//! Per-pass caches, as in mlx-lm `models/cache.py`: `KVCache` for global
//! passes, `RotatingKVCache` (keep = 0) for sliding-window passes, and the
//! four short-convolution states of every pass (`ArraysCache(4)`).
//!
//! The layouts and the update order are the reference's, because the order of
//! the keys changes how attention sums them.

use anyhow::Result;
use mlx_rs::Array;

use crate::ops;

const STEP: i32 = 256;

/// Growing bf16 cache for a global pass.
#[derive(Default)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    pub offset: i32,
}

impl KvCache {
    /// keys, values: (B, H, L, D). Returns all cached keys and values.
    pub fn update_and_fetch(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        let prev = self.offset;
        let l = keys.shape()[2];
        let grow = match &self.keys {
            None => true,
            Some(k) => prev + l > k.shape()[2],
        };
        if grow {
            let [b, h, _, dk] = keys.shape()[..] else {
                unreachable!()
            };
            let dv = values.shape()[3];
            let n_steps = (STEP + l - 1) / STEP;
            let new_k = ops::zeros(&[b, h, n_steps * STEP, dk], keys.dtype())?;
            let new_v = ops::zeros(&[b, h, n_steps * STEP, dv], values.dtype())?;
            match (self.keys.take(), self.values.take()) {
                (Some(mut k), Some(mut v)) => {
                    if prev % STEP != 0 {
                        k = ops::slice(&k, 2, 0, prev, 1)?;
                        v = ops::slice(&v, 2, 0, prev, 1)?;
                    }
                    self.keys = Some(ops::concatenate(&[&k, &new_k], 2)?);
                    self.values = Some(ops::concatenate(&[&v, &new_v], 2)?);
                }
                _ => {
                    self.keys = Some(new_k);
                    self.values = Some(new_v);
                }
            }
        }
        self.offset += l;
        let k = self.keys.as_ref().unwrap();
        let v = self.values.as_ref().unwrap();
        self.keys = Some(ops::slice_update(k, keys, 2, prev, self.offset)?);
        self.values = Some(ops::slice_update(v, values, 2, prev, self.offset)?);
        self.keys_and_values()
    }

    fn keys_and_values(&self) -> Result<(Array, Array)> {
        let k = self.keys.as_ref().unwrap();
        let v = self.values.as_ref().unwrap();
        if self.offset < k.shape()[2] {
            Ok((
                ops::slice(k, 2, 0, self.offset, 1)?,
                ops::slice(v, 2, 0, self.offset, 1)?,
            ))
        } else {
            Ok((k.clone(), v.clone()))
        }
    }

    pub fn arrays(&self) -> Vec<&Array> {
        self.keys.iter().chain(self.values.iter()).collect()
    }
}

/// Sliding-window cache that keeps the last `max_size` tokens, rotating in
/// place during decode.
pub struct RotatingKvCache {
    keys: Option<Array>,
    values: Option<Array>,
    pub offset: i32,
    max_size: i32,
    /// Next write position in the rotating buffer.
    pub idx: i32,
}

impl RotatingKvCache {
    pub fn new(max_size: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            max_size,
            idx: 0,
        }
    }

    /// Drops the oldest `trim_size` entries and appends `append`. With
    /// keep = 0 the reference's empty `v[..., :0, :]` part adds nothing.
    fn trim(trim_size: i32, v: &Array, append: Option<&Array>) -> Result<Array> {
        let base = if trim_size > 0 {
            ops::slice(v, 2, trim_size, v.shape()[2], 1)?
        } else {
            v.clone()
        };
        match append {
            Some(a) => ops::concatenate(&[&base, a], 2),
            None => ops::concatenate(&[&base], 2),
        }
    }

    /// Puts the cache in temporal order, slicing off the unused end.
    fn temporal_order(&self, v: &Array) -> Result<Array> {
        let n = v.shape()[2];
        if self.idx == n {
            Ok(v.clone())
        } else if self.idx < self.offset {
            let tail = ops::slice(v, 2, self.idx, n, 1)?;
            let head = ops::slice(v, 2, 0, self.idx, 1)?;
            ops::concatenate(&[&tail, &head], 2)
        } else {
            ops::slice(v, 2, 0, self.idx, 1)
        }
    }

    fn update_concat(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        match (self.keys.take(), self.values.take()) {
            (Some(k), Some(v)) => {
                let k = self.temporal_order(&k)?;
                let v = self.temporal_order(&v)?;
                self.idx = k.shape()[2];
                // At most max_size + L - 1, so every token sees max_size keys.
                let trim_size = self.idx - self.max_size + 1;
                self.keys = Some(Self::trim(trim_size, &k, Some(keys))?);
                self.values = Some(Self::trim(trim_size, &v, Some(values))?);
            }
            _ => {
                self.keys = Some(keys.clone());
                self.values = Some(values.clone());
            }
        }
        self.offset += keys.shape()[2];
        self.idx = self.keys.as_ref().unwrap().shape()[2];
        Ok((self.keys.clone().unwrap(), self.values.clone().unwrap()))
    }

    fn update_in_place(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        let [b, h, s, dk] = keys.shape()[..] else {
            unreachable!()
        };
        let prev = self.offset;
        let grow = match &self.keys {
            None => true,
            Some(k) => prev >= k.shape()[2] && k.shape()[2] < self.max_size,
        };
        if grow {
            let dv = values.shape()[3];
            let new_size = STEP.min(self.max_size - prev);
            let new_k = ops::zeros(&[b, h, new_size, dk], keys.dtype())?;
            let new_v = ops::zeros(&[b, h, new_size, dv], values.dtype())?;
            match (self.keys.take(), self.values.take()) {
                (Some(k), Some(v)) => {
                    self.keys = Some(ops::concatenate(&[&k, &new_k], 2)?);
                    self.values = Some(ops::concatenate(&[&v, &new_v], 2)?);
                }
                _ => {
                    self.keys = Some(new_k);
                    self.values = Some(new_v);
                }
            }
            self.idx = prev;
        }

        let trim_size = self.keys.as_ref().unwrap().shape()[2] - self.max_size;
        if trim_size > 0 {
            self.keys = Some(Self::trim(trim_size, self.keys.as_ref().unwrap(), None)?);
            self.values = Some(Self::trim(trim_size, self.values.as_ref().unwrap(), None)?);
            self.idx = self.max_size;
        }

        if self.idx == self.max_size {
            self.idx = 0;
        }

        let (i, j) = (self.idx, self.idx + s);
        self.keys = Some(ops::slice_update(
            self.keys.as_ref().unwrap(),
            keys,
            2,
            i,
            j,
        )?);
        self.values = Some(ops::slice_update(
            self.values.as_ref().unwrap(),
            values,
            2,
            i,
            j,
        )?);
        self.offset += s;
        self.idx += s;

        let k = self.keys.as_ref().unwrap();
        let v = self.values.as_ref().unwrap();
        if self.offset < k.shape()[2] {
            Ok((
                ops::slice(k, 2, 0, self.offset, 1)?,
                ops::slice(v, 2, 0, self.offset, 1)?,
            ))
        } else {
            Ok((k.clone(), v.clone()))
        }
    }

    pub fn update_and_fetch(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        if keys.shape()[2] == 1 {
            self.update_in_place(keys, values)
        } else {
            self.update_concat(keys, values)
        }
    }

    pub fn arrays(&self) -> Vec<&Array> {
        self.keys.iter().chain(self.values.iter()).collect()
    }
}

/// Storage format of the global passes' KV cache (`--kv-bits`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvFormat {
    /// 16 (bf16), 8 or 4 (MLX affine quantization).
    pub bits: i32,
    /// Channels per scale/bias group: 32, 64 or 128. Ignored for 16 bits.
    pub group_size: i32,
}

impl KvFormat {
    pub const BF16: Self = Self {
        bits: 16,
        group_size: 64,
    };

    pub fn new(bits: i32, group_size: i32) -> anyhow::Result<Self> {
        anyhow::ensure!(
            matches!(bits, 4 | 8 | 16),
            "KV cache bits must be 4, 8 or 16, not {bits}"
        );
        anyhow::ensure!(
            matches!(group_size, 32 | 64 | 128),
            "KV cache group size must be 32, 64 or 128, not {group_size}"
        );
        Ok(Self { bits, group_size })
    }

    pub fn is_quantized(self) -> bool {
        self.bits < 16
    }
}

/// Affine-quantized keys or values: packed u32, bf16 scales and biases, all
/// (B, H, capacity, ·).
#[derive(Clone)]
pub struct Packed {
    pub w: Array,
    pub scales: Array,
    pub biases: Array,
}

impl Packed {
    fn map(&self, f: impl Fn(&Array) -> Result<Array>) -> Result<Self> {
        Ok(Self {
            w: f(&self.w)?,
            scales: f(&self.scales)?,
            biases: f(&self.biases)?,
        })
    }

    fn zip(&self, other: &Self, f: impl Fn(&Array, &Array) -> Result<Array>) -> Result<Self> {
        Ok(Self {
            w: f(&self.w, &other.w)?,
            scales: f(&self.scales, &other.scales)?,
            biases: f(&self.biases, &other.biases)?,
        })
    }

    pub fn arrays(&self) -> [&Array; 3] {
        [&self.w, &self.scales, &self.biases]
    }
}

/// Global-pass cache in 8 or 4 bits. Same growth as `KvCache`, in blocks of
/// 256 tokens; new keys and values are quantized as they arrive.
pub struct QuantKvCache {
    pub keys: Option<Packed>,
    pub values: Option<Packed>,
    pub offset: i32,
    pub format: KvFormat,
}

impl QuantKvCache {
    pub fn new(format: KvFormat) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            format,
        }
    }

    fn quantize(&self, x: &Array) -> Result<Packed> {
        let (w, scales, biases) =
            ops::quantize_affine(x, self.format.group_size, self.format.bits)?;
        Ok(Packed { w, scales, biases })
    }

    /// Appends keys and values (B, H, L, D) in bf16. The full buffers stay
    /// in `keys`/`values`; the first `offset` tokens are valid.
    pub fn update(&mut self, keys: &Array, values: &Array) -> Result<()> {
        let prev = self.offset;
        let l = keys.shape()[2];
        let (k, v) = (self.quantize(keys)?, self.quantize(values)?);
        let grow = match &self.keys {
            None => true,
            Some(c) => prev + l > c.w.shape()[2],
        };
        if grow {
            let n = (STEP + l - 1) / STEP * STEP;
            let blank = |p: &Packed| {
                p.map(|a| {
                    let mut shape = a.shape().to_vec();
                    shape[2] = n;
                    ops::zeros(&shape, a.dtype())
                })
            };
            let (nk, nv) = (blank(&k)?, blank(&v)?);
            let extend = |old: Option<Packed>, new: Packed| -> Result<Packed> {
                match old {
                    None => Ok(new),
                    Some(o) => {
                        let o = if prev % STEP != 0 {
                            o.map(|a| ops::slice(a, 2, 0, prev, 1))?
                        } else {
                            o
                        };
                        o.zip(&new, |a, b| ops::concatenate(&[a, b], 2))
                    }
                }
            };
            self.keys = Some(extend(self.keys.take(), nk)?);
            self.values = Some(extend(self.values.take(), nv)?);
        }
        self.offset += l;
        let end = self.offset;
        self.keys = Some(
            self.keys
                .as_ref()
                .unwrap()
                .zip(&k, |a, b| ops::slice_update(a, b, 2, prev, end))?,
        );
        self.values = Some(
            self.values
                .as_ref()
                .unwrap()
                .zip(&v, |a, b| ops::slice_update(a, b, 2, prev, end))?,
        );
        Ok(())
    }

    /// All valid keys and values dequantized to bf16 (B, H, offset, D), as
    /// `mx.dequantize` does: the reference route.
    pub fn dequantized(&self) -> Result<(Array, Array)> {
        let f = self.format;
        let deq = |p: &Packed| -> Result<Array> {
            let p = p.map(|a| ops::slice(a, 2, 0, self.offset, 1))?;
            ops::dequantize_affine(&p.w, &p.scales, &p.biases, f.group_size, f.bits)
        };
        Ok((
            deq(self.keys.as_ref().unwrap())?,
            deq(self.values.as_ref().unwrap())?,
        ))
    }

    pub fn arrays(&self) -> Vec<&Array> {
        self.keys
            .iter()
            .chain(self.values.iter())
            .flat_map(|p| p.arrays())
            .collect()
    }
}

pub enum AttnCache {
    Global(KvCache),
    Quantized(QuantKvCache),
    Window(RotatingKvCache),
}

impl AttnCache {
    pub fn offset(&self) -> i32 {
        match self {
            Self::Global(c) => c.offset,
            Self::Quantized(c) => c.offset,
            Self::Window(c) => c.offset,
        }
    }

    pub fn update_and_fetch(&mut self, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        match self {
            Self::Global(c) => c.update_and_fetch(keys, values),
            Self::Window(c) => c.update_and_fetch(keys, values),
            Self::Quantized(_) => anyhow::bail!("quantized caches use update()"),
        }
    }
}

/// Everything one pass keeps between calls.
pub struct PassCache {
    pub attn: AttnCache,
    /// Short-convolution states: k, v, post-attention, post-MLP. Each is
    /// float32 (B, kernel_size - 1, C).
    pub conv: [Option<Array>; 4],
}

impl PassCache {
    pub fn arrays(&self) -> Vec<&Array> {
        let mut out = match &self.attn {
            AttnCache::Global(c) => c.arrays(),
            AttnCache::Quantized(c) => c.arrays(),
            AttnCache::Window(c) => c.arrays(),
        };
        out.extend(self.conv.iter().flatten());
        out
    }
}

/// Last `n` rows of a convolution window along axis 1, as a copy so the state
/// does not keep the window alive (`_conv_state` with no lengths).
pub fn conv_state(window: &Array, n: i32) -> Result<Array> {
    let w = window.shape()[1];
    let pos = ops::reshape(&ops::arange(w - n, w)?, &[1, n, 1])?;
    ops::take_along_axis(window, &pos, 1)
}
