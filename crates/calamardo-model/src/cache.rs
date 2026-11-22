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
#[derive(Clone, Default)]
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
            // Built from borrowed handles and assigned only on success, so
            // an error leaves the cache as it was.
            let (k, v) = match (&self.keys, &self.values) {
                (Some(k), Some(v)) => {
                    let (k, v) = if prev % STEP != 0 {
                        (ops::slice(k, 2, 0, prev, 1)?, ops::slice(v, 2, 0, prev, 1)?)
                    } else {
                        (k.clone(), v.clone())
                    };
                    (
                        ops::concatenate(&[&k, &new_k], 2)?,
                        ops::concatenate(&[&v, &new_v], 2)?,
                    )
                }
                _ => (new_k, new_v),
            };
            self.keys = Some(k);
            self.values = Some(v);
        }
        self.offset += l;
        let k = self.keys.as_ref().unwrap();
        let v = self.values.as_ref().unwrap();
        self.keys = Some(ops::slice_update(k, keys, 2, prev, self.offset)?);
        self.values = Some(ops::slice_update(v, values, 2, prev, self.offset)?);
        self.keys_and_values()
    }

    pub fn keys_and_values(&self) -> Result<(Array, Array)> {
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

    /// The whole key and value buffers (capacity may exceed `offset`), for
    /// kernels that read the first `offset` tokens in place.
    pub fn buffers(&self) -> Option<(&Array, &Array)> {
        Some((self.keys.as_ref()?, self.values.as_ref()?))
    }

    /// Forgets everything after the first `offset` tokens. The buffers stay;
    /// the next update overwrites from `offset`.
    pub fn truncate(&mut self, offset: i32) {
        self.offset = self.offset.min(offset);
    }
}

/// Sliding-window cache that keeps the last `max_size` tokens, rotating in
/// place during decode.
#[derive(Clone)]
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

    /// The buffers as they are, with the number of filled slots and the
    /// next write slot (slot s was written (idx - 1 - s) mod filled steps
    /// before the last token).
    pub fn ring(&self) -> Option<(&Array, &Array, i32, i32)> {
        let (k, v) = (self.keys.as_ref()?, self.values.as_ref()?);
        Some((k, v, self.offset.min(k.shape()[2]), self.idx))
    }

    /// The cached keys and values, oldest first.
    pub fn temporal(&self) -> Result<(Array, Array)> {
        let k = self.keys.as_ref().context("empty window cache")?;
        let v = self.values.as_ref().context("empty window cache")?;
        Ok((self.temporal_order(k)?, self.temporal_order(v)?))
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
#[derive(Clone)]
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

    /// Forgets everything after the first `offset` tokens.
    pub fn truncate(&mut self, offset: i32) {
        self.offset = self.offset.min(offset);
    }
}

#[derive(Clone)]
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
#[derive(Clone)]
pub struct PassCache {
    pub attn: AttnCache,
    /// Short-convolution states: k, v, post-attention, post-MLP. Each is
    /// float32 (B, kernel_size - 1, C).
    pub conv: [Option<Array>; 4],
    /// Convolution windows (state + inputs) of the last speculative
    /// verification, to rewind `conv` to an accepted token.
    pub windows: Option<[Array; 4]>,
    /// The current forward is a speculative verification (set and cleared
    /// by the layer around its attention).
    pub verifying: bool,
    /// Keys and values (B, Hkv, L, D) of the last verification of a
    /// sliding-window pass, not yet in its window: `rewind` writes the
    /// accepted ones.
    pub pending: Option<(Array, Array)>,
}

impl PassCache {
    pub fn arrays(&self) -> Vec<&Array> {
        let mut out = match &self.attn {
            AttnCache::Global(c) => c.arrays(),
            AttnCache::Quantized(c) => c.arrays(),
            AttnCache::Window(c) => c.arrays(),
        };
        out.extend(self.conv.iter().flatten());
        out.extend(self.windows.iter().flatten());
        if let Some((k, v)) = &self.pending {
            out.extend([k, v]);
        }
        out
    }
}

/// Last `n` rows of a convolution window along axis 1, as a copy so the state
/// does not keep the window alive (`_conv_state` with no lengths).
pub fn conv_state(window: &Array, n: i32) -> Result<Array> {
    let w = window.shape()[1];
    if w == n {
        // Already a state of its own (a decode step's kernel output).
        return Ok(window.clone());
    }
    // A new buffer in one kernel (K-6), with the values `take_along_axis`
    // gives: x * 1 is x for every float, -0 and NaN included. (`mx.copy`
    // would share the window's buffer and keep it alive.)
    ops::mul(&ops::slice(window, 1, w - n, w, 1)?, &ops::f32s(1.0))
}

/// What one pass needs to return to an earlier point of the same sequence.
/// Global caches only grow at the end, so their snapshot is a length; the
/// sliding windows and convolution states are kept by handle (MLX copies an
/// array only if it is written while shared, so the snapshot stays intact).
#[derive(Clone)]
pub enum PassSnapshot {
    Global {
        offset: i32,
        conv: [Option<Array>; 4],
    },
    Window {
        cache: RotatingKvCache,
        conv: [Option<Array>; 4],
    },
}

impl PassCache {
    pub fn snapshot(&self) -> PassSnapshot {
        match &self.attn {
            AttnCache::Window(c) => PassSnapshot::Window {
                cache: c.clone(),
                conv: self.conv.clone(),
            },
            other => PassSnapshot::Global {
                offset: other.offset(),
                conv: self.conv.clone(),
            },
        }
    }

    /// Returns to `snap`, which must come from this sequence (an ancestor of
    /// the current state).
    pub fn restore(&mut self, snap: &PassSnapshot) {
        self.windows = None;
        self.pending = None;
        match (snap, &mut self.attn) {
            (PassSnapshot::Window { cache, conv }, AttnCache::Window(c)) => {
                *c = cache.clone();
                self.conv = conv.clone();
            }
            (PassSnapshot::Global { offset, conv }, AttnCache::Global(c)) => {
                c.truncate(*offset);
                self.conv = conv.clone();
            }
            (PassSnapshot::Global { offset, conv }, AttnCache::Quantized(c)) => {
                c.truncate(*offset);
                self.conv = conv.clone();
            }
            _ => panic!("snapshot from a different cache layout"),
        }
    }

    pub fn nbytes(&self) -> usize {
        self.arrays().iter().map(|a| a.nbytes()).sum()
    }
}

impl PassSnapshot {
    /// Memory the snapshot keeps alive beyond the live cache (upper bound).
    pub fn nbytes(&self) -> usize {
        match self {
            Self::Window { cache, conv } => {
                cache.arrays().iter().map(|a| a.nbytes()).sum::<usize>()
                    + conv.iter().flatten().map(|a| a.nbytes()).sum::<usize>()
            }
            Self::Global { conv, .. } => conv.iter().flatten().map(|a| a.nbytes()).sum(),
        }
    }
}

/// Arrays and scalar fields of a saved cache (safetensors data and metadata).
#[derive(Default)]
pub struct Saved {
    pub arrays: Vec<(String, Array)>,
    pub meta: std::collections::BTreeMap<String, String>,
}

/// Arrays of a loaded file, taken by name.
pub struct Loaded {
    pub arrays: std::collections::HashMap<String, Array>,
    pub meta: std::collections::BTreeMap<String, String>,
}

impl Loaded {
    pub fn array(&mut self, k: &str) -> Result<Array> {
        self.arrays
            .remove(k)
            .with_context(|| format!("saved cache lacks {k}"))
    }

    pub fn opt(&mut self, k: &str) -> Option<Array> {
        self.arrays.remove(k)
    }

    pub fn int(&self, k: &str) -> Result<i32> {
        self.meta
            .get(k)
            .with_context(|| format!("saved cache lacks {k}"))?
            .parse()
            .with_context(|| format!("bad {k}"))
    }
}

use anyhow::Context;

fn export_conv(p: &str, conv: &[Option<Array>; 4], out: &mut Saved) {
    for (i, c) in conv.iter().enumerate() {
        if let Some(a) = c {
            out.arrays.push((format!("{p}.conv{i}"), a.clone()));
        }
    }
}

fn import_conv(p: &str, l: &mut Loaded) -> [Option<Array>; 4] {
    std::array::from_fn(|i| l.opt(&format!("{p}.conv{i}")))
}

fn export_packed(p: &str, x: &Packed, out: &mut Saved) {
    out.arrays.push((format!("{p}.w"), x.w.clone()));
    out.arrays.push((format!("{p}.scales"), x.scales.clone()));
    out.arrays.push((format!("{p}.biases"), x.biases.clone()));
}

fn import_packed(p: &str, l: &mut Loaded) -> Result<Packed> {
    Ok(Packed {
        w: l.array(&format!("{p}.w"))?,
        scales: l.array(&format!("{p}.scales"))?,
        biases: l.array(&format!("{p}.biases"))?,
    })
}

fn export_window(p: &str, c: &RotatingKvCache, out: &mut Saved) {
    if let (Some(k), Some(v)) = (&c.keys, &c.values) {
        out.arrays.push((format!("{p}.k"), k.clone()));
        out.arrays.push((format!("{p}.v"), v.clone()));
    }
    out.meta.insert(format!("{p}.offset"), c.offset.to_string());
    out.meta.insert(format!("{p}.idx"), c.idx.to_string());
}

fn import_window(p: &str, c: &mut RotatingKvCache, l: &mut Loaded) -> Result<()> {
    c.keys = l.opt(&format!("{p}.k"));
    c.values = l.opt(&format!("{p}.v"));
    c.offset = l.int(&format!("{p}.offset"))?;
    c.idx = l.int(&format!("{p}.idx"))?;
    Ok(())
}

impl PassCache {
    /// Adds this pass's state under prefix `p`. Buffers are saved whole (no
    /// slicing copy); the offsets say how much is valid.
    pub fn export(&self, p: &str, out: &mut Saved) {
        debug_assert!(
            self.pending.is_none(),
            "saving in the middle of a verification"
        );
        match &self.attn {
            AttnCache::Global(c) => {
                if let (Some(k), Some(v)) = (&c.keys, &c.values) {
                    out.arrays.push((format!("{p}.k"), k.clone()));
                    out.arrays.push((format!("{p}.v"), v.clone()));
                }
                out.meta.insert(format!("{p}.offset"), c.offset.to_string());
            }
            AttnCache::Quantized(c) => {
                if let (Some(k), Some(v)) = (&c.keys, &c.values) {
                    export_packed(&format!("{p}.k"), k, out);
                    export_packed(&format!("{p}.v"), v, out);
                }
                out.meta.insert(format!("{p}.offset"), c.offset.to_string());
            }
            AttnCache::Window(c) => export_window(p, c, out),
        }
        export_conv(p, &self.conv, out);
    }

    /// Fills this (fresh) pass cache from a saved one.
    pub fn import(&mut self, p: &str, l: &mut Loaded) -> Result<()> {
        match &mut self.attn {
            AttnCache::Global(c) => {
                c.keys = l.opt(&format!("{p}.k"));
                c.values = l.opt(&format!("{p}.v"));
                c.offset = l.int(&format!("{p}.offset"))?;
            }
            AttnCache::Quantized(c) => {
                if l.arrays.contains_key(&format!("{p}.k.w")) {
                    c.keys = Some(import_packed(&format!("{p}.k"), l)?);
                    c.values = Some(import_packed(&format!("{p}.v"), l)?);
                }
                c.offset = l.int(&format!("{p}.offset"))?;
            }
            AttnCache::Window(c) => import_window(p, c, l)?,
        }
        self.conv = import_conv(p, l);
        Ok(())
    }
}

impl PassSnapshot {
    pub fn export(&self, p: &str, out: &mut Saved) {
        match self {
            Self::Global { offset, conv } => {
                out.meta.insert(format!("{p}.offset"), offset.to_string());
                export_conv(p, conv, out);
            }
            Self::Window { cache, conv } => {
                export_window(p, cache, out);
                export_conv(p, conv, out);
            }
        }
    }

    /// Rebuilds a snapshot of the same kind as `like`.
    pub fn import(like: &PassCache, p: &str, l: &mut Loaded) -> Result<Self> {
        Ok(match &like.attn {
            AttnCache::Window(c) => {
                let mut cache = RotatingKvCache::new(c.max_size);
                import_window(p, &mut cache, l)?;
                Self::Window {
                    cache,
                    conv: import_conv(p, l),
                }
            }
            _ => Self::Global {
                offset: l.int(&format!("{p}.offset"))?,
                conv: import_conv(p, l),
            },
        })
    }
}

impl RotatingKvCache {
    /// Forgets the last `n` tokens of a multi-token (concatenating) update,
    /// whose keys are in temporal order.
    fn drop_last(&mut self, n: i32) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let k = self.keys.as_ref().context("empty window cache")?;
        let len = k.shape()[2];
        anyhow::ensure!(
            self.idx == len && n <= len,
            "window cache is not in temporal order"
        );
        self.keys = Some(ops::slice(k, 2, 0, len - n, 1)?);
        self.values = Some(ops::slice(self.values.as_ref().unwrap(), 2, 0, len - n, 1)?);
        self.offset -= n;
        self.idx = len - n;
        Ok(())
    }
}

impl PassCache {
    /// After a speculative verification that fed `fed` tokens, keeps only
    /// the first `keep` of them (rejected drafts leave no trace).
    pub fn rewind(&mut self, fed: i32, keep: i32) -> Result<()> {
        let drop = fed - keep;
        match (&mut self.attn, self.pending.take()) {
            (AttnCache::Window(c), Some((k, v))) => {
                // The window was left as it was: the accepted tokens go in
                // one by one, as decode steps would write them.
                anyhow::ensure!(k.shape()[2] == fed, "pending keys of another verification");
                for i in 0..keep {
                    let ki = ops::slice(&k, 2, i, i + 1, 1)?;
                    let vi = ops::slice(&v, 2, i, i + 1, 1)?;
                    drop_views(c.update_and_fetch(&ki, &vi)?);
                }
            }
            (_, Some(_)) => anyhow::bail!("pending window keys on a global pass"),
            (AttnCache::Global(c), None) => c.truncate(c.offset - drop),
            (AttnCache::Quantized(c), None) => c.truncate(c.offset - drop),
            (AttnCache::Window(c), None) => c.drop_last(drop)?,
        }
        let windows = self.windows.take();
        anyhow::ensure!(
            drop == 0 || windows.is_some(),
            "rewind without the verification's windows"
        );
        if let Some(windows) = windows
            && drop > 0
        {
            for (state, window) in self.conv.iter_mut().zip(windows.iter()) {
                // The window is the previous state followed by the `fed`
                // inputs; the state after `keep` of them starts at row `keep`.
                let n = state.as_ref().context("convolution state")?.shape()[1];
                anyhow::ensure!(
                    window.shape()[1] == n + fed,
                    "convolution window of {} rows for {fed} fed tokens",
                    window.shape()[1]
                );
                let pos = ops::reshape(&ops::arange(keep, keep + n)?, &[1, n, 1])?;
                *state = Some(ops::take_along_axis(window, &pos, 1)?);
            }
        }
        Ok(())
    }
}

/// Drops the views a cache update returns, so the buffers are not shared
/// when the next update writes them.
fn drop_views(_: (Array, Array)) {}

/// Capacity kept after a fork: the valid tokens rounded up to a block.
fn shrunk(offset: i32) -> i32 {
    (offset + STEP - 1) / STEP * STEP
}

impl KvCache {
    pub fn export(&self, p: &str, out: &mut Saved) {
        if let (Some(k), Some(v)) = (&self.keys, &self.values) {
            out.arrays.push((format!("{p}.k"), k.clone()));
            out.arrays.push((format!("{p}.v"), v.clone()));
        }
        out.meta
            .insert(format!("{p}.offset"), self.offset.to_string());
    }

    pub fn import(&mut self, p: &str, l: &mut Loaded) -> Result<()> {
        self.keys = l.opt(&format!("{p}.k"));
        self.values = l.opt(&format!("{p}.v"));
        self.offset = l.int(&format!("{p}.offset"))?;
        let cap = self.keys.as_ref().map_or(0, |k| k.shape()[2]);
        anyhow::ensure!(
            self.offset >= 0 && self.offset <= cap,
            "{p}: offset {} beyond {cap}",
            self.offset
        );
        Ok(())
    }

    /// Drops capacity beyond the valid tokens, so the first write after a
    /// fork copies only the prefix, not the whole shared buffer.
    pub fn shrink(&mut self) -> Result<()> {
        let n = shrunk(self.offset);
        if let (Some(k), Some(v)) = (&self.keys, &self.values)
            && k.shape()[2] > n
        {
            self.keys = Some(ops::slice(k, 2, 0, n, 1)?);
            self.values = Some(ops::slice(v, 2, 0, n, 1)?);
        }
        Ok(())
    }
}

impl QuantKvCache {
    fn shrink(&mut self) -> Result<()> {
        let n = shrunk(self.offset);
        if let (Some(k), Some(v)) = (&self.keys, &self.values)
            && k.w.shape()[2] > n
        {
            self.keys = Some(k.map(|a| ops::slice(a, 2, 0, n, 1))?);
            self.values = Some(v.map(|a| ops::slice(a, 2, 0, n, 1))?);
        }
        Ok(())
    }
}

impl RotatingKvCache {
    pub fn max_size(&self) -> i32 {
        self.max_size
    }

    pub fn export(&self, p: &str, out: &mut Saved) {
        export_window(p, self, out);
    }

    pub fn import(&mut self, p: &str, l: &mut Loaded) -> Result<()> {
        import_window(p, self, l)
    }
}

impl PassCache {
    /// See `KvCache::shrink`.
    pub fn shrink(&mut self) -> Result<()> {
        match &mut self.attn {
            AttnCache::Global(c) => c.shrink(),
            AttnCache::Quantized(c) => c.shrink(),
            AttnCache::Window(_) => Ok(()),
        }
    }

    /// Checks a loaded cache against the number of tokens it should hold.
    pub fn validate(&self, tokens: usize) -> Result<()> {
        let (offset, cap) = match &self.attn {
            AttnCache::Global(c) => (c.offset, c.keys.as_ref().map_or(0, |k| k.shape()[2])),
            AttnCache::Quantized(c) => (c.offset, c.keys.as_ref().map_or(0, |k| k.w.shape()[2])),
            AttnCache::Window(c) => (c.offset, i32::MAX),
        };
        anyhow::ensure!(
            offset as usize == tokens && offset <= cap,
            "cache holds {offset} tokens (capacity {cap}), expected {tokens}"
        );
        Ok(())
    }
}
