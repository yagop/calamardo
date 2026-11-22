//! Inkling 2 Mini forward pass, ported op by op from mlx-lm
//! `models/inkling2/{layers,model,checkpoint}.py` (text only, batch of one,
//! bf16 caches). W4A16: expert weights stay nvfp4/mxfp8, activations bf16.

use std::{collections::HashMap, ffi::CStr, path::Path};

use anyhow::{Context, Result, bail, ensure};
use calamardo_ckpt::{
    ModelConfig,
    config::TextConfig,
    hub::{Checkpoint, INDEX_FILE},
    is_skipped,
};
use mlx_rs::{Array, Dtype};

use crate::{
    cache::{
        AttnCache, KvCache, KvFormat, Packed, PassCache, QuantKvCache, RotatingKvCache, conv_state,
    },
    kernels::{Kernels, MAX_DECODE_QUERIES},
    ops::{self, f32s},
};

const ROUTER_SCORE_EPS: f32 = 1e-20;
const NVFP4_E2M1_MAX: f32 = 6.0;
const NVFP4_E4M3_MAX: f32 = 448.0;
const UNWEIGHTED_EPS: f32 = 1e-6;

fn unweighted_rms_norm(x: &Array) -> Result<Array> {
    ops::rms_norm(x, None, UNWEIGHTED_EPS)
}

// ---- Weights ------------------------------------------------------------

/// Checkpoint tensors by name, consumed as the model is built.
struct Weights(HashMap<String, Array>);

impl Weights {
    fn take(&mut self, name: &str) -> Result<Array> {
        self.0
            .remove(name)
            .with_context(|| format!("missing tensor {name}"))
    }
}

/// An expert projection: `SwitchLinear` quantized with nvfp4 or mxfp8.
struct QuantizedSwitchLinear {
    weight: Array,
    scales: Array,
    global_scale: Option<Array>,
    group_size: i32,
    bits: i32,
    mode: &'static CStr,
}

impl QuantizedSwitchLinear {
    /// `checkpoint.sanitize` for one `w13`/`w2` projection.
    fn load(w: &mut Weights, prefix: &str, in_dims: i32) -> Result<Self> {
        let weight = ops::view(&w.take(prefix)?, Dtype::Uint32)?;
        let scales = w.take(&format!("{prefix}_scale"))?;
        ensure!(
            scales.dtype() == Dtype::Uint8,
            "{prefix}_scale is {:?}",
            scales.dtype()
        );
        let group_size = in_dims / scales.shape()[2];
        let bits = weight.shape()[2] * 32 / in_dims;
        let mode: &'static CStr = match (bits, group_size) {
            (4, 16) => c"nvfp4",
            (4, 32) => c"mxfp4",
            (8, 32) => c"mxfp8",
            other => bail!("{prefix}: unsupported quantization {other:?}"),
        };
        let amax = w.0.remove(&format!("{prefix}_amax"));
        let denominator = w.0.remove(&format!("{prefix}_outer_scale_denominator"));
        let global_scale = match (amax, denominator) {
            (Some(amax), Some(den)) => {
                let scale = ops::div(&f32s(NVFP4_E2M1_MAX * NVFP4_E4M3_MAX), &ops::f32(&den)?)?;
                Some(ops::mul(&ops::f32(&amax)?, &scale)?)
            }
            (None, None) => None,
            _ => bail!("{prefix}: amax without denominator or the reverse"),
        };
        Ok(Self {
            weight,
            scales,
            global_scale,
            group_size,
            bits,
            mode,
        })
    }

    fn forward(&self, x: &Array, indices: &Array, sorted: bool) -> Result<Array> {
        ops::gather_qmm(
            x,
            &self.weight,
            &self.scales,
            indices,
            self.group_size,
            self.bits,
            self.mode,
            self.global_scale.as_ref(),
            sorted,
        )
    }

    fn arrays(&self) -> Vec<&Array> {
        let mut v = vec![&self.weight, &self.scales];
        v.extend(self.global_scale.iter());
        v
    }
}

/// Depthwise causal convolution over the last `kernel_size` tokens.
struct ShortConvolution {
    /// (C, K, 1), the checkpoint's (C, 1, K) with the last two axes swapped
    /// (row-contiguous as stored, the size-1 axis aside).
    weight: Array,
    kernel_size: i32,
}

impl ShortConvolution {
    fn load(w: &mut Weights, name: &str, kernel_size: i32) -> Result<Self> {
        Ok(Self {
            weight: ops::contiguous(&ops::swapaxes(&w.take(name)?, 1, 2)?)?,
            kernel_size,
        })
    }

    /// Returns the output and the window (state + input) for the next state;
    /// a decode step returns the next state itself (the window's last
    /// K - 1 rows, which `conv_state` then keeps as is). `f32_out`: the
    /// caller casts the output to float32 next; a decode step then returns
    /// it already as float32 (the same values, K-6). `keep_window`: return
    /// the full window even for one token (speculative verification).
    fn forward(
        &self,
        kern: &Kernels,
        x: &Array,
        state: Option<&Array>,
        f32_out: bool,
        keep_window: bool,
    ) -> Result<(Array, Array)> {
        let k = self.kernel_size;
        // A verification keeps the whole window to rewind with.
        if let Some(state) = state
            && !keep_window
            && x.shape()[..2] == [1, 1]
            && x.dtype() == self.weight.dtype()
        {
            // A decode step: one kernel, same results (K-6).
            let c = self.weight.shape()[0];
            return kern.short_conv_step(
                x,
                state,
                &ops::reshape(&self.weight, &[c, k])?,
                ((k + 1) as f64).sqrt() as f32,
                f32_out,
            );
        }
        let residual = ops::f32(x)?;
        let window = conv_input(x, state, k)?;
        let channels = self.weight.shape()[0];
        let out = ops::conv1d(
            &ops::astype(&window, self.weight.dtype())?,
            &self.weight,
            channels,
        )?;
        let out = ops::add(&ops::f32(&out)?, &residual)?;
        let out = ops::div(&out, &f32s(((k + 1) as f64).sqrt() as f32))?;
        Ok((ops::astype(&out, x.dtype())?, window))
    }
}

/// Short convolution whose weights also depend on the input (global passes).
struct DynamicShortConvolution {
    /// (K, C).
    static_weight: Array,
    /// (H * K, D).
    weight_proj: Array,
    kernel_size: i32,
    num_heads: i32,
}

impl DynamicShortConvolution {
    fn load(w: &mut Weights, prefix: &str, kernel_size: i32, num_heads: i32) -> Result<Self> {
        Ok(Self {
            static_weight: w.take(&format!("{prefix}.static_w"))?,
            weight_proj: w.take(&format!("{prefix}.weight_proj"))?,
            kernel_size,
            num_heads,
        })
    }

    fn forward(
        &self,
        x: &Array,
        dynamic_input: &Array,
        state: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let [b, l, c] = x.shape()[..] else {
            unreachable!()
        };
        let (k, h) = (self.kernel_size, self.num_heads);
        let dynamic = ops::f32(&ops::linear(dynamic_input, &self.weight_proj)?)?;
        let dynamic = ops::reshape(&dynamic, &[b, l, h, 1, k])?;
        let fixed = ops::reshape(&ops::t(&ops::f32(&self.static_weight)?)?, &[h, -1, k])?;
        let weights = ops::add(&fixed, &dynamic)?;

        let window = conv_input(x, state, k)?;
        let lags = (0..k)
            .map(|lag| ops::slice(&window, 1, k - 1 - lag, k - 1 - lag + l, 1))
            .collect::<Result<Vec<_>>>()?;
        let windows = ops::stack(&lags.iter().collect::<Vec<_>>(), -1)?;
        let windows = ops::reshape(&windows, &[b, l, h, -1, k])?;
        let out = ops::sum(&ops::mul(&windows, &weights)?, &[-1], false)?;
        let out = ops::reshape(&out, &[b, l, c])?;
        let out = ops::add(&out, &ops::f32(x)?)?;
        let out = ops::div(&out, &f32s(((2 * k + 1) as f64).sqrt() as f32))?;
        Ok((ops::astype(&out, x.dtype())?, window))
    }
}

/// `_conv_input`: the float32 input after the previous `kernel_size - 1`
/// rows (zeros at the start).
fn conv_input(x: &Array, state: Option<&Array>, kernel_size: i32) -> Result<Array> {
    let x = ops::f32(x)?;
    let zeros;
    let state = match state {
        Some(s) => s,
        None => {
            zeros = ops::zeros(
                &[x.shape()[0], kernel_size - 1, x.shape()[2]],
                Dtype::Float32,
            )?;
            &zeros
        }
    };
    ops::concatenate(&[state, &x], 1)
}

enum KvConv {
    Static(ShortConvolution),
    Dynamic(DynamicShortConvolution),
}

/// `l2_clip` for a few tokens (decode, verification) as one compiled
/// function per threshold (K-6): the same ops, fused around the sum.
fn l2_clip_short(
    kern: &Kernels,
    x: &Array,
    scale: Option<&Array>,
    threshold: f32,
) -> Result<Array> {
    if x.shape()[1] > MAX_DECODE_QUERIES {
        return l2_clip(x, scale, threshold);
    }
    let name = format!("l2_clip/{}/{}", threshold.to_bits(), scale.is_some());
    let f = kern.compiled(&name, || {
        ops::Compiled::new(move |a: &[Array]| Ok(vec![l2_clip(&a[0], a.get(1), threshold)?]))
    });
    match scale {
        Some(s) => f.call1(&[x, s]),
        None => f.call1(&[x]),
    }
}

/// `_l2_clip`: scales rows of float32 `x * scale` down to an L2 norm of at
/// most `threshold`.
fn l2_clip(x: &Array, scale: Option<&Array>, threshold: f32) -> Result<Array> {
    let mut states = ops::f32(x)?;
    states = match scale {
        Some(s) => ops::mul(&states, s)?,
        None => ops::mul(&states, &f32s(1.0))?,
    };
    let norm = ops::sqrt(&ops::sum(&ops::mul(&states, &states)?, &[-1], true)?)?;
    let factor = ops::div(&f32s(threshold), &ops::add(&norm, &f32s(1e-6))?)?;
    let scaled = ops::mul(&states, &ops::minimum(&factor, &f32s(1.0))?)?;
    ops::astype(&scaled, x.dtype())
}

struct Attention {
    /// q, r and gate projections stacked by rows, then k and v on sliding
    /// passes (their inputs are the same there when no messages mix in):
    /// one gemv for all of them in decode (K-6; the same rows bit for bit,
    /// MLX takes the same reduction for these sizes). The single
    /// projections below are views of it.
    merged: Array,
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    r_proj: Array,
    o_proj: Array,
    gate_proj: Array,
    sink_proj: Array,
    /// (d_rel, R).
    rel_proj: Array,
    q_scale: Array,
    rel_scale: Array,
    k_conv: KvConv,
    v_conv: KvConv,
    num_heads: i32,
    num_kv_heads: i32,
    scaling: f32,
    /// 0 for global passes.
    window: i32,
    qk_clip: f32,
    rel_clip: f32,
    eps: f32,
    log_scaling_alpha: f32,
    log_scaling_n_floor: f32,
}

impl Attention {
    fn load(w: &mut Weights, p: &str, cfg: &TextConfig, sliding: bool) -> Result<Self> {
        let k = cfg.sconv_kernel_size as i32;
        let (k_conv, v_conv) = if sliding {
            (
                KvConv::Static(ShortConvolution::load(
                    w,
                    &format!("{p}.k_sconv.weight"),
                    k,
                )?),
                KvConv::Static(ShortConvolution::load(
                    w,
                    &format!("{p}.v_sconv.weight"),
                    k,
                )?),
            )
        } else {
            let h = cfg.num_key_value_heads as i32;
            (
                KvConv::Dynamic(DynamicShortConvolution::load(
                    w,
                    &format!("{p}.k_sconv"),
                    k,
                    h,
                )?),
                KvConv::Dynamic(DynamicShortConvolution::load(
                    w,
                    &format!("{p}.v_sconv"),
                    k,
                    h,
                )?),
            )
        };
        let (heads, kv_heads, head_dim, window) = if sliding {
            (
                cfg.swa_num_attention_heads,
                cfg.swa_num_key_value_heads,
                cfg.swa_head_dim,
                cfg.sliding_window_size,
            )
        } else {
            (
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                0,
            )
        };
        let q = w.take(&format!("{p}.wq_du.weight"))?;
        let r = w.take(&format!("{p}.wr_du.weight"))?;
        let g = w.take(&format!("{p}.gate_attn.weight"))?;
        let kp = w.take(&format!("{p}.wk_dv.weight"))?;
        let vp = w.take(&format!("{p}.wv_dv.weight"))?;
        let mut parts = vec![&q, &r, &g];
        if sliding {
            parts.extend([&kp, &vp]);
        }
        let merged = ops::concatenate(&parts, 0)?;
        let mut start = 0;
        let mut rows = |a: &Array| -> Result<Array> {
            let n = a.shape()[0];
            start += n;
            ops::slice(&merged, 0, start - n, start, 1)
        };
        let (q_proj, r_proj, gate_proj) = (rows(&q)?, rows(&r)?, rows(&g)?);
        let (k_proj, v_proj) = if sliding {
            (rows(&kp)?, rows(&vp)?)
        } else {
            (kp, vp)
        };
        Ok(Self {
            merged,
            q_proj,
            k_proj,
            v_proj,
            r_proj,
            o_proj: w.take(&format!("{p}.wo_ud.weight"))?,
            gate_proj,
            sink_proj: w.take(&format!("{p}.attn_sink_proj.weight"))?,
            rel_proj: w.take(&format!("{p}.rel_logits_proj.proj"))?,
            q_scale: w.take(&format!("{p}.q_l2_clip.scale"))?,
            rel_scale: w.take(&format!("{p}.rel_l2_clip.scale"))?,
            k_conv,
            v_conv,
            num_heads: heads as i32,
            num_kv_heads: kv_heads as i32,
            scaling: (head_dim as f32).powf(-0.5),
            window: window as i32,
            qk_clip: cfg.qk_l2_clip_threshold,
            rel_clip: cfg.rel_l2_clip_threshold,
            eps: cfg.rms_norm_eps,
            log_scaling_alpha: cfg.log_scaling_alpha,
            log_scaling_n_floor: cfg.log_scaling_n_floor as f32,
        })
    }

    /// `shared_input`: the key and value inputs are `hidden` itself.
    fn forward(
        &self,
        k: &Kernels,
        hidden: &Array,
        key_hidden: &Array,
        value_hidden: &Array,
        shared_input: bool,
        cache: &mut PassCache,
    ) -> Result<Array> {
        let [b, l, _] = hidden.shape()[..] else {
            unreachable!()
        };
        // The projections of `hidden`: q, r, gate, then k and v when they
        // share it. One gemv over the stacked weights in decode (K-6).
        let rows: Vec<i32> = [
            &self.q_proj,
            &self.r_proj,
            &self.gate_proj,
            &self.k_proj,
            &self.v_proj,
        ]
        .iter()
        .map(|a| a.shape()[0])
        .collect();
        let with_kv = shared_input && self.merged.shape()[0] == rows.iter().sum::<i32>();
        let projected: Option<Array> = if b == 1 && l == 1 {
            let w = if with_kv {
                self.merged.clone()
            } else {
                ops::slice(&self.merged, 0, 0, rows[0] + rows[1] + rows[2], 1)?
            };
            Some(ops::linear(hidden, &w)?)
        } else {
            None
        };
        let part = |i: usize, w: &Array, x: &Array| -> Result<Array> {
            match &projected {
                Some(p) if i < 3 || with_kv => {
                    let start: i32 = rows[..i].iter().sum();
                    ops::slice(p, -1, start, start + rows[i], 1)
                }
                _ => ops::linear(x, w),
            }
        };
        let (key_states, kw) = match &self.k_conv {
            KvConv::Static(c) => c.forward(
                k,
                &part(3, &self.k_proj, key_hidden)?,
                cache.conv[0].as_ref(),
                false,
                cache.verifying,
            )?,
            KvConv::Dynamic(c) => c.forward(
                &part(3, &self.k_proj, key_hidden)?,
                key_hidden,
                cache.conv[0].as_ref(),
            )?,
        };
        let (value_states, vw) = match &self.v_conv {
            KvConv::Static(c) => c.forward(
                k,
                &part(4, &self.v_proj, value_hidden)?,
                cache.conv[1].as_ref(),
                false,
                cache.verifying,
            )?,
            KvConv::Dynamic(c) => c.forward(
                &part(4, &self.v_proj, value_hidden)?,
                value_hidden,
                cache.conv[1].as_ref(),
            )?,
        };
        cache.conv[0] = Some(kw);
        cache.conv[1] = Some(vw);

        let tau = if self.window == 0 {
            let l = hidden.shape()[1];
            let offset = cache.attn.offset();
            let pos = ops::f32(&ops::add(&ops::arange(0, l)?, &ops::i32s(offset))?)?;
            let x = ops::div(
                &ops::add(&pos, &f32s(1.0))?,
                &f32s(self.log_scaling_n_floor),
            )?;
            let x = ops::log(&ops::maximum(&x, &f32s(1.0))?)?;
            let tau = ops::add(&f32s(1.0), &ops::mul(&f32s(self.log_scaling_alpha), &x)?)?;
            Some(ops::reshape(&tau, &[l, 1, 1])?)
        } else {
            None
        };

        let query = part(0, &self.q_proj, hidden)?;
        let relative = part(1, &self.r_proj, hidden)?;
        let gate = part(2, &self.gate_proj, hidden)?;
        self.attend(
            k,
            hidden,
            &query,
            &key_states,
            &value_states,
            &relative,
            &gate,
            cache,
            tau.as_ref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        kern: &Kernels,
        hidden: &Array,
        query: &Array,
        key: &Array,
        value: &Array,
        relative: &Array,
        gate: &Array,
        cache: &mut PassCache,
        tau: Option<&Array>,
    ) -> Result<Array> {
        let [b, l, _] = hidden.shape()[..] else {
            unreachable!()
        };
        let sinks = ops::linear(hidden, &self.sink_proj)?;

        let mut q = ops::reshape(query, &[b, l, self.num_heads, -1])?;
        let k = ops::reshape(key, &[b, l, self.num_kv_heads, -1])?;
        let v = ops::reshape(value, &[b, l, self.num_kv_heads, -1])?;
        let mut r = ops::reshape(relative, &[b, l, self.num_heads, -1])?;
        q = l2_clip_short(kern, &q, Some(&self.q_scale), self.qk_clip)?;
        let k = l2_clip_short(kern, &k, None, self.qk_clip)?;
        let eps = self.eps;
        let v_norm = move |a: &[Array]| -> Result<Vec<Array>> {
            Ok(vec![ops::astype(
                &ops::rms_norm(&ops::f32(&a[0])?, None, eps)?,
                a[0].dtype(),
            )?])
        };
        let v = if l <= MAX_DECODE_QUERIES {
            kern.compiled(&format!("v_norm/{}", eps.to_bits()), || {
                ops::Compiled::new(v_norm)
            })
            .call1(&[&v])?
        } else {
            v_norm(&[v])?.remove(0)
        };
        r = l2_clip_short(kern, &r, Some(&self.rel_scale), self.rel_clip)?;
        if let Some(tau) = tau {
            q = ops::astype(&ops::mul(&ops::f32(&q)?, tau)?, q.dtype())?;
            r = ops::astype(&ops::mul(&ops::f32(&r)?, tau)?, r.dtype())?;
        }
        let position_bias = ops::matmul(&r, &ops::astype(&self.rel_proj, r.dtype())?)?;

        let q = ops::transpose(&q, &[0, 2, 1, 3])?;
        let k = ops::transpose(&k, &[0, 2, 1, 3])?;
        let v = ops::transpose(&v, &[0, 2, 1, 3])?;
        let verifying = cache.verifying;
        let mut pending = None;
        let out = match &mut cache.attn {
            AttnCache::Quantized(c) => {
                c.update(&k, &v)?;
                if l <= MAX_DECODE_QUERIES {
                    let (keys, values) = (c.keys.as_ref().unwrap(), c.values.as_ref().unwrap());
                    kern.quantized_decode_attention(
                        &q,
                        keys,
                        values,
                        c.offset,
                        &position_bias,
                        &sinks,
                        self.scaling,
                        c.format,
                    )?
                } else {
                    // Prefill reads the packed history in place (K-2, K-3):
                    // no dequantized copy, same output bit for bit.
                    let (keys, values) = (c.keys.as_ref().unwrap(), c.values.as_ref().unwrap());
                    kern.relative_attention_quantized(
                        &q,
                        keys,
                        values,
                        c.offset,
                        &position_bias,
                        &sinks,
                        self.scaling,
                        c.format,
                    )?
                }
            }
            // Only for verification: prefill chunks of a few tokens keep the
            // reference's kernel (exact parity).
            AttnCache::Global(c) if verifying && (2..=MAX_DECODE_QUERIES).contains(&l) => {
                // Speculative verification: the decode kernel over the bf16
                // cache in place (the prefill kernel has too few threadgroups
                // for a handful of queries over a long context).
                c.update_and_fetch(&k, &v)?;
                let (kb, vb) = c.buffers().context("empty cache")?;
                // Scales and biases are not read for 16 bits; the buffer
                // itself stands in (MLX passes small arrays as constants,
                // which the kernel's device pointers cannot take).
                let wrap = |a: &Array| Packed {
                    w: a.clone(),
                    scales: a.clone(),
                    biases: a.clone(),
                };
                kern.quantized_decode_attention(
                    &q,
                    &wrap(kb),
                    &wrap(vb),
                    c.offset,
                    &position_bias,
                    &sinks,
                    self.scaling,
                    KvFormat::BF16,
                )?
            }
            // Speculative verification of a sliding window: attend over the
            // ring buffer as it is plus the new keys, which wait in
            // `pending` until `rewind` writes the accepted ones. The
            // reference's path would put the window in temporal order and
            // concatenate (several copies of it per pass).
            AttnCache::Window(c)
                if verifying
                    && b == 1
                    && (2..=MAX_DECODE_QUERIES).contains(&l)
                    && c.ring().is_some() =>
            {
                let (rk, rv, rlen, ridx) = c.ring().context("empty window")?;
                let out = kern.window_verify_attention(
                    &q,
                    rk,
                    rv,
                    rlen,
                    ridx,
                    &ops::contiguous(&k)?,
                    &ops::contiguous(&v)?,
                    &position_bias,
                    &sinks,
                    self.scaling,
                    self.window,
                )?;
                pending = Some((k, v));
                out
            }
            attn => {
                let (k, v) = attn.update_and_fetch(&k, &v)?;
                self.attention(kern, &q, &k, &v, &position_bias, &sinks, attn)?
            }
        };
        if verifying {
            // This verification's keys, or none if it took the reference's
            // path (rewind then drops the last keys instead).
            cache.pending = pending;
        }

        let out = ops::reshape(&out, &[b, l, -1])?;
        let gated = |a: &[Array]| -> Result<Vec<Array>> {
            let out = ops::mul(&ops::f32(&a[0])?, &ops::sigmoid(&ops::f32(&a[1])?)?)?;
            Ok(vec![ops::astype(&out, a[1].dtype())?])
        };
        let out = if l <= MAX_DECODE_QUERIES {
            // One compiled function (K-6).
            kern.compiled("attention_gate", || ops::Compiled::new(gated))
                .call1(&[&out, gate])?
        } else {
            gated(&[out, gate.clone()])?.remove(0)
        };
        ops::linear(&out, &self.o_proj)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        kern: &Kernels,
        q: &Array,
        k: &Array,
        v: &Array,
        position_bias: &Array,
        sinks: &Array,
        cache: &AttnCache,
    ) -> Result<Array> {
        let [b, _, l, _] = q.shape()[..] else {
            unreachable!()
        };
        let s = k.shape()[2];
        let end = cache.offset();
        let bf16 = q.dtype() == Dtype::Bfloat16
            && k.dtype() == Dtype::Bfloat16
            && v.dtype() == Dtype::Bfloat16;
        ensure!(bf16, "attention expects bfloat16 inputs");
        if l > 1 {
            let key_start = ops::broadcast_to(&ops::i32s(s - end), &[b])?;
            return kern.relative_attention(
                q,
                k,
                v,
                position_bias,
                sinks,
                self.scaling,
                self.window,
                &key_start,
            );
        }
        ensure!(b == 1, "decode supports one sequence");

        // Decode: the fused attention with the bias and window as a mask,
        // built in one kernel (the reference's ops, bit for bit).
        let r = position_bias.shape()[3];
        let pb = ops::slice(&ops::slice(position_bias, 0, 0, 1, 1)?, 1, 0, 1, 1)?;
        let pb = ops::reshape(&pb, &[position_bias.shape()[2], r])?;
        let rot = match cache {
            // A decode step writes in place, so the keys can be rotated.
            AttnCache::Window(c) => c.idx,
            _ => 0,
        };
        let mask = kern.decode_mask(&pb, s, rot, end, self.window, q.dtype())?;
        let sinks = ops::reshape(
            &ops::slice(&ops::slice(sinks, 0, 0, 1, 1)?, 1, 0, 1, 1)?,
            &[sinks.shape()[2]],
        )?;
        let sinks = ops::astype(&sinks, q.dtype())?;
        let out = ops::sdpa(q, k, v, self.scaling, &mask, &sinks)?;
        ops::transpose(&out, &[0, 2, 1, 3])
    }
}

/// `MessageGate`: mixes the loop message into a stream.
struct MessageGate {
    down: Array,
    /// float32. A view of `fg`'s first rows.
    forget: Array,
    /// float32. A view of `fg`'s last rows.
    message: Array,
    /// float32 (2 D, r): the forget and gain projections stacked, one gemv
    /// for both in decode (K-6; MLX reduces each row the same way for
    /// these sizes, as K <= 64).
    fg: Array,
}

impl MessageGate {
    fn load(w: &mut Weights, p: &str) -> Result<Self> {
        let down = w.take(&format!("{p}.down_weight"))?;
        let forget = ops::f32(&w.take(&format!("{p}.forget_weight"))?)?;
        let gain = ops::f32(&w.take(&format!("{p}.gain_weight"))?)?;
        Self::from_parts(down, &forget, &gain)
    }

    fn from_parts(down: Array, forget: &Array, gain: &Array) -> Result<Self> {
        let n = forget.shape()[0];
        let fg = ops::concatenate(&[forget, gain], 0)?;
        Ok(Self {
            down,
            forget: ops::slice(&fg, 0, 0, n, 1)?,
            message: ops::slice(&fg, 0, n, 2 * n, 1)?,
            fg,
        })
    }

    /// One decode token: the forget and gain projections in one gemv, and
    /// the elementwise tail compiled (K-6). The same values as `gate`.
    fn forward_one(&self, kern: &Kernels, hidden: &Array, message: &Array) -> Result<Array> {
        let states = Self::states(hidden, message, &self.down)?;
        let fg = ops::linear(&states, &self.fg)?;
        Self::mix(kern, &fg, hidden, message)
    }

    /// The down projection of the normed hidden state and message, as
    /// float32.
    fn states(hidden: &Array, message: &Array, down: &Array) -> Result<Array> {
        let states = ops::concatenate(
            &[
                &unweighted_rms_norm(hidden)?,
                &unweighted_rms_norm(message)?,
            ],
            -1,
        )?;
        let states = ops::linear(&ops::astype(&states, down.dtype())?, down)?;
        ops::f32(&states)
    }

    /// `(1 - tanh(forget)) * hidden + tanh(gain) * message`, with forget
    /// and gain the two halves of `fg`'s last axis; compiled.
    fn mix(kern: &Kernels, fg: &Array, hidden: &Array, message: &Array) -> Result<Array> {
        kern.compiled("message_gate_mix", || {
            ops::Compiled::new(|a: &[Array]| {
                let n = a[0].shape()[a[0].ndim() - 1] / 2;
                let forget = ops::tanh(&ops::slice(&a[0], -1, 0, n, 1)?)?;
                let gain = ops::tanh(&ops::slice(&a[0], -1, n, 2 * n, 1)?)?;
                let kept = ops::mul(&ops::sub(&f32s(1.0), &forget)?, &a[1])?;
                Ok(vec![ops::add(&kept, &ops::mul(&gain, &a[2])?)?])
            })
        })
        .call1(&[fg, hidden, message])
    }

    /// One token goes through `forward_one`; a few (verification, short
    /// chunks) through one compiled function (K-6): the same ops with their
    /// elementwise chains fused.
    fn forward(&self, kern: &Kernels, hidden: &Array, message: &Array) -> Result<Array> {
        if hidden.shape()[..2] == [1, 1] {
            return self.forward_one(kern, hidden, message);
        }
        if hidden.shape()[1] > MAX_DECODE_QUERIES {
            return Self::gate(hidden, message, &self.down, &self.forget, &self.message);
        }
        let f = kern.compiled("message_gate", || {
            ops::Compiled::new(|a: &[Array]| {
                Ok(vec![Self::gate(&a[0], &a[1], &a[2], &a[3], &a[4])?])
            })
        });
        f.call1(&[hidden, message, &self.down, &self.forget, &self.message])
    }

    fn gate(
        hidden: &Array,
        message: &Array,
        down: &Array,
        forget: &Array,
        gain: &Array,
    ) -> Result<Array> {
        let states = ops::concatenate(
            &[
                &unweighted_rms_norm(hidden)?,
                &unweighted_rms_norm(message)?,
            ],
            -1,
        )?;
        let states = ops::linear(&ops::astype(&states, down.dtype())?, down)?;
        let states = ops::f32(&states)?;
        let forget = ops::tanh(&ops::linear(&states, forget)?)?;
        let gain = ops::tanh(&ops::linear(&states, gain)?)?;
        let kept = ops::mul(&ops::sub(&f32s(1.0), &forget)?, hidden)?;
        ops::add(&kept, &ops::mul(&gain, message)?)
    }
}

/// The q, k and v message gates of a core layer with their weights stacked:
/// one batched gemv per projection for the three in decode (K-6). MLX runs
/// each batch element through the same gemv as alone, so the values are
/// the same; the gates' own weights are views of these stacks.
struct GateStack {
    /// (3, r, 2 D).
    down: Array,
    /// float32 (3, 2 D, r).
    fg: Array,
}

impl GateStack {
    /// Stacks the first three gates and makes their weights views of the
    /// stacks.
    fn new(gates: &mut [MessageGate; 4]) -> Result<Self> {
        let down = ops::stack(&gates[..3].iter().map(|g| &g.down).collect::<Vec<_>>(), 0)?;
        let fg = ops::stack(&gates[..3].iter().map(|g| &g.fg).collect::<Vec<_>>(), 0)?;
        for (i, g) in gates[..3].iter_mut().enumerate() {
            let i = i as i32;
            let n = g.forget.shape()[0];
            g.down = ops::squeeze(&ops::slice(&down, 0, i, i + 1, 1)?, &[0])?;
            g.fg = ops::squeeze(&ops::slice(&fg, 0, i, i + 1, 1)?, &[0])?;
            g.forget = ops::slice(&g.fg, 0, 0, n, 1)?;
            g.message = ops::slice(&g.fg, 0, n, 2 * n, 1)?;
        }
        Ok(Self { down, fg })
    }

    /// The three gates for one decode token: hidden states and messages
    /// (1, 1, D) each, float32.
    fn forward(
        &self,
        kern: &Kernels,
        hidden: [&Array; 3],
        message: [&Array; 3],
    ) -> Result<[Array; 3]> {
        let d = hidden[0].shape()[2];
        let stack =
            |xs: [&Array; 3]| -> Result<Array> { ops::reshape(&ops::stack(&xs, 0)?, &[3, 1, d]) };
        let (h, m) = (stack(hidden)?, stack(message)?);
        let states = ops::concatenate(&[&unweighted_rms_norm(&h)?, &unweighted_rms_norm(&m)?], -1)?;
        let states = ops::matmul(
            &ops::astype(&states, self.down.dtype())?,
            &ops::transpose(&self.down, &[0, 2, 1])?,
        )?;
        let states = ops::f32(&states)?;
        let fg = ops::matmul(&states, &ops::transpose(&self.fg, &[0, 2, 1])?)?;
        let out = MessageGate::mix(kern, &fg, &h, &m)?;
        let row = |i: i32| -> Result<Array> {
            ops::reshape(&ops::slice(&out, 0, i, i + 1, 1)?, &[1, 1, d])
        };
        Ok([row(0)?, row(1)?, row(2)?])
    }

    fn arrays(&self) -> [&Array; 2] {
        [&self.down, &self.fg]
    }
}

struct Experts {
    gate_up: QuantizedSwitchLinear,
    down: QuantizedSwitchLinear,
    gate_softcap: f32,
    up_softcap: f32,
}

impl Experts {
    /// x: (B, L, D), indices: (B, L, K). Returns (B, L, K, D).
    fn forward(&self, kern: &Kernels, x: &Array, indices: &Array) -> Result<Array> {
        let [b, l, d] = x.shape()[..] else {
            unreachable!()
        };
        let top_k = indices.shape()[2];
        if b == 1 && (1..=MAX_DECODE_QUERIES).contains(&l) && l * top_k < 64 {
            // A few tokens (decode, speculative verification, short
            // chunks): each chosen expert is read once for all its tokens
            // (K-7; faster than gather_qmm even for one token). Bit for bit
            // gather_qmm's unsorted path, which the reference takes below
            // 64 routes (it sorts from 64 on, see below).
            let idx = ops::reshape(indices, &[l, top_k])?;
            let project = |p: &QuantizedSwitchLinear, x: &Array, per_route: bool| {
                kern.gather_qmv_shared(
                    x,
                    &p.weight,
                    &p.scales,
                    p.global_scale.as_ref(),
                    &idx,
                    per_route,
                    p.group_size,
                    p.bits,
                )
            };
            let gate_up = project(&self.gate_up, &ops::reshape(x, &[l, d])?, false)?;
            let n = gate_up.shape()[1];
            let gate = ops::slice(&gate_up, -1, 0, n, 2)?;
            let up = ops::slice(&gate_up, -1, 1, n, 2)?;
            let h = self.swiglu_short(kern, &gate, &up)?;
            let out = project(&self.down, &h, true)?;
            return ops::reshape(&out, &[1, l, top_k, -1]);
        }
        let x = ops::expand_dims(x, &[-2, -3])?;
        let do_sort = indices.size() >= 64;
        let (x, idx, inv_order) = if do_sort {
            // `_gather_sort`.
            let m = indices.shape()[indices.ndim() - 1];
            let flat = ops::flatten(indices, 0, -1)?;
            let order = ops::argsort(&flat, -1)?;
            let inv_order = ops::argsort(&order, -1)?;
            let rows = ops::floor_divide(&order, &ops::astype(&ops::i32s(m), order.dtype())?)?;
            let x = ops::take(&ops::flatten(&x, 0, -3)?, &rows, 0)?;
            (x, ops::take(&flat, &order, 0)?, Some(inv_order))
        } else {
            (x, indices.clone(), None)
        };
        let gate_up = self.gate_up.forward(&x, &idx, do_sort)?;
        let n = gate_up.shape()[gate_up.ndim() - 1];
        let gate = ops::slice(&gate_up, -1, 0, n, 2)?;
        let up = ops::slice(&gate_up, -1, 1, n, 2)?;
        let h = if l <= MAX_DECODE_QUERIES {
            self.swiglu_short(kern, &gate, &up)?
        } else {
            softcapped_swiglu(&gate, &up, self.gate_softcap, self.up_softcap)?
        };
        let mut out = self.down.forward(&h, &idx, do_sort)?;
        if let Some(inv) = inv_order {
            // `_scatter_unsort`.
            out = ops::take(&out, &inv, 0)?;
            out = ops::unflatten(&out, 0, indices.shape())?;
        }
        ops::squeeze(&out, &[-2])
    }
}

impl Experts {
    /// `softcapped_swiglu` as one compiled function (K-6).
    fn swiglu_short(&self, kern: &Kernels, gate: &Array, up: &Array) -> Result<Array> {
        let (g, u) = (self.gate_softcap, self.up_softcap);
        let name = format!("swiglu/{}/{}", g.to_bits(), u.to_bits());
        kern.compiled(&name, || {
            ops::Compiled::new(move |a: &[Array]| Ok(vec![softcapped_swiglu(&a[0], &a[1], g, u)?]))
        })
        .call1(&[gate, up])
    }
}

fn softcapped_swiglu(gate: &Array, up: &Array, gate_cap: f32, up_cap: f32) -> Result<Array> {
    let dtype = gate.dtype();
    let (gate, up) = (ops::f32(gate)?, ops::f32(up)?);
    let gate_capped = ops::mul(
        &ops::tanh(&ops::div(&gate, &f32s(gate_cap))?)?,
        &f32s(gate_cap),
    )?;
    let up_capped = ops::mul(&ops::tanh(&ops::div(&up, &f32s(up_cap))?)?, &f32s(up_cap))?;
    let out = ops::mul(&ops::mul(&gate_capped, &ops::sigmoid(&gate)?)?, &up_capped)?;
    ops::astype(&out, dtype)
}

struct Moe {
    /// float32 (E, D).
    router: Array,
    /// float32 (E,).
    bias: Array,
    experts: Experts,
    top_k: i32,
    route_scale: f32,
}

impl Moe {
    fn forward(&self, k: &Kernels, hidden: &Array) -> Result<Array> {
        let logits = ops::matmul(&ops::f32(hidden)?, &ops::t(&self.router)?)?;
        let (indices, weights) = k.expert_routing(
            &logits,
            &self.bias,
            self.top_k,
            ROUTER_SCORE_EPS,
            self.route_scale,
        )?;
        let expert_states = self.experts.forward(k, hidden, &indices)?;
        let dtype = hidden.dtype();
        let combine = move |a: &[Array]| -> Result<Vec<Array>> {
            let states = ops::mul(&ops::f32(&a[0])?, &ops::expand_dims(&a[1], &[-1])?)?;
            let out = ops::sum(&states, &[-2], false)?;
            Ok(vec![ops::astype(&out, dtype)?])
        };
        if hidden.shape()[1] <= MAX_DECODE_QUERIES {
            // One compiled function (K-6).
            k.compiled(&format!("moe_combine/{dtype:?}"), || {
                ops::Compiled::new(combine)
            })
            .call1(&[&expert_states, &weights])
        } else {
            Ok(combine(&[expert_states, weights])?.remove(0))
        }
    }
}

/// An RMS norm weight in the checkpoint's dtype and in float32. MLX's
/// rms_norm casts the weight to a float32 input's dtype on every call (one
/// kernel per norm); passing the float32 copy then gives the same result
/// without it (K-6).
struct NormWeight {
    w: Array,
    w32: Array,
}

impl NormWeight {
    fn new(w: Array) -> Result<Self> {
        Ok(Self {
            w32: ops::f32(&w)?,
            w,
        })
    }

    /// The weight MLX would use for input `x`.
    fn for_input(&self, x: &Array) -> &Array {
        if x.dtype() == Dtype::Float32 {
            &self.w32
        } else {
            &self.w
        }
    }

    fn arrays(&self) -> [&Array; 2] {
        [&self.w, &self.w32]
    }
}

enum InputNorms {
    Sliding(NormWeight),
    Full {
        q: NormWeight,
        k: NormWeight,
        v: NormWeight,
    },
}

struct Layer {
    index: usize,
    is_block_boundary: bool,
    /// float32 (Q, D); `None` for layer 0.
    attention_input: Option<Array>,
    /// float32 (1, D).
    mlp_input: Array,
    input_norms: InputNorms,
    post_attention_layernorm: NormWeight,
    post_attention_convolution: ShortConvolution,
    post_mlp_convolution: ShortConvolution,
    attn: Attention,
    moe: Moe,
    /// q, k, v and mlp gates, only in the loop core.
    message_gates: Option<[MessageGate; 4]>,
    /// The q, k and v gates stacked, in the loop core.
    qkv_gates: Option<GateStack>,
    eps: f32,
}

impl Layer {
    fn load(w: &mut Weights, cfg: &TextConfig, i: usize) -> Result<Self> {
        let p = format!("model.llm.layers.{i}");
        let d = cfg.hidden_size as i32;
        let k = cfg.sconv_kernel_size as i32;
        let sliding = !cfg.is_global_layer(i);
        let attention_input = if i > 0 {
            let q = w.take(&format!("model.llm.attn_res.attn_queries.{}", i - 1))?;
            Some(ops::reshape(&q, &[-1, d])?)
        } else {
            None
        };
        let mlp_input = ops::reshape(
            &w.take(&format!("model.llm.attn_res.mlp_queries.{i}"))?,
            &[-1, d],
        )?;
        let input_norms = if sliding {
            InputNorms::Sliding(NormWeight::new(w.take(&format!("{p}.attn_norm.weight"))?)?)
        } else {
            InputNorms::Full {
                q: NormWeight::new(w.take(&format!("{p}.q_input_norm.weight"))?)?,
                k: NormWeight::new(w.take(&format!("{p}.k_input_norm.weight"))?)?,
                v: NormWeight::new(w.take(&format!("{p}.v_input_norm.weight"))?)?,
            }
        };
        let core = cfg.loop_config.core_start..cfg.loop_config.core_end;
        let mut message_gates = if core.contains(&i) {
            let c = format!("{p}.controllers");
            Some([
                MessageGate::load(w, &format!("{c}.attn"))?,
                MessageGate::load(w, &format!("{c}.k"))?,
                MessageGate::load(w, &format!("{c}.v"))?,
                MessageGate::load(w, &format!("{c}.mlp"))?,
            ])
        } else {
            None
        };
        let qkv_gates = match message_gates.as_mut() {
            Some(g) => Some(GateStack::new(g)?),
            None => None,
        };
        let e = format!("{p}.mlp.experts");
        Ok(Self {
            index: i,
            is_block_boundary: i.is_multiple_of(cfg.n_layers_per_attn_res_block),
            attention_input,
            mlp_input,
            input_norms,
            post_attention_layernorm: NormWeight::new(w.take(&format!("{p}.mlp_norm.weight"))?)?,
            post_attention_convolution: ShortConvolution::load(
                w,
                &format!("{p}.attn_sconv.weight"),
                k,
            )?,
            post_mlp_convolution: ShortConvolution::load(w, &format!("{p}.mlp_sconv.weight"), k)?,
            attn: Attention::load(w, &format!("{p}.attn"), cfg, sliding)?,
            moe: Moe {
                router: ops::f32(&w.take(&format!("{p}.mlp.gate.weight"))?)?,
                bias: w.take(&format!("{p}.mlp.gate.bias"))?,
                experts: Experts {
                    gate_up: QuantizedSwitchLinear::load(w, &format!("{e}.w13_weight"), d)?,
                    down: QuantizedSwitchLinear::load(
                        w,
                        &format!("{e}.w2_weight"),
                        cfg.intermediate_size as i32,
                    )?,
                    gate_softcap: cfg.mlp_gate_softcap,
                    up_softcap: cfg.mlp_up_softcap,
                },
                top_k: cfg.num_experts_per_tok as i32,
                route_scale: cfg.route_scale,
            },
            message_gates,
            qkv_gates,
            eps: cfg.rms_norm_eps,
        })
    }

    fn mix_messages(
        &self,
        kern: &Kernels,
        q: &Array,
        k: &Array,
        v: &Array,
        messages: &Array,
    ) -> Result<[Array; 3]> {
        let g = self
            .message_gates
            .as_ref()
            .context("message gates outside the loop core")?;
        let m = |i: i32| -> Result<Array> {
            ops::squeeze(&ops::slice(messages, -2, i, i + 1, 1)?, &[-2])
        };
        if let Some(stack) = &self.qkv_gates
            && q.shape()[..2] == [1, 1]
        {
            return stack.forward(kern, [q, k, v], [&m(0)?, &m(1)?, &m(2)?]);
        }
        Ok([
            g[0].forward(kern, q, &m(0)?)?,
            g[1].forward(kern, k, &m(1)?)?,
            g[2].forward(kern, v, &m(2)?)?,
        ])
    }

    fn self_attention(
        &self,
        kern: &Kernels,
        hidden: &Array,
        attention_inputs: Option<&Array>,
        cache: &mut PassCache,
        messages: Option<&Array>,
    ) -> Result<Array> {
        let dtype = hidden.dtype();
        let row = |a: &Array, i: i32| -> Result<Array> {
            ops::squeeze(&ops::slice(a, -2, i, i + 1, 1)?, &[-2])
        };
        let norm = |x: &Array, w: &NormWeight| -> Result<Array> {
            ops::astype(&ops::rms_norm(x, Some(w.for_input(x)), self.eps)?, dtype)
        };
        match &self.input_norms {
            InputNorms::Sliding(w) => {
                let h = match attention_inputs {
                    Some(a) => row(a, 0)?,
                    None => hidden.clone(),
                };
                let [q, k, v] = match messages {
                    None => {
                        let n = norm(&h, w)?;
                        [n.clone(), n.clone(), n]
                    }
                    Some(m) => {
                        let [a, b, c] = self.mix_messages(kern, &h, &h, &h, m)?;
                        [norm(&a, w)?, norm(&b, w)?, norm(&c, w)?]
                    }
                };
                self.attn
                    .forward(kern, &q, &k, &v, messages.is_none(), cache)
            }
            InputNorms::Full {
                q: wq,
                k: wk,
                v: wv,
            } => {
                let (mut q, mut k, mut v) = match attention_inputs {
                    None => (hidden.clone(), hidden.clone(), hidden.clone()),
                    Some(a) => (row(a, 0)?, row(a, 1)?, row(a, 2)?),
                };
                if let Some(m) = messages {
                    [q, k, v] = self.mix_messages(kern, &q, &k, &v, m)?;
                }
                self.attn.forward(
                    kern,
                    &norm(&q, wq)?,
                    &norm(&k, wk)?,
                    &norm(&v, wv)?,
                    false,
                    cache,
                )
            }
        }
    }

    /// Returns the new hidden states and block states.
    fn forward(
        &self,
        kern: &Kernels,
        hidden: &Array,
        block_states: Option<&Array>,
        cache: &mut PassCache,
        messages: Option<&Array>,
        keep_windows: bool,
    ) -> Result<(Array, Array)> {
        let dtype = hidden.dtype();
        let block_residual = hidden;
        let normed_residual = unweighted_rms_norm(&ops::f32(block_residual)?)?;

        let attention_inputs = match &self.attention_input {
            Some(q) => {
                let blocks = block_states.context("block states before layer 1")?;
                Some(depth_attention(kern, blocks, &normed_residual, q)?)
            }
            None => None,
        };
        cache.verifying = keep_windows;
        let h = self.self_attention(kern, hidden, attention_inputs.as_ref(), cache, messages);
        cache.verifying = false;
        let h = h?;
        let (h, w2) = self.post_attention_convolution.forward(
            kern,
            &h,
            cache.conv[2].as_ref(),
            true,
            keep_windows,
        )?;
        cache.conv[2] = Some(w2);

        let mut h = ops::f32(&h)?;
        let residual;
        let block_states = if self.is_block_boundary {
            residual = h.clone();
            let block = ops::expand_dims(&normed_residual, &[0])?;
            match block_states {
                None => block,
                Some(b) => ops::concatenate(&[b, &block], 0)?,
            }
        } else {
            h = ops::add(&ops::f32(block_residual)?, &h)?;
            residual = h.clone();
            block_states.context("block states")?.clone()
        };

        let normed = unweighted_rms_norm(&h)?;
        let mut h = ops::squeeze(
            &ops::slice(
                &depth_attention(kern, &block_states, &normed, &self.mlp_input)?,
                -2,
                0,
                1,
                1,
            )?,
            &[-2],
        )?;
        if let Some(m) = messages {
            let g = self.message_gates.as_ref().context("message gates")?;
            h = g[3].forward(
                kern,
                &h,
                &ops::squeeze(&ops::slice(m, -2, 3, 4, 1)?, &[-2])?,
            )?;
        }
        let h = ops::astype(
            &ops::rms_norm(
                &h,
                Some(self.post_attention_layernorm.for_input(&h)),
                self.eps,
            )?,
            dtype,
        )?;
        let h = self.moe.forward(kern, &h)?;
        let (h, w3) = self.post_mlp_convolution.forward(
            kern,
            &h,
            cache.conv[3].as_ref(),
            true,
            keep_windows,
        )?;
        cache.conv[3] = Some(w3);

        let n = self.post_mlp_convolution.kernel_size - 1;
        if keep_windows {
            // Speculative verification: keep the windows so the states can be
            // rewound to any accepted token.
            let w: Vec<Array> = cache.conv.iter().flatten().cloned().collect();
            cache.windows = Some(
                w.try_into()
                    .map_err(|_| anyhow::anyhow!("missing convolution window"))?,
            );
        }
        for c in cache.conv.iter_mut() {
            let window = c.take().context("convolution window")?;
            *c = Some(conv_state(&window, n)?);
        }
        let out = ops::add(&residual, &ops::f32(&h)?)?;
        Ok((ops::astype(&out, dtype)?, block_states))
    }

    fn arrays(&self) -> Vec<&Array> {
        let a = &self.attn;
        let mut v = vec![
            &self.mlp_input,
            &self.post_attention_convolution.weight,
            &self.post_mlp_convolution.weight,
            &a.merged,
            &a.q_proj,
            &a.k_proj,
            &a.v_proj,
            &a.r_proj,
            &a.o_proj,
            &a.gate_proj,
            &a.sink_proj,
            &a.rel_proj,
            &a.q_scale,
            &a.rel_scale,
            &self.moe.router,
            &self.moe.bias,
        ];
        v.extend(self.attention_input.iter());
        v.extend(self.post_attention_layernorm.arrays());
        match &self.input_norms {
            InputNorms::Sliding(w) => v.extend(w.arrays()),
            InputNorms::Full { q, k, v: vv } => {
                v.extend(q.arrays());
                v.extend(k.arrays());
                v.extend(vv.arrays());
            }
        }
        for c in [&a.k_conv, &a.v_conv] {
            match c {
                KvConv::Static(s) => v.push(&s.weight),
                KvConv::Dynamic(d) => v.extend([&d.static_weight, &d.weight_proj]),
            }
        }
        for s in self.qkv_gates.iter() {
            v.extend(s.arrays());
        }
        for g in self.message_gates.iter().flatten() {
            v.extend([&g.fg, &g.down, &g.forget, &g.message]);
        }
        v.extend(self.moe.experts.gate_up.arrays());
        v.extend(self.moe.experts.down.arrays());
        v
    }
}

/// `_depth_attention` (GPU kernel path): attention of learned queries over
/// the block states and the current partial state.
fn depth_attention(
    kern: &Kernels,
    blocks: &Array,
    partial: &Array,
    query: &Array,
) -> Result<Array> {
    let query = ops::f32(query)?;
    let d = partial.shape()[partial.ndim() - 1];
    kern.depth_attention(blocks, partial, &query, (d as f32).powf(-0.5))
}

pub struct Model {
    pub config: ModelConfig,
    /// Calls with at least this many tokens evaluate pass by pass (bounded
    /// prefill); `i32::MAX` builds the whole forward as one graph.
    pub bounded_prefill_from: i32,
    kernels: Kernels,
    embed: Array,
    embed_norm: Array,
    layers: Vec<Layer>,
    final_query: Array,
    norm: Array,
    /// float32 (core layers, 4, D): q, k, v and mlp message queries.
    message_query: Array,
    lm_head: Array,
}

impl Model {
    /// Loads the text model from a Hugging Face snapshot and evaluates its
    /// weights (about 19 GB).
    pub fn load(dir: &Path) -> Result<Self> {
        ops::init();
        // Validates the index against the shard headers first.
        let ck = Checkpoint::open(dir)?;
        let config: ModelConfig = ck.read_json("config.json")?;
        config.validate()?;
        let cfg = &config.text_config;
        let d = cfg.hidden_size as i32;

        let mut files: Vec<_> = calamardo_ckpt::hub::shard_files(dir)?.into_iter().collect();
        files.sort();
        let mut map = HashMap::new();
        for f in files {
            for (name, a) in ops::load_safetensors(&dir.join(&f))? {
                if !is_skipped(&name) {
                    map.insert(name, a);
                }
            }
        }
        ensure!(dir.join(INDEX_FILE).exists(), "missing {INDEX_FILE}");
        let mut w = Weights(map);

        let kernels = Kernels::new()?;
        let embed = w.take("model.llm.embed.weight")?;
        let embed_norm = w.take("model.llm.embed_norm.weight")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = Layer::load(&mut w, cfg, i)?;
            ops::eval(&layer.arrays())?;
            layers.push(layer);
        }
        let final_query = ops::reshape(&w.take("model.llm.attn_res.output_query")?, &[-1, d])?;
        let norm = w.take("model.llm.norm.weight")?;
        let q = w.take("model.llm.middle_cycle_cross_pass_queries")?;
        let kv = w.take("model.llm.middle_cycle_cross_pass_kv_queries")?;
        // The checkpoint has [query, mlp] and [key, value] rows per looped layer.
        let q = ops::reshape(&q, &[-1, 2, d])?;
        let kv = ops::reshape(&kv, &[-1, 2, d])?;
        let message_query = ops::concatenate(
            &[
                &ops::slice(&q, 1, 0, 1, 1)?,
                &kv,
                &ops::slice(&q, 1, 1, 2, 1)?,
            ],
            1,
        )?;
        let lm_head = w.take("model.llm.unembed.weight")?;
        ops::eval(&[
            &embed,
            &embed_norm,
            &final_query,
            &norm,
            &message_query,
            &lm_head,
        ])?;
        if !w.0.is_empty() {
            let mut left: Vec<_> = w.0.keys().cloned().collect();
            left.sort();
            bail!("unused tensors: {}", left.join(", "));
        }
        Ok(Self {
            config,
            bounded_prefill_from: MAX_DECODE_QUERIES + 1,
            kernels,
            embed,
            embed_norm,
            layers,
            final_query,
            norm,
            message_query,
            lm_head,
        })
    }

    /// `model.embed`: embeddings of `tokens` (B, L) after `embed_norm`.
    pub fn embed(&self, tokens: &Array) -> Result<Array> {
        let h = ops::take(&self.embed, tokens, 0)?;
        ops::rms_norm(&h, Some(&self.embed_norm), self.text().rms_norm_eps)
    }

    /// Benchmarks only: the MoE of every pass in order, each added to its
    /// input. `hidden`: (1, L, D) bf16.
    #[doc(hidden)]
    pub fn moe_chain(&self, hidden: &Array) -> Result<Array> {
        let mut h = hidden.clone();
        for p in calamardo_ckpt::schedule::passes(&self.config.text_config) {
            let out = self.layers[p.layer].moe.forward(&self.kernels, &h)?;
            h = ops::add(&h, &ops::mul(&out, &ops::scalar(0.01, h.dtype())?)?)?;
        }
        Ok(h)
    }

    /// Tests only: layer `i`'s expert projections (weight, scales, global
    /// scale, group size, bits) for gate/up and down.
    #[doc(hidden)]
    pub fn expert_weights(&self, i: usize) -> [(Array, Array, Option<Array>, i32, i32); 2] {
        let e = &self.layers[i].moe.experts;
        [&e.gate_up, &e.down].map(|p| {
            (
                p.weight.clone(),
                p.scales.clone(),
                p.global_scale.clone(),
                p.group_size,
                p.bits,
            )
        })
    }

    /// The vocabulary projection (V, D).
    pub fn lm_head(&self) -> &Array {
        &self.lm_head
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    fn text(&self) -> &TextConfig {
        &self.config.text_config
    }

    /// One cache per pass, in pass order. Global passes store their keys and
    /// values in `kv`; sliding-window passes stay bf16.
    pub fn make_cache(&self, kv: KvFormat) -> Vec<PassCache> {
        calamardo_ckpt::schedule::passes(self.text())
            .into_iter()
            .map(|p| PassCache {
                attn: match p.attn {
                    calamardo_ckpt::schedule::AttnKind::Global if kv.is_quantized() => {
                        AttnCache::Quantized(QuantKvCache::new(kv))
                    }
                    calamardo_ckpt::schedule::AttnKind::Global => {
                        AttnCache::Global(KvCache::default())
                    }
                    calamardo_ckpt::schedule::AttnKind::Window => AttnCache::Window(
                        RotatingKvCache::new(self.text().sliding_window_size as i32),
                    ),
                },
                conv: [None, None, None, None],
                windows: None,
                verifying: false,
                pending: None,
            })
            .collect()
    }

    /// tokens: (1, L) int32. Returns float32 logits (1, L, unpadded vocab).
    pub fn forward(&self, tokens: &Array, cache: &mut [PassCache]) -> Result<Array> {
        Ok(self.forward_ex(tokens, cache, &[], false)?.0)
    }

    /// `forward`, also returning the hidden states entering the passes in
    /// `capture` (index `num_passes` is the final normalized output),
    /// concatenated on the last axis as the drafter reads them; and with
    /// `keep_windows`, keeping each pass's convolution windows for a
    /// speculative rollback.
    pub fn forward_ex(
        &self,
        tokens: &Array,
        cache: &mut [PassCache],
        capture: &[usize],
        keep_windows: bool,
    ) -> Result<(Array, Option<Array>)> {
        let cfg = self.text();
        let mut captured: Vec<(usize, Array)> = Vec::new();
        let mut pass = 0usize;
        let k = &self.kernels;
        let bounded = tokens.shape()[1] >= self.bounded_prefill_from;
        let h = ops::take(&self.embed, tokens, 0)?;
        let h = ops::rms_norm(&h, Some(&self.embed_norm), cfg.rms_norm_eps)?;
        let dtype = h.dtype();

        let lc = &cfg.loop_config;
        let mut caches = cache.iter_mut();
        let mut run = |layers: &[Layer],
                       mut h: Array,
                       mut bs: Option<Array>,
                       bank: Option<(&Array, &Array)>|
         -> Result<(Array, Option<Array>)> {
            for (i, layer) in layers.iter().enumerate() {
                let messages = match bank {
                    Some((blocks, partial)) => {
                        let q = ops::squeeze(
                            &ops::slice(&self.message_query, 0, i as i32, i as i32 + 1, 1)?,
                            &[0],
                        )?;
                        Some(depth_attention(k, blocks, partial, &q)?)
                    }
                    None => None,
                };
                if capture.contains(&pass) {
                    // DSpark taps the hidden state entering the pass.
                    captured.push((pass, h.clone()));
                }
                pass += 1;
                let c = caches.next().context("not enough caches")?;
                let (nh, nbs) =
                    layer.forward(k, &h, bs.as_ref(), c, messages.as_ref(), keep_windows)?;
                debug_assert!(layer.index < cfg.num_hidden_layers);
                if bounded {
                    // Bounded prefill (K-1, K-3): finish this pass before the
                    // next one starts, so only one pass's intermediates, and
                    // at most one dequantized history, are alive.
                    let mut live = vec![&nh, &nbs];
                    live.extend(c.arrays());
                    ops::eval(&live)?;
                }
                h = nh;
                bs = Some(nbs);
            }
            Ok((h, bs))
        };

        let prelude = run(&self.layers[..lc.core_start], h, None, None)?;
        let prelude_blocks = prelude.1.as_ref().context("prelude block states")?.shape()[0];
        let (mut h, mut bs) = prelude.clone();
        for pass in 0..lc.num_passes {
            let core = &self.layers[lc.core_start..lc.core_end];
            if pass == 0 {
                (h, bs) = run(core, h, bs, None)?;
            } else {
                let blocks = bs.as_ref().context("block states")?;
                let bank_blocks = ops::slice(blocks, 0, prelude_blocks + 1, blocks.shape()[0], 1)?;
                let bank_partial = unweighted_rms_norm(&ops::f32(&h)?)?;
                let (ph, pbs) = prelude.clone();
                (h, bs) = run(core, ph, pbs, Some((&bank_blocks, &bank_partial)))?;
            }
        }
        let (h, bs) = run(&self.layers[lc.core_end..], h, bs, None)?;
        let bs = bs.context("block states")?;

        let normed = unweighted_rms_norm(&ops::f32(&h)?)?;
        let h = ops::squeeze(
            &ops::slice(
                &depth_attention(k, &bs, &normed, &self.final_query)?,
                -2,
                0,
                1,
                1,
            )?,
            &[-2],
        )?;
        let h = ops::astype(
            &ops::rms_norm(&h, Some(&self.norm), cfg.rms_norm_eps)?,
            dtype,
        )?;

        if capture.contains(&pass) {
            captured.push((pass, h.clone()));
        }
        let hidden = if capture.is_empty() {
            None
        } else {
            let parts: Vec<&Array> = capture
                .iter()
                .map(|id| {
                    captured
                        .iter()
                        .find(|(p, _)| p == id)
                        .map(|(_, a)| a)
                        .context("capture id out of range")
                })
                .collect::<Result<_>>()?;
            Some(ops::concatenate(&parts, -1)?)
        };

        let logits = ops::f32(&ops::linear(&h, &self.lm_head)?)?;
        let cap = f32s(cfg.final_logit_softcapping);
        let logits = ops::mul(&ops::tanh(&ops::div(&logits, &cap)?)?, &cap)?;
        Ok((
            ops::slice(&logits, -1, 0, cfg.unpadded_vocab_size as i32, 1)?,
            hidden,
        ))
    }
}

#[cfg(test)]
mod tests {
    //! The compiled functions of K-6 against the plain ops, bit for bit.
    use super::*;

    fn rand(shape: &[i32], seed: u32, amp: f32, dtype: Dtype) -> Array {
        let n = shape.iter().product::<i32>() as usize;
        let mut s = seed;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * amp
            })
            .collect();
        ops::astype(&Array::from_slice(&v, shape), dtype).unwrap()
    }

    fn bits(a: &Array) -> Vec<u32> {
        let a = ops::f32(a).unwrap();
        ops::eval(&[&a]).unwrap();
        a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn compiled_bodies_match_plain_ops() {
        ops::init();
        let kern = Kernels::new().unwrap();
        let bf = Dtype::Bfloat16;
        for l in [1, 2, 5, 8] {
            // l2_clip with and without a scale (q, k, r shapes).
            let q = rand(&[1, l, 24, 128], 1, 40.0, bf);
            let scale = rand(&[24, 128], 2, 3.0, bf);
            for (x, s, t) in [
                (&q, Some(&scale), 30.0),
                (&q, None, 30.0),
                (&q, Some(&scale), 4.0),
            ] {
                assert_eq!(
                    bits(&l2_clip_short(&kern, x, s, t).unwrap()),
                    bits(&l2_clip(x, s, t).unwrap()),
                    "l2_clip L {l} threshold {t} scale {}",
                    s.is_some()
                );
            }
            // The experts' activation.
            let gate = rand(&[l * 8, 1152], 3, 30.0, bf);
            let up = rand(&[l * 8, 1152], 4, 30.0, bf);
            let experts = Experts {
                gate_up: dummy_switch(),
                down: dummy_switch(),
                gate_softcap: 4.0,
                up_softcap: 25.0,
            };
            assert_eq!(
                bits(&experts.swiglu_short(&kern, &gate, &up).unwrap()),
                bits(&softcapped_swiglu(&gate, &up, 4.0, 25.0).unwrap()),
                "swiglu L {l}"
            );
            // A message gate.
            let hidden = rand(&[1, l, 2304], 5, 4.0, Dtype::Float32);
            let message = rand(&[1, l, 2304], 6, 4.0, Dtype::Float32);
            let (gd, gf, gg) = (
                rand(&[64, 4608], 7, 0.1, bf),
                rand(&[2304, 64], 8, 0.5, Dtype::Float32),
                rand(&[2304, 64], 9, 0.5, Dtype::Float32),
            );
            let g = MessageGate::from_parts(gd.clone(), &gf, &gg).unwrap();
            assert_eq!(
                bits(&g.forward(&kern, &hidden, &message).unwrap()),
                bits(
                    &MessageGate::gate(&hidden, &message, &g.down, &g.forget, &g.message).unwrap()
                ),
                "message gate L {l}"
            );
        }
    }

    #[test]
    fn decode_message_gates_match_plain_ops() {
        ops::init();
        let kern = Kernels::new().unwrap();
        let bf = Dtype::Bfloat16;
        // The reference uses the raw weights, not views of the stacks.
        let plain: Vec<(Array, Array, Array)> = (0..4u32)
            .map(|i| {
                (
                    rand(&[64, 4608], 20 + i, 0.1, bf),
                    rand(&[2304, 64], 30 + i, 0.5, Dtype::Float32),
                    rand(&[2304, 64], 40 + i, 0.5, Dtype::Float32),
                )
            })
            .collect();
        let mut gates: [MessageGate; 4] = std::array::from_fn(|i| {
            let (d, f, g) = &plain[i];
            MessageGate::from_parts(d.clone(), f, g).unwrap()
        });
        let stack = GateStack::new(&mut gates).unwrap();
        let h: Vec<Array> = (0..3)
            .map(|i| rand(&[1, 1, 2304], 50 + i, 4.0, Dtype::Float32))
            .collect();
        let m: Vec<Array> = (0..3)
            .map(|i| rand(&[1, 1, 2304], 60 + i, 4.0, Dtype::Float32))
            .collect();
        let want: Vec<Vec<u32>> = (0..3)
            .map(|i| {
                let (d, f, g) = &plain[i];
                bits(&MessageGate::gate(&h[i], &m[i], d, f, g).unwrap())
            })
            .collect();
        let got = stack
            .forward(&kern, [&h[0], &h[1], &h[2]], [&m[0], &m[1], &m[2]])
            .unwrap();
        for i in 0..3 {
            assert_eq!(bits(&got[i]), want[i], "stacked gate {i}");
            assert_eq!(
                bits(&gates[i].forward(&kern, &h[i], &m[i]).unwrap()),
                want[i],
                "gate {i} alone"
            );
        }
        let (d, f, g) = &plain[3];
        assert_eq!(
            bits(&gates[3].forward(&kern, &h[0], &m[0]).unwrap()),
            bits(&MessageGate::gate(&h[0], &m[0], d, f, g).unwrap()),
            "mlp gate"
        );
        // Several tokens (compiled path, then plain ops) over the stacked views.
        for l in [2, 9] {
            let hl = rand(&[1, l, 2304], 70, 4.0, Dtype::Float32);
            let ml = rand(&[1, l, 2304], 71, 4.0, Dtype::Float32);
            let (d, f, g) = &plain[1];
            assert_eq!(
                bits(&gates[1].forward(&kern, &hl, &ml).unwrap()),
                bits(&MessageGate::gate(&hl, &ml, d, f, g).unwrap()),
                "gate 1 over {l} tokens"
            );
        }
    }

    fn dummy_switch() -> QuantizedSwitchLinear {
        QuantizedSwitchLinear {
            weight: Array::from_slice(&[0u32], &[1, 1, 1]),
            scales: Array::from_slice(&[0u8], &[1, 1, 1]),
            global_scale: None,
            group_size: 16,
            bits: 4,
            mode: c"nvfp4",
        }
    }
}
