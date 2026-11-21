//! Metal kernels of the reference (mlx-lm `models/inkling2/kernels.py` and
//! `kernels/expert_routing.py`), compiled at runtime through
//! `mx.fast.metal_kernel`. The sources and launch shapes are copied verbatim
//! so the outputs match the reference bit for bit.

use anyhow::Result;
use mlx_rs::{
    Array, Dtype,
    fast::{
        MetalKernel, MetalKernelApplyOptions, MetalKernelOptions, MetalKernelOutput, TemplateArg,
    },
};

use crate::ops::{self, f32s};

const DEPTH_ATTENTION_SOURCE: &str = r#"
    constexpr int N = K + 1;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    float acc[N * Q];
    for (int i = 0; i < N * Q; i++) {
        acc[i] = 0.0f;
    }
    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float qv = queries[j * D + d];
            for (int i = 0; i < N; i++) {
                acc[i * Q + j] += x[i] * qv;
            }
        }
    }

    threadgroup float buf[N * Q][TG / 32];
    for (int i = 0; i < N * Q; i++) {
        float v = simd_sum(acc[i]);
        if (lane == 0) {
            buf[i][sg] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float w[N * Q];
    for (int i = 0; i < N * Q; i++) {
        float t = 0.0f;
        for (int s = 0; s < TG / 32; s++) {
            t += buf[i][s];
        }
        w[i] = t * scale;
    }
    for (int j = 0; j < Q; j++) {
        float m = -INFINITY;
        for (int i = 0; i < N; i++) {
            m = max(m, w[i * Q + j]);
        }
        float z = 0.0f;
        for (int i = 0; i < N; i++) {
            w[i * Q + j] = metal::exp(w[i * Q + j] - m);
            z += w[i * Q + j];
        }
        for (int i = 0; i < N; i++) {
            w[i * Q + j] /= z;
        }
    }

    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float o = 0.0f;
            for (int i = 0; i < N; i++) {
                o += w[i * Q + j] * x[i];
            }
            out[(size_t(row) * Q + j) * D + d] = o;
        }
    }
"#;

const RELATIVE_ATTENTION_SOURCE: &str = r#"
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < S) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"#;

const ROUTING_SOURCE: &str = r#"
    constexpr int R = (E + 31) / 32;
    uint token = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    auto row = logits + size_t(token) * E;

    float score[R];
    float biased[R];
    for (int i = 0; i < R; i++) {
        int e = lane + 32 * i;
        score[i] = 0.0f;
        biased[i] = -INFINITY;
        if (e < E) {
            float x = row[e] * multiplier;
            float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
            score[i] = x < 0 ? y : 1.0f - y;
            biased[i] = score[i] + bias[e];
        }
    }

    float w[K];
    for (int k = 0; k < K; k++) {
        float best = -INFINITY;
        int best_e = E;
        for (int i = 0; i < R; i++) {
            int e = lane + 32 * i;
            if (biased[i] > best || (biased[i] == best && e < best_e)) {
                best = biased[i];
                best_e = e;
            }
        }
        float top = simd_max(best);
        int winner = simd_min(best == top ? best_e : E);
        float s = 0.0f;
        for (int i = 0; i < R; i++) {
            if (lane + 32 * i == winner) {
                s = score[i];
                biased[i] = -INFINITY;
            }
        }
        w[k] = simd_sum(s);
        if (lane == 0) {
            inds[size_t(token) * K + k] = winner;
        }
    }

    float total = 0.0f;
    for (int k = 0; k < K; k++) {
        total = w[k] + total;
    }
    if (lane == 0) {
        for (int k = 0; k < K; k++) {
            weights[size_t(token) * K + k] = w[k] / (total + eps) * scale;
        }
    }
"#;

/// Queries per threadgroup of the relative attention kernel.
const BQ: i32 = 32;
/// Threads per threadgroup of the relative attention kernel.
const TG: i32 = 128;

fn int(name: &str, v: i32) -> (String, TemplateArg) {
    (name.to_string(), TemplateArg::Int(v))
}

/// Threads for a threadgroup that handles one row of `dim` values.
fn row_threads(dim: i32) -> i32 {
    (dim / 8 / 32 * 32).clamp(32, 1024)
}

pub struct Kernels {
    depth: MetalKernel,
    relative: MetalKernel,
    routing: MetalKernel,
}

impl Kernels {
    pub fn new() -> Result<Self> {
        let contiguous = MetalKernelOptions {
            ensure_row_contiguous: true,
            ..Default::default()
        };
        Ok(Self {
            depth: ops::with_mlx_message(MetalKernel::new(
                "depth_attention",
                &["blocks", "partial", "queries", "scale"],
                &["out"],
                DEPTH_ATTENTION_SOURCE,
                contiguous.clone(),
            ))?,
            relative: ops::with_mlx_message(MetalKernel::new(
                "relative_attention",
                &["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
                &["out"],
                RELATIVE_ATTENTION_SOURCE,
                MetalKernelOptions::default(),
            ))?,
            routing: ops::with_mlx_message(MetalKernel::new(
                "expert_routing",
                &["logits", "bias", "multiplier", "eps", "scale"],
                &["inds", "weights"],
                ROUTING_SOURCE,
                contiguous,
            ))?,
        })
    }

    /// blocks: (K, ..., D) and partial: (..., D), float32, also the values.
    /// queries: float32 (Q, D). Returns (..., Q, D).
    pub fn depth_attention(
        &self,
        blocks: &Array,
        partial: &Array,
        queries: &Array,
        scale: f32,
    ) -> Result<Array> {
        let shape = partial.shape();
        let d = shape[shape.len() - 1];
        let q = queries.shape()[0];
        let tg = row_threads(d);
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.extend([q, d]);
        let rows = partial.size() as i32 / d;
        let scale = f32s(scale);
        // SAFETY: the launch shapes match the kernel's indexing, as in the reference.
        let out = unsafe {
            ops::with_mlx_message(self.depth.apply(
                &[blocks, partial, queries, &scale],
                &[MetalKernelOutput {
                    shape: out_shape,
                    dtype: Dtype::Float32,
                }],
                (rows * tg, 1, 1),
                (tg, 1, 1),
                MetalKernelApplyOptions {
                    template_args: vec![
                        int("K", blocks.shape()[0]),
                        int("D", d),
                        int("Q", q),
                        int("TG", tg),
                    ],
                    ..Default::default()
                },
            ))?
        };
        Ok(out.into_iter().next().unwrap())
    }

    /// q: (B, Hq, L, D), k and v: (B, Hkv, S, D), all bfloat16, with the
    /// queries at the last L keys. rel: (B, L, Hq, R), the bias at distances
    /// below R. sinks: (B, L, Hq). key_start: (B,), the first key of each
    /// row. Returns (B, L, Hq, D).
    #[allow(clippy::too_many_arguments)]
    pub fn relative_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        rel: &Array,
        sinks: &Array,
        scale: f32,
        window: i32,
        key_start: &Array,
    ) -> Result<Array> {
        let [b, hq, l, d] = q.shape()[..] else {
            unreachable!()
        };
        let (hkv, s) = (k.shape()[1], k.shape()[2]);
        let rel = ops::contiguous(rel)?;
        let sinks = ops::contiguous(&ops::f32(sinks)?)?;
        let key_start = ops::contiguous(&ops::astype(key_start, Dtype::Int32)?)?;
        let sizes = Array::from_slice(&[l, s], &[2]);
        let scale = f32s(scale);
        // SAFETY: the launch shapes match the kernel's indexing, as in the reference.
        let out = unsafe {
            ops::with_mlx_message(self.relative.apply(
                &[q, k, v, &rel, &sinks, &key_start, &sizes, &scale],
                &[MetalKernelOutput {
                    shape: vec![b, l, hq, d],
                    dtype: q.dtype(),
                }],
                ((l + BQ - 1) / BQ * TG, hq, b),
                (TG, 1, 1),
                MetalKernelApplyOptions {
                    template_args: vec![
                        ("T".to_string(), TemplateArg::Dtype(q.dtype())),
                        int("D", d),
                        int("HQ", hq),
                        int("HKV", hkv),
                        int("R", rel.shape()[3]),
                        int("WINDOW", window),
                        int("BQ", BQ),
                        int("TG", TG),
                    ],
                    ..Default::default()
                },
            ))?
        };
        Ok(out.into_iter().next().unwrap())
    }

    /// The experts and their weights from the router logits: (indices u32,
    /// weights f32), each `logits.shape[:-1] + (top_k,)`.
    pub fn expert_routing(
        &self,
        logits: &Array,
        bias: &Array,
        top_k: i32,
        score_eps: f32,
        route_scale: f32,
    ) -> Result<(Array, Array)> {
        let shape = logits.shape();
        let e = shape[shape.len() - 1];
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(top_k);
        let (multiplier, eps, scale) = (f32s(1.0), f32s(score_eps), f32s(route_scale));
        // SAFETY: one simdgroup per token, as in the reference.
        let mut out = unsafe {
            ops::with_mlx_message(self.routing.apply(
                &[logits, bias, &multiplier, &eps, &scale],
                &[
                    MetalKernelOutput {
                        shape: out_shape.clone(),
                        dtype: Dtype::Uint32,
                    },
                    MetalKernelOutput {
                        shape: out_shape,
                        dtype: Dtype::Float32,
                    },
                ],
                (logits.size() as i32 / e * 32, 1, 1),
                (256, 1, 1),
                MetalKernelApplyOptions {
                    template_args: vec![int("E", e), int("K", top_k)],
                    ..Default::default()
                },
            ))?
        }
        .into_iter();
        Ok((out.next().unwrap(), out.next().unwrap()))
    }
}
