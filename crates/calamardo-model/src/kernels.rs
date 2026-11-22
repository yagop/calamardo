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

// Decode attention straight over the affine-quantized global cache (K-4):
// no dequantized copy of the history. Pass 1 splits the keys of each KV head
// into NB parts; one simdgroup handles one part for the G query heads of that
// KV head and L queries (decode, or speculative verification up to 8), so the
// cache is read once per KV head. The simdgroup takes KPS keys per step, each
// over 32 / KPS lanes, so a dot product needs log2(32 / KPS) shuffles. Values are
// dequantized with the same bfloat expression as MLX's affine_dequantize
// (`scale * q + bias`), so they equal `mx.dequantize`. Pass 2 merges the parts
// and the sink logit.
const QUANT_DECODE_PARTIAL_SOURCE: &str = r#"
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int GL = G * L;
    constexpr int LPK = 32 / KPS;          // lanes per key
    constexpr int DPL = D / LPK;           // channels per lane
    constexpr int ROW_BYTES = D * BITS / 8;
    constexpr int NG = D / GROUP;
    constexpr int HQ = HKV * G;
    constexpr int WORDS = BITS == 16 ? 1 : DPL * BITS / 32; // u32 words per lane and row
    constexpr int PER_WORD = BITS == 16 ? 1 : 32 / BITS;
    const int S = sizes[0];
    const int cap = sizes[1];
    const uint lane = thread_index_in_simdgroup;
    const int sub = int(lane) / LPK;       // which key of the step
    const int li = int(lane) % LPK;
    const int part = threadgroup_position_in_grid.x;
    const int kvh = threadgroup_position_in_grid.y;
    const int d0 = li * DPL;
    const int gi = d0 / GROUP;

    float qv[GL][DPL];
    float m[GL];
    float ssum[GL];
    float o[GL][DPL];
    for (int qi = 0; qi < GL; qi++) {
        const int h = kvh * G + qi / L;
        const int l = qi % L;
        const device T* qp = q + (size_t(h) * L + l) * D + d0;
        for (int i = 0; i < DPL; i++) {
            qv[qi][i] = float(qp[i]);
            o[qi][i] = 0.0f;
        }
        m[qi] = -INFINITY;
        ssum[qi] = 0.0f;
    }

    const device uint8_t* kbytes = (const device uint8_t*)kw + size_t(kvh) * cap * ROW_BYTES;
    const device uint8_t* vbytes = (const device uint8_t*)vw + size_t(kvh) * cap * ROW_BYTES;
    const device T* ksc = ks + size_t(kvh) * cap * NG;
    const device T* kbi = kb + size_t(kvh) * cap * NG;
    const device T* vsc = vs + size_t(kvh) * cap * NG;
    const device T* vbi = vb + size_t(kvh) * cap * NG;

    const int per = (S + NB - 1) / NB;
    const int j0 = part * per;
    const int j1 = min(S, j0 + per);
    // All lanes run every step so the shuffles stay uniform; keys past the
    // part get a -inf score.
    for (int jb = j0; jb < j1; jb += KPS) {
        const int j = jb + sub;
        const bool valid = j < j1;
        float kf[DPL];
        float vf[DPL];
        if (valid && BITS == 16) {
            // A bf16 cache, read as is.
            const device T* kr = (const device T*)(kbytes + size_t(j) * ROW_BYTES) + d0;
            const device T* vr = (const device T*)(vbytes + size_t(j) * ROW_BYTES) + d0;
            for (int i = 0; i < DPL; i++) {
                kf[i] = float(kr[i]);
                vf[i] = float(vr[i]);
            }
        } else if (valid) {
            const T k_scale = ksc[size_t(j) * NG + gi];
            const T k_bias = kbi[size_t(j) * NG + gi];
            const T v_scale = vsc[size_t(j) * NG + gi];
            const T v_bias = vbi[size_t(j) * NG + gi];
            const float ksf = float(k_scale);
            const float kbf = float(k_bias);
            const float vsf = float(v_scale);
            const float vbf = float(v_bias);
            const device uint8_t* kp = kbytes + size_t(j) * ROW_BYTES + d0 * BITS / 8;
            const device uint8_t* vp = vbytes + size_t(j) * ROW_BYTES + d0 * BITS / 8;
            for (int w = 0; w < max(WORDS, 1); w++) {
                // A lane's channels fill whole u32 words, except 4 channels
                // of 4 bits: one u16.
                const uint kpack = WORDS > 0 ? ((const device uint*)kp)[w] : uint(*(const device ushort*)kp);
                const uint vpack = WORDS > 0 ? ((const device uint*)vp)[w] : uint(*(const device ushort*)vp);
                for (int e = 0; e < min(PER_WORD, DPL); e++) {
                    // Same expression and types as MLX's affine_dequantize.
                    const uint8_t kq = (kpack >> (BITS * e)) & ((1u << BITS) - 1);
                    const uint8_t vq = (vpack >> (BITS * e)) & ((1u << BITS) - 1);
                    // MLX's `scale * q + bias` in bfloat evaluates in float
                    // and rounds once to bfloat (checked bit for bit against
                    // mx.dequantize by the tests); doing that explicitly
                    // avoids the slower bfloat arithmetic.
                    const float kd = float(T(ksf * float(kq) + kbf));
                    const float vd = float(T(vsf * float(vq) + vbf));
                    kf[w * PER_WORD + e] = float(kd);
                    vf[w * PER_WORD + e] = float(vd);
                }
            }
        } else {
            for (int i = 0; i < DPL; i++) {
                kf[i] = 0.0f;
                vf[i] = 0.0f;
            }
        }
        for (int qi = 0; qi < GL; qi++) {
            float dot = 0.0f;
            for (int i = 0; i < DPL; i++) {
                dot += qv[qi][i] * kf[i];
            }
            for (int off = LPK / 2; off > 0; off /= 2) {
                dot += simd_shuffle_xor(dot, ushort(off));
            }
            const int l = qi % L;
            const int p = S - L + l;
            if (!valid || j > p) {
                continue;
            }
            const int h = kvh * G + qi / L;
            float sc = dot * scale;
            const int dist = p - j;
            if (dist < R) {
                sc += float(rel[(size_t(l) * HQ + h) * R + dist]);
            }
            sc *= LOG2E;
            const float nm = max(m[qi], sc);
            const float f = fast::exp2(m[qi] - nm);
            const float pr = fast::exp2(sc - nm);
            ssum[qi] = ssum[qi] * f + pr;
            for (int i = 0; i < DPL; i++) {
                o[qi][i] = o[qi][i] * f + pr * vf[i];
            }
            m[qi] = nm;
        }
    }

    // Merge the KPS keys of each step: lanes with the same `li` hold the same
    // channels.
    for (int qi = 0; qi < GL; qi++) {
        for (int off = LPK; off < 32; off *= 2) {
            const float m2 = simd_shuffle_xor(m[qi], ushort(off));
            const float s2 = simd_shuffle_xor(ssum[qi], ushort(off));
            const float nm = max(m[qi], m2);
            const float fa = m[qi] == -INFINITY ? 0.0f : fast::exp2(m[qi] - nm);
            const float fb = m2 == -INFINITY ? 0.0f : fast::exp2(m2 - nm);
            ssum[qi] = ssum[qi] * fa + s2 * fb;
            for (int i = 0; i < DPL; i++) {
                const float o2 = simd_shuffle_xor(o[qi][i], ushort(off));
                o[qi][i] = o[qi][i] * fa + o2 * fb;
            }
            m[qi] = nm;
        }
        const size_t base = (size_t(kvh) * NB + part) * GL + qi;
        if (lane == 0) {
            pm[base] = m[qi];
            ps[base] = ssum[qi];
        }
        if (sub == 0) {
            for (int i = 0; i < DPL; i++) {
                po[base * D + d0 + i] = o[qi][i];
            }
        }
    }
"#;

const QUANT_DECODE_REDUCE_SOURCE: &str = r#"
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int GL = G * L;
    constexpr int DPL = D / 32;
    constexpr int HQ = HKV * G;
    const uint lane = thread_index_in_simdgroup;
    const int t = threadgroup_position_in_grid.x;
    const int kvh = t / GL;
    const int qi = t % GL;
    const int h = kvh * G + qi / L;
    const int l = qi % L;
    const int d0 = int(lane) * DPL;

    const float sink = float(sinks[size_t(l) * HQ + h]) * LOG2E;
    float mx = sink;
    for (int part = 0; part < NB; part++) {
        mx = max(mx, pm[(size_t(kvh) * NB + part) * GL + qi]);
    }
    float total = fast::exp2(sink - mx);
    float acc[DPL];
    for (int i = 0; i < DPL; i++) {
        acc[i] = 0.0f;
    }
    for (int part = 0; part < NB; part++) {
        const size_t base = (size_t(kvh) * NB + part) * GL + qi;
        const float w = fast::exp2(pm[base] - mx);
        total += w * ps[base];
        for (int i = 0; i < DPL; i++) {
            acc[i] += w * po[base * D + d0 + i];
        }
    }
    for (int i = 0; i < DPL; i++) {
        out[(size_t(l) * HQ + h) * D + d0 + i] = T(acc[i] / total);
    }
"#;

// Pass 1 of the same attention for several queries (speculative
// verification, the drafter's block) with simdgroup matrices. The per-lane
// kernel above keeps G * L query rows in registers and spills from 2 queries
// on (8-bit at 516k tokens: 3.6 ms per pass for one query, 16.7 for 3, 168
// for 7). Here one simdgroup handles 8 query rows over one key part of one
// KV head, KS keys per step, and reads the keys and values straight from the
// cache into its matrix registers, dequantized as above (exactly
// `mx.dequantize`): no threadgroup memory or barriers, so many simdgroups
// stay in flight to hide the memory latency (staging tiles in threadgroup
// memory was 2-4x slower). Scores come from 8x8 float matrix products, the
// online softmax runs on the score registers and P V accumulates into the
// output tiles: 3.0 ms for 2 queries, 4.6 for 3-5, 6.6 for 7 (see
// bench_speculative_attention). The loops over the head dimension must stay
// fully unrolled: an index into the matrix arrays that is not a constant
// sends them to memory (4-5x slower). The partial results have the layout of
// the per-lane kernel, so the same pass 2 merges them. CAUSAL = 0 lets every
// query see every key (the drafter's bidirectional block); R = 0 skips the
// bias; WINDOW > 0 hides keys that many positions back or more. The body
// reads key v through `key_row`, `val_row` and `key_pos`, which a prelude
// defines: rows of a contiguous cache (VERIFY_PLAIN_KEYS), or a sliding
// window's ring buffer followed by the keys being verified
// (VERIFY_RING_KEYS, `window_verify_attention`).
fn quant_verify_source() -> String {
    format!("{VERIFY_HEAD}{VERIFY_PLAIN_KEYS}{VERIFY_BODY}")
}

fn window_verify_source() -> String {
    format!("{VERIFY_HEAD}{VERIFY_RING_KEYS}{VERIFY_BODY}")
}

const VERIFY_HEAD: &str = r#"
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int GL = G * L;
    constexpr int ROW_BYTES = D * BITS / 8;
    constexpr int NG = BITS == 16 ? 1 : D / GROUP;
    constexpr int HQ = HKV * G;
    constexpr int DT = D / 8;              // 8-wide tiles along a row
    constexpr int MASK = BITS == 16 ? 0 : (1 << BITS) - 1;
    const int S = sizes[0];
    const int cap = sizes[1];
    const int part = threadgroup_position_in_grid.x;
    const int kvh = threadgroup_position_in_grid.y;
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);

    // This lane's two elements of every 8x8 matrix: row fm, columns fn and
    // fn + 1 (the simdgroup matrix layout). The lanes of a row differ in
    // bits 0 and 3.
    const int fm = ((lane / 4) & 4) + ((lane / 2) % 4);
    const int fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int qr = sg * 8 + fm;            // this lane's query row
    const bool row_ok = qr < GL;
    const int hr = row_ok ? kvh * G + qr / L : 0;
    const int lr = row_ok ? qr % L : 0;
    const int pos = S - L + lr;            // position of the row's query

    simdgroup_matrix<float, 8, 8> qf[DT];
    simdgroup_matrix<float, 8, 8> o[DT];
    _Pragma("clang loop unroll(full)")
    for (int d = 0; d < DT; d++) {
        const device T* qp = q + (size_t(hr) * L + lr) * D + d * 8 + fn;
        qf[d].thread_elements()[0] = row_ok ? float(qp[0]) : 0.0f;
        qf[d].thread_elements()[1] = row_ok ? float(qp[1]) : 0.0f;
        o[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = -INFINITY;
    float ssum = 0.0f;

"#;

const VERIFY_PLAIN_KEYS: &str = r#"    const device uint8_t* kbytes = (const device uint8_t*)kw + size_t(kvh) * cap * ROW_BYTES;
    const device uint8_t* vbytes = (const device uint8_t*)vw + size_t(kvh) * cap * ROW_BYTES;
    const device T* ksc = ks + size_t(kvh) * cap * NG;
    const device T* kbi = kb + size_t(kvh) * cap * NG;
    const device T* vsc = vs + size_t(kvh) * cap * NG;
    const device T* vbi = vb + size_t(kvh) * cap * NG;

    // Key v is row v of the cache, at position v.
    auto key_row = [&](int v) { return kbytes + size_t(v) * ROW_BYTES; };
    auto val_row = [&](int v) { return vbytes + size_t(v) * ROW_BYTES; };
    auto key_pos = [&](int v) { return v; };

"#;

const VERIFY_RING_KEYS: &str = r#"    // A bf16 sliding window: keys 0..rlen-1 are the ring buffer's filled
    // slots (slot v was written (ridx - 1 - v) mod rlen steps ago), then the
    // L keys of this verification, not yet in the buffer. Positions count
    // from the oldest slot, so the queries are at rlen..rlen+L-1.
    const int rlen = sizes[2];
    const int ridx = sizes[3];
    const device uint8_t* kbytes = (const device uint8_t*)kw + size_t(kvh) * cap * ROW_BYTES;
    const device uint8_t* vbytes = (const device uint8_t*)vw + size_t(kvh) * cap * ROW_BYTES;
    const device uint8_t* nkb = (const device uint8_t*)nk + size_t(kvh) * L * ROW_BYTES;
    const device uint8_t* nvb = (const device uint8_t*)nv + size_t(kvh) * L * ROW_BYTES;
    // Not read for 16 bits.
    const device T* ksc = q;
    const device T* kbi = q;
    const device T* vsc = q;
    const device T* vbi = q;
    auto key_row = [&](int v) {
        return v < rlen ? kbytes + size_t(v) * ROW_BYTES : nkb + size_t(v - rlen) * ROW_BYTES;
    };
    auto val_row = [&](int v) {
        return v < rlen ? vbytes + size_t(v) * ROW_BYTES : nvb + size_t(v - rlen) * ROW_BYTES;
    };
    auto key_pos = [&](int v) {
        return v < rlen ? rlen - 1 - (ridx - 1 - v + rlen) % rlen : v;
    };

"#;

const VERIFY_BODY: &str = r#"    const int per = (S + NB - 1) / NB;
    const int j0 = part * per;
    const int j1 = min(S, j0 + per);
    constexpr int DG = BITS == 16 ? DT : GROUP / 8; // 8-wide tiles per quantization group
    for (int jt = j0; jt < j1; jt += KS) {
        simdgroup_matrix<float, 8, 8> sc[KS / 8];
        float tmax = -INFINITY;
        _Pragma("clang loop unroll(full)")
        for (int c8 = 0; c8 < KS / 8; c8++) {
            // Keys of this lane's K^T elements (fn, fn + 1), clamped to the
            // part: out-of-range keys get a -inf score below.
            const int ka = min(jt + c8 * 8 + fn, j1 - 1);
            const int kb2 = min(jt + c8 * 8 + fn + 1, j1 - 1);
            const device uint8_t* ra = key_row(ka);
            const device uint8_t* rb = key_row(kb2);
            sc[c8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            float sa = 0.0f, ba = 0.0f, sb = 0.0f, bb = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (int d = 0; d < DT; d++) {
                if (BITS != 16 && d % DG == 0) {
                    const int g = d / DG;
                    sa = float(ksc[size_t(ka) * NG + g]);
                    ba = float(kbi[size_t(ka) * NG + g]);
                    sb = float(ksc[size_t(kb2) * NG + g]);
                    bb = float(kbi[size_t(kb2) * NG + g]);
                }
                // K^T: element (dim fm, key fn) is K[fn][fm].
                const int col = d * 8 + fm;
                simdgroup_matrix<float, 8, 8> kt;
                if (BITS == 16) {
                    kt.thread_elements()[0] = float(((const device T*)ra)[col]);
                    kt.thread_elements()[1] = float(((const device T*)rb)[col]);
                } else {
                    const int sh = (col % (32 / BITS)) * BITS;
                    const uint wa = ((const device uint*)ra)[col * BITS / 32];
                    const uint wb = ((const device uint*)rb)[col * BITS / 32];
                    kt.thread_elements()[0] = float(T(sa * float((wa >> sh) & MASK) + ba));
                    kt.thread_elements()[1] = float(T(sb * float((wb >> sh) & MASK) + bb));
                }
                simdgroup_multiply_accumulate(sc[c8], qf[d], kt, sc[c8]);
            }
            _Pragma("clang loop unroll(full)")
            for (int e = 0; e < 2; e++) {
                const int jj = jt + c8 * 8 + fn + e;
                const int kp = key_pos(jj);
                float s = -INFINITY;
                if (row_ok && jj < j1 && (CAUSAL == 0 || kp <= pos) && (WINDOW == 0 || pos - kp < WINDOW)) {
                    s = sc[c8].thread_elements()[e] * scale;
                    if (R > 0) {
                        const int dist = pos - kp;
                        if (dist >= 0 && dist < R) {
                            s += float(rel[(size_t(lr) * HQ + hr) * R + dist]);
                        }
                    }
                    s *= LOG2E;
                }
                sc[c8].thread_elements()[e] = s;
                tmax = max(tmax, s);
            }
        }
        tmax = max(tmax, simd_shuffle_xor(tmax, ushort(1)));
        tmax = max(tmax, simd_shuffle_xor(tmax, ushort(8)));
        const float nm = max(m, tmax);
        // A row with no valid key yet keeps zeros (no NaN from -inf - -inf).
        const float f = nm == -INFINITY ? 1.0f : fast::exp2(m - nm);
        float psum = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (int c8 = 0; c8 < KS / 8; c8++) {
            _Pragma("clang loop unroll(full)")
            for (int e = 0; e < 2; e++) {
                const float p = nm == -INFINITY ? 0.0f : fast::exp2(sc[c8].thread_elements()[e] - nm);
                sc[c8].thread_elements()[e] = p;
                psum += p;
            }
        }
        psum += simd_shuffle_xor(psum, ushort(1));
        psum += simd_shuffle_xor(psum, ushort(8));
        ssum = ssum * f + psum;
        m = nm;
        _Pragma("clang loop unroll(full)")
        for (int d = 0; d < DT; d++) {
            o[d].thread_elements()[0] *= f;
            o[d].thread_elements()[1] *= f;
        }
        _Pragma("clang loop unroll(full)")
        for (int c8 = 0; c8 < KS / 8; c8++) {
            // V: element (key fm, dim fn) and (key fm, dim fn + 1).
            const int kvr = min(jt + c8 * 8 + fm, j1 - 1);
            const device uint8_t* vrow = val_row(kvr);
            float vsf = 0.0f, vbf = 0.0f;
            _Pragma("clang loop unroll(full)")
            for (int d = 0; d < DT; d++) {
                if (BITS != 16 && d % DG == 0) {
                    vsf = float(vsc[size_t(kvr) * NG + d / DG]);
                    vbf = float(vbi[size_t(kvr) * NG + d / DG]);
                }
                const int col = d * 8 + fn;
                simdgroup_matrix<float, 8, 8> vt;
                if (BITS == 16) {
                    const vec<T, 2> v2 = *(const device vec<T, 2>*)((const device T*)vrow + col);
                    vt.thread_elements()[0] = float(v2.x);
                    vt.thread_elements()[1] = float(v2.y);
                } else {
                    const int sh = (col % (32 / BITS)) * BITS;
                    const uint w = ((const device uint*)vrow)[col * BITS / 32];
                    vt.thread_elements()[0] = float(T(vsf * float((w >> sh) & MASK) + vbf));
                    vt.thread_elements()[1] = float(T(vsf * float((w >> (sh + BITS)) & MASK) + vbf));
                }
                simdgroup_multiply_accumulate(o[d], sc[c8], vt, o[d]);
            }
        }
    }

    const size_t base = (size_t(kvh) * NB + part) * GL + qr;
    if (row_ok) {
        if ((lane & 9) == 0) {
            pm[base] = m;
            ps[base] = ssum;
        }
        _Pragma("clang loop unroll(full)")
        for (int d = 0; d < DT; d++) {
            po[base * D + d * 8 + fn] = o[d].thread_elements()[0];
            po[base * D + d * 8 + fn + 1] = o[d].thread_elements()[1];
        }
    }
"#;

// K-7: the quantized experts of 2 to 7 tokens (speculative verification,
// short prefill chunks) reading each expert's weights once. MLX's gather_qmm runs
// one matrix-vector product per route, so an expert chosen by several tokens
// is read once per token (4 identical tokens cost 18.1 ms over the 48
// passes against 7.9 for one). One threadgroup per route: the first route
// of each expert takes all the routes on it (at most one per token, since a
// token never picks an expert twice) and the others exit. The arithmetic
// is MLX's, lane for lane: its fp_qmv_impl (one u32 of values per lane, in
// blocks of 32 lanes, with a last partial block where lanes past the end add
// nothing) or, when MLX takes it (N % 8 == 0 and K a multiple of
// 2 * 32 * 32 / BITS), its fp_qmv_fast_impl (two u32 per lane, PPT = 2):
// `scale * (x0 w0 + x1 w1 + x2 w2 + x3 w3 ...)` accumulated per block,
// simd_sum, then the global scale, so the results equal gather_qmm's bit for
// bit (tested with every format the checkpoint uses).
// Rows of `x`: one per token (XR = 0, the gate/up projection) or one per
// route (XR = 1, the down projection).
const GATHER_QMV_SHARED_HEADER: &str = r#"
inline float e2m1_to_float(uint bits) {
    half c = as_type<half>(ushort((bits & 7) << 9));
    c *= 16384.0;
    return float((bits & 8) ? -c : c);
}
inline float e4m3_to_float(uint bits) {
    ushort v = bits & 127;
    ushort sign_bit = ushort((bits >> 7) & 1) << 15;
    ushort u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half c = as_type<half>(u);
    return float(half(c * 256.0));
}
inline float e8m0_to_float(uint bits) {
    uint out = bits == 0 ? 0x400000 : (bits << 23);
    return as_type<float>(out);
}
"#;

const GATHER_QMV_SHARED_SOURCE: &str = r#"
    constexpr int VPT = 32 / BITS * PPT;    // values per lane and block (PPT u32)
    constexpr int BLOCK = VPT * 32;
    constexpr int ROW_BYTES = K * BITS / 8;
    constexpr int KG = K / GROUP;
    constexpr int SSPT = GROUP / VPT;       // lanes per scale
    static_assert(K % VPT == 0 && N % 8 == 0, "unsupported shape");
    const int r = int(threadgroup_position_in_grid.y);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    static_assert(NR <= 64 && MAXT <= 8, "at most 64 routes");
    const uint e = indices[r];
    // Every route on this expert, as a 64-bit mask from one vote per half.
    const bool m0 = lane < NR && indices[lane] == e;
    const bool m1 = lane + 32 < NR && indices[min(lane + 32, NR - 1)] == e;
    const ulong lo = static_cast<ulong>(static_cast<simd_vote::vote_t>(simd_ballot(m0))) & 0xffffffffUL;
    const ulong hi = static_cast<ulong>(static_cast<simd_vote::vote_t>(simd_ballot(m1))) & 0xffffffffUL;
    ulong mask = lo | (hi << 32);
    // Only the expert's first route does the work.
    if (int(ctz(mask)) != r) {
        return;
    }
    int routes[MAXT];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < MAXT; j++) {
        routes[j] = mask != 0 ? int(ctz(mask)) : -1;
        mask &= mask - 1;
    }

    const int out_row = int(threadgroup_position_in_grid.x) * 8 + sg * 4;
    const device uint8_t* ws = (const device uint8_t*)w + (size_t(e) * N + out_row) * ROW_BYTES + lane * 4 * PPT;
    const device uint8_t* sl = scales + (size_t(e) * N + out_row) * KG + lane / SSPT;
    float result[MAXT][4];
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < MAXT; j++) {
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < 4; row++) {
            result[j][row] = 0.0f;
        }
    }

    for (int k = 0; k < K; k += BLOCK) {
        // A last partial block, as MLX's: lanes past the end add nothing.
        if (k + lane * VPT >= K) {
            break;
        }
        // The 4 rows' weights of this block, decoded once for all routes.
        float wd[4][VPT];
        float s[4];
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < 4; row++) {
            const device uint8_t* wl = ws + row * ROW_BYTES + k * BITS / 8;
            const uint sbits = sl[row * KG + k / GROUP];
            s[row] = GROUP == 16 ? e4m3_to_float(sbits) : e8m0_to_float(sbits);
            if (BITS == 4) {
                const device uint16_t* w16 = (const device uint16_t*)wl;
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < VPT / 4; i++) {
                    const uint h = w16[i];
                    wd[row][4 * i] = e2m1_to_float(h & 15);
                    wd[row][4 * i + 1] = e2m1_to_float((h >> 4) & 15);
                    wd[row][4 * i + 2] = e2m1_to_float((h >> 8) & 15);
                    wd[row][4 * i + 3] = e2m1_to_float((h >> 12) & 15);
                }
            } else {
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < VPT; i++) {
                    wd[row][i] = e4m3_to_float(wl[i]);
                }
            }
        }
        // One route at a time: only its inputs are held.
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < MAXT; j++) {
            if (routes[j] < 0) {
                continue;
            }
            const int xrow = XR ? routes[j] : routes[j] / TOPK;
            const device T* xp = x + size_t(xrow) * K + k + lane * VPT;
            float xt[VPT];
            _Pragma("clang loop unroll(full)")
            for (int i = 0; i < VPT; i++) {
                xt[i] = float(xp[i]);
            }
            _Pragma("clang loop unroll(full)")
            for (int row = 0; row < 4; row++) {
                float accum = 0.0f;
                if (BITS == 4) {
                    _Pragma("clang loop unroll(full)")
                    for (int i = 0; i < VPT / 4; i++) {
                        accum += (xt[4 * i] * wd[row][4 * i] + xt[4 * i + 1] * wd[row][4 * i + 1] +
                                  xt[4 * i + 2] * wd[row][4 * i + 2] + xt[4 * i + 3] * wd[row][4 * i + 3]);
                    }
                } else {
                    _Pragma("clang loop unroll(full)")
                    for (int i = 0; i < VPT; i++) {
                        accum += xt[i] * wd[row][i];
                    }
                }
                // Two statements, as MLX's qdot returns `scale * accum`:
                // one expression would let the compiler fuse them (fma) and
                // round differently.
                const float scaled = s[row] * accum;
                result[j][row] += scaled;
            }
        }
    }

    const float inv_scale_enc = HAS_GS ? gs[e] / (448.0f * 6.0f) : 1.0f;
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < MAXT; j++) {
        _Pragma("clang loop unroll(full)")
        for (int row = 0; row < 4; row++) {
            const float v = simd_sum(result[j][row]);
            if (lane == 0 && routes[j] >= 0) {
                y[size_t(routes[j]) * N + out_row + row] = HAS_GS ? T(v * inv_scale_enc) : T(v);
            }
        }
    }
"#;

// K-6: the additive mask of a decode step's attention (MLX's fused SDPA), in
// one dispatch instead of the ~30 small ops that build it: for key slot j
// (rotated by `rot` in a sliding window's ring buffer) at position kp, the
// query at end - 1 sees it if 0 <= distance, kp >= 0 and (no window or
// distance < window); then the value is the relative bias at that distance
// (0 from R on), else -inf. Only selections and copies of bfloat values, so
// the mask is the one the ops build, bit for bit.
const DECODE_MASK_SOURCE: &str = r#"
    const int j = int(thread_position_in_grid.x);
    const int h = int(thread_position_in_grid.y);
    const int s = params[0];
    const int rot = params[1];
    const int end = params[2];
    const int window = params[3];
    if (j >= s) {
        return;
    }
    int slot = (j - rot) % s;
    if (slot < 0) {
        slot += s;
    }
    const int kp = slot + end - s;
    const int dist = end - 1 - kp;
    const bool visible = dist >= 0 && kp >= 0 && (window == 0 || dist < window);
    T out;
    if (!visible) {
        out = T(-INFINITY);
    } else if (dist < R) {
        out = T(pb[size_t(h) * R + dist]);
    } else {
        out = T(0.0f);
    }
    mask[size_t(h) * s + j] = out;
"#;

// K-6: a decode step of a static short convolution in one dispatch: the
// window (float state rows, then the input), MLX's depthwise conv1d over it
// in bfloat (`acc += float(in) * w` tap by tap, as its depthwise_conv_1d
// kernel; products of two bfloats are exact in float, so only this order
// matters), the residual and the scaling, as the ops do one by one.
// Outputs the result (in T, or float32 holding the same T-rounded value,
// for callers that would cast it to float32 next) and the next state (the
// window's last K - 1 rows, in a buffer of its own).
const SHORT_CONV_SOURCE: &str = r#"
    const int c = int(thread_position_in_grid.x);
    if (c >= C) {
        return;
    }
    const float xf = float(x[c]);
    float acc = 0.0f;
    for (int i = 0; i < K; i++) {
        const float v = i < K - 1 ? state[i * C + c] : xf;
        if (i > 0) {
            next[(i - 1) * C + c] = v;
        }
        acc += float(T(v)) * float(w[c * K + i]);
    }
    const float conv = float(T(acc));
    const float y = conv + xf;
    // Rounded to T once; stored as T, or exactly as that value in float.
    out[c] = OT(T(y / denom));
"#;

/// Most key parts per KV head of the quantized decode attention (K-4). On M4
/// Max, 512 parts of this kernel beat the 64 that suit MLX's own attention
/// (1.38 vs 2.08 ms per pass at 160k tokens, see `bench_decode_160k`).
pub const DECODE_PARTS: i32 = 512;
/// Fewest keys per part, so short contexts do not pay for many parts.
const KEYS_PER_PART: i32 = 256;
/// Keys per part of the window verification (about 1k keys in all).
const WINDOW_KEYS_PER_PART: i32 = 64;
/// Most queries the quantized decode attention takes at once.
pub const MAX_DECODE_QUERIES: i32 = 8;

// K-2: the relative attention kernel reading the affine-quantized cache in
// place, built from the same source with only the key/value tile loads
// changed. Each value is dequantized exactly as `mx.dequantize` does (see
// the decode kernel), then converted to half as the original converts the
// bf16 history, so the output is bit for bit the one over the dequantized
// cache, without materializing it.
const RELATIVE_PTRS: &str =
    "    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];
";

const QUANT_RELATIVE_PTRS: &str = "    constexpr int ROW_BYTES = D * BITS / 8;
    constexpr int NG = D / GROUP;
    const size_t row0 = (size_t(batch) * HKV + kv_head) * size_t(sizes[2]);
    const device uint8_t* kbytes = (const device uint8_t*)kw + row0 * ROW_BYTES;
    const device uint8_t* vbytes = (const device uint8_t*)vw + row0 * ROW_BYTES;
    const device T* ksc = ks + row0 * NG;
    const device T* kbi = kb + row0 * NG;
    const device T* vsc = vs + row0 * NG;
    const device T* vbi = vb + row0 * NG;
";

const RELATIVE_LOAD: &str = "            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
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
";

const QUANT_RELATIVE_LOAD: &str = "            float kf[8];
            float vf[8];
            if (key < S) {
                const size_t gi = size_t(key) * NG + col / GROUP;
                const float ksf = float(ksc[gi]);
                const float kbf = float(kbi[gi]);
                const float vsf = float(vsc[gi]);
                const float vbf = float(vbi[gi]);
                const device uint8_t* kp = kbytes + size_t(key) * ROW_BYTES + col * BITS / 8;
                const device uint8_t* vp = vbytes + size_t(key) * ROW_BYTES + col * BITS / 8;
                uint kwords[2];
                uint vwords[2];
                if (BITS == 8) {
                    const uint2 k2 = *(const device uint2*)kp;
                    const uint2 v2 = *(const device uint2*)vp;
                    kwords[0] = k2.x;
                    kwords[1] = k2.y;
                    vwords[0] = v2.x;
                    vwords[1] = v2.y;
                } else {
                    kwords[0] = *(const device uint*)kp;
                    vwords[0] = *(const device uint*)vp;
                    kwords[1] = 0;
                    vwords[1] = 0;
                }
                constexpr int PER_WORD = 32 / BITS;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t kq = (kwords[j / PER_WORD] >> (BITS * (j % PER_WORD))) & ((1u << BITS) - 1);
                    const uint8_t vq = (vwords[j / PER_WORD] >> (BITS * (j % PER_WORD))) & ((1u << BITS) - 1);
                    // As mx.dequantize: one rounding of `scale * q + bias` to bfloat.
                    kf[j] = float(T(ksf * float(kq) + kbf));
                    vf[j] = float(T(vsf * float(vq) + vbf));
                }
            } else {
                for (int j = 0; j < 8; ++j) {
                    kf[j] = 0.0f;
                    vf[j] = 0.0f;
                }
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(kf[j]);
                Vs[r * LDV + col + j] = half(vf[j]);
            }
";

fn quant_relative_source() -> Result<String> {
    anyhow::ensure!(
        RELATIVE_ATTENTION_SOURCE.contains(RELATIVE_PTRS)
            && RELATIVE_ATTENTION_SOURCE.contains(RELATIVE_LOAD),
        "relative attention source changed; update its quantized variant"
    );
    Ok(RELATIVE_ATTENTION_SOURCE
        .replace(RELATIVE_PTRS, QUANT_RELATIVE_PTRS)
        .replace(RELATIVE_LOAD, QUANT_RELATIVE_LOAD))
}

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
    quant_partial: MetalKernel,
    quant_reduce: MetalKernel,
    quant_verify: MetalKernel,
    relative_quant: MetalKernel,
    gather_qmv_shared: MetalKernel,
    window_verify: MetalKernel,
    decode_mask: MetalKernel,
    short_conv: MetalKernel,
    /// Key parts per KV head of the quantized decode attention; 0 adapts to
    /// the context length.
    pub decode_parts: i32,
    /// Keys per simdgroup step (1, 2, 4 or 8); 0 picks by query count.
    pub decode_keys_per_step: i32,
    /// Fewest queries that take the simdgroup-matrix kernel instead of the
    /// per-lane one.
    pub matrix_from: i32,
    /// Fewest keys from which a single query takes the matrix kernel too.
    pub matrix_decode_from: i32,
    /// Key parts per KV head of the window verification; 0 adapts.
    pub window_parts: i32,
    /// Functions compiled with `mx.compile` (K-6), by name.
    compiled: std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<ops::Compiled>>>,
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
                contiguous.clone(),
            ))?,
            quant_partial: ops::with_mlx_message(MetalKernel::new(
                "quant_decode_partial",
                &[
                    "q", "kw", "ks", "kb", "vw", "vs", "vb", "rel", "sizes", "scale",
                ],
                &["pm", "ps", "po"],
                QUANT_DECODE_PARTIAL_SOURCE,
                contiguous.clone(),
            ))?,
            quant_verify: ops::with_mlx_message(MetalKernel::new(
                "quant_verify_partial",
                &[
                    "q", "kw", "ks", "kb", "vw", "vs", "vb", "rel", "sizes", "scale",
                ],
                &["pm", "ps", "po"],
                &quant_verify_source(),
                contiguous.clone(),
            ))?,
            quant_reduce: ops::with_mlx_message(MetalKernel::new(
                "quant_decode_reduce",
                &["pm", "ps", "po", "sinks"],
                &["out"],
                QUANT_DECODE_REDUCE_SOURCE,
                contiguous.clone(),
            ))?,
            relative_quant: ops::with_mlx_message(MetalKernel::new(
                "relative_attention_quantized",
                &[
                    "q",
                    "kw",
                    "ks",
                    "kb",
                    "vw",
                    "vs",
                    "vb",
                    "rel",
                    "sinks",
                    "key_start",
                    "sizes",
                    "scale",
                ],
                &["out"],
                &quant_relative_source()?,
                MetalKernelOptions::default(),
            ))?,
            short_conv: ops::with_mlx_message(MetalKernel::new(
                "short_conv_step",
                &["x", "state", "w", "denom"],
                &["out", "next"],
                SHORT_CONV_SOURCE,
                contiguous.clone(),
            ))?,
            decode_mask: ops::with_mlx_message(MetalKernel::new(
                "decode_mask",
                &["pb", "params"],
                &["mask"],
                DECODE_MASK_SOURCE,
                contiguous.clone(),
            ))?,
            window_verify: ops::with_mlx_message(MetalKernel::new(
                "window_verify_partial",
                &["q", "kw", "vw", "nk", "nv", "rel", "sizes", "scale"],
                &["pm", "ps", "po"],
                &window_verify_source(),
                contiguous.clone(),
            ))?,
            gather_qmv_shared: ops::with_mlx_message(MetalKernel::new(
                "gather_qmv_shared",
                &["x", "w", "scales", "gs", "indices"],
                &["y"],
                GATHER_QMV_SHARED_SOURCE,
                MetalKernelOptions {
                    header: GATHER_QMV_SHARED_HEADER.to_string(),
                    ..contiguous
                },
            ))?,
            decode_parts: 0,
            decode_keys_per_step: 0,
            matrix_from: 2,
            // On M4 Max, 8-bit at 516k tokens: 2.9 ms per pass against 3.6
            // per lane; about even at 160k, slightly slower on short
            // contexts (see bench_speculative_attention).
            matrix_decode_from: 256 * 1024,
            window_parts: 0,
            compiled: Default::default(),
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

    /// `relative_attention` over the first `len` tokens of an
    /// affine-quantized cache (full buffers (1, Hkv, capacity, ·)), reading
    /// it packed (K-2). Same output as over the dequantized cache.
    #[allow(clippy::too_many_arguments)]
    pub fn relative_attention_quantized(
        &self,
        q: &Array,
        keys: &crate::cache::Packed,
        values: &crate::cache::Packed,
        len: i32,
        rel: &Array,
        sinks: &Array,
        scale: f32,
        format: crate::cache::KvFormat,
    ) -> Result<Array> {
        let [b, hq, l, d] = q.shape()[..] else {
            unreachable!()
        };
        let [_, hkv, cap, _] = keys.w.shape()[..] else {
            unreachable!()
        };
        anyhow::ensure!(
            matches!(format.bits, 4 | 8),
            "packed prefill needs 4 or 8 bits"
        );
        let rel = ops::contiguous(rel)?;
        let sinks = ops::contiguous(&ops::f32(sinks)?)?;
        let key_start = ops::contiguous(&ops::broadcast_to(&ops::i32s(0), &[b])?)?;
        let sizes = Array::from_slice(&[l, len, cap], &[3]);
        // The kernel indexes the buffers as dense rows (a sliced view, e.g.
        // after a fork, is copied first; fresh buffers are left as they are).
        let dense = |p: &crate::cache::Packed| -> Result<crate::cache::Packed> {
            Ok(crate::cache::Packed {
                w: ops::contiguous(&p.w)?,
                scales: ops::contiguous(&p.scales)?,
                biases: ops::contiguous(&p.biases)?,
            })
        };
        let (keys, values) = (&dense(keys)?, &dense(values)?);
        let scale = f32s(scale);
        // SAFETY: the launch shapes match the kernel's indexing, as in the
        // bf16 kernel; the packed buffers hold `cap` rows per KV head.
        let out = unsafe {
            ops::with_mlx_message(self.relative_quant.apply(
                &[
                    q,
                    &keys.w,
                    &keys.scales,
                    &keys.biases,
                    &values.w,
                    &values.scales,
                    &values.biases,
                    &rel,
                    &sinks,
                    &key_start,
                    &sizes,
                    &scale,
                ],
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
                        int("WINDOW", 0),
                        int("BQ", BQ),
                        int("TG", TG),
                        int("BITS", format.bits),
                        int("GROUP", format.group_size),
                    ],
                    ..Default::default()
                },
            ))?
        };
        Ok(out.into_iter().next().unwrap())
    }

    /// Attention of L <= 8 queries at the last L positions over the first
    /// `len` tokens of an affine-quantized cache, without dequantizing it.
    /// q: (1, Hq, L, D) bf16; keys and values: full cache buffers
    /// (1, Hkv, capacity, ·); rel: (1, L, Hq, R); sinks: (1, L, Hq).
    /// Returns (1, L, Hq, D) bf16, like `relative_attention`.
    #[allow(clippy::too_many_arguments)]
    pub fn quantized_decode_attention(
        &self,
        q: &Array,
        keys: &crate::cache::Packed,
        values: &crate::cache::Packed,
        len: i32,
        rel: &Array,
        sinks: &Array,
        scale: f32,
        format: crate::cache::KvFormat,
    ) -> Result<Array> {
        self.split_key_attention(q, keys, values, len, rel, sinks, scale, format, true)
    }

    /// Attention of L <= 8 queries that all see the first `len` keys of a
    /// bf16 cache (full buffers (1, Hkv, capacity, D)): no mask, bias or
    /// sink. The drafter's block. Returns (1, L, Hq, D).
    pub fn block_attention(
        &self,
        q: &Array,
        keys: &Array,
        values: &Array,
        len: i32,
        scale: f32,
    ) -> Result<Array> {
        let [_, hq, l, _] = q.shape()[..] else {
            unreachable!()
        };
        // Scales and biases are not read for 16 bits; the buffer itself
        // stands in (small arrays would be passed as constants).
        let wrap = |a: &Array| crate::cache::Packed {
            w: a.clone(),
            scales: a.clone(),
            biases: a.clone(),
        };
        // R = 0: the bias is never read. A -inf sink adds nothing.
        let rel = ops::zeros(&[1, l, hq, 1], q.dtype())?;
        let sinks = ops::broadcast_to(
            &ops::scalar(f32::NEG_INFINITY, Dtype::Float32)?,
            &[1, l, hq],
        )?;
        self.split_key_attention(
            q,
            &wrap(keys),
            &wrap(values),
            len,
            &rel,
            &sinks,
            scale,
            crate::cache::KvFormat::BF16,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn split_key_attention(
        &self,
        q: &Array,
        keys: &crate::cache::Packed,
        values: &crate::cache::Packed,
        len: i32,
        rel: &Array,
        sinks: &Array,
        scale: f32,
        format: crate::cache::KvFormat,
        causal: bool,
    ) -> Result<Array> {
        let [b, hq, l, d] = q.shape()[..] else {
            unreachable!()
        };
        anyhow::ensure!(b == 1, "quantized decode supports one sequence");
        anyhow::ensure!(
            l <= MAX_DECODE_QUERIES,
            "quantized decode takes at most {MAX_DECODE_QUERIES} queries"
        );
        anyhow::ensure!(
            d % 32 == 0 && (format.bits == 16 || d % format.group_size == 0),
            "unsupported head_dim {d}"
        );
        let [_, hkv, cap, _] = keys.w.shape()[..] else {
            unreachable!()
        };
        anyhow::ensure!(
            0 < len && len <= cap && (!causal || l <= len),
            "attention over {len} keys (capacity {cap}) with {l} queries"
        );
        let g = hq / hkv;
        let gl = g * l;
        let parts = match self.decode_parts {
            0 => (len / KEYS_PER_PART).clamp(1, DECODE_PARTS),
            p => p,
        };
        let template_args = vec![
            ("T".to_string(), TemplateArg::Dtype(q.dtype())),
            int("D", d),
            int("G", g),
            int("L", l),
            int("HKV", hkv),
            int("BITS", format.bits),
            int("GROUP", format.group_size),
            int("NB", parts),
            // One key per step was fastest on M4 Max (see bench_decode_160k).
            int(
                "KPS",
                match self.decode_keys_per_step {
                    0 => 1,
                    k => k,
                },
            ),
            int("R", if causal { rel.shape()[3] } else { 0 }),
            int("CAUSAL", causal as i32),
            int("WINDOW", 0),
            // Keys per step of the matrix kernel: 8, 16 and 32 were within
            // 5% of each other at 516k tokens.
            int("KS", 16),
        ];
        // Only the matrix kernel has the bidirectional mode.
        let matrix = !causal || l >= self.matrix_from || len >= self.matrix_decode_from;
        let q = ops::contiguous(q)?;
        let rel = ops::contiguous(rel)?;
        let sinks = ops::contiguous(&ops::f32(sinks)?)?;
        let sizes = Array::from_slice(&[len, cap], &[2]);
        let scale = f32s(scale);
        let n = hkv * parts * gl;
        // SAFETY: per lane, one simdgroup per (part, KV head); with
        // matrices, one simdgroup per 8 query rows of that pair. All
        // indexing stays within the shapes checked above, as the kernel
        // comments describe.
        let (kernel, threads) = if matrix {
            (&self.quant_verify, (g * l + 7) / 8 * 32)
        } else {
            (&self.quant_partial, 32)
        };
        let partial = unsafe {
            ops::with_mlx_message(kernel.apply(
                &[
                    &q,
                    &keys.w,
                    &keys.scales,
                    &keys.biases,
                    &values.w,
                    &values.scales,
                    &values.biases,
                    &rel,
                    &sizes,
                    &scale,
                ],
                &[
                    MetalKernelOutput {
                        shape: vec![n],
                        dtype: Dtype::Float32,
                    },
                    MetalKernelOutput {
                        shape: vec![n],
                        dtype: Dtype::Float32,
                    },
                    MetalKernelOutput {
                        shape: vec![n * d],
                        dtype: Dtype::Float32,
                    },
                ],
                (parts * threads, hkv, 1),
                (threads, 1, 1),
                MetalKernelApplyOptions {
                    template_args: template_args.clone(),
                    ..Default::default()
                },
            ))?
        };
        self.reduce_parts(&partial, &sinks, template_args, &q)
    }

    /// Pass 2 of the split-key kernels: merges the parts of every (KV head,
    /// query) with its sink. Returns (1, L, Hq, D) in q's dtype.
    fn reduce_parts(
        &self,
        partial: &[Array],
        sinks: &Array,
        template_args: Vec<(String, TemplateArg)>,
        q: &Array,
    ) -> Result<Array> {
        let [_, hq, l, d] = q.shape()[..] else {
            unreachable!()
        };
        // SAFETY: one simdgroup per (KV head, query); reads the parts of
        // pass 1, laid out as its template arguments say.
        let out = unsafe {
            ops::with_mlx_message(self.quant_reduce.apply(
                &[&partial[0], &partial[1], &partial[2], sinks],
                &[MetalKernelOutput {
                    shape: vec![1, l, hq, d],
                    dtype: q.dtype(),
                }],
                (hq * l * 32, 1, 1),
                (32, 1, 1),
                MetalKernelApplyOptions {
                    template_args,
                    ..Default::default()
                },
            ))?
        };
        Ok(out.into_iter().next().unwrap())
    }

    /// Speculative verification over a bf16 sliding window without
    /// touching it: q (1, Hq, L, D) at the positions after the window; the
    /// ring buffer (1, Hkv, capacity, D) with `rlen` filled slots and its
    /// next write slot `ridx`; the L new keys and values (1, Hkv, L, D),
    /// not yet in the buffer. Each query sees the keys less than `window`
    /// positions back, itself included. rel: (1, L, Hq, R); sinks:
    /// (1, L, Hq). Returns (1, L, Hq, D).
    #[allow(clippy::too_many_arguments)]
    pub fn window_verify_attention(
        &self,
        q: &Array,
        ring_k: &Array,
        ring_v: &Array,
        rlen: i32,
        ridx: i32,
        new_k: &Array,
        new_v: &Array,
        rel: &Array,
        sinks: &Array,
        scale: f32,
        window: i32,
    ) -> Result<Array> {
        let [b, hq, l, d] = q.shape()[..] else {
            unreachable!()
        };
        let [_, hkv, cap, _] = ring_k.shape()[..] else {
            anyhow::bail!("window buffer must be (1, Hkv, capacity, D)")
        };
        anyhow::ensure!(
            b == 1
                && (1..=MAX_DECODE_QUERIES).contains(&l)
                && d % 32 == 0
                && [q, ring_k, ring_v, new_k, new_v]
                    .iter()
                    .all(|a| a.dtype() == Dtype::Bfloat16)
                && ring_v.shape() == ring_k.shape()
                && new_k.shape() == [1, hkv, l, d]
                && new_v.shape() == [1, hkv, l, d]
                && 0 < rlen
                && rlen <= cap
                && (0..=rlen).contains(&ridx)
                && window > 0,
            "unsupported window verification: L {l}, {rlen} of {cap} slots, write slot {ridx}"
        );
        let g = hq / hkv;
        let gl = g * l;
        let len = rlen + l;
        let parts = match self.window_parts {
            0 => (len / WINDOW_KEYS_PER_PART).clamp(1, DECODE_PARTS),
            p => p,
        };
        let template_args = vec![
            ("T".to_string(), TemplateArg::Dtype(q.dtype())),
            int("D", d),
            int("G", g),
            int("L", l),
            int("HKV", hkv),
            int("BITS", 16),
            int("GROUP", 64),
            int("NB", parts),
            int("KPS", 1),
            int("R", rel.shape()[3]),
            int("CAUSAL", 1),
            int("WINDOW", window),
            int("KS", 16),
        ];
        let q = ops::contiguous(q)?;
        let rel = ops::contiguous(rel)?;
        let sinks = ops::contiguous(&ops::f32(sinks)?)?;
        let sizes = Array::from_slice(&[len, cap, rlen, ridx], &[4]);
        let scale = f32s(scale);
        let n = hkv * parts * gl;
        let threads = (gl + 7) / 8 * 32;
        // SAFETY: one simdgroup per 8 query rows of a (part, KV head) pair;
        // the virtual keys 0..len map to the rlen filled slots and the L new
        // rows, all within the shapes checked above.
        let partial = unsafe {
            ops::with_mlx_message(self.window_verify.apply(
                &[&q, ring_k, ring_v, new_k, new_v, &rel, &sizes, &scale],
                &[
                    MetalKernelOutput {
                        shape: vec![n],
                        dtype: Dtype::Float32,
                    },
                    MetalKernelOutput {
                        shape: vec![n],
                        dtype: Dtype::Float32,
                    },
                    MetalKernelOutput {
                        shape: vec![n * d],
                        dtype: Dtype::Float32,
                    },
                ],
                (parts * threads, hkv, 1),
                (threads, 1, 1),
                MetalKernelApplyOptions {
                    template_args: template_args.clone(),
                    ..Default::default()
                },
            ))?
        };
        self.reduce_parts(&partial, &sinks, template_args, &q)
    }

    /// `gather_qmm` (nvfp4, mxfp4 or mxfp8, transposed weights) for a few
    /// tokens, reading each chosen expert's weights once (K-7). x: (T, K)
    /// with one row per token, or (T * top_k, K) with one per route when
    /// `x_per_route`; w: (E, N, K * bits / 32) u32; scales: (E, N, K / group)
    /// u8; global_scale: (E,) f32 for nvfp4 with a global scale; indices:
    /// (T, top_k). Returns (T * top_k, N) in x's dtype, route by route.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_qmv_shared(
        &self,
        x: &Array,
        w: &Array,
        scales: &Array,
        global_scale: Option<&Array>,
        indices: &Array,
        x_per_route: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Array> {
        let [tokens, top_k] = indices.shape()[..] else {
            anyhow::bail!("indices must be (tokens, top_k)")
        };
        let [experts, n, words] = w.shape()[..] else {
            anyhow::bail!("expert weights must be (E, N, K * bits / 32)")
        };
        anyhow::ensure!(x.ndim() == 2, "x must be (rows, K)");
        let k = x.shape()[1];
        let routes = tokens * top_k;
        anyhow::ensure!(
            w.dtype() == Dtype::Uint32
                && words * 32 == k * bits
                && scales.dtype() == Dtype::Uint8
                && k % group_size == 0
                && scales.shape() == [experts, n, k / group_size]
                && global_scale
                    .is_none_or(|g| g.dtype() == Dtype::Float32 && g.shape() == [experts]),
            "expert weights, scales or global scale do not match x (K {k})"
        );
        anyhow::ensure!(
            (1..=MAX_DECODE_QUERIES).contains(&tokens)
                && matches!((bits, group_size), (4, 16) | (4, 32) | (8, 32))
                && k % (32 / bits) == 0
                && n % 8 == 0
                && x.shape()[0] == if x_per_route { routes } else { tokens },
            "unsupported shared expert gather: {tokens} tokens, bits {bits}, group {group_size}, K {k}, N {n}"
        );
        let indices = ops::astype(&ops::reshape(indices, &[routes])?, Dtype::Uint32)?;
        // The scales stand in for an absent global scale (never read).
        let gs = global_scale.unwrap_or(scales);
        // SAFETY: one threadgroup of two simdgroups per (8 output rows,
        // route); the shapes and dtypes are checked above and every index
        // stays within them, as the kernel describes. (Routing never picks
        // an expert twice for one token, so at most `tokens` routes share
        // one; only all-NaN router logits could break that, and those
        // already give garbage.)
        let out = unsafe {
            ops::with_mlx_message(self.gather_qmv_shared.apply(
                &[x, w, scales, gs, &indices],
                &[MetalKernelOutput {
                    shape: vec![routes, n],
                    dtype: x.dtype(),
                }],
                (n / 8 * 64, routes, 1),
                (64, 1, 1),
                MetalKernelApplyOptions {
                    template_args: vec![
                        ("T".to_string(), TemplateArg::Dtype(x.dtype())),
                        int("BITS", bits),
                        int("GROUP", group_size),
                        int("K", k),
                        int("N", n),
                        int("TOPK", top_k),
                        int("MAXT", tokens),
                        // MLX's choice between its qmv and qmv_fast kernels.
                        int("PPT", if k % (32 / bits * 2 * 32) == 0 { 2 } else { 1 }),
                        int("NR", routes),
                        int("XR", x_per_route as i32),
                        int("HAS_GS", global_scale.is_some() as i32),
                    ],
                    ..Default::default()
                },
            ))?
        };
        Ok(out.into_iter().next().unwrap())
    }

    /// The compiled function `name`, built by `build` on first use.
    pub fn compiled(
        &self,
        name: &str,
        build: impl FnOnce() -> ops::Compiled,
    ) -> std::sync::Arc<ops::Compiled> {
        let mut map = self.compiled.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(name.to_string())
            .or_insert_with(|| std::sync::Arc::new(build()))
            .clone()
    }

    /// One decode step of a static short convolution (K-6): x (1, 1, C) in
    /// the weights' dtype, state (1, K - 1, C) float32, weight (C, K)
    /// contiguous. Returns the output (1, 1, C) and the next state
    /// (1, K - 1, C) float32: the window's last K - 1 rows.
    pub fn short_conv_step(
        &self,
        x: &Array,
        state: &Array,
        weight: &Array,
        denom: f32,
        out_f32: bool,
    ) -> Result<(Array, Array)> {
        let [c, k] = weight.shape()[..] else {
            anyhow::bail!("short convolution weight must be (C, K)")
        };
        anyhow::ensure!(
            x.shape() == [1, 1, c]
                && x.dtype() == weight.dtype()
                && state.shape() == [1, k - 1, c]
                && state.dtype() == Dtype::Float32,
            "short convolution step: x {:?}, state {:?}, weight {:?}",
            x.shape(),
            state.shape(),
            weight.shape()
        );
        let denom = f32s(denom);
        // SAFETY: one thread per channel; reads and writes inside the
        // shapes checked above.
        let mut out = unsafe {
            ops::with_mlx_message(self.short_conv.apply(
                &[x, state, weight, &denom],
                &[
                    MetalKernelOutput {
                        shape: vec![1, 1, c],
                        dtype: if out_f32 { Dtype::Float32 } else { x.dtype() },
                    },
                    MetalKernelOutput {
                        shape: vec![1, k - 1, c],
                        dtype: Dtype::Float32,
                    },
                ],
                (c, 1, 1),
                (c.min(256), 1, 1),
                MetalKernelApplyOptions {
                    template_args: vec![
                        ("T".to_string(), TemplateArg::Dtype(x.dtype())),
                        (
                            "OT".to_string(),
                            TemplateArg::Dtype(if out_f32 { Dtype::Float32 } else { x.dtype() }),
                        ),
                        int("C", c),
                        int("K", k),
                    ],
                    ..Default::default()
                },
            ))?
        }
        .into_iter();
        Ok((out.next().unwrap(), out.next().unwrap()))
    }

    /// The additive attention mask (1, Hq, 1, s) of one decode query over
    /// `s` key slots, in `dtype` (K-6). pb: the relative bias (Hq, R) of
    /// the query; `rot`: the next write slot of a sliding window's ring
    /// buffer (0 for a global cache); `end`: tokens in the cache including
    /// the query; `window`: 0 for none.
    pub fn decode_mask(
        &self,
        pb: &Array,
        s: i32,
        rot: i32,
        end: i32,
        window: i32,
        dtype: Dtype,
    ) -> Result<Array> {
        let [hq, r] = pb.shape()[..] else {
            anyhow::bail!("relative bias must be (Hq, R)")
        };
        anyhow::ensure!(
            s > 0 && (0..=s).contains(&rot),
            "decode mask over {s} slots, rotation {rot}"
        );
        let params = Array::from_slice(&[s, rot, end, window], &[4]);
        // SAFETY: one thread per (slot, head); reads pb (Hq, R) at distances
        // below R and writes inside (Hq, s).
        let out = unsafe {
            ops::with_mlx_message(self.decode_mask.apply(
                &[pb, &params],
                &[MetalKernelOutput {
                    shape: vec![1, hq, 1, s],
                    dtype,
                }],
                (s, hq, 1),
                (s.min(256), 1, 1),
                MetalKernelApplyOptions {
                    template_args: vec![("T".to_string(), TemplateArg::Dtype(dtype)), int("R", r)],
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
