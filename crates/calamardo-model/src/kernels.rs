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

/// Most key parts per KV head of the quantized decode attention (K-4). On M4
/// Max, 512 parts of this kernel beat the 64 that suit MLX's own attention
/// (1.38 vs 2.08 ms per pass at 160k tokens, see `bench_decode_160k`).
pub const DECODE_PARTS: i32 = 512;
/// Fewest keys per part, so short contexts do not pay for many parts.
const KEYS_PER_PART: i32 = 256;
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
    relative_quant: MetalKernel,
    /// Key parts per KV head of the quantized decode attention; 0 adapts to
    /// the context length.
    pub decode_parts: i32,
    /// Keys per simdgroup step (1, 2, 4 or 8); 0 picks by query count.
    pub decode_keys_per_step: i32,
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
            quant_reduce: ops::with_mlx_message(MetalKernel::new(
                "quant_decode_reduce",
                &["pm", "ps", "po", "sinks"],
                &["out"],
                QUANT_DECODE_REDUCE_SOURCE,
                contiguous,
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
            decode_parts: 0,
            decode_keys_per_step: 0,
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
            int("R", rel.shape()[3]),
        ];
        let q = ops::contiguous(q)?;
        let rel = ops::contiguous(rel)?;
        let sinks = ops::contiguous(&ops::f32(sinks)?)?;
        let sizes = Array::from_slice(&[len, cap], &[2]);
        let scale = f32s(scale);
        let n = hkv * parts * gl;
        // SAFETY: one simdgroup per (part, KV head); all indexing stays
        // within the shapes checked above, as the kernel comments describe.
        let partial = unsafe {
            ops::with_mlx_message(self.quant_partial.apply(
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
                (parts * 32, hkv, 1),
                (32, 1, 1),
                MetalKernelApplyOptions {
                    template_args: template_args.clone(),
                    ..Default::default()
                },
            ))?
        };
        // SAFETY: one simdgroup per (KV head, query); reads the parts above.
        let out = unsafe {
            ops::with_mlx_message(self.quant_reduce.apply(
                &[&partial[0], &partial[1], &partial[2], &sinks],
                &[MetalKernelOutput {
                    shape: vec![1, l, hq, d],
                    dtype: q.dtype(),
                }],
                (hkv * gl * 32, 1, 1),
                (32, 1, 1),
                MetalKernelApplyOptions {
                    template_args,
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
