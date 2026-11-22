//! K-8 checks for the quantized-cache decode attention (needs the GPU).
//!
//! 1. Values are dequantized exactly as `mx.dequantize` (KV requirement 2):
//!    bit-for-bit equality.
//! 2. The attention matches a float64 reference over the dequantized cache,
//!    with the relative bias, the causal mask and the sink, within a
//!    documented tolerance: two bf16 ulps of the result (the kernel sums in
//!    f32 in a different order and uses exp2) plus 1e-3 absolute.

use calamardo_model::{
    cache::{KvFormat, QuantKvCache},
    kernels::Kernels,
    ops,
};
use mlx_rs::{Array, Dtype};

const HQ: i32 = 24;
const HKV: i32 = 8;
const D: i32 = 128;
const R: i32 = 64;

fn data(n: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            ((x as f32 / u32::MAX as f32) * 2.0 - 1.0) * amp
        })
        .collect()
}

fn bf16(v: Vec<f32>, shape: &[i32]) -> Array {
    ops::astype(&Array::from_slice(&v, shape), Dtype::Bfloat16).unwrap()
}

fn host(a: &Array) -> Vec<f32> {
    let a = ops::f32(a).unwrap();
    ops::eval(&[&a]).unwrap();
    a.as_slice::<f32>().to_vec()
}

fn cache_with(format: KvFormat, s: i32, seed: u32) -> QuantKvCache {
    ops::init();
    let mut c = QuantKvCache::new(format);
    let n = (HKV * s * D) as usize;
    // Fill in two updates so the buffers grow like at runtime.
    let first = s / 2;
    for (start, len, sd) in [(0, first, seed), (first, s - first, seed + 1)] {
        if len == 0 {
            continue;
        }
        let k = data((HKV * len * D) as usize, sd, 2.0);
        let v = data((HKV * len * D) as usize, sd + 100, 1.0);
        c.update(&bf16(k, &[1, HKV, len, D]), &bf16(v, &[1, HKV, len, D]))
            .unwrap();
        let _ = (start, n);
    }
    c
}

#[test]
fn dequantization_matches_mlx_bit_for_bit() {
    let mut kern = Kernels::new().unwrap();
    // Per lane, then with matrices.
    for (matrix_from, bits) in [(2, 4), (2, 8), (1, 4), (1, 8)] {
        kern.matrix_from = matrix_from;
        for group in [32, 64, 128] {
            let format = KvFormat::new(bits, group).unwrap();
            let c = cache_with(format, 1, 7 + bits as u32 + group as u32);
            let q = bf16(data((HQ * D) as usize, 3, 1.0), &[1, HQ, 1, D]);
            let rel = bf16(vec![0.0; (HQ * R) as usize], &[1, 1, HQ, R]);
            // A -inf sink and one key: the output is that key's value.
            let sinks = Array::from_slice(&vec![f32::NEG_INFINITY; HQ as usize], &[1, 1, HQ]);
            let out = kern
                .quantized_decode_attention(
                    &q,
                    c.keys.as_ref().unwrap(),
                    c.values.as_ref().unwrap(),
                    1,
                    &rel,
                    &sinks,
                    0.1,
                    format,
                )
                .unwrap();
            let got = host(&out);
            let (_, v) = c.dequantized().unwrap();
            let want = host(&v);
            for h in 0..HQ as usize {
                let kvh = h / (HQ / HKV) as usize;
                for d in 0..D as usize {
                    let (a, b) = (got[h * D as usize + d], want[kvh * D as usize + d]);
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "matrix from {matrix_from}, bits {bits} group {group} head {h} dim {d}: {a} vs {b}"
                    );
                }
            }
        }
    }
}

/// Float64 attention over the dequantized cache, as `_attention_reference`.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rel: &[f32],
    sinks: &[f32],
    s: i32,
    l: i32,
    scale: f64,
) -> Vec<f64> {
    let (s, l, d, r) = (s as usize, l as usize, D as usize, R as usize);
    let g = (HQ / HKV) as usize;
    let mut out = vec![0.0; l * HQ as usize * d];
    for h in 0..HQ as usize {
        let kvh = h / g;
        for qi in 0..l {
            let p = s - l + qi;
            let mut logits = Vec::with_capacity(s + 1);
            for j in 0..s {
                if j > p {
                    logits.push(f64::NEG_INFINITY);
                    continue;
                }
                let mut dot = 0.0;
                for c in 0..d {
                    dot += q[(h * l + qi) * d + c] as f64 * k[(kvh * s + j) * d + c] as f64;
                }
                let dist = p - j;
                let bias = if dist < r {
                    rel[(qi * HQ as usize + h) * r + dist] as f64
                } else {
                    0.0
                };
                logits.push(dot * scale + bias);
            }
            logits.push(sinks[qi * HQ as usize + h] as f64);
            let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = logits.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = w.iter().sum();
            for c in 0..d {
                let mut acc = 0.0;
                for j in 0..s {
                    acc += w[j] * v[(kvh * s + j) * d + c] as f64;
                }
                out[(qi * HQ as usize + h) * d + c] = acc / z;
            }
        }
    }
    out
}

#[test]
fn attention_matches_dequantized_reference() {
    let mut kern = Kernels::new().unwrap();
    let scale = (D as f64).powf(-0.5);
    let mut worst = 0.0f64;
    // Both kernels: per lane for one query and matrices for several, then
    // matrices for all.
    for (matrix_from, bits, group) in [
        (2, 8, 64),
        (2, 4, 64),
        (2, 8, 32),
        (2, 4, 128),
        (1, 8, 64),
        (1, 4, 32),
    ] {
        kern.matrix_from = matrix_from;
        let format = KvFormat::new(bits, group).unwrap();
        for s in [1, 37, 300, 2500] {
            for l in [1, 2, 3, 5, 8] {
                if l > s {
                    continue;
                }
                let c = cache_with(format, s, (s * 31 + l) as u32);
                let qv = data((HQ * l * D) as usize, 11 + l as u32, 0.2);
                let relv = data((l * HQ * R) as usize, 13, 1.0);
                let sinkv = data((l * HQ) as usize, 17, 2.0);
                let q = bf16(qv, &[1, HQ, l, D]);
                let rel = bf16(relv, &[1, l, HQ, R]);
                let sinks = bf16(sinkv, &[1, l, HQ]);
                let out = kern
                    .quantized_decode_attention(
                        &q,
                        c.keys.as_ref().unwrap(),
                        c.values.as_ref().unwrap(),
                        s,
                        &rel,
                        &sinks,
                        scale as f32,
                        format,
                    )
                    .unwrap();
                let got = host(&out);
                let (k, v) = c.dequantized().unwrap();
                let want = reference(
                    &host(&q),
                    &host(&k),
                    &host(&v),
                    &host(&rel),
                    &host(&sinks),
                    s,
                    l,
                    scale,
                );
                for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                    let err = (a as f64 - b).abs();
                    let tol = b.abs() * 2.0f64.powi(-7) + 1e-3;
                    worst = worst.max(err / tol);
                    assert!(
                        err <= tol,
                        "matrix from {matrix_from}, bits {bits} group {group} S {s} L {l} [{i}]: {a} vs {b}"
                    );
                }
            }
        }
    }
    eprintln!("worst error / tolerance: {worst:.3}");
}

/// Key parts whose keys are all masked for some queries (a last part
/// shorter than L) or empty (more parts than keys), with both kernels.
#[test]
fn masked_and_empty_parts_match_reference() {
    let mut kern = Kernels::new().unwrap();
    let scale = (D as f64).powf(-0.5);
    let format = KvFormat::new(8, 64).unwrap();
    let s = 37;
    let c = cache_with(format, s, 99);
    let (k, v) = c.dequantized().unwrap();
    for (matrix_from, parts) in [(2, 6), (2, 20), (2, 50), (1, 6), (1, 50)] {
        kern.matrix_from = matrix_from;
        kern.decode_parts = parts;
        for l in [1, 3, 8] {
            let q = bf16(
                data((HQ * l * D) as usize, 51 + l as u32, 0.2),
                &[1, HQ, l, D],
            );
            let rel = bf16(data((l * HQ * R) as usize, 53, 1.0), &[1, l, HQ, R]);
            let sinks = bf16(data((l * HQ) as usize, 57, 2.0), &[1, l, HQ]);
            let out = kern
                .quantized_decode_attention(
                    &q,
                    c.keys.as_ref().unwrap(),
                    c.values.as_ref().unwrap(),
                    s,
                    &rel,
                    &sinks,
                    scale as f32,
                    format,
                )
                .unwrap();
            let got = host(&out);
            let want = reference(
                &host(&q),
                &host(&k),
                &host(&v),
                &host(&rel),
                &host(&sinks),
                s,
                l,
                scale,
            );
            for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                let tol = b.abs() * 2.0f64.powi(-7) + 1e-3;
                assert!(
                    (a as f64 - b).abs() <= tol,
                    "matrix from {matrix_from}, {parts} parts, L {l} [{i}]: {a} vs {b}"
                );
            }
        }
    }
}

/// The drafter's block: every query sees every key, no bias or sink.
#[test]
fn block_attention_matches_reference() {
    let kern = Kernels::new().unwrap();
    let scale = (D as f64).powf(-0.5);
    let g = (HQ / HKV) as usize;
    for s in [8, 300, 2500] {
        let l = 8;
        let cap = s + 21;
        let kv = bf16(
            data((HKV * cap * D) as usize, 41 + s as u32, 2.0),
            &[1, HKV, cap, D],
        );
        let vv = bf16(
            data((HKV * cap * D) as usize, 42 + s as u32, 1.0),
            &[1, HKV, cap, D],
        );
        let q = bf16(data((HQ * l * D) as usize, 43, 0.2), &[1, HQ, l, D]);
        let got = host(&kern.block_attention(&q, &kv, &vv, s, scale as f32).unwrap());
        let (qh, kh, vh) = (host(&q), host(&kv), host(&vv));
        let (su, cu, d) = (s as usize, cap as usize, D as usize);
        for h in 0..HQ as usize {
            let kvh = h / g;
            for qi in 0..l as usize {
                let logits: Vec<f64> = (0..su)
                    .map(|j| {
                        (0..d)
                            .map(|c| {
                                qh[(h * 8 + qi) * d + c] as f64 * kh[(kvh * cu + j) * d + c] as f64
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = logits.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = w.iter().sum();
                for c in 0..d {
                    let b = (0..su)
                        .map(|j| w[j] * vh[(kvh * cu + j) * d + c] as f64)
                        .sum::<f64>()
                        / z;
                    let a = got[(qi * HQ as usize + h) * d + c] as f64;
                    let tol = b.abs() * 2.0f64.powi(-7) + 1e-3;
                    assert!((a - b).abs() <= tol, "S {s} h {h} q {qi} c {c}: {a} vs {b}");
                }
            }
        }
    }
}

/// Decode attention at 160k tokens for one global pass: quantized kernel vs
/// MLX's fused attention over a bf16 cache. Run with `--ignored`.
#[test]
#[ignore]
fn bench_decode_160k() {
    use std::time::Instant;
    let mut kern = Kernels::new().unwrap();
    let s = std::env::var("BENCH_LEN").map_or(160 * 1024, |v| v.parse().unwrap());
    let q = bf16(data((HQ * D) as usize, 3, 0.2), &[1, HQ, 1, D]);
    let rel = bf16(data((HQ * 1024) as usize, 5, 1.0), &[1, 1, HQ, 1024]);
    let sinks = bf16(data(HQ as usize, 7, 1.0), &[1, 1, HQ]);
    let iters = 50;
    for (bits, parts, kps) in [
        (8, 64, 1),
        (8, 512, 1),
        (8, 1024, 1),
        (8, 2048, 1),
        (4, 512, 1),
        (4, 1024, 1),
        (4, 2048, 1),
    ] {
        kern.decode_parts = parts;
        kern.decode_keys_per_step = kps;
        let format = KvFormat::new(bits, 64).unwrap();
        let c = cache_with(format, s, 1);
        let run = || {
            kern.quantized_decode_attention(
                &q,
                c.keys.as_ref().unwrap(),
                c.values.as_ref().unwrap(),
                s,
                &rel,
                &sinks,
                0.088,
                format,
            )
            .unwrap()
        };
        ops::eval(&[&run()]).unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            ops::eval(&[&run()]).unwrap();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let bytes = 2.0 * (HKV * s) as f64 * (D * bits / 8 + D / 64 * 4) as f64;
        eprintln!(
            "{bits}-bit kernel, {parts:>4} parts, {kps} keys/step: {ms:.2} ms/pass · {:.0} GB/s",
            bytes / ms / 1e6
        );
    }
    let k = bf16(data((HKV * s * D) as usize, 9, 2.0), &[1, HKV, s, D]);
    let v = bf16(data((HKV * s * D) as usize, 10, 1.0), &[1, HKV, s, D]);
    let mask = bf16(vec![0.0; (HQ * s) as usize], &[1, HQ, 1, s]);
    let sk = ops::reshape(&sinks, &[HQ]).unwrap();
    ops::eval(&[&k, &v, &mask]).unwrap();
    let run = || ops::sdpa(&q, &k, &v, 0.088, &mask, &sk).unwrap();
    ops::eval(&[&run()]).unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        ops::eval(&[&run()]).unwrap();
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    eprintln!(
        "bf16 sdpa:    {ms:.2} ms/pass · {:.0} GB/s",
        2.0 * (HKV * s * D * 2) as f64 / ms / 1e6
    );
}

/// The same kernel over a bf16 cache (speculative verification), against
/// the float64 reference.
#[test]
fn bf16_attention_matches_reference() {
    use calamardo_model::cache::Packed;
    let kern = Kernels::new().unwrap();
    let scale = (D as f64).powf(-0.5);
    for s in [1, 300, 2500] {
        for l in [2, 5, 8] {
            if l > s {
                continue;
            }
            // Capacity beyond the valid tokens, as in the growing cache.
            let cap = s + 37;
            let kv = bf16(
                data((HKV * cap * D) as usize, 21 + s as u32, 2.0),
                &[1, HKV, cap, D],
            );
            let vv = bf16(
                data((HKV * cap * D) as usize, 22 + s as u32, 1.0),
                &[1, HKV, cap, D],
            );
            let wrap = |a: &Array| Packed {
                w: a.clone(),
                scales: a.clone(),
                biases: a.clone(),
            };
            let q = bf16(
                data((HQ * l * D) as usize, 31 + l as u32, 0.2),
                &[1, HQ, l, D],
            );
            let rel = bf16(data((l * HQ * R) as usize, 33, 1.0), &[1, l, HQ, R]);
            let sinks = bf16(data((l * HQ) as usize, 37, 2.0), &[1, l, HQ]);
            let out = kern
                .quantized_decode_attention(
                    &q,
                    &wrap(&kv),
                    &wrap(&vv),
                    s,
                    &rel,
                    &sinks,
                    scale as f32,
                    KvFormat::BF16,
                )
                .unwrap();
            let got = host(&out);
            let k = ops::slice(&kv, 2, 0, s, 1).unwrap();
            let v = ops::slice(&vv, 2, 0, s, 1).unwrap();
            let want = reference(
                &host(&q),
                &host(&k),
                &host(&v),
                &host(&rel),
                &host(&sinks),
                s,
                l,
                scale,
            );
            for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                let tol = b.abs() * 2.0f64.powi(-7) + 1e-3;
                assert!(
                    (a as f64 - b).abs() <= tol,
                    "bf16 S {s} L {l} [{i}]: {a} vs {b}"
                );
            }
        }
    }
}

/// K-2: the prefill kernel reading the packed cache equals, bit for bit,
/// the bf16 kernel over the cache dequantized by `mx.dequantize`.
#[test]
fn packed_prefill_equals_dequantized_bit_for_bit() {
    let kern = Kernels::new().unwrap();
    let scale = (D as f32).powf(-0.5);
    for (bits, group) in [(8, 64), (4, 64), (8, 32), (4, 128)] {
        let format = KvFormat::new(bits, group).unwrap();
        for s in [40, 600, 3000] {
            for l in [9, 33, 100] {
                let c = cache_with(format, s, (s + l + bits) as u32);
                let q = bf16(
                    data((HQ * l * D) as usize, 41 + l as u32, 0.2),
                    &[1, HQ, l, D],
                );
                let rel = bf16(data((l * HQ * R) as usize, 43, 1.0), &[1, l, HQ, R]);
                let sinks = bf16(data((l * HQ) as usize, 47, 2.0), &[1, l, HQ]);
                let packed = kern
                    .relative_attention_quantized(
                        &q,
                        c.keys.as_ref().unwrap(),
                        c.values.as_ref().unwrap(),
                        s,
                        &rel,
                        &sinks,
                        scale,
                        format,
                    )
                    .unwrap();
                let (k, v) = c.dequantized().unwrap();
                let start = ops::broadcast_to(&ops::i32s(0), &[1]).unwrap();
                let want = kern
                    .relative_attention(&q, &k, &v, &rel, &sinks, scale, 0, &start)
                    .unwrap();
                let (a, b) = (host(&packed), host(&want));
                let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(same, "bits {bits} group {group} S {s} L {l}");
            }
        }
    }
}

/// Costs of the speculative pieces at long context: the drafter's global
/// layer (8 bidirectional queries over a bf16 cache) and verification
/// (L queries over the quantized cache). `BENCH_LEN` sets the keys.
#[test]
#[ignore]
fn bench_speculative_attention() {
    use calamardo_model::cache::Packed;
    use std::time::Instant;
    let mut kern = Kernels::new().unwrap();
    kern.matrix_from = std::env::var("MATRIX_FROM").map_or(2, |v| v.parse().unwrap());
    kern.decode_parts = std::env::var("PARTS").map_or(0, |v| v.parse().unwrap());
    let s = std::env::var("BENCH_LEN").map_or(160 * 1024, |v| v.parse().unwrap());
    let iters = 20;
    let time = |name: &str, f: &dyn Fn() -> Array| {
        ops::eval(&[&f()]).unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            ops::eval(&[&f()]).unwrap();
        }
        eprintln!(
            "{name:<40} {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        );
    };
    let cap = s + 256;
    let kb = bf16(data((HKV * cap * D) as usize, 9, 2.0), &[1, HKV, cap, D]);
    let vb = bf16(data((HKV * cap * D) as usize, 10, 1.0), &[1, HKV, cap, D]);
    let q8 = bf16(data((HQ * 8 * D) as usize, 3, 0.2), &[1, HQ, 8, D]);
    let nk = bf16(data((HKV * 8 * D) as usize, 4, 2.0), &[1, HKV, 8, D]);
    let ks = ops::slice(&kb, 2, 0, s, 1).unwrap();
    let vs = ops::slice(&vb, 2, 0, s, 1).unwrap();
    time("draft: concat ctx + block", &|| {
        // The slices keep MLX from skipping the (full) concatenations.
        let a = ops::concatenate(&[&ks, &nk], 2).unwrap();
        let b = ops::concatenate(&[&vs, &nk], 2).unwrap();
        ops::add(
            &ops::slice(&a, 2, s, s + 8, 1).unwrap(),
            &ops::slice(&b, 2, s, s + 8, 1).unwrap(),
        )
        .unwrap()
    });
    let kc = ops::slice(&kb, 2, 0, s + 8, 1).unwrap();
    let vc = ops::slice(&vb, 2, 0, s + 8, 1).unwrap();
    time("draft: sdpa 8 queries (strided view)", &|| {
        ops::sdpa_nomask(&q8, &kc, &vc, 0.088).unwrap()
    });
    let wrap = |a: &Array| Packed {
        w: a.clone(),
        scales: a.clone(),
        biases: a.clone(),
    };
    for l in [1, 2, 4, 8] {
        let q = bf16(data((HQ * l * D) as usize, 3, 0.2), &[1, HQ, l, D]);
        let rel = bf16(data((l * HQ * R) as usize, 5, 1.0), &[1, l, HQ, R]);
        let sinks = bf16(data((l * HQ) as usize, 7, 1.0), &[1, l, HQ]);
        time(&format!("bf16 split-key kernel, {l} queries"), &|| {
            kern.quantized_decode_attention(
                &q,
                &wrap(&kb),
                &wrap(&vb),
                s,
                &rel,
                &sinks,
                0.088,
                KvFormat::BF16,
            )
            .unwrap()
        });
    }
    drop((kb, vb, ks, vs, kc, vc));
    let format = KvFormat::new(8, 64).unwrap();
    let c = cache_with(format, s, 1);
    for l in [1, 2, 3, 5, 7] {
        let q = bf16(data((HQ * l * D) as usize, 3, 0.2), &[1, HQ, l, D]);
        let rel = bf16(data((l * HQ * R) as usize, 5, 1.0), &[1, l, HQ, R]);
        let sinks = bf16(data((l * HQ) as usize, 7, 1.0), &[1, l, HQ]);
        time(&format!("8-bit split-key kernel, {l} queries"), &|| {
            kern.quantized_decode_attention(
                &q,
                c.keys.as_ref().unwrap(),
                c.values.as_ref().unwrap(),
                s,
                &rel,
                &sinks,
                0.088,
                format,
            )
            .unwrap()
        });
    }
}
