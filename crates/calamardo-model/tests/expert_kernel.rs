//! The shared expert gather (K-7) against MLX's gather_qmm with unsorted
//! routes, bit for bit, with synthetic weights in every format the
//! checkpoint uses. (With 64 routes or more the model sorts them and MLX
//! takes another kernel, which rounds differently; the model keeps
//! gather_qmm there.)

use calamardo_ckpt::{DEFAULT_REPO, default_hub_cache, snapshot_dir};
use calamardo_model::{Model, kernels::Kernels, ops};
use mlx_rs::{Array, Dtype};

fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

fn floats(n: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| ((lcg(&mut s) >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * amp)
        .collect()
}

fn host(a: &Array) -> Vec<f32> {
    let a = ops::f32(a).unwrap();
    ops::eval(&[&a]).unwrap();
    a.as_slice::<f32>().to_vec()
}

#[test]
fn shared_gather_matches_gather_qmm_bit_for_bit() {
    ops::init();
    let kern = Kernels::new().unwrap();
    let mut failures = Vec::new();
    let (e, top_k) = (16, 8);
    // The model's N (2304): MLX's kernel choice depends on it.
    let n = 2304;
    for (bits, group, mode, global) in [
        (4, 16, c"nvfp4", true),
        (4, 16, c"nvfp4", false),
        (4, 32, c"mxfp4", false),
        (8, 32, c"mxfp8", false),
    ] {
        // 2304 and 1152 are the model's; 1024 takes MLX's fast kernel at 4 bits
        // too (2304 does at 8 bits).
        // 1088 leaves a partial last block at 8 bits.
        for k in [2304, 1152, 1024, 1088] {
            let mut seed = 7 + k as u32 + bits as u32 * 31 + global as u32;
            // No e4m3 NaN codes (0x7f, 0xff) in 8-bit weights: a NaN makes
            // every sum NaN and the comparison vacuous.
            let words: Vec<u32> = (0..e * n * k * bits / 32)
                .map(|_| {
                    let w = lcg(&mut seed);
                    if bits == 8 {
                        u32::from_le_bytes(
                            w.to_le_bytes()
                                .map(|b| if b & 0x7f == 0x7f { b - 1 } else { b }),
                        )
                    } else {
                        w
                    }
                })
                .collect();
            let w = Array::from_slice(&words, &[e as i32, n as i32, (k * bits / 32) as i32]);
            // Scales near 1: e4m3 0x30..0x3f, e8m0 125..129 (no NaN codes).
            let sc: Vec<u8> = (0..e * n * k / group)
                .map(|_| {
                    let r = (lcg(&mut seed) >> 16) as u8;
                    if group == 16 {
                        0x30 + r % 16
                    } else {
                        125 + r % 5
                    }
                })
                .collect();
            let scales = Array::from_slice(&sc, &[e as i32, n as i32, (k / group) as i32]);
            let gs = Array::from_slice(
                &floats(e, 3, 4.0)
                    .iter()
                    .map(|v| v.abs() + 0.5)
                    .collect::<Vec<_>>(),
                &[e as i32],
            );
            let gs = global.then_some(&gs);
            for t in [1usize, 2, 3, 5, 8] {
                // Each token picks top_k distinct experts out of 16, so
                // tokens share many.
                let idx: Vec<u32> = (0..t)
                    .flat_map(|i| (0..top_k).map(move |j| ((i * 3 + j * 2 + j / 4) % e) as u32))
                    .collect();
                let indices = Array::from_slice(&idx, &[t as i32, top_k as i32]);
                for per_route in [false, true] {
                    let rows = if per_route { t * top_k } else { t };
                    let x = ops::astype(
                        &Array::from_slice(
                            &floats(rows * k, 11 + t as u32, 2.0),
                            &[rows as i32, k as i32],
                        ),
                        Dtype::Bfloat16,
                    )
                    .unwrap();
                    let got = kern
                        .gather_qmv_shared(
                            &x,
                            &w,
                            &scales,
                            gs,
                            &indices,
                            per_route,
                            group as i32,
                            bits as i32,
                        )
                        .unwrap();
                    // gather_qmm as the model calls it.
                    let xq = if per_route {
                        ops::reshape(&x, &[1, t as i32, top_k as i32, 1, k as i32]).unwrap()
                    } else {
                        ops::reshape(&x, &[1, t as i32, 1, 1, k as i32]).unwrap()
                    };
                    let want = ops::gather_qmm(
                        &xq,
                        &w,
                        &scales,
                        &ops::reshape(&indices, &[1, t as i32, top_k as i32]).unwrap(),
                        group as i32,
                        bits as i32,
                        mode,
                        gs,
                        false,
                    )
                    .unwrap();
                    let (a, b) = (host(&got), host(&want));
                    assert_eq!(a.len(), b.len());
                    let diff = a
                        .iter()
                        .zip(&b)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    let worst = a
                        .iter()
                        .zip(&b)
                        .map(|(x, y)| (x - y).abs() / y.abs().max(1e-3))
                        .fold(0.0f32, f32::max);
                    if diff != 0 {
                        failures.push(format!(
                            "{mode:?} gs {global} K {k} T {t} per_route {per_route}: {diff} of {} differ (worst rel {worst})",
                            a.len()
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The same with the checkpoint's own experts (nvfp4 with global scales in
/// layers 0-19, mxfp8 in 20-23), both projections. Skipped if the
/// checkpoint is not downloaded.
#[test]
fn shared_gather_matches_gather_qmm_on_the_checkpoint() {
    let Ok(dir) = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main") else {
        eprintln!("checkpoint not downloaded; skipping");
        return;
    };
    let model = Model::load(&dir).unwrap();
    let kern = Kernels::new().unwrap();
    for layer in [0, 19, 20, 23] {
        for (p, (w, scales, gs, group, bits)) in model.expert_weights(layer).into_iter().enumerate()
        {
            let k = w.shape()[2] * 32 / bits;
            let per_route = p == 1;
            let mode = match (bits, group) {
                (8, 32) => c"mxfp8",
                (4, 32) => c"mxfp4",
                _ => c"nvfp4",
            };
            for t in [2, 6] {
                let idx: Vec<u32> = (0..t * 8).map(|r| ((r * 5 + r / 8) % 128) as u32).collect();
                let indices = Array::from_slice(&idx, &[t, 8]);
                let rows = if per_route { t * 8 } else { t };
                let x = ops::astype(
                    &Array::from_slice(
                        &floats((rows * k) as usize, 9 + layer as u32, 2.0),
                        &[rows, k],
                    ),
                    Dtype::Bfloat16,
                )
                .unwrap();
                let got = kern
                    .gather_qmv_shared(
                        &x,
                        &w,
                        &scales,
                        gs.as_ref(),
                        &indices,
                        per_route,
                        group,
                        bits,
                    )
                    .unwrap();
                let xq = if per_route {
                    ops::reshape(&x, &[1, t, 8, 1, k]).unwrap()
                } else {
                    ops::reshape(&x, &[1, t, 1, 1, k]).unwrap()
                };
                let want = ops::gather_qmm(
                    &xq,
                    &w,
                    &scales,
                    &ops::reshape(&indices, &[1, t, 8]).unwrap(),
                    group,
                    bits,
                    mode,
                    gs.as_ref(),
                    false,
                )
                .unwrap();
                let (a, b) = (host(&got), host(&want));
                let diff = a
                    .iter()
                    .zip(&b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count();
                assert_eq!(
                    diff,
                    0,
                    "layer {layer} projection {p}, {t} tokens: {diff} of {} differ",
                    a.len()
                );
            }
        }
    }
}
