//! bf16 projections with a few rows (decode and speculative verification),
//! with the model's shapes. `--ignored --nocapture`.

use std::time::Instant;

use calamardo_model::ops;
use mlx_rs::{Array, Dtype};

fn rand(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

#[test]
#[ignore]
fn bench_short_projections() {
    ops::init();
    let shapes = [
        (3072, 2304),
        (2304, 3072),
        (1024, 2304),
        (384, 2304),
        (64, 4608),
        (2304, 64),
        (201_024, 2304),
    ];
    for (n, k) in shapes {
        let w = ops::astype(
            &Array::from_slice(&rand(n * k, 7), &[n as i32, k as i32]),
            Dtype::Bfloat16,
        )
        .unwrap();
        ops::eval(&[&w]).unwrap();
        let mut line = format!("{n:>6} x {k:<5}");
        for m in [1, 2, 4, 8] {
            let x = ops::astype(
                &Array::from_slice(&rand(m * k, 3), &[1, m as i32, k as i32]),
                Dtype::Bfloat16,
            )
            .unwrap();
            ops::eval(&[&x]).unwrap();
            // 48 independent calls, as many as the passes of one forward.
            let run = || {
                let outs: Vec<Array> = (0..48).map(|_| ops::linear(&x, &w).unwrap()).collect();
                ops::eval(&outs.iter().collect::<Vec<_>>()).unwrap();
            };
            run();
            let reps = 10;
            let t = Instant::now();
            for _ in 0..reps {
                run();
            }
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64 / 48.0;
            line += &format!("  M={m}: {us:6.1} us");
        }
        let us = (n * k * 2) as f64 / 1e6;
        println!("{line}   (reading the weights at 1 TB/s: {us:.1} us)");
    }
}
