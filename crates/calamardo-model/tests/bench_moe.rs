//! Cost of the experts over the 48 passes for a few tokens (decode and
//! speculative verification). Run with `--ignored --nocapture`.

use std::time::Instant;

use calamardo_ckpt::{DEFAULT_REPO, default_hub_cache, snapshot_dir};
use calamardo_model::{Model, ops};
use mlx_rs::{Array, Dtype};

#[test]
#[ignore]
fn bench_moe_passes() {
    let dir = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main").unwrap();
    let model = Model::load(&dir).unwrap();
    let d = model.config.text_config.hidden_size as i32;
    for (l, same) in [
        (1, false),
        (2, false),
        (4, false),
        (8, false),
        (2, true),
        (4, true),
        (8, true),
    ] {
        let mut seed = 1u32 + l as u32;
        let rows = if same { 1 } else { l };
        let r: Vec<f32> = (0..rows * d)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        // Identical rows pick the same experts in every pass.
        let v: Vec<f32> = (0..(l * d) as usize).map(|i| r[i % r.len()]).collect();
        let x = ops::astype(&Array::from_slice(&v, &[1, l, d]), Dtype::Bfloat16).unwrap();
        ops::eval(&[&model.moe_chain(&x).unwrap()]).unwrap();
        let reps = 10;
        let t = Instant::now();
        for _ in 0..reps {
            ops::eval(&[&model.moe_chain(&x).unwrap()]).unwrap();
        }
        println!(
            "{l} tokens{}: {:.2} ms for the 48 passes' experts",
            if same { " (identical)" } else { "" },
            t.elapsed().as_secs_f64() * 1e3 / reps as f64
        );
    }
}
