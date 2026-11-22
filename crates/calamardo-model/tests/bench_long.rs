//! Decode and speculative round costs on a saved long session (e.g. 512k
//! tokens). Run with `BENCH_SESSION=<file.safetensors> --ignored --nocapture`.

use std::{sync::atomic::AtomicBool, time::Instant};

use calamardo_ckpt::{DEFAULT_REPO, default_hub_cache, snapshot_dir};
use calamardo_model::{
    Model,
    cache::KvFormat,
    draft::Drafter,
    generate::{Sampler, Session},
};

#[test]
#[ignore]
fn bench_long_session() {
    let dir = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main").unwrap();
    let model = Model::load(&dir).unwrap();
    let drafter = Drafter::load(&dir, &model).unwrap();
    let path = std::path::PathBuf::from(std::env::var("BENCH_SESSION").expect("BENCH_SESSION"));
    let kv = KvFormat::new(8, 64).unwrap();
    let t = Instant::now();
    let use_draft = std::env::var("BENCH_NO_DRAFT").is_err();
    let mut s = Session::load(
        &model,
        use_draft.then_some(&drafter),
        kv,
        &path,
        &Default::default(),
    )
    .unwrap();
    println!(
        "active {:.1} GB · MLX memory limit {:.1} GB",
        mlx_rs::memory::active_memory().unwrap() as f64 / 1e9,
        mlx_rs::memory::memory_limit().unwrap() as f64 / 1e9
    );
    println!(
        "loaded {} tokens in {:.2}s (drafter: {})",
        s.tokens.len(),
        t.elapsed().as_secs_f64(),
        s.has_drafter()
    );
    let no = AtomicBool::new(false);
    let mut sampler = Sampler::new(0.0, 1.0, 0);
    let mut last = 13u32;
    let reps = 6;
    let mut times = Vec::new();
    for _ in 0..12 {
        let t = Instant::now();
        let lp = s.feed(&[last], &no, &mut |_| {}).unwrap();
        last = sampler.sample(&lp).unwrap();
        times.push(format!("{:.0}", t.elapsed().as_secs_f64() * 1e3));
    }
    println!("decode ms: {}", times.join(" "));
    println!(
        "peak {:.1} GB",
        mlx_rs::memory::peak_memory().unwrap() as f64 / 1e9
    );
    if !s.has_drafter() {
        return;
    }
    let mut times = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let _ = drafter
            .draft(&model, s.draft_state_mut().unwrap(), last, 5)
            .unwrap();
        times.push(format!("{:.1}", t.elapsed().as_secs_f64() * 1e3));
    }
    println!("draft(5) ms: {}", times.join(" "));
    for block in [2usize, 4, 6] {
        let t = Instant::now();
        let mut got = 0;
        for _ in 0..reps {
            let r = s.speculate(last, block, &mut sampler, &[]).unwrap();
            got += r.tokens.len();
            last = *r.tokens.last().unwrap();
        }
        println!(
            "round({block})   {:7.1} ms, {:.1} tok",
            t.elapsed().as_secs_f64() * 1e3 / reps as f64,
            got as f64 / reps as f64
        );
    }
}
