//! Target forward cost by token count and mode (decode, prefill chunk,
//! speculative verification) on a short context. `--ignored --nocapture`.

use std::sync::atomic::AtomicBool;

use calamardo_ckpt::{DEFAULT_REPO, default_hub_cache, snapshot_dir};
use calamardo_model::{Model, cache::KvFormat, generate::Session};

#[test]
#[ignore]
fn bench_forward_modes() {
    let dir = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main").unwrap();
    let model = Model::load(&dir).unwrap();
    let prompt: Vec<u32> = (0..1024)
        .map(|i| 1000 + (i * 7919 % 50_000) as u32)
        .collect();
    let no = AtomicBool::new(false);
    for kv in [KvFormat::BF16, KvFormat::new(8, 64).unwrap()] {
        let mut s = Session::new(&model, kv);
        s.feed(&prompt, &no, &mut |_| {}).unwrap();
        for (l, verify) in [(1, false), (2, false), (2, true), (4, true), (8, true)] {
            let toks: Vec<u32> = (0..l).map(|i| 2000 + i as u32).collect();
            s.bench_forward(&toks, verify).unwrap();
            s.bench_forward(&toks, verify).unwrap();
            let reps = 10;
            let ms: f64 = (0..reps)
                .map(|_| s.bench_forward(&toks, verify).unwrap())
                .sum::<f64>()
                * 1e3
                / reps as f64;
            println!(
                "{kv:?} L {l} {}: {ms:.1} ms",
                if verify { "verify" } else { "chunk" }
            );
        }
    }
}
