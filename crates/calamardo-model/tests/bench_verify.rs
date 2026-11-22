//! Where a speculative round spends its time, at short and long context.
//! Run with `--ignored --nocapture`.

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
fn bench_speculative_round() {
    let dir = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main").unwrap();
    let model = Model::load(&dir).unwrap();
    let drafter = Drafter::load(&dir, &model).unwrap();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let path = std::env::var("BENCH_PROMPT").expect("BENCH_PROMPT: a long prompt file");
    let all = tok
        .encode(std::fs::read_to_string(path).unwrap(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    let no = AtomicBool::new(false);
    for len in [1024usize, 31_000] {
        let prompt = &all[..len.min(all.len())];
        let mut s = Session::new(&model, KvFormat::BF16).with_drafter(&drafter);
        let lp = s.feed(prompt, &no, &mut |_| {}).unwrap();
        let mut sampler = Sampler::new(0.0, 1.0, 0);
        let mut last = sampler.sample(&lp).unwrap();
        let reps = 8;
        // Plain decode steps.
        let t = Instant::now();
        for _ in 0..reps {
            let lp = s.feed(&[last], &no, &mut |_| {}).unwrap();
            last = sampler.sample(&lp).unwrap();
        }
        let decode = t.elapsed().as_secs_f64() / reps as f64;
        // Drafting alone.
        let t = Instant::now();
        for _ in 0..reps {
            let _ = drafter
                .draft(&model, s.draft_state().unwrap(), last, 7)
                .unwrap();
        }
        let draft = t.elapsed().as_secs_f64() / reps as f64;
        print!(
            "context {len:>6}: decode {:5.1} ms · draft(7) {:5.1} ms",
            decode * 1e3,
            draft * 1e3
        );
        for block in [2usize, 4, 8] {
            let t = Instant::now();
            let mut got = 0;
            for _ in 0..reps {
                let r = s.speculate(last, block, &mut sampler, &[]).unwrap();
                got += r.tokens.len();
                last = *r.tokens.last().unwrap();
            }
            let round = t.elapsed().as_secs_f64() / reps as f64;
            print!(
                " · round({block}) {:5.1} ms, {:.1} tok",
                round * 1e3,
                got as f64 / reps as f64
            );
        }
        println!();
    }
}
