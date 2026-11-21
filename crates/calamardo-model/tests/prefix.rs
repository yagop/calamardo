//! Prefix cache mechanics on the real model (milestone 5): returning to a
//! snapshot, or forking from it, reproduces the same greedy tokens as a cold
//! run. Skipped if the checkpoint is not downloaded.

use std::sync::atomic::AtomicBool;

use calamardo_ckpt::{DEFAULT_REPO, default_hub_cache, snapshot_dir};
use calamardo_model::{
    Model,
    cache::KvFormat,
    generate::{self, Reuse, Sampler, Session},
};

const PROMPT: [u32; 24] = [
    200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 200010, 200000, 200004, 100887,
    149881, 150011, 156021, 151661, 159804, 151505, 82, 149894, 150382, 13, 200010,
];

fn decode(session: &mut Session, first: &mlx_rs::Array, n: usize) -> Vec<u32> {
    let no = AtomicBool::new(false);
    let mut sampler = Sampler::new(0.0, 1.0, 0);
    let mut out = vec![sampler.sample(first).unwrap()];
    while out.len() < n {
        let lp = session
            .feed(&[*out.last().unwrap()], &no, &mut |_| {})
            .unwrap();
        out.push(sampler.sample(&lp).unwrap());
    }
    out
}

#[test]
fn restore_and_fork_reproduce_greedy() {
    let Ok(dir) = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main") else {
        eprintln!("checkpoint not downloaded; skipping");
        return;
    };
    let model = Model::load(&dir).unwrap();
    let no = AtomicBool::new(false);
    for kv in [KvFormat::BF16, KvFormat::new(8, 64).unwrap()] {
        let cold = generate::greedy(
            &model,
            &PROMPT,
            30,
            generate::PREFILL_STEP,
            kv,
            |_| {},
            |_| true,
        )
        .unwrap();

        let mut s = Session::new(&model, kv);
        let lp = s.feed(&PROMPT, &no, &mut |_| {}).unwrap();
        let first = decode(&mut s, &lp, 30);
        assert_eq!(first, cold, "{kv:?}: session vs greedy");

        // A retry of the same prompt restarts from the prompt-end snapshot.
        let reuse = s.reuse(&PROMPT, false).unwrap();
        assert_eq!(
            reuse,
            Reuse::Snapshot {
                index: 0,
                len: PROMPT.len() - 1
            }
        );
        let mut fork = s.fork(0);
        assert_eq!(fork.tokens.len(), PROMPT.len() - 1);
        let lp = fork
            .feed(&PROMPT[PROMPT.len() - 1..], &no, &mut |_| {})
            .unwrap();
        assert_eq!(decode(&mut fork, &lp, 30), cold, "{kv:?}: fork");

        // The original is untouched by the fork and can be rewound in place.
        assert_eq!(s.tokens.len(), PROMPT.len() + 29);
        s.restore(0);
        let lp = s
            .feed(&PROMPT[PROMPT.len() - 1..], &no, &mut |_| {})
            .unwrap();
        assert_eq!(decode(&mut s, &lp, 30), cold, "{kv:?}: restore");

        // The conversation continues from the live end.
        let mut next = s.tokens.clone();
        next.extend([200010, 200000, 200004, 13]);
        assert_eq!(
            s.reuse(&next, false),
            Some(Reuse::Live {
                len: s.tokens.len()
            })
        );
    }
}
