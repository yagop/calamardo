//! Speculative decoding (M-7) against plain greedy decoding on the real
//! model. Verification runs several tokens at once (other kernels, other
//! summation order), so the two can part at a near tie; any earlier or
//! clear-cut difference would mean the caches were not rewound correctly.
//! Skipped if the checkpoint is not downloaded.

use std::sync::atomic::AtomicBool;

use calamardo_ckpt::{DEFAULT_REPO, default_hub_cache, snapshot_dir};
use calamardo_model::{
    Model,
    cache::KvFormat,
    draft::Drafter,
    generate::{Sampler, Session},
    ops,
};

#[test]
fn speculative_matches_greedy_up_to_ties() {
    let Ok(dir) = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main") else {
        eprintln!("checkpoint not downloaded; skipping");
        return;
    };
    let model = Model::load(&dir).unwrap();
    let drafter = Drafter::load(&dir, &model).unwrap();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let text = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>List the planets of the solar system with one fact each.<|end_message|>";
    let long = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/prompts/needle-8k.txt"
    ))
    .unwrap();
    let no = AtomicBool::new(false);
    let n = 200;

    // Short (windows not full) and 8k tokens (full, trimmed windows), with
    // bf16, 8- and 4-bit caches.
    let cases = [
        (text.to_string(), KvFormat::BF16),
        (text.to_string(), KvFormat::new(8, 64).unwrap()),
        (long.clone(), KvFormat::BF16),
        (long, KvFormat::new(4, 64).unwrap()),
    ];
    for (text, kv) in cases {
        let prompt = tok.encode(text.as_str(), false).unwrap().get_ids().to_vec();
        // Plain greedy, keeping the gap between the two best tokens.
        let mut s = Session::new(&model, kv);
        let mut lp = s.feed(&prompt, &no, &mut |_| {}).unwrap();
        let (mut greedy, mut gaps) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let l = ops::f32(&lp).unwrap();
            ops::eval(&[&l]).unwrap();
            let v = l.as_slice::<f32>();
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.select_nth_unstable_by(1, |&a, &b| v[b].total_cmp(&v[a]));
            let (a, b) = if v[idx[0]] >= v[idx[1]] {
                (idx[0], idx[1])
            } else {
                (idx[1], idx[0])
            };
            greedy.push(a as u32);
            gaps.push(v[a] - v[b]);
            lp = s.feed(&[a as u32], &no, &mut |_| {}).unwrap();
        }

        // Speculative.
        let mut s = Session::new(&model, kv).with_drafter(&drafter);
        let lp = s.feed(&prompt, &no, &mut |_| {}).unwrap();
        let mut sampler = Sampler::new(0.0, 1.0, 0);
        let mut spec = vec![sampler.sample(&lp).unwrap()];
        let (mut drafted, mut accepted) = (0, 0);
        while spec.len() < n {
            // A fixed block: the test must not depend on timing.
            let block = 4.min(n - spec.len());
            if block < 2 {
                break;
            }
            let r = s
                .speculate(*spec.last().unwrap(), block, &mut sampler, &[])
                .unwrap();
            drafted += r.drafted;
            accepted += r.accepted;
            spec.extend(r.tokens);
        }
        spec.truncate(n);
        assert_eq!(
            s.tokens.len(),
            prompt.len() + spec.len() - 1,
            "{kv:?}: session tokens"
        );

        let first = greedy.iter().zip(&spec).position(|(a, b)| a != b);
        eprintln!(
            "{} tokens, {kv:?}: acceptance {:.0}% ({accepted}/{drafted}), first difference at {first:?}{}",
            prompt.len(),
            100.0 * accepted as f64 / drafted as f64,
            first.map_or(String::new(), |i| format!(" (gap {:.3} nats)", gaps[i]))
        );
        if let Some(i) = first {
            assert!(
                gaps[i] < 0.25,
                "{kv:?}: tokens differ at {i} where greedy was clear-cut (gap {})",
                gaps[i]
            );
        }
    }
}
