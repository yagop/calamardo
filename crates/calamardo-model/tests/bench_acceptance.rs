//! The drafter's acceptance curve: rounds of the full trained block (anchor
//! plus 7 drafts) with greedy verification, and how many drafts each round
//! accepts. From it, the acceptance rate (accepted / drafted) a fixed draft
//! depth would give, comparable with other runtimes whatever block policy
//! they use. `BENCH_NEEDLE=<29k prompt file> --ignored --nocapture`.

use std::sync::atomic::AtomicBool;

use calamardo_ckpt::{DEFAULT_REPO, default_hub_cache, snapshot_dir};
use calamardo_model::{
    Model,
    cache::KvFormat,
    draft::Drafter,
    generate::{Sampler, Session},
};

#[test]
#[ignore]
fn bench_acceptance_curve() {
    let dir = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main").unwrap();
    let model = Model::load(&dir).unwrap();
    let drafter = Drafter::load(&dir, &model).unwrap();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let chat = |user: &str| {
        format!(
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{user}<|end_message|>"
        )
    };
    let mut prompts = vec![
        (
            "code",
            chat(
                "Write a Python function that parses ISO 8601 dates with time zones, with tests, and explain the edge cases.",
            ),
        ),
        (
            "prose",
            chat(
                "Explain how a refrigerator works to a curious twelve-year-old, in about 300 words.",
            ),
        ),
    ];
    if let Ok(path) = std::env::var("BENCH_NEEDLE") {
        let raw = std::fs::read_to_string(path).unwrap();
        let essay = raw.replace(
            "Question: What is the secret access code for the archive room? Answer with the number only.",
            "Write a 300-word essay that connects the topics of these notes.",
        );
        prompts.push(("29k essay", essay));
    }
    let no = AtomicBool::new(false);
    let n = 300;
    for (name, text) in prompts {
        let prompt = tok.encode(text.as_str(), false).unwrap().get_ids().to_vec();
        let mut s = Session::new(&model, KvFormat::BF16).with_drafter(&drafter);
        let lp = s.feed(&prompt, &no, &mut |_| {}).unwrap();
        let mut sampler = Sampler::new(0.0, 1.0, 0);
        let mut last = sampler.sample(&lp).unwrap();
        let mut produced = 1;
        let mut hist = [0usize; 8];
        while produced < n {
            let r = s.speculate(last, 8, &mut sampler, &[]).unwrap();
            hist[r.accepted] += 1;
            produced += r.tokens.len();
            last = *r.tokens.last().unwrap();
        }
        let rounds: usize = hist.iter().sum();
        // P(at least k drafts accepted), k = 1..7.
        let at_least: Vec<f64> = (1..8)
            .map(|k| hist[k..].iter().sum::<usize>() as f64 / rounds as f64)
            .collect();
        // Rate with a fixed depth d: E[min(accepted, d)] / d.
        let rate: Vec<String> = (1..8)
            .map(|d| {
                let e: f64 = hist
                    .iter()
                    .enumerate()
                    .map(|(a, &c)| a.min(d) as f64 * c as f64)
                    .sum::<f64>()
                    / rounds as f64;
                format!("{d}:{:.0}%", 100.0 * e / d as f64)
            })
            .collect();
        println!(
            "{name} ({} prompt tokens, {rounds} rounds, {:.2} accepted per round)",
            prompt.len(),
            hist.iter().enumerate().map(|(a, &c)| a * c).sum::<usize>() as f64 / rounds as f64
        );
        println!(
            "  P(accepted >= k): {}",
            at_least
                .iter()
                .enumerate()
                .map(|(k, p)| format!("{}:{:.0}%", k + 1, 100.0 * p))
                .collect::<Vec<_>>()
                .join(" ")
        );
        println!("  rate at fixed depth: {}", rate.join(" "));
    }
}

/// The same curve on a given continuation (e.g. another runtime's output):
/// each round drafts 7 tokens after the text's own token, counts how many
/// match the text, and advances the session through the text itself.
/// `BENCH_TEXTS`: JSON files with "prompt" and "response", comma-separated.
#[test]
#[ignore]
fn bench_acceptance_on_given_text() {
    let dir = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main").unwrap();
    let model = Model::load(&dir).unwrap();
    let drafter = Drafter::load(&dir, &model).unwrap();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let no = AtomicBool::new(false);
    for path in std::env::var("BENCH_TEXTS").unwrap().split(',') {
        let j: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let prompt = tok
            .encode(j["prompt"].as_str().unwrap(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        let text = tok
            .encode(j["response"].as_str().unwrap(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        let mut s = Session::new(&model, KvFormat::BF16).with_drafter(&drafter);
        s.feed(&prompt, &no, &mut |_| {}).unwrap();
        let mut hist = [0usize; 8];
        let mut pos = 0;
        while pos + 8 < text.len() {
            let drafts = drafter
                .draft(&model, s.draft_state_mut().unwrap(), text[pos], 7)
                .unwrap();
            let m = drafts
                .iter()
                .zip(&text[pos + 1..])
                .take_while(|(a, b)| a == b)
                .count();
            hist[m] += 1;
            s.feed(&text[pos..pos + m + 1], &no, &mut |_| {}).unwrap();
            pos += m + 1;
        }
        let rounds: usize = hist.iter().sum();
        let at_least: Vec<String> = (1..8)
            .map(|k| {
                format!(
                    "{k}:{:.0}%",
                    100.0 * hist[k..].iter().sum::<usize>() as f64 / rounds as f64
                )
            })
            .collect();
        let rate: Vec<String> = (1..8)
            .map(|d| {
                let e: f64 = hist
                    .iter()
                    .enumerate()
                    .map(|(a, &c)| a.min(d) as f64 * c as f64)
                    .sum::<f64>()
                    / rounds as f64;
                format!("{d}:{:.0}%", 100.0 * e / d as f64)
            })
            .collect();
        println!(
            "{path}: {} prompt tokens, {} text tokens, {rounds} rounds",
            prompt.len(),
            text.len()
        );
        println!("  P(accepted >= k): {}", at_least.join(" "));
        println!("  rate at fixed depth: {}", rate.join(" "));
    }
}
