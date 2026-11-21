//! M-10: reasoning, content and tool calls from Inkling 2 output tokens.
//! Needs the checkpoint's tokenizer in the Hugging Face cache; skipped if absent.

use calamardo_server::parser::{Delta, ResponseParser, Specials};
use tokenizers::Tokenizer;

fn tokenizer() -> Option<Tokenizer> {
    let home = std::env::var("HOME").ok()?;
    let repo = format!("{home}/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini");
    let rev = std::fs::read_to_string(format!("{repo}/refs/main")).ok()?;
    Tokenizer::from_file(format!("{repo}/snapshots/{}/tokenizer.json", rev.trim())).ok()
}

fn parse(t: &Tokenizer, text: &str) -> (Vec<Delta>, bool) {
    let ids = t.encode(text, false).unwrap().get_ids().to_vec();
    let mut p = ResponseParser::new(Specials::from_tokenizer(t).unwrap());
    let mut out: Vec<Delta> = ids.iter().flat_map(|&id| p.push(t, id)).collect();
    out.extend(p.finish(t));
    // Merge consecutive text pieces of the same kind.
    let mut merged: Vec<Delta> = Vec::new();
    for d in out {
        match (merged.last_mut(), d) {
            (Some(Delta::Reasoning(a)), Delta::Reasoning(b)) => a.push_str(&b),
            (Some(Delta::Content(a)), Delta::Content(b)) => a.push_str(&b),
            (_, d) => merged.push(d),
        }
    }
    (merged, p.finished)
}

#[test]
fn reasoning_content_and_tool_calls() {
    let Some(t) = tokenizer() else {
        eprintln!("tokenizer not downloaded; skipping");
        return;
    };
    let (d, done) = parse(
        &t,
        "<|message_model|><|content_thinking|>Need the weather. 🐙<|end_message|>\
         <|message_model|><|content_text|>Let me check.<|end_message|>\
         <|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Vigo\"}}<|end_message|>\
         <|content_model_end_sampling|>",
    );
    assert!(done);
    assert_eq!(
        d,
        vec![
            Delta::Reasoning("Need the weather. 🐙".into()),
            Delta::Content("Let me check.".into()),
            Delta::ToolCall {
                name: "get_weather".into(),
                arguments: "{\"city\":\"Vigo\"}".into()
            },
        ]
    );

    // Cut by max_tokens inside the answer.
    let (d, done) = parse(&t, "<|message_model|><|content_text|>Partial answer");
    assert!(!done);
    assert_eq!(d, vec![Delta::Content("Partial answer".into())]);
}
