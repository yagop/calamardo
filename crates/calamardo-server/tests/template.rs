//! M-9: the chat template renders exactly as Hugging Face transformers does
//! (expected outputs produced with `apply_chat_template` on the reference).

use calamardo_server::template::ChatTemplate;
use serde_json::Value;

#[test]
fn renders_like_transformers() {
    let t = ChatTemplate::from_source(include_str!("data/chat_template.jinja")).unwrap();
    let cases: Vec<Value> = serde_json::from_str(include_str!("data/template_cases.json")).unwrap();
    for (i, c) in cases.iter().enumerate() {
        let got = t
            .render(&c["messages"], c.get("tools"), c.get("reasoning_effort"))
            .unwrap();
        assert_eq!(got, c["expected"].as_str().unwrap(), "case {i}");
    }
}
