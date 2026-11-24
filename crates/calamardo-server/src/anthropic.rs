//! Anthropic's Messages API (`/v1/messages`), for clients that speak it.

use std::{
    convert::Infallible,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::HeaderMap,
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
};
use serde_json::{Value, json};
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{
    parser::Delta,
    server::{
        App, Asked, Collector, Gen, Item, Refused, blocking, chat_prompt, completion_id, drive,
        effort_or_default, json_body, normalize_messages, precheck, random_seed, raw_prompt,
        stop_list,
    },
    worker::{Done, Finish},
};

fn error_response(e: &Refused) -> Response {
    let kind = match e {
        Refused::Unauthorized => "authentication_error",
        Refused::BadRequest(_) => "invalid_request_error",
        Refused::Busy(_) => "rate_limit_error",
        Refused::Unavailable(_) => "overloaded_error",
        Refused::Internal(_) => "api_error",
    };
    let body = json!({"type": "error", "error": {"type": kind, "message": e.message()}});
    (e.status(), Json(body)).into_response()
}

fn new_id(prefix: &str) -> String {
    format!(
        "{prefix}_{}",
        completion_id().trim_start_matches("chatcmpl-")
    )
}

/// Text of a content value: a string or blocks with text.
/// What stands in for an image or a document: the model reads text only,
/// and refusing the block would break every later turn that keeps it in the
/// history.
fn omitted(kind: &str) -> String {
    format!("[{kind} omitted: this model reads text only]")
}

fn blocks_text(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(blocks) => {
            let mut out = String::new();
            for b in blocks {
                match b.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        out.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                    }
                    Some(kind @ ("image" | "document")) => out.push_str(&omitted(kind)),
                    Some(other) => {
                        return Err(format!(
                            "content blocks of type {other} are not supported here"
                        ));
                    }
                    None => return Err("content blocks need a type".into()),
                }
            }
            Ok(out)
        }
        Value::Null => Ok(String::new()),
        _ => Err("content must be a string or an array of blocks".into()),
    }
}

/// Anthropic messages (and system) into the chat messages of the template.
fn chat_messages(body: &Value) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    if let Some(system) = body.get("system").filter(|s| !s.is_null()) {
        let text = blocks_text(system)?;
        if !text.is_empty() {
            out.push(json!({"role": "system", "content": text}));
        }
    }
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or("messages must be an array")?;
    for m in messages {
        let role = m
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or("each message needs a role")?;
        let content = m.get("content").unwrap_or(&Value::Null);
        let blocks = match content {
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            Value::Array(b) => b.clone(),
            _ => return Err("content must be a string or an array of blocks".into()),
        };
        match role {
            "user" => {
                // Tool results answer the previous assistant turn, so they
                // come first, as tool messages; the rest is the user's text.
                let mut text = String::new();
                for b in &blocks {
                    match b.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            text.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                        }
                        Some("tool_result") => {
                            let mut result = blocks_text(b.get("content").unwrap_or(&Value::Null))?;
                            if b.get("is_error").and_then(|e| e.as_bool()) == Some(true) {
                                result = format!("Error: {result}");
                            }
                            out.push(json!({
                                "role": "tool",
                                "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(Value::Null),
                                "content": result,
                            }));
                        }
                        Some(kind @ ("image" | "document")) => text.push_str(&omitted(kind)),
                        Some(other) => {
                            return Err(format!(
                                "content blocks of type {other} are not supported"
                            ));
                        }
                        None => return Err("content blocks need a type".into()),
                    }
                }
                if !text.is_empty() || !blocks.iter().any(|b| b["type"] == "tool_result") {
                    out.push(json!({"role": "user", "content": text}));
                }
            }
            "assistant" => {
                let (mut text, mut thinking, mut calls) =
                    (String::new(), String::new(), Vec::new());
                for b in &blocks {
                    match b.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            text.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                        }
                        Some("thinking") => thinking
                            .push_str(b.get("thinking").and_then(|t| t.as_str()).unwrap_or("")),
                        Some("redacted_thinking") => {}
                        Some("tool_use") => calls.push(json!({
                            "id": b.get("id").cloned().unwrap_or(Value::Null),
                            "type": "function",
                            "function": {
                                "name": b.get("name").cloned().unwrap_or(json!("")),
                                "arguments": b.get("input").cloned().unwrap_or(json!({})),
                            },
                        })),
                        Some(other) => {
                            return Err(format!(
                                "content blocks of type {other} are not supported"
                            ));
                        }
                        None => return Err("content blocks need a type".into()),
                    }
                }
                let mut msg = json!({"role": "assistant", "content": text});
                if !thinking.is_empty() {
                    msg["reasoning_content"] = json!(thinking);
                }
                if !calls.is_empty() {
                    msg["tool_calls"] = Value::Array(calls);
                }
                out.push(msg);
            }
            other => return Err(format!("unknown role {other}")),
        }
    }
    normalize_messages(&Value::Array(out))
}

/// Anthropic tools into chat function tools; `tool_choice` none drops them.
fn chat_tools(body: &Value) -> Result<Option<Value>, String> {
    if body.pointer("/tool_choice/type").and_then(|t| t.as_str()) == Some("none") {
        return Ok(None);
    }
    let Some(Value::Array(tools)) = body.get("tools") else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for t in tools {
        // Client tools have no type or "custom"; server tools (web search,
        // code execution...) need Anthropic's servers.
        match t.get("type").and_then(|v| v.as_str()) {
            None | Some("custom") => out.push(json!({
                "type": "function",
                "function": {
                    "name": t.get("name").cloned().unwrap_or(json!("")),
                    "description": t.get("description").cloned().unwrap_or(json!("")),
                    "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type": "object"})),
                },
            })),
            Some(other) => return Err(format!("tools of type {other} are not supported")),
        }
    }
    Ok((!out.is_empty()).then_some(Value::Array(out)))
}

/// Extended thinking as a reasoning effort: off is "none", a budget picks a
/// level, and no setting keeps the server's default.
fn thinking_effort(body: &Value) -> Option<Value> {
    let t = body.get("thinking")?;
    match t.get("type").and_then(|v| v.as_str()) {
        Some("disabled") => Some(json!("none")),
        Some("enabled") => {
            let budget = t.get("budget_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            Some(json!(match budget {
                0..2048 => "low",
                2048..8192 => "medium",
                _ => "high",
            }))
        }
        _ => None,
    }
}

/// How `thinking` became the effort, for the log.
fn thinking_note(body: &Value) -> Option<String> {
    let t = body.get("thinking")?;
    match t.get("type").and_then(|v| v.as_str()) {
        Some("enabled") => Some(format!(
            "from thinking budget {}",
            t.get("budget_tokens").and_then(|v| v.as_u64()).unwrap_or(0)
        )),
        Some("disabled") => Some("from thinking disabled".into()),
        Some(other) => Some(format!(
            "thinking type {:?} sent",
            other.chars().take(30).collect::<String>()
        )),
        None => None,
    }
}

struct MessagesRequest {
    model: String,
    job: Gen,
    stop: Vec<String>,
    stream: bool,
}

fn parse_messages(app: &App, body: &Value) -> Result<MessagesRequest, String> {
    let messages = chat_messages(body)?;
    let tools = chat_tools(body)?;
    let prompt = chat_prompt(
        app,
        &Value::Array(messages),
        tools.as_ref(),
        effort_or_default(app, thinking_effort(body)).as_ref(),
    )?;
    let num = |k: &str| body.get(k).and_then(|v| v.as_f64());
    Ok(MessagesRequest {
        model: body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&app.cfg.model_name)
            .to_string(),
        job: Gen {
            prompt,
            max_tokens: num("max_tokens")
                .map_or(app.cfg.default_max_tokens, |v| v as usize)
                .max(1),
            temperature: num("temperature").unwrap_or(1.0) as f32,
            top_p: num("top_p").unwrap_or(1.0) as f32,
            seed: random_seed(),
            summary: Asked {
                api: "messages",
                model: body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&app.cfg.model_name),
                messages: body["messages"].as_array().map(|m| m.len()),
                tools: tools
                    .as_ref()
                    .and_then(|t| t.as_array())
                    .map_or(0, |t| t.len()),
                effort: thinking_effort(body).as_ref(),
                effort_note: thinking_note(body),
                max_tokens: num("max_tokens"),
                max_note: None,
                stream: body.get("stream").and_then(|v| v.as_bool()) == Some(true),
            }
            .line(app),
        },
        stop: stop_list(body.get("stop_sequences")),
        stream: body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    })
}

fn stop_reason(c: &Collector, d: &Done) -> (&'static str, Value) {
    if !c.tool_calls.is_empty() {
        ("tool_use", Value::Null)
    } else if let Some(s) = &c.matched_stop {
        ("stop_sequence", json!(s))
    } else if d.finish == Finish::Length {
        ("max_tokens", Value::Null)
    } else {
        ("end_turn", Value::Null)
    }
}

fn usage(d: &Done) -> Value {
    json!({
        "input_tokens": d.prompt_tokens - d.cached_tokens,
        "cache_read_input_tokens": d.cached_tokens,
        "cache_creation_input_tokens": 0,
        "output_tokens": d.completion_tokens,
    })
}

fn tool_input(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
}

pub(crate) async fn messages(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let ready = async {
        precheck(&app, &headers)?;
        let body = json_body(&body)?;
        let req = {
            let app = app.clone();
            blocking(move || parse_messages(&app, &body)).await?
        };
        let run = crate::server::start(&app, &req.job).await?;
        Ok::<_, Refused>((req, run))
    };
    let (req, mut run) = match ready.await {
        Ok(v) => v,
        Err(e) => return error_response(&e),
    };
    let id = new_id("msg");
    let mut c = Collector::new(app.specials, req.stop.clone());
    if !req.stream {
        return match drive(&app, &mut run, &mut c, |_| {}).await {
            Ok(d) => {
                let mut content = Vec::new();
                if !c.reasoning.is_empty() {
                    content.push(
                        json!({"type": "thinking", "thinking": c.reasoning, "signature": ""}),
                    );
                }
                if !c.content.is_empty() {
                    content.push(json!({"type": "text", "text": c.content}));
                }
                for (name, args) in &c.tool_calls {
                    content.push(json!({"type": "tool_use", "id": new_id("toolu"), "name": name, "input": tool_input(args)}));
                }
                let (reason, sequence) = stop_reason(&c, &d);
                Json(json!({
                    "id": id, "type": "message", "role": "assistant", "model": req.model,
                    "content": content, "stop_reason": reason, "stop_sequence": sequence,
                    "usage": usage(&d),
                }))
                .into_response()
            }
            Err(e) => error_response(&Refused::Internal(e)),
        };
    }
    let (tx, rx) = unbounded_channel();
    tokio::spawn(async move {
        let mut s = MessagesStream {
            tx,
            cancel: run.cancel.clone(),
            next: 0,
            open: None,
        };
        s.event(
            "message_start",
            json!({"message": {
                "id": id, "type": "message", "role": "assistant", "model": req.model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": req.job.prompt.len() - run.cached,
                          "cache_read_input_tokens": run.cached,
                          "cache_creation_input_tokens": 0, "output_tokens": 0},
            }}),
        );
        let result = drive(&app, &mut run, &mut c, |item| match item {
            Item::Delta(d) => s.delta(d),
            // Anthropic's keep-alive; writing also notices a client that left.
            Item::Prefill { .. } => s.event("ping", json!({})),
        })
        .await;
        s.close();
        match result {
            Ok(d) => {
                let (reason, sequence) = stop_reason(&c, &d);
                s.event(
                    "message_delta",
                    json!({"delta": {"stop_reason": reason, "stop_sequence": sequence},
                           "usage": {"output_tokens": d.completion_tokens}}),
                );
                s.event("message_stop", json!({}));
            }
            Err(e) => s.event(
                "error",
                json!({"error": {"type": "api_error", "message": e}}),
            ),
        }
    });
    Sse::new(UnboundedReceiverStream::new(rx))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

#[derive(PartialEq)]
enum Block {
    Thinking,
    Text,
}

/// The streaming events: a content block opens when the kind of output
/// changes and closes before the next one.
struct MessagesStream {
    tx: UnboundedSender<Result<SseEvent, Infallible>>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    /// Index of the next block.
    next: usize,
    open: Option<Block>,
}

impl MessagesStream {
    fn event(&mut self, kind: &str, mut data: Value) {
        data["type"] = json!(kind);
        let ev = SseEvent::default().event(kind).data(data.to_string());
        if self.tx.send(Ok(ev)).is_err() {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn open(&mut self, kind: Block) {
        if self.open.as_ref() == Some(&kind) {
            return;
        }
        self.close();
        let block = match kind {
            Block::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
            Block::Text => json!({"type": "text", "text": ""}),
        };
        self.event(
            "content_block_start",
            json!({"index": self.next, "content_block": block}),
        );
        self.open = Some(kind);
    }

    fn close(&mut self) {
        if let Some(kind) = self.open.take() {
            if kind == Block::Thinking {
                let delta = json!({"type": "signature_delta", "signature": ""});
                self.event(
                    "content_block_delta",
                    json!({"index": self.next, "delta": delta}),
                );
            }
            self.event("content_block_stop", json!({"index": self.next}));
            self.next += 1;
        }
    }

    fn delta(&mut self, d: Delta) {
        match d {
            Delta::Reasoning(s) => {
                self.open(Block::Thinking);
                let delta = json!({"type": "thinking_delta", "thinking": s});
                self.event(
                    "content_block_delta",
                    json!({"index": self.next, "delta": delta}),
                );
            }
            Delta::Content(s) => {
                self.open(Block::Text);
                let delta = json!({"type": "text_delta", "text": s});
                self.event(
                    "content_block_delta",
                    json!({"index": self.next, "delta": delta}),
                );
            }
            Delta::ToolCall { name, arguments } => {
                self.close();
                let index = self.next;
                let block =
                    json!({"type": "tool_use", "id": new_id("toolu"), "name": name, "input": {}});
                self.event(
                    "content_block_start",
                    json!({"index": index, "content_block": block}),
                );
                let delta = json!({"type": "input_json_delta", "partial_json": arguments});
                self.event(
                    "content_block_delta",
                    json!({"index": index, "delta": delta}),
                );
                self.event("content_block_stop", json!({"index": index}));
                self.next += 1;
            }
        }
    }
}

/// `POST /v1/messages/count_tokens`: the prompt's length with the template.
pub(crate) async fn count_tokens(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let counted = async {
        if !crate::server::authorized(&app, &headers) {
            return Err(Refused::Unauthorized);
        }
        let body = json_body(&body)?;
        let app = app.clone();
        blocking(move || {
            let messages = chat_messages(&body)?;
            let tools = chat_tools(&body)?;
            let text = app
                .template
                .render(
                    &Value::Array(messages),
                    tools.as_ref(),
                    effort_or_default(&app, thinking_effort(&body)).as_ref(),
                )
                .map_err(|e| e.to_string())?;
            Ok(raw_prompt(&app, &text)?.len())
        })
        .await
    };
    match counted.await {
        Ok(n) => Json(json!({"input_tokens": n})).into_response(),
        Err(e) => error_response(&e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_map_to_chat() {
        let body = json!({
            "system": [{"type": "text", "text": "Be brief."}],
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Use the tool.", "signature": "x"},
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "t1", "name": "weather", "input": {"city": "Paris"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "18 C"}]},
                    {"type": "text", "text": "And tomorrow?"},
                ]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "weather", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "rain", "is_error": true}]},
            ],
        });
        let m = chat_messages(&body).unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "Be brief."}));
        assert_eq!(
            m[1],
            json!({"role": "user", "content": "Weather in Paris?"})
        );
        assert_eq!(m[2]["reasoning_content"], "Use the tool.");
        assert_eq!(m[2]["content"], "Checking.");
        assert_eq!(
            m[2]["tool_calls"][0]["function"]["arguments"],
            json!({"city": "Paris"})
        );
        assert_eq!(
            m[3],
            json!({"role": "tool", "tool_call_id": "t1", "content": "18 C"})
        );
        assert_eq!(m[4], json!({"role": "user", "content": "And tomorrow?"}));
        assert_eq!(
            m[6],
            json!({"role": "tool", "tool_call_id": "t2", "content": "Error: rain"})
        );
        assert_eq!(m.len(), 7);
        let image = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Look: "}, {"type": "image", "source": {}}]}]});
        let m = chat_messages(&image).unwrap();
        assert_eq!(
            m[0]["content"],
            "Look: [image omitted: this model reads text only]"
        );
    }

    #[test]
    fn thinking_budget_picks_effort() {
        assert_eq!(thinking_effort(&json!({})), None);
        assert_eq!(
            thinking_effort(&json!({"thinking": {"type": "disabled"}})),
            Some(json!("none"))
        );
        let budget =
            |b: u64| thinking_effort(&json!({"thinking": {"type": "enabled", "budget_tokens": b}}));
        assert_eq!(budget(1024), Some(json!("low")));
        assert_eq!(budget(4096), Some(json!("medium")));
        assert_eq!(budget(16000), Some(json!("high")));
    }

    #[test]
    fn tools_map_input_schema() {
        let body = json!({"tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]});
        let t = chat_tools(&body).unwrap().unwrap();
        assert_eq!(t[0]["function"]["parameters"], json!({"type": "object"}));
        assert!(
            chat_tools(&json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}))
                .is_err()
        );
        let none = json!({"tool_choice": {"type": "none"}, "tools": [{"name": "f"}]});
        assert!(chat_tools(&none).unwrap().is_none());
    }
}
