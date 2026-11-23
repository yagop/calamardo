//! Ollama's native API (`/api/*`), for clients that speak it.

use std::{
    sync::{Arc, atomic::Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use tokio::sync::mpsc::unbounded_channel;
use tokio_stream::{StreamExt, wrappers::UnboundedReceiverStream};

use crate::{
    parser::Delta,
    server::{
        App, Collector, Gen, Item, Refused, blocking, chat_prompt, drive, effort_or_default,
        json_body, normalize_messages, precheck, random_seed, raw_prompt, start, stop_list,
    },
    worker::{Done, Finish},
};

/// The Ollama API version this server follows (clients check it).
const OLLAMA_VERSION: &str = "0.12.0";

fn error_response(e: &Refused) -> Response {
    (e.status(), Json(json!({"error": e.message()}))).into_response()
}

/// RFC 3339 in UTC, as Ollama writes `created_at`.
fn rfc3339(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs();
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        d.subsec_nanos()
    )
}

/// `think`: false is "none", a level is that level, true or nothing keeps
/// the server's default.
fn think_effort(body: &Value) -> Option<Value> {
    match body.get("think") {
        Some(Value::Bool(false)) => Some(json!("none")),
        Some(Value::String(s)) => Some(json!(s)),
        _ => None,
    }
}

/// `format`: "json" or a JSON schema, as an instruction.
fn format_message(body: &Value) -> Option<Value> {
    let text = match body.get("format")? {
        Value::String(s) if s == "json" => {
            "Respond only with a valid JSON object, without any other text.".to_string()
        }
        v @ Value::Object(_) => format!(
            "Respond only with a valid JSON object that matches this JSON schema, without any other text: {v}"
        ),
        _ => return None,
    };
    Some(json!({"role": "system", "content": text}))
}

/// Ollama messages into chat messages: `thinking` is the reasoning, tool
/// calls have object arguments, tool results name the tool.
fn chat_messages(body: &Value) -> Result<Vec<Value>, String> {
    let list = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or("messages must be an array")?;
    let mut out: Vec<Value> = format_message(body).into_iter().collect();
    for m in list {
        if m.get("images")
            .and_then(|i| i.as_array())
            .is_some_and(|i| !i.is_empty())
        {
            return Err("images are not supported".into());
        }
        let mut m = m.clone();
        if let Some(t) = m.as_object_mut().and_then(|o| o.remove("thinking")) {
            m["reasoning_content"] = t;
        }
        if let Some(Value::Array(calls)) = m.get_mut("tool_calls") {
            for (i, c) in calls.iter_mut().enumerate() {
                let c = c
                    .as_object_mut()
                    .ok_or("each tool call must be an object")?;
                c.entry("type").or_insert(json!("function"));
                c.entry("id").or_insert(json!(format!("call_{i}")));
            }
        }
        out.push(m);
    }
    normalize_messages(&Value::Array(out))
}

fn options(app: &App, body: &Value, prompt: Vec<u32>) -> (Gen, Vec<String>) {
    let o = body.get("options").cloned().unwrap_or(json!({}));
    let num = |k: &str| o.get(k).and_then(|v| v.as_f64());
    let job = Gen {
        prompt,
        // -1 and -2 mean "no limit" in Ollama.
        max_tokens: num("num_predict")
            .filter(|&n| n > 0.0)
            .map_or(app.cfg.default_max_tokens, |n| n as usize),
        temperature: num("temperature").unwrap_or(1.0) as f32,
        top_p: num("top_p").unwrap_or(1.0) as f32,
        seed: num("seed").map_or_else(random_seed, |v| v as u64),
    };
    (job, stop_list(o.get("stop")))
}

struct OllamaRequest {
    model: String,
    job: Gen,
    stop: Vec<String>,
    stream: bool,
    /// `/api/generate` with `raw`: plain text out, no template.
    raw: bool,
}

fn model_name(app: &App, body: &Value) -> String {
    body.get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&app.cfg.model_name)
        .to_string()
}

fn parse_chat(app: &App, body: &Value) -> Result<Option<OllamaRequest>, String> {
    let messages = chat_messages(body)?;
    if messages.iter().all(|m| m["role"] == "system")
        && body
            .get("messages")
            .and_then(|m| m.as_array())
            .is_none_or(|m| m.is_empty())
    {
        return Ok(None);
    }
    let tools = body
        .get("tools")
        .filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
    let prompt = chat_prompt(
        app,
        &Value::Array(messages),
        tools,
        effort_or_default(app, think_effort(body)).as_ref(),
    )?;
    let (job, stop) = options(app, body, prompt);
    Ok(Some(OllamaRequest {
        model: model_name(app, body),
        job,
        stop,
        stream: body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true),
        raw: false,
    }))
}

fn parse_generate(app: &App, body: &Value) -> Result<Option<OllamaRequest>, String> {
    if body
        .get("images")
        .and_then(|i| i.as_array())
        .is_some_and(|i| !i.is_empty())
    {
        return Err("images are not supported".into());
    }
    if body
        .get("suffix")
        .and_then(|s| s.as_str())
        .is_some_and(|s| !s.is_empty())
    {
        return Err("suffix is not supported".into());
    }
    let text = body.get("prompt").and_then(|p| p.as_str()).unwrap_or("");
    if text.is_empty() {
        return Ok(None);
    }
    let raw = body.get("raw").and_then(|v| v.as_bool()).unwrap_or(false);
    let prompt = if raw {
        raw_prompt(app, text)?
    } else {
        let mut messages: Vec<Value> = format_message(body).into_iter().collect();
        if let Some(s) = body.get("system").and_then(|s| s.as_str()) {
            messages.push(json!({"role": "system", "content": s}));
        }
        messages.push(json!({"role": "user", "content": text}));
        chat_prompt(
            app,
            &Value::Array(messages),
            None,
            effort_or_default(app, think_effort(body)).as_ref(),
        )?
    };
    let (job, stop) = options(app, body, prompt);
    Ok(Some(OllamaRequest {
        model: model_name(app, body),
        job,
        stop,
        stream: body.get("stream").and_then(|v| v.as_bool()).unwrap_or(true),
        raw,
    }))
}

fn done_fields(d: &Done, c: &Collector, began: Instant) -> Value {
    let ns = |s: f64| (s * 1e9) as u64;
    let reason = if c.stopped || d.finish != Finish::Length {
        "stop"
    } else {
        "length"
    };
    json!({
        "done": true,
        "done_reason": reason,
        "total_duration": began.elapsed().as_nanos() as u64,
        "load_duration": 0,
        "prompt_eval_count": d.prompt_tokens - d.cached_tokens,
        "prompt_eval_duration": ns(d.prefill_seconds),
        "eval_count": d.completion_tokens,
        "eval_duration": ns(d.decode_seconds),
    })
}

fn merge(mut a: Value, b: Value) -> Value {
    for (k, v) in b.as_object().into_iter().flatten() {
        a[k] = v.clone();
    }
    a
}

fn tool_calls_json(calls: &[(String, String)]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|(name, args)| {
                let args: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                json!({"function": {"name": name, "arguments": args}})
            })
            .collect(),
    )
}

/// One chunk of output: a message (chat) or a response (generate).
fn piece(chat: bool, content: &str, thinking: &str, calls: &[(String, String)]) -> Value {
    if chat {
        let mut m = json!({"role": "assistant", "content": content});
        if !thinking.is_empty() {
            m["thinking"] = json!(thinking);
        }
        if !calls.is_empty() {
            m["tool_calls"] = tool_calls_json(calls);
        }
        json!({"message": m})
    } else {
        let mut r = json!({"response": content});
        if !thinking.is_empty() {
            r["thinking"] = json!(thinking);
        }
        r
    }
}

async fn run(app: Arc<App>, headers: HeaderMap, body: Bytes, chat: bool) -> Response {
    let began = Instant::now();
    let ready = async {
        precheck(&app, &headers)?;
        let body = json_body(&body)?;
        let req = {
            let app = app.clone();
            blocking(move || {
                if chat {
                    parse_chat(&app, &body)
                } else {
                    parse_generate(&app, &body)
                }
            })
            .await?
        };
        Ok::<_, Refused>(req)
    };
    let req = match ready.await {
        Ok(Some(r)) => r,
        // Nothing to generate: Ollama answers that the model is loaded.
        Ok(None) => {
            let p = piece(chat, "", "", &[]);
            let head = json!({"model": app.cfg.model_name, "created_at": rfc3339(SystemTime::now()), "done": true, "done_reason": "load"});
            return Json(merge(head, p)).into_response();
        }
        Err(e) => return error_response(&e),
    };
    let mut running = match start(&app, &req.job).await {
        Ok(r) => r,
        Err(e) => return error_response(&e),
    };
    let mut c = if req.raw {
        Collector::raw(req.stop.clone())
    } else {
        Collector::new(app.specials, req.stop.clone())
    };
    let head = |model: &str| json!({"model": model, "created_at": rfc3339(SystemTime::now())});
    if !req.stream {
        return match drive(&app, &mut running, &mut c, |_| {}).await {
            Ok(d) => {
                let out = merge(
                    head(&req.model),
                    piece(chat, &c.content, &c.reasoning, &c.tool_calls),
                );
                Json(merge(out, done_fields(&d, &c, began))).into_response()
            }
            Err(e) => error_response(&Refused::Internal(e)),
        };
    }
    let (tx, rx) = unbounded_channel::<Value>();
    tokio::spawn(async move {
        let cancel = running.cancel.clone();
        let send = |v: Value| {
            if tx.send(v).is_err() {
                cancel.store(true, Ordering::Relaxed);
            }
        };
        let result = drive(&app, &mut running, &mut c, |item| {
            let p = match item {
                Item::Delta(Delta::Content(s)) => piece(chat, &s, "", &[]),
                Item::Delta(Delta::Reasoning(s)) => piece(chat, "", &s, &[]),
                Item::Delta(Delta::ToolCall { name, arguments }) => {
                    piece(chat, "", "", &[(name, arguments)])
                }
                // NDJSON has no comments to write during the prefill (Ollama
                // sends nothing either); stop if the stream is already gone.
                Item::Prefill { .. } => {
                    if tx.is_closed() {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    return;
                }
            };
            send(merge(merge(head(&req.model), p), json!({"done": false})));
        })
        .await;
        match result {
            Ok(d) => send(merge(
                merge(head(&req.model), piece(chat, "", "", &[])),
                done_fields(&d, &c, began),
            )),
            Err(e) => send(json!({"error": e})),
        }
    });
    let lines = UnboundedReceiverStream::new(rx)
        .map(|v| Ok::<_, std::convert::Infallible>(format!("{v}\n")));
    Response::builder()
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from_stream(lines))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

pub(crate) async fn chat(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    run(app, headers, body, true).await
}

pub(crate) async fn generate(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    run(app, headers, body, false).await
}

/// Bytes of the model's weight files.
fn model_size(app: &App) -> u64 {
    std::fs::read_dir(&app.cfg.model_dir)
        .map(|d| {
            d.flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
                .filter_map(|e| std::fs::metadata(e.path()).ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

fn details() -> Value {
    json!({"parent_model": "", "format": "safetensors", "family": "inkling2", "families": ["inkling2"],
           "parameter_size": "", "quantization_level": ""})
}

fn digest(app: &App) -> String {
    format!("{:0>64}", crate::worker::model_id(&app.cfg.model_dir))
}

pub(crate) async fn tags(State(app): State<Arc<App>>) -> Response {
    let modified = rfc3339(app.started);
    Json(json!({"models": [{
        "name": app.cfg.model_name, "model": app.cfg.model_name, "modified_at": modified,
        "size": model_size(&app), "digest": digest(&app), "details": details(),
    }]}))
    .into_response()
}

pub(crate) async fn show(State(app): State<Arc<App>>) -> Response {
    Json(json!({
        "modelfile": "", "parameters": "", "template": "",
        "details": details(),
        "model_info": {"general.architecture": "inkling2", "inkling2.context_length": 1_048_576},
        "capabilities": ["completion", "tools", "thinking"],
        "modified_at": rfc3339(app.started),
    }))
    .into_response()
}

pub(crate) async fn ps(State(app): State<Arc<App>>) -> Response {
    let size = model_size(&app);
    Json(json!({"models": [{
        "name": app.cfg.model_name, "model": app.cfg.model_name, "size": size, "digest": digest(&app),
        "details": details(), "expires_at": "2318-01-01T00:00:00Z", "size_vram": size,
        "context_length": 1_048_576,
    }]}))
    .into_response()
}

pub(crate) async fn version() -> Response {
    Json(json!({"version": OLLAMA_VERSION})).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_formats_utc() {
        let t = UNIX_EPOCH + std::time::Duration::new(1_700_000_000, 5);
        assert_eq!(rfc3339(t), "2023-11-14T22:13:20.000000005Z");
        assert_eq!(rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00.000000000Z");
    }

    #[test]
    fn chat_messages_map_thinking_and_calls() {
        let body = json!({"format": "json", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "thinking": "t", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
            {"role": "tool", "content": "ok", "tool_name": "f"},
        ]});
        let m = chat_messages(&body).unwrap();
        assert_eq!(m[0]["role"], "system");
        assert_eq!(m[2]["reasoning_content"], "t");
        assert_eq!(
            m[2]["tool_calls"][0]["function"]["arguments"],
            json!({"a": 1})
        );
        assert_eq!(think_effort(&json!({"think": false})), Some(json!("none")));
        assert_eq!(think_effort(&json!({"think": "high"})), Some(json!("high")));
        assert_eq!(think_effort(&json!({"think": true})), None);
    }
}
