//! OpenAI's legacy text completions (`/v1/completions`) and the Responses
//! API (`/v1/responses`), on the same model slot as chat completions.

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
use tokio::sync::mpsc::unbounded_channel;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{
    parser::Delta,
    server::{
        App, Asked, Collector, Gen, Item, Refused, blocking, chat_prompt, completion_id, drive,
        effort_or_default, format_instruction, json_body, normalize_messages, now, precheck,
        random_seed, raw_prompt, start, stop_list,
    },
    worker::{Done, Finish},
};

fn sse(rx: tokio::sync::mpsc::UnboundedReceiver<Result<SseEvent, Infallible>>) -> Response {
    Sse::new(UnboundedReceiverStream::new(rx))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

fn num(body: &Value, k: &str) -> Option<f64> {
    body.get(k).and_then(|v| v.as_f64())
}

// ---- /v1/completions ------------------------------------------------------

struct Completion {
    model: String,
    prompt_text: Option<String>,
    job: Gen,
    stop: Vec<String>,
    stream: bool,
    include_usage: bool,
    echo: bool,
}

fn parse_completion(app: &App, body: &Value) -> Result<Completion, String> {
    if body.get("logprobs").is_some_and(|v| !v.is_null()) {
        return Err("logprobs are not supported".into());
    }
    if num(body, "n").is_some_and(|n| n > 1.0) || num(body, "best_of").is_some_and(|n| n > 1.0) {
        return Err("only one completion per request is supported".into());
    }
    // A string, one string in an array, or token ids.
    let (prompt_text, prompt) = match body.get("prompt") {
        Some(Value::String(s)) => (Some(s.clone()), raw_prompt(app, s)?),
        Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => {
            let s = a[0].as_str().unwrap_or_default().to_string();
            let ids = raw_prompt(app, &s)?;
            (Some(s), ids)
        }
        Some(Value::Array(a)) if !a.is_empty() && a.iter().all(|v| v.is_u64()) => {
            // An id outside the vocabulary would index past the embedding.
            let vocab = app.tokenizer.get_vocab_size(true) as u64;
            let ids = a.iter().filter_map(|v| v.as_u64()).filter(|&t| t < vocab);
            let ids: Vec<u32> = ids.map(|t| t as u32).collect();
            if ids.len() != a.len() {
                return Err(format!("token ids must be below {vocab}"));
            }
            (None, ids)
        }
        _ => return Err("prompt must be a string, one string in an array, or token ids".into()),
    };
    if prompt.is_empty() {
        return Err("the prompt is empty".into());
    }
    Ok(Completion {
        model: body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&app.cfg.model_name)
            .to_string(),
        prompt_text,
        job: Gen {
            prompt,
            // The server's default, like every other API (OpenAI's is 16).
            max_tokens: num(body, "max_tokens")
                .map_or(app.cfg.default_max_tokens, |v| v as usize)
                .max(1),
            temperature: num(body, "temperature").unwrap_or(1.0) as f32,
            top_p: num(body, "top_p").unwrap_or(1.0) as f32,
            seed: num(body, "seed").map_or_else(random_seed, |v| v as u64),
            summary: Asked {
                api: "completions",
                model: body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&app.cfg.model_name),
                messages: None,
                tools: 0,
                effort: None,
                effort_note: None,
                max_tokens: num(body, "max_tokens"),
                max_note: None,
                stream: body.get("stream").and_then(|v| v.as_bool()) == Some(true),
            }
            .line(app),
        },
        stop: stop_list(body.get("stop")),
        stream: body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        include_usage: body
            .pointer("/stream_options/include_usage")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        echo: body.get("echo").and_then(|v| v.as_bool()).unwrap_or(false),
    })
}

fn completion_usage(d: &Done) -> Value {
    json!({
        "prompt_tokens": d.prompt_tokens,
        "completion_tokens": d.completion_tokens,
        "total_tokens": d.prompt_tokens + d.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": d.cached_tokens},
    })
}

fn finish_reason(c: &Collector, d: &Done) -> &'static str {
    if !c.stopped && d.finish == Finish::Length {
        "length"
    } else {
        "stop"
    }
}

/// An SSE comment per prefill chunk: writing is how a client that left is
/// noticed before the first token.
fn prefill_comment(done: usize, total: usize) -> SseEvent {
    SseEvent::default().comment(format!("prefill {done}/{total}"))
}

pub(crate) async fn completions(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let ready = async {
        precheck(&app, &headers)?;
        let body = json_body(&body)?;
        let req = {
            let app = app.clone();
            blocking(move || parse_completion(&app, &body)).await?
        };
        let run = start(&app, &req.job).await?;
        Ok::<_, Refused>((req, run))
    };
    let (req, mut run) = match ready.await {
        Ok(v) => v,
        Err(e) => return e.openai(),
    };
    let id = completion_id().replacen("chatcmpl", "cmpl", 1);
    let created = now();
    let mut c = Collector::raw(req.stop.clone());
    let echo = if req.echo {
        req.prompt_text.clone()
    } else {
        None
    };
    if !req.stream {
        return match drive(&app, &mut run, &mut c, |_| {}).await {
            Ok(d) => Json(json!({
                "id": id, "object": "text_completion", "created": created, "model": req.model,
                "choices": [{
                    "text": echo.unwrap_or_default() + &c.content,
                    "index": 0, "logprobs": null, "finish_reason": finish_reason(&c, &d),
                }],
                "usage": completion_usage(&d),
            }))
            .into_response(),
            Err(e) => Refused::Internal(e).openai(),
        };
    }
    let (tx, rx) = unbounded_channel();
    tokio::spawn(async move {
        let cancel = run.cancel.clone();
        let send = |v: Value| {
            if tx
                .send(Ok(SseEvent::default().data(v.to_string())))
                .is_err()
            {
                cancel.store(true, Ordering::Relaxed);
            }
        };
        let chunk = |text: &str, finish: Option<&str>| {
            json!({
                "id": id, "object": "text_completion", "created": created, "model": req.model,
                "choices": [{"text": text, "index": 0, "logprobs": null, "finish_reason": finish}],
            })
        };
        if let Some(e) = &echo {
            send(chunk(e, None));
        }
        let result = drive(&app, &mut run, &mut c, |item| match item {
            Item::Delta(Delta::Content(s)) => send(chunk(&s, None)),
            Item::Prefill { done, total } => {
                if tx.send(Ok(prefill_comment(done, total))).is_err() {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Item::Delta(_) => {}
        })
        .await;
        match result {
            Ok(d) => {
                send(chunk("", Some(finish_reason(&c, &d))));
                if req.include_usage {
                    send(json!({
                        "id": id, "object": "text_completion", "created": created, "model": req.model,
                        "choices": [], "usage": completion_usage(&d),
                    }));
                }
            }
            Err(e) => send(json!({"error": {"message": e, "type": "server_error"}})),
        }
        let _ = tx.send(Ok(SseEvent::default().data("[DONE]")));
    });
    sse(rx)
}

// ---- /v1/responses --------------------------------------------------------

/// Text of a Responses content value: a string or parts with text.
fn content_text(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some(
                        "input_text" | "output_text" | "text" | "summary_text" | "reasoning_text",
                    ) => out.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or("")),
                    // Text-only model: a placeholder keeps the history usable.
                    Some("input_image") => {
                        out.push_str("[image omitted: this model reads text only]")
                    }
                    Some("input_file") => {
                        out.push_str("[file omitted: this model reads text only]")
                    }
                    Some(other) => return Err(format!("content of type {other} is not supported")),
                    None => {}
                }
            }
            Ok(out)
        }
        Value::Null => Ok(String::new()),
        _ => Err("content must be a string or an array of parts".into()),
    }
}

/// Responses input into chat messages for the template.
fn responses_messages(body: &Value) -> Result<Vec<Value>, String> {
    let mut msgs: Vec<Value> = Vec::new();
    if let Some(s) = body.get("instructions").and_then(|v| v.as_str()) {
        msgs.push(json!({"role": "system", "content": s}));
    }
    let mut reasoning: Option<String> = None;
    match body.get("input") {
        Some(Value::String(s)) => msgs.push(json!({"role": "user", "content": s})),
        Some(Value::Array(items)) => {
            for item in items {
                match item.get("type").and_then(|t| t.as_str()) {
                    Some("function_call") => {
                        let call = json!({
                            "id": item.get("call_id").cloned().unwrap_or(Value::Null),
                            "type": "function",
                            "function": {
                                "name": item.get("name").cloned().unwrap_or(json!("")),
                                "arguments": item.get("arguments").cloned().unwrap_or(json!("{}")),
                            },
                        });
                        // Calls belong to the assistant message just before
                        // them (text and calls of one turn come as separate
                        // items).
                        match msgs.last_mut() {
                            Some(m) if m["role"] == "assistant" && reasoning.is_none() => {
                                match m.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
                                    Some(calls) => calls.push(call),
                                    None => m["tool_calls"] = json!([call]),
                                }
                            }
                            _ => {
                                let mut m = json!({"role": "assistant", "content": "", "tool_calls": [call]});
                                if let Some(r) = reasoning.take() {
                                    m["reasoning_content"] = json!(r);
                                }
                                msgs.push(m);
                            }
                        }
                    }
                    Some("function_call_output") => {
                        let output = match item.get("output") {
                            Some(Value::String(s)) => s.clone(),
                            Some(v @ Value::Array(_)) => content_text(v)?,
                            Some(v) => v.to_string(),
                            None => String::new(),
                        };
                        msgs.push(json!({
                            "role": "tool",
                            "tool_call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
                            "content": output,
                        }));
                    }
                    Some("reasoning") => {
                        let text = item
                            .get("content")
                            .map(content_text)
                            .transpose()?
                            .filter(|t| !t.is_empty())
                            .or(item.get("summary").map(content_text).transpose()?)
                            .unwrap_or_default();
                        reasoning = Some(text);
                    }
                    None | Some("message") => {
                        let role = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                        let text = content_text(item.get("content").unwrap_or(&Value::Null))?;
                        let mut m = json!({"role": role, "content": text});
                        if role == "assistant"
                            && let Some(r) = reasoning.take()
                        {
                            m["reasoning_content"] = json!(r);
                        }
                        msgs.push(m);
                    }
                    Some(other) => {
                        return Err(format!("input items of type {other} are not supported"));
                    }
                }
            }
        }
        _ => return Err("input must be a string or an array of items".into()),
    }
    normalize_messages(&Value::Array(msgs))
}

/// Responses tools (function tools only) into chat tools.
fn responses_tools(body: &Value) -> Result<Option<Value>, String> {
    if body.get("tool_choice").and_then(|v| v.as_str()) == Some("none") {
        return Ok(None);
    }
    let Some(Value::Array(tools)) = body.get("tools") else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for t in tools {
        match t.get("type").and_then(|v| v.as_str()) {
            Some("function") => out.push(json!({
                "type": "function",
                "function": {
                    "name": t.get("name").cloned().unwrap_or(json!("")),
                    "description": t.get("description").cloned().unwrap_or(json!("")),
                    "parameters": t.get("parameters").cloned().unwrap_or(json!({"type": "object"})),
                },
            })),
            other => {
                return Err(format!(
                    "tools of type {} are not supported",
                    other.unwrap_or("?")
                ));
            }
        }
    }
    Ok((!out.is_empty()).then_some(Value::Array(out)))
}

struct ResponsesRequest {
    model: String,
    job: Gen,
    stream: bool,
    /// Echoed back in the response object.
    echo: Value,
}

fn parse_responses(app: &App, body: &Value) -> Result<ResponsesRequest, String> {
    if body
        .get("previous_response_id")
        .is_some_and(|v| !v.is_null())
    {
        return Err(
            "previous_response_id is not supported: send the whole conversation as input".into(),
        );
    }
    let mut messages = responses_messages(body)?;
    // `text.format` as chat's `response_format`.
    let format = match body.pointer("/text/format") {
        Some(f) if f["type"] == "json_schema" => Some(json!({
            "response_format": {"type": "json_schema", "json_schema": {"schema": f.get("schema").cloned().unwrap_or(json!({}))}}
        })),
        Some(f) if f["type"] == "json_object" => {
            Some(json!({"response_format": {"type": "json_object"}}))
        }
        _ => None,
    };
    if let Some(instruction) = format.as_ref().and_then(format_instruction) {
        messages.insert(0, json!({"role": "system", "content": instruction}));
    }
    let tools = responses_tools(body)?;
    let effort = body
        .pointer("/reasoning/effort")
        .filter(|v| !v.is_null())
        .cloned();
    let prompt = chat_prompt(
        app,
        &Value::Array(messages),
        tools.as_ref(),
        effort_or_default(app, effort).as_ref(),
    )?;
    Ok(ResponsesRequest {
        model: body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&app.cfg.model_name)
            .to_string(),
        job: Gen {
            prompt,
            max_tokens: num(body, "max_output_tokens")
                .map_or(app.cfg.default_max_tokens, |v| v as usize)
                .max(1),
            temperature: num(body, "temperature").unwrap_or(1.0) as f32,
            top_p: num(body, "top_p").unwrap_or(1.0) as f32,
            seed: random_seed(),
            summary: Asked {
                api: "responses",
                model: body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&app.cfg.model_name),
                messages: Some(match body.get("input") {
                    Some(Value::Array(items)) => items.len(),
                    _ => 1,
                }),
                tools: tools
                    .as_ref()
                    .and_then(|t| t.as_array())
                    .map_or(0, |t| t.len()),
                effort: body.pointer("/reasoning/effort").filter(|v| !v.is_null()),
                effort_note: None,
                max_tokens: num(body, "max_output_tokens"),
                max_note: None,
                stream: body.get("stream").and_then(|v| v.as_bool()) == Some(true),
            }
            .line(app),
        },
        stream: body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        echo: json!({
            "instructions": body.get("instructions").cloned().unwrap_or(Value::Null),
            "max_output_tokens": body.get("max_output_tokens").cloned().unwrap_or(Value::Null),
            "temperature": body.get("temperature").cloned().unwrap_or(json!(1.0)),
            "top_p": body.get("top_p").cloned().unwrap_or(json!(1.0)),
            "tools": body.get("tools").cloned().unwrap_or(json!([])),
            "tool_choice": body.get("tool_choice").cloned().unwrap_or(json!("auto")),
            "metadata": body.get("metadata").cloned().unwrap_or(json!({})),
        }),
    })
}

fn short_id(prefix: &str) -> String {
    format!(
        "{prefix}_{}",
        completion_id().trim_start_matches("chatcmpl-")
    )
}

/// Output items of a finished generation.
fn output_items(c: &Collector) -> Vec<Value> {
    let mut out = Vec::new();
    if !c.reasoning.is_empty() {
        out.push(reasoning_item(&short_id("rs"), &c.reasoning, "completed"));
    }
    if !c.content.is_empty() || c.tool_calls.is_empty() {
        out.push(message_item(&short_id("msg"), &c.content, "completed"));
    }
    for (name, args) in &c.tool_calls {
        out.push(call_item(
            &short_id("fc"),
            &short_id("call"),
            name,
            args,
            "completed",
        ));
    }
    out
}

fn reasoning_item(id: &str, text: &str, status: &str) -> Value {
    // A streamed item starts with its one part, for the deltas to fill.
    let content = if text.is_empty() && status != "in_progress" {
        json!([])
    } else {
        json!([{"type": "reasoning_text", "text": text}])
    };
    json!({"type": "reasoning", "id": id, "summary": [], "content": content})
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    let content = if status == "in_progress" {
        json!([])
    } else {
        json!([{"type": "output_text", "text": text, "annotations": []}])
    };
    json!({"type": "message", "id": id, "status": status, "role": "assistant", "content": content})
}

fn call_item(id: &str, call_id: &str, name: &str, args: &str, status: &str) -> Value {
    json!({"type": "function_call", "id": id, "call_id": call_id, "name": name, "arguments": args, "status": status})
}

fn responses_usage(d: &Done) -> Value {
    json!({
        "input_tokens": d.prompt_tokens,
        "input_tokens_details": {"cached_tokens": d.cached_tokens},
        "output_tokens": d.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": d.prompt_tokens + d.completion_tokens,
    })
}

/// The response object, in progress (no `done`) or finished.
fn response_object(
    id: &str,
    created: u64,
    req: &ResponsesRequest,
    output: Vec<Value>,
    done: Option<(&Collector, &Done)>,
) -> Value {
    let mut r = json!({
        "id": id, "object": "response", "created_at": created, "model": req.model,
        "status": "in_progress", "error": null, "incomplete_details": null,
        "output": output, "parallel_tool_calls": true, "usage": null,
    });
    for (k, v) in req.echo.as_object().into_iter().flatten() {
        r[k] = v.clone();
    }
    if let Some((c, d)) = done {
        let incomplete = !c.stopped && d.finish == Finish::Length && c.tool_calls.is_empty();
        r["status"] = json!(if incomplete {
            "incomplete"
        } else {
            "completed"
        });
        if incomplete {
            r["incomplete_details"] = json!({"reason": "max_output_tokens"});
        }
        r["usage"] = responses_usage(d);
    }
    r
}

pub(crate) async fn responses(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let ready = async {
        precheck(&app, &headers)?;
        let body = json_body(&body)?;
        let req = {
            let app = app.clone();
            blocking(move || parse_responses(&app, &body)).await?
        };
        let run = start(&app, &req.job).await?;
        Ok::<_, Refused>((req, run))
    };
    let (req, mut run) = match ready.await {
        Ok(v) => v,
        Err(e) => return e.openai(),
    };
    let id = short_id("resp");
    let created = now();
    let mut c = Collector::new(app.specials, Vec::new());
    if !req.stream {
        return match drive(&app, &mut run, &mut c, |_| {}).await {
            Ok(d) => Json(response_object(
                &id,
                created,
                &req,
                output_items(&c),
                Some((&c, &d)),
            ))
            .into_response(),
            Err(e) => Refused::Internal(e).openai(),
        };
    }
    let (tx, rx) = unbounded_channel();
    tokio::spawn(async move {
        let mut s = ResponsesStream {
            tx,
            cancel: run.cancel.clone(),
            seq: 0,
            output: Vec::new(),
            open: Open::None,
        };
        s.event(
            "response.created",
            json!({"response": response_object(&id, created, &req, Vec::new(), None)}),
        );
        s.event(
            "response.in_progress",
            json!({"response": response_object(&id, created, &req, Vec::new(), None)}),
        );
        let result = drive(&app, &mut run, &mut c, |item| match item {
            Item::Delta(d) => s.delta(d),
            Item::Prefill { done, total } => s.comment(prefill_comment(done, total)),
        })
        .await;
        if s.output.is_empty() && matches!(s.open, Open::None) {
            // Nothing generated: an empty message, as without streaming.
            s.delta(Delta::Content(String::new()));
        }
        s.close();
        match result {
            Ok(d) => {
                let kind = if !c.stopped && d.finish == Finish::Length && c.tool_calls.is_empty() {
                    "response.incomplete"
                } else {
                    "response.completed"
                };
                let output = std::mem::take(&mut s.output);
                s.event(
                    kind,
                    json!({"response": response_object(&id, created, &req, output, Some((&c, &d)))}),
                );
            }
            Err(e) => {
                let output = std::mem::take(&mut s.output);
                let mut r = response_object(&id, created, &req, output, None);
                r["status"] = json!("failed");
                r["error"] = json!({"code": "server_error", "message": e});
                s.event("response.failed", json!({"response": r}));
            }
        }
    });
    sse(rx)
}

/// The output item being streamed.
enum Open {
    None,
    Reasoning {
        index: usize,
        id: String,
        text: String,
    },
    Message {
        index: usize,
        id: String,
        text: String,
    },
}

/// The Responses streaming events: items open and close as the model moves
/// between reasoning, text and tool calls.
struct ResponsesStream {
    tx: tokio::sync::mpsc::UnboundedSender<Result<SseEvent, Infallible>>,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    seq: usize,
    /// Finished items, for the final response object.
    output: Vec<Value>,
    open: Open,
}

impl ResponsesStream {
    fn event(&mut self, kind: &str, mut data: Value) {
        data["type"] = json!(kind);
        data["sequence_number"] = json!(self.seq);
        self.seq += 1;
        let ev = SseEvent::default().event(kind).data(data.to_string());
        if self.tx.send(Ok(ev)).is_err() {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn comment(&mut self, ev: SseEvent) {
        if self.tx.send(Ok(ev)).is_err() {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn delta(&mut self, d: Delta) {
        match d {
            Delta::Reasoning(s) => {
                if !matches!(self.open, Open::Reasoning { .. }) {
                    self.close();
                    let (index, id) = (self.output.len(), short_id("rs"));
                    self.event(
                        "response.output_item.added",
                        json!({"output_index": index, "item": reasoning_item(&id, "", "in_progress")}),
                    );
                    self.open = Open::Reasoning {
                        index,
                        id,
                        text: String::new(),
                    };
                }
                if let Open::Reasoning { index, id, text } = &mut self.open {
                    text.push_str(&s);
                    let data = json!({"item_id": id, "output_index": index, "content_index": 0, "delta": s});
                    self.event("response.reasoning_text.delta", data);
                }
            }
            Delta::Content(s) => {
                if !matches!(self.open, Open::Message { .. }) {
                    self.close();
                    let (index, id) = (self.output.len(), short_id("msg"));
                    self.event(
                        "response.output_item.added",
                        json!({"output_index": index, "item": message_item(&id, "", "in_progress")}),
                    );
                    self.event(
                        "response.content_part.added",
                        json!({"item_id": id, "output_index": index, "content_index": 0,
                               "part": {"type": "output_text", "text": "", "annotations": []}}),
                    );
                    self.open = Open::Message {
                        index,
                        id,
                        text: String::new(),
                    };
                }
                if let Open::Message { index, id, text } = &mut self.open {
                    text.push_str(&s);
                    let data = json!({"item_id": id, "output_index": index, "content_index": 0, "delta": s});
                    self.event("response.output_text.delta", data);
                }
            }
            Delta::ToolCall { name, arguments } => {
                self.close();
                let (index, id, call_id) = (self.output.len(), short_id("fc"), short_id("call"));
                self.event(
                    "response.output_item.added",
                    json!({"output_index": index, "item": call_item(&id, &call_id, &name, "", "in_progress")}),
                );
                self.event(
                    "response.function_call_arguments.delta",
                    json!({"item_id": id, "output_index": index, "delta": arguments}),
                );
                self.event(
                    "response.function_call_arguments.done",
                    json!({"item_id": id, "output_index": index, "arguments": arguments}),
                );
                let item = call_item(&id, &call_id, &name, &arguments, "completed");
                self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                );
                self.output.push(item);
            }
        }
    }

    /// Finishes the open item, if any.
    fn close(&mut self) {
        match std::mem::replace(&mut self.open, Open::None) {
            Open::None => {}
            Open::Reasoning { index, id, text } => {
                self.event(
                    "response.reasoning_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                );
                let item = reasoning_item(&id, &text, "completed");
                self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                );
                self.output.push(item);
            }
            Open::Message { index, id, text } => {
                self.event(
                    "response.output_text.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0, "text": text}),
                );
                self.event(
                    "response.content_part.done",
                    json!({"item_id": id, "output_index": index, "content_index": 0,
                           "part": {"type": "output_text", "text": text, "annotations": []}}),
                );
                let item = message_item(&id, &text, "completed");
                self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                );
                self.output.push(item);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_input_maps_to_chat_messages() {
        let body = json!({
            "instructions": "Be brief.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Weather in Paris?"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Need the tool."}]},
                {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{\"city\":\"Paris\"}"},
                {"type": "function_call", "call_id": "c2", "name": "time", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "18 C"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "18 C."}]},
            ],
        });
        let m = responses_messages(&body).unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "Be brief."}));
        assert_eq!(m[1]["content"], "Weather in Paris?");
        assert_eq!(m[2]["role"], "assistant");
        assert_eq!(m[2]["reasoning_content"], "Need the tool.");
        assert_eq!(m[2]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(
            m[2]["tool_calls"][0]["function"]["arguments"],
            json!({"city": "Paris"})
        );
        assert_eq!(
            m[3],
            json!({"role": "tool", "tool_call_id": "c1", "content": "18 C"})
        );
        assert_eq!(m[4]["content"], "18 C.");
        assert!(responses_messages(&json!({"input": [{"type": "web_search_call"}]})).is_err());
        // A returned turn (reasoning, text, call) stays one assistant message.
        let round_trip = json!({"input": [
            {"role": "user", "content": "Weather?"},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Call it."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
            {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": "{}"},
        ]});
        let m = responses_messages(&round_trip).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[1]["reasoning_content"], "Call it.");
        assert_eq!(m[1]["content"], "Checking.");
        assert_eq!(m[1]["tool_calls"][0]["id"], "c1");
    }

    #[test]
    fn responses_tools_take_function_tools() {
        let t = responses_tools(&json!({"tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}]}))
            .unwrap()
            .unwrap();
        assert_eq!(t[0]["function"]["name"], "f");
        assert!(responses_tools(&json!({"tools": [{"type": "file_search"}]})).is_err());
        assert!(
            responses_tools(
                &json!({"tool_choice": "none", "tools": [{"type": "function", "name": "f"}]})
            )
            .unwrap()
            .is_none()
        );
    }
}
