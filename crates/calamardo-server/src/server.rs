//! HTTP routes: OpenAI Chat Completions (streaming and not), models, health
//! and metrics.

use std::{
    convert::Infallible,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, HeaderValue, StatusCode, header, request::Parts},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
    routing::{get, post},
};
use calamardo_model::cache::KvFormat;
use serde_json::{Value, json};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::{
    metrics::Metrics,
    parser::{Delta, ResponseParser, Specials},
    template::ChatTemplate,
    worker::{self, Event, Finish, Job, Queue, WorkerConfig},
};

/// Largest request body (long agent contexts, images in content parts).
const MAX_BODY: usize = 256 << 20;

pub struct ServerConfig {
    pub addr: SocketAddr,
    pub model_dir: PathBuf,
    pub kv: KvFormat,
    pub prefill_step: usize,
    /// Name reported by /v1/models. Requests may name any model.
    pub model_name: String,
    /// Requests that may wait for the slot; more get 429 at once.
    pub queue: usize,
    /// Longest wait for the slot before 429.
    pub max_queue_wait: Duration,
    /// Bearer token required on /v1 routes, if set.
    pub api_key: Option<String>,
    /// max_tokens when the request gives none.
    pub default_max_tokens: usize,
    /// reasoning_effort when the request gives none; `None` keeps the
    /// template's default (0.9).
    pub default_reasoning_effort: Option<String>,
    /// Bytes the prefix cache may keep in RAM.
    pub cache_budget: usize,
    /// Reuse prefixes only at chunk boundaries, for bit-exact output.
    pub aligned_prefix: bool,
    /// Prefix cache on disk: directory and byte budget (None disables it).
    pub disk_cache: Option<(PathBuf, u64)>,
}

struct App {
    cfg: ServerConfig,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    specials: Specials,
    queue: Arc<Queue>,
    metrics: Arc<Metrics>,
    worker: std::thread::JoinHandle<()>,
}

/// Browser clients on this machine only, as Ollama allows by default.
fn local_origin(origin: &HeaderValue, _: &Parts) -> bool {
    let Ok(o) = origin.to_str() else {
        return false;
    };
    let host = o
        .strip_prefix("http://")
        .or_else(|| o.strip_prefix("https://"))
        .unwrap_or("");
    let host = host.split(':').next().unwrap_or("");
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

pub async fn serve(cfg: ServerConfig) -> Result<()> {
    // Bind first: a busy port fails now, not after loading the model.
    let listener = tokio::net::TcpListener::bind(cfg.addr)
        .await
        .with_context(|| format!("listening on {}", cfg.addr))?;
    let tokenizer = Tokenizer::from_file(cfg.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
    let template = ChatTemplate::load(&cfg.model_dir.join("chat_template.jinja"))?;
    let specials = Specials::from_tokenizer(&tokenizer)?;
    let config: Value = serde_json::from_slice(&std::fs::read(cfg.model_dir.join("config.json"))?)?;
    let eos = config["eos_token_id"]
        .as_u64()
        .context("config.json has no eos_token_id")? as u32;
    let mut stop_tokens = vec![eos];
    for t in [
        "<|endoftext|>",
        "<|message_user|>",
        "<|message_system|>",
        "<|message_tool|>",
    ] {
        stop_tokens.extend(tokenizer.token_to_id(t));
    }

    let metrics = Arc::new(Metrics::default());
    let queue = Arc::new(Queue::new(cfg.queue));
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let worker = worker::spawn(
        WorkerConfig {
            model_dir: cfg.model_dir.clone(),
            kv: cfg.kv,
            prefill_step: cfg.prefill_step,
            stop_tokens,
            cache_budget: cfg.cache_budget,
            aligned_prefix: cfg.aligned_prefix,
            disk: cfg.disk_cache.clone(),
            model_id: cfg
                .model_dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
        },
        queue.clone(),
        metrics.clone(),
        ready_tx,
    );
    ready_rx.await.context("model thread died")??;

    let addr = cfg.addr;
    let app = Arc::new(App {
        cfg,
        tokenizer,
        template,
        specials,
        queue,
        metrics,
        worker,
    });
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(local_origin))
        .allow_methods(Any)
        .allow_headers(Any);
    let router = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_route))
        .route("/v1/models", get(models))
        .route("/v1/models/{id}", get(model))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .layer(cors)
        .with_state(app.clone());
    let (queue_for_shutdown, app_for_shutdown) = (app.queue.clone(), app);
    eprintln!("listening on http://{addr}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    // Let the model thread finish and save its conversations to disk.
    eprintln!("shutting down: saving conversations");
    queue_for_shutdown.close();
    while !app_for_shutdown.worker.is_finished() {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}

fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    let body = json!({"error": {"message": message.into(), "type": kind}});
    (status, Json(body)).into_response()
}

fn authorized(app: &App, headers: &HeaderMap) -> bool {
    let Some(key) = &app.cfg.api_key else {
        return true;
    };
    let Some(v) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    // The scheme is case-insensitive (RFC 7235); compare in constant time.
    let given = match v.split_once(' ') {
        Some((scheme, token)) if scheme.eq_ignore_ascii_case("bearer") => token.trim().as_bytes(),
        _ => return false,
    };
    let key = key.as_bytes();
    given.len() == key.len() && given.iter().zip(key).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

async fn health(State(app): State<Arc<App>>) -> Response {
    if app.worker.is_finished() {
        error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "the model thread is not running",
        )
    } else {
        Json(json!({"status": "ok"})).into_response()
    }
}

async fn metrics_route(State(app): State<Arc<App>>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        app.metrics.render(),
    )
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

async fn models(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid API key",
        );
    }
    Json(json!({"object": "list", "data": [model_json(&app)]})).into_response()
}

fn model_json(app: &App) -> Value {
    json!({"id": app.cfg.model_name, "object": "model", "created": now(), "owned_by": "calamardo"})
}

/// `GET /v1/models/{id}`: every name maps to the one model.
async fn model(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid API key",
        );
    }
    let mut m = model_json(&app);
    m["id"] = json!(id);
    Json(m).into_response()
}

/// A parsed request, ready for the model.
struct Request {
    model: String,
    prompt: Vec<u32>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
    stop: Vec<String>,
    stream: bool,
    include_usage: bool,
}

/// OpenAI messages into what the template expects: `developer` as `system`,
/// tool call arguments as objects, and `reasoning` (Ollama, newer vLLM) as
/// `reasoning_content`.
fn normalize_messages(messages: &Value) -> Result<Vec<Value>, String> {
    let list = messages.as_array().ok_or("messages must be an array")?;
    let mut out = Vec::with_capacity(list.len());
    for m in list {
        let mut m = m.clone();
        let obj = m.as_object_mut().ok_or("each message must be an object")?;
        if obj.get("role").and_then(|r| r.as_str()) == Some("developer") {
            obj.insert("role".into(), json!("system"));
        }
        if !obj.contains_key("reasoning_content")
            && let Some(r) = obj.remove("reasoning")
        {
            obj.insert("reasoning_content".into(), r);
        }
        if let Some(Value::Array(calls)) = obj.get_mut("tool_calls") {
            for c in calls {
                if let Some(args) = c.pointer_mut("/function/arguments")
                    && let Value::String(s) = args
                {
                    // Arguments that are not a JSON object (a broken call the
                    // client kept) become empty rather than failing every
                    // later turn of the conversation.
                    *args = match serde_json::from_str::<Value>(s) {
                        Ok(v @ Value::Object(_)) => v,
                        _ => json!({}),
                    };
                }
            }
        }
        out.push(m);
    }
    Ok(out)
}

/// `response_format` as an instruction: the output is not constrained, so a
/// client should still validate it.
fn format_instruction(body: &Value) -> Option<String> {
    match body
        .pointer("/response_format/type")
        .and_then(|t| t.as_str())?
    {
        "json_object" => {
            Some("Respond only with a valid JSON object, without any other text.".into())
        }
        "json_schema" => {
            let schema = body
                .pointer("/response_format/json_schema/schema")
                .map(|s| s.to_string())
                .unwrap_or_default();
            Some(format!(
                "Respond only with a valid JSON object that matches this JSON schema, without any other text: {schema}"
            ))
        }
        _ => None,
    }
}

fn parse_request(app: &App, body: &Value) -> Result<Request, String> {
    if body.get("logprobs").and_then(|v| v.as_bool()) == Some(true) {
        return Err("logprobs are not supported".into());
    }
    let mut messages = normalize_messages(body.get("messages").ok_or("missing messages")?)?;
    if let Some(instruction) = format_instruction(body) {
        messages.insert(0, json!({"role": "system", "content": instruction}));
    }
    let messages = Value::Array(messages);
    let tool_choice_none = body.get("tool_choice").and_then(|v| v.as_str()) == Some("none");
    let tools = body.get("tools").filter(|t| {
        !t.is_null() && !tool_choice_none && t.as_array().is_none_or(|a| !a.is_empty())
    });
    let effort = body
        .get("reasoning_effort")
        .or_else(|| body.pointer("/reasoning/effort"))
        .filter(|v| !v.is_null())
        .cloned()
        .or_else(|| app.cfg.default_reasoning_effort.clone().map(Value::String));
    let text = app
        .template
        .render(&messages, tools, effort.as_ref())
        .map_err(|e| e.to_string())?;
    let prompt = app
        .tokenizer
        .encode(text, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    let num = |k: &str| body.get(k).and_then(|v| v.as_f64());
    let max_tokens = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(|v| v.as_f64())
        .map_or(app.cfg.default_max_tokens, |v| v as usize);
    let stop = match body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|s| s.as_str().map(String::from))
            .collect(),
        _ => Vec::new(),
    }
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect();
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seed = num("seed").map(|v| v as u64).unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
            ^ COUNTER.fetch_add(1, Ordering::Relaxed)
    });
    Ok(Request {
        model: body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&app.cfg.model_name)
            .to_string(),
        prompt,
        max_tokens: max_tokens.max(1),
        temperature: num("temperature").unwrap_or(1.0) as f32,
        top_p: num("top_p").unwrap_or(1.0) as f32,
        seed,
        stop,
        stream: body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        include_usage: body
            .pointer("/stream_options/include_usage")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    })
}

/// Turns tokens into OpenAI deltas, applying stop sequences to the content.
struct Collector {
    parser: ResponseParser,
    stops: Vec<String>,
    content: String,
    reasoning: String,
    tool_calls: Vec<(String, String)>,
    stopped: bool,
}

impl Collector {
    fn new(specials: Specials, stops: Vec<String>) -> Self {
        Self {
            parser: ResponseParser::new(specials),
            stops,
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            stopped: false,
        }
    }

    fn accept(&mut self, deltas: Vec<Delta>) -> Vec<Delta> {
        let mut out = Vec::new();
        for d in deltas {
            if self.stopped {
                break;
            }
            match d {
                Delta::Content(s) => {
                    let start = self.content.len();
                    self.content.push_str(&s);
                    let hit = self
                        .stops
                        .iter()
                        .filter_map(|stop| {
                            // A stop may begin in earlier text; search from a
                            // char boundary before the new text.
                            let from = self
                                .content
                                .floor_char_boundary(start.saturating_sub(stop.len()));
                            self.content[from..].find(stop.as_str()).map(|i| i + from)
                        })
                        .min();
                    if let Some(cut) = hit {
                        self.content.truncate(cut);
                        self.stopped = true;
                        if cut > start {
                            out.push(Delta::Content(self.content[start..].to_string()));
                        }
                    } else {
                        out.push(Delta::Content(s));
                    }
                }
                Delta::Reasoning(s) => {
                    self.reasoning.push_str(&s);
                    out.push(Delta::Reasoning(s));
                }
                Delta::ToolCall { name, arguments } => {
                    self.tool_calls.push((name.clone(), arguments.clone()));
                    out.push(Delta::ToolCall { name, arguments });
                }
            }
        }
        out
    }

    fn token(&mut self, t: &Tokenizer, id: u32) -> Vec<Delta> {
        let d = self.parser.push(t, id);
        self.accept(d)
    }

    fn finish(&mut self, t: &Tokenizer) -> Vec<Delta> {
        let d = self.parser.finish(t);
        self.accept(d)
    }

    fn finish_reason(&self, finish: Finish) -> &'static str {
        if !self.tool_calls.is_empty() {
            "tool_calls"
        } else if self.stopped || finish == Finish::Stop {
            "stop"
        } else {
            "length"
        }
    }
}

fn completion_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_micros());
    format!("chatcmpl-{t:x}{n:x}")
}

fn tool_call_json(i: usize, id: &str, name: &str, arguments: &str) -> Value {
    json!({"index": i, "id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
}

fn usage(prompt: usize, cached: usize, completion: usize) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    })
}

async fn chat_completions(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid API key",
        );
    }
    if app.worker.is_finished() {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "the model thread is not running",
        );
    }
    // Cheap check before tokenizing a long prompt for nothing.
    if app.queue.is_full() {
        app.metrics.rejected();
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "the model is busy and the queue is full",
        );
    }
    // Any Content-Type, as Ollama; JSON errors in the OpenAI shape.
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("invalid JSON: {e}"),
            );
        }
    };
    // Rendering and tokenizing a long prompt is CPU work: off the async threads.
    let parsed = {
        let app = app.clone();
        tokio::task::spawn_blocking(move || parse_request(&app, &body)).await
    };
    let req = match parsed {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        Err(e) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                e.to_string(),
            );
        }
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let (etx, mut erx) = unbounded_channel();
    let job = Job {
        prompt: req.prompt.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        seed: req.seed,
        cancel: cancel.clone(),
        events: etx,
    };
    if app.queue.push(job).is_err() {
        app.metrics.rejected();
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "the model is busy and the queue is full",
        );
    }
    app.metrics.set_queue(app.queue.len());
    // Wait for the slot.
    match tokio::time::timeout(app.cfg.max_queue_wait, erx.recv()).await {
        Ok(Some(Event::Started { .. })) => {}
        Ok(Some(Event::Error(e))) => {
            app.metrics.error();
            return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e);
        }
        Ok(_) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "the model thread stopped",
            );
        }
        Err(_) => {
            cancel.store(true, Ordering::Relaxed);
            app.metrics.rejected();
            return error(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "waited too long for the model",
            );
        }
    }

    if req.stream {
        stream_response(app, req, cancel, erx)
    } else {
        full_response(app, req, cancel, erx).await
    }
}

async fn full_response(
    app: Arc<App>,
    req: Request,
    cancel: Arc<AtomicBool>,
    mut erx: UnboundedReceiver<Event>,
) -> Response {
    let id = completion_id();
    let mut c = Collector::new(app.specials, req.stop.clone());
    let (mut finish, mut prompt_tokens, mut cached_tokens, mut completion_tokens) =
        (Finish::Length, req.prompt.len(), 0, 0);
    let mut done = false;
    while let Some(ev) = erx.recv().await {
        match ev {
            Event::Token(id) => {
                c.token(&app.tokenizer, id);
                if c.stopped {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Event::Done {
                finish: f,
                prompt_tokens: p,
                cached_tokens: c,
                completion_tokens: n,
            } => {
                (finish, prompt_tokens, cached_tokens, completion_tokens) = (f, p, c, n);
                done = true;
                break;
            }
            Event::Error(e) => {
                app.metrics.error();
                return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e);
            }
            Event::Started { .. } | Event::Prefill(_) => {}
        }
    }
    if !done {
        app.metrics.error();
        return error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "generation ended unexpectedly",
        );
    }
    c.finish(&app.tokenizer);
    let mut message = json!({"role": "assistant", "content": c.content});
    if !c.reasoning.is_empty() {
        message["reasoning"] = json!(c.reasoning);
    }
    if !c.tool_calls.is_empty() {
        let calls: Vec<Value> = c
            .tool_calls
            .iter()
            .enumerate()
            .map(|(i, (n, a))| {
                let mut v = tool_call_json(i, &format!("call_{id}_{i}"), n, a);
                v.as_object_mut().unwrap().remove("index");
                v
            })
            .collect();
        message["tool_calls"] = json!(calls);
        if c.content.is_empty() {
            message["content"] = Value::Null;
        }
    }
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": req.model,
        "choices": [{"index": 0, "message": message, "finish_reason": c.finish_reason(finish)}],
        "usage": usage(prompt_tokens, cached_tokens, completion_tokens),
    }))
    .into_response()
}

fn stream_response(
    app: Arc<App>,
    req: Request,
    cancel: Arc<AtomicBool>,
    mut erx: UnboundedReceiver<Event>,
) -> Response {
    let (tx, rx) = unbounded_channel();
    tokio::spawn(async move {
        let id = completion_id();
        let created = now();
        let model = req.model.clone();
        let chunk = |delta: Value, finish: Option<&str>| {
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            })
        };
        // A send fails once the client is gone: stop generating.
        let send = |ev: SseEvent| {
            if tx.send(Ok(ev)).is_err() {
                cancel.store(true, Ordering::Relaxed);
            }
        };
        let data = |v: Value| SseEvent::default().data(v.to_string());
        send(data(chunk(
            json!({"role": "assistant", "content": ""}),
            None,
        )));

        let mut c = Collector::new(app.specials, req.stop.clone());
        let mut calls = 0;
        let emit = |deltas: Vec<Delta>, calls: &mut usize| {
            for d in deltas {
                let delta = match d {
                    Delta::Content(s) => json!({"content": s}),
                    Delta::Reasoning(s) => json!({"reasoning": s}),
                    Delta::ToolCall { name, arguments } => {
                        let v = json!({"tool_calls": [tool_call_json(*calls, &format!("call_{id}_{calls}"), &name, &arguments)]});
                        *calls += 1;
                        v
                    }
                };
                send(data(chunk(delta, None)));
            }
        };
        let mut finished = false;
        while let Some(ev) = erx.recv().await {
            match ev {
                Event::Prefill(p) => {
                    // Comments keep the connection alive during long prefills.
                    send(SseEvent::default().comment(format!("prefill {}/{}", p.done, p.total)));
                }
                Event::Token(t) => {
                    let d = c.token(&app.tokenizer, t);
                    emit(d, &mut calls);
                    if c.stopped {
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
                Event::Done {
                    finish,
                    prompt_tokens,
                    cached_tokens,
                    completion_tokens,
                } => {
                    let d = c.finish(&app.tokenizer);
                    emit(d, &mut calls);
                    send(data(chunk(json!({}), Some(c.finish_reason(finish)))));
                    if req.include_usage {
                        send(data(json!({
                            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [], "usage": usage(prompt_tokens, cached_tokens, completion_tokens),
                        })));
                    }
                    finished = true;
                    break;
                }
                Event::Error(e) => {
                    app.metrics.error();
                    send(data(
                        json!({"error": {"message": e, "type": "server_error"}}),
                    ));
                    finished = true;
                    break;
                }
                Event::Started { .. } => {}
            }
        }
        if !finished && !cancel.load(Ordering::Relaxed) {
            app.metrics.error();
            send(data(
                json!({"error": {"message": "generation ended unexpectedly", "type": "server_error"}}),
            ));
        }
        send(SseEvent::default().data("[DONE]"));
    });
    let stream: UnboundedReceiverStream<Result<SseEvent, Infallible>> =
        UnboundedReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}
