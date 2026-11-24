//! The router, OpenAI Chat Completions and Models, health and metrics, and
//! what every API shares: authentication, queueing a job, driving its events
//! and collecting reasoning, content and tool calls.

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
    parser::{Delta, ResponseParser, Specials, TextDecoder},
    template::ChatTemplate,
    worker::{self, Event, Finish, Job, Queue, WorkerConfig},
};

/// Largest request body (long agent contexts, images in content parts).
const MAX_BODY: usize = 256 << 20;

pub struct ServerConfig {
    pub addr: SocketAddr,
    pub model_dir: PathBuf,
    /// Where the model comes from, for the startup report (repository and
    /// revision, or a directory).
    pub model_source: String,
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
    /// Keep the disk cache across restarts instead of deleting it at exit.
    pub keep_disk_cache: bool,
    /// Speculative decoding with the DSpark drafter (M-7).
    pub speculative: bool,
    /// Largest draft block (anchor plus drafts).
    pub draft_block: usize,
}

pub(crate) struct App {
    pub(crate) cfg: ServerConfig,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) template: ChatTemplate,
    pub(crate) specials: Specials,
    pub(crate) queue: Arc<Queue>,
    pub(crate) metrics: Arc<Metrics>,
    pub(crate) worker: std::thread::JoinHandle<()>,
    /// When the server started, for Ollama's `/api/ps`.
    pub(crate) started: SystemTime,
    /// Tokens of context the model supports (prompt plus generation).
    pub(crate) max_context: usize,
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
    crate::report::print_startup(&cfg);
    let tokenizer = Tokenizer::from_file(cfg.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
    let template = ChatTemplate::load(&cfg.model_dir.join("chat_template.jinja"))?;
    let specials = Specials::from_tokenizer(&tokenizer)?;
    let config: Value = serde_json::from_slice(&std::fs::read(cfg.model_dir.join("config.json"))?)?;
    let eos = config["eos_token_id"]
        .as_u64()
        .context("config.json has no eos_token_id")? as u32;
    // The model's context: prompt and generated tokens together.
    let max_context = config
        .pointer("/text_config/model_max_length")
        .and_then(|v| v.as_u64())
        .context("config.json has no text_config.model_max_length")? as usize;
    eprintln!("  context            up to {max_context} tokens (the model's limit)");
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
            keep_disk_cache: cfg.keep_disk_cache,
            model_id: worker::model_id(&cfg.model_dir),
            speculative: cfg.speculative,
            draft_block: cfg.draft_block,
        },
        queue.clone(),
        metrics.clone(),
        ready_tx,
    );
    ready_rx.await.context("model thread died")??;

    let addr = cfg.addr;
    let keep_disk = cfg.keep_disk_cache;
    let app = Arc::new(App {
        cfg,
        tokenizer,
        template,
        specials,
        queue,
        metrics,
        worker,
        started: SystemTime::now(),
        max_context,
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
        .route("/v1/completions", post(crate::openai::completions))
        .route("/v1/responses", post(crate::openai::responses))
        .route("/v1/messages", post(crate::anthropic::messages))
        .route(
            "/v1/messages/count_tokens",
            post(crate::anthropic::count_tokens),
        )
        .route("/api/chat", post(crate::ollama::chat))
        .route("/api/generate", post(crate::ollama::generate))
        .route("/api/tags", get(crate::ollama::tags))
        .route("/api/show", post(crate::ollama::show))
        .route("/api/ps", get(crate::ollama::ps))
        .route("/api/version", get(crate::ollama::version))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .layer(cors)
        .layer(axum::middleware::from_fn(access_log))
        .with_state(app.clone());
    let (queue_for_shutdown, app_for_shutdown) = (app.queue.clone(), app);
    eprintln!("listening on http://{addr}");
    let closing = queue_for_shutdown.clone();
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        shutdown_signal().await;
        // Refuse queued requests and stop the running one at once, so
        // open connections finish quickly; a second signal exits now.
        eprintln!(
            "shutting down: {} (signal again to exit now)",
            if keep_disk {
                "saving conversations"
            } else {
                "deleting the cache"
            }
        );
        closing.close();
        tokio::spawn(async move {
            shutdown_signal().await;
            if !keep_disk {
                crate::disk::remove_owned();
            }
            eprintln!("exiting now");
            std::process::exit(130);
        });
    })
    .await?;
    // Let the model thread finish and save its conversations to disk.
    queue_for_shutdown.close();
    while !app_for_shutdown.worker.is_finished() {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}

/// One line per HTTP request once its response starts: time, client,
/// method, path, status and how long that took (for a stream, until it
/// started). `/health` and `/metrics` are left out: monitors poll them.
async fn access_log(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    // Present when served with connect info, as `serve` does.
    let peer = req
        .extensions()
        .get::<axum::extract::ConnectInfo<SocketAddr>>()
        .map_or_else(|| "-".to_string(), |c| c.0.ip().to_string());
    let (method, path) = (req.method().clone(), req.uri().path().to_owned());
    let began = std::time::Instant::now();
    let res = next.run(req).await;
    if !matches!(path.as_str(), "/health" | "/metrics") {
        eprintln!(
            "{} {} {method} {path} → {} in {:.2?}",
            crate::report::clock(),
            peer,
            res.status().as_u16(),
            began.elapsed()
        );
    }
    res
}

/// Ctrl-C or SIGTERM.
async fn shutdown_signal() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).ok();
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = async {
            match term.as_mut() {
                Some(t) => {
                    t.recv().await;
                }
                None => std::future::pending::<()>().await,
            }
        } => {}
    }
}

pub(crate) fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    let body = json!({"error": {"message": message.into(), "type": kind}});
    (status, Json(body)).into_response()
}

/// The API key, if one is set: `Authorization: Bearer KEY`, or `x-api-key:
/// KEY` as Anthropic clients send it.
pub(crate) fn authorized(app: &App, headers: &HeaderMap) -> bool {
    let Some(key) = &app.cfg.api_key else {
        return true;
    };
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| match v.split_once(' ') {
            // The scheme is case-insensitive (RFC 7235).
            Some((scheme, token)) if scheme.eq_ignore_ascii_case("bearer") => Some(token.trim()),
            _ => None,
        });
    let given = match bearer.or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok())) {
        Some(g) => g.as_bytes(),
        None => return false,
    };
    // Constant time.
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

pub(crate) fn now() -> u64 {
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
    /// For the log (`Asked::line`).
    summary: String,
}

/// OpenAI messages into what the template expects: `developer` as `system`,
/// tool call arguments as objects, and `reasoning` (Ollama, newer vLLM) as
/// `reasoning_content`.
pub(crate) fn normalize_messages(messages: &Value) -> Result<Vec<Value>, String> {
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
pub(crate) fn format_instruction(body: &Value) -> Option<String> {
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

/// Renders the chat template and tokenizes it.
pub(crate) fn chat_prompt(
    app: &App,
    messages: &Value,
    tools: Option<&Value>,
    effort: Option<&Value>,
) -> Result<Vec<u32>, String> {
    let text = app
        .template
        .render(messages, tools, effort)
        .map_err(|e| e.to_string())?;
    raw_prompt(app, &text)
}

/// Tokenizes text as is (special tokens written in it included).
pub(crate) fn raw_prompt(app: &App, text: &str) -> Result<Vec<u32>, String> {
    Ok(app
        .tokenizer
        .encode(text, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec())
}

/// The request's reasoning effort, or the server's default.
pub(crate) fn effort_or_default(app: &App, effort: Option<Value>) -> Option<Value> {
    effort.or_else(|| app.cfg.default_reasoning_effort.clone().map(Value::String))
}

/// Stop strings from a JSON string or array (empty ones dropped).
pub(crate) fn stop_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|s| s.as_str().map(String::from))
            .collect(),
        _ => Vec::new(),
    }
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect()
}

/// A seed for requests that give none.
pub(crate) fn random_seed() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
        ^ COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// What every API asks of the model.
pub(crate) struct Gen {
    pub(crate) prompt: Vec<u32>,
    pub(crate) max_tokens: usize,
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) seed: u64,
    /// What the request asked for, logged when it arrives (`Asked::line`).
    pub(crate) summary: String,
}

/// What a generation request asked for, as the client sent it, for the log.
pub(crate) struct Asked<'a> {
    /// Which API, e.g. "chat completions".
    pub(crate) api: &'a str,
    pub(crate) model: &'a str,
    /// Messages or input items; `None` for a raw prompt (no chat template).
    pub(crate) messages: Option<usize>,
    pub(crate) tools: usize,
    /// The reasoning effort the request sent, if any.
    pub(crate) effort: Option<&'a Value>,
    /// What the request sent when it is not literally an effort (e.g. a
    /// thinking budget), shown instead of "sent".
    pub(crate) effort_note: Option<String>,
    /// The output limit the request sent, if any.
    pub(crate) max_tokens: Option<f64>,
    /// What the request sent when it is not a usable limit (e.g. Ollama's
    /// `num_predict: -1`).
    pub(crate) max_note: Option<String>,
    pub(crate) stream: bool,
}

/// A client string for the log: escaped (no line breaks or terminal codes)
/// and at most 60 characters.
fn loggable(s: &str) -> String {
    let e: String = s.escape_debug().collect();
    if e.chars().count() > 60 {
        format!("{}…", e.chars().take(60).collect::<String>())
    } else {
        e
    }
}

impl Asked<'_> {
    /// One line: model, size, reasoning effort and output limit, and where
    /// the effort and the limit came from (the request or the server).
    pub(crate) fn line(&self, app: &App) -> String {
        let shown = |v: &Value| loggable(&v.as_str().map_or_else(|| v.to_string(), String::from));
        let mut parts = vec![format!("{} · model {}", self.api, loggable(self.model))];
        match self.messages {
            Some(n) => parts.push(format!(
                "{n} message{}{}",
                if n == 1 { "" } else { "s" },
                if self.tools > 0 {
                    format!(
                        " · {} tool{}",
                        self.tools,
                        if self.tools == 1 { "" } else { "s" }
                    )
                } else {
                    String::new()
                }
            )),
            None => parts.push("raw prompt (no chat template)".into()),
        }
        let effort_note = self.effort_note.as_deref();
        if self.messages.is_some() {
            parts.push(match (self.effort, &app.cfg.default_reasoning_effort) {
                (Some(e), _) => {
                    format!(
                        "reasoning effort {} ({})",
                        shown(e),
                        effort_note.unwrap_or("sent")
                    )
                }
                (None, Some(d)) => format!(
                    "reasoning effort {d} (server default, {})",
                    effort_note.unwrap_or("none sent")
                ),
                (None, None) => format!(
                    "reasoning effort template default ({})",
                    effort_note.unwrap_or("none sent")
                ),
            });
        }
        let max_note = self.max_note.as_deref();
        parts.push(match self.max_tokens {
            Some(n) => format!("max_tokens {n} ({})", max_note.unwrap_or("sent")),
            None => format!(
                "max_tokens {} (server default, {})",
                app.cfg.default_max_tokens,
                max_note.unwrap_or("none sent")
            ),
        });
        if self.stream {
            parts.push("stream".into());
        }
        parts.join(" · ")
    }
}

/// Why a request did not get the model, for each API to report its way.
pub(crate) enum Refused {
    Unauthorized,
    BadRequest(String),
    /// Queue full or waited too long.
    Busy(String),
    Unavailable(String),
    Internal(String),
}

impl Refused {
    pub(crate) fn status(&self) -> StatusCode {
        match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Busy(_) => StatusCode::TOO_MANY_REQUESTS,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub(crate) fn message(&self) -> String {
        match self {
            Self::Unauthorized => "invalid API key".into(),
            Self::BadRequest(m) | Self::Busy(m) | Self::Unavailable(m) | Self::Internal(m) => {
                m.clone()
            }
        }
    }

    /// The OpenAI error shape.
    pub(crate) fn openai(&self) -> Response {
        let kind = match self {
            Self::Unauthorized => "authentication_error",
            Self::BadRequest(_) => "invalid_request_error",
            Self::Busy(_) => "rate_limit_error",
            Self::Unavailable(_) | Self::Internal(_) => "server_error",
        };
        error(self.status(), kind, self.message())
    }
}

/// The checks every generating route makes before parsing its body.
pub(crate) fn precheck(app: &App, headers: &HeaderMap) -> Result<(), Refused> {
    if !authorized(app, headers) {
        return Err(Refused::Unauthorized);
    }
    if app.worker.is_finished() {
        return Err(Refused::Unavailable(
            "the model thread is not running".into(),
        ));
    }
    // Cheap check before tokenizing a long prompt for nothing.
    if app.queue.is_full() {
        app.metrics.rejected();
        return Err(Refused::Busy(
            "the model is busy and the queue is full".into(),
        ));
    }
    Ok(())
}

/// Parses a JSON body (any Content-Type, as Ollama).
pub(crate) fn json_body(body: &Bytes) -> Result<Value, Refused> {
    serde_json::from_slice(body).map_err(|e| Refused::BadRequest(format!("invalid JSON: {e}")))
}

/// Runs `f` (template rendering, tokenizing: CPU work) off the async threads.
pub(crate) async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, Refused> {
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(Refused::BadRequest(e)),
        Err(e) => Err(Refused::Internal(e.to_string())),
    }
}

/// A job that has the model.
pub(crate) struct Running {
    pub(crate) events: UnboundedReceiver<Event>,
    pub(crate) cancel: Arc<AtomicBool>,
    /// Prompt tokens served from the prefix cache.
    pub(crate) cached: usize,
}

/// What a response hears while the job runs.
pub(crate) enum Item {
    Delta(Delta),
    /// A prefill chunk finished: streams write something, so a client that
    /// left is noticed before the first token.
    Prefill {
        done: usize,
        total: usize,
    },
}

/// Runs the job's events through `c` until it ends, handing every delta and
/// prefill step to `emit`. A stop sequence cancels the job.
pub(crate) async fn drive(
    app: &App,
    run: &mut Running,
    c: &mut Collector,
    mut emit: impl FnMut(Item),
) -> Result<worker::Done, String> {
    while let Some(ev) = run.events.recv().await {
        match ev {
            Event::Prefill(p) => emit(Item::Prefill {
                done: p.done,
                total: p.total,
            }),
            Event::Token(t) => {
                for d in c.token(&app.tokenizer, t) {
                    emit(Item::Delta(d));
                }
                if c.stopped {
                    run.cancel.store(true, Ordering::Relaxed);
                }
            }
            Event::Done(d) => {
                for x in c.finish(&app.tokenizer) {
                    emit(Item::Delta(x));
                }
                return Ok(d);
            }
            Event::Error(e) => {
                app.metrics.error();
                return Err(e);
            }
            Event::Started { .. } => {}
        }
    }
    if !run.cancel.load(Ordering::Relaxed) {
        app.metrics.error();
    }
    Err("generation ended unexpectedly".into())
}

/// Queues the job and waits for the slot.
pub(crate) async fn start(app: &App, job: &Gen) -> Result<Running, Refused> {
    eprintln!("{} {}", crate::report::clock(), job.summary);
    if job.prompt.len() >= app.max_context {
        return Err(Refused::BadRequest(format!(
            "the prompt has {} tokens; the model's context holds {} with the reply",
            job.prompt.len(),
            app.max_context
        )));
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let (etx, mut erx) = unbounded_channel();
    let job = Job {
        prompt: job.prompt.clone(),
        // Prompt and reply stay within the model's context.
        max_tokens: job.max_tokens.min(app.max_context - job.prompt.len()),
        temperature: job.temperature,
        top_p: job.top_p,
        seed: job.seed,
        cancel: cancel.clone(),
        events: etx,
    };
    if app.queue.push(job).is_err() {
        app.metrics.rejected();
        return Err(Refused::Busy(
            "the model is busy and the queue is full".into(),
        ));
    }
    app.metrics.set_queue(app.queue.len());
    match tokio::time::timeout(app.cfg.max_queue_wait, erx.recv()).await {
        Ok(Some(Event::Started { cached })) => Ok(Running {
            events: erx,
            cancel,
            cached,
        }),
        Ok(Some(Event::Error(e))) => {
            app.metrics.error();
            Err(Refused::Internal(e))
        }
        Ok(_) => Err(Refused::Internal("the model thread stopped".into())),
        Err(_) => {
            cancel.store(true, Ordering::Relaxed);
            app.metrics.rejected();
            Err(Refused::Busy("waited too long for the model".into()))
        }
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
        .cloned();
    let prompt = chat_prompt(
        app,
        &messages,
        tools,
        effort_or_default(app, effort).as_ref(),
    )?;
    let num = |k: &str| body.get(k).and_then(|v| v.as_f64());
    let sent_max = body
        .get("max_completion_tokens")
        .filter(|v| !v.is_null())
        .or_else(|| body.get("max_tokens"))
        .and_then(|v| v.as_f64());
    let max_tokens = sent_max.map_or(app.cfg.default_max_tokens, |v| v as usize);
    let stop = stop_list(body.get("stop"));
    let seed = num("seed").map(|v| v as u64).unwrap_or_else(random_seed);
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&app.cfg.model_name)
        .to_string();
    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let summary = Asked {
        api: "chat completions",
        model: &model,
        // As sent (the format instruction is ours).
        messages: body["messages"].as_array().map(|m| m.len()),
        tools: tools.and_then(|t| t.as_array()).map_or(0, |t| t.len()),
        effort: body
            .get("reasoning_effort")
            .or_else(|| body.pointer("/reasoning/effort"))
            .filter(|v| !v.is_null()),
        effort_note: None,
        max_tokens: sent_max,
        max_note: None,
        stream,
    }
    .line(app);
    Ok(Request {
        model,
        prompt,
        max_tokens: max_tokens.max(1),
        temperature: num("temperature").unwrap_or(1.0) as f32,
        top_p: num("top_p").unwrap_or(1.0) as f32,
        seed,
        stop,
        stream,
        include_usage: body
            .pointer("/stream_options/include_usage")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        summary,
    })
}

enum Source {
    /// Chat output, split by the response template.
    Template(ResponseParser),
    /// Plain text (raw completions).
    Raw(TextDecoder),
}

/// Turns tokens into reasoning, content and tool call deltas, applying stop
/// sequences to the content.
pub(crate) struct Collector {
    source: Source,
    stops: Vec<String>,
    pub(crate) content: String,
    pub(crate) reasoning: String,
    pub(crate) tool_calls: Vec<(String, String)>,
    pub(crate) stopped: bool,
    /// The stop sequence that ended the content, if one did.
    pub(crate) matched_stop: Option<String>,
}

impl Collector {
    pub(crate) fn new(specials: Specials, stops: Vec<String>) -> Self {
        Self::with_source(Source::Template(ResponseParser::new(specials)), stops)
    }

    /// Plain text: every token is content.
    pub(crate) fn raw(stops: Vec<String>) -> Self {
        Self::with_source(Source::Raw(TextDecoder::default()), stops)
    }

    fn with_source(source: Source, stops: Vec<String>) -> Self {
        Self {
            source,
            stops,
            content: String::new(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            stopped: false,
            matched_stop: None,
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
                            self.content[from..]
                                .find(stop.as_str())
                                .map(|i| (i + from, stop))
                        })
                        .min_by_key(|(i, _)| *i);
                    if let Some((cut, stop)) = hit {
                        self.matched_stop = Some(stop.clone());
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

    pub(crate) fn token(&mut self, t: &Tokenizer, id: u32) -> Vec<Delta> {
        let d = match &mut self.source {
            Source::Template(p) => p.push(t, id),
            Source::Raw(d) => d.push(t, id).map(Delta::Content).into_iter().collect(),
        };
        self.accept(d)
    }

    pub(crate) fn finish(&mut self, t: &Tokenizer) -> Vec<Delta> {
        let d = match &mut self.source {
            Source::Template(p) => p.finish(t),
            Source::Raw(d) => d.finish(t).map(Delta::Content).into_iter().collect(),
        };
        self.accept(d)
    }

    pub(crate) fn finish_reason(&self, finish: Finish) -> &'static str {
        if !self.tool_calls.is_empty() {
            "tool_calls"
        } else if self.stopped || finish == Finish::Stop {
            "stop"
        } else {
            "length"
        }
    }
}

pub(crate) fn completion_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_micros());
    format!("chatcmpl-{t:x}{n:x}")
}

pub(crate) fn tool_call_json(i: usize, id: &str, name: &str, arguments: &str) -> Value {
    json!({"index": i, "id": id, "type": "function", "function": {"name": name, "arguments": arguments}})
}

pub(crate) fn usage(prompt: usize, cached: usize, completion: usize) -> Value {
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
    let parsed = async {
        precheck(&app, &headers)?;
        let body = json_body(&body)?;
        let req = {
            let app = app.clone();
            blocking(move || parse_request(&app, &body)).await?
        };
        let job = Gen {
            prompt: req.prompt.clone(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            seed: req.seed,
            summary: req.summary.clone(),
        };
        let run = start(&app, &job).await?;
        Ok::<_, Refused>((req, run))
    };
    let (req, run) = match parsed.await {
        Ok(v) => v,
        Err(e) => return e.openai(),
    };
    if req.stream {
        stream_response(app, req, run.cancel, run.events)
    } else {
        full_response(app, req, run.cancel, run.events).await
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
            Event::Done(d) => {
                (finish, prompt_tokens, cached_tokens, completion_tokens) = (
                    d.finish,
                    d.prompt_tokens,
                    d.cached_tokens,
                    d.completion_tokens,
                );
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
                Event::Done(worker::Done {
                    finish,
                    prompt_tokens,
                    cached_tokens,
                    completion_tokens,
                    ..
                }) => {
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
