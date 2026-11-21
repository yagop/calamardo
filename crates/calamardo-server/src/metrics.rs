//! Prometheus metrics (`GET /metrics`), including the memory peak of every
//! prefill step (K-9).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use calamardo_model::generate::PrefillChunk;

#[derive(Default)]
pub struct Metrics {
    requests: AtomicU64,
    rejected: AtomicU64,
    errors: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    /// f64 bits.
    prefill_seconds: AtomicU64,
    decode_seconds: AtomicU64,
    last_prefill_tps: AtomicU64,
    last_decode_tps: AtomicU64,
    queued: AtomicU64,
    busy: AtomicU64,
    active_bytes: AtomicU64,
    /// Highest MLX memory during a prefill chunk, ever and in the last request.
    prefill_peak_bytes_max: AtomicU64,
    last_prefill_peak_bytes: AtomicU64,
    /// Highest (peak - active) of a chunk in the last request.
    last_prefill_transient_bytes: AtomicU64,
}

fn add_f64(a: &AtomicU64, v: f64) {
    let _ = a.try_update(Relaxed, Relaxed, |x| {
        Some((f64::from_bits(x) + v).to_bits())
    });
}

fn set_f64(a: &AtomicU64, v: f64) {
    a.store(v.to_bits(), Relaxed);
}

impl Metrics {
    pub fn enqueued(&self) {
        self.queued.fetch_add(1, Relaxed);
    }

    pub fn dequeued(&self) {
        let _ = self
            .queued
            .try_update(Relaxed, Relaxed, |q| Some(q.saturating_sub(1)));
    }

    pub fn rejected(&self) {
        self.rejected.fetch_add(1, Relaxed);
    }

    pub fn error(&self) {
        self.errors.fetch_add(1, Relaxed);
    }

    pub fn request_started(&self) {
        self.requests.fetch_add(1, Relaxed);
        self.busy.store(1, Relaxed);
        self.last_prefill_peak_bytes.store(0, Relaxed);
        self.last_prefill_transient_bytes.store(0, Relaxed);
    }

    pub fn set_active(&self, bytes: usize) {
        self.active_bytes.store(bytes as u64, Relaxed);
    }

    pub fn prefill_chunk(&self, c: &PrefillChunk) {
        let peak = c.peak_bytes as u64;
        self.prefill_peak_bytes_max.fetch_max(peak, Relaxed);
        self.last_prefill_peak_bytes.fetch_max(peak, Relaxed);
        self.last_prefill_transient_bytes
            .fetch_max(peak.saturating_sub(c.active_bytes as u64), Relaxed);
        self.active_bytes.store(c.active_bytes as u64, Relaxed);
    }

    pub fn prefill_done(&self, tokens: usize, seconds: f64) {
        self.prompt_tokens.fetch_add(tokens as u64, Relaxed);
        add_f64(&self.prefill_seconds, seconds);
        set_f64(&self.last_prefill_tps, tokens as f64 / seconds.max(1e-9));
    }

    pub fn decode_done(&self, tokens: usize, seconds: f64) {
        self.completion_tokens.fetch_add(tokens as u64, Relaxed);
        add_f64(&self.decode_seconds, seconds);
        if tokens > 1 {
            set_f64(
                &self.last_decode_tps,
                (tokens - 1) as f64 / seconds.max(1e-9),
            );
        }
        self.busy.store(0, Relaxed);
    }

    pub fn render(&self) -> String {
        let u = |a: &AtomicU64| a.load(Relaxed);
        let f = |a: &AtomicU64| f64::from_bits(a.load(Relaxed));
        let mut s = String::new();
        let mut put = |name: &str, kind: &str, help: &str, value: String| {
            s.push_str(&format!("# HELP calamardo_{name} {help}\n# TYPE calamardo_{name} {kind}\ncalamardo_{name} {value}\n"));
        };
        put(
            "requests_total",
            "counter",
            "Requests that reached the model.",
            u(&self.requests).to_string(),
        );
        put(
            "rejected_total",
            "counter",
            "Requests rejected with 429.",
            u(&self.rejected).to_string(),
        );
        put(
            "errors_total",
            "counter",
            "Requests that failed.",
            u(&self.errors).to_string(),
        );
        put(
            "prompt_tokens_total",
            "counter",
            "Prompt tokens processed.",
            u(&self.prompt_tokens).to_string(),
        );
        put(
            "completion_tokens_total",
            "counter",
            "Tokens generated.",
            u(&self.completion_tokens).to_string(),
        );
        put(
            "prefill_seconds_total",
            "counter",
            "Time spent in prefill.",
            f(&self.prefill_seconds).to_string(),
        );
        put(
            "decode_seconds_total",
            "counter",
            "Time spent decoding.",
            f(&self.decode_seconds).to_string(),
        );
        put(
            "last_prefill_tokens_per_second",
            "gauge",
            "Prefill speed of the last request.",
            f(&self.last_prefill_tps).to_string(),
        );
        put(
            "last_decode_tokens_per_second",
            "gauge",
            "Decode speed of the last request.",
            f(&self.last_decode_tps).to_string(),
        );
        put(
            "queue_length",
            "gauge",
            "Requests waiting for the slot.",
            u(&self.queued).to_string(),
        );
        put(
            "busy",
            "gauge",
            "1 while a request holds the slot.",
            u(&self.busy).to_string(),
        );
        put(
            "active_memory_bytes",
            "gauge",
            "MLX memory in use (weights and caches).",
            u(&self.active_bytes).to_string(),
        );
        put(
            "prefill_peak_bytes_max",
            "gauge",
            "Highest MLX memory seen during any prefill step.",
            u(&self.prefill_peak_bytes_max).to_string(),
        );
        put(
            "last_prefill_peak_bytes",
            "gauge",
            "Highest MLX memory during the last request's prefill steps.",
            u(&self.last_prefill_peak_bytes).to_string(),
        );
        put(
            "last_prefill_transient_bytes",
            "gauge",
            "Largest peak above active memory of a prefill step in the last request.",
            u(&self.last_prefill_transient_bytes).to_string(),
        );
        s
    }
}
