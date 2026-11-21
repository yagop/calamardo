//! The single slot: one thread owns the model (MLX state is per thread) and
//! serves jobs from a bounded queue one at a time.

use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use anyhow::Result;
use calamardo_model::{
    Model,
    cache::KvFormat,
    generate::{Cancelled, PrefillChunk, Reuse, Sampler, Session},
};
use tokio::sync::{mpsc::UnboundedSender, oneshot};

use crate::metrics::Metrics;

pub struct Job {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    /// Set by the request side to stop: client gone, stop sequence, timeout.
    pub cancel: Arc<AtomicBool>,
    pub events: UnboundedSender<Event>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Finish {
    /// End-of-sampling token.
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug)]
pub enum Event {
    /// The job left the queue; `cached` prompt tokens come from the prefix
    /// cache.
    Started {
        cached: usize,
    },
    Prefill(PrefillChunk),
    Token(u32),
    Done {
        finish: Finish,
        prompt_tokens: usize,
        cached_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

/// Jobs waiting for the slot. Cancelled jobs (timed out in the queue, client
/// gone) are dropped as soon as anyone looks, so they never hold a place.
pub struct Queue {
    jobs: Mutex<VecDeque<Job>>,
    ready: Condvar,
    capacity: usize,
}

impl Queue {
    pub fn new(capacity: usize) -> Self {
        Self {
            jobs: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
            capacity,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<Job>> {
        self.jobs.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Waiting jobs that are still wanted.
    pub fn len(&self) -> usize {
        let mut q = self.lock();
        q.retain(|j| !j.cancel.load(Ordering::Relaxed));
        q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Queues `job`, or gives it back if the queue is full.
    pub fn push(&self, job: Job) -> std::result::Result<(), Box<Job>> {
        let mut q = self.lock();
        q.retain(|j| !j.cancel.load(Ordering::Relaxed));
        if q.len() >= self.capacity {
            return Err(Box::new(job));
        }
        q.push_back(job);
        self.ready.notify_one();
        Ok(())
    }

    fn pop(&self) -> Job {
        let mut q = self.lock();
        loop {
            while let Some(job) = q.pop_front() {
                if !job.cancel.load(Ordering::Relaxed) {
                    return job;
                }
            }
            q = self.ready.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }
}

pub struct WorkerConfig {
    pub model_dir: PathBuf,
    pub kv: KvFormat,
    pub prefill_step: usize,
    /// End of sampling, plus special tokens that would start a turn the model
    /// must not write (a user or system message, end of text).
    pub stop_tokens: Vec<u32>,
    /// Bytes the idle sessions and their snapshots may hold.
    pub cache_budget: usize,
    /// Reuse prefixes only at multiples of `prefill_step` (bit-exact output).
    pub aligned_prefix: bool,
}

/// Prefix cache in RAM (C-1, C-2): idle sessions with their snapshots.
struct PrefixCache<'m> {
    model: &'m Model,
    entries: Vec<(Session<'m>, u64)>,
    clock: u64,
    budget: usize,
    kv: KvFormat,
    prefill_step: usize,
    aligned: bool,
}

impl<'m> PrefixCache<'m> {
    fn bytes(&self) -> usize {
        self.entries.iter().map(|(s, _)| s.nbytes()).sum()
    }

    /// The session to serve `prompt` from, and how many prompt tokens it
    /// already holds.
    fn take(&mut self, prompt: &[u32]) -> (Session<'m>, usize) {
        let best = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, (s, _))| s.reuse(prompt, self.aligned).map(|r| (i, r)))
            .max_by_key(|(_, r)| r.tokens());
        match best {
            Some((i, Reuse::Live { len })) => (self.entries.remove(i).0, len),
            Some((i, Reuse::Snapshot { index, len })) => {
                let size = self.entries[i].0.nbytes();
                if self.bytes() + size <= self.budget {
                    // Keep the original conversation and branch off it.
                    (self.entries[i].0.fork(index), len)
                } else {
                    let (mut s, _) = self.entries.remove(i);
                    s.restore(index);
                    (s, len)
                }
            }
            None => {
                let mut s = Session::new(self.model, self.kv);
                s.prefill_step = self.prefill_step;
                (s, 0)
            }
        }
    }

    /// Returns a session after a request, evicting the least recently used
    /// ones (then old snapshots) beyond the budget.
    fn put(&mut self, session: Session<'m>) {
        self.clock += 1;
        self.entries.push((session, self.clock));
        while self.bytes() > self.budget {
            if self.entries.len() > 1 {
                let lru = (0..self.entries.len() - 1)
                    .min_by_key(|&i| self.entries[i].1)
                    .unwrap();
                self.entries.remove(lru);
            } else if !self.entries[0].0.drop_snapshot() {
                break;
            }
        }
    }
}

/// Starts the worker thread. `ready` gets the load result.
pub fn spawn(
    config: WorkerConfig,
    queue: Arc<Queue>,
    metrics: Arc<Metrics>,
    ready: oneshot::Sender<Result<()>>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("calamardo-model".into())
        .spawn(move || {
            let model = match Model::load(&config.model_dir) {
                Ok(m) => {
                    let _ = ready.send(Ok(()));
                    m
                }
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            };
            metrics.set_active(mlx_rs::memory::active_memory().unwrap_or(0));
            let mut cache = PrefixCache {
                model: &model,
                entries: Vec::new(),
                clock: 0,
                budget: config.cache_budget,
                kv: config.kv,
                prefill_step: config.prefill_step,
                aligned: config.aligned_prefix,
            };
            loop {
                let job = queue.pop();
                metrics.set_queue(queue.len());
                let (mut session, cached) = cache.take(&job.prompt);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(&mut session, cached, &config, &job, &metrics)
                }));
                match result {
                    Ok(Ok(())) => cache.put(session),
                    // Keep what was processed: a retry continues from it (C-7).
                    Ok(Err(e)) if e.downcast_ref::<Cancelled>().is_some() => cache.put(session),
                    Ok(Err(e)) => {
                        // The session may be half updated: drop it.
                        metrics.error();
                        let _ = job.events.send(Event::Error(format!("{e:#}")));
                    }
                    Err(panic) => {
                        metrics.error();
                        let msg = panic
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "unknown".into());
                        let _ = job
                            .events
                            .send(Event::Error(format!("internal error: {msg}")));
                    }
                }
                metrics.job_finished();
                // Return freed buffers to the system between jobs.
                calamardo_model::ops::clear_cache();
                metrics.set_cache(cache.entries.len(), cache.bytes());
                metrics.set_active(mlx_rs::memory::active_memory().unwrap_or(0));
            }
        })
        .expect("spawning the model thread")
}

fn run(
    session: &mut Session,
    cached: usize,
    config: &WorkerConfig,
    job: &Job,
    metrics: &Metrics,
) -> Result<()> {
    let send = |e: Event| {
        if job.events.send(e).is_err() {
            // The request is gone: stop at the next check.
            job.cancel.store(true, Ordering::Relaxed);
        }
    };
    send(Event::Started { cached });
    metrics.request_started(cached);

    let t = Instant::now();
    let mut lp = session.feed(&job.prompt[cached..], &job.cancel, &mut |mut c| {
        c.done += cached;
        c.total += cached;
        metrics.prefill_chunk(&c);
        send(Event::Prefill(c));
    })?;
    metrics.prefill_done(job.prompt.len() - cached, t.elapsed().as_secs_f64());

    let mut sampler = Sampler::new(job.temperature, job.top_p, job.seed);
    let t = Instant::now();
    let mut n = 0;
    let finish = loop {
        if job.cancel.load(Ordering::Relaxed) {
            break Finish::Cancelled;
        }
        let token = sampler.sample(&lp)?;
        n += 1;
        send(Event::Token(token));
        if config.stop_tokens.contains(&token) {
            break Finish::Stop;
        }
        if n >= job.max_tokens {
            break Finish::Length;
        }
        lp = session.feed(&[token], &job.cancel, &mut |_| {})?;
        if n % 256 == 0 {
            // As mlx-lm's generate_step.
            calamardo_model::ops::clear_cache();
        }
    };
    metrics.decode_done(n, t.elapsed().as_secs_f64());
    send(Event::Done {
        finish,
        prompt_tokens: job.prompt.len(),
        cached_tokens: cached,
        completion_tokens: n,
    });
    Ok(())
}
