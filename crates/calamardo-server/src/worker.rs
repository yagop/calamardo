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
    generate::{Cancelled, DraftPolicy, PrefillChunk, Reuse, Sampler, Session},
};
use tokio::sync::{mpsc::UnboundedSender, oneshot};

use crate::{disk::DiskCache, metrics::Metrics};
use calamardo_model::draft::Drafter;

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
    closed: AtomicBool,
    /// Cancel flag of the job holding the slot.
    current: Mutex<Option<Arc<AtomicBool>>>,
}

impl Queue {
    pub fn new(capacity: usize) -> Self {
        Self {
            jobs: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
            capacity,
            closed: AtomicBool::new(false),
            current: Mutex::new(None),
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

    fn set_current(&self, cancel: Option<Arc<AtomicBool>>) {
        *self.current.lock().unwrap_or_else(|e| e.into_inner()) = cancel;
    }

    /// Shutdown: refuses waiting jobs, stops the running one (its session is
    /// kept, C-7) and lets the worker save and exit.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
        if let Some(c) = self
            .current
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            c.store(true, Ordering::Relaxed);
        }
        let mut q = self.lock();
        for job in q.drain(..) {
            let _ = job
                .events
                .send(Event::Error("the server is shutting down".into()));
        }
        self.ready.notify_all();
    }

    fn pop(&self) -> Option<Job> {
        let mut q = self.lock();
        loop {
            while let Some(job) = q.pop_front() {
                if !job.cancel.load(Ordering::Relaxed) {
                    return Some(job);
                }
            }
            if self.closed.load(Ordering::Relaxed) {
                return None;
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
    /// Directory and byte budget of the prefix cache on disk, if enabled.
    pub disk: Option<(PathBuf, u64)>,
    /// Identifies the checkpoint in saved sessions (see `model_id`).
    pub model_id: String,
    /// Speculative decoding with the bundled DSpark drafter (M-7).
    pub speculative: bool,
    /// Largest draft block (anchor plus drafts); the drafter's trained block
    /// caps it.
    pub draft_block: usize,
}

/// Prefix cache in RAM (C-1, C-2): idle sessions with their snapshots.
struct PrefixCache<'m> {
    model: &'m Model,
    drafter: Option<&'m Drafter>,
    metrics: &'m Metrics,
    disk: Option<DiskCache>,
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

    fn fresh(&self) -> Session<'m> {
        let mut s = Session::new(self.model, self.kv);
        if let Some(d) = self.drafter {
            s = s.with_drafter(d);
        }
        s.prefill_step = self.prefill_step;
        s
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
        // A conversation saved on disk may hold more of the prompt (C-3).
        let aligned_to = self.aligned.then_some(self.prefill_step);
        let on_disk = self.disk.as_ref().and_then(|d| d.best(prompt, aligned_to));
        if let Some((di, reuse)) = on_disk
            && reuse.tokens() > best.map_or(0, |(_, r)| r.tokens())
        {
            let d = self.disk.as_mut().unwrap();
            match d.take(di, self.model, self.drafter, self.kv) {
                Ok((mut s, secs)) => {
                    s.prefill_step = self.prefill_step;
                    self.metrics.disk_load(secs);
                    eprintln!("restored {} tokens from disk in {secs:.2}s", s.tokens.len());
                    if let Reuse::Snapshot { index, .. } = reuse {
                        s.restore(index);
                    }
                    self.report_disk();
                    return (s, reuse.tokens());
                }
                Err(e) => eprintln!("could not restore a saved session: {e:#}"),
            }
            self.report_disk();
        }
        match best {
            Some((i, Reuse::Live { len })) => (self.entries.remove(i).0, len),
            Some((i, Reuse::Snapshot { index, len })) => {
                let original = &self.entries[i].0;
                // Branch off only if the original holds a later user turn
                // worth keeping, and memory allows; a regenerated answer just
                // replaces the old one.
                if original.branches_after(index) && self.bytes() + original.nbytes() <= self.budget
                {
                    match original.fork(index) {
                        Ok(s) => return (s, len),
                        Err(e) => eprintln!("could not fork a session: {e:#}"),
                    }
                }
                let (mut s, _) = self.entries.remove(i);
                s.restore(index);
                (s, len)
            }
            None => (self.fresh(), 0),
        }
    }

    /// Returns a session after a request, evicting the least recently used
    /// ones (then old snapshots) beyond the budget.
    fn put(&mut self, session: Session<'m>) {
        if session.tokens.is_empty() {
            // Nothing to reuse (the client left before the first chunk).
            self.forget(&session);
            return;
        }
        self.clock += 1;
        self.entries.push((session, self.clock));
        while self.bytes() > self.budget {
            if self.entries.len() > 1 {
                let lru = (0..self.entries.len() - 1)
                    .min_by_key(|&i| self.entries[i].1)
                    .unwrap();
                let (evicted, _) = self.entries.remove(lru);
                self.save_to_disk(&evicted);
            } else if !self.entries[0].0.drop_snapshot() {
                break;
            }
        }
    }

    /// A session leaves RAM without being saved: its file, if any, is no
    /// longer resident.
    fn forget(&mut self, session: &Session<'m>) {
        if let (Some(d), Some(origin)) = (self.disk.as_mut(), &session.origin) {
            d.release(origin);
        }
    }

    /// Saves a session leaving RAM instead of losing it (C-3).
    fn save_to_disk(&mut self, s: &Session<'m>) {
        if s.tokens.is_empty() {
            return;
        }
        if let Some(d) = self.disk.as_mut() {
            match d.save(s) {
                Ok(secs) => {
                    self.metrics.disk_save(secs);
                    eprintln!("saved {} tokens to disk in {secs:.2}s", s.tokens.len());
                }
                Err(e) => {
                    eprintln!("could not save a session to disk: {e:#}");
                    self.forget(s);
                }
            }
        }
        self.report_disk();
    }

    fn report_disk(&self) {
        if let Some(d) = &self.disk {
            self.metrics.set_disk(d.len(), d.bytes() as usize);
        }
    }

    /// Loads the most recent saved conversation into RAM (C-9).
    fn prewarm(&mut self) {
        let Some(i) = self.disk.as_ref().and_then(|d| d.latest()) else {
            return;
        };
        match self
            .disk
            .as_mut()
            .unwrap()
            .take(i, self.model, self.drafter, self.kv)
        {
            Ok((mut s, secs)) => {
                s.prefill_step = self.prefill_step;
                self.metrics.disk_load(secs);
                eprintln!(
                    "prewarmed {} tokens from disk in {secs:.2}s",
                    s.tokens.len()
                );
                self.put(s);
            }
            Err(e) => eprintln!("could not prewarm: {e:#}"),
        }
        self.report_disk();
    }

    /// Saves every session in RAM, least recently used first so the most
    /// recent one is the newest file (and the one prewarmed next time).
    fn flush(&mut self) {
        let mut entries = std::mem::take(&mut self.entries);
        entries.sort_by_key(|(_, used)| *used);
        for (s, _) in entries {
            self.save_to_disk(&s);
        }
    }
}

/// A stable identity of the checkpoint for saved sessions (C-4): the
/// snapshot's configuration, tensor index and file sizes.
pub fn model_id(dir: &std::path::Path) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for name in ["config.json", "model.safetensors.index.json"] {
        feed(&std::fs::read(dir.join(name)).unwrap_or_default());
    }
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map(|d| d.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    files.sort();
    for f in files {
        if f.extension().is_some_and(|e| e == "safetensors") {
            feed(f.file_name().unwrap_or_default().as_encoded_bytes());
            feed(
                &std::fs::metadata(&f)
                    .map(|m| m.len())
                    .unwrap_or(0)
                    .to_le_bytes(),
            );
        }
    }
    format!("{h:016x}")
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
            let loaded = Model::load(&config.model_dir).and_then(|m| {
                // Drafting decides by measured speed, so its output is not
                // reproducible: off when bit-exact output is asked for.
                let drafter = if config.speculative
                    && !config.aligned_prefix
                    && config.model_dir.join("mtp.safetensors").exists()
                {
                    Some(Drafter::load(&config.model_dir, &m)?)
                } else {
                    None
                };
                Ok((m, drafter))
            });
            let (model, drafter) = match loaded {
                Ok(m) => {
                    let _ = ready.send(Ok(()));
                    m
                }
                Err(e) => {
                    let _ = ready.send(Err(e));
                    return;
                }
            };
            eprintln!(
                "speculative decoding: {}",
                if drafter.is_some() {
                    "DSpark drafter"
                } else {
                    "off"
                }
            );
            metrics.set_active(mlx_rs::memory::active_memory().unwrap_or(0));
            let mut manifest: std::collections::BTreeMap<String, String> = [
                ("model".to_string(), config.model_id.clone()),
                ("kv_bits".to_string(), config.kv.bits.to_string()),
                ("kv_group".to_string(), config.kv.group_size.to_string()),
            ]
            .into_iter()
            .collect();
            if config.aligned_prefix {
                manifest.insert("prefill_step".into(), config.prefill_step.to_string());
            }
            let disk = config.disk.as_ref().and_then(|(dir, budget)| {
                DiskCache::open(dir, *budget, manifest.clone())
                    .map_err(|e| eprintln!("disk cache disabled: {e:#}"))
                    .ok()
            });
            let mut cache = PrefixCache {
                model: &model,
                drafter: drafter.as_ref(),
                metrics: &metrics,
                disk,
                entries: Vec::new(),
                clock: 0,
                budget: config.cache_budget,
                kv: config.kv,
                prefill_step: config.prefill_step,
                aligned: config.aligned_prefix,
            };
            cache.prewarm();
            // One policy for the server: it learns block sizes across requests.
            let mut policy = drafter
                .as_ref()
                .map(|d| DraftPolicy::new(config.draft_block.min(d.block_size)));
            while let Some(job) = queue.pop() {
                metrics.set_queue(queue.len());
                queue.set_current(Some(job.cancel.clone()));
                let (mut session, cached) = cache.take(&job.prompt);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(
                        &mut session,
                        cached,
                        &config,
                        &job,
                        &metrics,
                        policy.as_mut(),
                    )
                }));
                queue.set_current(None);
                match result {
                    Ok(Ok(())) => cache.put(session),
                    // Keep what was processed: a retry continues from it (C-7).
                    Ok(Err(e)) if e.downcast_ref::<Cancelled>().is_some() => cache.put(session),
                    Ok(Err(e)) => {
                        metrics.error();
                        let _ = job.events.send(Event::Error(format!("{e:#}")));
                        // The caches may hold half a chunk: return to the
                        // last snapshot rather than lose the conversation.
                        if session.recover() {
                            cache.put(session);
                        } else {
                            cache.forget(&session);
                        }
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
                        cache.forget(&session);
                    }
                }
                metrics.job_finished();
                // Return freed buffers to the system between jobs.
                calamardo_model::ops::clear_cache();
                metrics.set_cache(cache.entries.len(), cache.bytes());
                metrics.set_active(mlx_rs::memory::active_memory().unwrap_or(0));
            }
            // Shutting down: keep the conversations for the next start.
            cache.flush();
        })
        .expect("spawning the model thread")
}

fn run(
    session: &mut Session,
    cached: usize,
    config: &WorkerConfig,
    job: &Job,
    metrics: &Metrics,
    mut policy: Option<&mut DraftPolicy>,
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
    let lp = session.feed(&job.prompt[cached..], &job.cancel, &mut |mut c| {
        c.done += cached;
        c.total += cached;
        metrics.prefill_chunk(&c);
        send(Event::Prefill(c));
    })?;
    metrics.prefill_done(job.prompt.len() - cached, t.elapsed().as_secs_f64());

    let mut sampler = Sampler::new(job.temperature, job.top_p, job.seed);
    if !session.has_drafter() {
        policy = None;
    }
    let mut cleared = 0;
    let t = Instant::now();
    let stop = &config.stop_tokens;
    let mut last = sampler.sample(&lp)?;
    let mut n = 1;
    send(Event::Token(last));
    let finish = 'decode: loop {
        if stop.contains(&last) {
            break Finish::Stop;
        }
        if n >= job.max_tokens {
            break Finish::Length;
        }
        if job.cancel.load(Ordering::Relaxed) {
            break Finish::Cancelled;
        }
        // At most as many tokens as are still allowed.
        let block = policy.as_mut().map_or(1, |p| p.next(job.max_tokens - n));
        let step = Instant::now();
        if block >= 2 {
            let r = session.speculate(last, block, &mut sampler, stop)?;
            if let Some(p) = policy.as_mut() {
                p.record(block, r.tokens.len(), step.elapsed().as_secs_f64());
            }
            metrics.draft(r.accepted, r.drafted);
            for &tok in &r.tokens {
                n += 1;
                send(Event::Token(tok));
                last = tok;
                if stop.contains(&tok) {
                    break 'decode Finish::Stop;
                }
            }
        } else {
            let lp = session.feed(&[last], &job.cancel, &mut |_| {})?;
            last = sampler.sample(&lp)?;
            if let Some(p) = policy.as_mut() {
                p.record(1, 1, step.elapsed().as_secs_f64());
            }
            n += 1;
            send(Event::Token(last));
        }
        if n / 256 > cleared {
            // As mlx-lm's generate_step, every 256 tokens.
            cleared = n / 256;
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
