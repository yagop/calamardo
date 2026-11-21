//! The single slot: one thread owns the model (MLX state is per thread) and
//! serves jobs from a bounded queue one at a time.

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Instant,
};

use anyhow::Result;
use calamardo_model::{
    Model,
    cache::KvFormat,
    generate::{Cancelled, PrefillChunk, Sampler, Session},
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
    /// The job left the queue.
    Started,
    Prefill(PrefillChunk),
    Token(u32),
    Done {
        finish: Finish,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

pub struct WorkerConfig {
    pub model_dir: PathBuf,
    pub kv: KvFormat,
    pub prefill_step: usize,
    pub eos: u32,
}

/// Starts the worker thread. `ready` gets the load result.
pub fn spawn(
    config: WorkerConfig,
    jobs: mpsc::Receiver<Job>,
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
            for job in jobs {
                metrics.dequeued();
                if job.cancel.load(Ordering::Relaxed) {
                    continue;
                }
                let events = job.events.clone();
                if let Err(e) = run(&model, &config, &job, &metrics)
                    && e.downcast_ref::<Cancelled>().is_none()
                {
                    let _ = events.send(Event::Error(format!("{e:#}")));
                }
                metrics.set_active(mlx_rs::memory::active_memory().unwrap_or(0));
            }
        })
        .expect("spawning the model thread")
}

fn run(model: &Model, config: &WorkerConfig, job: &Job, metrics: &Metrics) -> Result<()> {
    let send = |e: Event| {
        if job.events.send(e).is_err() {
            // The request is gone: stop at the next check.
            job.cancel.store(true, Ordering::Relaxed);
        }
    };
    send(Event::Started);
    metrics.request_started();

    let mut session = Session::new(model, config.kv);
    session.prefill_step = config.prefill_step;
    let t = Instant::now();
    let mut lp = session.feed(&job.prompt, &job.cancel, &mut |c| {
        metrics.prefill_chunk(&c);
        send(Event::Prefill(c));
    })?;
    metrics.prefill_done(job.prompt.len(), t.elapsed().as_secs_f64());

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
        if token == config.eos {
            break Finish::Stop;
        }
        if n >= job.max_tokens {
            break Finish::Length;
        }
        lp = session.feed(&[token], &job.cancel, &mut |_| {})?;
    };
    metrics.decode_done(n, t.elapsed().as_secs_f64());
    send(Event::Done {
        finish,
        prompt_tokens: job.prompt.len(),
        completion_tokens: n,
    });
    Ok(())
}
