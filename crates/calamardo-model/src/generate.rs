//! Greedy generation, mirroring mlx-lm `generate_step`: the prompt minus its
//! last token is prefilled in chunks of `prefill_step` tokens, then each step
//! feeds one token and takes the argmax of the log-probabilities.

use anyhow::Result;
use mlx_rs::{Array, Dtype};

use crate::{
    cache::{KvFormat, PassCache},
    model::Model,
    ops,
};

/// mlx-lm's default `prefill_step_size`.
pub const PREFILL_STEP: usize = 2048;

fn eval_cache(cache: &[PassCache]) -> Result<()> {
    let arrays: Vec<&Array> = cache.iter().flat_map(|c| c.arrays()).collect();
    ops::eval(&arrays)
}

/// tokens: (L,). Returns the log-probabilities (1, V) of the next token.
fn step_logprobs(model: &Model, tokens: &Array, cache: &mut [PassCache]) -> Result<Array> {
    let l = tokens.shape()[0];
    let logits = model.forward(&ops::reshape(tokens, &[1, l])?, cache)?;
    let logits = ops::squeeze(&ops::slice(&logits, 1, l - 1, l, 1)?, &[1])?;
    ops::sub(&logits, &ops::logsumexp(&logits, &[0, 1], true)?)
}

/// tokens: (L,). Returns the greedy next token as a (1,) array.
fn step(model: &Model, tokens: &Array, cache: &mut [PassCache]) -> Result<Array> {
    ops::argmax(&step_logprobs(model, tokens, cache)?, -1)
}

/// One prefill chunk, reported as it finishes (K-9, API requirement 6).
#[derive(Clone, Copy, Debug)]
pub struct PrefillChunk {
    /// Prompt tokens processed so far, out of `total`.
    pub done: usize,
    pub total: usize,
    /// Tokens in this chunk.
    pub tokens: usize,
    pub seconds: f64,
    /// MLX memory in use after the chunk (weights and caches).
    pub active_bytes: usize,
    /// Highest MLX memory in use during the chunk.
    pub peak_bytes: usize,
}

/// Prefills all of `prompt` but its last token, as `greedy` does.
fn prefill(
    model: &Model,
    prompt: &[i32],
    prefill_step: usize,
    cache: &mut [PassCache],
    on_chunk: &mut dyn FnMut(PrefillChunk),
) -> Result<usize> {
    let mut done = 0;
    while prompt.len() - done > 1 {
        let n = prefill_step.min(prompt.len() - done - 1);
        let t = std::time::Instant::now();
        mlx_rs::memory::reset_peak_memory()?;
        let chunk = Array::from_slice(&prompt[done..done + n], &[n as i32]);
        model.forward(&ops::reshape(&chunk, &[1, n as i32])?, cache)?;
        eval_cache(cache)?;
        let peak_bytes = mlx_rs::memory::peak_memory()?;
        done += n;
        ops::clear_cache();
        on_chunk(PrefillChunk {
            done,
            total: prompt.len(),
            tokens: n,
            seconds: t.elapsed().as_secs_f64(),
            active_bytes: mlx_rs::memory::active_memory()?,
            peak_bytes,
        });
    }
    Ok(done)
}

/// Teacher forcing: after `prompt`, feeds `forced` one token at a time and
/// calls `on_logprobs(i, logprobs)` with the distribution that predicts
/// `forced[i]`. Every step after the prompt goes through the decode path.
pub fn teacher_forced(
    model: &Model,
    prompt: &[u32],
    forced: &[u32],
    prefill_step: usize,
    kv: KvFormat,
    mut on_logprobs: impl FnMut(usize, &Array) -> Result<()>,
) -> Result<()> {
    anyhow::ensure!(!prompt.is_empty(), "empty prompt");
    let mut cache = model.make_cache(kv);
    let prompt: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let done = prefill(model, &prompt, prefill_step, &mut cache, &mut |_| {})?;
    let last = &prompt[done..];
    let mut input = Array::from_slice(last, &[last.len() as i32]);
    for (i, &tok) in forced.iter().enumerate() {
        let lp = step_logprobs(model, &input, &mut cache)?;
        ops::eval(&[&lp])?;
        on_logprobs(i, &lp)?;
        input = Array::from_slice(&[tok as i32], &[1]);
        if i % 256 == 0 {
            ops::clear_cache();
        }
    }
    Ok(())
}

/// Generates up to `max_tokens` tokens after `prompt`, calling `on_token`
/// for each one; generation stops when it returns false.
pub fn greedy(
    model: &Model,
    prompt: &[u32],
    max_tokens: usize,
    prefill_step: usize,
    kv: KvFormat,
    mut on_prefill: impl FnMut(PrefillChunk),
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<Vec<u32>> {
    anyhow::ensure!(!prompt.is_empty(), "empty prompt");
    let mut cache = model.make_cache(kv);
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let done = prefill(
        model,
        &prompt_i32,
        prefill_step,
        &mut cache,
        &mut on_prefill,
    )?;

    let last = &prompt_i32[done..];
    let mut y = step(
        model,
        &Array::from_slice(last, &[last.len() as i32]),
        &mut cache,
    )?;
    let mut out = Vec::new();
    while out.len() < max_tokens {
        ops::eval(&[&y])?;
        let token = ops::astype(&y, Dtype::Uint32)?.item_exact::<u32>();
        out.push(token);
        if !on_token(token) || out.len() == max_tokens {
            break;
        }
        // As mlx-lm: return freed buffers every 256 steps.
        if (out.len() - 1) % 256 == 0 {
            ops::clear_cache();
        }
        y = step(model, &y, &mut cache)?;
    }
    Ok(out)
}

/// Generation was cancelled (client gone, stop sequence...).
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// A conversation's caches plus the tokens they hold. `feed` follows the
/// same schedule as `greedy`, so feeding a prompt and then one token at a
/// time reproduces it.
pub struct Session<'m> {
    model: &'m Model,
    cache: Vec<PassCache>,
    /// Every token fed so far, in order.
    pub tokens: Vec<u32>,
    pub prefill_step: usize,
}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model, kv: KvFormat) -> Self {
        Self {
            model,
            cache: model.make_cache(kv),
            tokens: Vec::new(),
            prefill_step: PREFILL_STEP,
        }
    }

    /// Feeds `tokens` and returns the log-probabilities (1, V) of the next
    /// one. All but the last token go through the chunked prefill; the
    /// last through a single-token step. Checks `cancel` between chunks.
    pub fn feed(
        &mut self,
        tokens: &[u32],
        cancel: &std::sync::atomic::AtomicBool,
        on_chunk: &mut dyn FnMut(PrefillChunk),
    ) -> Result<Array> {
        use std::sync::atomic::Ordering;
        anyhow::ensure!(!tokens.is_empty(), "nothing to feed");
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut done = 0;
        while ids.len() - done > 1 {
            if cancel.load(Ordering::Relaxed) {
                return Err(Cancelled.into());
            }
            let n = self.prefill_step.min(ids.len() - done - 1);
            let t = std::time::Instant::now();
            mlx_rs::memory::reset_peak_memory()?;
            let chunk = Array::from_slice(&ids[done..done + n], &[1, n as i32]);
            self.model.forward(&chunk, &mut self.cache)?;
            eval_cache(&self.cache)?;
            let peak_bytes = mlx_rs::memory::peak_memory()?;
            self.tokens.extend_from_slice(&tokens[done..done + n]);
            done += n;
            ops::clear_cache();
            on_chunk(PrefillChunk {
                done,
                total: ids.len(),
                tokens: n,
                seconds: t.elapsed().as_secs_f64(),
                active_bytes: mlx_rs::memory::active_memory()?,
                peak_bytes,
            });
        }
        let last = Array::from_slice(&ids[done..], &[(ids.len() - done) as i32]);
        let lp = step_logprobs(self.model, &last, &mut self.cache)?;
        self.tokens.extend_from_slice(&tokens[done..]);
        Ok(lp)
    }
}

/// Picks the next token from log-probabilities (1, V).
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    rng: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self {
            temperature,
            top_p,
            // splitmix64 never sees a zero state.
            rng: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn sample(&mut self, logprobs: &Array) -> Result<u32> {
        if self.temperature <= 0.0 {
            // Greedy on the GPU, exactly as `greedy` does.
            let y = ops::astype(&ops::argmax(logprobs, -1)?, Dtype::Uint32)?;
            ops::eval(&[&y])?;
            return Ok(y.item_exact::<u32>());
        }
        let lp = ops::f32(logprobs)?;
        ops::eval(&[&lp])?;
        let lp = lp.as_slice::<f32>();
        let t = self.temperature as f64;
        let max = lp.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut probs: Vec<(u32, f64)> = lp
            .iter()
            .enumerate()
            .map(|(i, &x)| (i as u32, ((x as f64 - max) / t).exp()))
            .filter(|&(_, p)| p > 0.0)
            .collect();
        let z: f64 = probs.iter().map(|p| p.1).sum();
        if self.top_p < 1.0 {
            probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut acc = 0.0;
            let mut keep = probs.len();
            for (i, p) in probs.iter().enumerate() {
                acc += p.1 / z;
                if acc >= self.top_p as f64 {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
        }
        let total: f64 = probs.iter().map(|p| p.1).sum();
        let mut r = self.next_f64() * total;
        for &(id, p) in &probs {
            r -= p;
            if r <= 0.0 {
                return Ok(id);
            }
        }
        Ok(probs.last().map_or(0, |p| p.0))
    }
}
