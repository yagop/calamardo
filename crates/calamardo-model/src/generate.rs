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
