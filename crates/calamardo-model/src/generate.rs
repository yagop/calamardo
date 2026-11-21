//! Greedy generation, mirroring mlx-lm `generate_step`: the prompt minus its
//! last token is prefilled in chunks of `prefill_step` tokens, then each step
//! feeds one token and takes the argmax of the log-probabilities.

use anyhow::Result;
use mlx_rs::{Array, Dtype};

use crate::{cache::PassCache, model::Model, ops};

/// mlx-lm's default `prefill_step_size`.
pub const PREFILL_STEP: usize = 2048;

fn eval_cache(cache: &[PassCache]) -> Result<()> {
    let arrays: Vec<&Array> = cache.iter().flat_map(|c| c.arrays()).collect();
    ops::eval(&arrays)
}

/// tokens: (L,). Returns the greedy next token as a (1,) array.
fn step(model: &Model, tokens: &Array, cache: &mut [PassCache]) -> Result<Array> {
    let l = tokens.shape()[0];
    let logits = model.forward(&ops::reshape(tokens, &[1, l])?, cache)?;
    let logits = ops::squeeze(&ops::slice(&logits, 1, l - 1, l, 1)?, &[1])?;
    let logprobs = ops::sub(&logits, &ops::logsumexp(&logits, &[0, 1], true)?)?;
    ops::argmax(&logprobs, -1)
}

/// Generates up to `max_tokens` tokens after `prompt`, calling `on_token`
/// for each one; generation stops when it returns false.
pub fn greedy(
    model: &Model,
    prompt: &[u32],
    max_tokens: usize,
    prefill_step: usize,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<Vec<u32>> {
    anyhow::ensure!(!prompt.is_empty(), "empty prompt");
    let mut cache = model.make_cache();
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();

    let mut done = 0;
    while prompt.len() - done > 1 {
        let n = prefill_step.min(prompt.len() - done - 1);
        let chunk = Array::from_slice(&prompt_i32[done..done + n], &[n as i32]);
        model.forward(&ops::reshape(&chunk, &[1, n as i32])?, &mut cache)?;
        eval_cache(&cache)?;
        done += n;
        ops::clear_cache();
    }

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
