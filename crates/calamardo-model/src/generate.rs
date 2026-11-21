//! Greedy generation, mirroring mlx-lm `generate_step`: the prompt minus its
//! last token is prefilled in chunks of `prefill_step` tokens, then each step
//! feeds one token and takes the argmax of the log-probabilities.

use anyhow::Result;
use mlx_rs::{Array, Dtype};

use anyhow::Context;

use crate::{
    cache::{KvFormat, PassCache},
    draft::{DraftSnapshot, DraftState, Drafter},
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

/// Periodic snapshots during a prefill (C-1).
pub const SNAPSHOT_EVERY: usize = 8192;

/// A point of a session's sequence it can return to.
#[derive(Clone)]
pub struct Snapshot {
    /// Tokens before this point.
    pub len: usize,
    /// Taken just before a prompt's last token (vs. periodic).
    pub prompt_end: bool,
    /// Reached only through full prefill chunks from the start, as a cold run
    /// would: resuming here is bit-exact (C-2, aligned mode).
    pub aligned: bool,
    passes: Vec<crate::cache::PassSnapshot>,
    draft: Option<DraftSnapshot>,
}

impl Snapshot {
    pub fn nbytes(&self) -> usize {
        self.passes.iter().map(|p| p.nbytes()).sum::<usize>()
            + self.draft.as_ref().map_or(0, |d| d.nbytes())
    }

    fn arrays(&self) -> Vec<Array> {
        let mut out: Vec<Array> = self
            .passes
            .iter()
            .flat_map(|p| match p {
                crate::cache::PassSnapshot::Global { conv, .. } => {
                    conv.iter().flatten().cloned().collect::<Vec<_>>()
                }
                crate::cache::PassSnapshot::Window { cache, conv } => cache
                    .arrays()
                    .into_iter()
                    .cloned()
                    .chain(conv.iter().flatten().cloned())
                    .collect(),
            })
            .collect();
        if let Some(d) = &self.draft {
            out.extend(d.arrays());
        }
        out
    }
}

/// Where a new prompt can start from in a session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reuse {
    /// The prompt extends everything the session holds.
    Live { len: usize },
    /// The prompt shares the first `len` tokens, saved in snapshot `index`.
    Snapshot { index: usize, len: usize },
}

impl Reuse {
    /// Prompt tokens the session already holds from this point.
    pub fn tokens(self) -> usize {
        match self {
            Self::Live { len } | Self::Snapshot { len, .. } => len,
        }
    }
}

/// One speculative round's outcome.
pub struct Speculated {
    /// Accepted drafts followed by the target's own next token (the new
    /// last token, not yet fed).
    pub tokens: Vec<u32>,
    pub drafted: usize,
    pub accepted: usize,
}

/// A conversation's caches plus the tokens they hold. `feed` follows the
/// same schedule as `greedy`, so feeding a prompt and then one token at a
/// time reproduces it.
#[derive(Clone)]
pub struct Session<'m> {
    model: &'m Model,
    drafter: Option<&'m Drafter>,
    cache: Vec<PassCache>,
    draft: Option<DraftState>,
    /// Every token fed so far, in order.
    pub tokens: Vec<u32>,
    pub prefill_step: usize,
    /// Earlier points of `tokens`, by increasing length.
    pub snapshots: Vec<Snapshot>,
    /// Everything so far came in full prefill chunks from the start.
    pure: bool,
    /// The saved file this session was loaded from, if any.
    pub origin: Option<std::path::PathBuf>,
}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model, kv: KvFormat) -> Self {
        Self {
            model,
            drafter: None,
            cache: model.make_cache(kv),
            draft: None,
            tokens: Vec::new(),
            prefill_step: PREFILL_STEP,
            snapshots: Vec::new(),
            pure: true,
            origin: None,
        }
    }

    /// Enables speculative decoding with `drafter`, which then follows every
    /// token fed (on a fresh session).
    pub fn with_drafter(mut self, drafter: &'m Drafter) -> Self {
        self.draft = Some(drafter.new_state());
        self.drafter = Some(drafter);
        self
    }

    pub fn has_drafter(&self) -> bool {
        self.drafter.is_some()
    }

    fn take_snapshot(&mut self, prompt_end: bool) {
        if let Some(last) = self.snapshots.last_mut()
            && last.len == self.tokens.len()
        {
            last.prompt_end |= prompt_end;
            return;
        }
        self.snapshots.push(Snapshot {
            len: self.tokens.len(),
            prompt_end,
            aligned: self.pure,
            passes: self.cache.iter().map(|c| c.snapshot()).collect(),
            draft: self.draft.as_ref().map(|d| d.snapshot()),
        });
    }

    /// The longest start for `prompt`: the live end if the prompt extends
    /// it, else the longest snapshot inside the common prefix. At least one
    /// prompt token is always left to feed. With `aligned`, only snapshots
    /// reached through full prefill chunks qualify (never the live end), so
    /// the chunks match a cold run and the output is bit for bit the same
    /// (C-2).
    pub fn reuse(&self, prompt: &[u32], aligned: bool) -> Option<Reuse> {
        let common = self
            .tokens
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        let ok = |len: usize| len > 0 && len < prompt.len();
        let live = (!aligned && common == self.tokens.len() && ok(common))
            .then_some(Reuse::Live { len: common });
        let snap = self
            .snapshots
            .iter()
            .enumerate()
            .rev()
            .find(|(_, s)| s.len <= common && ok(s.len) && (!aligned || s.aligned))
            .map(|(index, s)| Reuse::Snapshot { index, len: s.len });
        match (live, snap) {
            (Some(l), Some(s)) if s.tokens() > l.tokens() => Some(s),
            (Some(l), _) => Some(l),
            (None, s) => s,
        }
    }

    /// Returns to snapshot `index`, forgetting what came after it.
    pub fn restore(&mut self, index: usize) {
        let snap = self.snapshots[index].clone();
        for (c, p) in self.cache.iter_mut().zip(&snap.passes) {
            c.restore(p);
        }
        if let (Some(d), Some(sd)) = (self.draft.as_mut(), snap.draft.as_ref()) {
            d.restore(sd);
        }
        self.tokens.truncate(snap.len);
        self.snapshots.truncate(index + 1);
        self.pure = snap.aligned;
    }

    /// Returns to the latest snapshot, if any (after a failed request: the
    /// caches may hold half of a chunk).
    pub fn recover(&mut self) -> bool {
        match self.snapshots.len() {
            0 => false,
            n => {
                self.restore(n - 1);
                true
            }
        }
    }

    /// Whether anything after snapshot `index` is worth keeping apart: a
    /// later prompt (a new user turn), not just a generated answer.
    pub fn branches_after(&self, index: usize) -> bool {
        self.snapshots[index + 1..].iter().any(|s| s.prompt_end)
    }

    /// A new session at snapshot `index`; this one is left as it is. The
    /// caches are shared until one side writes; their capacity is cut to the
    /// prefix so that write copies only the prefix.
    pub fn fork(&self, index: usize) -> Result<Session<'m>> {
        let mut s = self.clone();
        s.restore(index);
        for c in &mut s.cache {
            c.shrink()?;
        }
        if let Some(d) = s.draft.as_mut() {
            d.shrink()?;
        }
        s.origin = None;
        Ok(s)
    }

    /// Memory held by the caches and the snapshots (an upper bound when
    /// arrays are shared).
    pub fn nbytes(&self) -> usize {
        self.cache.iter().map(|c| c.nbytes()).sum::<usize>()
            + self.draft.as_ref().map_or(0, |d| d.nbytes())
            + self.snapshots.iter().map(|s| s.nbytes()).sum::<usize>()
    }

    /// Drops the oldest periodic snapshot, or else the oldest prompt-end one
    /// but the last. Returns false if none can go.
    pub fn drop_snapshot(&mut self) -> bool {
        let n = self.snapshots.len();
        let victim = self
            .snapshots
            .iter()
            .position(|s| !s.prompt_end)
            .or_else(|| (n > 1).then_some(0));
        match victim {
            Some(i) => {
                self.snapshots.remove(i);
                true
            }
            None => false,
        }
    }

    fn capture_ids(&self) -> &[usize] {
        self.drafter.map_or(&[], |d| d.target_layer_ids.as_slice())
    }

    /// Runs the target on `tokens` (1, L), keeping the drafter's context in
    /// step with the accepted positions.
    fn forward(&mut self, tokens: &Array, keep_windows: bool) -> Result<(Array, Option<Array>)> {
        let ids = self.capture_ids().to_vec();
        self.model
            .forward_ex(tokens, &mut self.cache, &ids, keep_windows)
    }

    fn add_context(&mut self, hidden: Option<&Array>) -> Result<()> {
        if let (Some(d), Some(state), Some(h)) = (self.drafter, self.draft.as_mut(), hidden) {
            d.add_context(state, h)?;
        }
        Ok(())
    }

    fn eval_state(&self) -> Result<()> {
        let mut arrays: Vec<&Array> = self.cache.iter().flat_map(|c| c.arrays()).collect();
        if let Some(d) = &self.draft {
            arrays.extend(d.arrays());
        }
        ops::eval(&arrays)
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
        anyhow::ensure!(self.prefill_step > 0, "prefill step must be positive");
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
            let (_, hidden) = self.forward(&chunk, false)?;
            self.add_context(hidden.as_ref())?;
            self.eval_state()?;
            let peak_bytes = mlx_rs::memory::peak_memory()?;
            let before = self.tokens.len();
            self.tokens.extend_from_slice(&tokens[done..done + n]);
            self.pure &= n == self.prefill_step;
            done += n;
            if self.tokens.len() / SNAPSHOT_EVERY > before / SNAPSHOT_EVERY {
                self.take_snapshot(false);
            }
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
        if ids.len() > 1 {
            // End of a prompt: a retry of it only recomputes the last token.
            self.take_snapshot(true);
        }
        let last = Array::from_slice(&ids[done..], &[1, (ids.len() - done) as i32]);
        let (logits, hidden) = self.forward(&last, false)?;
        self.add_context(hidden.as_ref())?;
        let l = logits.shape()[1];
        let logits = ops::squeeze(&ops::slice(&logits, 1, l - 1, l, 1)?, &[1])?;
        let lp = ops::sub(&logits, &ops::logsumexp(&logits, &[0, 1], true)?)?;
        self.tokens.extend_from_slice(&tokens[done..]);
        self.pure = false;
        Ok(lp)
    }

    /// One speculative round (M-7): drafts up to `block - 1` tokens after
    /// `last` (sampled, not yet fed), verifies them with one target forward
    /// and keeps the accepted prefix. A stop token among the accepted ends
    /// the round there, so the session never holds tokens past it.
    pub fn speculate(
        &mut self,
        last: u32,
        block: usize,
        sampler: &mut Sampler,
        stop: &[u32],
    ) -> Result<Speculated> {
        let drafter = self.drafter.context("speculation needs a drafter")?;
        let state = self.draft.as_ref().context("speculation needs a drafter")?;
        let drafts = drafter.draft(self.model, state, last, block.saturating_sub(1).max(1))?;
        let mut input: Vec<i32> = vec![last as i32];
        input.extend(drafts.iter().map(|&t| t as i32));
        let l = input.len() as i32;
        let (logits, hidden) = self.forward(&Array::from_slice(&input, &[1, l]), true)?;
        let logits = ops::squeeze(&logits, &[0])?;
        let lp = ops::sub(&logits, &ops::logsumexp(&logits, &[-1], true)?)?;
        let rows = sampler.verify(&lp, &drafts)?;
        let mut accepted = rows.accepted;
        let mut next = rows.next;
        if let Some(j) = drafts[..accepted].iter().position(|t| stop.contains(t)) {
            accepted = j;
            next = drafts[j];
        }
        let keep = accepted as i32 + 1;
        for c in &mut self.cache {
            c.rewind(l, keep)?;
        }
        if let Some(h) = &hidden {
            let h = ops::slice(h, 1, 0, keep, 1)?;
            self.add_context(Some(&h))?;
        }
        self.eval_state()?;
        self.tokens.push(last);
        self.tokens.extend_from_slice(&drafts[..accepted]);
        self.pure = false;
        let mut tokens = drafts[..accepted].to_vec();
        tokens.push(next);
        Ok(Speculated {
            tokens,
            drafted: drafts.len(),
            accepted,
        })
    }
}

/// Picks how many tokens to draft next from recent acceptance (M-7): the
/// policy mlx-vlm uses for DFlash drafters, within the trained block.
pub struct DraftPolicy {
    history: std::collections::VecDeque<(usize, usize)>,
    ceiling: usize,
    /// Measured speeds (tokens per second, smoothed) with and without
    /// drafting: drafting stops when it does not pay (long contexts, where
    /// verifying a block costs more), probing again now and then.
    spec_tps: Option<f64>,
    plain_tps: Option<f64>,
    decisions: usize,
}

impl DraftPolicy {
    /// Shortest block the policy backs off to.
    const MIN_BLOCK: usize = 4;

    pub fn new(block_size: usize) -> Self {
        Self {
            history: Default::default(),
            ceiling: block_size,
            spec_tps: None,
            plain_tps: None,
            decisions: 0,
        }
    }

    /// Every this many decisions the slower mode is tried again.
    const PROBE_EVERY: usize = 16;

    /// Time of one round (speculative) or one plain step.
    pub fn record_time(&mut self, tokens: usize, seconds: f64, speculative: bool) {
        let tps = tokens as f64 / seconds.max(1e-6);
        let slot = if speculative {
            &mut self.spec_tps
        } else {
            &mut self.plain_tps
        };
        *slot = Some(slot.map_or(tps, |old| 0.7 * old + 0.3 * tps));
    }

    pub fn record(&mut self, accepted: usize, drafted: usize) {
        self.history.push_back((accepted, drafted));
        if self.history.len() > 8 {
            self.history.pop_front();
        }
    }

    /// Block size (anchor plus drafts) for the next round, given the tokens
    /// still allowed. 1 means: decode without drafting.
    pub fn next(&mut self, remaining: usize) -> usize {
        let total = self.ceiling.min(remaining + 1);
        if total <= 1 {
            return total;
        }
        self.decisions += 1;
        let probe = self.decisions.is_multiple_of(Self::PROBE_EVERY);
        match (self.spec_tps, self.plain_tps) {
            // Measure plain decoding once drafting has a few rounds.
            (Some(_), None) if self.decisions >= 4 => return 1,
            (Some(s), Some(p)) if s < p && !probe => return 1,
            _ => {}
        }
        let recent: Vec<_> = self.history.iter().filter(|(_, d)| *d > 0).collect();
        let Some(&&(_, last_drafted)) = recent.last() else {
            return total;
        };
        let current = total.min((last_drafted + 1).max(2));
        let min_total = total.min(Self::MIN_BLOCK.max(2));
        let drafted: usize = recent.iter().map(|(_, d)| d).sum();
        let accepted: usize = recent.iter().map(|(a, _)| a).sum();
        let rate = accepted as f64 / drafted as f64;
        let mean = accepted as f64 / recent.len() as f64;
        if rate < 0.30 || mean < 2.0 {
            return if current >= 8 {
                min_total.max(total.min(current / 2))
            } else {
                min_total.max(total.min(current.saturating_sub(2)))
            };
        }
        if rate < 0.50 {
            return min_total.max(total.min(current.saturating_sub(2)));
        }
        let full = recent.iter().filter(|(a, d)| a >= d).count() as f64 / recent.len() as f64;
        if rate >= 0.85 && full >= 0.75 {
            return total.min(current + 2);
        }
        current
    }
}

/// Verification of a block of drafts against the target.
pub struct Verified {
    /// Leading drafts accepted.
    pub accepted: usize,
    /// The target's token after them.
    pub next: u32,
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

    /// The sampling distribution (unnormalized) after temperature and top_p.
    fn probs(&self, lp: &[f32]) -> Vec<(u32, f64)> {
        let t = self.temperature as f64;
        let max = lp.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut probs: Vec<(u32, f64)> = lp
            .iter()
            .enumerate()
            .map(|(i, &x)| (i as u32, ((x as f64 - max) / t).exp()))
            // Relative to the best token; what is dropped cannot change the
            // draw, and top_p then sorts thousands instead of the vocabulary.
            .filter(|&(_, p)| p > 1e-12)
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
        probs
    }

    fn draw(&mut self, probs: &[(u32, f64)]) -> u32 {
        let total: f64 = probs.iter().map(|p| p.1).sum();
        let mut r = self.next_f64() * total;
        for &(id, p) in probs {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        probs.last().map_or(0, |p| p.0)
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
        let probs = self.probs(lp.as_slice::<f32>());
        Ok(self.draw(&probs))
    }

    /// Checks greedy drafts (one per row but the last of `logprobs`, (L, V))
    /// against the target. Greedy: a draft stays if it is the argmax. With
    /// temperature: a draft `d` stays with probability p(d) and, if it does
    /// not, the token comes from p without `d`; this samples exactly from the
    /// target distribution.
    pub fn verify(&mut self, logprobs: &Array, drafts: &[u32]) -> Result<Verified> {
        let rows = logprobs.shape()[0] as usize;
        anyhow::ensure!(
            rows == drafts.len() + 1,
            "{rows} rows for {} drafts",
            drafts.len()
        );
        if self.temperature <= 0.0 {
            let y = ops::astype(&ops::argmax(logprobs, -1)?, Dtype::Uint32)?;
            ops::eval(&[&y])?;
            let target = y.as_slice::<u32>();
            let accepted = drafts
                .iter()
                .zip(target)
                .take_while(|(d, t)| d == t)
                .count();
            return Ok(Verified {
                accepted,
                next: target[accepted],
            });
        }
        let lp = ops::f32(logprobs)?;
        ops::eval(&[&lp])?;
        let all = lp.as_slice::<f32>();
        let v = all.len() / rows;
        for (i, &d) in drafts.iter().enumerate() {
            let mut probs = self.probs(&all[i * v..(i + 1) * v]);
            let total: f64 = probs.iter().map(|p| p.1).sum();
            let pd = probs.iter().find(|p| p.0 == d).map_or(0.0, |p| p.1) / total;
            if self.next_f64() < pd {
                continue;
            }
            probs.retain(|p| p.0 != d);
            let next = if probs.is_empty() {
                d
            } else {
                self.draw(&probs)
            };
            return Ok(Verified { accepted: i, next });
        }
        let probs = self.probs(&all[drafts.len() * v..]);
        Ok(Verified {
            accepted: drafts.len(),
            next: self.draw(&probs),
        })
    }
}

/// Version of the on-disk session format (C-4).
pub const SAVE_FORMAT: &str = "calamardo-session-2";

/// A saved session that does not match this model or cache format.
#[derive(Debug)]
pub struct Incompatible(pub String);

impl std::fmt::Display for Incompatible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "incompatible saved session: {}", self.0)
    }
}

impl std::error::Error for Incompatible {}

fn fsync(path: &std::path::Path) -> Result<()> {
    std::fs::File::open(path)?.sync_all()?;
    Ok(())
}

impl<'m> Session<'m> {
    /// Saves the live state, the tokens and the latest prompt-end snapshot
    /// as one safetensors file. `manifest` (model, cache format...) goes into
    /// its metadata and must match on load. Written to a temporary file,
    /// synced, then renamed, so a crash or power loss never leaves half a
    /// file under `path`.
    pub fn save(
        &self,
        path: &std::path::Path,
        manifest: &std::collections::BTreeMap<String, String>,
    ) -> Result<()> {
        use crate::cache::Saved;
        anyhow::ensure!(!self.tokens.is_empty(), "nothing to save");
        let mut out = Saved::default();
        out.meta.extend(manifest.clone());
        out.meta.insert("format".into(), SAVE_FORMAT.into());
        out.meta.insert("pure".into(), self.pure.to_string());
        out.arrays.push((
            "tokens".into(),
            Array::from_slice(&self.tokens, &[self.tokens.len() as i32]),
        ));
        for (i, c) in self.cache.iter().enumerate() {
            c.export(&format!("live.{i}"), &mut out);
        }
        if let Some(d) = &self.draft {
            d.export("draft", &mut out);
        }
        if let Some(s) = self.snapshots.iter().rev().find(|s| s.prompt_end) {
            out.meta.insert("snap.len".into(), s.len.to_string());
            out.meta
                .insert("snap.aligned".into(), s.aligned.to_string());
            for (i, p) in s.passes.iter().enumerate() {
                p.export(&format!("snap.{i}"), &mut out);
            }
            if let Some(d) = &s.draft {
                d.export("snap.draft", &mut out);
            }
        }
        // MLX appends ".safetensors" to names that lack it.
        let tmp = path.with_extension("tmp.safetensors");
        let result = (|| -> Result<()> {
            ops::save_safetensors(&tmp, &out.arrays, &out.meta)?;
            fsync(&tmp)?;
            std::fs::rename(&tmp, path)?;
            if let Some(dir) = path.parent() {
                fsync(dir)?;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
    }

    /// Loads a session saved by `save`, checking `manifest`, the format and
    /// that every cache holds exactly the saved tokens.
    pub fn load(
        model: &'m Model,
        drafter: Option<&'m Drafter>,
        kv: KvFormat,
        path: &std::path::Path,
        manifest: &std::collections::BTreeMap<String, String>,
    ) -> Result<Session<'m>> {
        use crate::cache::{Loaded, PassSnapshot};
        let (arrays, meta) = ops::load_safetensors_with_metadata(path)?;
        if meta.get("format").map(String::as_str) != Some(SAVE_FORMAT) {
            return Err(Incompatible(format!("format {:?}", meta.get("format"))).into());
        }
        for (k, v) in manifest {
            if meta.get(k) != Some(v) {
                return Err(
                    Incompatible(format!("{k} is {:?}, expected {v:?}", meta.get(k))).into(),
                );
            }
        }
        let mut l = Loaded {
            arrays: arrays.into_iter().collect(),
            meta,
        };
        let mut s = Session::new(model, kv);
        if let Some(d) = drafter {
            s = s.with_drafter(d);
        }
        let tokens = l.array("tokens")?;
        ops::eval(&[&tokens])?;
        s.tokens = tokens.as_slice::<u32>().to_vec();
        s.pure = l.meta.get("pure").is_some_and(|v| v == "true");
        for (i, c) in s.cache.iter_mut().enumerate() {
            c.import(&format!("live.{i}"), &mut l)?;
            c.validate(s.tokens.len())?;
        }
        if let Some(d) = s.draft.as_mut() {
            d.import("draft", &mut l)?;
            anyhow::ensure!(
                d.len as usize == s.tokens.len(),
                "drafter context does not match the tokens"
            );
        }
        if let Some(len) = l.meta.get("snap.len").and_then(|v| v.parse::<usize>().ok()) {
            anyhow::ensure!(len < s.tokens.len(), "snapshot beyond the tokens");
            let passes = s
                .cache
                .iter()
                .enumerate()
                .map(|(i, c)| PassSnapshot::import(c, &format!("snap.{i}"), &mut l))
                .collect::<Result<Vec<_>>>()?;
            let draft = match &s.draft {
                Some(d) => Some(DraftSnapshot::import(d, "snap.draft", &mut l)?),
                None => None,
            };
            s.snapshots.push(Snapshot {
                len,
                prompt_end: true,
                aligned: l.meta.get("snap.aligned").is_some_and(|v| v == "true"),
                passes,
                draft,
            });
        }
        // Read everything now, not on the first request that touches it.
        let mut all: Vec<&Array> = s.cache.iter().flat_map(|c| c.arrays()).collect();
        if let Some(d) = &s.draft {
            all.extend(d.arrays());
        }
        let snap_arrays: Vec<Array> = s.snapshots.iter().flat_map(|sn| sn.arrays()).collect();
        all.extend(snap_arrays.iter());
        ops::eval(&all)?;
        s.origin = Some(path.to_path_buf());
        Ok(s)
    }
}
