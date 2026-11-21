//! Inkling 2's bundled DSpark drafter (M-7): three layers with DFlare fusion
//! of the target's hidden states and an RNN ("Markov") head. Ported op by op
//! from mlx-vlm `speculative/drafters/inkling2_dspark/dspark.py` (itself from
//! vLLM).
//!
//! The drafter proposes up to `block_size - 1` tokens in one forward: the
//! last accepted token followed by mask embeddings attends to the context
//! (projections of the target's hidden states at passes 5, 23, 41 and the
//! final output) and to the whole block.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use calamardo_ckpt::ModelConfig;
use mlx_rs::{Array, Dtype};

use crate::{
    cache::{KvCache, RotatingKvCache},
    model::Model,
    ops::{self, f32s},
};

struct Layer {
    input_layernorm: Array,
    post_attention_layernorm: Array,
    dflare_norm: Array,
    q_proj: Array,
    k_proj: Array,
    v_proj: Array,
    k_proj_ctx: Array,
    v_proj_ctx: Array,
    o_proj: Array,
    q_norm: Array,
    k_norm: Array,
    k_norm_ctx: Array,
    gate_proj: Array,
    up_proj: Array,
    down_proj: Array,
    /// Sliding window over the context, or `None` for the global layer.
    window: Option<i32>,
}

/// The drafter's own context cache, one per layer. Only target context goes
/// in; proposals never do.
#[derive(Clone)]
pub enum DraftCache {
    Global(KvCache),
    Window(RotatingKvCache),
}

/// Per-conversation drafter state.
#[derive(Clone)]
pub struct DraftState {
    pub caches: Vec<DraftCache>,
    /// Context positions added so far.
    pub len: i32,
    /// Hidden states (1, 1, 4 * D) of the last context position.
    pub last_hidden: Option<Array>,
}

pub struct Drafter {
    layers: Vec<Layer>,
    norm: Array,
    markov_w1: Array,
    rnn_w: Array,
    rnn_b: Array,
    markov_norm: Array,
    /// `lm_head @ hidden_out` folded once (V, rank) and its bias (V,).
    fold_w: Array,
    fold_b: Array,
    /// softmax(fuse_weight) rows, one per layer plus the Markov head.
    alpha: Array,
    mask_embed: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rope_theta: f32,
    eps: f32,
    rank: i32,
    hidden: i32,
    /// Trained block: anchor plus up to `block_size - 1` proposals.
    pub block_size: usize,
    /// Pass indices whose hidden states the drafter reads.
    pub target_layer_ids: Vec<usize>,
    unpadded_vocab: i32,
    softcap: f32,
}

fn heads(x: &Array, n: i32, head_dim: i32) -> Result<Array> {
    let [b, l, _] = x.shape()[..] else {
        unreachable!()
    };
    ops::transpose(&ops::reshape(x, &[b, l, n, head_dim])?, &[0, 2, 1, 3])
}

impl Drafter {
    /// Loads `mtp.safetensors` from the snapshot and folds the Markov head
    /// into the target's vocabulary projection.
    pub fn load(dir: &Path, target: &Model) -> Result<Self> {
        let cfg: ModelConfig = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let m = &cfg.mtp_config;
        ensure!(
            m.markov_head_type == "rnn_hidden",
            "unsupported Markov head {}",
            m.markov_head_type
        );
        let mut w: std::collections::HashMap<String, Array> =
            ops::load_safetensors(&dir.join("mtp.safetensors"))?
                .into_iter()
                .collect();
        let mut take = |k: &str| {
            w.remove(k)
                .with_context(|| format!("mtp.safetensors lacks {k}"))
        };
        let mut layers = Vec::new();
        for i in 0..m.num_hidden_layers {
            let p = format!("layers.{i}");
            let a = format!("{p}.self_attn");
            layers.push(Layer {
                input_layernorm: take(&format!("{p}.input_layernorm.weight"))?,
                post_attention_layernorm: take(&format!("{p}.post_attention_layernorm.weight"))?,
                dflare_norm: take(&format!("{p}.dflare_norm.weight"))?,
                q_proj: take(&format!("{a}.q_proj.weight"))?,
                k_proj: take(&format!("{a}.k_proj.weight"))?,
                v_proj: take(&format!("{a}.v_proj.weight"))?,
                k_proj_ctx: take(&format!("{a}.k_proj_ctx.weight"))?,
                v_proj_ctx: take(&format!("{a}.v_proj_ctx.weight"))?,
                o_proj: take(&format!("{a}.o_proj.weight"))?,
                q_norm: take(&format!("{a}.q_norm.weight"))?,
                k_norm: take(&format!("{a}.k_norm.weight"))?,
                k_norm_ctx: take(&format!("{a}.k_norm_ctx.weight"))?,
                gate_proj: take(&format!("{p}.mlp.gate_proj.weight"))?,
                up_proj: take(&format!("{p}.mlp.up_proj.weight"))?,
                down_proj: take(&format!("{p}.mlp.down_proj.weight"))?,
                window: m
                    .local_layer_ids
                    .contains(&i)
                    .then_some(m.sliding_window_size as i32),
            });
        }
        let hidden_out = take("markov_head.hidden_out.weight")?;
        let hidden_out_b = take("markov_head.hidden_out.bias")?;
        // The reference folds in float64 on the CPU; float32 on the GPU gives
        // the same bf16 values but for rare ties of the last rounding.
        let head = ops::f32(target.lm_head())?;
        let fold_w = ops::astype(
            &ops::matmul(&head, &ops::f32(&hidden_out)?)?,
            hidden_out.dtype(),
        )?;
        let fold_b = ops::astype(
            &ops::squeeze(
                &ops::matmul(&head, &ops::reshape(&ops::f32(&hidden_out_b)?, &[-1, 1])?)?,
                &[1],
            )?,
            hidden_out.dtype(),
        )?;
        let fuse = take("fuse_weight")?;
        let alpha = ops::astype(
            &ops::softmax(&ops::f32(&fuse)?, &[-1], false)?,
            fuse.dtype(),
        )?;
        let d = Self {
            norm: take("norm.weight")?,
            markov_w1: take("markov_head.markov_w1.weight")?,
            rnn_w: take("markov_head.rnn_proj.weight")?,
            rnn_b: take("markov_head.rnn_proj.bias")?,
            markov_norm: take("markov_head.dflare_norm.weight")?,
            mask_embed: take("mask_embed")?,
            layers,
            fold_w,
            fold_b,
            alpha,
            heads: m.num_attention_heads as i32,
            kv_heads: m.num_key_value_heads as i32,
            head_dim: m.head_dim as i32,
            rope_theta: m.rope_theta,
            eps: m.rms_norm_eps,
            rank: m.markov_rank as i32,
            hidden: m.hidden_size as i32,
            block_size: m.block_size,
            target_layer_ids: m.target_layer_ids.clone(),
            unpadded_vocab: cfg.text_config.unpadded_vocab_size as i32,
            softcap: cfg.text_config.final_logit_softcapping,
        };
        let mut all = vec![
            &d.norm,
            &d.markov_w1,
            &d.rnn_w,
            &d.rnn_b,
            &d.markov_norm,
            &d.mask_embed,
            &d.fold_w,
            &d.fold_b,
            &d.alpha,
        ];
        for l in &d.layers {
            all.extend([
                &l.input_layernorm,
                &l.post_attention_layernorm,
                &l.dflare_norm,
                &l.q_proj,
                &l.k_proj,
                &l.v_proj,
                &l.k_proj_ctx,
                &l.v_proj_ctx,
                &l.o_proj,
                &l.q_norm,
                &l.k_norm,
                &l.k_norm_ctx,
                &l.gate_proj,
                &l.up_proj,
                &l.down_proj,
            ]);
        }
        ops::eval(&all)?;
        Ok(d)
    }

    pub fn new_state(&self) -> DraftState {
        DraftState {
            caches: self
                .layers
                .iter()
                .map(|l| match l.window {
                    Some(w) => DraftCache::Window(RotatingKvCache::new(w)),
                    None => DraftCache::Global(KvCache::default()),
                })
                .collect(),
            len: 0,
            last_hidden: None,
        }
    }

    /// `mix`: weighted sum of the target hidden states (B, L, 4 * D), then
    /// RMS norm with `norm`.
    fn mix(&self, hidden: &Array, row: i32, norm: &Array) -> Result<Array> {
        let [b, l, _] = hidden.shape()[..] else {
            unreachable!()
        };
        let features = ops::f32(&ops::reshape(hidden, &[b, l, -1, self.hidden])?)?;
        let n = features.shape()[2];
        let alpha = ops::squeeze(&ops::slice(&self.alpha, 0, row, row + 1, 1)?, &[0])?;
        let mut mixed = ops::zeros(&[b, l, self.hidden], Dtype::Float32)?;
        for i in 0..n {
            let f = ops::squeeze(&ops::slice(&features, 2, i, i + 1, 1)?, &[2])?;
            let a = ops::f32(&ops::slice(&alpha, 0, i, i + 1, 1)?)?;
            mixed = ops::add(&mixed, &ops::mul(&f, &a)?)?;
        }
        ops::astype(
            &ops::rms_norm(&mixed, Some(norm), self.eps)?,
            hidden.dtype(),
        )
    }

    /// Adds target hidden states (1, L, 4 * D) of accepted positions to the
    /// drafter's context.
    pub fn add_context(&self, state: &mut DraftState, hidden: &Array) -> Result<()> {
        let l = hidden.shape()[1];
        for (i, (layer, cache)) in self.layers.iter().zip(state.caches.iter_mut()).enumerate() {
            let mut ctx = self.mix(hidden, i as i32, &layer.dflare_norm)?;
            let mut offset = state.len;
            if let Some(w) = layer.window
                && l > w
            {
                // Only the last window of the context can be attended to.
                ctx = ops::slice(&ctx, 1, l - w, l, 1)?;
                offset += l - w;
            }
            let k = heads(
                &ops::linear(&ctx, &layer.k_proj_ctx)?,
                self.kv_heads,
                self.head_dim,
            )?;
            let k = ops::rms_norm(&k, Some(&layer.k_norm_ctx), self.eps)?;
            let k = ops::rope(&k, self.head_dim, self.rope_theta, offset)?;
            let v = heads(
                &ops::linear(&ctx, &layer.v_proj_ctx)?,
                self.kv_heads,
                self.head_dim,
            )?;
            match cache {
                DraftCache::Global(c) => {
                    c.update_and_fetch(&k, &v)?;
                }
                DraftCache::Window(c) => {
                    if c.offset < offset {
                        // Keep positions aligned after skipping old context.
                        c.offset = offset;
                    }
                    c.update_and_fetch(&k, &v)?;
                }
            }
        }
        state.len += l;
        state.last_hidden = Some(ops::slice(hidden, 1, l - 1, l, 1)?);
        Ok(())
    }

    fn attention(
        &self,
        layer: &Layer,
        x: &Array,
        cache: &DraftCache,
        offset: i32,
    ) -> Result<Array> {
        let [b, l, _] = x.shape()[..] else {
            unreachable!()
        };
        let (ck, cv) = match cache {
            DraftCache::Global(c) => c.keys_and_values()?,
            DraftCache::Window(c) => {
                let (k, v) = c.temporal()?;
                let w = layer.window.unwrap();
                let n = k.shape()[2];
                if n > w {
                    (
                        ops::slice(&k, 2, n - w, n, 1)?,
                        ops::slice(&v, 2, n - w, n, 1)?,
                    )
                } else {
                    (k, v)
                }
            }
        };
        let q = ops::rms_norm(
            &heads(&ops::linear(x, &layer.q_proj)?, self.heads, self.head_dim)?,
            Some(&layer.q_norm),
            self.eps,
        )?;
        let bk = ops::rms_norm(
            &heads(
                &ops::linear(x, &layer.k_proj)?,
                self.kv_heads,
                self.head_dim,
            )?,
            Some(&layer.k_norm),
            self.eps,
        )?;
        let k = ops::concatenate(
            &[
                &ck,
                &ops::rope(&bk, self.head_dim, self.rope_theta, offset)?,
            ],
            2,
        )?;
        let v = ops::concatenate(
            &[
                &cv,
                &heads(
                    &ops::linear(x, &layer.v_proj)?,
                    self.kv_heads,
                    self.head_dim,
                )?,
            ],
            2,
        )?;
        let q = ops::rope(&q, self.head_dim, self.rope_theta, offset)?;
        let out = ops::sdpa_nomask(&q, &k, &v, (self.head_dim as f32).powf(-0.5))?;
        let out = ops::reshape(&ops::transpose(&out, &[0, 2, 1, 3])?, &[b, l, -1])?;
        ops::linear(&out, &layer.o_proj)
    }

    /// Proposes `count` tokens after `anchor` (the last accepted token, not
    /// yet in the context), greedily.
    pub fn draft(
        &self,
        target: &Model,
        state: &DraftState,
        anchor: u32,
        count: usize,
    ) -> Result<Vec<u32>> {
        ensure!(
            count < self.block_size,
            "draft of {count} exceeds the trained block"
        );
        if count == 0 {
            return Ok(Vec::new());
        }
        let last = state
            .last_hidden
            .as_ref()
            .context("the drafter has no context")?;
        let embed = target.embed(&Array::from_slice(&[anchor as i32], &[1, 1]))?;
        let dtype = embed.dtype();
        let masks = ops::broadcast_to(
            &ops::astype(&self.mask_embed, dtype)?,
            &[1, self.block_size as i32 - 1, self.hidden],
        )?;
        // Bidirectional attention needs the full trained block.
        let mut x = ops::concatenate(&[&embed, &masks], 1)?;
        let mut residual: Option<Array> = None;
        for (layer, cache) in self.layers.iter().zip(&state.caches) {
            let summed = match &residual {
                None => x.clone(),
                Some(r) => ops::add(&ops::f32(&x)?, &ops::f32(r)?)?,
            };
            let res = ops::astype(&summed, dtype)?;
            let normed = ops::astype(
                &ops::rms_norm(&summed, Some(&layer.input_layernorm), self.eps)?,
                dtype,
            )?;
            let h = self.attention(layer, &normed, cache, state.len)?;
            let summed = ops::add(&ops::f32(&h)?, &ops::f32(&res)?)?;
            let normed = ops::astype(
                &ops::rms_norm(&summed, Some(&layer.post_attention_layernorm), self.eps)?,
                dtype,
            )?;
            let gate = ops::f32(&ops::linear(&normed, &layer.gate_proj)?)?;
            let up = ops::f32(&ops::linear(&normed, &layer.up_proj)?)?;
            let act = ops::mul(&ops::mul(&gate, &ops::sigmoid(&gate)?)?, &up)?;
            x = ops::linear(&ops::astype(&act, dtype)?, &layer.down_proj)?;
            residual = Some(ops::astype(&summed, dtype)?);
        }
        let x = ops::add(&ops::f32(&x)?, &ops::f32(residual.as_ref().unwrap())?)?;
        let x = ops::astype(&ops::rms_norm(&x, Some(&self.norm), self.eps)?, dtype)?;
        let x = ops::slice(&x, 1, 1, count as i32 + 1, 1)?;
        // Draft logits are bf16, unlike the target's float32 projection.
        let base = ops::astype(&ops::linear(&x, target.lm_head())?, dtype)?;

        // Markov head: the context part of the RNN projection, once per block.
        let r = self.rank;
        let f_rnn = self.mix(last, self.layers.len() as i32, &self.markov_norm)?;
        let inputs = ops::concatenate(&[&x, &ops::broadcast_to(&f_rnn, x.shape())?], -1)?;
        let w_ctx = ops::f32(&ops::slice(
            &self.rnn_w,
            1,
            2 * r,
            self.rnn_w.shape()[1],
            1,
        )?)?;
        let context = ops::add(
            &ops::matmul(&ops::f32(&inputs)?, &ops::t(&w_ctx)?)?,
            &ops::f32(&self.rnn_b)?,
        )?;
        let w_state = ops::f32(&ops::slice(&self.rnn_w, 1, 0, 2 * r, 1)?)?;
        let mut state_v = ops::zeros(&[1, r], dtype)?;
        let mut previous = Array::from_slice(&[anchor as i32], &[1]);
        let cap = f32s(self.softcap);
        let mut tokens = Vec::with_capacity(count);
        for i in 0..count as i32 {
            let emb = ops::take(&self.markov_w1, &previous, 0)?;
            let inp = ops::concatenate(&[&state_v, &emb], -1)?;
            let ctx_i = ops::squeeze(&ops::slice(&context, 1, i, i + 1, 1)?, &[1])?;
            let raw = ops::astype(
                &ops::add(&ops::matmul(&ops::f32(&inp)?, &ops::t(&w_state)?)?, &ctx_i)?,
                dtype,
            )?;
            let parts = ops::split_equal(&raw, 3, -1)?;
            let gate = ops::astype(&ops::sigmoid(&ops::f32(&parts[0])?)?, dtype)?;
            let cand = ops::astype(&ops::tanh(&ops::f32(&parts[1])?)?, dtype)?;
            let one = ops::scalar(1.0, dtype)?;
            state_v = ops::add(
                &ops::mul(&gate, &state_v)?,
                &ops::mul(&ops::sub(&one, &gate)?, &cand)?,
            )?;
            let latent = ops::astype(&ops::tanh(&ops::f32(&parts[2])?)?, dtype)?;
            let bias = ops::addmm(&self.fold_b, &latent, &ops::t(&self.fold_w)?)?;
            let logits = ops::add(
                &ops::squeeze(&ops::slice(&base, 1, i, i + 1, 1)?, &[1])?,
                &bias,
            )?;
            let logits = ops::astype(
                &ops::mul(&cap, &ops::tanh(&ops::div(&ops::f32(&logits)?, &cap)?)?)?,
                dtype,
            )?;
            let logits = ops::slice(&logits, -1, 0, self.unpadded_vocab, 1)?;
            previous = ops::argmax(&logits, -1)?;
            tokens.push(previous.clone());
        }
        let all = ops::concatenate(&tokens.iter().collect::<Vec<_>>(), 0)?;
        let all = ops::astype(&all, Dtype::Uint32)?;
        ops::eval(&[&all])?;
        Ok(all.as_slice::<u32>().to_vec())
    }
}

/// A point of the drafter's context to return to (same rules as the
/// target's snapshots: the global layer keeps a length, windows a handle).
#[derive(Clone)]
pub struct DraftSnapshot {
    caches: Vec<DraftSnapCache>,
    len: i32,
    last_hidden: Option<Array>,
}

#[derive(Clone)]
enum DraftSnapCache {
    Global(i32),
    Window(RotatingKvCache),
}

impl DraftState {
    pub fn snapshot(&self) -> DraftSnapshot {
        DraftSnapshot {
            caches: self
                .caches
                .iter()
                .map(|c| match c {
                    DraftCache::Global(c) => DraftSnapCache::Global(c.offset),
                    DraftCache::Window(c) => DraftSnapCache::Window(c.clone()),
                })
                .collect(),
            len: self.len,
            last_hidden: self.last_hidden.clone(),
        }
    }

    pub fn restore(&mut self, s: &DraftSnapshot) {
        for (c, sc) in self.caches.iter_mut().zip(&s.caches) {
            match (c, sc) {
                (DraftCache::Global(c), DraftSnapCache::Global(offset)) => c.truncate(*offset),
                (DraftCache::Window(c), DraftSnapCache::Window(w)) => *c = w.clone(),
                _ => panic!("draft snapshot from a different layout"),
            }
        }
        self.len = s.len;
        self.last_hidden = s.last_hidden.clone();
    }

    pub fn shrink(&mut self) -> Result<()> {
        for c in &mut self.caches {
            if let DraftCache::Global(c) = c {
                c.shrink()?;
            }
        }
        Ok(())
    }

    pub fn arrays(&self) -> Vec<&Array> {
        let mut out: Vec<&Array> = Vec::new();
        for c in &self.caches {
            match c {
                DraftCache::Global(c) => out.extend(c.arrays()),
                DraftCache::Window(c) => out.extend(c.arrays()),
            }
        }
        out.extend(self.last_hidden.iter());
        out
    }

    pub fn nbytes(&self) -> usize {
        self.arrays().iter().map(|a| a.nbytes()).sum()
    }

    pub fn export(&self, p: &str, out: &mut crate::cache::Saved) {
        for (i, c) in self.caches.iter().enumerate() {
            match c {
                DraftCache::Global(c) => c.export(&format!("{p}.{i}"), out),
                DraftCache::Window(c) => c.export(&format!("{p}.{i}"), out),
            }
        }
        out.meta.insert(format!("{p}.len"), self.len.to_string());
        if let Some(h) = &self.last_hidden {
            out.arrays.push((format!("{p}.last_hidden"), h.clone()));
        }
    }

    /// Fills a fresh state (from `Drafter::new_state`).
    pub fn import(&mut self, p: &str, l: &mut crate::cache::Loaded) -> Result<()> {
        for (i, c) in self.caches.iter_mut().enumerate() {
            match c {
                DraftCache::Global(c) => c.import(&format!("{p}.{i}"), l)?,
                DraftCache::Window(c) => c.import(&format!("{p}.{i}"), l)?,
            }
        }
        self.len = l.int(&format!("{p}.len"))?;
        self.last_hidden = l.opt(&format!("{p}.last_hidden"));
        Ok(())
    }
}

impl DraftSnapshot {
    pub fn nbytes(&self) -> usize {
        self.arrays().iter().map(|a| a.nbytes()).sum()
    }

    pub fn arrays(&self) -> Vec<Array> {
        let mut out: Vec<Array> = Vec::new();
        for c in &self.caches {
            if let DraftSnapCache::Window(c) = c {
                out.extend(c.arrays().into_iter().cloned());
            }
        }
        out.extend(self.last_hidden.iter().cloned());
        out
    }

    pub fn export(&self, p: &str, out: &mut crate::cache::Saved) {
        for (i, c) in self.caches.iter().enumerate() {
            match c {
                DraftSnapCache::Global(offset) => {
                    out.meta
                        .insert(format!("{p}.{i}.offset"), offset.to_string());
                }
                DraftSnapCache::Window(c) => c.export(&format!("{p}.{i}"), out),
            }
        }
        out.meta.insert(format!("{p}.len"), self.len.to_string());
        if let Some(h) = &self.last_hidden {
            out.arrays.push((format!("{p}.last_hidden"), h.clone()));
        }
    }

    /// Rebuilds a snapshot shaped like `like`.
    pub fn import(like: &DraftState, p: &str, l: &mut crate::cache::Loaded) -> Result<Self> {
        let caches = like
            .caches
            .iter()
            .enumerate()
            .map(|(i, c)| {
                Ok(match c {
                    DraftCache::Global(_) => {
                        DraftSnapCache::Global(l.int(&format!("{p}.{i}.offset"))?)
                    }
                    DraftCache::Window(w) => {
                        let mut w = RotatingKvCache::new(w.max_size());
                        w.import(&format!("{p}.{i}"), l)?;
                        DraftSnapCache::Window(w)
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            caches,
            len: l.int(&format!("{p}.len"))?,
            last_hidden: l.opt(&format!("{p}.last_hidden")),
        })
    }
}
