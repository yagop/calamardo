//! The checkpoint's `config.json` (`Inkling2LoopForConditionalGeneration`).
//!
//! Only the fields the runtime uses are modeled; serde ignores the rest
//! (e.g. `vision_config`, out of scope for the first version).

use anyhow::{Result, ensure};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub eos_token_id: u32,
    pub text_config: TextConfig,
    pub mtp_config: MtpConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    pub is_varloop: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub swa_head_dim: usize,
    pub swa_num_attention_heads: usize,
    pub swa_num_key_value_heads: usize,
    pub local_layer_ids: Vec<usize>,
    pub sliding_window_size: usize,
    pub loop_config: LoopConfig,
    pub n_layers_per_attn_res_block: usize,
    pub attn_res_normalize_values: bool,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub route_scale: f32,
    pub use_gate_bias: bool,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub rel_l2_clip_threshold: f32,
    pub qk_l2_clip_threshold: f32,
    pub log_scaling_alpha: f32,
    pub log_scaling_n_floor: usize,
    pub sconv_kernel_size: usize,
    pub use_value_norm_no_gamma: bool,
    pub final_logit_softcapping: f32,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
}

/// DSpark draft model (MTP) configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct MtpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub local_layer_ids: Vec<usize>,
    pub sliding_window_size: usize,
    pub block_size: usize,
    pub markov_rank: usize,
    pub markov_head_type: String,
    /// Pass indices (not physical layers) whose hidden states the drafter reads.
    pub target_layer_ids: Vec<usize>,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl TextConfig {
    pub fn is_global_layer(&self, layer: usize) -> bool {
        !self.local_layer_ids.contains(&layer)
    }
}

impl ModelConfig {
    /// Checks what the runtime takes for granted (requirements M-1 to M-3).
    pub fn validate(&self) -> Result<()> {
        let t = &self.text_config;
        let l = &t.loop_config;
        ensure!(
            self.model_type == "inkling2_loop" && t.model_type == "inkling2_loop_text",
            "unexpected model_type: {}",
            self.model_type
        );
        ensure!(!l.is_varloop, "variable-length loop is not supported");
        ensure!(
            l.core_start < l.core_end && l.core_end <= t.num_hidden_layers && l.num_passes >= 1,
            "inconsistent loop_config: {l:?}"
        );
        ensure!(
            t.num_attention_heads == t.swa_num_attention_heads
                && t.num_key_value_heads == t.swa_num_key_value_heads
                && t.head_dim == t.swa_head_dim,
            "global and sliding-window layers must share the attention shape"
        );
        ensure!(
            t.num_attention_heads.is_multiple_of(t.num_key_value_heads),
            "GQA: {} query heads are not a multiple of {} KV heads",
            t.num_attention_heads,
            t.num_key_value_heads
        );
        ensure!(t.n_shared_experts == 0, "shared experts are not supported");
        Ok(())
    }
}
