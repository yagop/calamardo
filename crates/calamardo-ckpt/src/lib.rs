//! Inkling 2 Mini checkpoint downloaded from Hugging Face: tensor index,
//! configuration, quantization formats and pass order. Does not depend on MLX.

pub mod config;
pub mod hub;
pub mod quant;
pub mod safetensors;
pub mod schedule;

pub use config::ModelConfig;
pub use hub::{Checkpoint, default_hub_cache, snapshot_dir};
pub use quant::{ExpertFormat, QuantConfig};
pub use safetensors::{Dtype, TensorInfo};

/// Official checkpoint repository.
pub const DEFAULT_REPO: &str = "thinkingmachines/Inkling-2-Mini";

/// Tensor prefixes the first version does not load.
pub const SKIPPED_PREFIXES: &[&str] = &["model.visual."];

pub fn is_skipped(name: &str) -> bool {
    SKIPPED_PREFIXES.iter().any(|p| name.starts_with(p))
}
