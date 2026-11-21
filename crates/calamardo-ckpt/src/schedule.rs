//! Per-token pass order (requirements M-1 and M-2).
//!
//! Layers `0..core_start` run once, the core `core_start..core_end` repeats
//! `num_passes` times back to back, and `core_end..num_hidden_layers` runs
//! once. Every pass has its own KV cache.

use crate::config::TextConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnKind {
    /// Attention over the full context: configurable 4, 8 or 16-bit cache.
    Global,
    /// Sliding window: bf16 cache of `sliding_window_size` tokens.
    Window,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pass {
    /// Pass index, `0..48`. Also the index of its KV cache.
    pub index: usize,
    /// Physical layer whose weights are used.
    pub layer: usize,
    /// Core cycle (`0..num_passes`), or `None` outside the core.
    pub cycle: Option<usize>,
    pub attn: AttnKind,
}

pub fn passes(cfg: &TextConfig) -> Vec<Pass> {
    let l = &cfg.loop_config;
    let mut layers: Vec<(usize, Option<usize>)> = (0..l.core_start).map(|i| (i, None)).collect();
    for c in 0..l.num_passes {
        layers.extend((l.core_start..l.core_end).map(|i| (i, Some(c))));
    }
    layers.extend((l.core_end..cfg.num_hidden_layers).map(|i| (i, None)));
    layers
        .into_iter()
        .enumerate()
        .map(|(index, (layer, cycle))| Pass {
            index,
            layer,
            cycle,
            attn: if cfg.is_global_layer(layer) {
                AttnKind::Global
            } else {
                AttnKind::Window
            },
        })
        .collect()
}

/// KV cache bytes per token for one global pass (K and V).
///
/// `bits == 16` is bf16. With 4 or 8 bits it is MLX affine quantization:
/// packed values plus a bf16 scale and bias per group.
pub fn global_kv_bytes_per_token(cfg: &TextConfig, bits: usize, group_size: usize) -> usize {
    let per_head = match bits {
        16 => cfg.head_dim * 2,
        4 | 8 => {
            assert!(
                cfg.head_dim.is_multiple_of(group_size),
                "head_dim is not a multiple of the group size"
            );
            cfg.head_dim * bits / 8 + (cfg.head_dim / group_size) * 2 * 2
        }
        _ => panic!("unsupported KV cache bits: {bits}"),
    };
    2 * cfg.num_key_value_heads * per_head
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    /// The official `config.json`, without `vision_config`.
    pub(crate) fn inkling_config() -> ModelConfig {
        serde_json::from_str(include_str!("../tests/data/config.json")).unwrap()
    }

    #[test]
    fn forty_eight_passes_eight_global() {
        let cfg = inkling_config();
        cfg.validate().unwrap();
        let p = passes(&cfg.text_config);
        assert_eq!(p.len(), 48);
        let globals: Vec<_> = p
            .iter()
            .filter(|p| p.attn == AttnKind::Global)
            .map(|p| (p.index, p.layer))
            .collect();
        assert_eq!(
            globals,
            [
                (5, 5),
                (11, 11),
                (17, 17),
                (23, 11),
                (29, 17),
                (35, 11),
                (41, 17),
                (47, 23)
            ]
        );
        // The 6..18 core repeats three times back to back.
        assert_eq!(p[18].layer, 6);
        assert_eq!(p[18].cycle, Some(1));
        assert_eq!(p[42].layer, 18);
        assert_eq!(p[42].cycle, None);
    }

    #[test]
    fn bytes_per_token_match_spec() {
        let t = inkling_config().text_config;
        let g = passes(&t)
            .iter()
            .filter(|p| p.attn == AttnKind::Global)
            .count();
        assert_eq!(g * global_kv_bytes_per_token(&t, 16, 64), 32_768);
        assert_eq!(g * global_kv_bytes_per_token(&t, 8, 64), 17_408);
        assert_eq!(g * global_kv_bytes_per_token(&t, 4, 64), 9_216);
        let windows = 48 - g;
        let window_bytes = windows * t.sliding_window_size * global_kv_bytes_per_token(&t, 16, 64);
        assert_eq!(window_bytes, 167_772_160);
    }
}
