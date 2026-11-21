//! `hf_quant_config.json`: the format of every checkpoint tensor.
//!
//! MoE experts come in two formats (M-5):
//! - layers 0–19: nvfp4 (E2M1 packed in `u8`, E4M3 scale per block of 16,
//!   plus a per-expert outer scale `amax / outer_scale_denominator`);
//! - layers 20–23: mxfp8 (E4M3, UE8M0 scale per block of 32).
//!
//! Both declare `required_activation_quantization`: the model was trained
//! with the experts' input activations quantized as well.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::Deserialize;

use crate::{Dtype, TensorInfo};

#[derive(Clone, Debug, Deserialize)]
pub struct QuantConfig {
    pub quant_method: String,
    pub model: Section,
    pub mtp: Section,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Section {
    pub parameter_groups: Vec<ParamGroup>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ParamGroup {
    pub tensors: Vec<String>,
    pub compute_dtype: String,
    pub output_dtype: String,
    pub weight_transform: String,
    #[serde(default)]
    pub weight_block_shape: Vec<usize>,
    pub weight_scale_format: Option<String>,
    pub required_activation_quantization: Option<ActivationQuant>,
    /// Only on `unembed`: logits are produced in f32.
    pub fwd_output_dtype: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ActivationQuant {
    pub format: String,
    pub block_size: usize,
    pub scale_dtype: String,
    pub rounding: String,
    pub amax_type: Option<String>,
    pub scale_selection_rule: Option<String>,
}

/// Weight format of an expert block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertFormat {
    /// E2M1, two values per byte; E4M3 scale per 16; outer scale.
    Nvfp4,
    /// E4M3, one value per byte; UE8M0 scale per 32.
    Mxfp8,
}

impl ExpertFormat {
    pub fn group_size(self) -> usize {
        match self {
            Self::Nvfp4 => 16,
            Self::Mxfp8 => 32,
        }
    }

    /// Values per byte in the weight tensor.
    pub fn values_per_byte(self) -> usize {
        match self {
            Self::Nvfp4 => 2,
            Self::Mxfp8 => 1,
        }
    }

    /// Infers the format of an expert weight and its scale, and checks that
    /// the shapes agree with `in_features`.
    pub fn detect(weight: &TensorInfo, scale: &TensorInfo, in_features: usize) -> Result<Self> {
        let fmt = match (weight.dtype, scale.dtype) {
            (Dtype::U8, Dtype::F8E4M3) => Self::Nvfp4,
            (Dtype::F8E4M3, Dtype::U8) => Self::Mxfp8,
            (w, s) => bail!(
                "{}: unknown expert format ({w:?}, scale {s:?})",
                weight.name
            ),
        };
        let [e, rows, packed] = weight.shape[..] else {
            bail!(
                "{}: expected 3 dimensions, got {:?}",
                weight.name,
                weight.shape
            );
        };
        let ok = packed * fmt.values_per_byte() == in_features
            && scale.shape == [e, rows, in_features / fmt.group_size()];
        if !ok {
            bail!(
                "{}: shape {:?} / scale {:?} do not match {fmt:?} with in={in_features}",
                weight.name,
                weight.shape,
                scale.shape
            );
        }
        Ok(fmt)
    }
}

impl QuantConfig {
    /// Parameter group of every main-model tensor.
    pub fn model_groups(&self) -> Result<BTreeMap<&str, &ParamGroup>> {
        index(&self.model)
    }

    pub fn mtp_groups(&self) -> Result<BTreeMap<&str, &ParamGroup>> {
        index(&self.mtp)
    }
}

fn index(s: &Section) -> Result<BTreeMap<&str, &ParamGroup>> {
    let mut out = BTreeMap::new();
    for g in &s.parameter_groups {
        for t in &g.tensors {
            if out.insert(t.as_str(), g).is_some() {
                bail!("{t} appears in two groups of hf_quant_config.json");
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_param_groups() {
        let q: QuantConfig =
            serde_json::from_str(include_str!("../tests/data/hf_quant_config.json")).unwrap();
        assert_eq!(q.quant_method, "inkling2");
        let g = q.model_groups().unwrap();

        let fp4 = g["model.llm.layers.0.mlp.experts.w13_weight"];
        assert_eq!(fp4.weight_transform, "nvfp4_nearest");
        assert_eq!(fp4.weight_scale_format.as_deref(), Some("amax_denominator"));
        let a = fp4.required_activation_quantization.as_ref().unwrap();
        assert_eq!((a.format.as_str(), a.block_size), ("nvfp4_e2m1", 16));

        let fp8 = g["model.llm.layers.20.mlp.experts.w2_weight"];
        assert_eq!(fp8.weight_transform, "mxfp8_nearest");
        assert_eq!(fp8.weight_block_shape, [32, 32]);

        assert_eq!(
            g["model.llm.attn_res.attn_queries.0"].compute_dtype,
            "float32"
        );
        assert_eq!(
            g["model.llm.unembed.weight"].fwd_output_dtype.as_deref(),
            Some("float32")
        );
        assert_eq!(q.mtp_groups().unwrap().len(), 54);
    }
}
