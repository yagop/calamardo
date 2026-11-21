//! Reads the official checkpoint from the Hugging Face cache.
//!
//! Skipped when it is not fully downloaded
//! (`hf download thinkingmachines/Inkling-2-Mini`), so the other tests pass on
//! any machine.

use calamardo_ckpt::{
    Checkpoint, DEFAULT_REPO, Dtype, ExpertFormat, ModelConfig, QuantConfig, default_hub_cache,
    hub::{DRAFT_FILE, shard_files},
    is_skipped, snapshot_dir,
};

fn open() -> Option<Checkpoint> {
    let dir = snapshot_dir(&default_hub_cache().ok()?, DEFAULT_REPO, "main").ok()?;
    let complete = shard_files(&dir)
        .map(|s| s.iter().all(|f| dir.join(f).exists()))
        .unwrap_or(false)
        && dir.join(DRAFT_FILE).exists();
    if !complete {
        eprintln!("{DEFAULT_REPO} is not fully downloaded; skipping");
        return None;
    }
    Some(Checkpoint::open(&dir).expect("opening the checkpoint"))
}

#[test]
fn full_tensor_index() {
    let Some(ck) = open() else { return };
    let cfg: ModelConfig = ck.read_json("config.json").unwrap();
    cfg.validate().unwrap();
    let t = &cfg.text_config;
    let q: QuantConfig = ck.read_json("hf_quant_config.json").unwrap();
    let groups = q.model_groups().unwrap();

    // Every checkpoint tensor has a quantization group (scales belong to
    // their weight's group), and every group tensor exists.
    for name in ck.tensors.keys() {
        assert!(
            groups.contains_key(name.as_str())
                || name.ends_with("_scale")
                || name.contains("_amax")
                || name.contains("outer_scale"),
            "{name} has no group"
        );
    }
    for name in groups.keys() {
        ck.tensor(name).unwrap();
    }

    for l in 0..t.num_hidden_layers {
        for suffix in [
            "attn.wq_du.weight",
            "attn.wk_dv.weight",
            "attn.wv_dv.weight",
            "attn.wo_ud.weight",
            "attn.attn_sink_proj.weight",
            "attn.rel_logits_proj.proj",
            "mlp.gate.weight",
            "mlp.gate.bias",
        ] {
            ck.tensor(&format!("model.llm.layers.{l}.{suffix}"))
                .unwrap();
        }
        let core = (t.loop_config.core_start..t.loop_config.core_end).contains(&l);
        let has_ctrl = ck.tensors.contains_key(&format!(
            "model.llm.layers.{l}.controllers.attn.gain_weight"
        ));
        assert_eq!(core, has_ctrl, "loop controller in layer {l}");

        // Experts (M-5): nvfp4 in 0–19, mxfp8 in 20–23.
        let p = format!("model.llm.layers.{l}.mlp.experts");
        let w13 = ck.tensor(&format!("{p}.w13_weight")).unwrap();
        let s13 = ck.tensor(&format!("{p}.w13_weight_scale")).unwrap();
        let w2 = ck.tensor(&format!("{p}.w2_weight")).unwrap();
        let s2 = ck.tensor(&format!("{p}.w2_weight_scale")).unwrap();
        let f13 = ExpertFormat::detect(w13, s13, t.hidden_size).unwrap();
        let f2 = ExpertFormat::detect(w2, s2, t.intermediate_size).unwrap();
        assert_eq!(f13, f2);
        let expected = if l < 20 {
            ExpertFormat::Nvfp4
        } else {
            ExpertFormat::Mxfp8
        };
        assert_eq!(f13, expected, "expert format in layer {l}");
        assert_eq!(
            w13.shape[..2],
            [t.n_routed_experts, 2 * t.intermediate_size]
        );
        assert_eq!(w2.shape[..2], [t.n_routed_experts, t.hidden_size]);
        let outer = ck.tensors.contains_key(&format!("{p}.w13_weight_amax"));
        assert_eq!(outer, f13 == ExpertFormat::Nvfp4);
    }

    let emb = ck.tensor("model.llm.embed.weight").unwrap();
    assert_eq!(emb.dtype, Dtype::BF16);
    assert_eq!(emb.shape, [t.vocab_size, t.hidden_size]);

    // DSpark draft model (M-7), in mtp.safetensors.
    assert_eq!(ck.draft.len(), q.mtp_groups().unwrap().len());
    for i in 0..cfg.mtp_config.num_hidden_layers {
        ck.draft_tensor(&format!("layers.{i}.self_attn.q_proj.weight"))
            .unwrap();
    }

    assert!(ck.tensors.keys().any(|k| is_skipped(k)));
}
