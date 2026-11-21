use std::{collections::BTreeMap, path::PathBuf};

use anyhow::Result;
use calamardo_ckpt::{
    Checkpoint, DEFAULT_REPO, ExpertFormat, ModelConfig, default_hub_cache, is_skipped,
    schedule::{self, AttnKind},
    snapshot_dir,
};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about = "Inkling 2 Mini runtime on MLX")]
struct Cli {
    /// Hugging Face repository (read from the local Hub cache).
    #[arg(long, default_value = DEFAULT_REPO, global = true)]
    repo: String,

    /// Snapshot branch or commit hash.
    #[arg(long, default_value = "main", global = true)]
    revision: String,

    /// Checkpoint directory; when given, the Hub cache is not used.
    #[arg(long, env = "CALAMARDO_MODEL_DIR", global = true)]
    model_dir: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Summarizes the checkpoint: tensors, formats, memory and pass order.
    Inspect {
        /// Global KV cache bits used for the memory estimate.
        #[arg(long, env = "CALAMARDO_KV_BITS", default_value_t = 8)]
        kv_bits: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let dir = match cli.model_dir {
        Some(d) => d,
        None => snapshot_dir(&default_hub_cache()?, &cli.repo, &cli.revision)?,
    };
    match cli.cmd {
        Cmd::Inspect { kv_bits } => inspect(&Checkpoint::open(&dir)?, kv_bits),
    }
}

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

/// Version reported by the libmlx.dylib loaded at runtime.
fn mlx_version() -> Result<String> {
    // SAFETY: the string handle is created, read and freed here; the data
    // pointer is copied before the handle is freed.
    unsafe {
        let mut s = mlx_sys::mlx_string_new();
        anyhow::ensure!(mlx_sys::mlx_version(&mut s) == 0, "mlx_version failed");
        let v = std::ffi::CStr::from_ptr(mlx_sys::mlx_string_data(s))
            .to_string_lossy()
            .into_owned();
        mlx_sys::mlx_string_free(s);
        Ok(v)
    }
}

fn inspect(ck: &Checkpoint, kv_bits: usize) -> Result<()> {
    let cfg: ModelConfig = ck.read_json("config.json")?;
    cfg.validate()?;
    let t = &cfg.text_config;

    println!(
        "mlx         {} on {:?} · memory limit {:.1} GB",
        mlx_version()?,
        mlx_rs::Device::default(),
        mlx_rs::memory::memory_limit()? as f64 / 1e9
    );

    let mut groups: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    let mut dtypes: BTreeMap<String, u64> = BTreeMap::new();
    for (name, info) in &ck.tensors {
        let g = if is_skipped(name) {
            "vision (not loaded)"
        } else if name.contains(".mlp.experts.") {
            "MoE experts"
        } else {
            "rest of the model"
        };
        let e = groups.entry(g).or_default();
        e.0 += 1;
        e.1 += info.nbytes;
        if !is_skipped(name) {
            *dtypes.entry(format!("{:?}", info.dtype)).or_default() += info.nbytes;
        }
    }
    let draft_bytes: u64 = ck.draft.values().map(|t| t.nbytes).sum();
    groups.insert("draft model", (ck.draft.len(), draft_bytes));

    println!("model       {}", cfg.architectures.join(", "));
    println!("snapshot    {}", ck.dir.display());
    let mut loaded = 0;
    for (g, (n, b)) in &groups {
        println!("  {g:<22} {n:>5} tensors   {:>6.2} GB", gb(*b));
        if !g.starts_with("vision") {
            loaded += b;
        }
    }
    println!("  {:<22} {:>14} {:>6.2} GB", "to load", "", gb(loaded));
    let v: Vec<_> = dtypes
        .iter()
        .map(|(d, b)| format!("{d} {:.2} GB", gb(*b)))
        .collect();
    println!("dtypes      {}", v.join(" · "));

    let mut formats: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for l in 0..t.num_hidden_layers {
        let p = format!("model.llm.layers.{l}.mlp.experts.w13_weight");
        let f = ExpertFormat::detect(
            ck.tensor(&p)?,
            ck.tensor(&format!("{p}_scale"))?,
            t.hidden_size,
        )?;
        formats.entry(format!("{f:?}")).or_default().push(l);
    }
    for (f, layers) in &formats {
        println!(
            "experts     {f:<6} layers {}–{}",
            layers[0],
            layers[layers.len() - 1]
        );
    }

    let passes = schedule::passes(t);
    let globals: Vec<_> = passes
        .iter()
        .filter(|p| p.attn == AttnKind::Global)
        .collect();
    println!(
        "passes      {} ({} global, {} sliding-window {})",
        passes.len(),
        globals.len(),
        passes.len() - globals.len(),
        t.sliding_window_size
    );
    let g: Vec<_> = globals
        .iter()
        .map(|p| format!("{}←L{}", p.index, p.layer))
        .collect();
    println!("  global    {}", g.join(" "));
    println!(
        "  draft     reads passes {:?}",
        cfg.mtp_config.target_layer_ids
    );

    let per_tok = globals.len() * schedule::global_kv_bytes_per_token(t, kv_bits, 64);
    let window = (passes.len() - globals.len())
        * t.sliding_window_size
        * schedule::global_kv_bytes_per_token(t, 16, 64);
    println!(
        "KV cache    {kv_bits} bits: {per_tok} B/token · windows {:.0} MB",
        window as f64 / 1e6
    );
    for ctx in [32_768u64, 163_840, 327_680] {
        println!(
            "  {:>4}k     {:>6.2} GB",
            ctx / 1024,
            gb(ctx * per_tok as u64 + window as u64)
        );
    }
    Ok(())
}
