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
    /// KV cache quality: generates greedily with a bf16 cache, then replays
    /// the same tokens with each quantized format and reports the mean KL
    /// divergence and top-1 agreement against bf16 (verification 2).
    KvEval {
        /// File with the full prompt text (chat format included).
        prompt_file: PathBuf,
        #[arg(long, default_value_t = 256)]
        tokens: usize,
        /// Formats to compare, as bits/group.
        #[arg(long, value_delimiter = ',', default_value = "8/64,4/64")]
        formats: Vec<String>,
    },
    /// Greedy generation (milestone 1 check uses --kv-bits 16).
    Generate {
        /// User message, wrapped in the chat format.
        #[arg(long, conflicts_with_all = ["raw", "tokens"])]
        prompt: Option<String>,
        /// Text tokenized as is, without the chat format.
        #[arg(long, conflicts_with = "tokens")]
        raw: Option<String>,
        /// File whose text is tokenized as is, like --raw.
        #[arg(long, conflicts_with_all = ["raw", "tokens", "prompt"])]
        raw_file: Option<PathBuf>,
        /// Comma-separated token ids.
        #[arg(long, value_delimiter = ',')]
        tokens: Option<Vec<u32>>,
        /// Reasoning effort for --prompt: none, minimal, low, medium, high.
        #[arg(long, default_value = "none")]
        reasoning_effort: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Print the token ids instead of the text.
        #[arg(long)]
        ids: bool,
        /// Bits of the global passes' KV cache: 16, 8 or 4.
        #[arg(long, env = "CALAMARDO_KV_BITS", default_value_t = 8)]
        kv_bits: i32,
        /// Channels per scale/bias group of the quantized KV cache.
        #[arg(long, env = "CALAMARDO_KV_GROUP", default_value_t = 64)]
        kv_group: i32,
        /// Prompt tokens per prefill chunk.
        #[arg(long, default_value_t = calamardo_model::generate::PREFILL_STEP)]
        prefill_step: usize,
        /// Log time and memory of each prefill chunk.
        #[arg(long, short)]
        verbose: bool,
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
        Cmd::KvEval {
            prompt_file,
            tokens,
            formats,
        } => {
            let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
            let text = std::fs::read_to_string(&prompt_file)?;
            let prompt = tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
                .get_ids()
                .to_vec();
            let formats = formats
                .iter()
                .map(|f| {
                    let (b, g) = f.split_once('/').unwrap_or((f, "64"));
                    calamardo_model::cache::KvFormat::new(b.parse()?, g.parse()?)
                })
                .collect::<Result<Vec<_>>>()?;
            kv_eval(&dir, &prompt, tokens, &formats)
        }
        Cmd::Generate {
            prompt,
            mut raw,
            raw_file,
            tokens,
            reasoning_effort,
            max_tokens,
            ids,
            kv_bits,
            kv_group,
            prefill_step,
            verbose,
        } => {
            let kv = calamardo_model::cache::KvFormat::new(kv_bits, kv_group)?;
            let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
            if let Some(f) = raw_file {
                raw = Some(std::fs::read_to_string(f)?);
            }
            let text = match (prompt, raw) {
                (Some(p), _) => Some(chat_prompt(&p, &reasoning_effort)?),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            };
            let prompt_ids = match (text, tokens) {
                (Some(t), _) => tokenizer
                    .encode(t, false)
                    .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
                    .get_ids()
                    .to_vec(),
                (None, Some(t)) => t,
                (None, None) => anyhow::bail!("pass --prompt, --raw or --tokens"),
            };
            generate(
                &dir,
                &tokenizer,
                &prompt_ids,
                max_tokens,
                ids,
                kv,
                prefill_step,
                verbose,
            )
        }
    }
}

/// Turns a token stream into text without splitting multi-token characters:
/// byte-level BPE can spread one UTF-8 character over several tokens.
#[derive(Default)]
struct StreamDecoder {
    ids: Vec<u32>,
    /// Start of the tokens decoded together for context.
    prefix: usize,
    /// End of the tokens whose text was already printed.
    read: usize,
}

impl StreamDecoder {
    /// Returns the new text once it is complete.
    fn push(&mut self, tokenizer: &tokenizers::Tokenizer, id: u32) -> Option<String> {
        self.ids.push(id);
        let printed = tokenizer
            .decode(&self.ids[self.prefix..self.read], false)
            .ok()?;
        let current = tokenizer.decode(&self.ids[self.prefix..], false).ok()?;
        if current.len() > printed.len() && !current.ends_with('\u{FFFD}') {
            let new = current[printed.len()..].to_string();
            self.prefix = self.read;
            self.read = self.ids.len();
            Some(new)
        } else {
            None
        }
    }
}

/// A single user message in the chat format of the checkpoint's template.
/// Temporary until the template is rendered for real (M-9).
fn chat_prompt(message: &str, effort: &str) -> Result<String> {
    let level = match effort {
        "none" => "0",
        "minimal" => "0.1",
        "low" => "0.2",
        "medium" => "0.7",
        "high" => "0.9",
        "xhigh" | "max" => "0.99",
        other => anyhow::bail!("unknown reasoning effort: {other}"),
    };
    Ok(format!(
        "<|message_system|><|content_text|>Thinking effort level: {level}<|end_message|>\
         <|message_user|><|content_text|>{message}<|end_message|>"
    ))
}

#[allow(clippy::too_many_arguments)]
fn generate(
    dir: &std::path::Path,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &[u32],
    max_tokens: usize,
    ids: bool,
    kv: calamardo_model::cache::KvFormat,
    prefill_step: usize,
    verbose: bool,
) -> Result<()> {
    use std::io::Write;
    use std::time::Instant;

    let t = Instant::now();
    let model = calamardo_model::Model::load(dir)?;
    let eos = model.config.eos_token_id;
    eprintln!(
        "loaded in {:.1}s · {:.2} GB active",
        t.elapsed().as_secs_f64(),
        mlx_rs::memory::active_memory()? as f64 / 1e9
    );
    eprintln!(
        "prompt: {} tokens · global KV cache {} bits",
        prompt.len(),
        kv.bits
    );

    mlx_rs::memory::reset_peak_memory()?;
    let t = Instant::now();
    let mut first: Option<f64> = None;
    let mut out = std::io::stdout();
    let mut text = StreamDecoder::default();
    let tokens = calamardo_model::generate::greedy(
        &model,
        prompt,
        max_tokens,
        prefill_step,
        kv,
        |c| {
            if verbose {
                eprintln!(
                    "prefill {:>7}/{} · {:>4} tokens in {:.2}s · {:.0} tok/s · active {:.2} GB · peak {:.2} GB (+{:.2})",
                    c.done,
                    c.total,
                    c.tokens,
                    c.seconds,
                    c.tokens as f64 / c.seconds,
                    c.active_bytes as f64 / 1e9,
                    c.peak_bytes as f64 / 1e9,
                    (c.peak_bytes as f64 - c.active_bytes as f64) / 1e9
                );
            }
        },
        |tok| {
            first.get_or_insert(t.elapsed().as_secs_f64());
            if ids {
                print!("{tok} ");
            } else if let Some(s) = text.push(tokenizer, tok) {
                print!("{s}");
            }
            let _ = out.flush();
            tok != eos
        },
    )?;
    println!();
    let total = t.elapsed().as_secs_f64();
    let first = first.unwrap_or(total);
    let decode = (tokens.len().saturating_sub(1)) as f64 / (total - first).max(1e-9);
    eprintln!(
        "prefill {:.0} tok/s ({:.2}s) · decode {:.1} tok/s · {} tokens · peak {:.2} GB",
        prompt.len() as f64 / first,
        first,
        decode,
        tokens.len(),
        mlx_rs::memory::peak_memory()? as f64 / 1e9
    );
    Ok(())
}

fn kv_eval(
    dir: &std::path::Path,
    prompt: &[u32],
    n: usize,
    formats: &[calamardo_model::cache::KvFormat],
) -> Result<()> {
    use calamardo_model::{cache::KvFormat, generate, ops};
    let model = calamardo_model::Model::load(dir)?;
    eprintln!("prompt: {} tokens, {n} generated", prompt.len());

    // bf16 reference: greedy tokens and their distributions.
    let tokens = generate::greedy(
        &model,
        prompt,
        n,
        generate::PREFILL_STEP,
        KvFormat::BF16,
        |_| {},
        |_| true,
    )?;
    let mut reference = Vec::with_capacity(tokens.len());
    generate::teacher_forced(
        &model,
        prompt,
        &tokens,
        generate::PREFILL_STEP,
        KvFormat::BF16,
        |_, lp| {
            reference.push(lp.clone());
            Ok(())
        },
    )?;

    println!("format     mean KL     max KL     top-1 agree");
    for &f in formats {
        let (mut sum, mut max, mut agree) = (0.0f64, 0.0f64, 0usize);
        generate::teacher_forced(
            &model,
            prompt,
            &tokens,
            generate::PREFILL_STEP,
            f,
            |i, lp| {
                let r = &reference[i];
                let kl = ops::sum(&ops::mul(&ops::exp(r)?, &ops::sub(r, lp)?)?, &[0, 1], false)?;
                let same = ops::astype(&ops::argmax(lp, -1)?, mlx_rs::Dtype::Uint32)?;
                ops::eval(&[&kl, &same])?;
                let kl = kl.item_exact::<f32>() as f64;
                sum += kl;
                max = max.max(kl);
                agree += usize::from(same.item_exact::<u32>() == tokens[i]);
                Ok(())
            },
        )?;
        println!(
            "{:>2} bits/{:<3} {:>9.5} {:>10.5} {:>8}/{}",
            f.bits,
            f.group_size,
            sum / tokens.len() as f64,
            max,
            agree,
            tokens.len()
        );
    }
    Ok(())
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
