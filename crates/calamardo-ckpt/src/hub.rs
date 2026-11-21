//! Checkpoint downloaded from Hugging Face.
//!
//! Reads a snapshot from the standard Hub cache
//! (`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>/`) or any
//! directory with the same layout: `config.json`,
//! `model.safetensors.index.json` with its shards, and `mtp.safetensors`
//! (the draft model).

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, de::DeserializeOwned};

use crate::safetensors::{TensorInfo, read_header};

/// Hub cache directory: `$HF_HUB_CACHE`, `$HF_HOME/hub` or
/// `~/.cache/huggingface/hub`, in that order (same as `huggingface_hub`).
pub fn default_hub_cache() -> Result<PathBuf> {
    if let Some(d) = env::var_os("HF_HUB_CACHE") {
        return Ok(PathBuf::from(d));
    }
    if let Some(d) = env::var_os("HF_HOME") {
        return Ok(PathBuf::from(d).join("hub"));
    }
    let home = env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// Snapshot of `repo` (`org/name`) at revision `rev` (branch or commit hash).
pub fn snapshot_dir(cache: &Path, repo: &str, rev: &str) -> Result<PathBuf> {
    let (org, name) = repo
        .split_once('/')
        .with_context(|| format!("repo without organization: {repo}"))?;
    let valid = |s: &str| {
        !s.is_empty()
            && !s.starts_with('.')
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    };
    ensure!(
        valid(org) && valid(name) && valid(rev),
        "invalid repo or revision: {repo}@{rev}"
    );
    let root = cache.join(format!("models--{org}--{name}"));
    let ref_file = root.join("refs").join(rev);
    let hash = if ref_file.exists() {
        fs::read_to_string(&ref_file)?.trim().to_string()
    } else {
        rev.to_string()
    };
    let dir = root.join("snapshots").join(&hash);
    ensure!(
        dir.is_dir(),
        "{repo}@{rev} is not downloaded in {} (run `hf download {repo}`)",
        cache.display()
    );
    Ok(dir)
}

#[derive(Deserialize)]
struct ShardIndex {
    weight_map: BTreeMap<String, String>,
}

/// Name of the index that lists the main model's shards.
pub const INDEX_FILE: &str = "model.safetensors.index.json";

/// Draft model (DSpark) file inside the snapshot.
pub const DRAFT_FILE: &str = "mtp.safetensors";

/// Tensor index of a snapshot. Weights are not read.
#[derive(Debug)]
pub struct Checkpoint {
    pub dir: PathBuf,
    /// Main model tensors, by name.
    pub tensors: BTreeMap<String, TensorInfo>,
    /// Draft model tensors (`mtp.safetensors`), by unprefixed name.
    pub draft: BTreeMap<String, TensorInfo>,
}

/// Shard files listed in the index of `dir`.
pub fn shard_files(dir: &Path) -> Result<BTreeSet<String>> {
    let index: ShardIndex = read_json_file(&dir.join(INDEX_FILE))?;
    let shards: BTreeSet<String> = index.weight_map.into_values().collect();
    for shard in &shards {
        ensure!(
            !shard.contains('/') && shard.ends_with(".safetensors"),
            "invalid shard in index: {shard}"
        );
    }
    Ok(shards)
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self> {
        let index: ShardIndex = read_json_file(&dir.join(INDEX_FILE))?;

        let mut tensors = BTreeMap::new();
        for shard in shard_files(dir)? {
            for t in read_header(&dir.join(&shard))? {
                if let Some(prev) = tensors.insert(t.name.clone(), t) {
                    bail!("duplicate tensor: {}", prev.name);
                }
            }
        }
        // The index and the headers must match exactly.
        for (name, shard) in &index.weight_map {
            let t = tensors
                .get(name)
                .with_context(|| format!("{name} is in the index but not in {shard}"))?;
            ensure!(
                t.file.file_name().and_then(|f| f.to_str()) == Some(shard.as_str()),
                "{name}: index says {shard}, found in {}",
                t.file.display()
            );
        }
        ensure!(
            tensors.len() == index.weight_map.len(),
            "shards hold {} tensors but the index lists {}",
            tensors.len(),
            index.weight_map.len()
        );

        let draft_path = dir.join(DRAFT_FILE);
        let draft = if draft_path.exists() {
            read_header(&draft_path)?
                .into_iter()
                .map(|t| (t.name.clone(), t))
                .collect()
        } else {
            BTreeMap::new()
        };

        Ok(Self {
            dir: dir.to_path_buf(),
            tensors,
            draft,
        })
    }

    pub fn read_json<T: DeserializeOwned>(&self, name: &str) -> Result<T> {
        read_json_file(&self.dir.join(name))
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    pub fn draft_tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.draft
            .get(name)
            .with_context(|| format!("missing draft tensor {name}"))
    }
}

fn read_json_file<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}
