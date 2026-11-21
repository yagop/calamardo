//! Prefix cache on the SSD (C-3 to C-5, C-9): conversations evicted from RAM
//! are saved instead of lost, and read back when a prompt continues them.
//!
//! One safetensors file per conversation, named after a hash of its tokens,
//! with a manifest in its metadata (format, model, KV cache format). The
//! index is rebuilt at startup from the headers and the token tensors;
//! files that do not match the manifest are deleted. The file's modification
//! time is its last use, for LRU eviction within the byte budget.

use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::{Instant, SystemTime},
};

use anyhow::{Context, Result};
use calamardo_ckpt::{Dtype, safetensors::read_header};
use calamardo_model::{
    Model,
    cache::KvFormat,
    draft::Drafter,
    generate::{Incompatible, Reuse, SAVE_FORMAT, Session},
};

struct Entry {
    /// Loaded and still in RAM: the file stays as a backup until the session
    /// is saved again, but it is not offered for reuse or evicted.
    resident: bool,
    path: PathBuf,
    tokens: Vec<u32>,
    /// Length of the saved prompt-end snapshot, if any.
    snapshot: Option<usize>,
    bytes: u64,
    used: SystemTime,
}

pub struct DiskCache {
    dir: PathBuf,
    budget: u64,
    manifest: BTreeMap<String, String>,
    entries: Vec<Entry>,
}

fn hash(tokens: &[u32]) -> u64 {
    // FNV-1a over the token ids.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Reads the manifest and the token tensor of a saved session.
fn read_entry(path: &Path) -> Result<(BTreeMap<String, String>, Vec<u32>)> {
    let tensors = read_header(path)?;
    let t = tensors
        .iter()
        .find(|t| t.name == "tokens")
        .context("no tokens tensor")?;
    anyhow::ensure!(t.dtype == Dtype::U32, "tokens are {:?}", t.dtype);
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(t.offset))?;
    let mut buf = vec![0u8; t.nbytes as usize];
    f.read_exact(&mut buf)?;
    let tokens = buf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    Ok((t.metadata.clone(), tokens))
}

impl DiskCache {
    pub fn open(dir: &Path, budget: u64, manifest: BTreeMap<String, String>) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let mut entries = Vec::new();
        for item in std::fs::read_dir(dir)? {
            let path = item?.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.ends_with(".tmp.safetensors") {
                // Left by a crash mid-save.
                let _ = std::fs::remove_file(&path);
                continue;
            }
            if !name.ends_with(".safetensors") {
                continue;
            }
            let ok = read_entry(&path).ok().filter(|(meta, _)| {
                meta.get("format").map(String::as_str) == Some(SAVE_FORMAT)
                    && manifest.iter().all(|(k, v)| meta.get(k) == Some(v))
            });
            match ok {
                Some((meta, tokens)) => {
                    let md = std::fs::metadata(&path)?;
                    entries.push(Entry {
                        resident: false,
                        snapshot: meta.get("snap.len").and_then(|v| v.parse().ok()),
                        path,
                        tokens,
                        bytes: md.len(),
                        used: md.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                    });
                }
                None => {
                    // Another model or cache format (C-4): discard.
                    eprintln!("discarding incompatible saved session {}", path.display());
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        let mut cache = Self {
            dir: dir.to_path_buf(),
            budget,
            manifest,
            entries,
        };
        cache.evict();
        Ok(cache)
    }

    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| !e.resident).count()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    /// A resident file whose session left RAM without being saved becomes
    /// an ordinary entry again.
    pub fn release(&mut self, path: &Path) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.path == path) {
            e.resident = false;
        }
    }

    /// The saved conversation that holds the longest start of `prompt`.
    pub fn best(&self, prompt: &[u32], aligned_to: Option<usize>) -> Option<(usize, Reuse)> {
        let ok = |len: usize| {
            len > 0 && len < prompt.len() && aligned_to.is_none_or(|a| len.is_multiple_of(a))
        };
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.resident)
            .filter_map(|(i, e)| {
                let common = e
                    .tokens
                    .iter()
                    .zip(prompt)
                    .take_while(|(a, b)| a == b)
                    .count();
                if common == e.tokens.len() && ok(common) {
                    return Some((i, Reuse::Live { len: common }));
                }
                e.snapshot
                    .filter(|&s| s <= common && ok(s))
                    .map(|len| (i, Reuse::Snapshot { index: 0, len }))
            })
            .max_by_key(|(_, r)| r.tokens())
    }

    /// The most recently used conversation (for prewarming).
    pub fn latest(&self) -> Option<usize> {
        (0..self.entries.len())
            .filter(|&i| !self.entries[i].resident && !self.entries[i].tokens.is_empty())
            .max_by_key(|&i| self.entries[i].used)
    }

    /// Loads conversation `i` into RAM. The file stays (marked resident) so
    /// a crash does not lose it; it is replaced when the session is saved
    /// again. Only an incompatible file is deleted. Returns the session and
    /// the seconds it took.
    pub fn take<'m>(
        &mut self,
        i: usize,
        model: &'m Model,
        drafter: Option<&'m Drafter>,
        kv: KvFormat,
    ) -> Result<(Session<'m>, f64)> {
        let t = Instant::now();
        let path = self.entries[i].path.clone();
        match Session::load(model, drafter, kv, &path, &self.manifest) {
            Ok(s) => {
                self.entries[i].resident = true;
                Ok((s, t.elapsed().as_secs_f64()))
            }
            Err(e) => {
                if e.downcast_ref::<Incompatible>().is_some() {
                    let _ = std::fs::remove_file(&path);
                    self.entries.remove(i);
                }
                Err(e)
            }
        }
    }

    /// Saves a conversation evicted from RAM, then keeps the disk within
    /// its budget, oldest first (C-5).
    pub fn save(&mut self, session: &Session) -> Result<f64> {
        let t = Instant::now();
        let path = self
            .dir
            .join(format!("{:016x}.safetensors", hash(&session.tokens)));
        session.save(&path, &self.manifest)?;
        self.entries.retain(|e| e.path != path);
        // The file it was loaded from is now superseded.
        if let Some(origin) = &session.origin
            && *origin != path
        {
            let _ = std::fs::remove_file(origin);
            self.entries.retain(|e| e.path != *origin);
        }
        self.entries.push(Entry {
            resident: false,
            bytes: std::fs::metadata(&path)?.len(),
            path,
            tokens: session.tokens.clone(),
            snapshot: session
                .snapshots
                .iter()
                .rev()
                .find(|s| s.prompt_end)
                .map(|s| s.len),
            used: SystemTime::now(),
        });
        self.evict();
        Ok(t.elapsed().as_secs_f64())
    }

    fn evict(&mut self) {
        while self.bytes() > self.budget {
            let Some(oldest) = (0..self.entries.len())
                .filter(|&i| !self.entries[i].resident)
                .min_by_key(|&i| self.entries[i].used)
            else {
                break;
            };
            let e = self.entries.remove(oldest);
            eprintln!("disk cache full: removing {} tokens", e.tokens.len());
            let _ = std::fs::remove_file(&e.path);
        }
    }
}
