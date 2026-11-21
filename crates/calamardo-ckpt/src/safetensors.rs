//! Safetensors headers: only the index is read, never the data.

use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

/// Data type as written by safetensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
}

impl Dtype {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "U16" => Self::U16,
            "I16" => Self::I16,
            "U32" => Self::U32,
            "I32" => Self::I32,
            "U64" => Self::U64,
            "I64" => Self::I64,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            other => bail!("unknown safetensors dtype: {other}"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// A tensor inside a safetensors file.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub file: PathBuf,
    /// Absolute offset of the data within `file`.
    pub offset: u64,
    pub nbytes: u64,
    /// The file's `__metadata__` (e.g. `format`).
    pub metadata: BTreeMap<String, String>,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Sanity limit for the JSON header (safetensors uses 100 MB).
const MAX_HEADER: u64 = 100 << 20;

/// Reads the header of `path` and returns its tensors sorted by offset.
pub fn read_header(path: &Path) -> Result<Vec<TensorInfo>> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = f.metadata()?.len();
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .with_context(|| format!("reading header of {}", path.display()))?;
    let header_len = u64::from_le_bytes(len);
    ensure!(
        header_len <= MAX_HEADER && 8 + header_len <= file_len,
        "{}: invalid safetensors header ({header_len} bytes)",
        path.display()
    );
    let mut buf = vec![0u8; header_len as usize];
    f.read_exact(&mut buf)?;
    let mut raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&buf)
        .with_context(|| format!("header JSON in {}", path.display()))?;

    let metadata: BTreeMap<String, String> = match raw.remove("__metadata__") {
        Some(v) => serde_json::from_value(v).context("__metadata__")?,
        None => BTreeMap::new(),
    };

    let data_start = 8 + header_len;
    let data_len = file_len - data_start;
    let mut out = Vec::with_capacity(raw.len());
    for (name, v) in raw {
        let e: RawEntry =
            serde_json::from_value(v).with_context(|| format!("{name} in {}", path.display()))?;
        let dtype = Dtype::parse(&e.dtype)?;
        let [start, end] = e.data_offsets;
        let expected = e.shape.iter().product::<usize>() as u64 * dtype.size() as u64;
        ensure!(
            start <= end && end <= data_len && end - start == expected,
            "{name}: data_offsets {start}..{end} do not match {:?} {:?} (data: {data_len})",
            e.shape,
            dtype
        );
        out.push(TensorInfo {
            name,
            dtype,
            shape: e.shape,
            file: path.to_path_buf(),
            offset: data_start + start,
            nbytes: end - start,
            metadata: metadata.clone(),
        });
    }
    out.sort_by_key(|t| t.offset);
    Ok(out)
}
