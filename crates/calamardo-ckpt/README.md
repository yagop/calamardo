# calamardo-ckpt

Reads the Inkling 2 Mini checkpoint from a Hugging Face snapshot: the tensor index, the configuration, the quantization formats and the pass order. It reads headers and metadata only, never the weights, and does not depend on MLX.

| Module | Contents |
|---|---|
| `hub` | Finds a snapshot in the Hub cache (`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<rev>/`, honouring `HF_HUB_CACHE` and `HF_HOME`) or in any directory with the same layout, and indexes its shards: `config.json`, `model.safetensors.index.json` and `mtp.safetensors` (the drafter) |
| `config` | `ModelConfig`: the fields of `config.json` the runtime uses (text model, loop, MoE, relative attention, drafter). The rest, such as `vision_config`, is ignored |
| `quant` | `hf_quant_config.json`: nvfp4 experts in layers 0–19, mxfp8 experts in layers 20–23, and the format of every other tensor |
| `safetensors` | Safetensors header parsing: names, dtypes, shapes, offsets and metadata |
| `schedule` | Per-token pass order: the layers before the core run once, the core repeats `num_passes` times back to back, and the layers after it run once. Every pass has its own KV cache |

```rust
use calamardo_ckpt::{Checkpoint, DEFAULT_REPO, ModelConfig, default_hub_cache, snapshot_dir};

let dir = snapshot_dir(&default_hub_cache()?, DEFAULT_REPO, "main")?;
let ckpt = Checkpoint::open(&dir)?;
let config: ModelConfig = ckpt.read_json("config.json")?;
```

## Tests

`tests/hub_snapshot.rs` reads the official checkpoint from the Hugging Face cache (and skips if it is not downloaded): every tensor has a quantization format and every format's tensor exists. `tests/data` holds the checkpoint's `config.json` and `hf_quant_config.json` for the tests that do not need the download.
