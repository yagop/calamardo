# Calamardo

A Rust runtime that serves [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on Apple Silicon Macs, built on [MLX](https://github.com/ml-explore/mlx).

- **Long contexts without memory spikes:** chunked prefill with a bounded transient (under 3 GB above active memory at 516k tokens), a 4-, 8- or 16-bit global KV cache, and bf16 sliding windows.
- **Prefix cache in RAM and on disk:** each turn only processes what is new. A 516k-token conversation is saved in 4 s and restored in 2 s.
- **Speculative decoding** with the DSpark drafter that ships with the checkpoint.
- **Custom Metal kernels**, compiled at run time. Each one is tested bit for bit (or against a float64 reference) against the MLX ops it replaces.
- **No Xcode:** it links the official prebuilt MLX, so there is no Metal compiler and no CMake.

## Performance

Mac Studio M4 Max (40-core GPU, 48 GB), compared with a patched Ollama 0.40.17 on the same machine:

| Measure | Ollama | Calamardo |
|---|---|---|
| Decode at 29k context, with drafter | 36.7 tok/s | ~49 tok/s (median) |
| Prefill at ~145k context | ~90 tok/s | 401 tok/s |
| Cold prefill of 164k tokens | ~15 min | 5.3 min |
| Context kept between turns | ~150k tokens | 516k (RAM and disk) |

At 516k tokens of context with an 8-bit cache, decode runs at 20.8 tok/s with the drafter and 17.4 tok/s without it.

## Requirements

- An Apple Silicon Mac with macOS 26 or later. Development and measurements use an M4 Max with 48 GB; the weights take about 20 GB. The model accepts up to 1,048,576 tokens of context, and Calamardo serves prompts up to that length; how much of it fits depends on the memory left after the weights and on the KV cache bits.
- Rust 1.89 or later (edition 2024).
- `curl`, `unzip` and `shasum`, to download MLX.
- Python 3 for `scripts/bench.py` (standard library only). The reference check (`scripts/check-reference.sh`) uses Python 3.12 and creates its own virtual environment.

## Installation

1. Download the prebuilt MLX 0.32.4. The script takes the `mlx-metal` wheel from PyPI, verifies its sha256 and unpacks it into `third_party/mlx`:

   ```bash
   scripts/fetch-mlx.sh
   ```

2. Build:

   ```bash
   cargo build --release
   ```

3. Download the model into the Hugging Face cache. Calamardo reads it from there and never goes online. The drafter (`mtp.safetensors`) is part of the same checkpoint.

   ```bash
   hf download thinkingmachines/Inkling-2-Mini
   ```

   `--repo` and `--revision` pick another repository or revision from the cache; `--model-dir` (or `CALAMARDO_MODEL_DIR`) uses any directory.

## Usage

### Server

```bash
target/release/calamardo serve
```

It listens on `127.0.0.1:11434`, Ollama's port. To serve the local network:

```bash
target/release/calamardo serve --addr 0.0.0.0:11434 --api-key "$CALAMARDO_API_KEY"
```

At startup it prints the environment (macOS, chip, memory, GPU, MLX version, and the `CALAMARDO_*`, `MLX_*` and `MTL_*` variables) and the configuration. For each request served, it prints a line with prefill and generation speed in tok/s and the drafter's acceptance; at shutdown, a summary.

Main options (`calamardo serve --help` lists them all):

| Option | Default | What it does |
|---|---|---|
| `--kv-bits` | 8 | Bits of the global passes' KV cache: 16, 8 or 4 |
| `--cache-gb` | 16 | RAM for the prefix cache |
| `--disk-cache-dir` | `/tmp/calamardo-cache` | Prefix cache on disk |
| `--disk-cache-gb` | 50 | Disk space; 0 disables it |
| `--keep-disk-cache` | off | Keep the disk cache at exit and preload the latest conversation at startup; otherwise it is deleted at exit |
| `--api-key` | none | Token required as `Authorization: Bearer` or `x-api-key` |
| `--default-reasoning-effort` | the template's | Reasoning effort when a request gives none |
| `--no-draft` | | Disable speculative decoding |
| `--queue`, `--max-queue-wait` | 8, 900 s | Queue for the single slot; when it is full or the wait runs out, 429 |

Ctrl-C stops the running request and shuts down; a second signal exits at once.

### Endpoints

| Route | API |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions: streaming, tools, `reasoning_effort`, `response_format` |
| `POST /v1/completions` | OpenAI, plain text or token ids |
| `POST /v1/responses` | OpenAI Responses, stateless |
| `GET /v1/models` | OpenAI |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | Anthropic Messages: `thinking`, tools, `stop_sequences` |
| `POST /api/chat`, `POST /api/generate` | Ollama (NDJSON) |
| `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version` | Ollama |
| `GET /health`, `GET /metrics` | Health and Prometheus metrics |

All of them share the slot, the prefix cache, and the parser for reasoning and tool calls. Any model name is accepted. Reasoning comes back in `reasoning` (Chat Completions), `reasoning` items (Responses), `thinking` blocks (Anthropic) or `thinking` (Ollama).

```bash
curl http://127.0.0.1:11434/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "reasoning_effort": "low"
}'
```

Limitations: text only (images and documents are replaced by a note saying they were omitted), one conversation at a time, and `tool_choice` only honours `"none"`. JSON output is requested with an instruction, without constrained decoding.

### Other commands

```bash
target/release/calamardo generate --prompt "Hello" --max-tokens 64   # greedy generation
target/release/calamardo inspect                                     # checkpoint tensors, formats and memory
target/release/calamardo kv-eval tests/prompts/ice.txt               # 8- and 4-bit cache quality against bf16
```

## Verification

- `cargo test --release` runs the kernel, fusion and server tests. The ones that need the model read it from the Hugging Face cache. The benchmarks (`crates/calamardo-model/tests/bench_*.rs`) are ignored by default; run one with, for example, `cargo test --release -p calamardo-model --test bench_long -- --ignored --nocapture`.
- `scripts/check-reference.sh tests/prompts/ice.txt` compares greedy generation with a bf16 cache against mlx-lm, token by token.
- `scripts/bench.py` is the reproducible benchmark. It measures prefill and decode from the client side on any OpenAI-compatible server, Ollama included. Every prompt starts with a nonce, so no cache is reused, and rounds alternate the order of the versions:

  ```bash
  scripts/bench.py \
    --target draft=http://127.0.0.1:8093 --launch "draft=target/release/calamardo serve --addr 127.0.0.1:8093" \
    --target nodraft=http://127.0.0.1:8094 --launch "nodraft=target/release/calamardo serve --addr 127.0.0.1:8094 --no-draft" \
    --context 1000,8000,30000 --rounds 3
  ```

## Layout

| Crate | Contents |
|---|---|
| `crates/calamardo` | The binary: command line |
| `crates/calamardo-server` | HTTP server (axum), APIs, queue, on-disk prefix cache and metrics |
| `crates/calamardo-model` | The model, the drafter, KV caches, Metal kernels and generation |
| `crates/calamardo-ckpt` | Safetensors and checkpoint reading |
| `crates/mlx-sys` | MLX-C bindings over the prebuilt MLX, instead of building it with CMake |

## License

[MIT](LICENSE).
