# calamardo

The `calamardo` binary: the command line over the other crates.

| Command | What it does |
|---|---|
| `serve` | Runs the HTTP server from [`calamardo-server`](../calamardo-server) (OpenAI, Anthropic and Ollama APIs, one slot) |
| `generate` | Greedy generation from a chat message (`--prompt`), raw text (`--raw`, `--raw-file`) or token ids (`--tokens`); `--kv-bits 16` is the milestone 1 parity check |
| `inspect` | Summarizes the checkpoint: tensors, formats, memory estimate and pass order |
| `kv-eval` | Generates greedily with a bf16 cache, replays the same tokens with each quantized format, and reports the mean KL divergence and top-1 agreement against bf16 |

Every command reads the model from the Hugging Face cache (`--repo`, `--revision`) or from a directory (`--model-dir`, `CALAMARDO_MODEL_DIR`). `calamardo <command> --help` lists each command's options; most also read a `CALAMARDO_*` environment variable.

```bash
calamardo serve --addr 0.0.0.0:11434 --kv-bits 8
calamardo generate --prompt "Hello" --max-tokens 64
calamardo kv-eval tests/prompts/ice.txt --formats 8/64,4/64
```

## Build

`build.rs` adds an rpath to the prebuilt `libmlx.dylib` that [`mlx-sys`](../mlx-sys) linked, so the binary runs without `DYLD_LIBRARY_PATH`. MLX finds `mlx.metallib` next to the library.

## Tests

`tests/mlx_smoke.rs` checks the toolchain end to end: the prebuilt libmlx, our `mlx-sys` and `mlx-rs` computing on the GPU.
