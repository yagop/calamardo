# calamardo-server

The HTTP server behind `calamardo serve`: OpenAI, Anthropic and Ollama APIs over a single model slot, with a prefix cache in RAM and on disk.

| Module | Contents |
|---|---|
| `server` | Router (axum), `ServerConfig`, authentication, CORS for local origins, shutdown, and OpenAI Chat Completions and Models. Also the helpers every API shares: queueing a job, driving its events, collecting reasoning, content and tool calls, and stop sequences |
| `openai` | `/v1/completions` and `/v1/responses` |
| `anthropic` | `/v1/messages` and `/v1/messages/count_tokens` |
| `ollama` | `/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/ps` and `/api/version` |
| `worker` | The single slot: one thread owns the model (MLX state is per thread) and serves jobs from a bounded queue one at a time. Holds the prefix cache in RAM, runs speculative decoding, and prints a tok/s line per request and a summary at shutdown |
| `disk` | The prefix cache on the SSD: one safetensors file per conversation, with a manifest in its metadata (model, format, KV cache bits). One process per directory (lock file); only cache-named files are read or deleted. Deleted at exit unless `--keep-disk-cache` |
| `template` | The checkpoint's `chat_template.jinja`, rendered with minijinja exactly as Hugging Face transformers renders it |
| `parser` | Streams the model's output into reasoning, content and tool calls, following the checkpoint's `response_template` |
| `metrics` | Prometheus metrics (`/metrics`): speeds, tokens, queue, cache sizes and the memory peak of every prefill step |
| `report` | The startup report: system, GPU, MLX, environment variables and configuration |

Requests flow the same way on every API: the body becomes chat messages, the template renders them, and the prompt goes to the queue. The worker reuses the longest cached prefix, prefills the rest and streams tokens back; the parser splits them, and each API formats the deltas its own way. A client that disconnects stops the generation, and what was already processed stays in the cache.

Prompts are limited by the model's context, read from `config.json` (`text_config.model_max_length`, 1,048,576 tokens): longer ones are refused, and `max_tokens` is capped so prompt and reply fit.

## Tests

- `cargo test -p calamardo-server` runs the unit tests (the request mappings of each API) and `tests/parser.rs` and `tests/template.rs`.
- The template test renders the cases in `tests/data/template_cases.json` and compares them with the output of transformers' `apply_chat_template`.
- The APIs are checked live with curl, and performance with [`scripts/bench.py`](../../scripts/bench.py).
