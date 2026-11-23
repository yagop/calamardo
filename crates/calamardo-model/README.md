# calamardo-model

Inkling 2 Mini on MLX: weights, caches, Metal kernels, the forward pass, the DSpark drafter and generation. Text only, batch of one.

The model is ported op by op from mlx-lm (`models/inkling2`), so a greedy run with a bf16 cache produces exactly the reference's tokens. Every optimization keeps that property or documents where rounding changes: fused kernels are tested bit for bit against the ops they replace, and attention kernels against a float64 reference.

| Module | Contents |
|---|---|
| `model` | `Model`: loads the weights (nvfp4 and mxfp8 experts stay quantized, activations are bf16) and runs the forward pass over the 48 passes: sliding-window and global attention with relative position bias, short convolutions, message gates and the MoE |
| `cache` | Per-pass caches: `KvCache` for global passes (bf16, or quantized to 8 or 4 bits with `KvFormat`), `RotatingKvCache` for sliding windows, and the short-convolution states. Same layouts and update order as mlx-lm |
| `kernels` | Metal kernels compiled at run time: the reference's kernels, plus decode attention over the quantized cache, block attention for speculative verification (8×8 simdgroup matrices), a shared expert gather, decode masks and a fused short-convolution step |
| `draft` | The bundled DSpark drafter: three layers that read the target's hidden states at passes 5, 23 and 41, with a Markov head folded into the target's vocabulary projection. Proposes up to 7 tokens per round |
| `generate` | `Session`: chunked prefill, greedy and sampled decode, speculative rounds, and the prefix-cache operations (snapshots, reuse, fork, save and load) |
| `ops` | Thin wrappers over MLX-C, one per MLX op, with explicit scalar dtypes; `Compiled` wraps `mlx_compile` |

The context limit is the model's: 1,048,576 tokens. Prefill runs in chunks, so the memory transient stays bounded (under 3 GB above active memory at 516k tokens).

## Tests

GPU tests run with `cargo test --release -p calamardo-model`; the ones that need the model read it from the Hugging Face cache.

- `fused`, `expert_kernel`, `quant_attention`: kernels and fusions against the ops they replace.
- `speculative`: speculative decoding produces the same tokens as plain greedy decoding.
- `prefix`: returning to a snapshot or a fork reproduces a cold run.

The benchmarks are ignored by default:

| Test | Measures |
|---|---|
| `bench_forward` | Forward cost by token count (decode, prefill chunk, verification) |
| `bench_moe`, `bench_matmul` | Experts and dense projections with a few rows |
| `bench_decode_profile` | Where a decode step's time goes: graph by primitive, timing, dispatch cost, GPU capture |
| `bench_verify` | Where a speculative round spends its time |
| `bench_long` | Decode and speculative rounds on a saved long session (e.g. 512k tokens) |
| `bench_acceptance` | The drafter's acceptance curve by position |

```bash
cargo test --release -p calamardo-model --test bench_long -- --ignored --nocapture
```
