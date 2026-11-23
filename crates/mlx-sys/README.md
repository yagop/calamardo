# mlx-sys

Raw bindings to [MLX-C](https://github.com/ml-explore/mlx-c), linked against the official prebuilt MLX instead of building MLX from source. It replaces crates.io `mlx-sys` through `[patch.crates-io]` in the workspace, so `mlx-rs` 0.34 runs on it unchanged.

- **No CMake, no Metal compiler, no Xcode.** `build.rs` compiles the vendored MLX-C sources with the system C++ compiler against the headers of the prebuilt MLX, links `libmlx.dylib` dynamically, and generates the bindings with bindgen.
- **The prebuilt MLX** comes from the `mlx-metal` wheel on PyPI. [`scripts/fetch-mlx.sh`](../../scripts/fetch-mlx.sh) downloads it, checks its sha256 and unpacks it into `third_party/mlx`; `CALAMARDO_MLX_DIR` points elsewhere. The build fails with a hint if the MLX version is not 0.32.4.
- **Downstream rpath:** the build exports the library directory (`DEP_MLX_LIB_DIR`), so binaries can set an rpath to it.

## Vendored code

| Path | Contents |
|---|---|
| `vendor/mlx-c` | MLX-C 0.8.0 (commit a26f2d584a14), the revision pinned to MLX 0.32.4, under its own license |
| `vendor/compat` | The MLX-C 0.7.0 signature of `mlx_gather_qmm`, which `mlx-rs` 0.34 calls. MLX-C 0.8.0 added a `global_scale` argument; `build.rs` renames the new function to `mlx_gather_qmm_gs`, and `compat.cpp` defines the old one on top of it |
| `vendor/shim` | A declaration of `mlx::core::cu::is_available`: the wheel does not ship the CUDA backend headers, but MLX-C's `cuda.cpp` needs this one, and `libmlx.dylib` exports it |

To update MLX, change the version and hash in `fetch-mlx.sh`, vendor the matching MLX-C revision, and update `MLX_VERSION` in `build.rs`.
