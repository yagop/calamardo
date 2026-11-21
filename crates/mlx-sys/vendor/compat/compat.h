// MLX-C 0.7.0 signature of mlx_gather_qmm, which mlx-rs 0.34 calls.
//
// MLX-C 0.8.0 added a `global_scale` argument. build.rs renames the 0.8.0
// function to `mlx_gather_qmm_gs` (exposed to Rust with the new signature)
// and compat.cpp defines this one on top of it with no global scale.
#pragma once

#include "mlx/c/array.h"
#include "mlx/c/optional.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

int mlx_gather_qmm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases /* may be null */,
    const mlx_array lhs_indices /* may be null */,
    const mlx_array rhs_indices /* may be null */,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    bool sorted_indices,
    const mlx_stream s);

#ifdef __cplusplus
}
#endif
