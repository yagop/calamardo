// See compat.h. Compiled with -Dmlx_gather_qmm=mlx_gather_qmm_gs like the
// rest of MLX-C, so ops.h declares the 0.8.0 function under its new name.
#include "mlx/c/ops.h"

#undef mlx_gather_qmm
#include "compat.h"

extern "C" int mlx_gather_qmm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases,
    const mlx_array lhs_indices,
    const mlx_array rhs_indices,
    bool transpose,
    mlx_optional_int group_size,
    mlx_optional_int bits,
    const char* mode,
    bool sorted_indices,
    const mlx_stream s) {
  const mlx_array no_global_scale = {nullptr};
  return mlx_gather_qmm_gs(
      res,
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      transpose,
      group_size,
      bits,
      mode,
      no_global_scale,
      sorted_indices,
      s);
}
