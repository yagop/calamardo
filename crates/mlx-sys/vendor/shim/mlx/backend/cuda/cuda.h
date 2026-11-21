// The mlx-metal wheel does not ship the CUDA backend headers, but libmlx.dylib
// still exports this symbol (it returns false on Metal builds). MLX-C's
// cuda.cpp only needs this declaration.
#pragma once

namespace mlx::core::cu {

bool is_available();

} // namespace mlx::core::cu
