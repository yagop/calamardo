//! Points the binary's rpath at the prebuilt libmlx.dylib that mlx-sys linked.
//! MLX finds mlx.metallib next to libmlx.dylib.

fn main() {
    if let Ok(dir) = std::env::var("DEP_MLX_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
}
