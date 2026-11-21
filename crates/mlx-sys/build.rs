//! Compiles the vendored MLX-C against the headers of the prebuilt MLX and
//! links libmlx.dylib dynamically. No CMake, no Metal compiler.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

/// MLX version the vendored MLX-C revision targets. Must match the wheel.
const MLX_VERSION: &str = "0.32.2";

fn mlx_dir() -> PathBuf {
    if let Some(d) = env::var_os("CALAMARDO_MLX_DIR") {
        return PathBuf::from(d);
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest.join("../../third_party/mlx")
}

fn main() {
    println!("cargo:rerun-if-env-changed=CALAMARDO_MLX_DIR");
    println!("cargo:rerun-if-changed=vendor");

    let mlx = mlx_dir();
    let version_file = mlx.join("VERSION");
    println!("cargo:rerun-if-changed={}", version_file.display());
    let version = fs::read_to_string(&version_file).unwrap_or_else(|_| {
        panic!(
            "prebuilt MLX not found in {}; run scripts/fetch-mlx.sh",
            mlx.display()
        )
    });
    assert_eq!(
        version.trim(),
        MLX_VERSION,
        "MLX in {} is {}, but MLX-C needs {MLX_VERSION}; run scripts/fetch-mlx.sh",
        mlx.display(),
        version.trim()
    );
    let mlx = mlx.canonicalize().expect("canonicalize MLX dir");
    let include = mlx.join("include");
    let lib = mlx.join("lib");

    let mlx_c = PathBuf::from("vendor/mlx-c");
    let shim = PathBuf::from("vendor/shim");

    let mut sources: Vec<PathBuf> = fs::read_dir(mlx_c.join("mlx/c"))
        .expect("read vendored MLX-C")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "cpp"))
        .collect();
    sources.sort();

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .files(&sources)
        .include(&mlx_c)
        .include(&shim)
        .include(&include)
        .include(include.join("metal_cpp"))
        // Same definitions MLX exports in MLXConfig.cmake.
        .define("_METAL_", None)
        .define("ACCELERATE_NEW_LAPACK", None)
        .flag("-fvisibility=hidden")
        .warnings(false)
        .compile("mlxc");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=c++");
    // Dependents (with a direct dependency on this crate) read this as
    // DEP_MLX_LIB_DIR to set the rpath of their binaries.
    println!("cargo:lib_dir={}", lib.display());
    println!("cargo:version={MLX_VERSION}");
    // This crate's own tests and examples.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());

    let headers = discover_headers(&mlx_c);
    let bindings = headers
        .iter()
        .fold(
            bindgen::Builder::default()
                .rust_target("1.73.0".parse().expect("rust-version"))
                .clang_arg(format!("-I{}", mlx_c.display()))
                // Same blocklist as upstream mlx-sys: bindgen's `c_ulong`
                // sizes trip rustc's suspicious_runtime_symbol_definitions.
                .blocklist_function("malloc")
                .blocklist_function("realloc"),
            |b, h| b.header(h.to_string_lossy()),
        )
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("generate MLX-C bindings");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings");
}

/// Public MLX-C headers (`mlx/c/*.h`), in the same order as upstream mlx-sys.
fn discover_headers(mlx_c: &Path) -> Vec<PathBuf> {
    let mut headers: Vec<PathBuf> = fs::read_dir(mlx_c.join("mlx/c"))
        .expect("read vendored MLX-C headers")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "h"))
        .collect();
    headers.sort();
    headers
}
