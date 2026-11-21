//! End-to-end check of the MLX toolchain: prebuilt libmlx + our mlx-sys +
//! mlx-rs on the GPU. Covers what the runtime relies on: runtime-compiled
//! Metal kernels (K-5), nvfp4/mxfp8 quantized matmul (M-5), `mlx_depends`
//! (K-1) and peak memory (K-9).

use mlx_rs::{
    Array, Device, Dtype,
    fast::{MetalKernel, MetalKernelOptions, MetalKernelOutput},
    ops::{self, DequantizeOptions, QuantizationMode, QuantizeOptions, QuantizedMatmulOptions},
};

fn vec_f32(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .to_vec_exact::<f32>()
        .unwrap()
}

/// Deterministic pseudo-random values in [-1, 1).
fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn gpu_is_default_and_matmul_works() {
    assert_eq!(Device::default(), Device::gpu());
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);
    assert_eq!(vec_f32(&a.matmul(&b).unwrap()), [19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn runtime_compiled_metal_kernel() {
    let kernel = MetalKernel::new(
        "calamardo_axpb",
        &["x"],
        &["out"],
        "uint i = thread_position_in_grid.x; out[i] = x[i] * 2.0f + 1.0f;",
        MetalKernelOptions {
            ensure_row_contiguous: true,
            ..Default::default()
        },
    )
    .unwrap();
    let n = 4096;
    let x = Array::from_slice(&data(n, 1), &[n as i32]);
    // SAFETY: one thread per element; input and output both hold `n` f32.
    let out = unsafe {
        kernel.apply(
            &[&x],
            &[MetalKernelOutput {
                shape: vec![n as i32],
                dtype: Dtype::Float32,
            }],
            (n as i32, 1, 1),
            (256, 1, 1),
            Default::default(),
        )
    }
    .unwrap();
    let got = vec_f32(&out[0]);
    for (g, x) in got.iter().zip(vec_f32(&x)) {
        assert_eq!(*g, x * 2.0 + 1.0);
    }
}

fn check_quantized(mode: QuantizationMode) {
    let (rows, cols, tokens) = (256, 512, 4);
    let w = Array::from_slice(&data(rows * cols, 2), &[rows as i32, cols as i32]);
    let x = Array::from_slice(&data(tokens * cols, 3), &[tokens as i32, cols as i32]);

    let (wq, scales, biases) = ops::quantize(
        &w,
        QuantizeOptions {
            mode,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(biases.is_none());
    let wd = ops::dequantize(
        &wq,
        &scales,
        None,
        DequantizeOptions {
            mode,
            ..Default::default()
        },
    )
    .unwrap();

    let fused = ops::quantized_matmul(
        &x,
        &wq,
        &scales,
        None,
        QuantizedMatmulOptions {
            mode,
            transpose: true,
            ..Default::default()
        },
    )
    .unwrap();
    let reference = x
        .matmul(wd.as_dtype(Dtype::Float32).unwrap().transpose().unwrap())
        .unwrap();

    let (f, r) = (vec_f32(&fused), vec_f32(&reference));
    assert_eq!(f.len(), tokens * rows);
    let max_err = f
        .iter()
        .zip(&r)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-2, "{mode:?}: max relative error {max_err}");
}

#[test]
fn nvfp4_quantized_matmul() {
    check_quantized(QuantizationMode::Nvfp4);
}

#[test]
fn mxfp8_quantized_matmul() {
    check_quantized(QuantizationMode::Mxfp8);
}

#[test]
fn depends_through_ffi() {
    let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
    let b = a.matmul(Array::from_slice(&[3.0f32, 4.0], &[2])).unwrap();
    // SAFETY: the vectors hold borrowed handles that MLX retains; every
    // handle created here is freed or wrapped in an owning `Array`.
    let out = unsafe {
        let inputs = [a.as_ptr()];
        let deps = [b.as_ptr()];
        let vin = mlx_sys::mlx_vector_array_new_data(inputs.as_ptr(), 1);
        let vdep = mlx_sys::mlx_vector_array_new_data(deps.as_ptr(), 1);
        let mut res = mlx_sys::mlx_vector_array_new();
        assert_eq!(mlx_sys::mlx_depends(&mut res, vin, vdep), 0);
        let mut out = mlx_sys::mlx_array_new();
        assert_eq!(mlx_sys::mlx_vector_array_get(&mut out, res, 0), 0);
        mlx_sys::mlx_vector_array_free(vin);
        mlx_sys::mlx_vector_array_free(vdep);
        mlx_sys::mlx_vector_array_free(res);
        Array::from_ptr(out)
    };
    assert_eq!(vec_f32(&out), [1.0, 2.0]);
}

#[test]
fn peak_memory_is_tracked() {
    mlx_rs::memory::reset_peak_memory().unwrap();
    let x = Array::from_slice(&vec![1.0f32; 1 << 20], &[1 << 20]);
    x.add(&x).unwrap().eval().unwrap();
    assert!(mlx_rs::memory::peak_memory().unwrap() >= 4 << 20);
}
