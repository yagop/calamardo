//! Thin wrappers over MLX-C.
//!
//! The model mirrors the Python reference op by op, so every op here maps to
//! exactly one MLX op with the same arguments Python would pass. Scalars are
//! explicit arrays: Python's weakly typed scalars take the dtype of the other
//! operand, so callers pass the dtype they need.

use std::ffi::{CStr, CString, c_char};

use anyhow::{Result, bail};
use mlx_rs::{Array, Dtype, Stream};
use mlx_sys as sys;

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

extern "C" fn on_error(msg: *const c_char, _data: *mut std::ffi::c_void) {
    // SAFETY: MLX-C passes a valid C string for the duration of the call.
    let msg = unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg));
}

/// Routes MLX-C errors to this module so failures carry MLX's message.
///
/// mlx-rs installs its own handler once, on its first fallible call, and
/// keeps the message where only mlx-rs can read it. This forces that install
/// first and then replaces it, so this handler stays in place; errors from
/// the mlx-rs calls we still make (Metal kernels) get the message attached
/// through [`with_mlx_message`]. Idempotent; call before any MLX work.
pub fn init() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // Any fallible mlx-rs call triggers its one-time handler install.
        let _ = mlx_rs::fast::MetalKernel::new(
            "calamardo_init",
            &["x"],
            &["y"],
            "y[0] = x[0];",
            Default::default(),
        );
        // SAFETY: the handler has static lifetime and no user data.
        unsafe { sys::mlx_set_error_handler(Some(on_error), std::ptr::null_mut(), None) };
    });
}

/// Adds MLX's message to an error returned by mlx-rs.
pub fn with_mlx_message<T>(r: std::result::Result<T, mlx_rs::error::Exception>) -> Result<T> {
    r.map_err(|e| anyhow::anyhow!("{e}: {}", last_error()))
}

/// Frees MLX's buffer cache (`mx.clear_cache()`).
pub fn clear_cache() {
    // SAFETY: no arguments; only returns memory to the system.
    unsafe { sys::mlx_clear_cache() };
}

fn last_error() -> String {
    LAST_ERROR
        .with(|e| e.borrow_mut().take())
        .unwrap_or_else(|| "no message".to_string())
}

fn stream() -> Stream {
    Stream::thread_local_or_default()
}

fn dt(d: Dtype) -> sys::mlx_dtype {
    d as u32 as sys::mlx_dtype
}

/// Runs an MLX-C op that writes one array.
fn op(name: &str, f: impl FnOnce(&mut sys::mlx_array, sys::mlx_stream) -> i32) -> Result<Array> {
    let s = stream();
    // SAFETY: `res` is a fresh handle that MLX-C fills; ownership moves into
    // the returned `Array`, or it is freed on failure.
    unsafe {
        let mut res = sys::mlx_array_new();
        if f(&mut res, s.as_ptr()) != 0 {
            sys::mlx_array_free(res);
            bail!("MLX op {name} failed: {}", last_error());
        }
        Ok(Array::from_ptr(res))
    }
}

/// A borrowed list of arrays as an MLX-C vector.
struct Vector(sys::mlx_vector_array);

impl Vector {
    fn new(arrays: &[&Array]) -> Self {
        let ptrs: Vec<sys::mlx_array> = arrays.iter().map(|a| a.as_ptr()).collect();
        // SAFETY: MLX-C copies the handles (it retains each array).
        Self(unsafe { sys::mlx_vector_array_new_data(ptrs.as_ptr(), ptrs.len()) })
    }

    fn get(&self, i: usize) -> Result<Array> {
        // SAFETY: `i` is checked by MLX-C; the result is a new owned handle.
        unsafe {
            let mut a = sys::mlx_array_new();
            if sys::mlx_vector_array_get(&mut a, self.0, i) != 0 {
                sys::mlx_array_free(a);
                bail!("vector index {i} out of range");
            }
            Ok(Array::from_ptr(a))
        }
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        // SAFETY: the vector is owned and freed once.
        unsafe { sys::mlx_vector_array_free(self.0) };
    }
}

fn null() -> sys::mlx_array {
    // SAFETY: an empty handle, used for optional arguments.
    unsafe { sys::mlx_array_new() }
}

// ---- Constructors -------------------------------------------------------

/// 0-d array of `v` in `dtype` (a Python scalar next to an array of `dtype`).
pub fn scalar(v: f32, dtype: Dtype) -> Result<Array> {
    let a = Array::from_f32(v);
    if dtype == Dtype::Float32 {
        Ok(a)
    } else {
        astype(&a, dtype)
    }
}

pub fn f32s(v: f32) -> Array {
    Array::from_f32(v)
}

pub fn i32s(v: i32) -> Array {
    Array::from_int(v)
}

pub fn arange(start: i32, stop: i32) -> Result<Array> {
    op("arange", |r, s| unsafe {
        sys::mlx_arange(r, start as f64, stop as f64, 1.0, dt(Dtype::Int32), s)
    })
}

pub fn zeros(shape: &[i32], dtype: Dtype) -> Result<Array> {
    op("zeros", |r, s| unsafe {
        sys::mlx_zeros(r, shape.as_ptr(), shape.len(), dt(dtype), s)
    })
}

// ---- Elementwise --------------------------------------------------------

macro_rules! binary {
    ($($name:ident => $c:ident),* $(,)?) => {$(
        pub fn $name(a: &Array, b: &Array) -> Result<Array> {
            op(stringify!($name), |r, s| unsafe { sys::$c(r, a.as_ptr(), b.as_ptr(), s) })
        }
    )*};
}

macro_rules! unary {
    ($($name:ident => $c:ident),* $(,)?) => {$(
        pub fn $name(a: &Array) -> Result<Array> {
            op(stringify!($name), |r, s| unsafe { sys::$c(r, a.as_ptr(), s) })
        }
    )*};
}

binary! {
    add => mlx_add,
    sub => mlx_subtract,
    mul => mlx_multiply,
    div => mlx_divide,
    maximum => mlx_maximum,
    minimum => mlx_minimum,
    matmul => mlx_matmul,
    less => mlx_less,
    greater_equal => mlx_greater_equal,
    logical_and => mlx_logical_and,
    remainder => mlx_remainder,
    floor_divide => mlx_floor_divide,
}

unary! {
    sqrt => mlx_sqrt,
    exp => mlx_exp,
    tanh => mlx_tanh,
    sigmoid => mlx_sigmoid,
    log => mlx_log,
}

pub fn where_(cond: &Array, x: &Array, y: &Array) -> Result<Array> {
    op("where", |r, s| unsafe {
        sys::mlx_where(r, cond.as_ptr(), x.as_ptr(), y.as_ptr(), s)
    })
}

pub fn clip(a: &Array, lo: &Array, hi: &Array) -> Result<Array> {
    op("clip", |r, s| unsafe {
        sys::mlx_clip(r, a.as_ptr(), lo.as_ptr(), hi.as_ptr(), s)
    })
}

pub fn astype(a: &Array, dtype: Dtype) -> Result<Array> {
    op("astype", |r, s| unsafe {
        sys::mlx_astype(r, a.as_ptr(), dt(dtype), s)
    })
}

pub fn f32(a: &Array) -> Result<Array> {
    astype(a, Dtype::Float32)
}

/// Reinterprets the bytes of `a` as `dtype` (`mx.array.view`).
pub fn view(a: &Array, dtype: Dtype) -> Result<Array> {
    op("view", |r, s| unsafe {
        sys::mlx_view(r, a.as_ptr(), dt(dtype), s)
    })
}

/// `mx.contiguous(a)`.
pub fn contiguous(a: &Array) -> Result<Array> {
    op("contiguous", |r, s| unsafe {
        sys::mlx_contiguous(r, a.as_ptr(), false, s)
    })
}

// ---- Shape --------------------------------------------------------------

pub fn reshape(a: &Array, shape: &[i32]) -> Result<Array> {
    op("reshape", |r, s| unsafe {
        sys::mlx_reshape(r, a.as_ptr(), shape.as_ptr(), shape.len(), s)
    })
}

pub fn transpose(a: &Array, axes: &[i32]) -> Result<Array> {
    op("transpose", |r, s| unsafe {
        sys::mlx_transpose_axes(r, a.as_ptr(), axes.as_ptr(), axes.len(), s)
    })
}

/// `a.T` for a 2-d array.
pub fn t(a: &Array) -> Result<Array> {
    transpose(a, &[1, 0])
}

pub fn swapaxes(a: &Array, x: i32, y: i32) -> Result<Array> {
    op("swapaxes", |r, s| unsafe {
        sys::mlx_swapaxes(r, a.as_ptr(), x, y, s)
    })
}

pub fn expand_dims(a: &Array, axes: &[i32]) -> Result<Array> {
    op("expand_dims", |r, s| unsafe {
        sys::mlx_expand_dims_axes(r, a.as_ptr(), axes.as_ptr(), axes.len(), s)
    })
}

pub fn squeeze(a: &Array, axes: &[i32]) -> Result<Array> {
    op("squeeze", |r, s| unsafe {
        sys::mlx_squeeze_axes(r, a.as_ptr(), axes.as_ptr(), axes.len(), s)
    })
}

pub fn flatten(a: &Array, start: i32, end: i32) -> Result<Array> {
    op("flatten", |r, s| unsafe {
        sys::mlx_flatten(r, a.as_ptr(), start, end, s)
    })
}

pub fn unflatten(a: &Array, axis: i32, shape: &[i32]) -> Result<Array> {
    op("unflatten", |r, s| unsafe {
        sys::mlx_unflatten(r, a.as_ptr(), axis, shape.as_ptr(), shape.len(), s)
    })
}

pub fn broadcast_to(a: &Array, shape: &[i32]) -> Result<Array> {
    op("broadcast_to", |r, s| unsafe {
        sys::mlx_broadcast_to(r, a.as_ptr(), shape.as_ptr(), shape.len(), s)
    })
}

pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    let v = Vector::new(arrays);
    op("concatenate", |r, s| unsafe {
        sys::mlx_concatenate_axis(r, v.0, axis, s)
    })
}

pub fn stack(arrays: &[&Array], axis: i32) -> Result<Array> {
    let v = Vector::new(arrays);
    op("stack", |r, s| unsafe {
        sys::mlx_stack_axis(r, v.0, axis, s)
    })
}

/// `a[start:stop:step]` along `axis`, full range on the other axes.
pub fn slice(a: &Array, axis: i32, start: i32, stop: i32, step: i32) -> Result<Array> {
    let shape = a.shape();
    let axis = axis.rem_euclid(shape.len() as i32) as usize;
    let mut lo = vec![0; shape.len()];
    let mut hi = shape.to_vec();
    let mut st = vec![1; shape.len()];
    lo[axis] = start;
    hi[axis] = stop;
    st[axis] = step;
    op("slice", |r, s| unsafe {
        sys::mlx_slice(
            r,
            a.as_ptr(),
            lo.as_ptr(),
            lo.len(),
            hi.as_ptr(),
            hi.len(),
            st.as_ptr(),
            st.len(),
            s,
        )
    })
}

/// `src[..., start:stop, ...] = update` along `axis` (a new array).
pub fn slice_update(
    src: &Array,
    update: &Array,
    axis: i32,
    start: i32,
    stop: i32,
) -> Result<Array> {
    let shape = src.shape();
    let axis = axis.rem_euclid(shape.len() as i32) as usize;
    let mut lo = vec![0; shape.len()];
    let mut hi = shape.to_vec();
    let st = vec![1; shape.len()];
    lo[axis] = start;
    hi[axis] = stop;
    op("slice_update", |r, s| unsafe {
        sys::mlx_slice_update(
            r,
            src.as_ptr(),
            update.as_ptr(),
            lo.as_ptr(),
            lo.len(),
            hi.as_ptr(),
            hi.len(),
            st.as_ptr(),
            st.len(),
            s,
        )
    })
}

// ---- Indexing -----------------------------------------------------------

pub fn take(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    op("take", |r, s| unsafe {
        sys::mlx_take_axis(r, a.as_ptr(), indices.as_ptr(), axis, s)
    })
}

pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    op("take_along_axis", |r, s| unsafe {
        sys::mlx_take_along_axis(r, a.as_ptr(), indices.as_ptr(), axis, s)
    })
}

pub fn argsort(a: &Array, axis: i32) -> Result<Array> {
    op("argsort", |r, s| unsafe {
        sys::mlx_argsort_axis(r, a.as_ptr(), axis, s)
    })
}

pub fn argmax(a: &Array, axis: i32) -> Result<Array> {
    op("argmax", |r, s| unsafe {
        sys::mlx_argmax_axis(r, a.as_ptr(), axis, false, s)
    })
}

// ---- Reductions ---------------------------------------------------------

pub fn sum(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
    op("sum", |r, s| unsafe {
        sys::mlx_sum_axes(r, a.as_ptr(), axes.as_ptr(), axes.len(), keepdims, s)
    })
}

pub fn softmax(a: &Array, axes: &[i32], precise: bool) -> Result<Array> {
    op("softmax", |r, s| unsafe {
        sys::mlx_softmax_axes(r, a.as_ptr(), axes.as_ptr(), axes.len(), precise, s)
    })
}

pub fn logsumexp(a: &Array, axes: &[i32], keepdims: bool) -> Result<Array> {
    op("logsumexp", |r, s| unsafe {
        sys::mlx_logsumexp_axes(r, a.as_ptr(), axes.as_ptr(), axes.len(), keepdims, s)
    })
}

// ---- Neural network ops -------------------------------------------------

/// `mx.fast.rms_norm(x, weight, eps)`.
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
    let w = weight.map_or_else(null, |w| w.as_ptr());
    op("rms_norm", |r, s| unsafe {
        sys::mlx_fast_rms_norm(r, x.as_ptr(), w, eps, s)
    })
}

/// `nn.Linear` without bias: `x @ w.T`.
pub fn linear(x: &Array, w: &Array) -> Result<Array> {
    matmul(x, &t(w)?)
}

pub fn conv1d(x: &Array, w: &Array, groups: i32) -> Result<Array> {
    op("conv1d", |r, s| unsafe {
        sys::mlx_conv1d(r, x.as_ptr(), w.as_ptr(), 1, 0, 1, groups, s)
    })
}

/// `mx.fast.scaled_dot_product_attention` with an additive mask and sinks.
pub fn sdpa(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: &Array,
    sinks: &Array,
) -> Result<Array> {
    op("scaled_dot_product_attention", |r, s| unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            r,
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            scale,
            c"".as_ptr(),
            mask.as_ptr(),
            sinks.as_ptr(),
            false,
            s,
        )
    })
}

/// `mx.gather_qmm(x, w, scales, None, rhs_indices=..., transpose=True, ...)`
/// with an optional nvfp4 global scale.
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    rhs_indices: &Array,
    group_size: i32,
    bits: i32,
    mode: &CStr,
    global_scale: Option<&Array>,
    sorted_indices: bool,
) -> Result<Array> {
    let gs = global_scale.map_or_else(null, |a| a.as_ptr());
    let group = sys::mlx_optional_int {
        value: group_size,
        has_value: true,
    };
    let bits = sys::mlx_optional_int {
        value: bits,
        has_value: true,
    };
    op("gather_qmm", |r, s| unsafe {
        sys::mlx_gather_qmm_gs(
            r,
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            null(),
            null(),
            rhs_indices.as_ptr(),
            true,
            group,
            bits,
            mode.as_ptr(),
            gs,
            sorted_indices,
            s,
        )
    })
}

/// `mx.quantize(w, group_size, bits)` (affine): packed u32, scales, biases.
pub fn quantize_affine(w: &Array, group_size: i32, bits: i32) -> Result<(Array, Array, Array)> {
    let s = stream();
    let opt = |v: i32| sys::mlx_optional_int {
        value: v,
        has_value: true,
    };
    // SAFETY: `res` is a fresh vector, wrapped so it is freed.
    let res = unsafe {
        let mut res = sys::mlx_vector_array_new();
        if sys::mlx_quantize(
            &mut res,
            w.as_ptr(),
            opt(group_size),
            opt(bits),
            c"affine".as_ptr(),
            null(),
            s.as_ptr(),
        ) != 0
        {
            sys::mlx_vector_array_free(res);
            bail!("MLX op quantize failed: {}", last_error());
        }
        Vector(res)
    };
    Ok((res.get(0)?, res.get(1)?, res.get(2)?))
}

/// `mx.dequantize(w, scales, biases, group_size, bits)` (affine), in the
/// dtype of the scales.
pub fn dequantize_affine(
    w: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> Result<Array> {
    let opt = |v: i32| sys::mlx_optional_int {
        value: v,
        has_value: true,
    };
    let no_dtype = sys::mlx_optional_dtype {
        value: 0,
        has_value: false,
    };
    op("dequantize", |r, s| unsafe {
        sys::mlx_dequantize(
            r,
            w.as_ptr(),
            scales.as_ptr(),
            biases.as_ptr(),
            opt(group_size),
            opt(bits),
            c"affine".as_ptr(),
            null(),
            no_dtype,
            s,
        )
    })
}

/// `mx.depends(inputs, dependencies)`: `inputs` are only computed after
/// `dependencies`.
pub fn depends(inputs: &[&Array], dependencies: &[&Array]) -> Result<Vec<Array>> {
    let ins = Vector::new(inputs);
    let deps = Vector::new(dependencies);
    // SAFETY: `res` is a fresh vector, wrapped so it is freed.
    let res = unsafe {
        let mut res = sys::mlx_vector_array_new();
        if sys::mlx_depends(&mut res, ins.0, deps.0) != 0 {
            sys::mlx_vector_array_free(res);
            bail!("MLX op depends failed: {}", last_error());
        }
        Vector(res)
    };
    (0..inputs.len()).map(|i| res.get(i)).collect()
}

// ---- I/O ----------------------------------------------------------------

/// `mx.load(path)` of a safetensors file: lazily loaded arrays by name.
pub fn load_safetensors(path: &std::path::Path) -> Result<Vec<(String, Array)>> {
    let file = CString::new(path.to_string_lossy().as_bytes())?;
    let mut out = Vec::new();
    // SAFETY: the stream, maps and iterator are created and freed here; each
    // array handle is copied into an owned `Array`.
    unsafe {
        // Loading only runs on the CPU, as with `mx.load`.
        let cpu = sys::mlx_default_cpu_stream_new();
        let mut arrays = sys::mlx_map_string_to_array_new();
        let mut meta = sys::mlx_map_string_to_string_new();
        let status = sys::mlx_load_safetensors(&mut arrays, &mut meta, file.as_ptr(), cpu);
        sys::mlx_stream_free(cpu);
        sys::mlx_map_string_to_string_free(meta);
        if status != 0 {
            sys::mlx_map_string_to_array_free(arrays);
            bail!("loading {} failed: {}", path.display(), last_error());
        }
        let it = sys::mlx_map_string_to_array_iterator_new(arrays);
        loop {
            let mut key: *const c_char = std::ptr::null();
            let mut value = sys::mlx_array_new();
            if sys::mlx_map_string_to_array_iterator_next(&mut key, &mut value, it) != 0 {
                sys::mlx_array_free(value);
                break;
            }
            let name = CStr::from_ptr(key).to_string_lossy().into_owned();
            out.push((name, Array::from_ptr(value)));
        }
        sys::mlx_map_string_to_array_iterator_free(it);
        sys::mlx_map_string_to_array_free(arrays);
    }
    Ok(out)
}

/// `mx.eval(arrays)`.
pub fn eval(arrays: &[&Array]) -> Result<()> {
    let v = Vector::new(arrays);
    // SAFETY: evaluates borrowed arrays.
    if unsafe { sys::mlx_eval(v.0) } != 0 {
        bail!("MLX eval failed: {}", last_error());
    }
    Ok(())
}
