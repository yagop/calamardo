//! Where the time of one decode step goes (K-6): the lazy graph by primitive,
//! CPU graph build vs encode vs GPU wait, the cost of a dispatch, and the
//! effect of `compile` on small-op chains. `--ignored --nocapture`, one test
//! at a time (`--test-threads=1`).

use std::{
    collections::BTreeMap,
    ffi::{CString, c_void},
    path::Path,
    time::Instant,
};

use calamardo_ckpt::{DEFAULT_REPO, default_hub_cache, snapshot_dir};
use calamardo_model::{
    Model,
    cache::{KvFormat, PassCache},
    ops::{self, f32s},
};
use mlx_rs::{Array, Dtype};
use mlx_sys as sys;

fn rand(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn randn(shape: &[i32], seed: u32, dtype: Dtype) -> Array {
    let n = shape.iter().product::<i32>() as usize;
    let a = ops::astype(&Array::from_slice(&rand(n, seed), shape), dtype).unwrap();
    ops::eval(&[&a]).unwrap();
    a
}

fn vector(arrays: &[&Array]) -> sys::mlx_vector_array {
    let ptrs: Vec<sys::mlx_array> = arrays.iter().map(|a| a.as_ptr()).collect();
    // SAFETY: MLX-C retains each handle; the caller frees the vector.
    unsafe { sys::mlx_vector_array_new_data(ptrs.as_ptr(), ptrs.len()) }
}

/// `mx.async_eval`: encodes and commits the graph, returns before the GPU
/// finishes.
fn async_eval(arrays: &[&Array]) {
    let v = vector(arrays);
    // SAFETY: evaluates borrowed arrays; the vector is freed once.
    unsafe {
        assert_eq!(sys::mlx_async_eval(v), 0);
        sys::mlx_vector_array_free(v);
    }
}

/// Primitive nodes of the unevaluated graph behind `outputs`, by name, as
/// `mx.export_to_dot` lists them (one per primitive, multi-output ones
/// once). The dot file is written to `path`.
fn graph_counts(outputs: &[&Array], path: &Path) -> BTreeMap<String, usize> {
    let c = CString::new(path.to_str().unwrap()).unwrap();
    // SAFETY: a fresh file and namer, closed and freed once.
    unsafe {
        let f = sys::fopen(c.as_ptr(), c"w".as_ptr());
        assert!(!f.is_null(), "cannot write {}", path.display());
        let namer = sys::mlx_node_namer_new();
        let v = vector(outputs);
        assert_eq!(sys::mlx_export_to_dot(f, namer, v), 0);
        sys::mlx_vector_array_free(v);
        sys::mlx_node_namer_free(namer);
        sys::fclose(f);
    }
    let text = std::fs::read_to_string(path).unwrap();
    let mut counts = BTreeMap::new();
    for line in text.lines() {
        if !line.contains("shape=rectangle") {
            continue;
        }
        let Some(start) = line.find("label =\"").map(|i| i + 8) else {
            continue;
        };
        let name = &line[start..];
        let name = &name[..name.find('"').unwrap()];
        // Custom kernels print their name; keep the primitive's kind.
        let name = name.split_whitespace().next().unwrap_or(name).to_string();
        *counts.entry(name).or_insert(0) += 1;
    }
    counts
}

/// Primitives that never dispatch a kernel on the GPU (views, or no-ops on
/// contiguous inputs).
const VIEWS: &[&str] = &[
    "Reshape",
    "Slice",
    "Squeeze",
    "ExpandDims",
    "Transpose",
    "Broadcast",
    "BroadcastAxes",
    "AsStrided",
    "Depends",
    "Unflatten",
    "Flatten",
];

fn print_counts(title: &str, counts: &BTreeMap<String, usize>) {
    let total: usize = counts.values().sum();
    let views: usize = counts
        .iter()
        .filter(|(k, _)| VIEWS.contains(&k.as_str()))
        .map(|(_, v)| v)
        .sum();
    println!(
        "{title}: {total} primitives, {views} views, {} kernels",
        total - views
    );
    let mut by: Vec<_> = counts.iter().collect();
    by.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in by {
        let tag = if VIEWS.contains(&k.as_str()) {
            " (view)"
        } else {
            ""
        };
        println!("  {k:<28} {v:>6}{tag}");
    }
}

fn scratch() -> std::path::PathBuf {
    std::env::var_os("PROFILE_DIR")
        .map(Into::into)
        .unwrap_or_else(std::env::temp_dir)
}

fn load() -> Model {
    let dir = snapshot_dir(&default_hub_cache().unwrap(), DEFAULT_REPO, "main").unwrap();
    Model::load(&dir).unwrap()
}

/// Caches after `n` prompt tokens, prefilled in chunks of 512.
fn prefilled(model: &Model, kv: KvFormat, n: usize) -> Vec<PassCache> {
    let mut cache = model.make_cache(kv);
    let prompt: Vec<i32> = (0..n).map(|i| 1000 + (i * 7919 % 50_000) as i32).collect();
    for chunk in prompt.chunks(512) {
        let t = Array::from_slice(chunk, &[1, chunk.len() as i32]);
        let logits = model.forward(&t, &mut cache).unwrap();
        let mut live = vec![&logits];
        live.extend(cache.iter().flat_map(|c| c.arrays()));
        ops::eval(&live).unwrap();
    }
    cache
}

unsafe extern "C" {
    fn clock_gettime_nsec_np(clock: u32) -> u64;
}

/// CPU time of the calling thread, in seconds (`CLOCK_THREAD_CPUTIME_ID`).
fn thread_cpu() -> f64 {
    const CLOCK_THREAD_CPUTIME_ID: u32 = 16;
    // SAFETY: plain libSystem call.
    unsafe { clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) as f64 * 1e-9 }
}

/// One decode step split into graph build (CPU), encode + commit
/// (`async_eval`) and the wait for the GPU, plus the thread's CPU time
/// inside `async_eval` (less than its wall time when it waits for the GPU).
/// Seconds.
fn timed_step(model: &Model, cache: &mut [PassCache], token: i32) -> (f64, f64, f64, f64) {
    let t0 = Instant::now();
    let logits = model
        .forward(&Array::from_slice(&[token], &[1, 1]), cache)
        .unwrap();
    let t1 = Instant::now();
    let mut live = vec![&logits];
    live.extend(cache.iter().flat_map(|c| c.arrays()));
    let c1 = thread_cpu();
    async_eval(&live);
    let c2 = thread_cpu();
    let t2 = Instant::now();
    ops::eval(&live).unwrap();
    let t3 = Instant::now();
    (
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        (t3 - t2).as_secs_f64(),
        c2 - c1,
    )
}

/// The decode graph at 1k context by primitive, and the step's time split
/// between graph build, encoding and the GPU.
#[test]
#[ignore]
fn decode_graph_and_timing() {
    let model = load();
    for kv in [KvFormat::BF16, KvFormat::new(8, 64).unwrap()] {
        let mut cache = prefilled(&model, kv, 1024);
        // Warm up (kernels compiled, buffers cached).
        for i in 0..3 {
            timed_step(&model, &mut cache, 2000 + i);
        }
        let logits = model
            .forward(&Array::from_slice(&[2003], &[1, 1]), &mut cache)
            .unwrap();
        let mut live = vec![&logits];
        live.extend(cache.iter().flat_map(|c| c.arrays()));
        let counts = graph_counts(&live, &scratch().join(format!("decode_{kv:?}.dot")));
        print_counts(&format!("{kv:?} decode graph"), &counts);
        ops::eval(&live).unwrap();

        let reps = 20;
        let (mut b, mut e, mut w, mut c) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..reps {
            let (x, y, z, cpu) = timed_step(&model, &mut cache, 3000 + i);
            b += x;
            e += y;
            w += z;
            c += cpu;
        }
        let ms = |s: f64| s * 1e3 / reps as f64;
        println!(
            "{kv:?}: build {:.2} ms, encode+commit {:.2} ms (thread CPU {:.2} ms), wait {:.2} ms, total {:.2} ms",
            ms(b),
            ms(e),
            ms(c),
            ms(w),
            ms(b + e + w)
        );
    }
}

/// Whole decode steps with the time split as in `timed_step`
/// (`PROFILE_KV=8` for the 8-bit cache, `PROFILE_REPS` steps). With
/// `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` large enough for one
/// command buffer per step, the GPU starts only after encoding, so the wait
/// is the step's GPU time.
#[test]
#[ignore]
fn decode_step_time() {
    let model = load();
    let kv = match std::env::var("PROFILE_KV").as_deref() {
        Ok("8") => KvFormat::new(8, 64).unwrap(),
        _ => KvFormat::BF16,
    };
    let mut cache = prefilled(&model, kv, 1024);
    for i in 0..3 {
        timed_step(&model, &mut cache, 2000 + i);
    }
    {
        let logits = model
            .forward(&Array::from_slice(&[2003], &[1, 1]), &mut cache)
            .unwrap();
        let mut live = vec![&logits];
        live.extend(cache.iter().flat_map(|c| c.arrays()));
        let counts = graph_counts(&live, &scratch().join("decode_step.dot"));
        let total: usize = counts.values().sum();
        let views: usize = counts
            .iter()
            .filter(|(k, _)| VIEWS.contains(&k.as_str()))
            .map(|(_, v)| v)
            .sum();
        println!(
            "{kv:?} graph: {total} primitives, {views} views, {} kernels",
            total - views
        );
        ops::eval(&live).unwrap();
    }
    let reps: i32 = std::env::var("PROFILE_REPS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(30);
    let mut v: Vec<(f64, f64, f64, f64)> = (0..reps)
        .map(|i| timed_step(&model, &mut cache, 3000 + i))
        .collect();
    v.sort_by(|a, b| (a.0 + a.1 + a.2).total_cmp(&(b.0 + b.1 + b.2)));
    let med = v[reps as usize / 2];
    let mean =
        |f: fn(&(f64, f64, f64, f64)) -> f64| v.iter().map(f).sum::<f64>() * 1e3 / reps as f64;
    println!(
        "{kv:?} decode: median {:.2} ms; mean build {:.2}, encode {:.2} (thread CPU {:.2}), wait {:.2}, total {:.2} ms",
        (med.0 + med.1 + med.2) * 1e3,
        mean(|x| x.0),
        mean(|x| x.1),
        mean(|x| x.3),
        mean(|x| x.2),
        mean(|x| x.0 + x.1 + x.2)
    );
}

type Graph<'a> = dyn Fn() -> Vec<Array> + 'a;

/// Seconds to build and evaluate `f` (median of `reps` after a warm-up),
/// and of that, the CPU time until `async_eval` returns.
fn time_graph(reps: usize, f: &dyn Fn() -> Vec<Array>) -> (f64, f64) {
    for _ in 0..2 {
        let outs = f();
        ops::eval(&outs.iter().collect::<Vec<_>>()).unwrap();
    }
    let mut v: Vec<(f64, f64)> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            let outs = f();
            let refs: Vec<&Array> = outs.iter().collect();
            async_eval(&refs);
            let cpu = t.elapsed().as_secs_f64();
            ops::eval(&refs).unwrap();
            (t.elapsed().as_secs_f64(), cpu)
        })
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    v[reps / 2]
}

/// Cost per dispatch: chains of N dependent small ops in one eval, and N
/// independent ones.
#[test]
#[ignore]
fn dispatch_cost() {
    ops::init();
    let d = 2304;
    let x = randn(&[1, 1, d], 1, Dtype::Float32);
    let xb = randn(&[1, 1, d], 1, Dtype::Bfloat16);
    let c = f32s(1.0001);
    let w_norm = randn(&[d], 2, Dtype::Bfloat16);
    let w_down = randn(&[64, d], 3, Dtype::Bfloat16);
    let w_up = randn(&[d, 64], 4, Dtype::Bfloat16);
    let w_big = randn(&[3072, d], 5, Dtype::Bfloat16);
    let w_back = randn(&[d, 3072], 6, Dtype::Bfloat16);
    let per = |n: usize, s: f64| s * 1e6 / n as f64;
    for n in [100usize, 500, 1000, 2000, 5000] {
        let chain = || {
            let mut y = x.clone();
            for _ in 0..n {
                y = ops::mul(&y, &c).unwrap();
            }
            vec![y]
        };
        let independent = || (0..n).map(|_| ops::mul(&x, &c).unwrap()).collect();
        let (t1, c1) = time_graph(7, &chain);
        let (t2, c2) = time_graph(7, &independent);
        println!(
            "N {n:>5}: chained mul {:.2} us/op (CPU to commit {:.2}), independent {:.2} us/op (CPU {:.2})",
            per(n, t1),
            per(n, c1),
            per(n, t2),
            per(n, c2)
        );
    }
    let n = 1000;
    let cases: Vec<(&str, Box<Graph<'_>>)> = vec![
        (
            "rms_norm chain (2304, bf16)",
            Box::new(|| {
                let mut y = xb.clone();
                for _ in 0..n {
                    y = ops::rms_norm(&y, Some(&w_norm), 1e-6).unwrap();
                }
                vec![y]
            }),
        ),
        (
            "astype f32<->bf16 chain",
            Box::new(|| {
                let mut y = xb.clone();
                for i in 0..n {
                    let to = if i % 2 == 0 {
                        Dtype::Float32
                    } else {
                        Dtype::Bfloat16
                    };
                    y = ops::astype(&y, to).unwrap();
                }
                vec![y]
            }),
        ),
        (
            "matmul 2304->64->2304 chain (per matmul)",
            Box::new(|| {
                let mut y = xb.clone();
                for i in 0..n {
                    let w = if i % 2 == 0 { &w_down } else { &w_up };
                    y = ops::linear(&y, w).unwrap();
                }
                vec![y]
            }),
        ),
        (
            "matmul 2304->3072->2304 chain (per matmul)",
            Box::new(|| {
                let mut y = xb.clone();
                for i in 0..n {
                    let w = if i % 2 == 0 { &w_big } else { &w_back };
                    y = ops::linear(&y, w).unwrap();
                }
                vec![y]
            }),
        ),
        (
            "concatenate 2 x 2304 chain",
            Box::new(|| {
                let mut y = x.clone();
                for _ in 0..n {
                    let z = ops::concatenate(&[&y, &y], -1).unwrap();
                    y = ops::slice(&z, -1, 0, d, 1).unwrap();
                }
                vec![y]
            }),
        ),
    ];
    for (name, f) in cases {
        let (t, c) = time_graph(7, &*f);
        println!(
            "{name}: {:.2} us/op (CPU to commit {:.2})",
            per(n, t),
            per(n, c)
        );
    }
}

// ---- compile ------------------------------------------------------------

type Body = dyn Fn(&[Array]) -> Vec<Array>;

/// An MLX-C closure running `body`; the box is freed with the closure.
fn closure(body: Box<Body>) -> sys::mlx_closure {
    unsafe extern "C" fn call(
        res: *mut sys::mlx_vector_array,
        input: sys::mlx_vector_array,
        payload: *mut c_void,
    ) -> i32 {
        // SAFETY: `payload` is the `Box<Box<Body>>` below, alive while the
        // closure is; the input handles are retained by the copies.
        unsafe {
            let body = &*(payload as *const Box<Body>);
            let n = sys::mlx_vector_array_size(input);
            let args: Vec<Array> = (0..n)
                .map(|i| {
                    let mut a = sys::mlx_array_new();
                    sys::mlx_vector_array_get(&mut a, input, i);
                    Array::from_ptr(a)
                })
                .collect();
            let outs = body(&args);
            let ptrs: Vec<sys::mlx_array> = outs.iter().map(|a| a.as_ptr()).collect();
            sys::mlx_vector_array_set_data(res, ptrs.as_ptr(), ptrs.len())
        }
    }
    unsafe extern "C" fn free(payload: *mut c_void) {
        // SAFETY: created by `Box::into_raw` below, freed once.
        drop(unsafe { Box::from_raw(payload as *mut Box<Body>) });
    }
    let payload = Box::into_raw(Box::new(body)) as *mut c_void;
    // SAFETY: the payload outlives the closure through `free`.
    unsafe { sys::mlx_closure_new_func_payload(Some(call), payload, Some(free)) }
}

/// A compiled function (`mx.compile`) over arrays.
struct Compiled(sys::mlx_closure);

impl Compiled {
    fn new(body: Box<Body>, shapeless: bool) -> Self {
        let f = closure(body);
        // SAFETY: fresh handles; `f` is retained by the compiled closure.
        unsafe {
            let mut res = sys::mlx_closure_new();
            assert_eq!(sys::mlx_compile(&mut res, f, shapeless), 0, "mlx_compile");
            sys::mlx_closure_free(f);
            Self(res)
        }
    }

    fn call(&self, args: &[&Array]) -> Vec<Array> {
        let input = vector(args);
        // SAFETY: borrowed inputs; the result vector is read and freed once.
        unsafe {
            let mut out = sys::mlx_vector_array_new();
            assert_eq!(sys::mlx_closure_apply(&mut out, self.0, input), 0, "apply");
            sys::mlx_vector_array_free(input);
            let n = sys::mlx_vector_array_size(out);
            let v = (0..n)
                .map(|i| {
                    let mut a = sys::mlx_array_new();
                    sys::mlx_vector_array_get(&mut a, out, i);
                    Array::from_ptr(a)
                })
                .collect();
            sys::mlx_vector_array_free(out);
            v
        }
    }
}

impl Drop for Compiled {
    fn drop(&mut self) {
        // SAFETY: owned and freed once.
        unsafe { sys::mlx_closure_free(self.0) };
    }
}

/// `MessageGate::forward` with its weights as inputs: hidden, message (f32
/// (1, 1, D)), down (bf16 (64, 2D)), forget, gain (f32 (D, 64)).
fn message_gate(a: &[Array]) -> Vec<Array> {
    let (hidden, message, down, forget, gain) = (&a[0], &a[1], &a[2], &a[3], &a[4]);
    let n = |x: &Array| ops::rms_norm(x, None, 1e-6).unwrap();
    let states = ops::concatenate(&[&n(hidden), &n(message)], -1).unwrap();
    let states = ops::linear(&ops::astype(&states, Dtype::Bfloat16).unwrap(), down).unwrap();
    let states = ops::f32(&states).unwrap();
    let f = ops::tanh(&ops::linear(&states, forget).unwrap()).unwrap();
    let g = ops::tanh(&ops::linear(&states, gain).unwrap()).unwrap();
    let kept = ops::mul(&ops::sub(&f32s(1.0), &f).unwrap(), hidden).unwrap();
    vec![ops::add(&kept, &ops::mul(&g, message).unwrap()).unwrap()]
}

/// `softcapped_swiglu` on (1, 1, 8, 2 I) gate/up interleaved, bf16.
fn swiglu(a: &[Array]) -> Vec<Array> {
    let gu = &a[0];
    let n = gu.shape()[3];
    let gate = ops::f32(&ops::slice(gu, -1, 0, n, 2).unwrap()).unwrap();
    let up = ops::f32(&ops::slice(gu, -1, 1, n, 2).unwrap()).unwrap();
    let cap = |x: &Array, c: f32| {
        ops::mul(
            &ops::tanh(&ops::div(x, &f32s(c)).unwrap()).unwrap(),
            &f32s(c),
        )
        .unwrap()
    };
    let out = ops::mul(
        &ops::mul(&cap(&gate, 10.0), &ops::sigmoid(&gate).unwrap()).unwrap(),
        &cap(&up, 10.0),
    )
    .unwrap();
    vec![ops::astype(&out, Dtype::Bfloat16).unwrap()]
}

/// The pass tail around the MoE: post-MLP short convolution (state 3
/// rows), residual add, cast. x bf16 (1, 1, D), state f32 (1, 3, D), weight
/// bf16 (D, 4, 1), residual f32 (1, 1, D).
fn conv_residual(a: &[Array]) -> Vec<Array> {
    let (x, state, w, residual) = (&a[0], &a[1], &a[2], &a[3]);
    let xf = ops::f32(x).unwrap();
    let window = ops::concatenate(&[state, &xf], 1).unwrap();
    let out = ops::conv1d(&ops::astype(&window, Dtype::Bfloat16).unwrap(), w, 2304).unwrap();
    let out = ops::add(&ops::f32(&out).unwrap(), &xf).unwrap();
    let out = ops::div(&out, &f32s(5f32.sqrt())).unwrap();
    let out = ops::astype(&out, Dtype::Bfloat16).unwrap();
    let h = ops::add(residual, &ops::f32(&out).unwrap()).unwrap();
    let pos = ops::reshape(&ops::arange(1, 4).unwrap(), &[1, 3, 1]).unwrap();
    let new_state = ops::take_along_axis(&window, &pos, 1).unwrap();
    vec![ops::astype(&h, Dtype::Bfloat16).unwrap(), new_state]
}

/// Eager vs `compile` for chains of 48 small blocks (one per pass), each
/// fed by the previous one so nothing runs concurrently.
#[test]
#[ignore]
fn compile_small_blocks() {
    ops::init();
    let d = 2304;
    let passes = 48;
    let hidden = randn(&[1, 1, d], 1, Dtype::Float32);
    let message = randn(&[1, 1, d], 2, Dtype::Float32);
    let down = randn(&[64, 2 * d], 3, Dtype::Bfloat16);
    let forget = ops::mul(&randn(&[d, 64], 4, Dtype::Float32), &f32s(0.1)).unwrap();
    let gain = ops::mul(&randn(&[d, 64], 5, Dtype::Float32), &f32s(0.1)).unwrap();
    ops::eval(&[&forget, &gain]).unwrap();

    let report = |name: &str, eager: &dyn Fn() -> Vec<Array>, comp: &dyn Fn() -> Vec<Array>| {
        let e = eager();
        let c = comp();
        let (e, c) = (ops::f32(&e[0]).unwrap(), ops::f32(&c[0]).unwrap());
        ops::eval(&[&e, &c]).unwrap();
        let diff = e
            .as_slice::<f32>()
            .iter()
            .zip(c.as_slice::<f32>())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let kernels = |outs: Vec<Array>| {
            let refs: Vec<&Array> = outs.iter().collect();
            let c = graph_counts(&refs, &scratch().join("block.dot"));
            let views: usize = c
                .iter()
                .filter(|(k, _)| VIEWS.contains(&k.as_str()))
                .map(|(_, v)| v)
                .sum();
            let kinds: Vec<String> = c
                .iter()
                .filter(|(k, _)| !VIEWS.contains(&k.as_str()))
                .map(|(k, v)| format!("{k} {:.1}", *v as f64 / passes as f64))
                .collect();
            (
                (c.values().sum::<usize>() - views) as f64 / passes as f64,
                kinds.join(", "),
            )
        };
        let (ke, kinds_e) = kernels(eager());
        let (kc, kinds_c) = kernels(comp());
        println!(
            "{name}: kernels per block eager {ke:.1} [{kinds_e}], compiled {kc:.1} [{kinds_c}]"
        );
        let (te, ce) = time_graph(15, eager);
        let (tc, cc) = time_graph(15, comp);
        println!(
            "{name}: eager {:.1} us/block (CPU {:.1}), compiled {:.1} us/block (CPU {:.1}), max diff {}",
            te * 1e6 / passes as f64,
            ce * 1e6 / passes as f64,
            tc * 1e6 / passes as f64,
            cc * 1e6 / passes as f64,
            diff
        );
    };

    let gate = Compiled::new(Box::new(message_gate), false);
    report(
        "MessageGate x48",
        &|| {
            let mut h = hidden.clone();
            for _ in 0..passes {
                h = message_gate(&[
                    h,
                    message.clone(),
                    down.clone(),
                    forget.clone(),
                    gain.clone(),
                ])
                .remove(0);
            }
            vec![h]
        },
        &|| {
            let mut h = hidden.clone();
            for _ in 0..passes {
                h = gate.call(&[&h, &message, &down, &forget, &gain]).remove(0);
            }
            vec![h]
        },
    );

    let gu = randn(&[1, 1, 8, 2 * 768], 6, Dtype::Bfloat16);
    let sw = Compiled::new(Box::new(swiglu), false);
    // Chained through a cheap add so each block waits for the previous.
    let chain = |f: &dyn Fn(&Array) -> Array| {
        let mut x = gu.clone();
        for _ in 0..passes {
            let h = f(&x);
            x = ops::add(
                &gu,
                &ops::mul(
                    &ops::concatenate(&[&h, &h], -1).unwrap(),
                    &ops::scalar(1e-3, Dtype::Bfloat16).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        }
        vec![x]
    };
    report(
        "softcapped swiglu x48 (+3 chain ops)",
        &|| chain(&|x| swiglu(std::slice::from_ref(x)).remove(0)),
        &|| chain(&|x| sw.call(&[x]).remove(0)),
    );

    let state = randn(&[1, 3, d], 7, Dtype::Float32);
    let w = randn(&[d, 4, 1], 8, Dtype::Bfloat16);
    let xb = randn(&[1, 1, d], 9, Dtype::Bfloat16);
    let cr = Compiled::new(Box::new(conv_residual), false);
    report(
        "short conv + residual x48",
        &|| {
            let (mut h, mut s) = (xb.clone(), state.clone());
            for _ in 0..passes {
                let mut o = conv_residual(&[h, s, w.clone(), hidden.clone()]);
                s = o.pop().unwrap();
                h = o.pop().unwrap();
            }
            vec![h, s]
        },
        &|| {
            let (mut h, mut s) = (xb.clone(), state.clone());
            for _ in 0..passes {
                let mut o = cr.call(&[&h, &s, &w, &hidden]);
                s = o.pop().unwrap();
                h = o.pop().unwrap();
            }
            vec![h, s]
        },
    );
}

/// Captures one decode step (1k context, bf16 cache) as a Metal GPU trace
/// for offline profiling (e.g. `gputrace report`). Needs
/// `MTL_CAPTURE_ENABLED=1`; writes `PROFILE_DIR/decode_step.gputrace`.
#[test]
#[ignore]
fn capture_decode_step() {
    let model = load();
    let mut cache = prefilled(&model, KvFormat::BF16, 1024);
    for t in 0..4 {
        timed_step(&model, &mut cache, 3000 + t);
    }
    let path = scratch().join("decode_step.gputrace");
    let _ = std::fs::remove_dir_all(&path);
    let c = CString::new(path.to_str().unwrap()).unwrap();
    // SAFETY: plain MLX-C calls; the path outlives them.
    assert_eq!(
        unsafe { sys::mlx_metal_start_capture(c.as_ptr()) },
        0,
        "start capture"
    );
    let logits = model
        .forward(&Array::from_slice(&[3100], &[1, 1]), &mut cache)
        .unwrap();
    let mut live = vec![&logits];
    live.extend(cache.iter().flat_map(|c| c.arrays()));
    ops::eval(&live).unwrap();
    assert_eq!(unsafe { sys::mlx_metal_stop_capture() }, 0, "stop capture");
    println!("captured {}", path.display());
}
