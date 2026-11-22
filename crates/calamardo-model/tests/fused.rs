//! K-6 fusions against the op sequences they replace, bit for bit.

use calamardo_model::{kernels::Kernels, ops};
use mlx_rs::{Array, Dtype};

fn data(n: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * amp
        })
        .collect()
}

fn bf16(v: Vec<f32>, shape: &[i32]) -> Array {
    ops::astype(&Array::from_slice(&v, shape), Dtype::Bfloat16).unwrap()
}

fn bits(a: &Array) -> Vec<u32> {
    let a = ops::f32(a).unwrap();
    ops::eval(&[&a]).unwrap();
    a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect()
}

/// The decode mask as the reference builds it, op by op.
fn mask_ops(pb: &Array, s: i32, rot: Option<i32>, end: i32, window: i32, dtype: Dtype) -> Array {
    let r = pb.shape()[1];
    let query_positions = ops::add(&ops::arange(0, 1).unwrap(), &ops::i32s(end - 1)).unwrap();
    let mut key_positions = ops::arange(0, s).unwrap();
    if let Some(idx) = rot {
        key_positions = ops::remainder(
            &ops::sub(&key_positions, &ops::i32s(idx)).unwrap(),
            &ops::i32s(s),
        )
        .unwrap();
    }
    let key_positions = ops::add(&key_positions, &ops::i32s(end - s)).unwrap();
    let distance = ops::sub(
        &ops::reshape(&query_positions, &[1]).unwrap(),
        &ops::reshape(&key_positions, &[s]).unwrap(),
    )
    .unwrap();
    let bias = ops::take(
        pb,
        &ops::clip(&distance, &ops::i32s(0), &ops::i32s(r - 1)).unwrap(),
        -1,
    )
    .unwrap();
    let zeros = ops::zeros(bias.shape(), bias.dtype()).unwrap();
    let bias = ops::where_(&ops::less(&distance, &ops::i32s(r)).unwrap(), &bias, &zeros).unwrap();
    let mut visible = ops::logical_and(
        &ops::greater_equal(&distance, &ops::i32s(0)).unwrap(),
        &ops::greater_equal(&ops::reshape(&key_positions, &[s]).unwrap(), &ops::i32s(0)).unwrap(),
    )
    .unwrap();
    if window > 0 {
        visible =
            ops::logical_and(&visible, &ops::less(&distance, &ops::i32s(window)).unwrap()).unwrap();
    }
    let neg_inf = ops::scalar(f32::NEG_INFINITY, bias.dtype()).unwrap();
    let mask = ops::astype(&ops::where_(&visible, &bias, &neg_inf).unwrap(), dtype).unwrap();
    ops::expand_dims(&ops::expand_dims(&mask, &[0]).unwrap(), &[2]).unwrap()
}

#[test]
fn decode_mask_matches_ops() {
    ops::init();
    let kern = Kernels::new().unwrap();
    let hq = 24;
    // (slots, rotation (None: global cache), tokens with the query, window, R)
    let cases = [
        (1024, Some(0), 5000, 1024, 1024),
        (1024, Some(317), 5000, 1024, 1024),
        (1024, Some(1024), 1500, 1024, 1024),
        (700, Some(700), 700, 1024, 1024),
        (1030, Some(1030), 1030, 1024, 1024),
        (3000, None, 3000, 0, 1024),
        (9, None, 9, 0, 16),
        (40, Some(13), 77, 8, 16),
        // A buffer with more slots than tokens: keys at negative positions.
        (40, Some(13), 20, 1024, 16),
    ];
    for (s, rot, end, window, r) in cases {
        let pb = bf16(data(hq * r, 5 + s as u32, 4.0), &[hq as i32, r as i32]);
        let got = kern
            .decode_mask(&pb, s, rot.unwrap_or(0), end, window, Dtype::Bfloat16)
            .unwrap();
        let want = mask_ops(&pb, s, rot, end, window, Dtype::Bfloat16);
        assert_eq!(got.shape(), want.shape());
        assert_eq!(
            bits(&got),
            bits(&want),
            "s {s} rot {rot:?} end {end} window {window} R {r}"
        );
    }
}

#[test]
fn short_conv_step_matches_ops() {
    ops::init();
    let kern = Kernels::new().unwrap();
    for (c, k) in [(2304, 4), (1024, 4), (40, 3)] {
        let x = bf16(data(c, 11 + c as u32, 6.0), &[1, 1, c as i32]);
        let state = Array::from_slice(&data((k - 1) * c, 12, 6.0), &[1, (k - 1) as i32, c as i32]);
        let w = bf16(data(c * k, 13, 1.5), &[c as i32, k as i32, 1]);
        let denom = ((k + 1) as f64).sqrt() as f32;
        let (out, win) = kern
            .short_conv_step(
                &x,
                &state,
                &ops::reshape(&w, &[c as i32, k as i32]).unwrap(),
                denom,
            )
            .unwrap();
        // ShortConvolution::forward, op by op.
        let residual = ops::f32(&x).unwrap();
        let window = ops::concatenate(&[&state, &ops::f32(&x).unwrap()], 1).unwrap();
        let conv = ops::conv1d(
            &ops::astype(&window, Dtype::Bfloat16).unwrap(),
            &w,
            c as i32,
        )
        .unwrap();
        let want = ops::add(&ops::f32(&conv).unwrap(), &residual).unwrap();
        let want = ops::div(&want, &ops::f32s(denom)).unwrap();
        let want = ops::astype(&want, Dtype::Bfloat16).unwrap();
        assert_eq!(bits(&out), bits(&want), "C {c} K {k}: output");
        assert_eq!(bits(&win), bits(&window), "C {c} K {k}: window");
    }
}

/// A compiled function's own error reaches the caller.
#[test]
fn compiled_error_keeps_its_message() {
    ops::init();
    let f = ops::Compiled::new(|_: &[Array]| anyhow::bail!("the body says no"));
    let err = f.call1(&[&Array::from_slice(&[1.0f32], &[1])]).unwrap_err();
    assert!(format!("{err:#}").contains("the body says no"), "{err:#}");
}

/// The convolution state is a buffer of its own: dropping the window frees
/// the window.
#[test]
fn conv_state_does_not_keep_the_window() {
    ops::init();
    let window = Array::from_slice(&data(2051 * 2304, 21, 1.0), &[1, 2051, 2304]);
    ops::eval(&[&window]).unwrap();
    let state = calamardo_model::cache::conv_state(&window, 3).unwrap();
    ops::eval(&[&state]).unwrap();
    // MLX releases what a command buffer held from its completion handler,
    // which may run just after the wait returns.
    std::thread::sleep(std::time::Duration::from_millis(100));
    let with = mlx_rs::memory::active_memory().unwrap();
    drop(window);
    let without = mlx_rs::memory::active_memory().unwrap();
    assert!(
        with - without >= 2051 * 2304 * 4,
        "the window stayed alive ({with} -> {without})"
    );
    assert_eq!(state.shape(), &[1, 3, 2304]);
}
