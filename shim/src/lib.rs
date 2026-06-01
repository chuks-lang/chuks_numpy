// chuks_numpy_shim — C-ABI bridge around the `ndarray` crate.
//
// Scaffold (v0.1) — exposes a version string and a minimal NDArray
// (float64, row-major, owned) sufficient for the NU1 smoke test:
//   - np_version()                       → *mut c_char
//   - np_free_string(s)                  → ()
//   - np_array_zeros_f64(rank, dims_ptr) → *mut NDArrayF64
//   - np_array_len(arr)                  → i64
//   - np_array_rank(arr)                 → i64
//   - np_array_dim(arr, axis)            → i64
//   - np_array_get_f64(arr, idxs_ptr)    → f64   (panic-free; out-of-range → 0.0)
//   - np_array_set_f64(arr, idxs_ptr, v) → i32   (0 ok, -1 out of range)
//   - np_free_array_f64(arr)             → ()
//
// All `*mut X` handles are Box::into_raw'd; release with the matching
// `np_free_*` function. Phases NU1–NU5 will replace this scaffold with
// the full surface (broadcasting kernels, slicing, BLAS linalg, FFT,
// PPP_I ternary ops).

use ndarray::{ArrayD, IxDyn, SliceInfoElem};
use std::ffi::{c_char, c_void, CString};
use std::slice;

#[repr(C)]
pub struct NDArrayF64 {
    inner: ArrayD<f64>,
}

#[inline]
fn box_arr(a: ArrayD<f64>) -> *mut NDArrayF64 {
    Box::into_raw(Box::new(NDArrayF64 { inner: a }))
}

#[inline]
unsafe fn arr_ref<'a>(p: *const NDArrayF64) -> Option<&'a NDArrayF64> {
    if p.is_null() { None } else { Some(&*p) }
}

#[inline]
unsafe fn arr_mut<'a>(p: *mut NDArrayF64) -> Option<&'a mut NDArrayF64> {
    if p.is_null() { None } else { Some(&mut *p) }
}

#[unsafe(no_mangle)]
pub extern "C" fn np_version() -> *mut c_char {
    CString::new("chuks_numpy_shim 0.6.0 (ndarray 0.16 + nalgebra 0.33 + rustfft 6 + rand 0.8 + zip 0.6)")
        .unwrap()
        .into_raw()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

// ── NDArray (f64) ───────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_zeros_f64(rank: i64, dims_ptr: *const i64) -> *mut NDArrayF64 {
    if rank < 0 || dims_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let dims: Vec<usize> = slice::from_raw_parts(dims_ptr, rank as usize)
        .iter()
        .map(|&d| if d < 0 { 0 } else { d as usize })
        .collect();
    box_arr(ArrayD::<f64>::zeros(IxDyn(&dims)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_len(p: *const NDArrayF64) -> i64 {
    match arr_ref(p) {
        Some(a) => a.inner.len() as i64,
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_rank(p: *const NDArrayF64) -> i64 {
    match arr_ref(p) {
        Some(a) => a.inner.ndim() as i64,
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_dim(p: *const NDArrayF64, axis: i64) -> i64 {
    match arr_ref(p) {
        Some(a) => {
            if axis < 0 || (axis as usize) >= a.inner.ndim() {
                return -1;
            }
            a.inner.shape()[axis as usize] as i64
        }
        None => -1,
    }
}

unsafe fn ixdyn_from(p: *const NDArrayF64, idxs_ptr: *const i64) -> Option<IxDyn> {
    let a = arr_ref(p)?;
    if idxs_ptr.is_null() {
        return None;
    }
    let rank = a.inner.ndim();
    let raw = slice::from_raw_parts(idxs_ptr, rank);
    let mut ix = Vec::with_capacity(rank);
    let shape = a.inner.shape();
    for i in 0..rank {
        let v = raw[i];
        if v < 0 || (v as usize) >= shape[i] {
            return None;
        }
        ix.push(v as usize);
    }
    Some(IxDyn(&ix))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_get_f64(p: *const NDArrayF64, idxs_ptr: *const i64) -> f64 {
    let a = match arr_ref(p) { Some(a) => a, None => return 0.0 };
    let ix = match ixdyn_from(p, idxs_ptr) { Some(i) => i, None => return 0.0 };
    a.inner[ix]
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_set_f64(
    p: *mut NDArrayF64,
    idxs_ptr: *const i64,
    v: f64,
) -> i32 {
    let ix = match ixdyn_from(p as *const NDArrayF64, idxs_ptr) {
        Some(i) => i,
        None => return -1,
    };
    let a = match arr_mut(p) { Some(a) => a, None => return -1 };
    a.inner[ix] = v;
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_free_array_f64(p: *mut NDArrayF64) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

// ── NU1: Construction ───────────────────────────────────────────────

unsafe fn read_dims(rank: i64, dims_ptr: *const i64) -> Option<Vec<usize>> {
    if rank < 0 || dims_ptr.is_null() {
        return None;
    }
    Some(
        slice::from_raw_parts(dims_ptr, rank as usize)
            .iter()
            .map(|&d| if d < 0 { 0 } else { d as usize })
            .collect(),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_ones_f64(rank: i64, dims_ptr: *const i64) -> *mut NDArrayF64 {
    let dims = match read_dims(rank, dims_ptr) { Some(d) => d, None => return std::ptr::null_mut() };
    box_arr(ArrayD::<f64>::ones(IxDyn(&dims)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_full_f64(
    rank: i64,
    dims_ptr: *const i64,
    value: f64,
) -> *mut NDArrayF64 {
    let dims = match read_dims(rank, dims_ptr) { Some(d) => d, None => return std::ptr::null_mut() };
    box_arr(ArrayD::<f64>::from_elem(IxDyn(&dims), value))
}

#[unsafe(no_mangle)]
pub extern "C" fn np_array_arange_f64(start: f64, stop: f64, step: f64) -> *mut NDArrayF64 {
    if step == 0.0 || !step.is_finite() || !start.is_finite() || !stop.is_finite() {
        return std::ptr::null_mut();
    }
    let span = stop - start;
    let n_raw = (span / step).ceil();
    let n = if n_raw <= 0.0 { 0 } else { n_raw as usize };
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(start + (i as f64) * step);
    }
    box_arr(ArrayD::from_shape_vec(IxDyn(&[n]), v).expect("arange shape"))
}

#[unsafe(no_mangle)]
pub extern "C" fn np_array_linspace_f64(start: f64, stop: f64, n: i64) -> *mut NDArrayF64 {
    if n < 0 {
        return std::ptr::null_mut();
    }
    let n = n as usize;
    let mut v = Vec::with_capacity(n);
    if n == 0 {
        // empty
    } else if n == 1 {
        v.push(start);
    } else {
        let step = (stop - start) / ((n - 1) as f64);
        for i in 0..n {
            v.push(start + (i as f64) * step);
        }
    }
    box_arr(ArrayD::from_shape_vec(IxDyn(&[n]), v).expect("linspace shape"))
}

#[unsafe(no_mangle)]
pub extern "C" fn np_array_eye_f64(n: i64) -> *mut NDArrayF64 {
    if n < 0 {
        return std::ptr::null_mut();
    }
    let n = n as usize;
    let mut a = ArrayD::<f64>::zeros(IxDyn(&[n, n]));
    for i in 0..n {
        a[IxDyn(&[i, i])] = 1.0;
    }
    box_arr(a)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_from_data_f64(
    rank: i64,
    dims_ptr: *const i64,
    data_ptr: *const f64,
) -> *mut NDArrayF64 {
    let dims = match read_dims(rank, dims_ptr) { Some(d) => d, None => return std::ptr::null_mut() };
    if data_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let total: usize = dims.iter().product();
    let buf = slice::from_raw_parts(data_ptr, total).to_vec();
    match ArrayD::from_shape_vec(IxDyn(&dims), buf) {
        Ok(a) => box_arr(a),
        Err(_) => std::ptr::null_mut(),
    }
}

// Copies the array's elements (row-major / standard layout) into the
// caller-allocated buffer at `out_ptr`. Returns the number of elements
// written, or -1 on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_to_data_f64(
    p: *const NDArrayF64,
    out_ptr: *mut f64,
) -> i64 {
    let a = match arr_ref(p) { Some(a) => a, None => return -1 };
    if out_ptr.is_null() {
        return -1;
    }
    // Materialize to a row-major owned array so we can copy as a flat slice.
    // If the array is already standard-layout, as_slice() is zero-cost.
    let owned;
    let flat: &[f64] = if let Some(s) = a.inner.as_slice() {
        s
    } else {
        owned = a.inner.iter().copied().collect::<Vec<f64>>();
        &owned[..]
    };
    std::ptr::copy_nonoverlapping(flat.as_ptr(), out_ptr, flat.len());
    flat.len() as i64
}

// ── NU1: Reshape / transpose / slice / take / copy ──────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_reshape_f64(
    p: *const NDArrayF64,
    rank: i64,
    dims_ptr: *const i64,
) -> *mut NDArrayF64 {
    let a = match arr_ref(p) { Some(a) => a, None => return std::ptr::null_mut() };
    let dims = match read_dims(rank, dims_ptr) { Some(d) => d, None => return std::ptr::null_mut() };
    let total: usize = dims.iter().product();
    if total != a.inner.len() {
        return std::ptr::null_mut();
    }
    // Materialize a standard-layout flat copy, then reshape.
    let flat: Vec<f64> = a.inner.iter().copied().collect();
    match ArrayD::from_shape_vec(IxDyn(&dims), flat) {
        Ok(b) => box_arr(b),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_transpose_f64(p: *const NDArrayF64) -> *mut NDArrayF64 {
    let a = match arr_ref(p) { Some(a) => a, None => return std::ptr::null_mut() };
    // Reverse axes view, then materialize to owned standard layout so the
    // resulting array is contiguous (callers don't see strides).
    let view = a.inner.t();
    let shape = view.shape().to_vec();
    let data: Vec<f64> = view.iter().copied().collect();
    match ArrayD::from_shape_vec(IxDyn(&shape), data) {
        Ok(b) => box_arr(b),
        Err(_) => std::ptr::null_mut(),
    }
}

// Strided slice. `starts`, `ends`, `steps` each have length == rank.
// Negative start/end means "from the end" (Python style). step > 0 only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_slice_f64(
    p: *const NDArrayF64,
    rank: i64,
    starts: *const i64,
    ends: *const i64,
    steps: *const i64,
) -> *mut NDArrayF64 {
    let a = match arr_ref(p) { Some(a) => a, None => return std::ptr::null_mut() };
    if rank < 0 || a.inner.ndim() != rank as usize {
        return std::ptr::null_mut();
    }
    if starts.is_null() || ends.is_null() || steps.is_null() {
        return std::ptr::null_mut();
    }
    let r = rank as usize;
    let s_starts = slice::from_raw_parts(starts, r);
    let s_ends = slice::from_raw_parts(ends, r);
    let s_steps = slice::from_raw_parts(steps, r);
    let shape = a.inner.shape();
    let mut elems: Vec<SliceInfoElem> = Vec::with_capacity(r);
    for i in 0..r {
        let dim = shape[i] as i64;
        let mut st = s_starts[i];
        let mut en = s_ends[i];
        if st < 0 { st += dim; }
        if en < 0 { en += dim; }
        if st < 0 { st = 0; }
        if en > dim { en = dim; }
        if en < st { en = st; }
        let step = s_steps[i];
        if step <= 0 {
            return std::ptr::null_mut();
        }
        elems.push(SliceInfoElem::Slice { start: st as isize, end: Some(en as isize), step: step as isize });
    }
    let view = a.inner.slice_each_axis(|ax| {
        if let SliceInfoElem::Slice { start, end, step } = elems[ax.axis.index()] {
            ndarray::Slice { start, end, step }
        } else {
            unreachable!()
        }
    });
    let shape_out = view.shape().to_vec();
    let data: Vec<f64> = view.iter().copied().collect();
    match ArrayD::from_shape_vec(IxDyn(&shape_out), data) {
        Ok(b) => box_arr(b),
        Err(_) => std::ptr::null_mut(),
    }
}

// Fancy index along one axis: pick the rows/cols/etc. listed in `idxs`.
// Out-of-range index → null returned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_take_f64(
    p: *const NDArrayF64,
    axis: i64,
    n_idx: i64,
    idxs_ptr: *const i64,
) -> *mut NDArrayF64 {
    let a = match arr_ref(p) { Some(a) => a, None => return std::ptr::null_mut() };
    if axis < 0 || (axis as usize) >= a.inner.ndim() {
        return std::ptr::null_mut();
    }
    if n_idx < 0 || (n_idx > 0 && idxs_ptr.is_null()) {
        return std::ptr::null_mut();
    }
    let ax = ndarray::Axis(axis as usize);
    let dim_len = a.inner.shape()[axis as usize] as i64;
    let raw = if n_idx == 0 { &[][..] } else { slice::from_raw_parts(idxs_ptr, n_idx as usize) };
    let mut picks: Vec<usize> = Vec::with_capacity(raw.len());
    for &i in raw {
        if i < 0 || i >= dim_len {
            return std::ptr::null_mut();
        }
        picks.push(i as usize);
    }
    let selected = a.inner.select(ax, &picks);
    box_arr(selected)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_copy_f64(p: *const NDArrayF64) -> *mut NDArrayF64 {
    let a = match arr_ref(p) { Some(a) => a, None => return std::ptr::null_mut() };
    box_arr(a.inner.clone())
}

// ── NU2: Elementwise + broadcasting + reductions ────────────────────
//
// Op-code dispatch keeps the C ABI compact. Chuks side maps ergonomic
// method names (`add`, `sub`, `sqrt`, …) onto these op ids.
//
//   Binary  : 0=add 1=sub 2=mul 3=div 4=pow 5=mod 6=min 7=max
//   Unary   : 0=sqrt 1=exp 2=log 3=sin 4=cos 5=tan 6=abs
//             7=floor 8=ceil 9=round 10=neg
//   Compare : 0=eq 1=ne 2=lt 3=le 4=gt 5=ge   (mask: 1.0=true, 0.0=false)
//   Reduce  : 0=sum 1=mean 2=min 3=max 4=prod
//             5=std 6=var 7=any 8=all 9=argmin 10=argmax

#[inline]
fn binop_apply(op: i32, a: f64, b: f64) -> f64 {
    match op {
        0 => a + b,
        1 => a - b,
        2 => a * b,
        3 => a / b,
        4 => a.powf(b),
        // NumPy `mod`: result takes the sign of the divisor.
        5 => {
            if b == 0.0 { f64::NAN } else { a - (a / b).floor() * b }
        }
        6 => a.min(b),
        7 => a.max(b),
        _ => f64::NAN,
    }
}

#[inline]
fn unary_apply(op: i32, x: f64) -> f64 {
    match op {
        0 => x.sqrt(),
        1 => x.exp(),
        2 => x.ln(),
        3 => x.sin(),
        4 => x.cos(),
        5 => x.tan(),
        6 => x.abs(),
        7 => x.floor(),
        8 => x.ceil(),
        9 => x.round(),
        10 => -x,
        _ => f64::NAN,
    }
}

#[inline]
fn cmp_apply(op: i32, a: f64, b: f64) -> f64 {
    let r = match op {
        0 => a == b,
        1 => a != b,
        2 => a < b,
        3 => a <= b,
        4 => a > b,
        5 => a >= b,
        _ => false,
    };
    if r { 1.0 } else { 0.0 }
}

// NumPy-style broadcast: align right, dims must match or one of them is 1.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let r = a.len().max(b.len());
    let mut out = Vec::with_capacity(r);
    for i in 0..r {
        let da = if i + a.len() < r { 1 } else { a[i + a.len() - r] };
        let db = if i + b.len() < r { 1 } else { b[i + b.len() - r] };
        if da == db {
            out.push(da);
        } else if da == 1 {
            out.push(db);
        } else if db == 1 {
            out.push(da);
        } else {
            return None;
        }
    }
    Some(out)
}

// Returns a broadcast view of `arr` at `target` shape, or None if the
// shapes are not broadcast-compatible. `ndarray::broadcast` handles
// inserting leading length-1 axes and stretching length-1 axes for us.
fn broadcast_view<'a>(
    arr: &'a ArrayD<f64>,
    target: &[usize],
) -> Option<ndarray::ArrayViewD<'a, f64>> {
    arr.broadcast(IxDyn(target))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_binop_f64(
    pa: *const NDArrayF64,
    pb: *const NDArrayF64,
    op: i32,
) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let b = match arr_ref(pb) { Some(a) => a, None => return std::ptr::null_mut() };
    let out_shape = match broadcast_shape(a.inner.shape(), b.inner.shape()) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let av = match broadcast_view(&a.inner, &out_shape) {
        Some(v) => v, None => return std::ptr::null_mut(),
    };
    let bv = match broadcast_view(&b.inner, &out_shape) {
        Some(v) => v, None => return std::ptr::null_mut(),
    };
    let out = ndarray::Zip::from(&av).and(&bv).map_collect(|x, y| binop_apply(op, *x, *y));
    box_arr(out)
}

// Binary op with a scalar broadcast against `pa`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_binop_scalar_f64(
    pa: *const NDArrayF64,
    scalar: f64,
    op: i32,
    // 0: arr op scalar   1: scalar op arr   (matters for sub/div/pow/mod)
    rev: i32,
) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let out = a.inner.mapv(|x| {
        if rev == 0 { binop_apply(op, x, scalar) } else { binop_apply(op, scalar, x) }
    });
    box_arr(out)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_unary_f64(
    pa: *const NDArrayF64,
    op: i32,
) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let out = a.inner.mapv(|x| unary_apply(op, x));
    box_arr(out)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_cmp_f64(
    pa: *const NDArrayF64,
    pb: *const NDArrayF64,
    op: i32,
) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let b = match arr_ref(pb) { Some(a) => a, None => return std::ptr::null_mut() };
    let out_shape = match broadcast_shape(a.inner.shape(), b.inner.shape()) {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let av = match broadcast_view(&a.inner, &out_shape) {
        Some(v) => v, None => return std::ptr::null_mut(),
    };
    let bv = match broadcast_view(&b.inner, &out_shape) {
        Some(v) => v, None => return std::ptr::null_mut(),
    };
    let out = ndarray::Zip::from(&av).and(&bv).map_collect(|x, y| cmp_apply(op, *x, *y));
    box_arr(out)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_cmp_scalar_f64(
    pa: *const NDArrayF64,
    scalar: f64,
    op: i32,
    rev: i32,
) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let out = a.inner.mapv(|x| {
        if rev == 0 { cmp_apply(op, x, scalar) } else { cmp_apply(op, scalar, x) }
    });
    box_arr(out)
}

// Reduce over a 1-D lane (used by both reduce_all and reduce_axis).
fn reduce_lane_iter<I: IntoIterator<Item = f64>>(op: i32, iter: I, n: usize) -> f64 {
    if n == 0 {
        return match op {
            0 => 0.0,
            4 => 1.0,
            7 => 0.0,
            8 => 1.0,
            9 | 10 => -1.0,
            _ => f64::NAN,
        };
    }
    match op {
        0 => iter.into_iter().sum(),
        1 => {
            let s: f64 = iter.into_iter().sum();
            s / (n as f64)
        }
        2 => iter.into_iter().fold(f64::INFINITY, f64::min),
        3 => iter.into_iter().fold(f64::NEG_INFINITY, f64::max),
        4 => iter.into_iter().product(),
        5 | 6 => {
            let xs: Vec<f64> = iter.into_iter().collect();
            let mean = xs.iter().sum::<f64>() / (n as f64);
            let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n as f64);
            if op == 6 { var } else { var.sqrt() }
        }
        7 => {
            for v in iter { if v != 0.0 && !v.is_nan() { return 1.0; } }
            0.0
        }
        8 => {
            for v in iter { if v == 0.0 || v.is_nan() { return 0.0; } }
            1.0
        }
        9 => {
            let mut best_i = 0usize;
            let mut best_v = f64::INFINITY;
            for (i, v) in iter.into_iter().enumerate() {
                if v < best_v { best_v = v; best_i = i; }
            }
            best_i as f64
        }
        10 => {
            let mut best_i = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for (i, v) in iter.into_iter().enumerate() {
                if v > best_v { best_v = v; best_i = i; }
            }
            best_i as f64
        }
        _ => f64::NAN,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_reduce_all_f64(pa: *const NDArrayF64, op: i32) -> f64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return f64::NAN };
    let n = a.inner.len();
    reduce_lane_iter(op, a.inner.iter().copied(), n)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_reduce_axis_f64(
    pa: *const NDArrayF64,
    axis: i64,
    keepdims: i32,
    op: i32,
) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    if axis < 0 || (axis as usize) >= a.inner.ndim() {
        return std::ptr::null_mut();
    }
    let ax = ndarray::Axis(axis as usize);
    let reduced = a.inner.map_axis(ax, |lane| {
        let n = lane.len();
        reduce_lane_iter(op, lane.iter().copied(), n)
    });
    let out: ArrayD<f64> = if keepdims != 0 {
        let mut shape = reduced.shape().to_vec();
        shape.insert(axis as usize, 1);
        let flat: Vec<f64> = reduced.iter().copied().collect();
        match ArrayD::from_shape_vec(IxDyn(&shape), flat) {
            Ok(a) => a,
            Err(_) => return std::ptr::null_mut(),
        }
    } else {
        reduced
    };
    box_arr(out)
}

// In-place binary op: dst[i] = binop(dst[i], src[i]) with broadcasting.
// `src` must broadcast to `dst`'s shape (one-way; dst shape is the target).
// Returns 0 on success, -1 on shape mismatch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_binop_inplace_f64(
    p_dst: *mut NDArrayF64,
    p_src: *const NDArrayF64,
    op: i32,
) -> i32 {
    let src = match arr_ref(p_src) { Some(a) => a, None => return -1 };
    let dst = match arr_mut(p_dst) { Some(a) => a, None => return -1 };
    let target = dst.inner.shape().to_vec();
    let sv = match src.inner.broadcast(IxDyn(&target)) {
        Some(v) => v,
        None => return -1,
    };
    // Snapshot src values (broadcast view borrows from src; we need to
    // own the values so we can mutate dst freely afterwards).
    let svals: Vec<f64> = sv.iter().copied().collect();
    let mut idx = 0usize;
    for d in dst.inner.iter_mut() {
        *d = binop_apply(op, *d, svals[idx]);
        idx += 1;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_binop_inplace_scalar_f64(
    p_dst: *mut NDArrayF64,
    scalar: f64,
    op: i32,
) -> i32 {
    let dst = match arr_mut(p_dst) { Some(a) => a, None => return -1 };
    for d in dst.inner.iter_mut() {
        *d = binop_apply(op, *d, scalar);
    }
    0
}

// Touch c_void so the `use` is not warned-unused once more handles land.
#[doc(hidden)]
pub fn _link_c_void(_: *mut c_void) {}

// ── NU3: BLAS-backed linear algebra + FFT ───────────────────────────
//
// Backend: pure-Rust `nalgebra` 0.33 for linalg, `rustfft` 6 for FFT.
// All inputs/outputs round-trip through `ArrayD<f64>` (row-major). For
// multi-output decompositions (SVD, QR, Eig) we expose an opaque handle
// + `_take_*` accessors so callers pay the decomposition cost exactly
// once.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::sync::Arc;

// ── ndarray ↔ nalgebra bridges ──────────────────────────────────────

fn nd_to_dmatrix(a: &ArrayD<f64>) -> Option<DMatrix<f64>> {
    if a.ndim() != 2 {
        return None;
    }
    let r = a.shape()[0];
    let c = a.shape()[1];
    let flat: Vec<f64> = a.iter().copied().collect(); // row-major
    Some(DMatrix::from_row_slice(r, c, &flat))
}

fn nd_to_dvector(a: &ArrayD<f64>) -> Option<DVector<f64>> {
    if a.ndim() != 1 {
        return None;
    }
    let n = a.shape()[0];
    let flat: Vec<f64> = a.iter().copied().collect();
    Some(DVector::from_row_slice(&flat[..n]))
}

fn dmatrix_to_nd(m: &DMatrix<f64>) -> ArrayD<f64> {
    let r = m.nrows();
    let c = m.ncols();
    let mut flat = Vec::with_capacity(r * c);
    for i in 0..r {
        for j in 0..c {
            flat.push(m[(i, j)]);
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[r, c]), flat).expect("dmatrix_to_nd shape")
}

fn dvector_to_nd(v: &DVector<f64>) -> ArrayD<f64> {
    let n = v.nrows();
    let flat: Vec<f64> = v.iter().copied().collect();
    ArrayD::from_shape_vec(IxDyn(&[n]), flat).expect("dvector_to_nd shape")
}

// ── matmul / dot ────────────────────────────────────────────────────
//
// `matmul` handles the four NumPy cases:
//   2D × 2D → 2D
//   2D × 1D → 1D
//   1D × 2D → 1D
//   1D × 1D → 0D scalar (returned as shape=[] NDArray with single elem)

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_matmul_f64(
    pa: *const NDArrayF64,
    pb: *const NDArrayF64,
) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let b = match arr_ref(pb) { Some(a) => a, None => return std::ptr::null_mut() };
    match (a.inner.ndim(), b.inner.ndim()) {
        (2, 2) => {
            let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
            let n = match nd_to_dmatrix(&b.inner) { Some(m) => m, None => return std::ptr::null_mut() };
            if m.ncols() != n.nrows() {
                return std::ptr::null_mut();
            }
            box_arr(dmatrix_to_nd(&(m * n)))
        }
        (2, 1) => {
            let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
            let v = match nd_to_dvector(&b.inner) { Some(v) => v, None => return std::ptr::null_mut() };
            if m.ncols() != v.nrows() {
                return std::ptr::null_mut();
            }
            box_arr(dvector_to_nd(&(m * v)))
        }
        (1, 2) => {
            let v = match nd_to_dvector(&a.inner) { Some(v) => v, None => return std::ptr::null_mut() };
            let m = match nd_to_dmatrix(&b.inner) { Some(m) => m, None => return std::ptr::null_mut() };
            if v.nrows() != m.nrows() {
                return std::ptr::null_mut();
            }
            // (v.T) * m → 1×k row, then flatten to length-k vector.
            let rv = v.transpose() * m;
            let flat: Vec<f64> = rv.iter().copied().collect();
            let n = flat.len();
            box_arr(ArrayD::from_shape_vec(IxDyn(&[n]), flat).expect("1×2D matmul"))
        }
        (1, 1) => {
            if a.inner.len() != b.inner.len() {
                return std::ptr::null_mut();
            }
            let s: f64 = a.inner.iter().zip(b.inner.iter()).map(|(x, y)| x * y).sum();
            box_arr(ArrayD::from_shape_vec(IxDyn(&[]), vec![s]).expect("scalar matmul"))
        }
        _ => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_dot_f64(pa: *const NDArrayF64, pb: *const NDArrayF64) -> f64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return f64::NAN };
    let b = match arr_ref(pb) { Some(a) => a, None => return f64::NAN };
    if a.inner.ndim() != 1 || b.inner.ndim() != 1 || a.inner.len() != b.inner.len() {
        return f64::NAN;
    }
    a.inner.iter().zip(b.inner.iter()).map(|(x, y)| x * y).sum()
}

// ── inv / solve / det / trace ───────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_inv_f64(pa: *const NDArrayF64) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
    if m.nrows() != m.ncols() {
        return std::ptr::null_mut();
    }
    match m.try_inverse() {
        Some(inv) => box_arr(dmatrix_to_nd(&inv)),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_solve_f64(
    pa: *const NDArrayF64,
    pb: *const NDArrayF64,
) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let b = match arr_ref(pb) { Some(a) => a, None => return std::ptr::null_mut() };
    let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
    if m.nrows() != m.ncols() {
        return std::ptr::null_mut();
    }
    // Accept b as 1D (returns 1D) or 2D (returns 2D).
    match b.inner.ndim() {
        1 => {
            let v = match nd_to_dvector(&b.inner) { Some(v) => v, None => return std::ptr::null_mut() };
            if v.nrows() != m.nrows() {
                return std::ptr::null_mut();
            }
            let lu = m.lu();
            match lu.solve(&v) {
                Some(x) => box_arr(dvector_to_nd(&x)),
                None => std::ptr::null_mut(),
            }
        }
        2 => {
            let bm = match nd_to_dmatrix(&b.inner) { Some(m) => m, None => return std::ptr::null_mut() };
            if bm.nrows() != m.nrows() {
                return std::ptr::null_mut();
            }
            let lu = m.lu();
            match lu.solve(&bm) {
                Some(x) => box_arr(dmatrix_to_nd(&x)),
                None => std::ptr::null_mut(),
            }
        }
        _ => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_det_f64(pa: *const NDArrayF64) -> f64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return f64::NAN };
    let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return f64::NAN };
    if m.nrows() != m.ncols() {
        return f64::NAN;
    }
    m.determinant()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_trace_f64(pa: *const NDArrayF64) -> f64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return f64::NAN };
    if a.inner.ndim() != 2 {
        return f64::NAN;
    }
    let n = a.inner.shape()[0].min(a.inner.shape()[1]);
    let mut s = 0.0;
    for i in 0..n {
        s += a.inner[IxDyn(&[i, i])];
    }
    s
}

// ── norm ────────────────────────────────────────────────────────────
//
// `ord` encoding:
//   vector input (1D): 0=L1, 1=L2/fro, 2=Linf, -1=L-inf-min
//   matrix input (2D): 0=1-norm (max col sum), 1=fro, 2=inf-norm (max row sum),
//                      3=spectral (largest singular value)

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_norm_f64(pa: *const NDArrayF64, ord: i32) -> f64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return f64::NAN };
    match a.inner.ndim() {
        1 => {
            match ord {
                0 => a.inner.iter().map(|x| x.abs()).sum(),
                1 => a.inner.iter().map(|x| x * x).sum::<f64>().sqrt(),
                2 => a.inner.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
                -1 => a.inner.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min),
                _ => f64::NAN,
            }
        }
        2 => {
            let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return f64::NAN };
            match ord {
                0 => {
                    // max column-sum of |a_ij|
                    let mut best = 0.0_f64;
                    for j in 0..m.ncols() {
                        let mut s = 0.0;
                        for i in 0..m.nrows() { s += m[(i, j)].abs(); }
                        if s > best { best = s; }
                    }
                    best
                }
                1 => a.inner.iter().map(|x| x * x).sum::<f64>().sqrt(), // frobenius
                2 => {
                    let mut best = 0.0_f64;
                    for i in 0..m.nrows() {
                        let mut s = 0.0;
                        for j in 0..m.ncols() { s += m[(i, j)].abs(); }
                        if s > best { best = s; }
                    }
                    best
                }
                3 => {
                    // spectral norm = largest singular value
                    let svd = m.svd(false, false);
                    svd.singular_values.iter().copied().fold(0.0_f64, f64::max)
                }
                _ => f64::NAN,
            }
        }
        _ => f64::NAN,
    }
}

// ── QR / Cholesky / pinv / lstsq ────────────────────────────────────

#[repr(C)]
pub struct QrHandle {
    q: DMatrix<f64>,
    r: DMatrix<f64>,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_qr_compute_f64(pa: *const NDArrayF64) -> *mut QrHandle {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
    let qr = m.qr();
    Box::into_raw(Box::new(QrHandle { q: qr.q(), r: qr.r() }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_qr_q_f64(h: *const QrHandle) -> *mut NDArrayF64 {
    if h.is_null() { return std::ptr::null_mut(); }
    box_arr(dmatrix_to_nd(&(*h).q))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_qr_r_f64(h: *const QrHandle) -> *mut NDArrayF64 {
    if h.is_null() { return std::ptr::null_mut(); }
    box_arr(dmatrix_to_nd(&(*h).r))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_qr_free_f64(h: *mut QrHandle) {
    if !h.is_null() { drop(Box::from_raw(h)); }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_cholesky_f64(pa: *const NDArrayF64) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
    if m.nrows() != m.ncols() {
        return std::ptr::null_mut();
    }
    match m.cholesky() {
        Some(chol) => box_arr(dmatrix_to_nd(&chol.l())),
        None => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_pinv_f64(pa: *const NDArrayF64) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
    let rows = m.nrows();
    let cols = m.ncols();
    let svd = m.svd(true, true);
    let eps_factor = rows.max(cols) as f64 * f64::EPSILON;
    let max_sv = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = max_sv * eps_factor;
    match svd.pseudo_inverse(eps) {
        Ok(pi) => box_arr(dmatrix_to_nd(&pi)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_lstsq_f64(
    pa: *const NDArrayF64,
    pb: *const NDArrayF64,
) -> *mut NDArrayF64 {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let b = match arr_ref(pb) { Some(a) => a, None => return std::ptr::null_mut() };
    let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
    let svd = m.clone().svd(true, true);
    let eps_factor = (m.nrows().max(m.ncols())) as f64 * f64::EPSILON;
    let max_sv = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = max_sv * eps_factor;
    match b.inner.ndim() {
        1 => {
            let v = match nd_to_dvector(&b.inner) { Some(v) => v, None => return std::ptr::null_mut() };
            if v.nrows() != m.nrows() { return std::ptr::null_mut(); }
            match svd.solve(&v, eps) {
                Ok(x) => box_arr(dvector_to_nd(&x)),
                Err(_) => std::ptr::null_mut(),
            }
        }
        2 => {
            let bm = match nd_to_dmatrix(&b.inner) { Some(m) => m, None => return std::ptr::null_mut() };
            if bm.nrows() != m.nrows() { return std::ptr::null_mut(); }
            match svd.solve(&bm, eps) {
                Ok(x) => box_arr(dmatrix_to_nd(&x)),
                Err(_) => std::ptr::null_mut(),
            }
        }
        _ => std::ptr::null_mut(),
    }
}

// ── SVD ─────────────────────────────────────────────────────────────

#[repr(C)]
pub struct SvdHandle {
    u: DMatrix<f64>,
    s: DVector<f64>,
    vt: DMatrix<f64>,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_svd_compute_f64(pa: *const NDArrayF64) -> *mut SvdHandle {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
    let svd = m.svd(true, true);
    let u = match svd.u { Some(u) => u, None => return std::ptr::null_mut() };
    let vt = match svd.v_t { Some(v) => v, None => return std::ptr::null_mut() };
    let s = svd.singular_values;
    Box::into_raw(Box::new(SvdHandle { u, s, vt }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_svd_u_f64(h: *const SvdHandle) -> *mut NDArrayF64 {
    if h.is_null() { return std::ptr::null_mut(); }
    box_arr(dmatrix_to_nd(&(*h).u))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_svd_s_f64(h: *const SvdHandle) -> *mut NDArrayF64 {
    if h.is_null() { return std::ptr::null_mut(); }
    box_arr(dvector_to_nd(&(*h).s))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_svd_vt_f64(h: *const SvdHandle) -> *mut NDArrayF64 {
    if h.is_null() { return std::ptr::null_mut(); }
    box_arr(dmatrix_to_nd(&(*h).vt))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_svd_free_f64(h: *mut SvdHandle) {
    if !h.is_null() { drop(Box::from_raw(h)); }
}

// ── Eigendecomposition (general real matrix → complex spectrum) ─────
//
// nalgebra exposes `Schur` (real Schur form) for general real matrices,
// from which we extract complex eigenvalues. Eigenvectors of a general
// real matrix are also complex; we expose them split into Re/Im parts.

#[repr(C)]
pub struct EigHandle {
    vals_re: Vec<f64>,
    vals_im: Vec<f64>,
    // For NU3 we ship eigenvalues only (the common case for sklearn-style
    // workflows). Eigenvectors would require a complex linear-algebra
    // routine that nalgebra doesn't provide directly without unsafe
    // FFI to LAPACK. Reserved for NU3.1 if demand emerges.
    n: usize,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_eig_compute_f64(pa: *const NDArrayF64) -> *mut EigHandle {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    let m = match nd_to_dmatrix(&a.inner) { Some(m) => m, None => return std::ptr::null_mut() };
    if m.nrows() != m.ncols() {
        return std::ptr::null_mut();
    }
    let n = m.nrows();
    let schur = m.schur();
    let mut vals_re = Vec::with_capacity(n);
    let mut vals_im = Vec::with_capacity(n);
    let cvals = schur.complex_eigenvalues();
    for v in cvals.iter() {
        vals_re.push(v.re);
        vals_im.push(v.im);
    }
    Box::into_raw(Box::new(EigHandle { vals_re, vals_im, n }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_eig_vals_re_f64(h: *const EigHandle) -> *mut NDArrayF64 {
    if h.is_null() { return std::ptr::null_mut(); }
    let r = &(*h).vals_re;
    box_arr(ArrayD::from_shape_vec(IxDyn(&[(*h).n]), r.clone()).expect("eig re"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_eig_vals_im_f64(h: *const EigHandle) -> *mut NDArrayF64 {
    if h.is_null() { return std::ptr::null_mut(); }
    let i = &(*h).vals_im;
    box_arr(ArrayD::from_shape_vec(IxDyn(&[(*h).n]), i.clone()).expect("eig im"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_eig_free_f64(h: *mut EigHandle) {
    if !h.is_null() { drop(Box::from_raw(h)); }
}

// ── FFT ─────────────────────────────────────────────────────────────
//
// Inputs and outputs are real f64 NDArrays. Complex spectra are split
// into separate Re/Im 1-D arrays. All FFTs are 1-D over the last axis
// for now (NU3 scope). Multi-axis FFT is NU5 work.

fn fft_planner() -> FftPlanner<f64> {
    FftPlanner::new()
}

// Forward complex-input FFT. Real input → Re/Im pair output (full length).
// Layout: caller passes one real 1D array; we return the *real* component;
// `np_fft_im_f64` returns the imaginary part computed from the *same*
// input. Since complex FFT of a real-only input has Hermitian symmetry,
// we compute and cache: callers must call `np_fft_compute` first.

#[repr(C)]
pub struct FftHandle {
    re: Vec<f64>,
    im: Vec<f64>,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_fft_compute_f64(pa: *const NDArrayF64, inverse: i32) -> *mut FftHandle {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    if a.inner.ndim() != 1 { return std::ptr::null_mut(); }
    let n = a.inner.len();
    if n == 0 {
        return Box::into_raw(Box::new(FftHandle { re: vec![], im: vec![] }));
    }
    let mut buf: Vec<Complex64> = a.inner.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let mut planner = fft_planner();
    let fft = if inverse != 0 { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
    fft.process(&mut buf);
    let scale = if inverse != 0 { 1.0 / (n as f64) } else { 1.0 };
    let re: Vec<f64> = buf.iter().map(|c| c.re * scale).collect();
    let im: Vec<f64> = buf.iter().map(|c| c.im * scale).collect();
    Box::into_raw(Box::new(FftHandle { re, im }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_fft_complex_compute_f64(
    pa_re: *const NDArrayF64,
    pa_im: *const NDArrayF64,
    inverse: i32,
) -> *mut FftHandle {
    let re_in = match arr_ref(pa_re) { Some(a) => a, None => return std::ptr::null_mut() };
    let im_in = match arr_ref(pa_im) { Some(a) => a, None => return std::ptr::null_mut() };
    if re_in.inner.ndim() != 1 || im_in.inner.ndim() != 1 || re_in.inner.len() != im_in.inner.len() {
        return std::ptr::null_mut();
    }
    let n = re_in.inner.len();
    if n == 0 {
        return Box::into_raw(Box::new(FftHandle { re: vec![], im: vec![] }));
    }
    let mut buf: Vec<Complex64> = re_in.inner.iter().zip(im_in.inner.iter())
        .map(|(&r, &i)| Complex64::new(r, i)).collect();
    let mut planner = fft_planner();
    let fft = if inverse != 0 { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
    fft.process(&mut buf);
    let scale = if inverse != 0 { 1.0 / (n as f64) } else { 1.0 };
    let re: Vec<f64> = buf.iter().map(|c| c.re * scale).collect();
    let im: Vec<f64> = buf.iter().map(|c| c.im * scale).collect();
    Box::into_raw(Box::new(FftHandle { re, im }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_fft_re_f64(h: *const FftHandle) -> *mut NDArrayF64 {
    if h.is_null() { return std::ptr::null_mut(); }
    let r = &(*h).re;
    box_arr(ArrayD::from_shape_vec(IxDyn(&[r.len()]), r.clone()).expect("fft re"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_fft_im_f64(h: *const FftHandle) -> *mut NDArrayF64 {
    if h.is_null() { return std::ptr::null_mut(); }
    let i = &(*h).im;
    box_arr(ArrayD::from_shape_vec(IxDyn(&[i.len()]), i.clone()).expect("fft im"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_fft_free_f64(h: *mut FftHandle) {
    if !h.is_null() { drop(Box::from_raw(h)); }
}

// rfft: real input → Hermitian-truncated complex output (length n/2+1).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_rfft_compute_f64(pa: *const NDArrayF64) -> *mut FftHandle {
    let a = match arr_ref(pa) { Some(a) => a, None => return std::ptr::null_mut() };
    if a.inner.ndim() != 1 { return std::ptr::null_mut(); }
    let n = a.inner.len();
    if n == 0 {
        return Box::into_raw(Box::new(FftHandle { re: vec![], im: vec![] }));
    }
    let mut buf: Vec<Complex64> = a.inner.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let mut planner = fft_planner();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buf);
    let out_n = n / 2 + 1;
    let re: Vec<f64> = buf.iter().take(out_n).map(|c| c.re).collect();
    let im: Vec<f64> = buf.iter().take(out_n).map(|c| c.im).collect();
    Box::into_raw(Box::new(FftHandle { re, im }))
}

// irfft: Hermitian-truncated input (Re/Im length n_out/2+1) → real
// output of length `n_out`. `n_out` is required because length is
// ambiguous (n_out could be 2k or 2k+1 for the same input length k+1).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_irfft_f64(
    pa_re: *const NDArrayF64,
    pa_im: *const NDArrayF64,
    n_out: i64,
) -> *mut NDArrayF64 {
    let re_in = match arr_ref(pa_re) { Some(a) => a, None => return std::ptr::null_mut() };
    let im_in = match arr_ref(pa_im) { Some(a) => a, None => return std::ptr::null_mut() };
    if re_in.inner.ndim() != 1 || im_in.inner.ndim() != 1 { return std::ptr::null_mut(); }
    if re_in.inner.len() != im_in.inner.len() { return std::ptr::null_mut(); }
    if n_out < 0 { return std::ptr::null_mut(); }
    let n = n_out as usize;
    if n == 0 {
        return box_arr(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).expect("irfft empty"));
    }
    let half = re_in.inner.len();
    // Reconstruct the full Hermitian-symmetric complex spectrum of length n.
    let mut buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..half.min(n) {
        buf[k] = Complex64::new(re_in.inner[IxDyn(&[k])], im_in.inner[IxDyn(&[k])]);
    }
    for k in 1..n {
        let mirror = n - k;
        if mirror < half && mirror != k {
            // Already filled below half via direct path, but mirror may
            // exceed the supplied truncated bins; fill via conjugate.
            if mirror >= half {
                // can't happen since mirror < half
            }
        }
        if k >= half && n - k < half {
            let src = n - k;
            buf[k] = Complex64::new(re_in.inner[IxDyn(&[src])], -im_in.inner[IxDyn(&[src])]);
        }
    }
    let mut planner = fft_planner();
    let fft = planner.plan_fft_inverse(n);
    fft.process(&mut buf);
    let scale = 1.0 / (n as f64);
    let out: Vec<f64> = buf.iter().map(|c| c.re * scale).collect();
    box_arr(ArrayD::from_shape_vec(IxDyn(&[n]), out).expect("irfft out"))
}

// Touch Arc so the `use` is not warned-unused (rustfft re-exports it).
#[doc(hidden)]
pub fn _link_arc(_: Arc<()>) {}

// ── NU4: Ternary / masked kernels ────────────────────────────────────
//
// Mask convention (consistent with NU2 `np_cmp_f64`): an `f64` array
// where `0.0` ≡ false and any non-zero value ≡ true. NaN is treated as
// false (NumPy-aligned).
//
// All kernels broadcast against the destination/output shape. In-place
// variants validate that `mask` and `src` broadcast onto `dst`'s shape
// without reshaping `dst`.

#[inline]
fn is_true(m: f64) -> bool {
    !(m == 0.0 || m.is_nan())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_where_f64(
    p_mask: *const NDArrayF64,
    p_x: *const NDArrayF64,
    p_y: *const NDArrayF64,
) -> *mut NDArrayF64 {
    let m = match arr_ref(p_mask) { Some(a) => a, None => return std::ptr::null_mut() };
    let x = match arr_ref(p_x)    { Some(a) => a, None => return std::ptr::null_mut() };
    let y = match arr_ref(p_y)    { Some(a) => a, None => return std::ptr::null_mut() };
    // Broadcast all three together: shape(mask) ⨂ shape(x) ⨂ shape(y).
    let s1 = match broadcast_shape(m.inner.shape(), x.inner.shape()) {
        Some(s) => s, None => return std::ptr::null_mut(),
    };
    let out_shape = match broadcast_shape(&s1, y.inner.shape()) {
        Some(s) => s, None => return std::ptr::null_mut(),
    };
    let mv = match broadcast_view(&m.inner, &out_shape) { Some(v) => v, None => return std::ptr::null_mut() };
    let xv = match broadcast_view(&x.inner, &out_shape) { Some(v) => v, None => return std::ptr::null_mut() };
    let yv = match broadcast_view(&y.inner, &out_shape) { Some(v) => v, None => return std::ptr::null_mut() };
    let out = ndarray::Zip::from(&mv).and(&xv).and(&yv)
        .map_collect(|m, x, y| if is_true(*m) { *x } else { *y });
    box_arr(out)
}

// In-place copy: where mask is true, dst <- src. mask and src must
// broadcast onto dst's shape (dst is not resized).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_copyto_f64(
    p_dst: *mut NDArrayF64,
    p_src: *const NDArrayF64,
    p_mask: *const NDArrayF64,
) -> i32 {
    let dst = match arr_mut(p_dst)  { Some(a) => a, None => return -1 };
    let src = match arr_ref(p_src)  { Some(a) => a, None => return -1 };
    let msk = match arr_ref(p_mask) { Some(a) => a, None => return -1 };
    let target: Vec<usize> = dst.inner.shape().to_vec();
    // src and mask must broadcast onto target.
    let sv = match broadcast_view(&src.inner, &target) { Some(v) => v, None => return -2 };
    let mv = match broadcast_view(&msk.inner, &target) { Some(v) => v, None => return -2 };
    ndarray::Zip::from(&mut dst.inner).and(&sv).and(&mv).for_each(|d, s, m| {
        if is_true(*m) { *d = *s; }
    });
    0
}

// In-place scalar masked-assign: where mask is true, dst <- value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_masked_assign_scalar_f64(
    p_dst: *mut NDArrayF64,
    p_mask: *const NDArrayF64,
    value: f64,
) -> i32 {
    let dst = match arr_mut(p_dst)  { Some(a) => a, None => return -1 };
    let msk = match arr_ref(p_mask) { Some(a) => a, None => return -1 };
    let target: Vec<usize> = dst.inner.shape().to_vec();
    let mv = match broadcast_view(&msk.inner, &target) { Some(v) => v, None => return -2 };
    ndarray::Zip::from(&mut dst.inner).and(&mv).for_each(|d, m| {
        if is_true(*m) { *d = value; }
    });
    0
}

// Clamp: out[i] = min(max(arr[i], lo), hi). NaN in lo/hi disables that bound.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_clip_f64(
    p: *const NDArrayF64,
    lo: f64,
    hi: f64,
) -> *mut NDArrayF64 {
    let a = match arr_ref(p) { Some(a) => a, None => return std::ptr::null_mut() };
    let out = a.inner.mapv(|x| {
        let mut v = x;
        if !lo.is_nan() && v < lo { v = lo; }
        if !hi.is_nan() && v > hi { v = hi; }
        v
    });
    box_arr(out)
}

// ── NU5: PRNG + .npy / .npz IO ──────────────────────────────────────

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform, Binomial, Poisson, Gamma, Beta};
use std::cell::RefCell;
use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom, Cursor};
use std::path::Path;

thread_local! {
    static GLOBAL_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
}

#[unsafe(no_mangle)]
pub extern "C" fn np_random_seed(seed: u64) {
    GLOBAL_RNG.with(|r| *r.borrow_mut() = StdRng::seed_from_u64(seed));
}

unsafe fn dims_vec(rank: i64, dims_ptr: *const i64) -> Vec<usize> {
    if rank <= 0 || dims_ptr.is_null() { return vec![]; }
    let s = slice::from_raw_parts(dims_ptr, rank as usize);
    s.iter().map(|&d| if d < 0 { 0 } else { d as usize }).collect()
}

fn fill_array<F: FnMut() -> f64>(shape: Vec<usize>, mut f: F) -> *mut NDArrayF64 {
    let n: usize = shape.iter().product();
    let mut data: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n { data.push(f()); }
    match ArrayD::from_shape_vec(IxDyn(&shape), data) {
        Ok(a) => box_arr(a),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_random_uniform_f64(
    low: f64, high: f64, rank: i64, dims_ptr: *const i64,
) -> *mut NDArrayF64 {
    let shape = dims_vec(rank, dims_ptr);
    if low >= high { return std::ptr::null_mut(); }
    let dist = Uniform::new(low, high);
    GLOBAL_RNG.with(|r| {
        let mut rng = r.borrow_mut();
        fill_array(shape, || dist.sample(&mut *rng))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_random_normal_f64(
    mean: f64, std: f64, rank: i64, dims_ptr: *const i64,
) -> *mut NDArrayF64 {
    let shape = dims_vec(rank, dims_ptr);
    let dist = match Normal::new(mean, std) { Ok(d) => d, Err(_) => return std::ptr::null_mut() };
    GLOBAL_RNG.with(|r| {
        let mut rng = r.borrow_mut();
        fill_array(shape, || dist.sample(&mut *rng))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_random_binomial_f64(
    n_trials: i64, p: f64, rank: i64, dims_ptr: *const i64,
) -> *mut NDArrayF64 {
    if n_trials < 0 { return std::ptr::null_mut(); }
    let dist = match Binomial::new(n_trials as u64, p) { Ok(d) => d, Err(_) => return std::ptr::null_mut() };
    let shape = dims_vec(rank, dims_ptr);
    GLOBAL_RNG.with(|r| {
        let mut rng = r.borrow_mut();
        fill_array(shape, || dist.sample(&mut *rng) as f64)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_random_poisson_f64(
    lambda: f64, rank: i64, dims_ptr: *const i64,
) -> *mut NDArrayF64 {
    let dist = match Poisson::new(lambda) { Ok(d) => d, Err(_) => return std::ptr::null_mut() };
    let shape = dims_vec(rank, dims_ptr);
    GLOBAL_RNG.with(|r| {
        let mut rng = r.borrow_mut();
        fill_array(shape, || {
            let v: f64 = dist.sample(&mut *rng);
            v.round()
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_random_gamma_f64(
    shape_k: f64, scale: f64, rank: i64, dims_ptr: *const i64,
) -> *mut NDArrayF64 {
    let dist = match Gamma::new(shape_k, scale) { Ok(d) => d, Err(_) => return std::ptr::null_mut() };
    let shape = dims_vec(rank, dims_ptr);
    GLOBAL_RNG.with(|r| {
        let mut rng = r.borrow_mut();
        fill_array(shape, || dist.sample(&mut *rng))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_random_beta_f64(
    alpha: f64, beta: f64, rank: i64, dims_ptr: *const i64,
) -> *mut NDArrayF64 {
    let dist = match Beta::new(alpha, beta) { Ok(d) => d, Err(_) => return std::ptr::null_mut() };
    let shape = dims_vec(rank, dims_ptr);
    GLOBAL_RNG.with(|r| {
        let mut rng = r.borrow_mut();
        fill_array(shape, || dist.sample(&mut *rng))
    })
}

// Sample `n` elements from a 1-D `arr`. If `replace != 0`, sampling is
// with replacement (any `n` is allowed). Otherwise `n <= arr.len()` and
// each index is drawn at most once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_random_choice_f64(
    p_arr: *const NDArrayF64,
    n: i64,
    replace: i32,
) -> *mut NDArrayF64 {
    let a = match arr_ref(p_arr) { Some(a) => a, None => return std::ptr::null_mut() };
    if a.inner.ndim() != 1 || n < 0 { return std::ptr::null_mut(); }
    let len = a.inner.len();
    if len == 0 { return std::ptr::null_mut(); }
    let n_usz = n as usize;
    let src: Vec<f64> = a.inner.iter().copied().collect();
    let mut out: Vec<f64> = Vec::with_capacity(n_usz);
    GLOBAL_RNG.with(|r| {
        let mut rng = r.borrow_mut();
        if replace != 0 {
            for _ in 0..n_usz {
                out.push(src[rng.gen_range(0..len)]);
            }
        } else {
            if n_usz > len { return; }
            // Fisher-Yates partial shuffle.
            let mut pool: Vec<usize> = (0..len).collect();
            for i in 0..n_usz {
                let j = rng.gen_range(i..len);
                pool.swap(i, j);
                out.push(src[pool[i]]);
            }
        }
    });
    if out.len() != n_usz { return std::ptr::null_mut(); }
    box_arr(ArrayD::from_shape_vec(IxDyn(&[n_usz]), out).expect("choice out"))
}

// In-place shuffle of a 1-D array (Fisher-Yates).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_random_shuffle_f64(p: *mut NDArrayF64) -> i32 {
    let a = match arr_mut(p) { Some(a) => a, None => return -1 };
    if a.inner.ndim() != 1 { return -2; }
    let len = a.inner.len();
    if len < 2 { return 0; }
    // Build a flat &mut [f64] view; `as_slice_mut` succeeds for owned C-contiguous arrays.
    let s = match a.inner.as_slice_mut() { Some(s) => s, None => return -3 };
    GLOBAL_RNG.with(|r| {
        let mut rng = r.borrow_mut();
        for i in (1..len).rev() {
            let j = rng.gen_range(0..=i);
            s.swap(i, j);
        }
    });
    0
}

// ── NPY (NumPy on-disk) format v1, f64 little-endian, C-contiguous ──
//
// File layout:
//   magic   : b"\x93NUMPY"          (6 bytes)
//   version : (1, 0)                (2 bytes)
//   hlen    : u16 little-endian     (2 bytes)
//   header  : ASCII dict, padded with spaces + b'\n' so the total
//             header (magic+ver+hlen+header) is 16-byte aligned.
//   data    : raw little-endian f64 bytes, row-major (C order).
//
// On read, we accept v1.0 / v2.0 / v3.0 (only hlen-width differs)
// and require `descr` in `<f8`/`=f8`/`|f8` and `fortran_order: False`.

fn build_npy_header(shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let mut header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );
    // Pre-prefix length = 6 magic + 2 version + 2 hlen = 10
    let unpadded = 10 + header.len() + 1; // +1 for trailing \n
    let pad = (16 - (unpadded % 16)) % 16;
    for _ in 0..pad { header.push(' '); }
    header.push('\n');
    header.into_bytes()
}

unsafe fn cstr_path<'a>(p: *const c_char) -> Option<&'a Path> {
    if p.is_null() { return None; }
    let cs = std::ffi::CStr::from_ptr(p);
    cs.to_str().ok().map(Path::new)
}

fn write_npy_to<W: Write>(w: &mut W, arr: &ArrayD<f64>) -> std::io::Result<()> {
    let header = build_npy_header(arr.shape());
    w.write_all(&[0x93])?;
    w.write_all(b"NUMPY")?;
    w.write_all(&[1, 0])?; // version 1.0
    let hlen = header.len() as u16;
    w.write_all(&hlen.to_le_bytes())?;
    w.write_all(&header)?;
    // Ensure C-contiguous; clone if not.
    if arr.is_standard_layout() {
        let bytes: &[u8] = unsafe {
            slice::from_raw_parts(
                arr.as_ptr() as *const u8,
                arr.len() * std::mem::size_of::<f64>(),
            )
        };
        w.write_all(bytes)?;
    } else {
        let owned = arr.as_standard_layout();
        let bytes: &[u8] = unsafe {
            slice::from_raw_parts(
                owned.as_ptr() as *const u8,
                owned.len() * std::mem::size_of::<f64>(),
            )
        };
        w.write_all(bytes)?;
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_save_npy(
    p: *const NDArrayF64,
    path_cstr: *const c_char,
) -> i32 {
    let a = match arr_ref(p) { Some(a) => a, None => return -1 };
    let path = match cstr_path(path_cstr) { Some(p) => p, None => return -2 };
    let mut f = match File::create(path) { Ok(f) => f, Err(_) => return -3 };
    match write_npy_to(&mut f, &a.inner) {
        Ok(_) => 0,
        Err(_) => -4,
    }
}

// Parse the dict-ish header for shape + dtype + fortran_order. Returns
// (shape, fortran_order) or Err on malformed.
fn parse_npy_header(h: &str) -> Result<(Vec<usize>, bool), ()> {
    // Find descr value.
    let descr_key = "'descr':";
    let i = h.find(descr_key).ok_or(())?;
    let rest = &h[i + descr_key.len()..];
    let q1 = rest.find('\'').ok_or(())?;
    let after = &rest[q1 + 1..];
    let q2 = after.find('\'').ok_or(())?;
    let descr = &after[..q2];
    // Accept <f8 / =f8 / |f8 (little-endian or native f64 only).
    let ok = matches!(descr, "<f8" | "=f8" | "|f8");
    if !ok { return Err(()); }

    // fortran_order
    let fo_key = "'fortran_order':";
    let j = h.find(fo_key).ok_or(())?;
    let fo_rest = &h[j + fo_key.len()..];
    let fortran_order = fo_rest.trim_start().starts_with("True");

    // shape tuple
    let sh_key = "'shape':";
    let k = h.find(sh_key).ok_or(())?;
    let after_sh = &h[k + sh_key.len()..];
    let lp = after_sh.find('(').ok_or(())?;
    let rp = after_sh[lp..].find(')').ok_or(())?;
    let body = &after_sh[lp + 1..lp + rp];
    let mut dims: Vec<usize> = Vec::new();
    for tok in body.split(',') {
        let t = tok.trim();
        if t.is_empty() { continue; }
        let d: usize = t.parse().map_err(|_| ())?;
        dims.push(d);
    }
    Ok((dims, fortran_order))
}

fn read_npy_from<R: Read>(r: &mut R) -> std::io::Result<ArrayD<f64>> {
    let mut prefix = [0u8; 10];
    r.read_exact(&mut prefix)?;
    if prefix[0] != 0x93 || &prefix[1..6] != b"NUMPY" {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    let major = prefix[6];
    let _minor = prefix[7];
    // For v1.0 the hlen is u16; for v2.0/v3.0 it's u32 (we read 2 more bytes).
    let hlen: usize = if major >= 2 {
        let mut more = [0u8; 2];
        r.read_exact(&mut more)?;
        let buf = [prefix[8], prefix[9], more[0], more[1]];
        u32::from_le_bytes(buf) as usize
    } else {
        u16::from_le_bytes([prefix[8], prefix[9]]) as usize
    };
    let mut hbuf = vec![0u8; hlen];
    r.read_exact(&mut hbuf)?;
    let header = std::str::from_utf8(&hbuf)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "header utf8"))?;
    let (shape, fortran) = parse_npy_header(header)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "header parse"))?;
    if fortran {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "fortran_order not supported"));
    }
    let n: usize = shape.iter().product();
    let mut data: Vec<f64> = vec![0.0; n];
    let bytes: &mut [u8] = unsafe {
        slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * std::mem::size_of::<f64>())
    };
    r.read_exact(bytes)?;
    ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "shape mismatch"))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_load_npy(path_cstr: *const c_char) -> *mut NDArrayF64 {
    let path = match cstr_path(path_cstr) { Some(p) => p, None => return std::ptr::null_mut() };
    let mut f = match File::open(path) { Ok(f) => f, Err(_) => return std::ptr::null_mut() };
    match read_npy_from(&mut f) {
        Ok(a) => box_arr(a),
        Err(_) => std::ptr::null_mut(),
    }
}

// NPZ — multi-array container (NumPy-compatible). On disk it is a ZIP
// archive where each entry is named `<key>.npy` and stores the raw NPY
// payload. We use the deflate variant (NumPy reads either).
//
// API is incremental: open a writer handle, add entries, close it. On
// the read side we open an archive handle, query names, then fetch
// individual arrays.

pub struct NpzWriter {
    inner: Option<zip::ZipWriter<File>>,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_npz_writer_open(path_cstr: *const c_char) -> *mut NpzWriter {
    let path = match cstr_path(path_cstr) { Some(p) => p, None => return std::ptr::null_mut() };
    let f = match File::create(path) { Ok(f) => f, Err(_) => return std::ptr::null_mut() };
    let w = zip::ZipWriter::new(f);
    Box::into_raw(Box::new(NpzWriter { inner: Some(w) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_npz_writer_add(
    h: *mut NpzWriter,
    name_cstr: *const c_char,
    p_arr: *const NDArrayF64,
) -> i32 {
    if h.is_null() { return -1; }
    let writer = match (*h).inner.as_mut() { Some(w) => w, None => return -1 };
    let a = match arr_ref(p_arr) { Some(a) => a, None => return -2 };
    let name_cs = if name_cstr.is_null() { return -3 } else { std::ffi::CStr::from_ptr(name_cstr) };
    let name = match name_cs.to_str() { Ok(s) => format!("{}.npy", s), Err(_) => return -3 };
    let opts = zip::write::FileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .large_file(false);
    if writer.start_file(name, opts).is_err() { return -4; }
    if write_npy_to(writer, &a.inner).is_err() { return -5; }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_npz_writer_close(h: *mut NpzWriter) -> i32 {
    if h.is_null() { return -1; }
    let mut boxed = Box::from_raw(h);
    if let Some(mut w) = boxed.inner.take() {
        if w.finish().is_err() { return -2; }
    }
    0
}

pub struct NpzReader {
    inner: zip::ZipArchive<File>,
    names: Vec<String>,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_npz_reader_open(path_cstr: *const c_char) -> *mut NpzReader {
    let path = match cstr_path(path_cstr) { Some(p) => p, None => return std::ptr::null_mut() };
    let f = match File::open(path) { Ok(f) => f, Err(_) => return std::ptr::null_mut() };
    let archive = match zip::ZipArchive::new(f) { Ok(a) => a, Err(_) => return std::ptr::null_mut() };
    let names: Vec<String> = archive.file_names()
        .map(|s| {
            // Strip trailing `.npy` for ergonomic exposure.
            if let Some(stripped) = s.strip_suffix(".npy") { stripped.to_string() } else { s.to_string() }
        })
        .collect();
    Box::into_raw(Box::new(NpzReader { inner: archive, names }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_npz_reader_count(h: *const NpzReader) -> i64 {
    if h.is_null() { return -1; }
    (*h).names.len() as i64
}

// Returns a *mut c_char that the caller must free with np_free_string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_npz_reader_name(h: *const NpzReader, i: i64) -> *mut c_char {
    if h.is_null() || i < 0 { return std::ptr::null_mut(); }
    let names = &(*h).names;
    let idx = i as usize;
    if idx >= names.len() { return std::ptr::null_mut(); }
    match CString::new(names[idx].clone()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// Look up `<name>.npy` in the archive, return decoded NDArray (or null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_npz_reader_get(
    h: *mut NpzReader,
    name_cstr: *const c_char,
) -> *mut NDArrayF64 {
    if h.is_null() || name_cstr.is_null() { return std::ptr::null_mut(); }
    let name = match std::ffi::CStr::from_ptr(name_cstr).to_str() {
        Ok(s) => format!("{}.npy", s),
        Err(_) => return std::ptr::null_mut(),
    };
    let archive = &mut (*h).inner;
    let mut entry = match archive.by_name(&name) {
        Ok(e) => e,
        Err(_) => return std::ptr::null_mut(),
    };
    // Pull the whole entry into memory then parse as NPY. Entries in a
    // typical NPZ are small (per-tensor); streaming would complicate
    // error handling without measurable benefit here.
    let mut buf: Vec<u8> = Vec::with_capacity(entry.size() as usize);
    if entry.read_to_end(&mut buf).is_err() { return std::ptr::null_mut(); }
    let mut cur = Cursor::new(buf);
    cur.seek(SeekFrom::Start(0)).ok();
    match read_npy_from(&mut cur) {
        Ok(a) => box_arr(a),
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_npz_reader_close(h: *mut NpzReader) {
    if h.is_null() { return; }
    drop(Box::from_raw(h));
}

// ── NU5 (interop foundation): raw data pointer ──────────────────────
//
// Returns a read-only pointer to the underlying contiguous f64 buffer.
// If the array isn't C-contiguous, returns null (caller should `.copy()`
// first). Pointer is valid as long as the source NDArray is alive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_data_ptr_f64(p: *const NDArrayF64) -> *const f64 {
    let a = match arr_ref(p) { Some(a) => a, None => return std::ptr::null() };
    if !a.inner.is_standard_layout() { return std::ptr::null(); }
    a.inner.as_ptr()
}

// ── NU5b: Arrow C Data Interface export (zero-copy via consume) ─────
//
// Consumes the NDArray handle: steals its Vec<f64> via
// into_raw_vec_and_offset(), hands ownership to an Arrow Float64Array,
// and exports it through the Arrow C Data Interface into caller-
// allocated FFI_ArrowSchema (72 B) + FFI_ArrowArray (80 B) structs.
//
// After this returns 0, the NDArray pointer is invalid — the caller
// MUST NOT call np_array_close on it. The Chuks wrapper should mark
// itself closed so .close() is a no-op.
//
// Constraints: 1-D, standard (C-contig) layout, zero offset.
// Returns: 0 ok, -1 null arg, -2 ffi export failure, -3 layout error.
use arrow::array::{Array as ArrowArrayTrait, Float64Array};
use arrow::buffer::ScalarBuffer as ArrowScalarBuffer;
use arrow::ffi::{to_ffi, FFI_ArrowArray, FFI_ArrowSchema};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn np_array_to_arrow_cdi(
    p: *mut NDArrayF64,
    out_schema: *mut FFI_ArrowSchema,
    out_array: *mut FFI_ArrowArray,
) -> std::os::raw::c_int {
    if p.is_null() || out_schema.is_null() || out_array.is_null() {
        return -1;
    }
    {
        let r = match arr_ref(p) { Some(r) => r, None => return -1 };
        if r.inner.ndim() != 1 { return -3; }
        if !r.inner.is_standard_layout() { return -3; }
    }
    let owned: Box<NDArrayF64> = Box::from_raw(p);
    let (vec, off) = owned.inner.into_raw_vec_and_offset();
    if off.unwrap_or(0) != 0 { return -3; }
    let scalar: ArrowScalarBuffer<f64> = ArrowScalarBuffer::from(vec);
    let arr = Float64Array::new(scalar, None);
    let data = ArrowArrayTrait::to_data(&arr);
    match to_ffi(&data) {
        Ok((ffi_arr, ffi_sch)) => {
            std::ptr::write(out_schema, ffi_sch);
            std::ptr::write(out_array, ffi_arr);
            0
        }
        Err(_) => -2,
    }
}

