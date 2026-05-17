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
    CString::new("chuks_numpy_shim 0.1.0 (ndarray 0.16)")
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
