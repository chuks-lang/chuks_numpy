# chuks_numpy — User Guide

This guide explains the concepts behind `chuks_numpy`: arrays, shapes,
indexing, broadcasting, lifetimes, and performance. It complements the
method-by-method [NDArray API](./ndarray.md) and [NumPy API](./numpy.md)
references, and is organized roughly the same way as the NumPy user guide on
[numpy.org](https://numpy.org/doc/stable/user/).

## Contents

1. [What is an NDArray?](#1-what-is-an-ndarray)
2. [Constructing arrays](#2-constructing-arrays)
3. [Shape, rank, and `size()`](#3-shape-rank-and-size)
4. [Indexing & slicing](#4-indexing--slicing)
5. [Reshape, transpose, take](#5-reshape-transpose-take)
6. [Elementwise arithmetic & broadcasting](#6-elementwise-arithmetic--broadcasting)
7. [Unary math & comparisons](#7-unary-math--comparisons)
8. [Reductions (whole-array and axis-wise)](#8-reductions-whole-array-and-axis-wise)
9. [Boolean masks & ternary kernels](#9-boolean-masks--ternary-kernels)
10. [Linear algebra](#10-linear-algebra)
11. [Random sampling](#11-random-sampling)
12. [FFT](#12-fft)
13. [Persistence — `.npy` / `.npz`](#13-persistence--npy--npz)
14. [Zero-copy Arrow interop](#14-zero-copy-arrow-interop)
15. [Memory model and `close()`](#15-memory-model-and-close)
16. [Performance notes](#16-performance-notes)
17. [Errors & edge cases](#17-errors--edge-cases)
18. [Migrating from Python NumPy](#18-migrating-from-python-numpy)

---

## 1. What is an NDArray?

An `NDArray` is an n-dimensional dense array of `float` (IEEE-754 binary64).
The numbers are stored row-major (C order) in a single Rust-side buffer; the
Chuks `NDArray` value is a thin handle that holds a pointer to that buffer
plus a cached shape vector.

Three consequences:

- **All numbers are `float`.** There is no integer or complex tensor type
  yet (complex outputs from `fft` / `eig` are returned as a pair of real
  arrays — see [§12](#12-fft) and `EigResult` in
  [NumPy API](./numpy.md#result-wrapper-classes)).
- **The data is not visible to Chuks introspection.** `println(a)` prints
  `Instance(NDArray)`, not the contents. Use `a.toString()` (NumPy-style any
  rank) or `a.toFloats()` (flat row-major copy) instead.
- **You must call `close()` on every array you receive.** The Chuks GC
  releases the wrapper, not the Rust buffer. Details in
  [§15](#15-memory-model-and-close).

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()
const a = np.from1d([1.0, 2.0, 3.0])

println(a)              // Instance(NDArray)         ← not useful
println(a.toString())   // [1, 2, 3]
println(a.toFloats())   // [1, 2, 3]                 ← flat row-major copy

a.close()
```

---

## 2. Constructing arrays

| Constructor                      | Result                                                                |
| -------------------------------- | --------------------------------------------------------------------- |
| `np.zeros([2, 3])`               | `2×3` of `0.0`                                                        |
| `np.ones([4])`                   | length-4 vector of `1.0`                                              |
| `np.full([2, 2], 7.5)`           | `2×2` filled with `7.5`                                               |
| `np.eye(3)`                      | `3×3` identity                                                        |
| `np.arange(0.0, 10.0, 1.0)`      | `[0.0, 1.0, …, 9.0]` (half-open `[start, stop)`, `step > 0`)          |
| `np.linspace(0.0, 1.0, 11)`      | 11 evenly spaced points in `[0.0, 1.0]` _inclusive_ of both endpoints |
| `np.from1d([1.0, 2.0, 3.0])`     | 1-D from a `[]float` literal                                          |
| `np.from2d([[1.0, 2.0], …])`     | 2-D from rows (`[][]float`, rectangular only — see below)             |
| `np.fromNd([2, 3, 4], flatVals)` | _any_ rank: pass shape vector + flat row-major `[]float`              |

### `fromNd` — the general constructor

Because Chuks does not have rank-polymorphic nested literals, `from1d` and
`from2d` are sugar over the underlying `fromNd(shape, flat)`. For 3-D, 4-D,
and higher you call `fromNd` directly (or `reshape` a 1-D array):

```chuks
// 3-D, shape [2, 2, 3] — 12 values, row-major
const a = np.fromNd([2, 2, 3], [
     1.0,  2.0,  3.0,    4.0,  5.0,  6.0,
     7.0,  8.0,  9.0,   10.0, 11.0, 12.0,
])

// equivalent via reshape
const b = np.arange(1.0, 13.0, 1.0).reshape([2, 2, 3])

// 4-D, shape [2, 3, 4, 5] — 120 values
const c = np.zeros([2, 3, 4, 5])      // or fromNd + a flat slice

a.close(); b.close(); c.close()
```

**Rules.** `from2d` requires all rows to have equal length and throws on
ragged input. `fromNd` enforces `product(shape) == values.length` and throws
otherwise.

---

## 3. Shape, rank, and `size()`

| Call          | Returns | Meaning                                       |
| ------------- | ------- | --------------------------------------------- |
| `a.ndim()`    | `int`   | number of dimensions                          |
| `a.shape()`   | `[]int` | per-axis sizes, e.g. `[2, 3, 4]`              |
| `a.dim(axis)` | `int`   | size of one axis, `-1` if `axis` out of range |
| `a.size()`    | `int`   | total elements (product of `shape`)           |

```chuks
const a = np.zeros([2, 3, 4])
println(a.ndim())       // 3
println(a.shape())      // [2, 3, 4]
println(a.dim(1))       // 3
println(a.size())        // 24
a.close()
```

---

## 4. Indexing & slicing

### Single-element access

`get(idxs)` and `set(idxs, v)` take an `[]int` of length `ndim()`.
0-based, row-major.

```chuks
const m = np.zeros([2, 3])
m.set([0, 1], 7.0)
println(m.get([0, 1]))   // 7
println(m.get([5, 5]))   // 0  (out-of-range returns 0.0 silently)
m.close()
```

`set` returns `bool` — `true` on success, `false` if `idxs` is the wrong
length or any index is out of range. `get` returns `0.0` on out-of-range and
does not throw. _Check the return values if correctness matters._

### Slicing — `slice(starts, ends, steps)`

Per-axis half-open `[start, end)` interval, with positive `step`. Negative
indices are normalized Python-style (`-1` = last). Lengths of `starts`,
`ends`, `steps` must all equal `ndim()`.

```chuks
const m = np.arange(0.0, 12.0, 1.0).reshape([3, 4])
// 0  1  2  3
// 4  5  6  7
// 8  9 10 11

// rows [0, 2), cols [1, 4)  → upper-right 2×3 block
const s = m.slice([0, 1], [2, 4], [1, 1])
println(s.toFloats())     // [1, 2, 3, 5, 6, 7]

// every other column (step 2)
const e = m.slice([0, 0], [3, 4], [1, 2])
println(e.shape())        // [3, 2]

m.close(); s.close(); e.close()
```

`slice` always returns a new contiguous array (copy, not view). To save the
allocation when reading once, use `take` for fancy-index selection along a
single axis:

```chuks
const cols = m.take(1, [0, 3])    // columns 0 and 3
```

### What is _not_ supported (yet)

- **Bracket / colon notation.** You write `m.get([i, j])`, not `m[i, j]`,
  and `m.slice(...)`, not `m[0:2, 1:4]`.
- **Boolean fancy indexing as an indexer.** Build a mask and combine with
  `np.where` / `np.maskedAssign` / `np.copyto` — see [§9](#9-boolean-masks--ternary-kernels).
- **Negative `step`.** All steps must be `> 0`. Flip an axis by combining
  `transpose` with a normal slice.

---

## 5. Reshape, transpose, take

```chuks
const v = np.arange(0.0, 12.0, 1.0)
const m = v.reshape([3, 4])    // total must match: 12 = 3·4
const t = m.transpose()        // reverses all axes — `m.T` style for 2-D
const c = m.take(1, [0, 3])    // pick columns 0 and 3 (fancy-index along axis 1)

v.close(); m.close(); t.close(); c.close()
```

`reshape` throws when the product of the new shape ≠ `size()`. `transpose`
reverses _all_ axes; there is no `swapaxes` yet — reshape + transpose covers
most needs.

---

## 6. Elementwise arithmetic & broadcasting

Every binary op (`add`, `sub`, `mul`, `div`, `pow`, `modulo`, `minimum`,
`maximum`) follows NumPy broadcasting rules:

1. Align shapes **right** (last axis first).
2. Two dimensions are compatible if they are equal _or_ one of them is `1`.
3. The result shape is the elementwise maximum.

```chuks
const a = np.ones([3, 4])
const b = np.from1d([10.0, 20.0, 30.0, 40.0])  // shape [4]

const c = a.add(b)        // broadcasts → shape [3, 4]; each row is [11,21,31,41]
const d = a.add(np.from2d([[100.0], [200.0], [300.0]]))   // shape [3, 1] → [3, 4]

a.close(); b.close(); c.close(); d.close()
```

If shapes can't broadcast, the Rust kernel throws. Inspect `a.shape()` and
`b.shape()` first if you're unsure.

### Scalar variants

`addScalar(v)`, `mulScalar(v)`, etc. avoid allocating a one-element array.
The reversed `r…Scalar(v)` forms compute `v OP a` (useful when subtraction
or division order matters):

```chuks
const a = np.from1d([1.0, 2.0, 4.0])
const b = a.rsubScalar(10.0)    // 10 - a → [9, 8, 6]
const c = a.rdivScalar(1.0)     //  1 / a → [1, 0.5, 0.25]
a.close(); b.close(); c.close()
```

### In-place variants

`addInPlace`, `mulInPlaceScalar`, etc. mutate the destination and return
`this`. Use these inside hot loops to avoid per-step allocation:

```chuks
const buf = np.zeros([1_000_000])
const inc = np.full([1_000_000], 0.001)

for (var i: int = 0; i < 100; i = i + 1) {
    buf.addInPlace(inc)        // 0 allocations per iteration
}
buf.close(); inc.close()
```

In-place ops require the destination shape to already be broadcast-compatible
with the source — they do _not_ allocate to grow `dst`.

---

## 7. Unary math & comparisons

Unary ops (`sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `abs`, `floor`,
`ceil`, `round`, `neg`) all return a new NDArray with the same shape.

Comparison ops (`eq`, `ne`, `lt`, `le`, `gt`, `ge`, plus scalar variants
`eqScalar(v)`, …) return a **mask** — an `NDArray` of `0.0` / `1.0` with the
same shape as the input. Combine with the ternary kernels in
[§9](#9-boolean-masks--ternary-kernels):

```chuks
const a = np.from1d([1.0, -2.0, 3.0, -4.0])
const m = a.gtScalar(0.0)        // [1, 0, 1, 0]
const r = a.abs()                // [1, 2, 3, 4]
a.close(); m.close(); r.close()
```

---

## 8. Reductions (whole-array and axis-wise)

Whole-array (scalar out): `sum`, `mean`, `min`, `max`, `prod`, `std`,
`variance`, `argmin`, `argmax`, `any`, `all`.

Axis-wise (NDArray out): same names with `Axis` suffix and signature
`(axis: int, keepdims: bool)`. `keepdims = true` preserves the reduced axis
with size 1, matching NumPy.

```chuks
const m = np.from2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
println(m.sum())                          // 21
println(m.sumAxis(1, false).toFloats())   // [6, 15]            shape [2]
println(m.sumAxis(1, true).shape())       // [2, 1]             keepdims
m.close()
```

`std` and `variance` are _population_ statistics (divide by N, not N-1).
`argmin` / `argmax` (whole-array) return the flat row-major index.

---

## 9. Boolean masks & ternary kernels

Build masks with comparison ops, then route values through the fused
ternary kernels — these are typed-FFI fast paths that avoid Chuks-side
loops.

| Kernel                              | What it does                              |
| ----------------------------------- | ----------------------------------------- |
| `np.where(mask, x, y)`              | `mask ? x : y`, new array                 |
| `np.copyto(dst, src, mask)`         | in-place `dst[i] = src[i]` where mask     |
| `np.maskedAssign(dst, mask, value)` | in-place `dst[i] = value` where mask      |
| `np.clip(arr, lo, hi)`              | clamp into `[lo, hi]`                     |
| `np.select(conds, choices, def)`    | first matching condition; otherwise `def` |

```chuks
const a = np.from1d([-2.0, -1.0, 0.0, 1.0, 2.0])

// ReLU: max(a, 0) via a mask
const mask = a.geScalar(0.0)
const zeros = np.zeros([5])
const relu = np.where(mask, a, zeros)
println(relu.toFloats())                 // [0, 0, 0, 1, 2]

// Equivalent in-place
np.maskedAssign(a, a.ltScalar(0.0), 0.0) // mask freed by Rust kernel
println(a.toFloats())                    // [0, 0, 0, 1, 2]

a.close(); mask.close(); zeros.close(); relu.close()
```

---

## 10. Linear algebra

All operate on 2-D matrices (`matmul` also supports batched batched-matmul
for higher-rank inputs).

| Routine           | Returns     | Notes                                          |
| ----------------- | ----------- | ---------------------------------------------- |
| `np.matmul(a, b)` | `NDArray`   | matrix × matrix; also batched on higher ndims  |
| `np.dot(a, b)`    | `float`     | inner product on 1-D vectors                   |
| `np.inv(a)`       | `NDArray`   | matrix inverse                                 |
| `np.solve(a, b)`  | `NDArray`   | `a·x = b`; faster than `inv(a) · b`            |
| `np.det(a)`       | `float`     |                                                |
| `np.trace(a)`     | `float`     |                                                |
| `np.norm(a, ord)` | `float`     | `ord = 1`/`2`/`-1` etc.                        |
| `np.cholesky(a)`  | `NDArray`   | lower-triangular `L`, `L·Lᵀ = a`               |
| `np.pinv(a)`      | `NDArray`   | Moore–Penrose pseudoinverse                    |
| `np.lstsq(a, b)`  | `NDArray`   | least-squares `argmin ‖a·x − b‖`               |
| `np.qr(a)`        | `QrResult`  | `.q`, `.r`                                     |
| `np.svd(a)`       | `SvdResult` | `.u`, `.s`, `.vt`                              |
| `np.eig(a)`       | `EigResult` | `.valuesRe`, `.valuesIm` (complex eigenvalues) |

```chuks
const A = np.from2d([[3.0, 1.0], [1.0, 2.0]])
const b = np.from1d([9.0, 8.0])
const x = np.solve(A, b)             // [2, 3]
A.close(); b.close(); x.close()
```

Result wrappers (`QrResult`, etc.) own all their parts — calling
`.close()` on the wrapper frees every contained NDArray.

---

## 11. Random sampling

`np.seed(s)` controls the PRNG globally per `NumPy` instance. Distributions
return a freshly allocated NDArray of the requested shape.

| Routine                        | Distribution             |
| ------------------------------ | ------------------------ |
| `np.uniform(low, high, shape)` | Uniform `[low, high)`    |
| `np.normal(mean, std, shape)`  | Gaussian                 |
| `np.binomial(n, p, shape)`     | Binomial(n, p)           |
| `np.poisson(lambda, shape)`    | Poisson                  |
| `np.gamma(k, scale, shape)`    | Gamma                    |
| `np.beta(alpha, b, shape)`     | Beta                     |
| `np.choice(arr, n, replace)`   | random sample from `arr` |
| `np.shuffle(arr)`              | in-place Fisher–Yates    |

```chuks
np.seed(42)
const u = np.normal(0.0, 1.0, [10_000])
println("mean ≈ " + string(u.mean()))    // ≈ 0
println("std  ≈ " + string(u.std()))     // ≈ 1
u.close()
```

---

## 12. FFT

| Routine                  | Returns                                                   |
| ------------------------ | --------------------------------------------------------- |
| `np.fft(a)`              | `FftResult` (`.re`, `.im`) — full complex spectrum        |
| `np.ifft(re, im)`        | `FftResult` (`.re`, `.im`) — inverse, complex out         |
| `np.rfft(a)`             | `FftResult` (`.re`, `.im`) — half spectrum for real input |
| `np.irfft(re, im, nOut)` | `NDArray` — real inverse of `rfft`, output length `nOut`  |

```chuks
const x  = np.from1d([1.0, 2.0, 3.0, 4.0])
const X  = np.fft(x)             // .re, .im
const xr = np.ifft(X.re, X.im)
println(xr.re.toFloats())        // ≈ [1, 2, 3, 4]
x.close(); X.close(); xr.close()
```

---

## 13. Persistence — `.npy` / `.npz`

| Call                    | Effect                                                 |
| ----------------------- | ------------------------------------------------------ |
| `np.saveNpy(arr, path)` | write one array, NumPy-compatible `.npy`               |
| `np.loadNpy(path)`      | read one `.npy` into a new NDArray                     |
| `np.npzWrite(path)`     | open writer → `.add(name, arr)` per entry → `.close()` |
| `np.npzRead(path)`      | open reader → `.names()`, `.get(name)`, `.close()`     |

```chuks
const a = np.arange(0.0, 10.0, 1.0)
np.saveNpy(a, "/tmp/a.npy")

const r = np.loadNpy("/tmp/a.npy")
println(r.toFloats())            // [0..9]
a.close(); r.close()
```

`.npz` is a zip-of-`.npy`s. Names are arbitrary strings; `.get(name)`
allocates a fresh NDArray per call.

---

## 14. Zero-copy Arrow interop

`np.exportArrowCDI(arr, schemaPtr, arrayPtr)` hands ownership of `arr`'s
buffer to a `chuks_arrow` `Float64Array` through the
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html).
**No copy.**

Constraints: input must be 1-D and C-contiguous with zero offset. For
higher-rank or sliced inputs, call `.reshape([size])` and/or `.copy()` first.

Full example: [examples.md §12](./examples.md#12-zero-copy-arrow-interop).

After export, the NDArray is consumed — calling `.close()` is a no-op, and
attempting to use the array further is undefined behaviour.

---

## 15. Memory model and `close()`

Each `NDArray` wraps a `Box<ArrayD<f64>>` on the Rust side. The Chuks GC
reclaims the wrapper but not the Rust allocation, so:

```chuks
const a = np.zeros([1_000_000])
// ... use a ...
a.close()      // mandatory; idempotent
```

**Rules of thumb**

- Operations returning a _new_ NDArray (`add`, `reshape`, `slice`,
  `matmul`, `mean`, …) hand you an owned array — close it.
- _In-place_ ops (`*InPlace`, `*InPlaceScalar`, `np.copyto`,
  `np.maskedAssign`) mutate the destination and return `this`. You still
  close the destination once at the end.
- Composite result wrappers (`QrResult`, `SvdResult`, `EigResult`,
  `FftResult`, `NpzReader`, `NpzWriter`) own all their fields. Calling
  `.close()` on the wrapper closes _every_ contained NDArray.

If you forget `close()`, the buffer leaks until the process exits. The
test harness has no automatic detection, so be disciplined in long-running
servers.

---

## 16. Performance notes

- **FFI calls are not free.** Every NDArray method that crosses into Rust
  pays a fixed call cost (typically 50–200 ns). For loops over many small
  arrays, fuse work into one larger array first.
- **In-place beats new.** Allocating and freeing million-element arrays
  inside a loop dominates. `addInPlace` / `mulInPlaceScalar` / `np.copyto`
  reuse storage.
- **Ternary kernels are fused.** `np.where(mask, x, y)`, `np.clip`,
  `np.maskedAssign` do the comparison + selection + write in one Rust
  pass. Building the same logic with `if`/`get`/`set` is orders of
  magnitude slower.
- **Reshape is free; transpose is a view.** Neither copies. The followup
  arithmetic op materializes a contiguous result.
- **Vector vs matrix dot.** `np.dot(v, w)` on 1-D inputs uses BLAS-style
  inner product. For matrix products use `np.matmul`.
- **`.toFloats()` copies.** Avoid in hot paths — prefer `np.exportArrowCDI`
  for zero-copy handoff to `chuks_arrow`.

---

## 17. Errors & edge cases

| Situation                                            | Behavior                          |
| ---------------------------------------------------- | --------------------------------- |
| `get(idxs)` wrong rank or out of range               | returns `0.0`, no throw           |
| `set(idxs, v)` wrong rank or out of range            | returns `false`, no throw         |
| `reshape(s)` when `product(s) != size()`             | throws `Error`                    |
| `from2d` with ragged rows                            | throws `Error("…ragged rows…")`   |
| `fromNd(s, v)` with `product(s) != v.length`         | throws `Error("…shape product…")` |
| Broadcasting incompatible shapes in `add`/`mul`/…    | Rust kernel throws                |
| `solve(a, b)` with singular `a`                      | Rust kernel throws                |
| `cholesky(a)` with non-PD `a`                        | Rust kernel throws                |
| `slice` step `<= 0`                                  | Rust kernel throws                |
| `close()` called twice                               | no-op                             |
| Using an array after `np.exportArrowCDI` consumed it | undefined; treat as freed         |

`println(err.message())` after `try`/`catch` gives the Rust-side message.

---

## 18. Migrating from Python NumPy

A pocket cheat-sheet. The table below maps the calls Python users reach for
first to their `chuks_numpy` equivalents.

| Python NumPy                        | chuks_numpy                                                              |
| ----------------------------------- | ------------------------------------------------------------------------ |
| `import numpy as np`                | `import { NumPy } from "chuks_numpy"; const np = new NumPy()`            |
| `np.array([1, 2, 3])`               | `np.from1d([1.0, 2.0, 3.0])`                                             |
| `np.array([[1, 2], [3, 4]])`        | `np.from2d([[1.0, 2.0], [3.0, 4.0]])`                                    |
| `np.array(... 3-D nested list ...)` | `np.fromNd([d0, d1, d2], flat)` _or_ `from1d(...).reshape([d0, d1, d2])` |
| `np.zeros((2, 3))`                  | `np.zeros([2, 3])`                                                       |
| `np.ones(4)`                        | `np.ones([4])`                                                           |
| `np.full((2, 2), 7.5)`              | `np.full([2, 2], 7.5)`                                                   |
| `np.eye(3)`                         | `np.eye(3)`                                                              |
| `np.arange(0, 10)`                  | `np.arange(0.0, 10.0, 1.0)` (always 3 args)                              |
| `np.linspace(0, 1, 11)`             | `np.linspace(0.0, 1.0, 11)`                                              |
| `a + b`                             | `a.add(b)`                                                               |
| `a * 2`                             | `a.mulScalar(2.0)`                                                       |
| `10 - a`                            | `a.rsubScalar(10.0)`                                                     |
| `a ** 2`                            | `a.powScalar(2.0)`                                                       |
| `a == b`                            | `a.eq(b)` (returns mask `NDArray`)                                       |
| `a < 0`                             | `a.ltScalar(0.0)`                                                        |
| `a += b`                            | `a.addInPlace(b)`                                                        |
| `a.sum()`                           | `a.sum()`                                                                |
| `a.sum(axis=1)`                     | `a.sumAxis(1, false)`                                                    |
| `a.sum(axis=1, keepdims=True)`      | `a.sumAxis(1, true)`                                                     |
| `a.mean()`                          | `a.mean()`                                                               |
| `np.sqrt(a)`                        | `a.sqrt()`                                                               |
| `np.where(mask, x, y)`              | `np.where(mask, x, y)`                                                   |
| `np.clip(a, 0, 1)`                  | `np.clip(a, 0.0, 1.0)`                                                   |
| `a[i, j]`                           | `a.get([i, j])`                                                          |
| `a[i, j] = v`                       | `a.set([i, j], v)`                                                       |
| `a[0:2, 1:4]`                       | `a.slice([0, 1], [2, 4], [1, 1])`                                        |
| `a[:, [0, 3]]`                      | `a.take(1, [0, 3])`                                                      |
| `a.T`                               | `a.transpose()`                                                          |
| `a.reshape(3, 4)`                   | `a.reshape([3, 4])`                                                      |
| `a.copy()`                          | `a.copy()`                                                               |
| `a.flatten()`                       | `a.toFloats()` _(returns `[]float`, not NDArray)_                        |
| `np.matmul(a, b)` or `a @ b`        | `np.matmul(a, b)`                                                        |
| `np.dot(v, w)` (1-D)                | `np.dot(v, w)`                                                           |
| `np.linalg.inv(a)`                  | `np.inv(a)`                                                              |
| `np.linalg.solve(a, b)`             | `np.solve(a, b)`                                                         |
| `np.linalg.det(a)`                  | `np.det(a)`                                                              |
| `np.linalg.norm(a)`                 | `np.norm(a, 2)`                                                          |
| `np.linalg.qr(a)`                   | `np.qr(a)` → `.q`, `.r`                                                  |
| `U, s, Vt = np.linalg.svd(a)`       | `const r = np.svd(a)` → `r.u`, `r.s`, `r.vt`                             |
| `np.linalg.eig(a)`                  | `np.eig(a)` → `.valuesRe`, `.valuesIm`                                   |
| `np.fft.fft(x)`                     | `np.fft(x)` → `.re`, `.im`                                               |
| `np.fft.ifft(X)`                    | `np.ifft(X.re, X.im)`                                                    |
| `np.fft.rfft(x)`                    | `np.rfft(x)`                                                             |
| `np.random.seed(42)`                | `np.seed(42)`                                                            |
| `np.random.uniform(0, 1, 1000)`     | `np.uniform(0.0, 1.0, [1000])`                                           |
| `np.random.normal(0, 1, (3, 3))`    | `np.normal(0.0, 1.0, [3, 3])`                                            |
| `np.save("a.npy", a)`               | `np.saveNpy(a, "a.npy")`                                                 |
| `np.load("a.npy")`                  | `np.loadNpy("a.npy")`                                                    |
| `np.savez("bundle.npz", a=a, b=b)`  | `const w = np.npzWrite(...); w.add(...); w.close()`                      |
| _(implicit GC)_                     | **explicit `a.close()` on every NDArray**                                |

### Five gotchas for Python users

1. **Everything is `float`.** `np.array([1, 2, 3])` becomes
   `np.from1d([1.0, 2.0, 3.0])` — integer literals would type-fail.
2. **No operator overloading.** `a + b` doesn't compile. Write `a.add(b)`.
3. **`println(a)` prints a handle, not the data.** Use `a.toString()` or
   `a.toFloats()`.
4. **You manage lifetimes.** Every `NDArray` you get needs `close()`.
   Wrappers (`QrResult`, …) close all their fields when you close them.
5. **Argument is a `[]int` shape vector, not a tuple.** `np.zeros((2, 3))`
   becomes `np.zeros([2, 3])`.

---

## See also

- [API Index](./api-index.md) — every public method, alphabetically.
- [NDArray API](./ndarray.md) — full method reference, organized by category.
- [NumPy API](./numpy.md) — factory + algorithms reference.
- [Examples & Recipes](./examples.md) — 13 runnable snippets.
