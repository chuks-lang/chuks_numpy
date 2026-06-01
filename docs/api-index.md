# chuks_numpy — API Index

Alphabetical index of every public method on `NumPy` and `NDArray`. For
detailed signatures, semantics, and examples, follow the link to the
appropriate reference page.

- [`NumPy` factory](./numpy.md) — algorithms, construction, I/O.
- [`NDArray`](./ndarray.md) — per-array methods.
- [User Guide](./user-guide.md) — concepts and migration.

> **Convention.** Names ending in `*Scalar` take a `float` instead of an
> `NDArray`. Names ending in `*InPlace` mutate the receiver and return
> `this`. Names ending in `Axis` take `(axis: int, keepdims: bool)`.

---

## `NumPy` factory

| Name             | One-liner                                                   |
| ---------------- | ----------------------------------------------------------- |
| `arange`         | Half-open range `[start, stop)` with step                   |
| `beta`           | Random Beta(α, b) samples                                   |
| `binomial`       | Random Binomial(n, p) samples                               |
| `choice`         | Random sample from an existing array                        |
| `cholesky`       | Lower-triangular Cholesky factor `L`                        |
| `clip`           | Clamp into `[lo, hi]`                                       |
| `copyto`         | In-place `dst[i] = src[i]` where mask                       |
| `det`            | Determinant                                                 |
| `dot`            | Inner product of 1-D vectors                                |
| `eig`            | Eigenvalues — returns `EigResult`                           |
| `exportArrowCDI` | Zero-copy hand-off to Arrow C Data Interface                |
| `eye`            | `n×n` identity matrix                                       |
| `fft`            | Complex forward FFT — returns `FftResult`                   |
| `from1d`         | NDArray from a `[]float`                                    |
| `from2d`         | NDArray from a `[][]float`                                  |
| `fromNd`         | NDArray from shape vector + flat row-major `[]float`        |
| `full`           | NDArray of given shape filled with one value                |
| `gamma`          | Random Gamma(k, scale) samples                              |
| `ifft`           | Inverse FFT of `(re, im)`                                   |
| `inv`            | Matrix inverse                                              |
| `irfft`          | Real inverse of `rfft`, output length `nOut`                |
| `linspace`       | `n` evenly spaced points in `[start, stop]`                 |
| `loadNpy`        | Read a NumPy `.npy` file into a new NDArray                 |
| `lstsq`          | Least-squares solution `argmin ‖A·x − b‖`                   |
| `maskedAssign`   | In-place `dst[i] = value` where mask                        |
| `matmul`         | Matrix product (also batched on higher ranks)               |
| `norm`           | Vector / matrix norm (`ord = 1`, `2`, `-1`, …)              |
| `normal`         | Random Gaussian samples                                     |
| `npzRead`        | Open an `.npz` reader                                       |
| `npzWrite`       | Open an `.npz` writer                                       |
| `ones`           | NDArray of given shape filled with `1.0`                    |
| `pinv`           | Moore–Penrose pseudoinverse                                 |
| `poisson`        | Random Poisson(λ) samples                                   |
| `qr`             | QR decomposition — returns `QrResult`                       |
| `rfft`           | Real-input forward FFT, half spectrum — returns `FftResult` |
| `saveNpy`        | Write a NumPy `.npy` file                                   |
| `seed`           | Seed the per-`NumPy` PRNG                                   |
| `select`         | First matching condition; fall back to default array        |
| `shuffle`        | Fisher–Yates shuffle (in-place)                             |
| `solve`          | Solve `A·x = b`                                             |
| `svd`            | Singular value decomposition — returns `SvdResult`          |
| `trace`          | Sum of diagonal                                             |
| `uniform`        | Random Uniform(low, high) samples                           |
| `version`        | Version string of the shim + ndarray + nalgebra             |
| `where`          | `mask ? x : y`, allocates a new NDArray                     |
| `zeros`          | NDArray of given shape filled with `0.0`                    |

### Result wrappers

| Class       | Fields                 | `close()` behaviour    |
| ----------- | ---------------------- | ---------------------- |
| `EigResult` | `valuesRe`, `valuesIm` | closes both            |
| `FftResult` | `re`, `im`             | closes both            |
| `NpzReader` | _(opaque)_             | flushes/closes archive |
| `NpzWriter` | _(opaque)_             | flushes/closes archive |
| `QrResult`  | `q`, `r`               | closes both            |
| `SvdResult` | `u`, `s`, `vt`         | closes all three       |

---

## `NDArray`

### Inspection / lifecycle

| Name       | One-liner                                                |
| ---------- | -------------------------------------------------------- |
| `close`    | Release the Rust buffer (idempotent)                     |
| `copy`     | Deep clone into a new contiguous NDArray                 |
| `dim`      | Size of one axis, `-1` if out of range                   |
| `get`      | Single element at `idxs` (returns `0.0` if out of range) |
| `len`      | Total element count                                      |
| `rank`     | Number of dimensions                                     |
| `set`      | Set single element; returns `bool` success               |
| `shape`    | Per-axis size vector                                     |
| `toFloats` | Flat row-major `[]float` copy                            |
| `toString` | NumPy-style pretty print, any rank                       |

### Shape transforms

| Name        | One-liner                                                   |
| ----------- | ----------------------------------------------------------- |
| `reshape`   | New view with same total length                             |
| `slice`     | Per-axis `[start, end)` with positive step (new contiguous) |
| `take`      | Fancy-index along one axis                                  |
| `transpose` | Reverse all axes                                            |

### Elementwise binary

Each accepts another NDArray with broadcast-compatible shape.

| Name      | Operation       |
| --------- | --------------- |
| `add`     | `a + b`         |
| `div`     | `a / b`         |
| `maximum` | elementwise max |
| `minimum` | elementwise min |
| `modulo`  | `a mod b`       |
| `mul`     | `a * b`         |
| `pow`     | `a ^ b`         |
| `sub`     | `a - b`         |

### Elementwise scalar variants

Each takes a single `float` `v` (no allocation of an intermediate array).
`r…Scalar(v)` computes `v OP a` (reversed operand order).

| Name            | Operation   |
| --------------- | ----------- |
| `addScalar`     | `a + v`     |
| `divScalar`     | `a / v`     |
| `maximumScalar` | `max(a, v)` |
| `minimumScalar` | `min(a, v)` |
| `mulScalar`     | `a * v`     |
| `powScalar`     | `a ^ v`     |
| `rdivScalar`    | `v / a`     |
| `rmodScalar`    | `v mod a`   |
| `rpowScalar`    | `v ^ a`     |
| `rsubScalar`    | `v - a`     |
| `subScalar`     | `a - v`     |

### In-place arithmetic

All return `this` (chainable) and require broadcast-compatible shapes.

| Name               | Effect   |
| ------------------ | -------- |
| `addInPlace`       | `a += b` |
| `addInPlaceScalar` | `a += v` |
| `divInPlace`       | `a /= b` |
| `divInPlaceScalar` | `a /= v` |
| `mulInPlace`       | `a *= b` |
| `mulInPlaceScalar` | `a *= v` |
| `subInPlace`       | `a -= b` |
| `subInPlaceScalar` | `a -= v` |

### Unary math

All return a new NDArray of the same shape.

| Name    | Operation      |
| ------- | -------------- | --- | --- |
| `abs`   | `              | a   | `   |
| `ceil`  | `⌈a⌉`          |
| `cos`   | `cos a`        |
| `exp`   | `eᵃ`           |
| `floor` | `⌊a⌋`          |
| `log`   | `ln a`         |
| `neg`   | `-a`           |
| `round` | banker's round |
| `sin`   | `sin a`        |
| `sqrt`  | `√a`           |
| `tan`   | `tan a`        |

### Comparison (returns mask NDArray of 0.0 / 1.0)

| Name       | Operation |
| ---------- | --------- |
| `eq`       | `a == b`  |
| `eqScalar` | `a == v`  |
| `ge`       | `a >= b`  |
| `geScalar` | `a >= v`  |
| `gt`       | `a > b`   |
| `gtScalar` | `a > v`   |
| `le`       | `a <= b`  |
| `leScalar` | `a <= v`  |
| `lt`       | `a < b`   |
| `ltScalar` | `a < v`   |
| `ne`       | `a != b`  |
| `neScalar` | `a != v`  |

### Whole-array reductions (scalar out)

| Name       | Returns                                        |
| ---------- | ---------------------------------------------- |
| `all`      | `1.0` if every element is non-zero, else `0.0` |
| `any`      | `1.0` if any element is non-zero, else `0.0`   |
| `argmax`   | flat index of max                              |
| `argmin`   | flat index of min                              |
| `max`      | maximum element                                |
| `mean`     | arithmetic mean                                |
| `min`      | minimum element                                |
| `prod`     | product of all elements                        |
| `std`      | population standard deviation                  |
| `sum`      | sum of all elements                            |
| `variance` | population variance                            |

### Axis-wise reductions (NDArray out)

Signature: `name(axis: int, keepdims: bool)`.

| Name           | Operation along axis |
| -------------- | -------------------- |
| `argmaxAxis`   | index of max         |
| `argminAxis`   | index of min         |
| `maxAxis`      | max                  |
| `meanAxis`     | mean                 |
| `minAxis`      | min                  |
| `prodAxis`     | product              |
| `stdAxis`      | population std       |
| `sumAxis`      | sum                  |
| `varianceAxis` | population variance  |

---

## See also

- [User Guide](./user-guide.md) — concepts, broadcasting, indexing, NumPy migration.
- [NDArray API](./ndarray.md) — full method tables organized by category.
- [NumPy API](./numpy.md) — full factory tables with signatures.
- [Examples & Recipes](./examples.md) — 13 runnable snippets.
