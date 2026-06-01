# NDArray API

`NDArray` is the f64 n-dimensional array. Construct one with a `NumPy`
factory (see [NumPy API](./numpy.md) for constructors).

> Every `NDArray` you receive owns Rust-side memory. Call `close()` exactly
> once. `close()` is idempotent.

## Shape & metadata

| Method           | Returns  | Notes                                    |
| ---------------- | -------- | ---------------------------------------- |
| `size()`          | `int`    | Total elements (product of shape).       |
| `ndim()`         | `int`    | Number of dimensions.                    |
| `dim(axis: int)` | `int`    | Size of `axis`. Returns `-1` if invalid. |
| `shape()`        | `[]int`  | Full shape as a Chuks array.             |
| `toString()`     | `string` | NumPy-style pretty-print (any rank).     |

```chuks
const a = np.zeros([2, 3, 4])
println(a.ndim())          // 3
println(a.size())           // 24
println(a.shape())         // [2, 3, 4]
a.close()
```

### Pretty-printing with `toString()`

`println(a)` on its own prints `Instance(NDArray)` — Chuks's default object
formatter has no idea what's inside the Rust buffer. Use `a.toString()` to
get a NumPy-style rendering for any rank:

```chuks
const a = np.from1d([1.0, 2.0, 3.3])
println(a.toString())
// [1, 2, 3.3]

const b = np.from2d([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
println(b.toString())
// [[7, 8],
//  [9, 10],
//  [11, 12]]

const c = np.arange(0.0, 24.0, 1.0).reshape([2, 3, 4])
println(c.toString())
// [[[0, 1, 2, 3],
//   [4, 5, 6, 7],
//   [8, 9, 10, 11]],
//  [[12, 13, 14, 15],
//   [16, 17, 18, 19],
//   [20, 21, 22, 23]]]
```

For raw data instead of a string, use `toFloats()` (flat row-major) plus
`shape()`.

## Element access

| Method                       | Returns   | Notes                                                     |
| ---------------------------- | --------- | --------------------------------------------------------- |
| `get(idxs: []int)`           | `float`   | `idxs.length` must equal `ndim()`. Returns `0.0` on miss. |
| `set(idxs: []int, v: float)` | `bool`    | `true` on success, `false` if out-of-range/wrong-rank.    |
| `toFloats()`                 | `[]float` | Copy to row-major flat slice (length = `size()`).          |

```chuks
const m = np.zeros([2, 2])
m.set([0, 0], 1.0); m.set([1, 1], 1.0)
println(m.get([0, 0]))     // 1.0
println(m.toFloats())      // [1, 0, 0, 1]
m.close()
```

## Shape transforms (return new array)

| Method                                            | Notes                                                                                    |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `reshape(shape: []int)`                           | Throws if product mismatches `size()`.                                                    |
| `transpose()`                                     | Reverses all axes (`.T` for 2-D).                                                        |
| `slice(starts: []int, ends: []int, steps: []int)` | Per-axis half-open `[start, end)` slice; `steps > 0`. Negatives normalized Python-style. |
| `take(axis: int, idxs: []int)`                    | Fancy-index along `axis`.                                                                |
| `copy()`                                          | Deep independent copy.                                                                   |

```chuks
const v = np.arange(0.0, 12.0, 1.0)      // [0..11]
const m = v.reshape([3, 4])
const t = m.transpose()                   // 4×3
const s = m.slice([0, 1], [3, 3], [1, 1]) // first 3 rows, columns [1, 3)
const c = m.take(1, [0, 3])               // columns 0 and 3

v.close(); m.close(); t.close(); s.close(); c.close()
```

## Elementwise arithmetic (binary, broadcasting)

All return a _new_ NDArray. Shapes broadcast NumPy-style.

| Method           | Operation       |
| ---------------- | --------------- |
| `add(other)`     | `a + b`         |
| `sub(other)`     | `a - b`         |
| `mul(other)`     | `a * b`         |
| `div(other)`     | `a / b`         |
| `pow(other)`     | `a ** b`        |
| `modulo(other)`  | `a % b`         |
| `minimum(other)` | elementwise min |
| `maximum(other)` | elementwise max |

## Elementwise arithmetic (scalar)

`xScalar(v)` computes `a OP v`. The `r…Scalar` variants compute `v OP a`
(reversed) — useful when scalar division/subtraction order matters.

| Method          | Operation   |
| --------------- | ----------- |
| `addScalar(v)`  | `a + v`     |
| `subScalar(v)`  | `a - v`     |
| `mulScalar(v)`  | `a * v`     |
| `divScalar(v)`  | `a / v`     |
| `powScalar(v)`  | `a ** v`    |
| `modScalar(v)`  | `a % v`     |
| `minScalar(v)`  | `min(a, v)` |
| `maxScalar(v)`  | `max(a, v)` |
| `rsubScalar(v)` | `v - a`     |
| `rdivScalar(v)` | `v / a`     |
| `rpowScalar(v)` | `v ** a`    |
| `rmodScalar(v)` | `v % a`     |

```chuks
const a = np.from1d([1.0, 2.0, 3.0])
const b = a.mulScalar(10.0).addScalar(1.0)  // [11, 21, 31]
b.close(); a.close()
```

## Unary functions

| Method    | Notes             |
| --------- | ----------------- |
| `sqrt()`  |                   |
| `exp()`   |                   |
| `log()`   | natural log       |
| `sin()`   |                   |
| `cos()`   |                   |
| `tan()`   |                   |
| `abs()`   |                   |
| `floor()` |                   |
| `ceil()`  |                   |
| `round()` | banker's rounding |
| `neg()`   | unary `-`         |

## Comparisons → mask arrays (`0.0`/`1.0`)

| Method                      | Operation |
| --------------------------- | --------- |
| `eq(other)` / `eqScalar(v)` | `a == b`  |
| `ne(other)` / `neScalar(v)` | `a != b`  |
| `lt(other)` / `ltScalar(v)` | `a < b`   |
| `le(other)` / `leScalar(v)` | `a <= b`  |
| `gt(other)` / `gtScalar(v)` | `a > b`   |
| `ge(other)` / `geScalar(v)` | `a >= b`  |

Masks are themselves `NDArray<f64>` (1.0 = true, 0.0 = false). Combine with
`np.where`, `np.maskedAssign`, `np.copyto`, or `np.select`.

## Whole-array reductions

| Method       | Returns | Notes                       |
| ------------ | ------- | --------------------------- |
| `sum()`      | `float` |                             |
| `mean()`     | `float` |                             |
| `min()`      | `float` |                             |
| `max()`      | `float` |                             |
| `prod()`     | `float` |                             |
| `std()`      | `float` | population std              |
| `variance()` | `float` |                             |
| `any()`      | `bool`  | true if any non-zero        |
| `all()`      | `bool`  | true if all non-zero        |
| `argmin()`   | `int`   | flat index of first minimum |
| `argmax()`   | `int`   | flat index of first maximum |

## Axis-wise reductions (return new NDArray)

Each takes `(axis: int, keepdims: bool)`. With `keepdims = true` the reduced
axis is preserved with size 1 (NumPy-compatible).

| Method                       | Notes            |
| ---------------------------- | ---------------- |
| `sumAxis(axis, keepdims)`    |                  |
| `meanAxis(axis, keepdims)`   |                  |
| `minAxis(axis, keepdims)`    |                  |
| `maxAxis(axis, keepdims)`    |                  |
| `prodAxis(axis, keepdims)`   |                  |
| `stdAxis(axis, keepdims)`    |                  |
| `varAxis(axis, keepdims)`    |                  |
| `anyAxis(axis, keepdims)`    |                  |
| `allAxis(axis, keepdims)`    |                  |
| `argminAxis(axis, keepdims)` | indices as `f64` |
| `argmaxAxis(axis, keepdims)` | indices as `f64` |

```chuks
const m = np.from2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
const rowSum = m.sumAxis(1, false)       // shape [2]: [6, 15]
const colMean = m.meanAxis(0, true)      // shape [1, 3]: [[2.5, 3.5, 4.5]]
m.close(); rowSum.close(); colMean.close()
```

## In-place updates (return `this`)

The destination is mutated; the source/scalar is not. Useful inside hot loops
where allocation matters. All return `this`, so you can chain.

| Method                | Op       |
| --------------------- | -------- |
| `addInPlace(other)`   | `a += b` |
| `subInPlace(other)`   | `a -= b` |
| `mulInPlace(other)`   | `a *= b` |
| `divInPlace(other)`   | `a /= b` |
| `addInPlaceScalar(v)` | `a += v` |
| `subInPlaceScalar(v)` | `a -= v` |
| `mulInPlaceScalar(v)` | `a *= v` |
| `divInPlaceScalar(v)` | `a /= v` |

## Lifecycle

| Method       | Notes                                           |
| ------------ | ----------------------------------------------- |
| `toString()` | NumPy-style pretty-print (see above). Any rank. |
| `close()`    | Free the Rust allocation. Idempotent.           |
