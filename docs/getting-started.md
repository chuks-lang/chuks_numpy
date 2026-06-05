# Getting Started

## Install

Add to your project's `chuks.json`:

```jsonc
chuks add @chuks/numpy
```

## Your first program

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

// 1-D from a Chuks []float
const v = np.from1d([1.0, 2.0, 3.0, 4.0])
println("len = "  + string(v.size()))   // 4
println("sum = "  + string(v.sum()))   // 10
println("mean = " + string(v.mean()))  // 2.5

// 2-D from rows
const m = np.from2d([[1.0, 2.0], [3.0, 4.0]])
println("shape = " + string(m.shape())) // [2, 2]
println("trace = " + string(np.trace(m)))

v.close()
m.close()
```

Run it:

```bash
chuks run app.chuks       # VM mode
chuks build app.chuks     # AOT — native binary in build/
```

## Inspecting an array

`NDArray` is a thin Chuks class that holds an opaque pointer to a Rust-side
buffer — the numbers do **not** live in any Chuks field. So:

```chuks
const a = np.from1d([1.0, 2.0, 3.0])
println(a)              // Instance(NDArray)   ← not what you want
```

### `toString()` — NumPy-style pretty print _(recommended)_

Use `a.toString()` to get a familiar NumPy-style rendering for any rank:

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

> Coming from Python? In NumPy, `print(arr)` calls `arr.__repr__`. Chuks does
> not yet auto-dispatch to user-defined formatters, so you have to spell it
> `println(a.toString())` explicitly. The output is the same shape NumPy
> users expect.

### Lower-level inspection

| Call            | What it gives you                               |
| --------------- | ----------------------------------------------- |
| `a.toFloats()`  | `[]float` — flat row-major copy of all elements |
| `a.shape()`     | `[]int` — e.g. `[3, 2]`                         |
| `a.ndim()`      | number of dimensions                            |
| `a.size()`      | total element count                             |
| `a.get([i, j])` | single element                                  |

```chuks
println(a.toFloats())   // [1, 2, 3]      flat row-major copy
println(a.shape())      // [3]
println(a.ndim())       // 1
println(a.size())        // 3
println(a.get([0]))     // 1.0
```

## Constructing arrays

| Call                              | Result                              |
| --------------------------------- | ----------------------------------- |
| `np.zeros([2, 3])`                | 2×3 of zeros                        |
| `np.ones([4])`                    | length-4 vector of ones             |
| `np.full([2, 2], 7.5)`            | 2×2 filled with `7.5`               |
| `np.arange(0.0, 10.0, 1.0)`       | `[0, 1, …, 9]`                      |
| `np.linspace(0.0, 1.0, 11)`       | 11 evenly spaced points in `[0, 1]` |
| `np.eye(3)`                       | 3×3 identity                        |
| `np.from1d([1.0, 2.0, 3.0])`      | from a flat `[]float`               |
| `np.from2d([[1.0, 2.0], …])`      | from rows                           |
| `np.fromNd([2, 3], [1.0, … 6.0])` | from shape + flat row-major values  |

## Memory model — `close()` is mandatory

Every `NDArray` wraps a Rust `Box<ArrayD<f64>>`. The Chuks GC will free the
_handle wrapper_, but not the Rust allocation behind it. Always:

```chuks
const a = np.zeros([1_000_000])
// … use a …
a.close()
```

`close()` is **idempotent** — calling it twice is a no-op.

Operations that _return a new_ array allocate new memory; you own the result and
must close it. In-place ops (`addInPlace`, `mulInPlaceScalar`, …) reuse the
destination's storage and return `this` for chaining.

```chuks
const a = np.ones([3])
const b = np.full([3], 2.0)

a.addInPlace(b).mulInPlaceScalar(10.0)    // a is now [30, 30, 30]
println(string(a.toFloats()))

a.close()
b.close()
```

## Lifetime rules for composite results

A few methods return small wrapper objects that own multiple NDArrays.
Calling `.close()` on the wrapper closes _all_ of them:

```chuks
const qr = np.qr(a)
// use qr.q, qr.r
qr.close()       // closes both q and r

const svd = np.svd(a)
// use svd.u, svd.s, svd.vt
svd.close()
```

Same applies to `EigResult`, `FftResult`, `NpzReader`, `NpzWriter`.

## Where to next

- Full method-by-method reference: [NDArray API](./ndarray.md), [NumPy API](./numpy.md)
- Runnable recipes: [Examples](./examples.md)
