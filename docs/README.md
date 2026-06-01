# chuks_numpy — Documentation

NumPy-style n-dimensional arrays for [Chuks](https://chuks.org), backed by a
Rust cdylib shim around the `ndarray`, `nalgebra`, and `rustfft` crates.

## Contents

| Doc                                     | What's inside                                                                                                  |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [Getting Started](./getting-started.md) | Install, import, first program, lifetime rules (`close()`).                                                    |
| [User Guide](./user-guide.md)           | Concepts: shapes, indexing, broadcasting, masks, lifetimes, performance, and a NumPy → chuks_numpy cheatsheet. |
| [NDArray API](./ndarray.md)             | Every method on `NDArray` — shape, indexing, arithmetic, reductions, axis-wise reductions, in-place updates.   |
| [NumPy API](./numpy.md)                 | Every method on the `NumPy` factory — construction, linear algebra, FFT, random, `.npy`/`.npz` I/O, Arrow.     |
| [API Index](./api-index.md)             | Alphabetical index of every public method, with one-line summaries.                                            |
| [Examples & Recipes](./examples.md)     | End-to-end runnable snippets (basics, linalg, FFT, RNG, persistence, Arrow zero-copy).                         |

## At a glance

```chuks
import { NumPy } from "chuks_numpy"

const np = new NumPy()

const a = np.from2d([[1.0, 2.0], [3.0, 4.0]])
const b = np.from2d([[5.0, 6.0], [7.0, 8.0]])

const c = np.matmul(a, b)            // linear algebra
println("c[0,0] = " + string(c.get([0, 0])))

const d = a.mulScalar(2.0).addScalar(1.0)   // chained elementwise
println("mean = " + string(d.mean()))

a.close(); b.close(); c.close(); d.close()
```

## Conventions used in this doc

- `NDArray` always holds `f64` (Chuks `float`). Integer-tensor types are not yet exposed.
- All shape-taking APIs use `[]int`, e.g. `np.zeros([2, 3])`.
- `int` indexing follows Python rules: 0-based, last axis fastest in row-major (C) order.
- Every NDArray you receive must be `close()`d exactly once. `close()` is idempotent.
- Operations that return a _new_ NDArray do **not** mutate inputs. In-place variants
  end in `InPlace`/`InPlaceScalar` and return `this` for chaining.
- To print an array, use `println(a.toString())` — `println(a)` alone prints
  `Instance(NDArray)` because the data lives in Rust, not in Chuks fields.

## Version

This documentation tracks `chuks_numpy` **v1.0.1** (phases NU0–NU5b).
See the top-level [`README.md`](../README.md) for roadmap/status.
