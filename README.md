# chuks_numpy

NumPy-style n-dimensional arrays for [Chuks](https://chuks.org), backed by a Rust cdylib shim around the [`ndarray`](https://crates.io/crates/ndarray), [`nalgebra`](https://crates.io/crates/nalgebra), and [`rustfft`](https://crates.io/crates/rustfft) crates.

> **Status:** v1.0.0 — Phases NU0–NU5b shipped (263/263 tests pass VM + AOT, including a 9-test cross-package round-trip with `@chuks/arrow`).

## Roadmap

| Phase | Scope                                                                                                         | Status |
| ----- | ------------------------------------------------------------------------------------------------------------- | ------ |
| NU0   | Scaffold: shim build, NDArray<f64> handle, get/set/close, smoke test                                          | ✅     |
| NU1   | Construction (`ones`, `full`, `arange`, `linspace`, `from1d`/`from2d`/`fromNd`), reshape, transpose, slice    | ✅     |
| NU2   | Elementwise + broadcasting + unary + comparisons + reductions (axis-wise + keepdims) + in-place               | ✅     |
| NU3   | Linear algebra (matmul, dot, inv, solve, det, trace, norm, cholesky, qr, svd, pinv, lstsq, eig) + FFT         | ✅     |
| NU4   | Ternary / masked kernels via typed-FFI `PPP_I` fast path: `where`, `copyto`, `maskedAssign`, `clip`, `select` | ✅     |
| NU5a  | PRNG (`uniform`, `normal`, `binomial`, `poisson`, `gamma`, `beta`, `choice`, `shuffle`) + `.npy`/`.npz` IO    | ✅     |
| NU5b  | Zero-copy `NDArray → chuks_arrow Float64Array` via Arrow C Data Interface (`np.exportArrowCDI`)               | ✅     |

## Install

```jsonc
 chuks add @chuks/numpy
```

## Quick start

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

// Construction
const a = np.from2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
const b = np.from2d([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

// Elementwise + reductions
const c = a.mul(2.0)               // scalar broadcast
println("sum: " + string(a.sumAll()))

// Linear algebra (NU3)
const m = np.matmul(a, b)
println("det: " + string(np.det(np.from2d([[4.0, 7.0], [2.0, 6.0]]))))

const qr = np.qr(a)
const svd = np.svd(a)
println("singular values: " + string(svd.s.toFloats()))

// FFT (NU3)
const x = np.from1d([1.0, 2.0, 3.0, 4.0])
const X = np.fft(x)
const xr = np.ifft(X.re, X.im)

// Ternary / masked (NU4)
const mask = a.gtScalar(3.0)         // where a > 3
const clipped = np.clip(a, 0.0, 5.0) // clamp to [0, 5]
np.maskedAssign(clipped, mask, -1.0) // in-place: where mask, set to -1

// Random (NU5)
np.seed(42)
const u = np.uniform(0.0, 1.0, [1000])
const g = np.normal(0.0, 1.0, [1000])
println("normal mean: " + string(g.mean()))

// .npy / .npz IO (NU5)
np.saveNpy(a, "/tmp/a.npy")
const a2 = np.loadNpy("/tmp/a.npy")

const w = np.npzWrite("/tmp/bundle.npz")
w.add("a", a); w.add("b", b); w.close()
const r = np.npzRead("/tmp/bundle.npz")
const loadedA = r.get("a")
r.close()

a.close(); b.close(); c.close(); m.close()
qr.close(); svd.close(); X.close(); xr.close(); x.close()
mask.close(); clipped.close(); u.close(); g.close()
a2.close(); loadedA.close()
```

## Zero-copy Arrow interop (NU5b)

Export a 1-D `NDArray<f64>` to a `chuks_arrow.Float64Array` through the Arrow C Data Interface — no buffer copy. The NDArray's underlying `Vec<f64>` is handed to Arrow; closing the resulting Arrow array invokes the release callback that frees it.

```chuks
import { NumPy } from "pkg/@chuks/numpy"
import { Arrow } from "pkg/@chuks/arrow"
import { ArrowSchema, ArrowArray } from "std/chuksArrow"

const np = new NumPy()
const ar = new Arrow()

const nd = np.from1d([10.0, 20.0, 30.0, 40.0, 50.0])

// Allocate the 72 B / 80 B CDI structs.
const sch = ArrowSchema.alloc()
const arr = ArrowArray.alloc()

// Hand ownership of nd's buffer to Arrow (CONSUMES nd).
np.exportArrowCDI(nd, sch.ptr(), arr.ptr())

// Materialize on the chuks_arrow side — zero-copy.
const imported = ar.importArray(sch, arr)
sch.free(); arr.free()

println("len: " + string(imported.len()))         // 5
println("[0]: " + string(imported.getFloat(0)))   // 10

imported.close()  // releases the buffer
nd.close()        // no-op (already consumed)
```

Constraints: input must be 1-D and C-contiguous with zero offset. For higher-rank or sliced inputs, call `.reshape([len])` and/or `.copy()` first to normalize.

## Tests

```bash
chuks run tests/index.test.chuks
```

## Documentation

Full docs live in [`docs/`](./docs/README.md):

- [Getting Started](./docs/getting-started.md) — install, first program, lifetimes.
- [User Guide](./docs/user-guide.md) — broadcasting, indexing, masks, memory, performance, NumPy migration cheatsheet.
- [NDArray API](./docs/ndarray.md) — every method on `NDArray`.
- [NumPy API](./docs/numpy.md) — every method on the `NumPy` factory.
- [API Index](./docs/api-index.md) — alphabetical index with one-liners.
- [Examples & Recipes](./docs/examples.md) — 13 runnable end-to-end snippets.

## License

MIT
