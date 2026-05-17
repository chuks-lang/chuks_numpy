# chuks_numpy

NumPy-style n-dimensional arrays for [Chuks](https://chuks.org), backed by a Rust cdylib shim around the [`ndarray`](https://crates.io/crates/ndarray) crate.

> **Status:** Phase NU0 (scaffold). Surface: load, `version()`, `zeros(shape)`, get/set, close. Phases NU1–NU5 in progress.

## Roadmap

| Phase | Scope                                                                                       |
| ----- | ------------------------------------------------------------------------------------------- |
| NU0   | Scaffold: shim build, NDArray<f64> handle, get/set/close, smoke test                        |
| NU1   | Construction (`ones`, `full`, `arange`, `linspace`, `from1d`/`from2d`), reshape, slicing    |
| NU2   | Elementwise + broadcasting kernels; reductions (sum/mean/std/max/min/argmax, axis-wise)     |
| NU3   | BLAS-backed linalg (matmul, dot, solve, inv, det, svd, eig, qr, cholesky), FFT              |
| NU4   | Ternary kernels via typed-FFI `PPP_I` fast path: `where`, `select`, `copyto`, masked-assign |
| NU5   | IO (`.npy`/`.npz`), random (PRNG), zero-copy interop with `chuks_arrow`                     |

## Install

```jsonc
// chuks.json
{
  "dependencies": {
    "chuks_numpy": "0.1.0",
  },
}
```

## Quick start

```chuks
import { NumPy } from "chuks_numpy"

const np = new NumPy("/abs/path/to/chuks_numpy/shim")
println(np.version())

const a = np.zeros([2, 3])
a.set([0, 1], 3.14)
println(string(a.get([0, 1])))   // 3.14

println("shape: " + string(a.shape()))   // [2, 3]
println("len:   " + string(a.len()))     // 6

a.close()
```

## Building the shim

The shim is built on first import via `r.cargoBuild()` and cached. To build manually:

```bash
cd chuks_numpy/shim
cargo build --release
```

## Tests

```bash
chuks run tests/index.test.chuks
```

## License

MIT
