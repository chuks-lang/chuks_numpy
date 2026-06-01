# NumPy API

`NumPy` is the factory class. One instance is enough per process; reuse it
to construct arrays and call algorithms.

```chuks
import { NumPy } from "chuks_numpy"
const np = new NumPy()
```

The constructor takes an optional shim-manifest directory:

```chuks
new NumPy(shimManifestDir: string = "chuks_packages/chuks_numpy")
```

You only need to override it when calling from outside the standard
`chuks_packages/` layout (e.g. tests in the package repo itself).

> Underscore-prefixed methods on `NumPy` (`_arrayLen`, `_binop`, …) are
> internal FFI plumbing. They are public so the `NDArray` class can reach
> them across the package boundary, but they are **not** stable API.
> Skip them unless you are extending the package itself.

## Meta

| Method      | Returns  | Notes                                     |
| ----------- | -------- | ----------------------------------------- |
| `version()` | `string` | `chuks_numpy_shim 1.0.0; ndarray 0.16; …` |

## Construction

| Method                                           | Returns   |
| ------------------------------------------------ | --------- |
| `zeros(shape: []int)`                            | `NDArray` |
| `ones(shape: []int)`                             | `NDArray` |
| `full(shape: []int, value: float)`               | `NDArray` |
| `arange(start: float, stop: float, step: float)` | `NDArray` |
| `linspace(start: float, stop: float, n: int)`    | `NDArray` |
| `eye(n: int)`                                    | `NDArray` |
| `from1d(values: []float)`                        | `NDArray` |
| `from2d(rows: [][]float)`                        | `NDArray` |
| `fromNd(shape: []int, values: []float)`          | `NDArray` |

## Linear algebra

| Method              | Returns     | Notes                                  |
| ------------------- | ----------- | -------------------------------------- |
| `matmul(a, b)`      | `NDArray`   | matrix × matrix (or batched mat-mul)   |
| `dot(a, b)`         | `float`     | inner product on 1-D vectors           |
| `inv(a)`            | `NDArray`   | matrix inverse                         |
| `solve(a, b)`       | `NDArray`   | solve `a·x = b`                        |
| `det(a)`            | `float`     |                                        |
| `trace(a)`          | `float`     |                                        |
| `norm(a, ord: int)` | `float`     | `ord=2` for L2, `ord=1` for L1, etc.   |
| `cholesky(a)`       | `NDArray`   | lower-triangular factor                |
| `pinv(a)`           | `NDArray`   | Moore–Penrose pseudoinverse            |
| `lstsq(a, b)`       | `NDArray`   | least-squares solution                 |
| `qr(a)`             | `QrResult`  | `.q`, `.r`; `close()` releases both    |
| `svd(a)`            | `SvdResult` | `.u`, `.s`, `.vt`                      |
| `eig(a)`            | `EigResult` | `.valuesRe`, `.valuesIm` (eigenvalues) |

```chuks
const a = np.from2d([[4.0, 7.0], [2.0, 6.0]])
println("det = " + string(np.det(a)))      // 10

const inv = np.inv(a)
const id = np.matmul(a, inv)               // ~ identity
a.close(); inv.close(); id.close()

const svd = np.svd(np.from2d([[1.0, 0.0], [0.0, 2.0]]))
println("singular values: " + string(svd.s.toFloats()))
svd.close()
```

## FFT

| Method                     | Returns     | Notes                                                  |
| -------------------------- | ----------- | ------------------------------------------------------ |
| `fft(a)`                   | `FftResult` | Real-input forward FFT; returns complex (`.re`, `.im`) |
| `ifft(re, im)`             | `FftResult` | Inverse of `fft`                                       |
| `rfft(a)`                  | `FftResult` | Real-input forward, half-spectrum                      |
| `irfft(re, im, nOut: int)` | `NDArray`   | Real inverse, output length `nOut`                     |

```chuks
const x = np.from1d([1.0, 2.0, 3.0, 4.0])
const X = np.fft(x)
println(X.re.toFloats())
println(X.im.toFloats())
const xr = np.ifft(X.re, X.im)
X.close(); xr.close(); x.close()
```

## Ternary / masked kernels

Fast typed-FFI fused kernels — avoid Chuks-side loops.

| Method                               | Behaviour                                            |
| ------------------------------------ | ---------------------------------------------------- |
| `where(mask, x, y)`                  | `mask ? x : y` (returns new)                         |
| `copyto(dst, src, mask)`             | in-place: `dst[i] = src[i]` where mask               |
| `maskedAssign(dst, mask, value)`     | in-place: `dst[i] = value` where mask                |
| `clip(arr, lo, hi)`                  | clamp to `[lo, hi]`                                  |
| `select(conds, choices, defaultArr)` | first matching condition; falls back to `defaultArr` |

```chuks
const a = np.arange(0.0, 6.0, 1.0).reshape([2, 3])
const m = a.gtScalar(3.0)
const out = np.where(m, a, np.zeros([2, 3]))
// out is a with values <= 3 zeroed
a.close(); m.close(); out.close()
```

## Random (PRNG)

`seed(s)` is global per `NumPy` instance — calling it makes subsequent random
calls reproducible.

| Method                                 | Returns   |
| -------------------------------------- | --------- |
| `seed(s: int)`                         | `void`    |
| `uniform(low, high, shape)`            | `NDArray` |
| `normal(mean, std, shape)`             | `NDArray` |
| `binomial(n: int, p: float, shape)`    | `NDArray` |
| `poisson(lambda: float, shape)`        | `NDArray` |
| `gamma(k: float, scale: float, shape)` | `NDArray` |
| `beta(alpha: float, b: float, shape)`  | `NDArray` |
| `choice(arr, n: int, replace: bool)`   | `NDArray` |
| `shuffle(arr)`                         | `NDArray` |

```chuks
np.seed(42)
const u = np.uniform(0.0, 1.0, [1_000_000])
println("mean ≈ " + string(u.mean()))
u.close()
```

## File I/O — `.npy` / `.npz`

| Method                       | Returns     | Notes                                  |
| ---------------------------- | ----------- | -------------------------------------- |
| `saveNpy(arr, path: string)` | `int`       | returns 0 on success                   |
| `loadNpy(path: string)`      | `NDArray`   |                                        |
| `npzWrite(path: string)`     | `NpzWriter` | call `.add(name, arr)` then `.close()` |
| `npzRead(path: string)`      | `NpzReader` | `.names()`, `.get(name)`, `.close()`   |

### `NpzWriter`

| Method                            | Returns | Notes               |
| --------------------------------- | ------- | ------------------- |
| `add(name: string, arr: NDArray)` | `void`  |                     |
| `close()`                         | `void`  | flushes the archive |

### `NpzReader`

| Method              | Returns    |
| ------------------- | ---------- |
| `count()`           | `int`      |
| `names()`           | `[]string` |
| `get(name: string)` | `NDArray`  |
| `close()`           | `void`     |

```chuks
const a = np.arange(0.0, 10.0, 1.0)
const b = np.linspace(0.0, 1.0, 5)

const w = np.npzWrite("/tmp/bundle.npz")
w.add("a", a)
w.add("b", b)
w.close()

const r = np.npzRead("/tmp/bundle.npz")
println(r.names())                 // ["a", "b"]
const aLoaded = r.get("a")
r.close()

a.close(); b.close(); aLoaded.close()
```

## Arrow zero-copy

| Method                                                          | Notes                                       |
| --------------------------------------------------------------- | ------------------------------------------- |
| `exportArrowCDI(arr: NDArray, schemaPtr: CPtr, arrayPtr: CPtr)` | Hands ownership of `arr`'s buffer to Arrow. |

See [examples.md](./examples.md#zero-copy-arrow-interop) for the full handshake
with `chuks_arrow`.

Constraints: input must be 1-D and C-contiguous with zero offset. For other
inputs, call `.copy()` or `.reshape([len])` first.

## Result wrapper classes

These small classes simply hold the parts of a multi-output algorithm; their
`close()` closes every contained `NDArray`.

| Class       | Fields                 |
| ----------- | ---------------------- |
| `QrResult`  | `q`, `r`               |
| `SvdResult` | `u`, `s`, `vt`         |
| `EigResult` | `valuesRe`, `valuesIm` |
| `FftResult` | `re`, `im`             |
