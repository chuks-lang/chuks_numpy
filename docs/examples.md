# Examples & Recipes

Every snippet below is self-contained. Save as e.g. `app.chuks` and run
`chuks run app.chuks` (or `chuks build app.chuks` for a native binary).

## 1. Hello, arrays

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

const a = np.arange(0.0, 10.0, 1.0)
println("len  = " + string(a.size()))    // 10
println("sum  = " + string(a.sum()))    // 45
println("mean = " + string(a.mean()))   // 4.5
println(a.toString())                   // [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

a.close()
```

> Tip: `println(a)` on its own prints `Instance(NDArray)`. Always go through
> `a.toString()` (NumPy-style, any rank) or `a.toFloats()` (raw `[]float`).

## 2. Reshaping & slicing

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

const m = np.arange(0.0, 12.0, 1.0).reshape([3, 4])
println(m.shape())                        // [3, 4]

// rows [0, 2), cols [1, 4) — i.e. the upper-right 2×3 block
const block = m.slice([0, 1], [2, 4], [1, 1])
println(block.toFloats())                 // [1, 2, 3, 5, 6, 7]

// pick columns 0 and 3
const cols = m.take(1, [0, 3])
println(cols.shape())                     // [3, 2]

m.close(); block.close(); cols.close()
```

## 3. Broadcasting & chained scalar math

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

// Standardize: (x - mean) / std
const x   = np.uniform(0.0, 100.0, [10_000])
const mu  = x.mean()
const sd  = x.std()

const z = x.subScalar(mu).divScalar(sd)
println("z mean ≈ " + string(z.mean()))  // ≈ 0
println("z std  ≈ " + string(z.std()))   // ≈ 1

x.close(); z.close()
```

## 4. Axis reductions

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

const grades = np.from2d([
    [85.0, 90.0, 78.0],
    [88.0, 76.0, 92.0],
    [70.0, 85.0, 95.0],
])

// per-student average (axis 1 = across subjects)
const perStudent = grades.meanAxis(1, false)
println(perStudent.toFloats())            // [84.33, 85.33, 83.33]

// per-subject min (axis 0 = across students)
const perSubject = grades.minAxis(0, false)
println(perSubject.toFloats())            // [70, 76, 78]

grades.close(); perStudent.close(); perSubject.close()
```

## 5. In-place updates (hot loop)

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

const buf = np.zeros([1_000_000])
const inc = np.full([1_000_000], 0.001)

// 100 steps, zero new allocations
for (var i: int = 0; i < 100; i = i + 1) {
    buf.addInPlace(inc)
}
println("buf[0] = " + string(buf.get([0])))   // 0.1

buf.close(); inc.close()
```

## 6. Linear algebra — solve a system

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

// [[3, 1], [1, 2]] · x = [9, 8]
const A = np.from2d([[3.0, 1.0], [1.0, 2.0]])
const b = np.from1d([9.0, 8.0])
const x = np.solve(A, b)

println(x.toFloats())                    // [2, 3]
println("det(A) = " + string(np.det(A))) // 5

A.close(); b.close(); x.close()
```

## 7. SVD + low-rank reconstruction

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

const A = np.from2d([
    [1.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 3.0],
])

const svd = np.svd(A)
println("U  shape = " + string(svd.u.shape()))
println("s        = " + string(svd.s.toFloats()))
println("Vt shape = " + string(svd.vt.shape()))

svd.close(); A.close()
```

## 8. FFT round-trip

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

const x = np.from1d([1.0, 2.0, 3.0, 4.0])
const X = np.fft(x)
const xr = np.ifft(X.re, X.im)

println("original     : " + string(x.toFloats()))
println("ifft(fft(x)) : " + string(xr.re.toFloats()))

x.close(); X.close(); xr.close()
```

## 9. Masks + `where` / `clip` / `maskedAssign`

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

const a = np.from1d([-2.0, -1.0, 0.0, 1.0, 2.0])

// clamp to [-1, 1]
const clipped = np.clip(a, -1.0, 1.0)
println(clipped.toFloats())              // [-1, -1, 0, 1, 1]

// where a < 0 → 0, else a
const zeros = np.zeros([5])
const relu  = np.where(a.geScalar(0.0), a, zeros)
println(relu.toFloats())                 // [0, 0, 0, 1, 2]

// in-place: set negatives to 0
const mask = a.ltScalar(0.0)
np.maskedAssign(a, mask, 0.0)
println(a.toFloats())                    // [0, 0, 0, 1, 2]

a.close(); clipped.close(); zeros.close(); relu.close(); mask.close()
```

## 10. Random sampling & summary stats

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()
np.seed(42)

const samples = np.normal(0.0, 1.0, [1_000_000])
println("mean = " + string(samples.mean()))   // ≈ 0
println("std  = " + string(samples.std()))    // ≈ 1
println("min  = " + string(samples.min()))
println("max  = " + string(samples.max()))

samples.close()
```

## 11. Persist with `.npz`

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()

const a = np.arange(0.0, 10.0, 1.0)
const b = np.linspace(0.0, 1.0, 5)

// save
const w = np.npzWrite("/tmp/bundle.npz")
w.add("a", a)
w.add("b", b)
w.close()

// load
const r = np.npzRead("/tmp/bundle.npz")
println(r.names())                        // ["a", "b"]
const aLoaded = r.get("a")
println(aLoaded.toFloats())
r.close()

a.close(); b.close(); aLoaded.close()
```

## 12. Zero-copy Arrow interop

Hands an `NDArray`'s buffer to a `chuks_arrow.Float64Array` through the
[Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html).
No copy.

```chuks
import { NumPy } from "pkg/@chuks/numpy"
import { Arrow } from "pkg/@chuks/arrow"
import { ArrowSchema, ArrowArray } from "std/chuksArrow"

const np = new NumPy()
const ar = new Arrow()

const nd = np.from1d([10.0, 20.0, 30.0, 40.0, 50.0])

// 72 B + 80 B C structs.
const sch = ArrowSchema.alloc()
const arr = ArrowArray.alloc()

// Hand ownership of nd's buffer to Arrow.
np.exportArrowCDI(nd, sch.ptr(), arr.ptr())

// Materialize on the chuks_arrow side — zero-copy.
const imported = ar.importArray(sch, arr)
sch.free(); arr.free()

println("len   = " + string(imported.len()))        // 5
println("[2]   = " + string(imported.getFloat(2)))  // 30

imported.close()    // releases the buffer
nd.close()          // no-op (already consumed)
```

Constraints: input must be 1-D and C-contiguous. For higher-rank or sliced
inputs, call `.reshape([len])` and/or `.copy()` first.

## 13. End-to-end: PCA-style centering + SVD

```chuks
import { NumPy } from "pkg/@chuks/numpy"

const np = new NumPy()
np.seed(7)

// 1000 samples × 4 features
const X = np.normal(0.0, 1.0, [1000, 4])

// subtract per-column mean
const colMeans = X.meanAxis(0, true)     // shape [1, 4]
const Xc = X.sub(colMeans)

// SVD
const svd = np.svd(Xc)
println("top-4 singular values: " + string(svd.s.toFloats()))

X.close(); colMeans.close(); Xc.close(); svd.close()
```

---

More end-to-end demos live in
[`forge-007/todoApp/demo/`](../../../forge-007/todoApp/demo/) — see
`01_numpy_basics.chuks`, `04_arrow_zero_copy.chuks`, and the benchmark
runner [`bench_all.sh`](../../../forge-007/todoApp/demo/scripts/bench_all.sh).
