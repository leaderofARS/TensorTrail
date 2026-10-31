# Benchmarking

TensorTrail reports two kinds of measurements:

1. **End-to-end Python benchmarks** time the public `tt.compare` call, including
   input normalization, backend dispatch, the native binding, and report
   construction.
2. **Criterion kernel benchmarks** time the Rust numerical kernels directly,
   without Python, NumPy conversion, or report objects.

The two suites answer different questions. A fast kernel can still be hidden by
Python overhead for small inputs; a fast end-to-end call does not prove that the
reduction itself scales. Results below are reported separately for that reason.

## Environment

| Item | Value |
| --- | --- |
| Machine | Windows 11 (10.0.26200) |
| CPU | Intel Core i7-12700H class, 14 physical / 20 logical cores |
| Memory | 15.68 GiB |
| Python | 3.13.5 (CPython, MSVC) |
| NumPy | 2.1.3 |
| TensorTrail | 0.1.0 |
| Native build | release, Rayon pool with 20 threads |
| Rust | 1.95.0 stable |

These numbers describe one machine. They are not a promise of the same speedup
on another CPU, operating system, Python version, or data distribution. Run the
suites locally before making decisions that depend on performance.

## End-to-end Python benchmark

### Methodology

Script: `benchmarks/benchmark_python.py`

```bash
python benchmarks/benchmark_python.py \
  --json benchmarks/results/python-windows.json
```

Defaults:

- sizes: 1,000; 100,000; 1,000,000; 5,000,000 elements;
- dtypes: float32 and float64;
- backends: `rust` and `numpy`;
- 2 warm-up calls, then 7 timed repetitions; the median is reported;
- both inputs are one-dimensional, C-contiguous arrays generated from a
  fixed-seed normal distribution, with the candidate perturbed by Gaussian noise
  scaled by `1e-4`;
- `metrics` is the default set (MAE, MSE, RMSE, relative error, cosine
  similarity), so both backends perform equivalent comparison work;
- `threads=None`, `chunk_size=min(32_768, size)`, and
  `parallel_threshold=32_768`; inputs below the threshold run sequentially;
- wall-clock timing uses `time.perf_counter()` around `tt.compare` only. Report
  serialization is not included.

The NumPy backend is a readable reference implementation, not a tuned C kernel.
It also promotes working arrays to float64 even when both inputs are float32,
whereas the native backend keeps float32 input buffers and widens elements during
accumulation. The comparison is therefore an API-level comparison between the
two backends as shipped, not an isolated study of Rust versus NumPy intrinsics.

### Results

Median wall-clock time for the full default metric set:

| Elements | dtype | Rust | NumPy | Rust speedup |
| ---: | --- | ---: | ---: | ---: |
| 1,000 | float32 | 0.083 ms | 0.211 ms | 2.5× |
| 1,000 | float64 | 0.079 ms | 0.193 ms | 2.4× |
| 100,000 | float32 | 0.801 ms | 7.05 ms | 8.8× |
| 100,000 | float64 | 0.751 ms | 5.58 ms | 7.4× |
| 1,000,000 | float32 | 1.685 ms | 91.46 ms | 54.3× |
| 1,000,000 | float64 | 1.828 ms | 80.32 ms | 43.9× |
| 5,000,000 | float32 | 16.35 ms | 436.7 ms | 26.7× |
| 5,000,000 | float64 | 20.90 ms | 409.7 ms | 19.6× |

Raw samples, means, minima, maxima, throughput, and MAE values are in
`benchmarks/results/python-windows.json`.

### Interpretation

- At 1,000 elements, fixed Python and report overhead dominates. The Rust call is
  about 2.4–2.5× faster, not tens of times faster.
- Throughput improves through 1,000,000 elements as the native parallel path
  engages and per-call overhead is amortized.
- The 5,000,000-element case is memory-bandwidth bound. Its speedup is lower than
  the 1,000,000-element case because both backends spend more time streaming
  arrays and less time on arithmetic.
- The NumPy reference performs several full-array temporaries (absolute
  difference, square, relative denominator, tolerance array, masks), while the
  native fused kernel traverses the inputs once. That difference is a property of
  these implementations, not evidence that NumPy is intrinsically slow.
- `metrics=("mae",)` is also recorded in the raw JSON. It still executes the fused
  native kernel; subsetting changes report exposure, not the kernel traversal.
  Differences between the `all` and `mae` rows are small and within run-to-run
  noise at these sizes.

## Criterion kernel benchmarks

### Methodology

Benchmark: `native/tensortrail-core/benches/metrics.rs`

```bash
cargo bench --manifest-path native/tensortrail-core/Cargo.toml --bench metrics -- \
  --warm-up-time 0.3 --measurement-time 1.0 --sample-size 10
```

The benchmark uses deterministic float32 data (`sin` reference plus a small
`cos` perturbation), Criterion's default outlier handling, and the median
estimate. The parallel configuration enables parallelism at every size
(`threshold = 1`) with `chunk_size = 32_768` and the global Rayon pool, so these
numbers isolate kernel scaling rather than the public API's small-input
threshold. The sequential configuration disables the parallel path entirely.

### Individual kernels at 1,000,000 elements

| Kernel | Sequential | Parallel | Parallel speedup |
| --- | ---: | ---: | ---: |
| MAE | 2.000 ms | 0.446 ms | 4.5× |
| MSE | — | 0.600 ms | — |
| RMSE | — | 0.585 ms | — |
| Relative error | — | 0.717 ms | — |
| Cosine similarity | — | 0.747 ms | — |
| Statistics (one tensor) | — | 0.956 ms | — |

MAE reaches about 2.24 billion elements/second on the parallel path. MSE, RMSE,
relative error, and cosine each allocate their own pass and therefore pay
separate memory traversals. Statistics uses Welford accumulation and is the most
expensive individual kernel per element.

### Fused kernel

| Size | Configuration | Time | Throughput |
| ---: | --- | ---: | ---: |
| 1,000 | sequential, all + statistics | 12.16 µs | 82.2 Melem/s |
| 1,000 | parallel, all + statistics | 12.16 µs | 82.2 Melem/s |
| 1,000 | parallel, binary metrics only | 7.36 µs | 135.8 Melem/s |
| 100,000 | sequential, all + statistics | 1.214 ms | 82.3 Melem/s |
| 100,000 | parallel, all + statistics | 0.602 ms | 166.1 Melem/s |
| 100,000 | parallel, binary metrics only | 0.350 ms | 285.4 Melem/s |
| 1,000,000 | sequential, all + statistics | 13.33 ms | 75.0 Melem/s |
| 1,000,000 | parallel, all + statistics | 2.577 ms | 388.1 Melem/s |
| 1,000,000 | parallel, binary metrics only | 1.861 ms | 537.3 Melem/s |

The fused kernel computes MAE, MSE, RMSE, relative error, cosine, maximum absolute
error, tolerance mismatches, and non-finite counts in one traversal. With
statistics enabled it also folds Welford updates for both tensors into the same
traversal. At 1,000 elements, thread-pool overhead cancels the benefit of
parallelism; the public API consequently keeps small inputs on the sequential
path.

### Fusion versus equivalent separate calls

A fair fusion comparison must count the same metrics. The `passes` benchmark
therefore compares the fused kernel against separate calls that compute the same
set: MAE, MSE, relative error, and cosine (RMSE is derived from MSE). The
statistics variant additionally computes per-tensor statistics for both sides.

| Size | Work | Separate calls | Fused | Fusion ratio |
| ---: | --- | ---: | ---: | ---: |
| 100,000 | binary metrics | 1.564 ms | 1.040 ms | 1.50× |
| 100,000 | binary + statistics | 2.305 ms | 1.174 ms | 1.96× |
| 1,000,000 | binary metrics | 3.671 ms | 2.676 ms | 1.37× |
| 1,000,000 | binary + statistics | 5.463 ms | 3.243 ms | 1.68× |

The fused side performs a little more work than the separate side: it also
computes maximum absolute error, tolerance mismatch counts, and non-finite
counts. The ratios are therefore conservative. A repeat run with 20 samples and a
longer measurement window produced the same 1.4–1.5× binary-metric ratio and
1.7–2.0× statistics ratio, although its absolute parallel times were higher.

### Measurement variance

Absolute parallel timings on this machine drifted substantially under sustained
load: the same 1,000,000-element fused binary configuration measured 1.86 ms in
one run and 4.02 ms in another. Ratios measured back-to-back within one group
were much more stable than absolute times. Treat the absolute numbers as
observations from the stated run, not as a hardware-independent baseline, and
avoid comparing numbers from different runs when the difference is small.

Raw Criterion output is in `benchmarks/results/criterion-windows.txt`
(individual kernels and the first fusion comparison) and
`benchmarks/results/criterion-fusion-windows.txt` (repeated fusion run).

## What the benchmarks do and do not show

Verified:

- the native backend is faster than the NumPy reference for this API and data on
  the recorded machine;
- the fused kernel scales from one thread to the Rayon pool for large inputs;
- small inputs are dominated by fixed overhead, which is why a parallel threshold
  exists;
- statistics and additional metrics have measurable costs; fusion avoids
  repeated traversals but does not make work free.

Not established:

- speedups on other CPUs, GPUs, operating systems, Python builds, or NumPy
  versions;
- performance for sparse, irregular, or non-contiguous inputs (they are copied
  before timing in the public API);
- bit-identical results between the sequential and parallel paths;
- the effect of explicit SIMD, which is not implemented. The native build reports
  `simd: "scalar"`, and no auto-vectorization claim is made;
- end-to-end latency in a serving system, including serialization, network, and
  framework overhead.

## Reproducing and recording results

1. Build the release extension: `maturin develop --release`.
2. Confirm `tt.native_info()` reports `build_profile: "release"` and the expected
   Rayon thread count.
3. Close other CPU-heavy applications; record the machine and versions.
4. Run both suites and keep the raw JSON/text output.
5. Report medians and the environment with every claim. Do not compare a release
   build against a debug build, or a warm cache against a cold one, without
   saying so.

## Related documents

- [Numerical methods](numerical_methods.md)
- [Architecture](architecture.md)
- [Development](development.md)
- [Engineering report](engineering_report.md)
