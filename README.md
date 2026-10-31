# TensorTrail

**Numerical drift detection and tensor comparison for machine-learning inference.**

TensorTrail compares a reference tensor with a candidate tensor and returns a
structured, reproducible diagnostic report: fused MAE/MSE/RMSE, relative error,
cosine similarity, tolerance outcomes, tensor statistics, non-finite counts, and
explicit conversion provenance. The default engine is a Rust extension built
with PyO3 and Rayon. A NumPy implementation is included as a readable reference
backend and parity oracle.

> **Status: 0.1.0 alpha.** The numerical core, Python API, reports, tests, and
> benchmarks in this repository are implemented and verified. GPU execution,
> activation capture, and explicit SIMD kernels are **not** implemented; see
> [Scope and non-goals](#scope-and-non-goals).

## Why TensorTrail

- **One traversal for all binary metrics.** MAE, MSE, RMSE, relative error,
  cosine terms, tolerance mismatches, and non-finite counts are accumulated
  together. Statistics can be folded into the same pass.
- **Deterministic parallel reductions.** Fixed chunk boundaries and ordered
  merges make results independent of Rayon worker count and scheduling for a
  fixed chunk size. Results do not silently depend on how many cores are busy.
- **Explicit numerical policy.** Empty tensors, NaN/±Inf, zero denominators,
  dtype conversion, non-contiguous layouts, and overflow are defined and
  recorded rather than hidden behind a scalar.
- **No silent backend fallback.** A missing native extension raises an
  actionable error. `backend="numpy"` is an explicit choice.
- **Reports you can keep.** Stable JSON schema and a self-contained HTML report
  with inputs, conversions, metrics, counts, tolerances, and execution metadata.

## Installation

TensorTrail requires Python 3.11+ (3.11–3.14 tested) and NumPy. Building the
native engine from source requires a Rust toolchain (1.85+) and a C/C++ build
environment supported by PyO3.

```bash
git clone https://github.com/leaderofARS/tensortrail.git
cd tensortrail
python -m pip install .
```

For development (editable install with the native extension):

```bash
python -m pip install "maturin>=1.16,<2.0"
maturin develop --release
```

Optional PyTorch input support:

```bash
python -m pip install "tensortrail[torch]"
```

No package is published to PyPI by this repository state. Build and install
from source, or use a wheel produced by CI.

## Quick start

```python
import numpy as np
import tensortrail as tt

reference = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
candidate = np.array([1.0, 4.0, 2.0, 8.0], dtype=np.float32)

report = tt.compare(reference, candidate)

print(report.summary())
print(report.metrics.mae)              # 1.75
print(report.metrics.relative_error)   # RelativeErrorValue(mean=..., maximum=...)
print(report.counts.mismatches)        # 3
print(report.status)                   # "fail"

report.save("layer12.json")
report.save("layer12.html")
```

```text
TensorTrail comparison: FAIL
  reference: 'reference' shape=(4,) dtype=float32 device=cpu
  candidate: 'candidate' shape=(4,) dtype=float32 device=cpu
  mae: 1.75
  mse: 5.25
  rmse: 2.2912878
  relative_error: mean=0.58333333 max=1
  cosine_similarity: 0.93073869
  max_abs_error: 4
  elements: 4 (finite pairs: 4)
  mismatches: 3 (3 finite, 0 non-finite pattern)
  tolerance: atol=1e-06 rtol=1e-05 equal_nan=True
  backend: rust threads=1 elapsed=0.002391s
```

### Configuration

```python
config = tt.ComparisonConfig(
    backend="rust",          # default; "numpy" is the explicit reference backend
    metrics=("mae", "rmse", "relative_error", "cosine_similarity", "statistics"),
    atol=1e-5,
    rtol=1e-4,
    equal_nan=True,
    relative_epsilon=1e-12,
    threads=8,               # None uses the global Rayon pool; 1 is sequential
    parallel_threshold=32_768,
    chunk_size=32_768,
    release_gil=True,
)

report = tt.compare(
    reference,
    candidate,
    config=config,
    reference_id="decoder.layer_12.output",
    candidate_id="decoder.layer_12.output.optimized",
)
```

Tolerance is evaluated per element:

```text
|reference - candidate| <= atol + rtol * |reference|
```

### Metrics

| Metric | Definition | Empty/all-non-finite policy |
| --- | --- | --- |
| `mae` | `mean(abs(r - c))` over finite pairs | `None` |
| `mse` | `mean((r - c)^2)` over finite pairs | `None` |
| `rmse` | `sqrt(mse)` | `None` |
| `relative_error` | `abs(r - c) / max(abs(r), epsilon)` | `None` |
| `cosine_similarity` | `dot(r, c) / sqrt(||r||^2 * ||c||^2)`, clamped to `[-1, 1]` | `None` |
| `statistics` | min, max, mean, population variance, std dev, finite/non-finite counts | numeric fields `None` |

Cosine policy: two all-zero vectors have similarity `1.0`; exactly one all-zero
vector has similarity `0.0`. Extreme magnitudes use a scaled two-pass fallback
so that `1e-300` and `1e308` inputs do not overflow or underflow the result.

### Policies at a glance

- **Empty tensors:** metrics are `None`, never a fabricated zero; status is
  `pass` when there are no mismatches.
- **NaN/±Inf:** excluded from metric sums and statistics, counted by category on
  each side. Non-finite pattern differences (for example `NaN` vs `1.0`, or
  `+Inf` vs `-Inf`) fail the comparison. Corresponding NaNs pass when
  `equal_nan=True`.
- **Shape mismatch:** rejected before execution with both shapes in the error.
- **dtypes:** float32 and float64 stay in their dtype when both sides match;
  float16 and bfloat16 are widened to float32; integers and bools become
  float64; mixed float32/float64 becomes float64. Complex, object, string,
  datetime, and extended-precision arrays are rejected. Every conversion is
  listed in `report.reference.conversion_notes`.
- **Layouts:** non-contiguous, negative-stride, unaligned, and non-native-endian
  arrays are copied to contiguous native buffers and the conversion is recorded.
- **Overflow:** accumulation is float64; a mathematically out-of-range result is
  `Infinity`, which fails tolerance checking. Underflowed squared differences can
  make MSE report `0.0`; this is a documented floating-point limit.
- **No silent fallback:** if `backend="rust"` and the extension is unavailable,
  `BackendError` is raised. TensorTrail never switches to NumPy implicitly.
- **No silent device copy:** CUDA/MPS/meta PyTorch tensors are rejected rather
  than transferred and presented as a GPU comparison.

Full details: [numerical methods](docs/numerical_methods.md).

## PyTorch input

PyTorch is optional and is never imported at package import time.

```python
import torch
import tensortrail as tt

reference = torch.randn(32, 768)             # CPU tensor
candidate = reference + torch.randn(32, 768) * 1e-5

report = tt.compare(reference, candidate)
assert report.reference.source == "torch"
```

CPU tensors are converted explicitly; `requires_grad` tensors are detached and
the note is recorded. Accelerator and quantized tensors are rejected. TensorTrail
does not claim GPU support and does not silently move data to the host.

## Reports

`report.metrics`, `report.statistics`, `report.counts`, `report.reference`,
`report.candidate`, `report.tolerances`, and `report.metadata` are typed,
immutable dataclasses. `report.save(path)` writes JSON or HTML based on the
extension; `format=` overrides it. `report.to_dict()` and `report.to_json()`
expose the same schema (`schema_version: "1.0"`). Strict JSON is emitted: any
non-finite float that survives overflow is encoded as `"NaN"`, `"Infinity"`, or
`"-Infinity"`.

## Performance

Measured on the development machine (Windows 11, Intel Core i7-12700H class,
14 physical/20 logical cores, Python 3.13.5, NumPy 2.1.3, release build). These
are end-to-end `tt.compare` medians over 7 repetitions; they are not universal
claims. Full methodology and raw results are in
[docs/benchmarking.md](docs/benchmarking.md).

| Elements | dtype | Rust backend | NumPy backend | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 100,000 | float64 | 0.75 ms | 5.58 ms | 7.4× |
| 1,000,000 | float64 | 1.83 ms | 80.3 ms | 43.9× |
| 5,000,000 | float64 | 20.9 ms | 409.7 ms | 19.6× |
| 1,000,000 | float32 | 1.69 ms | 91.5 ms | 54.3× |

For equivalent work, the fused native kernel measured about 1.4–1.5× faster than
separate kernel calls for the binary metric set, and about 1.7–2.0× faster with
statistics; it also scales with Rayon workers. Absolute parallel timings vary
under sustained load; see the benchmark document for the full results and
caveats. No hand-written SIMD intrinsics are present. The native build reports
`simd: "scalar"`; explicit SIMD remains deferred.

## Scope and non-goals

Implemented and verified:

- Rust numerical engine with deterministic Rayon reductions and a sequential
  path.
- NumPy reference backend and randomized cross-backend parity tests.
- Python API, typed reports, JSON/HTML serialization, and atomic saves.
- Optional CPU PyTorch tensors with explicit conversion and rejection of
  accelerator/quantized/complex layouts.
- Unit, parity, determinism, layout, reporting, import-isolation, integration,
  and optional PyTorch tests.

Not implemented (not claimed):

- GPU/CUDA/MPS-native metrics, multi-device comparison, or distributed runs.
- Activation capture/hooks, graph tracing, and model patching.
- Explicit AVX/NEON/SVE kernels or runtime CPU dispatch.
- Complex-valued tensors, quantized dequantization, sparse tensors, or masked
  arrays.

## Development

```bash
python -m pip install -e ".[dev]"
maturin develop --release

cargo test --manifest-path native/tensortrail-core/Cargo.toml --release --all-features
cargo clippy --manifest-path native/tensortrail-core/Cargo.toml --all-targets --all-features -- -D warnings
cargo fmt --manifest-path native/tensortrail-core/Cargo.toml --all -- --check

ruff check python tests
mypy
pytest
```

See [docs/development.md](docs/development.md) for platform notes, CI, and
release steps. Contribution guidelines are in [CONTRIBUTING.md](CONTRIBUTING.md),
and release history is in [CHANGELOG.md](CHANGELOG.md). The verified status of
this release is documented in the
[engineering report](docs/engineering_report.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
