# TensorTrail 0.1.0 Engineering Report

**Date:** 2026-10-31  
**Status:** alpha, source release candidate  
**Scope:** numerical drift detection and tensor comparison for ML inference  
**Distribution:** source, wheel, and sdist built locally; not published

This report separates what was executed and observed from what is planned,
configured, or deliberately out of scope. It describes the repository state at
the date above.

## 1. Summary

TensorTrail 0.1.0 is a Python package with a Rust numerical engine. It compares a
reference tensor and a candidate tensor, then returns a typed report containing
fused MAE/MSE/RMSE, relative error, cosine similarity, tolerance outcomes,
per-tensor statistics, non-finite counts, input provenance, and execution
metadata. The native engine is optional at import time and loaded on first use;
the NumPy backend is a readable reference implementation and parity oracle.

Verified in this workspace:

- the Rust core compiles in release mode and passes 57 unit tests;
- the Python suite passes 208 tests with torch installed (2 platform/hardware
  skips), and 193 tests with 17 skips in NumPy-only clean environments;
- Ruff, strict mypy, `cargo fmt --check`, and Clippy with `-D warnings` pass;
- wheel and sdist build successfully, both install into clean environments, and
  the installed package performs a native comparison;
- the package builds and its tests pass on CPython 3.11, 3.12, 3.13, and 3.14;
- the NumPy 1.24.4 dependency floor was built against and tested on Python 3.11;
- measured end-to-end and kernel benchmarks are recorded with environment and
  methodology.

Not verified in this workspace:

- execution of the GitHub Actions workflow on GitHub's runners;
- any operating system other than Windows 11;
- GPU, distributed, or accelerator execution (not implemented);
- sustained performance across machines or thermal states;
- publication to PyPI or any other package index (not performed).

## 2. Implemented capability

| Area | Implementation | Status |
| --- | --- | --- |
| Metrics | Fused MAE, MSE, RMSE, relative mean/max, cosine, max absolute error, tolerance mismatches, non-finite counts | Verified |
| Statistics | Welford min/max/mean/population variance/std dev, optional fused pass | Verified |
| Native engine | Rust crate with PyO3/numpy bindings and Rayon reductions | Verified |
| Reference engine | NumPy implementation of the same policy | Verified |
| Parallelism | Fixed chunks, ordered merges, cached Rayon pools, sequential fallback | Verified |
| Configuration | Backend, metrics, tolerances, epsilon, threads, threshold, chunk size, GIL release | Verified |
| Inputs | NumPy arrays, numeric array-like objects, optional CPU PyTorch tensors | Verified |
| Normalization | Float16, integers, bools, mixed dtypes, layouts, byte order; bfloat16 via CPU torch | Verified |
| Rejections | Complex, object, string, datetime, masked, extended precision, accelerator/quantized tensors | Verified |
| Reports | Immutable typed objects, summary text, strict JSON, self-contained HTML, atomic writes | Verified |
| Packaging | Maturin wheel/sdist, `py.typed`, Apache-2.0 license, SBOM in wheel | Verified |
| CI | Rust, lint/type, OS/Python matrix, NumPy floor, packaging jobs | Configured, not executed remotely |

## 3. Architecture

The Python layer owns configuration, provenance, input normalization, backend
selection, report assembly, and serialization. The Rust layer owns shape and
option validation, chunk partitioning, numerical kernels, and parallel
execution. A backend returns a backend-neutral `BackendResult`; the engine
exposes only requested metrics and constructs an immutable `ComparisonReport`.

The native extension is `tensortrail._core`. It is imported lazily. A missing or
version-mismatched extension raises `BackendError`; TensorTrail never falls back
to NumPy without an explicit `backend="numpy"` configuration. CUDA, MPS, meta,
quantized, sparse, and complex PyTorch tensors are rejected rather than copied to
the host or dequantized silently.

Details: [architecture.md](architecture.md).

## 4. Numerical policy

- Working dtypes are float32 or float64. Accumulation and reported scalars are
  float64.
- Float16 and bfloat16 widen to float32. Integers and bools widen to float64.
  Mixed float32/float64 inputs widen to float64. The default suite verifies
  bfloat16 through CPU torch tensors; NumPy has no built-in bfloat16 dtype, and a
  third-party dtype supplying that name is supported by the same path but is not
  part of the default test matrix.
- Non-contiguous, negative-stride, unaligned, and non-native-endian arrays are
  copied to contiguous native buffers, with each action recorded in
  `conversion_notes`.
- Non-finite elements are excluded from metric sums and statistics and counted by
  category. Non-finite pattern differences fail the comparison; corresponding
  NaNs match when `equal_nan=True`.
- Empty or entirely non-finite inputs produce `None` metrics, never fabricated
  zeros.
- Relative error uses `|r - c| / max(|r|, epsilon)` with `epsilon=1e-12` by
  default.
- Cosine handles two zero vectors as `1.0`, one zero vector as `0.0`, and uses a
  scaled two-pass fallback when norms overflow or underflow.
- Variance is population variance (`n`), computed with Welford updates and
  Chan-style chunk merges.
- Tolerance is `|r - c| <= atol + rtol * |r|` with defaults `atol=1e-6` and
  `rtol=1e-5`.

Determinism: fixed chunk boundaries and ordered partial merges make parallel
results identical across worker counts for a fixed chunk size. Sequential and
parallel paths can differ in the last bits because their reduction trees differ.
Changing `chunk_size` can also change rounding. This is documented rather than
presented as bit-identical reproducibility.

Details and justified parity tolerances: [numerical_methods.md](numerical_methods.md).

## 5. Verification evidence

All commands below were run from the repository root on Windows 11 unless noted.

### Rust

```text
cargo fmt --manifest-path native/tensortrail-core/Cargo.toml --all -- --check
  -> exit 0

cargo clippy --manifest-path native/tensortrail-core/Cargo.toml \
  --all-targets --all-features -- -D warnings
  -> Finished; no warnings

cargo test --manifest-path native/tensortrail-core/Cargo.toml \
  --release --all-features
  -> 57 passed; 0 failed; 0 ignored
```

The 57 tests cover hand-calculated values, empty inputs, NaN/Inf handling,
overflow, zero vectors, epsilon floors, Welford statistics, fused-versus-standalone
agreement, parallel thread-count invariance, chunk tails, cached pools, and shape
validation.

### Python

```text
ruff check python tests benchmarks
  -> All checks passed

mypy
  -> Success: no issues found in 17 source files
```

Primary environment (CPython 3.13.5, NumPy 2.1.3, torch 2.9.1+cpu):

```text
pytest -q -p no:anchorpy
  -> 208 passed, 2 skipped
```

Skips:

- extended-precision `longdouble` rejection test: on this Windows build,
  `longdouble` is the same as `float64`, so the test has no distinct dtype to
  exercise;
- CUDA rejection test: CUDA is not available.

Clean NumPy-only environments (no torch):

| Python | NumPy | Result |
| --- | --- | --- |
| 3.11.15 | 1.24.4 | 193 passed, 17 skipped |
| 3.12.13 | 2.5.4 | 193 passed, 17 skipped |
| 3.13.13 | 2.5.4 | 193 passed, 17 skipped |
| 3.14.3 | 2.5.4 | 193 passed, 17 skipped |

The extra skips are the optional PyTorch tests. The Python 3.11 run used the
oldest supported NumPy and was built against that NumPy version, validating the
declared `numpy>=1.24` floor on that interpreter.

The test suite covers hand-calculated metrics, NumPy/Rust parity on randomized
data, parallel determinism across 2/3/4/8 workers, layouts and dtypes, rejection
paths, configuration validation, JSON/HTML escaping and non-finite sentinels,
atomic saves, import isolation, missing-extension behavior, version mismatch,
optional CPU torch conversion, and an end-to-end workflow.

The `-p no:anchorpy` flag was used only in the Anaconda environment because an
unrelated globally installed plugin fails to import there; it is not required by
the project's CI configuration.

### Packaging

```text
maturin build --release --sdist --out dist
  -> dist/tensortrail-0.1.0.tar.gz
  -> dist/tensortrail-0.1.0-cp313-cp313-win_amd64.whl
```

Wheel contents verified:

- `tensortrail/` Python package and `py.typed`;
- `tensortrail/_core.cp313-win_amd64.pyd`;
- `LICENSE` under `dist-info/licenses/`;
- `METADATA` with `License-Expression: Apache-2.0`, project URLs, requires-python,
  extras, and the NumPy dependency;
- a CycloneDX SBOM of the Rust dependency tree.

Sdist contents verified:

- Rust crate, `Cargo.toml`, `Cargo.lock`, and workspace manifest;
- Python package and `py.typed`;
- tests, documentation, benchmark scripts/results, contributing/changelog/conduct
  files;
- `README.md`, `LICENSE`, and `pyproject.toml`.

Clean installation:

- the cp313 wheel installed into a standalone CPython 3.13.13 virtual
  environment; `tt.compare` returned MAE `1.75`, status `fail`, and the native
  engine reported version `0.1.0`, release profile, 20 Rayon threads, and
  `simd: "scalar"`;
- the sdist was built and installed with pip into a fresh CPython 3.13.13
  environment; the full test suite then passed;
- wheels were also built and tested for CPython 3.11, 3.12, and 3.14.

Maturin's Windows audit reported a transitive `zlib.dll` dependency originating
from Anaconda's `python313.dll`, not from the extension itself. The extension's
import table references `python313.dll` plus Windows system DLLs. The standalone
CPython 3.13.13 install ran successfully, confirming that bundling Anaconda's
zlib would be incorrect.

### Benchmarks

Environment: Windows 11, i7-12700H class (14 physical/20 logical cores), 15.68 GiB,
CPython 3.13.5, NumPy 2.1.3, release build, Rayon 20 threads.

End-to-end `tt.compare` medians (7 repetitions):

| Elements | dtype | Rust | NumPy | Rust speedup |
| ---: | --- | ---: | ---: | ---: |
| 100,000 | float64 | 0.751 ms | 5.58 ms | 7.4× |
| 1,000,000 | float64 | 1.828 ms | 80.32 ms | 43.9× |
| 5,000,000 | float64 | 20.90 ms | 409.7 ms | 19.6× |
| 1,000,000 | float32 | 1.685 ms | 91.46 ms | 54.3× |

Kernel-level fusion, equivalent work:

| Size | Work | Separate | Fused | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 100,000 | binary metrics | 1.564 ms | 1.040 ms | 1.50× |
| 100,000 | binary + statistics | 2.305 ms | 1.174 ms | 1.96× |
| 1,000,000 | binary metrics | 3.671 ms | 2.676 ms | 1.37× |
| 1,000,000 | binary + statistics | 5.463 ms | 3.243 ms | 1.68× |

Absolute parallel timings varied under sustained load (the same 1M fused
configuration measured 1.86–4.02 ms across runs), while within-group ratios were
stable at roughly 1.4–1.5× for binary metrics and 1.7–2.0× with statistics.
These are observations from one machine, not universal claims. The NumPy backend
is a reference implementation, not a tuned competitor, and it upcasts float32
working arrays to float64.

Methodology and caveats: [benchmarking.md](benchmarking.md). Raw data:
`benchmarks/results/python-windows.json`,
`benchmarks/results/criterion-windows.txt`, and
`benchmarks/results/criterion-fusion-windows.txt`.

### CI

`.github/workflows/ci.yml` defines:

- Rust formatting, Clippy, and release tests;
- Ruff and strict mypy;
- a Linux/macOS/Windows matrix for Python 3.11–3.14 that builds a wheel and runs
  the suite against it;
- a NumPy 1.24.4 floor job on Python 3.11;
- wheel/sdist build, `twine check`, sdist content inspection, clean wheel
  install, smoke test, and artifact upload.

The commands were executed locally as described above, but the workflow has not
run on GitHub in this workspace. No publication step exists.

## 6. Release readiness

Satisfied for the scoped 0.1.0 alpha:

- all requested numerical features are implemented and tested;
- the default backend is the Rust engine, with explicit failure when unavailable;
- NumPy and Rust backends agree within documented tolerances;
- reports are typed, serializable, and include conversion/execution provenance;
- packaging is complete for the tested platforms;
- benchmark evidence and limitations are recorded.

Outstanding before a public release:

- first hosted CI run and fixes for any platform-specific findings;
- a repository commit/tag and release notes review;
- a decision on supported wheel platforms and an authorized publication process;
- optional longer benchmark runs on additional hardware.

No package index publication has been performed. Doing so requires explicit
authorization.

## 7. Deliberately not implemented

- GPU/CUDA/MPS-native metrics, multi-device comparison, or distributed execution.
- Activation capture, framework hooks, graph tracing, or model patching.
- Explicit AVX/NEON/SVE intrinsics or runtime CPU dispatch. The native build
  reports `simd: "scalar"`; no auto-vectorization claim is made.
- Complex-valued, quantized, sparse, or masked-array semantics.
- Cross-backend bit-identical results. Different reduction layouts can differ in
  the last bits, as documented.
- Automatic host transfers for accelerator tensors.

These are scope decisions, not hidden fallbacks. Each rejection path is explicit
and covered by tests where applicable.

## 8. Reproducing the verification

Prerequisites: Python 3.11+, Rust 1.85+ (tested 1.95.0), Maturin 1.16.0, MSVC on
Windows. From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install "maturin>=1.16,<2.0" "numpy>=1.24" pytest mypy ruff
maturin develop --release

cargo fmt --manifest-path native/tensortrail-core/Cargo.toml --all -- --check
cargo clippy --manifest-path native/tensortrail-core/Cargo.toml \
  --all-targets --all-features -- -D warnings
cargo test --manifest-path native/tensortrail-core/Cargo.toml --release --all-features

ruff check python tests benchmarks
mypy
pytest -q

maturin build --release --sdist --out dist
```

See [development.md](development.md) for platform notes and packaging details.

## 9. Conclusion

TensorTrail 0.1.0 delivers the requested numerical comparison engine and Python
API without claiming GPU, capture, or SIMD capabilities it does not have. The
numerical policy is explicit, the backends are cross-checked, the native failure
path is enforced, and the package installs from both wheel and sdist. The main
remaining risks are the untested hosted-CI environments and the normal
variability of parallel benchmark timings; both are stated rather than hidden.
