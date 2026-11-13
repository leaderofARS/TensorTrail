# TensorTrail 0.2.0 Engineering Report

**Date:** 2026-11-13  
**Status:** alpha, released to PyPI and GitHub  
**Scope:** execution-aware inference validation on top of the 0.1.0 numerical comparison engine  
**Distribution:** PyPI `0.2.0` (CPython 3.13 manylinux x86_64 wheel and sdist); GitHub release `v0.2.0`

This report separates what was executed and observed from what is planned,
configured, or deliberately out of scope. It describes the repository state at
the date above. Version `0.2.0` was tagged `v0.2.0`, published as a GitHub
release, and uploaded to PyPI on 2026-11-13 by the release-triggered OIDC
workflow.

## 1. Summary

TensorTrail 0.2.0 adds an execution-level validation layer above the existing
tensor comparison API. An `ExecutionSession` records the provenance of one
inference run and a set of checkpoint metadata/payload pairs. Two completed
sessions can be aligned deterministically and validated checkpoint by
checkpoint. Every matched pair is still evaluated by `tt.compare()`; the Rust
numerical engine and the 0.1.0 comparison policy are unchanged.

The execution layer adds:

- immutable execution and checkpoint metadata;
- explicit session lifecycle rules;
- deterministic alignment by explicit mapping or unique checkpoint name;
- a validation policy that reuses `ComparisonConfig`;
- explicit outcomes for missing, shape-incompatible, dtype-rejected, invalid,
  and backend-failed checkpoints;
- `passed` / `failed` / `incomplete` aggregation;
- local execution and validation artifacts with versioned manifests, SHA-256
  checksums, safe paths, atomic publication, and pickle-free loading;
- validation JSON serialization and reload.

Verified locally on Windows 11 with CPython 3.13.5 and NumPy 2.1.3:

- the Rust core passes 57 tests; `cargo fmt --check` and Clippy with
  `-D warnings` pass;
- Ruff passes over `python/`, `tests/`, and `benchmarks/`; strict mypy passes on
  29 source files;
- the Python suite passes **399 tests with 2 skips**;
- wheel and sdist `0.2.0` artifacts build successfully;
- both artifacts install into separate clean virtual environments, and the
  installed package performs native comparison and execution validation;
- the built extension reports version `0.2.0` and the Python loader accepts it.

Additionally, for the published release:

- the `v0.2.0` release workflow built the Linux wheel and sdist, validated
  metadata, installed the wheel, passed the native-backend smoke test, and
  uploaded both distributions to PyPI;
- a clean `pip install tensortrail==0.2.0` from PyPI built the extension from
  the sdist and ran native comparison plus persisted execution validation.

Not verified for this revision:

- GPU, distributed, or accelerator execution (not implemented);
- sustained performance of the new validation layer (no new benchmark run).

## 2. Implemented capability

| Area | Implementation | Status |
| --- | --- | --- |
| Comparison metrics | Fused MAE, MSE, RMSE, relative mean/max, cosine, max absolute error, tolerance mismatches, non-finite counts | Verified, unchanged from 0.1.0 |
| Statistics | Welford min/max/mean/population variance/std dev, optional fused pass | Verified, unchanged |
| Native engine | Rust crate with PyO3/numpy bindings and Rayon reductions | Verified |
| Reference engine | NumPy implementation of the same numerical policy | Verified |
| Execution metadata | Immutable provenance: model, framework, precision, device, input, dataset, timestamps, JSON configuration | Verified |
| Checkpoints | Immutable tensor metadata separated from payloads; metadata-only NumPy/PyTorch description | Verified |
| Sessions | `created` / `recording` / `completed` / `failed` lifecycle, ordered registration, explicit payload access | Verified |
| Alignment | Explicit ID/name mappings first, unique names second, structured unmatched entries, metadata mismatches | Verified |
| Validation policy | Comparison settings plus checkpoint selection, missing/shape/dtype policy, `all`/`any` aggregation | Verified |
| Validation engine | Per-checkpoint `tt.compare()` calls, explicit skip outcomes, status aggregation | Verified |
| Results and reporting | Typed checkpoint/validation results, summary, strict JSON, reload, non-finite sentinels | Verified |
| Artifacts | Local versioned manifests, SHA-256 checksums, safe paths, atomic directory publication, `allow_pickle=False` | Verified |
| Comparison reports | Existing JSON/HTML plus comparison-report reload for embedded validation reports | Verified |
| Packaging | Maturin wheel/sdist, `py.typed`, Apache-2.0 license, SBOM in wheel | Verified locally for 0.2.0 |
| CI | Rust, lint/type, OS/Python matrix, NumPy floor, packaging jobs | Not run for 0.2.0; 0.1.0 CI was green |
| Publishing | Separate `publish.yml`, OIDC trusted publishing, version/tag checks | 0.1.0 published; 0.2.0 not authorized |

## 3. Architecture and data boundaries

The existing split is preserved:

- the Rust core owns shape/option validation, numerical kernels, reductions, and
  deterministic parallel execution;
- the Python comparison layer owns normalization, provenance, backend
  selection, comparison reports, and JSON/HTML serialization;
- the new Python execution layer owns session lifecycle, checkpoint metadata,
  payload references, alignment, validation policy/results, and local artifact
  persistence.

The execution and validation packages do not import PyTorch or the native
extension at package import time. The Rust crate is independent of PyTorch,
TensorFlow, ONNX, and model architectures.

Checkpoint metadata and payloads are separate. Registering a NumPy array stores
a reference to the array; registering a CPU PyTorch tensor records its shape,
dtype, byte size, and device without converting it. Validation calls the
existing explicit CPU adapter when a comparison is executed. Accelerator,
quantized, and unsupported tensors are rejected rather than transferred.
Persistence currently requires NumPy-compatible payloads; a PyTorch payload must
be converted explicitly by the caller first. See
[execution_model.md](execution_model.md) for the full contract.

## 4. Execution validation semantics

Lifecycle:

```text
created -> recording -> completed
   |          |
   +----------+-----> failed
```

- The first checkpoint registration moves `created` to `recording`.
- `finalize()` accepts `created` or `recording` and produces `completed`.
- `fail(reason)` accepts `created` or `recording` and produces `failed`.
- Both terminal states reject further registration or transitions.
- Validation requires two completed sessions. Empty completed sessions produce
  `incomplete`; they never fabricate a pass.

Alignment:

- explicit mapping (checkpoint IDs or names) first;
- unique checkpoint name second;
- otherwise an explicit `missing_reference` / `missing_candidate` entry;
- list position is never an alignment rule;
- a supplied alignment must cover every checkpoint unless a checkpoint selection
  is configured;
- duplicate names, ambiguous mappings, unknown endpoints, and selection
  mismatches raise `AlignmentError`.

Policy and status:

- `ValidationPolicy` validates its comparison fields by constructing a
  `ComparisonConfig`; tolerance and metric logic are not reimplemented.
- Missing required checkpoints and shape mismatches produce `incomplete` by
  default. `fail_on_missing_checkpoints=True` and
  `fail_on_shape_mismatch=True` turn them into `failed`.
- A dtype change is reported and compared unless `fail_on_dtype_change=True`
  makes it a policy failure without a comparison.
- Unsupported dtypes, invalid tensors, and backend failures are explicit
  checkpoint outcomes, never passes.
- `aggregation_strategy="all"` is the default. `"any"` passes when at least one
  matched checkpoint passes, but required incomplete and structural
  policy-failure conditions take precedence.

Persistence:

- execution layout: `manifest.json` plus per-checkpoint `metadata.json` and
  `tensor.npy`;
- validation layout: `manifest.json` plus `validation.json`;
- manifests are versioned and typed, and record library version, identity,
  status/timestamps, policy/alignment/results, and SHA-256 checksums;
- IDs are validated as safe single path segments;
- artifacts are staged and published by directory rename, and existing
  artifacts are never overwritten;
- `.npy` files are loaded with `allow_pickle=False`;
- metadata, checksums, byte sizes, shapes, and dtypes are verified on load;
- missing, malformed, or tampered artifacts raise `ArtifactError` or
  `ArtifactIntegrityError`.

## 5. Numerical policy

The numerical policy is unchanged from 0.1.0:

- working dtypes are float32 or float64; accumulation and reported scalars are
  float64;
- float16 and bfloat16 widen to float32; integers and bools widen to float64;
  mixed float32/float64 widens to float64;
- non-contiguous, negative-stride, unaligned, and non-native-endian arrays are
  copied to contiguous native buffers with conversion notes;
- non-finite values are excluded from metric sums and statistics and counted by
  category; non-finite pattern differences fail the comparison;
- empty or entirely non-finite inputs produce `None` metrics, never zeros;
- relative error uses `|r - c| / max(|r|, epsilon)` with default
  `epsilon=1e-12`;
- cosine handles two zero vectors as `1.0`, one zero vector as `0.0`, and uses a
  scaled fallback for extreme magnitudes;
- variance is population variance (`n`);
- tolerance is `|r - c| <= atol + rtol * |r|` with defaults `atol=1e-6` and
  `rtol=1e-5`.

Determinism is unchanged: fixed chunk boundaries and ordered partial merges make
parallel results identical across worker counts for a fixed chunk size.
Sequential and parallel paths can differ in the last bits, and changing
`chunk_size` can change rounding. This is documented rather than presented as
bit-identical reproducibility.

Validation adds policy around comparisons, not a new numerical implementation.
Each matched checkpoint pair is passed to the existing comparison engine with the
policy's `ComparisonConfig`.

## 6. Verification evidence

All commands below were run from the repository root on Windows 11 unless noted.

### Rust

```text
cargo fmt --check
  -> exit 0

cargo clippy --all-targets --all-features -- -D warnings
  -> Finished; no warnings

cargo test
  -> 57 passed; 0 failed; 0 ignored
```

The 57 tests cover hand-calculated values, empty inputs, NaN/Inf handling,
overflow, zero vectors, epsilon floors, Welford statistics, fused-versus-standalone
agreement, parallel thread-count invariance, chunk tails, cached pools, and shape
validation. The Rust sources were not modified for 0.2.0; only the crate version
changed.

### Python

```text
ruff check python/tensortrail tests benchmarks
  -> All checks passed

mypy python/tensortrail
  -> Success: no issues found in 29 source files
```

Primary environment (CPython 3.13.5, NumPy 2.1.3, torch 2.9.1+cpu):

```text
pytest -q -p no:anchorpy
  -> 399 passed, 2 skipped
```

Skips:

- extended-precision `longdouble` rejection test: on this Windows build,
  `longdouble` is the same as `float64`, so there is no distinct dtype to
  exercise;
- CUDA rejection test: CUDA is not available.

The suite grew from 208 to 399 tests. The new coverage includes:

- metadata validation, identifiers, timestamps, JSON round-trips, and rejection
  of non-JSON values;
- checkpoint metadata, tensor description, registry uniqueness/order, session
  lifecycle, and payload separation;
- persistence round-trips, non-overwriting saves, checksums, tamper detection,
  missing files, unsafe IDs, and pickle-free loading;
- policy defaults/validation, alignment rules, explicit mappings, selections,
  missing entries, and metadata mismatches;
- validation outcomes for pass/fail, numerical failure, missing checkpoints,
  shape and dtype policies, unsupported dtypes, backend failures, invalid
  tensors, empty sessions, and aggregation strategies;
- validation JSON schema, strict reload, non-finite sentinels, and comparison
  report reload;
- an end-to-end persisted workflow and public-API compatibility checks;
- optional PyTorch metadata, persistence rejection, and in-memory validation.

The `-p no:anchorpy` flag is used only in the Anaconda environment because an
unrelated globally installed plugin fails to import there. It is not required by
the project's CI configuration.

Torch tests run after the large NumPy numerical tests in this Anaconda
environment. Importing torch before a large NumPy dot product aborts with
`OMP: Error #15`; this is an environment-specific OpenMP conflict, documented in
[development.md](development.md).

### Packaging

```text
maturin build --release --sdist --out dist
  -> dist/tensortrail-0.2.0.tar.gz
  -> dist/tensortrail-0.2.0-cp313-cp313-win_amd64.whl
```

Wheel contents verified:

- `tensortrail/` Python package and `py.typed`;
- the new `execution/` and `validation/` packages;
- `tensortrail/_core.cp313-win_amd64.pyd`;
- `LICENSE` under `dist-info/licenses/`;
- `METADATA` with version `0.2.0`, project URLs, requires-python, extras, and the
  NumPy dependency;
- a CycloneDX SBOM of the Rust dependency tree.

Sdist contents verified:

- Rust crate, `Cargo.toml`, `Cargo.lock`, and workspace manifest;
- Python package and `py.typed`, including the new execution/validation modules;
- tests, documentation (including `docs/execution_model.md`), benchmark
  scripts/results, contributing/changelog/conduct files;
- `README.md`, `LICENSE`, and `pyproject.toml` with version `0.2.0`.

### Clean installation

- The cp313 wheel installed into a standalone CPython 3.13 virtual environment
  with NumPy from PyPI. `tt.compare` returned status `fail` for a deliberate
  difference; a persisted execution pair validated as `passed`; the native
  extension reported version `0.2.0`.
- The sdist installed with pip into a second fresh virtual environment, building
  the Rust extension from source. The installed package reported Python version
  `0.2.0`, native version `0.2.0`, and `is_native_available() == True`.
- The installed wheel's public namespace still exposes the complete 0.1.0 API
  alongside `ExecutionSession`, `ValidationPolicy`, `ValidationResult`,
  `align_checkpoints`, `validate`, and `FilesystemStorage`.

Maturin's Windows audit again reported a transitive `zlib.dll` dependency
originating from Anaconda's `python313.dll`, not from the extension itself. The
extension's import table references `python313.dll` plus Windows system DLLs. The
standalone CPython virtual environments installed and ran successfully,
confirming that bundling Anaconda's zlib would be incorrect.

### Compatibility

- `tt.compare()`, the metric helpers, configuration types, comparison report
  types, and 0.1.0 report schema `"1.0"` are unchanged.
- The full 0.1.0 test suite continues to pass; new tests were added without
  weakening existing assertions.
- Importing `tensortrail` still does not import PyTorch or the native extension.
- Validation uses the existing comparison engine for every matched pair.

### CI status

The hosted CI matrix passed for the publication follow-up commit `a1e1ff1`
(16 of 16 jobs): Rust checks, Python lint/types, the NumPy 1.24 floor, Ubuntu,
Windows, and macOS test jobs for Python 3.11–3.14, and the distribution build.
The earlier push of the 0.2.0 code commit `f423575` had its macOS/Python 3.13
job cancelled by the runner before executing any steps. The publication
follow-up run exercised all four macOS versions; two jobs were initially
cancelled for capacity and passed when retried.

The release-triggered publish workflow for `v0.2.0` passed its build,
version/tag verification, metadata validation, wheel install, and native smoke
test, then uploaded the wheel and sdist to PyPI. A clean install of the
published package passed native comparison and persisted validation.

### Benchmarks

The Rust numerical engine and its kernels are unchanged. The 0.1.0 end-to-end
comparison benchmarks therefore remain representative of `tt.compare()`, but
they do not measure session construction, alignment, or artifact I/O. No new
benchmark run was performed for 0.2.0, and no performance claim is made for the
validation layer.

Previously recorded `tt.compare` medians (Windows 11, i7-12700H class, CPython
3.13.5, NumPy 2.1.3, release build) remain in
[benchmarking.md](benchmarking.md) and `benchmarks/results/`.

## 7. Release status

Published for the scoped 0.2.0 alpha:

- execution metadata, sessions, checkpoints, alignment, validation policy,
  validation engine, results, and local artifacts are implemented and tested;
- missing, shape-incompatible, dtype-rejected, unsupported, invalid, and
  backend-failed checkpoints are explicit outcomes, never passes;
- every matched pair is compared by the existing engine;
- persistence uses versioned manifests, checksums, safe paths, atomic writes,
  and pickle-free loading;
- the complete 0.1.0 public API remains available;
- wheel and sdist build and install cleanly;
- documentation distinguishes implemented behavior from deferred work;
- the `v0.2.0` GitHub release and PyPI upload were authorized and completed on
  2026-11-13.

Known gaps at publication:

- no 0.2.0 benchmark run covers the new orchestration layer;
- PyTorch checkpoints are supported in memory but require explicit conversion
  before persistence.

The 0.1.0 production upload occurred on 2026-10-31. The 0.2.0 production upload
occurred on 2026-11-13; TestPyPI was not used.

## 8. Deliberately not implemented

- GPU/CUDA/MPS-native metrics, multi-device comparison, or distributed execution.
- Automatic activation capture, framework hooks, graph tracing, or model
  patching. Checkpoint capture is manual.
- Cloud or object storage for execution artifacts.
- Explicit AVX/NEON/SVE intrinsics or runtime CPU dispatch. The native build
  reports `simd: "scalar"`; no auto-vectorization claim is made.
- Complex-valued, quantized, sparse, or masked-array semantics.
- Cross-backend bit-identical results.
- Automatic host transfers for accelerator tensors, or implicit conversion of
  framework payloads during persistence.
- A diagnostic hypothesis engine. The result records measurements and explicit
  status; it does not identify the internal operation that caused a difference.

These are scope decisions, not hidden fallbacks. Each rejection path is explicit
and covered by tests where applicable.

## 9. Reproducing the verification

Prerequisites: Python 3.11+, Rust 1.85+ (tested 1.95.0), Maturin 1.16.0, MSVC on
Windows. From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install "maturin>=1.16,<2.0" "numpy>=1.24" pytest mypy ruff
maturin develop --release

cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test

ruff check python/tensortrail tests benchmarks
mypy python/tensortrail
pytest -q

maturin build --release --sdist --out dist
```

On Windows/Anaconda, select the intended interpreter explicitly when building
(`-i C:/path/to/python.exe`) and ensure `PYO3_PYTHON` points at the same
interpreter. If maturin reports `Access is denied` while running `cargo rustc`,
check for a stray `cargo` environment variable that points at a directory rather
than the executable.

See [development.md](development.md) for platform notes, packaging details, and
the clean-install smoke tests.

## 10. Release history

- **0.1.0** (2026-10-31): initial alpha; published to PyPI through the
  release-triggered OIDC workflow. The numerical core, Python API, comparison
  reports, tests, and benchmarks were verified.
- **0.2.0** (2026-11-13): execution-aware validation layer; tagged `v0.2.0`,
  released on GitHub, and published to PyPI through the release-triggered OIDC
  workflow.

## 11. Conclusion

TensorTrail 0.2.0 delivers execution-aware inference validation without changing
the numerical engine or claiming capabilities it does not have. Sessions and
checkpoint metadata make provenance explicit; alignment is deterministic;
missing and incompatible checkpoints cannot become passes; persistence is
versioned, checksummed, and pickle-free. The complete 0.1.0 comparison API
remains available.

The main remaining risks are platform coverage (only a Linux x86_64 wheel is
published), the lack of performance measurements for the new orchestration
layer, and the absence of a new benchmark run. These are stated rather than
hidden. Tagging and publication were authorized and completed on 2026-11-13.
