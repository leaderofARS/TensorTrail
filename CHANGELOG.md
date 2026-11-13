# Changelog

All notable changes to TensorTrail are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
TensorTrail uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The project is currently in the `0.x` series: public APIs may change in a minor
release, and such changes will be called out here.

## [0.2.0] - 2026-11-13

Published to PyPI on 2026-11-13 from the `v0.2.0` tag through the release-triggered
OIDC workflow. The uploaded artifacts are a CPython 3.13
`manylinux_2_34_x86_64` wheel and an sdist.

Execution-aware inference validation is added on top of the 0.1.0 numerical
engine. The comparison engine itself is unchanged, and every checkpoint pair is
still evaluated by `tt.compare()`.

### Added

- `ExecutionMetadata` for immutable execution provenance: model identity and
  version, framework, precision, device, input/dataset identifiers, timestamps,
  and JSON-compatible configuration/custom metadata.
- `TensorCheckpoint` for tensor metadata separate from payloads, plus
  metadata-only description of NumPy and CPU PyTorch tensors without conversion.
- `CheckpointRegistry` with unique IDs, names, and indices; deterministic
  iteration by execution order; structured lookup errors.
- `ExecutionSession` with explicit `created -> recording -> completed` and
  `created/recording -> failed` lifecycles, payload access by checkpoint ID, and
  no implicit tensor copies or accelerator transfers.
- `ValidationPolicy`, reusing `ComparisonConfig` for metrics, tolerances,
  epsilon, backend, and dtype normalization policy. It adds checkpoint
  selection, required-checkpoint handling, shape/dtype failure policy, and
  `all`/`any` aggregation.
- `align_checkpoints()` and `CheckpointAlignment` with explicit ID/name
  mappings, unique-name matching, metadata mismatch reporting, and structured
  missing-reference/missing-candidate entries. Position is never an alignment
  rule.
- `tt.validate()` and `ValidationResult` with per-checkpoint outcomes
  (`passed`, `failed`, `missing_reference`, `missing_candidate`,
  `shape_mismatch`, `dtype_mismatch`, `unsupported_dtype`, `invalid_tensor`,
  `backend_error`), overall `passed`/`failed`/`incomplete` status, derived
  counts, highest observed deviation, and a text summary.
- `FilesystemStorage` and the `ExecutionStorage` protocol for local execution
  and validation artifacts. Artifacts use versioned manifests, SHA-256
  checksums, safe path segments, atomic directory publication, and
  `np.load(..., allow_pickle=False)`.
- Validation JSON serialization/reload (`ValidationResult.save/load/to_dict/
  to_json/from_dict`) with strict schema checks and non-finite sentinels.
- `report_from_dict()` for reloading comparison reports embedded in validation
  artifacts.
- New exceptions: `ExecutionStateError`, `CheckpointError`,
  `CheckpointNotFoundError`, `AlignmentError`, `ArtifactError`, and
  `ArtifactIntegrityError`.
- `docs/execution_model.md`, the public contract for lifecycle, alignment,
  status semantics, persistence, errors, compatibility, and limitations.
- Tests for metadata, checkpoints, registries, sessions, persistence and
  tamper detection, policy, alignment, validation engine, reporting, an
  end-to-end persisted workflow, and optional PyTorch payloads.

### Changed

- The package version is now `0.2.0`; the native extension reports the same
  version and the Python loader continues to reject a mismatch.
- Validation requires completed sessions and rejects partial explicit
  alignments unless a checkpoint selection is configured.
- `CheckpointValidationResult` rejects outcomes that disagree with the status of
  their attached comparison report.
- The public namespace exports the execution/validation types alongside the
  complete 0.1.0 API.

### Verification

- Rust: `cargo fmt --check`, Clippy with `-D warnings`, and 57 tests pass.
- Python: Ruff passes; strict mypy passes on 29 source files; the full suite
  reports **399 passed, 2 skipped** on Windows with CPython 3.13 and torch
  installed.
- `maturin build --release --sdist` produces `0.2.0` wheel and sdist artifacts;
  both install into clean virtual environments, and the installed package runs
  native comparison and persisted validation.
- The built extension reports version `0.2.0`.
- The `v0.2.0` release workflow built and validated the Linux wheel and sdist,
  passed the native smoke test, and uploaded both to PyPI.
- A clean `pip install tensortrail==0.2.0` from PyPI built the native extension
  and ran native comparison plus persisted execution validation.

### Known limitations

- Activation capture is manual; there are no framework hooks, graph tracing, or
  model-patching APIs.
- In-memory validation accepts CPU PyTorch tensors, but artifact persistence
  requires an explicitly converted NumPy payload. PyTorch payloads are rejected
  for persistence rather than transferred or converted implicitly.
- Checkpoint names must be unique within a session; repeated stage/name pairs
  must be distinguished by the caller.
- Only local filesystem persistence is implemented; cloud/object storage is not.
- Alignment matches names or explicit mappings; it does not infer graph
  topology or compare by list position.
- The result reports per-checkpoint measurements and the largest observed
  deviation. It does not claim to identify the internal operation that caused a
  difference.
- GPU, distributed execution, explicit SIMD, complex, quantized, sparse, and
  masked-array support remain out of scope.

## [0.1.0] - 2026-10-31

Initial alpha release. The numerical core, Python package, reports, tests, and
benchmarks described below are implemented and verified on the documented
development platform. Explicit SIMD, GPU execution, and activation capture are
not included; see [Scope and non-goals](README.md#scope-and-non-goals).

### Added

- Rust numerical engine with fused MAE, MSE, RMSE, relative error, cosine
  similarity, maximum absolute error, tolerance mismatch counts, non-finite
  counts, and optional Welford statistics.
- Deterministic Rayon chunk reduction with fixed chunk boundaries, ordered
  partial merges, cached thread pools, and a sequential path.
- PyO3/numpy bindings exposed as the optional `tensortrail._core` extension, with
  lazy loading, a native/Python version check, and explicit `BackendError` when
  the extension is unavailable.
- NumPy reference backend implementing the same numerical and tolerance policy,
  used as a parity oracle.
- Public Python API: `tt.compare`, `tt.mae`, `tt.mse`, `tt.rmse`,
  `tt.relative_error`, `tt.cosine_similarity`, `tt.statistics`,
  `tt.is_native_available`, `tt.available_backends`, and `tt.native_info`.
- Typed, immutable report objects with metrics, counts, tolerances, tensor
  provenance, conversion notes, execution metadata, and schema version `1.0`.
- Human-readable `report.summary()`, strict JSON serialization with non-finite
  sentinels, self-contained HTML reports, and atomic report writes.
- Explicit input normalization for float16/bfloat16, integers, bools, mixed
  dtypes, non-contiguous and negative-stride arrays, unaligned buffers, and
  non-native byte order. Complex, object, string, datetime, masked, and extended
  precision arrays are rejected with actionable errors.
- Optional CPU PyTorch input support with lazy import, explicit conversion notes,
  autograd detachment, and rejection of CUDA, MPS, meta, quantized, sparse, and
  complex tensors.
- Test suite covering hand-calculated values, NumPy/Rust parity, parallel
  determinism, chunk tails, dtype and layout handling, configuration validation,
  report serialization and escaping, import isolation, missing-extension
  behavior, and an end-to-end workflow.
- End-to-end Python benchmarks and Criterion kernel benchmarks, with raw results
  and methodology recorded in `benchmarks/results/` and
  [docs/benchmarking.md](docs/benchmarking.md).
- Documentation for architecture, numerical policy, development, benchmarking,
  contribution, and release verification.
- GitHub Actions CI for Rust checks, Python lint/type checks, a Python/OS test
  matrix, and wheel/sdist packaging smoke tests.
- Apache-2.0 license, `py.typed`, Maturin build configuration, wheel and sdist
  build support.
- Separate OIDC-based GitHub Actions publishing workflow for TestPyPI and PyPI,
  with version/tag checks, a build-only validation path, and wheel smoke testing
  before upload.

### Known limitations

- No GPU, CUDA, MPS, multi-device, or distributed execution.
- No activation capture, hooks, graph tracing, or model patching.
- No explicit SIMD intrinsics or runtime CPU feature dispatch. The native build
  reports `simd: "scalar"`.
- No complex-valued, quantized, sparse, or masked-array support.
- Non-finite values are excluded from metric sums and statistics rather than
  propagated; non-finite pattern differences still fail the comparison.
- Metrics over empty or entirely non-finite comparisons are `None`, not zero.
- Cross-backend results can differ in the last bits because summation orders
  differ; parity tests use justified tolerances.
- The 0.1.0 wheel targets CPython 3.13 on Linux x86_64; other platforms build
  from source. The package is published on PyPI.

[0.2.0]: https://github.com/leaderofARS/tensortrail/releases/tag/v0.2.0
[0.1.0]: https://github.com/leaderofARS/tensortrail/releases/tag/v0.1.0
