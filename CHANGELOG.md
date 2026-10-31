# Changelog

All notable changes to TensorTrail are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
TensorTrail uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The project is currently in the `0.x` series: public APIs may change in a minor
release, and such changes will be called out here.

## [Unreleased]

Nothing yet.

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
- No package has been published to PyPI by this repository state.

[Unreleased]: https://github.com/leaderofARS/tensortrail/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/leaderofARS/tensortrail/releases/tag/v0.1.0
