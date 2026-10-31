# Contributing to TensorTrail

Thank you for considering a contribution. TensorTrail is a numerical library:
correctness, explicit policies, and reproducible evidence matter more than adding
surface area quickly.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

- Read [Architecture](docs/architecture.md) and
  [Numerical methods](docs/numerical_methods.md). The numerical policy is a
  contract, not an implementation detail.
- Search existing issues and pull requests before opening a new one.
- For a large change (new backend, new metric, public API change, new
  dependency), open an issue first so the design can be agreed on.
- Security issues should not be filed publicly. Contact the maintainers through
  the repository's private security reporting channel.

## Development setup

Follow [Development](docs/development.md). The short version:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "maturin>=1.16,<2.0" "numpy>=1.24" pytest mypy ruff
maturin develop --release
```

## What makes a good change

### Numerical behavior

- Define behavior for empty inputs, non-finite values, zeros, mixed dtypes, shape
  mismatch, non-contiguous layouts, and overflow before implementing it.
- Keep the NumPy reference backend and the Rust engine in agreement. Add a parity
  test for every new numerical branch.
- Use tolerances justified by floating-point error analysis, not tolerances
  chosen to make a failing test pass.
- Never return a fabricated zero or a fallback number when a result is
  mathematically undefined.
- Never add a silent backend or device fallback. If execution is impossible,
  raise an actionable error.

### Public API

- Keep imports lightweight: do not import torch or the native extension at
  package import time.
- Preserve the typed result schema unless the change is an intentional,
  documented schema revision.
- New configuration fields need validation, documentation, tests, and JSON/HTML
  report coverage.
- Prefer additive changes. A breaking change requires a major version and a
  migration note.

### Performance

- Include a benchmark when claiming a performance improvement. State the
  machine, input sizes, dtype, build profile, and baseline.
- Optimize the measured bottleneck, not an assumed one.
- Do not trade correctness or determinism for speed without documenting the
  trade-off.
- Keep the sequential path available. Parallel execution must remain
  deterministic for a fixed chunk size.

## Tests and checks

Run the full set before opening a pull request:

```bash
cargo fmt --manifest-path native/tensortrail-core/Cargo.toml --all -- --check
cargo clippy --manifest-path native/tensortrail-core/Cargo.toml --all-targets --all-features -- -D warnings
cargo test --manifest-path native/tensortrail-core/Cargo.toml --release --all-features

ruff check python tests benchmarks
mypy
pytest
```

Add tests at the lowest useful level:

- Rust unit tests for kernel logic and edge cases;
- Python tests for configuration, normalization, reports, and API behavior;
- cross-backend tests for numerical parity;
- integration tests for end-to-end workflows.

If a test is platform-specific, skip it with a reason rather than deleting it.
Do not mark a test as skipped to hide a failure.

## Documentation

Update the relevant document when behavior changes:

- `README.md` for user-visible usage and scope;
- `docs/numerical_methods.md` for formulas, policies, and tolerances;
- `docs/architecture.md` for module boundaries and execution flow;
- `docs/benchmarking.md` for measurement methodology and results;
- `CHANGELOG.md` for every user-visible change;
- `docs/engineering_report.md` for the verified status of the release.

Examples must run. Do not document a feature that is not implemented.

## Commit and pull request guidelines

- Keep commits focused; separate refactoring from behavior changes.
- Write imperative commit subjects, for example
  `add float16 normalization parity test`.
- Describe what changed, why, and how it was verified in the pull request.
- Link the issue being addressed.
- Include the exact commands and results for new numerical or performance
  behavior.
- Call out anything not verified, platform-specific, or intentionally deferred.

## Review criteria

A pull request is ready when:

- the numerical policy is explicit and tested;
- Rust and Python behavior agree where both backends implement a feature;
- public errors are actionable and typed;
- no silent fallbacks or device transfers were introduced;
- the full quality gate passes;
- documentation and changelog are updated;
- performance claims have reproducible evidence.

## License

By contributing, you agree that your contribution is licensed under the
Apache License 2.0, the same license as the project.
