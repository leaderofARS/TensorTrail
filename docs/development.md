# Development

This document covers the local toolchain, quality gates, packaging, and release
checks. TensorTrail has two build products: the Python package and the native
Rust extension that backs `tensortrail._core`.

## Prerequisites

| Requirement | Version used for verification | Notes |
| --- | --- | --- |
| Python | 3.13.5 | `requires-python = ">=3.11"`; 3.11–3.14 tested |
| NumPy | 2.1.3 | Runtime dependency; `>=1.24` is allowed |
| Rust | 1.95.0 stable | `rust-version = "1.85"` |
| Maturin | 1.16.0 | Build backend and `maturin develop` |
| C/C++ toolchain | MSVC 2022 on Windows | Xcode CLT on macOS, GCC/Clang on Linux |

PyTorch is optional. Installing it is only required for `tests/python/test_torch.py`;
the rest of the suite skips those tests when torch is absent.

## Bootstrap

Create and activate a virtual environment first. Maturin needs an active
environment for `maturin develop`.

```bash
git clone https://github.com/leaderofARS/tensortrail.git
cd tensortrail
python -m venv .venv
```

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the development dependencies and build the extension in release mode:

```bash
python -m pip install --upgrade pip
python -m pip install "maturin>=1.16,<2.0" "numpy>=1.24" pytest mypy ruff
maturin develop --release
```

`maturin develop` builds the Rust crate and installs the Python package in
editable form. After changing Rust code, run it again. Python-only changes are
picked up without rebuilding when the editable install points at `python/`.

To test the explicit NumPy reference backend without a native extension, install
the package without building Rust and use `ComparisonConfig(backend="numpy")`.
The default backend deliberately raises `BackendError` when the extension is
missing.

### Windows

The Rust host toolchain must match the Python interpreter. CPython on
python.org and Anaconda are MSVC builds, so use the MSVC Rust target:

```powershell
rustup default stable-x86_64-pc-windows-msvc
```

Run the build from a shell where the MSVC environment is available (the
"x64 Native Tools Command Prompt for VS 2022" or a shell that called
`vcvars64.bat`). If `link.exe` is not found, the Visual Studio C++ workload is
not installed or the environment was not initialized.

Some Anaconda installations combine Intel OpenMP with PyTorch's OpenMP runtime.
Importing torch before large NumPy operations can then abort with
`OMP: Error #15`. The test suite imports torch lazily inside individual tests and
runs torch tests last; it does not require a particular import order from users.
If the error occurs in your own process, set `KMP_DUPLICATE_LIB_OK=TRUE` only as
a local workaround and investigate the mixed runtimes.

### macOS and Linux

Install the platform compiler:

```bash
# macOS
xcode-select --install

# Debian / Ubuntu
sudo apt-get install build-essential
```

No CUDA toolkit, GPU, or vendor BLAS is required. The native engine is CPU-only.

## Tests

```bash
pytest
```

The suite parameterizes the NumPy and native backends. Native tests are skipped
when `tensortrail._core` cannot be imported; torch tests are skipped when torch
is absent. Platform-specific cases (for example 80-bit `longdouble`) are skipped
where the dtype does not exist.

Focused runs:

```bash
pytest tests/numerical -q
pytest tests/python/test_reporting.py -q
pytest -k parity -q
```

If an unrelated globally installed pytest plugin fails during collection, disable
it explicitly rather than changing project configuration:

```bash
pytest -q -p no:anchorpy
```

## Rust checks

Run these from the repository root:

```bash
cargo test --manifest-path native/tensortrail-core/Cargo.toml --release --all-features
cargo clippy --manifest-path native/tensortrail-core/Cargo.toml --all-targets --all-features -- -D warnings
cargo fmt --manifest-path native/tensortrail-core/Cargo.toml --all -- --check
```

The Rust tests cover hand-calculated metrics, edge cases, deterministic chunk
merges, thread-count invariance, overflow handling, and the fused kernel.

## Python checks

```bash
ruff check python tests benchmarks
mypy
pytest
```

Ruff is configured for Python 3.11, a 100-character line length, and a broad
rule set. Mypy runs in strict mode over `python/tensortrail`. Do not weaken either
configuration to make a change pass; fix the code or document a narrowly scoped
exception.

## Benchmarks

Kernel-level benchmarks use Criterion:

```bash
cargo bench --manifest-path native/tensortrail-core/Cargo.toml --bench metrics -- \
  --warm-up-time 0.3 --measurement-time 1.0 --sample-size 10
```

End-to-end Python benchmarks use the public API:

```bash
python benchmarks/benchmark_python.py \
  --json benchmarks/results/python-windows.json
```

Record the machine, Python/NumPy versions, commit, and build profile with any
reported number. Do not compare results across machines or build profiles
without saying so. See [benchmarking](benchmarking.md) for methodology and the
current results.

## Packaging

Build a wheel and source distribution:

```bash
maturin build --release --sdist --out dist
```

Inspect both artifacts before installing:

```bash
python -m zipfile -l dist/tensortrail-0.1.0-*.whl
tar -tzf dist/tensortrail-0.1.0.tar.gz | head -40
```

The wheel must contain:

- `tensortrail/` Python modules and `py.typed`;
- the compiled `tensortrail/_core` extension for the target interpreter;
- `LICENSE` and the project metadata.

The sdist must contain the Rust crate (`native/tensortrail-core`), `Cargo.toml`,
`Cargo.lock`, `pyproject.toml`, `README.md`, `LICENSE`, and the Python package.
Building from the sdist requires the Rust toolchain and a C compiler.

Smoke-test a clean install:

```bash
python -m venv /tmp/tt-smoke
/tmp/tt-smoke/bin/python -m pip install --no-deps dist/tensortrail-0.1.0-*.whl
/tmp/tt-smoke/bin/python -c "
import numpy as np, tensortrail as tt
r = tt.compare(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
assert r.metrics.mae == 1.0
print(tt.__version__, r.status)
"
```

The wheel is CPython- and platform-specific (the extension does not use the
stable ABI), so one wheel per OS/architecture/Python version is expected.

## CI

`.github/workflows/ci.yml` runs:

- formatting, Clippy, Rust tests, Ruff, and strict mypy;
- Python tests on Linux, macOS, and Windows for Python 3.11, 3.12, 3.13, and
  3.14, building the extension before testing;
- an sdist/wheel build with metadata checks and a clean-install smoke test.

CI never publishes packages. Publishing a release to PyPI or anywhere else is a
separate, explicitly authorized action.

## Versioning and release checklist

The version appears in three places and they must agree:

1. `pyproject.toml` (`project.version`);
2. `python/tensortrail/_version.py` (`__version__`);
3. `native/tensortrail-core/Cargo.toml` (`package.version`).

The Python adapter checks the native extension version at load time and raises
`BackendError` on a mismatch. Update `Cargo.lock` by building after changing the
crate version.

Checklist:

1. Bump the version in all three files and update `CHANGELOG.md`.
2. Run the Rust tests, Clippy, fmt check, Ruff, mypy, and pytest.
3. Run both benchmark suites if performance claims changed.
4. Build the wheel and sdist; inspect their contents.
5. Install the wheel into a clean environment and run the smoke test.
6. Update affected documentation and the engineering report.
7. Tag only after the checks above pass. Publication requires separate approval.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `BackendError: native TensorTrail extension is not available` | Run `maturin develop --release` in the active environment, or use `backend="numpy"` explicitly. |
| `does not match native extension version` | Rebuild the extension after a version bump. |
| `Couldn't find a virtualenv` from maturin | Activate the environment or set `VIRTUAL_ENV`. |
| `link.exe not found` on Windows | Install the MSVC C++ workload and initialize `vcvars64.bat`. |
| Importing torch aborts with `OMP: Error #15` | Conflicting OpenMP runtimes in the environment; see the Windows note above. |
| Native tests skipped | The extension is not installed in the interpreter running pytest. |
| Wheel imports a different copy of the package | Remove stale editable installs and reinstall the wheel; check `python -c "import tensortrail; print(tensortrail.__file__)"`. |

## Related documents

- [Architecture](architecture.md)
- [Numerical methods](numerical_methods.md)
- [Benchmarking](benchmarking.md)
- [Engineering report](engineering_report.md)
