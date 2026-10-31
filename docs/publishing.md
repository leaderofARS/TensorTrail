# Publishing

TensorTrail's release workflow is configured to use PyPI Trusted Publishing. No
repository secret or long-lived API token is stored in GitHub. When it
publishes, the workflow exchanges a GitHub OIDC token for a short-lived PyPI
upload credential.

Publication is intentionally separate from CI:

- `.github/workflows/ci.yml` tests pull requests and pushes and never publishes;
- `.github/workflows/publish.yml` builds release artifacts and is the only
  workflow with `id-token: write`.

## Trusted publisher configuration

Before the first upload, configure a pending publisher on each index. Both
entries must use the same repository and workflow filename.

| Field | Value |
| --- | --- |
| Project name | `tensortrail` |
| Owner | `leaderofARS` |
| Repository | `TensorTrail` |
| Workflow | `publish.yml` |
| Environment | empty / any |

- PyPI: <https://pypi.org/manage/account/publishing/>
- TestPyPI: <https://test.pypi.org/manage/account/publishing/>

If an earlier publisher entry points at `ci.yml`, update it to `publish.yml`.
The filename is part of the identity check; an OIDC token from a different
workflow will be rejected.

## Workflow targets

`publish.yml` can be started in four ways:

- a published GitHub release, which builds and publishes to PyPI;
- `workflow_dispatch` with target `testpypi`, which builds and publishes to
  TestPyPI;
- `workflow_dispatch` with target `build-only`, which builds, validates, and
  uploads the distributions as a GitHub Actions artifact without publishing;
- `workflow_dispatch` with target `pypi`, which is intended for recovery and
  must be dispatched from the `v<version>` tag.

The build job checks that the version in `pyproject.toml`, `_version.py`, and the
Rust crate agree. For a GitHub release, or a manual `pypi` dispatch, it also
requires the checked-out ref to be exactly `v<version>`. It installs the built
wheel and runs a native-backend smoke test before the artifact is offered to the
publish jobs. The publish jobs download that validated artifact rather than
rebuilding it.

To validate the workflow without uploading:

```bash
gh workflow run publish.yml -f target=build-only
```

## Validation status

The `build-only` target has been exercised on GitHub Actions for 0.1.0. The
Ubuntu build produced a CPython 3.13 `manylinux_2_34_x86_64` wheel and an sdist,
installed the wheel, and passed the native-backend smoke test. The TestPyPI and
PyPI publish jobs were skipped. No package-index upload has been performed.

## TestPyPI validation

Run the manual workflow first:

```bash
gh workflow run publish.yml -f target=testpypi
```

Watch the resulting `Publish TensorTrail` run in the Actions tab. Then install in
a fresh environment, using PyPI as the dependency fallback:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "tensortrail==0.1.0"
python -c "import numpy as np, tensortrail as tt; print(tt.compare(np.array([1.0]), np.array([2.0])).metrics.mae)"
```

The same version cannot be uploaded twice. If a TestPyPI upload fails after the
files are accepted, bump the version rather than assuming a file can be replaced.

## Production release

1. Ensure CI is green on `main`.
2. Review `CHANGELOG.md`, `README.md`, and `docs/engineering_report.md`.
3. Create and publish a GitHub release tagged `v0.1.0`. The workflow rejects a
   tag that does not match the three version declarations.
4. Confirm the `Publish TensorTrail` workflow succeeds.
5. Verify a clean install from PyPI.
6. Only then update the README's publication status.

For a manual production run, select the release tag and dispatch:

```bash
gh workflow run publish.yml --ref v0.1.0 -f target=pypi
```

Until the production upload succeeds, the README must continue to say no package
has been published.

## Platform scope

The release workflow currently builds on Ubuntu for CPython 3.13 on x86_64 and
uploads one Linux wheel plus the source distribution. Windows, macOS, and other
Python-version wheels require additional build jobs on their native runners. Do
not claim broad platform wheel coverage until those artifacts are built, tested,
and published.

Source installations require a Rust toolchain and a C compiler because the
native extension does not use the stable ABI.

## Release rules

- Never reuse a published version or filename.
- Do not store PyPI tokens in GitHub secrets when Trusted Publishing is
  available.
- Keep the workflow filename and environment synchronized with the pending
  publisher entry.
- Require explicit approval before production publication.
