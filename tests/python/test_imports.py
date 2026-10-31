"""Import isolation and explicit backend failure behavior."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import tensortrail as tt
from tensortrail.backends import rust_backend
from tests.conftest import requires_native


def _run(code: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout.strip().splitlines()


def test_public_import_does_not_load_optional_dependencies() -> None:
    lines = _run(
        "import sys\n"
        "import tensortrail as tt\n"
        "print(tt.__version__)\n"
        "print('torch' in sys.modules)\n"
        "print('tensortrail._core' in sys.modules)\n"
    )
    assert lines[0] == tt.__version__
    assert lines[1] == "False"
    assert lines[2] == "False"


def test_numpy_backend_never_loads_the_native_extension() -> None:
    lines = _run(
        "import sys\n"
        "import numpy as np\n"
        "import tensortrail as tt\n"
        "tt.compare(np.ones(4), np.ones(4), config=tt.ComparisonConfig(backend='numpy'))\n"
        "print('tensortrail._core' in sys.modules)\n"
    )
    assert lines[0] == "False"


@requires_native
def test_using_the_default_backend_loads_native_only_on_demand() -> None:
    lines = _run(
        "import sys\n"
        "import numpy as np\n"
        "import tensortrail as tt\n"
        "print('tensortrail._core' in sys.modules)\n"
        "tt.compare(np.ones(4), np.ones(4))\n"
        "print('tensortrail._core' in sys.modules)\n"
        "print('torch' in sys.modules)\n"
    )
    assert lines == ["False", "True", "False"]


def test_missing_extension_raises_instead_of_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_backend, "_CORE", None)
    monkeypatch.setattr(
        rust_backend,
        "_LOAD_ERROR",
        tt.BackendError("simulated missing native extension"),
    )

    with pytest.raises(tt.BackendError, match="simulated missing native extension"):
        tt.compare(np.ones(4), np.ones(4))
    with pytest.raises(tt.BackendError, match="simulated missing native extension"):
        tt.native_info()
    assert tt.is_native_available() is False
    assert tt.available_backends() == ("numpy",)

    # The explicit reference backend remains fully usable.
    report = tt.compare(
        np.ones(4),
        np.ones(4),
        config=tt.ComparisonConfig(backend="numpy"),
    )
    assert report.status == "pass"
    assert report.metadata.backend == "numpy"


def test_native_version_mismatch_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(__version__="9.9.9")
    monkeypatch.setattr(rust_backend, "_CORE", None)
    monkeypatch.setattr(rust_backend, "_LOAD_ERROR", None)
    monkeypatch.setattr(
        rust_backend,
        "importlib",
        SimpleNamespace(import_module=lambda _name: fake_module),
    )

    with pytest.raises(tt.BackendError, match="does not match native extension version"):
        tt.compare(np.ones(2), np.ones(2))
    assert tt.is_native_available() is False


@requires_native
def test_native_module_reports_matching_version() -> None:
    import tensortrail._core as core

    assert core.__version__ == tt.__version__
    assert isinstance(core.native_info(), dict)
