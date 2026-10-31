"""Input normalization, layout conversion, and rejected dtypes."""

from __future__ import annotations

import numpy as np
import pytest

import tensortrail as tt
from tests.conftest import BACKEND_PARAMS


def _config(backend: str, **overrides: object) -> tt.ComparisonConfig:
    values: dict[str, object] = {"backend": backend, "metrics": tt.SUPPORTED_METRICS}
    values.update(overrides)
    return tt.ComparisonConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_c_and_fortran_order_agree(backend: str) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(17, 31))
    c_order = np.ascontiguousarray(values)
    fortran_order = np.asfortranarray(values)
    assert not fortran_order.flags.c_contiguous

    c_report = tt.compare(c_order, c_order.copy(), config=_config(backend))
    f_report = tt.compare(fortran_order, fortran_order.copy(), config=_config(backend))

    assert c_report.metrics == f_report.metrics
    assert c_report.reference.shape == (17, 31)
    assert "non_contiguous_to_contiguous" in f_report.reference.conversion_notes
    assert f_report.reference.converted is True
    assert c_report.reference.converted is False


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_transposed_view_is_converted_explicitly(backend: str) -> None:
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    transposed = values.T
    assert not transposed.flags.c_contiguous

    report = tt.compare(transposed, transposed.copy(), config=_config(backend))
    assert report.reference.shape == (6, 4)
    assert "non_contiguous_to_contiguous" in report.reference.conversion_notes
    assert report.metrics.mae == 0.0
    assert report.status == "pass"


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_negative_stride_is_converted_explicitly(backend: str) -> None:
    base = np.arange(32, dtype=np.float64)
    reversed_view = base[::-1]
    assert reversed_view.strides[0] < 0

    report = tt.compare(reversed_view, reversed_view.copy(), config=_config(backend))
    assert "non_contiguous_to_contiguous" in report.reference.conversion_notes
    assert report.reference.elements == 32
    assert report.metrics.mae == 0.0


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_mixed_layouts_compare_consistently(backend: str) -> None:
    values = np.arange(20, dtype=np.float64).reshape(4, 5)
    fortran = np.asfortranarray(values)
    reversed_rows = values[::-1]
    report = tt.compare(fortran, reversed_rows, config=_config(backend))
    expected = np.abs(fortran - reversed_rows).mean()
    assert report.metrics.mae == pytest.approx(expected)
    assert "non_contiguous_to_contiguous" in report.candidate.conversion_notes


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_unaligned_buffer_is_copied_and_recorded(backend: str) -> None:
    backing = bytearray(8 * 4 + 1)
    unaligned = np.frombuffer(backing, dtype=np.float64, count=4, offset=1)
    assert unaligned.flags.aligned is False
    unaligned[:] = [1.0, 2.0, 3.0, 4.0]

    report = tt.compare(unaligned, unaligned.copy(), config=_config(backend))
    assert "unaligned_to_aligned" in report.reference.conversion_notes
    assert report.metrics.mae == 0.0


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_non_native_byte_order_is_normalized(backend: str) -> None:
    native = np.array([1.0, 2.0, 3.5], dtype=np.float64)
    swapped = native.astype(native.dtype.newbyteorder())
    assert swapped.dtype.isnative is False

    report = tt.compare(swapped, native, config=_config(backend))
    assert "non_native_byteorder_to_native" in report.reference.conversion_notes
    assert report.reference.dtype == "float64"
    assert report.metrics.mae == 0.0
    assert report.status == "pass"


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_zero_dimensional_scalar(backend: str) -> None:
    report = tt.compare(np.array(2.0), np.array(2.5), config=_config(backend))
    assert report.reference.shape == ()
    assert report.reference.elements == 1
    assert report.metrics.mae == pytest.approx(0.5)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_empty_multidimensional_tensor(backend: str) -> None:
    empty = np.empty((0, 3), dtype=np.float64)
    report = tt.compare(empty, empty.copy(), config=_config(backend))
    assert report.reference.shape == (0, 3)
    assert report.reference.elements == 0
    assert report.metrics.mae is None
    assert report.statistics is not None
    assert report.statistics.reference.elements == 0


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_float16_is_widened_to_float32(backend: str) -> None:
    reference = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    candidate = np.array([1.0, 2.5, 3.0], dtype=np.float16)
    report = tt.compare(reference, candidate, config=_config(backend))
    assert report.reference.dtype == "float16"
    assert report.reference.working_dtype == "float32"
    assert "float16_to_float32" in report.reference.conversion_notes
    assert report.metrics.mae == pytest.approx(0.5 / 3)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_bool_inputs_are_numeric(backend: str) -> None:
    reference = np.array([True, False, True, True])
    candidate = np.array([True, False, False, True])
    report = tt.compare(reference, candidate, config=_config(backend))
    assert report.reference.dtype == "bool"
    assert report.reference.working_dtype == "float64"
    assert "bool_to_float64" in report.reference.conversion_notes
    assert report.metrics.mae == pytest.approx(0.25)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_integer_dtypes_are_supported(backend: str) -> None:
    reference = np.array([1, 2, 3], dtype=np.int32)
    candidate = np.array([1, 4, 3], dtype=np.int32)
    report = tt.compare(reference, candidate, config=_config(backend))
    assert report.reference.dtype == "int32"
    assert report.reference.working_dtype == "float64"
    assert "int32_to_float64" in report.reference.conversion_notes
    assert report.metrics.mae == pytest.approx(2 / 3)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_wide_integer_conversion_is_flagged(backend: str) -> None:
    values = np.array([1, 2, 3], dtype=np.int64)
    report = tt.compare(values, values, config=_config(backend))
    assert "integer_to_float64_may_lose_precision_above_2^53" in (
        report.reference.conversion_notes
    )

    unsigned = values.astype(np.uint64)
    report = tt.compare(unsigned, unsigned, config=_config(backend))
    assert "integer_to_float64_may_lose_precision_above_2^53" in (
        report.reference.conversion_notes
    )


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_mixed_dtypes_upcast_to_float64(backend: str) -> None:
    reference = np.array([1.0, 2.0], dtype=np.float32)
    candidate = np.array([1.0, 2.0], dtype=np.float64)
    report = tt.compare(reference, candidate, config=_config(backend))
    assert report.metadata.working_dtype == "float64"
    assert report.reference.working_dtype == "float64"
    assert "upcast_to_float64_for_mixed_dtypes" in report.reference.conversion_notes


@pytest.mark.parametrize(
    "array",
    [
        np.array([1 + 2j], dtype=np.complex64),
        np.array([1 + 2j], dtype=np.complex128),
        np.array(["a", "b"]),
        np.array([b"a"], dtype=np.bytes_),
        np.array(["2026-01-01"], dtype="datetime64[D]"),
        np.array([1], dtype="timedelta64[s]"),
    ],
    ids=[
        "complex64",
        "complex128",
        "unicode",
        "bytes",
        "datetime64",
        "timedelta64",
    ],
)
def test_unsupported_dtypes_are_rejected(array: np.ndarray) -> None:
    with pytest.raises(tt.UnsupportedDtypeError, match="unsupported dtype"):
        tt.compare(array, array.copy())


@pytest.mark.skipif(
    np.dtype(np.longdouble).itemsize <= 8,
    reason="longdouble is float64 on this platform",
)
def test_extended_precision_is_rejected() -> None:
    values = np.array([1.0], dtype=np.longdouble)
    with pytest.raises(tt.UnsupportedDtypeError, match="extended precision"):
        tt.compare(values, values.copy())


def test_object_arrays_are_rejected() -> None:
    with pytest.raises(tt.UnsupportedDtypeError, match="object array"):
        tt.compare(np.array([1, "two"], dtype=object), np.array([1, 2], dtype=object))


def test_masked_arrays_are_rejected_instead_of_ignoring_masks() -> None:
    masked = np.ma.array([1.0, 2.0], mask=[False, True])
    with pytest.raises(tt.TensorValidationError, match="masked array"):
        tt.compare(masked, masked.copy())


def test_ragged_sequences_are_rejected() -> None:
    with pytest.raises(tt.TensorValidationError, match="could not interpret"):
        tt.compare([[1.0, 2.0], [3.0]], [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_lists_are_accepted_as_array_like(backend: str) -> None:
    report = tt.compare([1, 2, 3], (1.0, 4.0, 3.0), config=_config(backend))
    assert report.reference.source == "array-like"
    assert report.reference.dtype == np.asarray([1, 2, 3]).dtype.name
    assert report.metrics.mae == pytest.approx(2 / 3)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_shape_mismatch_reports_both_shapes(backend: str) -> None:
    with pytest.raises(tt.TensorShapeError) as raised:
        tt.compare(
            np.zeros((2, 3)),
            np.zeros((3, 2)),
            config=_config(backend),
        )
    message = str(raised.value)
    assert "(2, 3)" in message
    assert "(3, 2)" in message


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_inputs_are_not_mutated(backend: str) -> None:
    reference = np.arange(12, dtype=np.float64).reshape(3, 4)
    candidate = reference + 1.0
    reference_before = reference.copy()
    candidate_before = candidate.copy()
    tt.compare(reference, candidate, config=_config(backend))
    np.testing.assert_array_equal(reference, reference_before)
    np.testing.assert_array_equal(candidate, candidate_before)


@pytest.mark.parametrize("backend", BACKEND_PARAMS)
def test_byte_accounting_uses_working_buffer(backend: str) -> None:
    values = np.arange(6, dtype=np.float16)
    report = tt.compare(values, values, config=_config(backend))
    assert report.reference.elements == 6
    assert report.reference.nbytes == 6 * np.dtype(np.float32).itemsize
