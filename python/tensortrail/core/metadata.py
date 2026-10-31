"""Input normalization and tensor provenance tracking.

TensorTrail accepts NumPy arrays, array-like objects, and CPU PyTorch tensors.
Conversions are explicit and recorded in :class:`TensorInfo`; the native engine
only ever receives C-contiguous, aligned, native-endian ``float32``/``float64``
buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backends.pytorch_backend import looks_like_torch_tensor, torch_tensor_to_numpy
from ..core.result import TensorInfo
from ..exceptions import TensorShapeError, TensorValidationError, UnsupportedDtypeError

__all__ = ["NormalizedTensor", "normalize_pair", "normalize_tensor"]

Array = NDArray[Any]


@dataclass(frozen=True, slots=True)
class NormalizedTensor:
    """A working buffer plus provenance metadata."""

    array: Array
    info: TensorInfo

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.array.dtype)


def _unsupported(role: str, dtype: object, detail: str = "") -> UnsupportedDtypeError:
    message = (
        f"{role} has unsupported dtype {dtype!s}. Supported inputs are float32, float64, "
        "float16, bfloat16, integers, and bool; other dtypes are converted explicitly or "
        "rejected so memory is never silently misinterpreted"
    )
    if detail:
        message = f"{message}. {detail}"
    return UnsupportedDtypeError(message)


def _to_working_dtype(
    array: Array,
    role: str,
    original_dtype: str,
    notes: list[str],
) -> Array:
    dtype = array.dtype
    name = dtype.name

    if dtype.kind in {"c", "O", "S", "U", "M", "m"}:
        raise _unsupported(role, dtype)

    if name == "bfloat16":
        array = array.astype(np.float32, copy=True)
        notes.append("bfloat16_to_float32")
    elif dtype.kind == "f" and dtype.itemsize == 2:
        # Check the item size so byte-swapped float16 is widened too.
        array = array.astype(np.float32, copy=True)
        notes.append("float16_to_float32")
    elif dtype.kind == "f" and dtype.itemsize in (4, 8):
        # float32/float64, including non-native byte orders handled below.
        pass
    elif dtype.kind in {"b", "i", "u"}:
        # Bool is converted to 0.0/1.0; integer arrays are compared in float64.
        array = array.astype(np.float64, copy=True)
        notes.append(f"{name}_to_float64")
        if dtype.kind in {"i", "u"} and dtype.itemsize == 8:
            notes.append("integer_to_float64_may_lose_precision_above_2^53")
    elif dtype.kind == "f":
        raise _unsupported(
            role,
            dtype,
            "Only float16/float32/float64 are supported; downcast extended precision "
            "explicitly if that is the desired comparison",
        )
    else:
        raise _unsupported(role, dtype, f"Original dtype: {original_dtype}")

    if not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="), copy=True)
        notes.append("non_native_byteorder_to_native")

    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
        notes.append("non_contiguous_to_contiguous")

    if not array.flags.aligned:
        array = array.copy(order="C")
        notes.append("unaligned_to_aligned")

    return array


def normalize_tensor(value: object, role: str, tensor_id: str) -> NormalizedTensor:
    """Normalize one input to a contiguous native-endian float buffer.

    Parameters
    ----------
    value:
        NumPy array, array-like numeric object, or CPU PyTorch tensor.
    role:
        ``"reference"`` or ``"candidate"``; used in actionable errors.
    tensor_id:
        Identifier recorded in the report.

    Raises
    ------
    TensorValidationError
        For invalid array-like inputs.
    UnsupportedDtypeError
        For complex, object, string, datetime, or extended-precision dtypes.
    BackendError
        For accelerator PyTorch tensors (never silently copied).
    """
    notes: list[str] = []
    source = "array-like"
    device = "cpu"
    original_dtype: str

    if looks_like_torch_tensor(value):
        conversion = torch_tensor_to_numpy(value, role)
        if conversion is not None:
            array = conversion.array
            original_dtype = conversion.original_dtype
            source = "torch"
            device = conversion.device
            notes.extend(conversion.notes)
        else:
            array = _as_array(value, role)
            original_dtype = array.dtype.name
    else:
        if isinstance(value, np.ma.MaskedArray):
            raise TensorValidationError(
                f"{role} is a masked array; TensorTrail does not silently ignore masks. "
                "Call `.filled(np.nan)` or remove the mask explicitly."
            )
        if isinstance(value, np.ndarray):
            source = "numpy"
        array = _as_array(value, role)
        original_dtype = array.dtype.name

    working = _to_working_dtype(array, role, original_dtype, notes)

    # Deduplicate while preserving order (a torch conversion plus a generic
    # layout fix must remain readable in the report).
    unique_notes = tuple(dict.fromkeys(notes))
    info = TensorInfo(
        id=tensor_id,
        shape=tuple(int(dimension) for dimension in working.shape),
        dtype=original_dtype,
        working_dtype=working.dtype.name,
        device=device,
        elements=int(working.size),
        nbytes=int(working.nbytes),
        source=source,
        conversion_notes=unique_notes,
    )
    return NormalizedTensor(array=working, info=info)


def _as_array(value: object, role: str) -> Array:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TensorValidationError(
            f"could not interpret {role} as a numeric array: {error}"
        ) from error
    if array.dtype.kind == "O":
        raise UnsupportedDtypeError(
            f"{role} produced an object array. Ragged sequences and mixed Python objects "
            "are not numeric tensors; pass a homogeneous numeric array."
        )
    return array


def normalize_pair(
    reference: object,
    candidate: object,
    *,
    reference_id: str = "reference",
    candidate_id: str = "candidate",
) -> tuple[NormalizedTensor, NormalizedTensor, str]:
    """Normalize two inputs, validate shapes, and select a common dtype.

    Returns the two working tensors and the name of the common working dtype.
    ``float32`` inputs stay ``float32`` when both sides are ``float32``;
    mixed/integer inputs use ``float64``.
    """
    normalized_reference = normalize_tensor(reference, "reference", reference_id)
    normalized_candidate = normalize_tensor(candidate, "candidate", candidate_id)

    if normalized_reference.info.shape != normalized_candidate.info.shape:
        raise TensorShapeError(
            "reference and candidate shapes must match exactly: "
            f"{normalized_reference.info.shape} vs {normalized_candidate.info.shape}. "
            "Reshape or slice explicitly before comparison."
        )

    if normalized_reference.array.dtype == normalized_candidate.array.dtype:
        return (
            normalized_reference,
            normalized_candidate,
            normalized_reference.array.dtype.name,
        )

    # Only float32/float64 reach this point. Upcast the float32 side so both
    # buffers share a monomorphized native kernel.
    target = np.dtype(np.float64)
    normalized_reference = _cast_common(normalized_reference, target)
    normalized_candidate = _cast_common(normalized_candidate, target)
    return normalized_reference, normalized_candidate, target.name


def _cast_common(tensor: NormalizedTensor, target: np.dtype[Any]) -> NormalizedTensor:
    if tensor.array.dtype == target:
        return tensor
    array = tensor.array.astype(target, copy=True)
    info = replace(
        tensor.info,
        working_dtype=target.name,
        nbytes=int(array.nbytes),
        conversion_notes=(*tensor.info.conversion_notes, "upcast_to_float64_for_mixed_dtypes"),
    )
    return NormalizedTensor(array=array, info=info)
