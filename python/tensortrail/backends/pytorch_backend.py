"""Optional PyTorch input support.

PyTorch is never imported at package import time. A tensor is recognized only
when the caller supplies one; the dependency is imported lazily in that case.

CUDA/MPS tensors are rejected rather than copied to the host and presented as
GPU comparison. GPU-native metrics are a separate future backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BackendError, UnsupportedDtypeError

__all__ = [
    "TorchConversion",
    "TorchDescription",
    "describe_torch_tensor",
    "looks_like_torch_tensor",
    "torch_tensor_to_numpy",
]


@dataclass(frozen=True, slots=True)
class TorchConversion:
    """Result of an explicit, recorded PyTorch-to-NumPy conversion."""

    array: NDArray[Any]
    original_dtype: str
    device: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TorchDescription:
    """Metadata read from a CPU PyTorch tensor without converting its payload."""

    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    device: str


def looks_like_torch_tensor(value: object) -> bool:
    """Detect torch tensors without importing torch for ordinary inputs."""
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def describe_torch_tensor(value: object, role: str) -> TorchDescription | None:
    """Describe a CPU tensor without copying or converting its payload.

    Returns ``None`` when ``value`` is not a PyTorch tensor. Accelerator,
    quantized, and non-strided tensors are rejected with the same policy as
    comparison.
    """
    if not looks_like_torch_tensor(value):
        return None

    try:
        import torch  # deliberately lazy and optional
    except ImportError as error:  # pragma: no cover - defensive
        raise BackendError(
            "received a PyTorch tensor, but PyTorch is not installed in this environment"
        ) from error

    if not isinstance(value, torch.Tensor):
        return None

    if value.device.type != "cpu":
        raise BackendError(
            f"{role} is on device {value.device!s}; TensorTrail does not silently copy GPU "
            "or accelerator tensors to CPU. Move it explicitly with `.detach().cpu()` and "
            "record the conversion before registering the checkpoint."
        )
    if value.is_quantized:
        raise UnsupportedDtypeError(
            f"{role} is a quantized PyTorch tensor; dequantize it explicitly and record the "
            "conversion before registering the checkpoint"
        )
    if value.layout != torch.strided:
        raise UnsupportedDtypeError(
            f"{role} has layout {value.layout}; only dense strided CPU tensors are supported"
        )

    return TorchDescription(
        shape=tuple(int(dimension) for dimension in value.shape),
        dtype=str(value.dtype).removeprefix("torch."),
        nbytes=int(value.numel() * value.element_size()),
        device="cpu",
    )


def torch_tensor_to_numpy(value: object, role: str) -> TorchConversion | None:
    """Convert a CPU PyTorch tensor to NumPy, or return ``None`` if not a tensor.

    Raises
    ------
    BackendError
        For GPU/MPS tensors or unsupported tensor layouts. TensorTrail does not
        silently transfer device memory or claim GPU acceleration.
    UnsupportedDtypeError
        For complex or otherwise unsupported dtypes.
    """
    if not looks_like_torch_tensor(value):
        return None

    try:
        import torch  # deliberately lazy and optional
    except ImportError as error:  # pragma: no cover - defensive
        raise BackendError(
            "received a PyTorch tensor, but PyTorch is not installed in this environment"
        ) from error

    if not isinstance(value, torch.Tensor):
        return None

    if value.device.type != "cpu":
        raise BackendError(
            f"{role} is on device {value.device!s}; TensorTrail does not silently copy GPU "
            "or accelerator tensors to CPU. Move it explicitly with `.detach().cpu()` and "
            "label the report accordingly. GPU-native comparison is planned, not implemented."
        )
    if value.is_quantized:
        raise UnsupportedDtypeError(
            f"{role} is a quantized PyTorch tensor; dequantize it explicitly and record the "
            "conversion before comparison"
        )
    if value.layout != torch.strided:
        raise UnsupportedDtypeError(
            f"{role} has layout {value.layout}; only dense strided CPU tensors are supported"
        )

    notes: list[str] = []
    tensor = value
    if tensor.requires_grad:
        tensor = tensor.detach()
        notes.append("torch_detached_from_autograd")

    dtype = tensor.dtype
    original_dtype = str(dtype).removeprefix("torch.")
    if dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.float()
        notes.append(f"torch_{original_dtype}_to_float32")
    elif dtype in (torch.float32, torch.float64):
        pass
    elif dtype == torch.bool or dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        tensor = tensor.to(torch.float64)
        notes.append(f"torch_{original_dtype}_to_float64")
    elif dtype.is_complex:
        raise UnsupportedDtypeError(
            f"{role} has complex dtype {dtype}; complex tensors are not supported"
        )
    else:
        raise UnsupportedDtypeError(
            f"{role} has unsupported PyTorch dtype {dtype}; supported dtypes are "
            "float32, float64, float16, bfloat16, float8 variants, integers, and bool"
        )

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
        notes.append("torch_non_contiguous_to_contiguous")

    try:
        array = tensor.numpy()
    except (TypeError, RuntimeError) as error:  # pragma: no cover - defensive
        raise BackendError(f"could not convert {role} to NumPy: {error}") from error

    return TorchConversion(
        array=np.asarray(array),
        original_dtype=original_dtype,
        device="cpu",
        notes=tuple(notes),
    )
