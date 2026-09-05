"""Strict Apple Silicon GPU selection and runtime auditing."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")

import torch


class MPSUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MPSSelection:
    device: torch.device
    dtype: torch.dtype
    name: str
    probe_ms: float


def _precision_dtype(precision: str | None) -> torch.dtype:
    value = (precision or os.getenv("RVC_MPS_PRECISION", "float32")).lower()
    if value in {"float32", "fp32", "32"}:
        return torch.float32
    if value in {"float16", "fp16", "16"}:
        return torch.float16
    raise ValueError("precision must be float32 or float16")


def select_mps(precision: str | None = None, strict: bool = True) -> MPSSelection:
    reasons = []
    if platform.system() != "Darwin":
        reasons.append("macOS is required")
    if platform.machine() != "arm64":
        reasons.append("a native arm64 Python process is required (not Rosetta)")
    if not torch.backends.mps.is_built():
        reasons.append("this PyTorch build has no MPS support")
    if not torch.backends.mps.is_available():
        reasons.append("MPS is not available to this process")
    if reasons:
        message = "; ".join(reasons)
        if strict:
            raise MPSUnavailableError(message)
        return MPSSelection(torch.device("cpu"), torch.float32, "CPU", 0.0)

    device = torch.device("mps")
    dtype = _precision_dtype(precision)
    started = time.perf_counter()
    try:
        # Availability flags are not enough: execute and synchronize a Metal kernel.
        left = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64)
        right = torch.eye(64, device=device, dtype=torch.float32)
        result = torch.relu(left @ right)
        torch.mps.synchronize()
        if result.device.type != "mps" or float(result[63, 63].item()) != 4095.0:
            raise RuntimeError("MPS kernel returned an invalid result")
    except Exception as exc:
        raise MPSUnavailableError(f"MPS execution probe failed: {exc}") from exc
    probe_ms = (time.perf_counter() - started) * 1000.0
    return MPSSelection(device, dtype, "Apple Silicon GPU (Metal/MPS)", probe_ms)


def synchronize(device: Any = "mps") -> None:
    if torch.device(device).type == "mps":
        torch.mps.synchronize()


def require_mps_tensor(tensor: torch.Tensor, stage: str) -> None:
    if not torch.is_tensor(tensor) or tensor.device.type != "mps":
        actual = getattr(getattr(tensor, "device", None), "type", type(tensor).__name__)
        raise RuntimeError(f"{stage} left Apple GPU; actual device: {actual}")


def require_mps_module(module: torch.nn.Module, stage: str) -> None:
    devices = {value.device.type for value in module.parameters()}
    devices.update(value.device.type for value in module.buffers())
    if not devices:
        raise RuntimeError(f"{stage} has no auditable tensors")
    if devices != {"mps"}:
        raise RuntimeError(f"{stage} is not fully on Apple GPU: {sorted(devices)}")


def memory_snapshot() -> dict[str, int | None]:
    def read(name: str):
        function = getattr(torch.mps, name, None)
        return int(function()) if function else None

    return {
        "current_allocated_bytes": read("current_allocated_memory"),
        "driver_allocated_bytes": read("driver_allocated_memory"),
        "recommended_max_bytes": read("recommended_max_memory"),
    }
