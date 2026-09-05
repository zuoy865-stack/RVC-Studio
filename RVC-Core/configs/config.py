"""Runtime configuration for the macOS RVC inference service.

This module intentionally has no CUDA/DirectML/CLI compatibility layer. The desktop
backend is Apple-Silicon-only and fails closed when MPS cannot execute a real kernel.
"""

import json
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

# PyTorch reads this when it initializes MPS. Unsupported operators must raise
# instead of silently executing on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")

import torch

from rvc_core.device import select_mps


CONFIGS_DIR = Path(__file__).resolve().parent
MODEL_CONFIG_FILES = (
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
)


def get_device_dtype_sm(_idx):
    """Compatibility shim for a legacy RMVPE import path."""
    return torch.device("mps"), torch.float32, 0.0, 0.0


infer_device = torch.device("mps")
infer_dtype = torch.float32


def get_training_dtype():
    """Training is deliberately FP32 on MPS until every RVC op is FP16-safe."""
    select_mps(precision="float32", strict=True)
    return torch.float32


class Config:
    """The small configuration surface consumed by the legacy RVC pipeline."""

    def __init__(self, precision=None, strict=True):
        selected = select_mps(precision=precision, strict=strict)
        self.device = selected.device
        self.dtype = selected.dtype
        self.is_half = selected.dtype == torch.float16
        self.device_name = selected.name
        self.device_probe_ms = selected.probe_ms
        self.backend = "pytorch-mps"
        self.strict_gpu = strict
        self.cuda_graph = False
        self.dml = False
        self.instead = ""
        self.n_cpu = max(1, cpu_count())
        self.python_cmd = sys.executable
        self.noparallel = False
        self.gpu_name = selected.name
        self.gpu_mem = None
        self.preprocess_per = 3.0
        self.json_config = self.load_config_json()
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json():
        return {
            name: json.loads((CONFIGS_DIR / name).read_text(encoding="utf-8"))
            for name in MODEL_CONFIG_FILES
        }

    def device_config(self):
        # Larger chunks reduce Python dispatch and help keep the GPU fed.
        if self.is_half:
            return 3, 10, 60, 65
        return 2, 8, 48, 52
