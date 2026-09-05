"""High-level, UI-independent RVC conversion engine."""

from __future__ import annotations

import os
import time
import gc
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from configs.config import Config
from infer.audio import wav2
from infer.vc.modules import VC
from infer.vc.singing import normalize_inference_mode
from rvc_core.device import memory_snapshot, require_mps_module, synchronize


CORE_ROOT = Path(__file__).resolve().parent.parent
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
OUTPUT_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a"}


class RVCEngine:
    def __init__(self, precision: str = "float32"):
        os.environ.setdefault("weight_root", str(CORE_ROOT / "assets" / "weights"))
        os.environ.setdefault("index_root", str(CORE_ROOT / "logs"))
        os.environ.setdefault("outside_index_root", str(CORE_ROOT / "assets" / "indices"))
        os.environ.setdefault("rmvpe_root", str(CORE_ROOT / "assets" / "rmvpe"))
        self.config = Config(precision=precision, strict=True)
        self.vc = VC(self.config)
        self.model_path: Path | None = None

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.config.backend,
            "device": str(self.config.device),
            "device_name": self.config.device_name,
            "dtype": str(self.config.dtype).replace("torch.", ""),
            "strict_gpu": True,
            "cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            "probe_ms": round(self.config.device_probe_ms, 3),
            "model": str(self.model_path) if self.model_path else None,
            "memory": memory_snapshot(),
            "assets": self.asset_status(),
        }

    @staticmethod
    def asset_status() -> dict[str, bool]:
        hubert = CORE_ROOT / "assets" / "hubert_base"
        return {
            "hubert": (hubert / "config.json").is_file()
            and ((hubert / "model.safetensors").is_file() or (hubert / "pytorch_model.bin").is_file()),
            "rmvpe": (CORE_ROOT / "assets" / "rmvpe" / "rmvpe.pt").is_file(),
        }

    @staticmethod
    def inspect_model(path_value: str) -> dict[str, Any]:
        path = Path(path_value).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".pth":
            raise FileNotFoundError(f"RVC .pth model not found: {path}")
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        embedding = checkpoint.get("weight", {}).get("emb_g.weight")
        if embedding is None:
            raise ValueError("model has no weight/emb_g.weight")
        speaker_count = int(embedding.shape[0])
        speakers = []
        for item in checkpoint.get("speaker_info", []):
            try:
                speaker_id, name = int(item["id"]), str(item["name"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= speaker_id < speaker_count and name:
                speakers.append({"id": speaker_id, "name": name})
        speakers.sort(key=lambda value: value["id"])
        return {
            "path": str(path),
            "name": path.name,
            "version": checkpoint.get("version", "v1"),
            "uses_f0": bool(checkpoint.get("f0", 1)),
            "sample_rate": int(checkpoint["config"][-1]),
            "speaker_count": speaker_count,
            "speakers": speakers,
        }

    def load_model(self, path_value: str) -> dict[str, Any]:
        metadata = self.inspect_model(path_value)
        path = Path(metadata["path"])
        os.environ["weight_root"] = str(path.parent)
        self.vc.get_vc(path.name)
        require_mps_module(self.vc.net_g, "RVC generator")
        synchronize(self.config.device)
        self.model_path = path
        return metadata

    def unload_model(self) -> dict[str, Any]:
        self.vc.net_g = None
        self.vc.hubert_model = None
        self.vc.pipeline = None
        self.vc.cpt = None
        self.model_path = None
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        synchronize(self.config.device)
        return {"unloaded": True, "memory": memory_snapshot()}

    def convert_batch(self, request: dict[str, Any], progress=None) -> dict[str, Any]:
        input_value = Path(str(request["input"])).expanduser().resolve()
        output_dir = Path(str(request["output_dir"])).expanduser().resolve()
        if input_value.is_file():
            inputs = [input_value]
        elif input_value.is_dir():
            inputs = sorted(path for path in input_value.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES)
        else:
            raise FileNotFoundError(f"批量输入不存在：{input_value}")
        if not inputs:
            raise RuntimeError("批量输入中没有支持的音频")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_format = str(request.get("format", "wav")).lower()
        if f".{output_format}" not in OUTPUT_SUFFIXES:
            raise ValueError("批量输出格式必须是 wav、flac、mp3 或 m4a")
        completed, failed = [], []
        for index, input_path in enumerate(inputs, 1):
            child = dict(request)
            child["input"] = str(input_path)
            child["output"] = str(output_dir / f"{input_path.stem}_rvc.{output_format}")
            child["overwrite"] = bool(request.get("overwrite", True))
            if progress:
                progress(index, len(inputs), input_path.name)
            try:
                completed.append(self.convert(child)["output"])
            except Exception as exc:
                failed.append({"input": str(input_path), "error": str(exc)})
        return {"total": len(inputs), "completed": completed, "failed": failed, "output_dir": str(output_dir), "gpu_verified": True}

    def convert(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = normalize_inference_mode(request.get("mode", "speech"))
        model = Path(str(request["model"])).expanduser().resolve()
        if self.model_path != model:
            metadata = self.load_model(str(model))
        else:
            metadata = self.inspect_model(str(model))

        input_path = Path(str(request["input"])).expanduser().resolve()
        output_path = Path(str(request["output"])).expanduser().resolve()
        if not input_path.is_file() or input_path.suffix.lower() not in AUDIO_SUFFIXES:
            raise FileNotFoundError(f"supported input audio not found: {input_path}")
        if output_path.suffix.lower() not in OUTPUT_SUFFIXES:
            raise ValueError("output must end in .wav, .flac, .mp3, or .m4a")
        if output_path.exists() and not bool(request.get("overwrite", False)):
            raise FileExistsError(f"output exists: {output_path}")

        index_value = str(request.get("index") or "").strip()
        if index_value:
            index_path = Path(index_value).expanduser().resolve()
            if not index_path.is_file():
                raise FileNotFoundError(f"FAISS index not found: {index_path}")
            index_value = str(index_path)

        assets = self.asset_status()
        if not assets["hubert"]:
            raise FileNotFoundError(f"HuBERT assets are missing under {CORE_ROOT / 'assets' / 'hubert_base'}")
        f0_method = str(request.get("f0_method", "rmvpe"))
        if f0_method == "rmvpe" and not assets["rmvpe"]:
            raise FileNotFoundError(f"RMVPE weight is missing: {CORE_ROOT / 'assets' / 'rmvpe' / 'rmvpe.pt'}")

        started = time.perf_counter()
        status, result = self.vc.vc_single(
            int(request.get("speaker_id", 0)),
            str(input_path),
            int(request.get("pitch", 0)),
            f0_method,
            index_value,
            float(request.get("index_rate", 0.75)),
            int(request.get("resample_sr", 0)),
            float(request.get("rms_mix_rate", 1.0)),
            float(request.get("protect", 0.33)),
            mode=mode,
        )
        synchronize(self.config.device)
        if not result or result[0] is None or result[1] is None:
            raise RuntimeError(status)

        require_mps_module(self.vc.net_g, "RVC generator")
        require_mps_module(self.vc.hubert_model, "HuBERT model")
        if f0_method == "rmvpe" and hasattr(self.vc.pipeline, "model_rmvpe"):
            require_mps_module(self.vc.pipeline.model_rmvpe.model, "RMVPE model")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_audio(output_path, np.asarray(result[1]), int(result[0]))
        return {
            "output": str(output_path),
            "sample_rate": int(result[0]),
            "duration_seconds": round(len(result[1]) / int(result[0]), 3),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model": metadata,
            "device": "mps",
            "gpu_verified": True,
            "memory": memory_snapshot(),
        }

    @staticmethod
    def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        suffix = path.suffix.lower()
        if suffix in {".wav", ".flac"}:
            sf.write(str(path), audio, sample_rate)
            return
        with BytesIO() as buffer:
            sf.write(buffer, audio, sample_rate, format="wav")
            buffer.seek(0)
            with path.open("wb") as target:
                wav2(buffer, target, suffix.lstrip("."))
