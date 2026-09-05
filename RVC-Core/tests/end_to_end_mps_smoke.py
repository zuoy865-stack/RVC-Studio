"""Run the complete file-to-file pipeline with a temporary synthetic checkpoint."""

import gc
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT))
os.chdir(CORE_ROOT)

import numpy as np
import soundfile as sf
import torch

from infer.module.models import SynthesizerTrnMs768NSFsid
from rvc_core.engine import RVCEngine


CONFIG = [
    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [12, 10, 2, 2], 512,
    [24, 20, 4, 4], 1, 256, 48000,
]


def main():
    with tempfile.TemporaryDirectory(prefix="rvc-studio-smoke-") as temporary:
        root = Path(temporary)
        model_path = root / "synthetic-v2.pth"
        input_path = root / "input.wav"
        output_path = root / "output.wav"

        model = SynthesizerTrnMs768NSFsid(*CONFIG, is_half=False)
        weights = {
            name: value.detach().half()
            for name, value in model.state_dict().items()
            if "enc_q" not in name
        }
        torch.save({
            "weight": weights,
            "config": CONFIG,
            "f0": 1,
            "version": "v2",
            "sr": "48k",
            "info": "synthetic MPS smoke fixture",
        }, model_path)
        del weights, model
        gc.collect()

        sample_rate = 16000
        seconds = 0.25
        samples = np.arange(int(sample_rate * seconds), dtype=np.float32)
        sf.write(input_path, 0.05 * np.sin(2 * np.pi * 220 * samples / sample_rate), sample_rate)

        engine = RVCEngine("float32")
        result = engine.convert({
            "model": str(model_path),
            "input": str(input_path),
            "output": str(output_path),
            "speaker_id": 0,
            "pitch": 0,
            "f0_method": "rmvpe",
            "index": "",
            "index_rate": 0,
            "rms_mix_rate": 1,
            "protect": 0.33,
            "resample_sr": 0,
            "overwrite": True,
        })
        assert result["gpu_verified"] is True
        assert output_path.is_file() and output_path.stat().st_size > 44
        print(result)


if __name__ == "__main__":
    main()
