"""Exercise every neural stage on MPS without requiring a user voice model."""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_ROOT))
os.chdir(CORE_ROOT)

import torch

from infer.hubert import extract_hubert_features, load_hubert_model
from infer.module.models import SynthesizerTrnMs768NSFsid
from infer.rmvpe import RMVPE
from rvc_core.device import memory_snapshot, require_mps_module, require_mps_tensor


def main():
    hubert = load_hubert_model("mps", False)
    features = extract_hubert_features(
        hubert, torch.zeros(1, 16000, device="mps"), "v2"
    )
    require_mps_tensor(features, "HuBERT smoke")

    rmvpe = RMVPE("assets/rmvpe/rmvpe.pt", False, "mps")
    f0 = rmvpe.infer_from_audio(torch.zeros(16000))

    config = [
        1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
        [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [12, 10, 2, 2], 512,
        [24, 20, 4, 4], 1, 256, 48000,
    ]
    generator = SynthesizerTrnMs768NSFsid(*config, is_half=False)
    del generator.enc_q
    generator.eval().to("mps").float()
    require_mps_module(generator, "RVC generator smoke")
    length = 64
    with torch.inference_mode():
        output = generator.infer(
            torch.randn(1, length, 768, device="mps"),
            torch.tensor([length], device="mps").long(),
            torch.full((1, length), 128, device="mps").long(),
            torch.full((1, length), 220.0, device="mps"),
            torch.zeros(1, device="mps").long(),
        )[0]
    require_mps_tensor(output, "RVC generator smoke")
    torch.mps.synchronize()
    print({
        "hubert": list(features.shape),
        "rmvpe_frames": len(f0),
        "generator": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
        "memory": memory_snapshot(),
    })


if __name__ == "__main__":
    main()
