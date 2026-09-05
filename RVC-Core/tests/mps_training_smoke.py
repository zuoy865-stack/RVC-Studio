"""One real generator/discriminator forward+backward step on strict MPS."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from infer.module.models import MultiPeriodDiscriminatorV2, SynthesizerTrnMs768NSFsid
from rvc_core.device import require_mps_module, select_mps, synchronize

config = json.loads((root / "configs/v2/32k.json").read_text())
device = select_mps("float32", strict=True).device
frames = 48
model = SynthesizerTrnMs768NSFsid(
    config["data"]["filter_length"] // 2 + 1,
    config["train"]["segment_size"] // config["data"]["hop_length"],
    **config["model"], is_half=False, sr="32k",
).to(device)
discriminator = MultiPeriodDiscriminatorV2(False).to(device)
require_mps_module(model, "training smoke generator")
require_mps_module(discriminator, "training smoke discriminator")
phone = torch.randn(1, frames, 768, device=device)
lengths = torch.tensor([frames], device=device)
pitch = torch.randint(1, 255, (1, frames), device=device)
pitchf = torch.full((1, frames), 220.0, device=device)
spec = torch.randn(1, config["data"]["filter_length"] // 2 + 1, frames, device=device)
speaker = torch.zeros(1, dtype=torch.long, device=device)
output = model(phone, lengths, pitch, pitchf, spec, lengths, speaker)[0]
real = torch.randn_like(output)
real_scores, fake_scores, _, _ = discriminator(real, output.detach())
loss_d = sum(value.float().mean() for value in real_scores + fake_scores)
loss_d.backward()
fake_scores = discriminator(real, output)[1]
loss_g = sum(value.float().mean() for value in fake_scores) + output.abs().mean()
loss_g.backward()
synchronize(device)
print({"device": str(device), "output_shape": list(output.shape), "generator_grad": any(p.grad is not None for p in model.parameters()), "discriminator_grad": any(p.grad is not None for p in discriminator.parameters())})
