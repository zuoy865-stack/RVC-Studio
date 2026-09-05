"""Self-contained PyMSS worker that routes the two output stems separately."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")

from pymss import create_separator
from rvc_core.device import select_mps

MODEL_STEMS = {
    "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt": ("noreverb", "reverb"),
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt": ("noreverb", "reverb"),
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt": ("Vocals", "Instrumental"),
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt": ("Vocals", "Instrumental"),
    "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt": ("vocals", "instrumental"),
}


def mlx_probe():
    select_mps(precision="float32", strict=True)
    import mlx.core as mx
    value = mx.arange(4096, dtype=mx.float32).reshape(64, 64) @ mx.eye(64, dtype=mx.float32)
    mx.eval(value)
    if float(value[63, 63].item()) != 4095.0:
        raise RuntimeError("MLX Metal kernel probe returned an invalid result")
    print("MLX Metal kernel probe passed", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODEL_STEMS))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--primary-output", required=True)
    parser.add_argument("--secondary-output", required=True)
    parser.add_argument("--format", default="wav", choices=("wav", "flac", "mp3", "m4a"))
    args = parser.parse_args(argv)
    mlx_probe()
    for path in (args.primary_output, args.secondary_output):
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)
    primary, secondary = MODEL_STEMS[args.model]
    store_dirs = {primary: args.primary_output, secondary: args.secondary_output}
    with create_separator(
        args.model, model_dir=args.model_dir, device="mlx", output_format=args.format,
        store_dirs=store_dirs, save_as_folder=False,
        inference_params={"mps_model_backend": "mlx_full", "mps_model_compute_dtype": "float16", "mps_mlx_clear_cache": True},
    ) as separator:
        backend = str(separator.config.inference.get("mps_model_backend", ""))
        if backend != "mlx_full" or str(separator.device) != "mps":
            raise RuntimeError(f"PyMSS did not select strict MLX/MPS: backend={backend}, device={separator.device}")
        files = separator.process_folder(args.input)
    print(f"PyMSS MLX separation completed: {len(files)} file(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
