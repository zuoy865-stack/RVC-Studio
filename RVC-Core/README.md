# RVC-Core

`RVC-Core` is the UI-independent inference and training backend for Apple Silicon. It exposes a
newline-delimited JSON protocol over stdin/stdout and deliberately has no HTTP server,
Gradio dependency, or Qt dependency.

## GPU contract

- native `arm64` macOS only;
- PyTorch `mps` device only;
- `PYTORCH_ENABLE_MPS_FALLBACK=0` before PyTorch initialization;
- a synchronized Metal matrix-multiply probe must pass before the service becomes ready;
- HuBERT, RMVPE, and the RVC generator are audited to ensure every parameter/buffer is on MPS;
- the RVC generator/discriminator used for training are moved to and audited on MPS;
- bundled PyMSS source explicitly selects its MLX full-model backend after a real MLX Metal probe;
- unsupported GPU operations fail visibly instead of silently moving neural inference to CPU.

Audio decoding, filtering, dataset slicing, FAISS lookup/index construction, and final file encoding remain CPU-side DSP/I/O.
The neural inference stages remain on Apple GPU. `float32` is the compatibility default;
`float16` is available as an experimental performance mode.

## Setup

```zsh
chmod +x setup_macos.sh
./setup_macos.sh
```

Add required runtime assets:

```text
assets/hubert_base/config.json
assets/hubert_base/preprocessor_config.json
assets/hubert_base/pytorch_model.bin   (or model.safetensors)
assets/rmvpe/rmvpe.pt
assets/weights/your-model.pth
assets/indices/your-model.index        (optional)
```

The `pymss` and `pymss_core` source packages are included in `RVC-Core`; no system
PyMSS install is used. Separation checkpoints are not stored in this repository.
They are selected and downloaded individually by the native UI into the user's
application data directory.

The original RVC assets can be fetched from `lj1995/VoiceConversionWebUI`:

```zsh
runtime/python-env/bin/python -m pip install huggingface_hub
runtime/python-env/bin/hf download lj1995/VoiceConversionWebUI --include "hubert_base/*" --local-dir assets
runtime/python-env/bin/hf download lj1995/VoiceConversionWebUI rmvpe.pt --local-dir assets/rmvpe
```

Verify all three neural networks with real, synchronized MPS forwards:

```zsh
PYTORCH_ENABLE_MPS_FALLBACK=0 runtime/python-env/bin/python tests/mps_network_smoke.py
```

`tests/end_to_end_mps_smoke.py` also creates a temporary synthetic RVC v2 checkpoint
and exercises the complete audio-file pipeline. Its audio is intentionally meaningless;
the test exists to verify integration and device placement without shipping a voice model.

## Protocol

Start with `runtime/python-env/bin/python -u -m rvc_core.server`. The server emits a `ready` event.
Commands are JSON objects with `id` and `command`. The service implements device
status, RVC model listing/inspection/loading/unloading, single and batch conversion,
separation model listing/download/delete and MLX inference, preprocessing/feature/
model/index/one-click training tasks, ckpt tools, task status/cancellation, and shutdown.
Long operations emit `task` events so Qt stays responsive.
