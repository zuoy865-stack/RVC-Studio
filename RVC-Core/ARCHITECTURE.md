# RVC Studio / Core architecture

```text
RVC-Studio (SwiftUI + AppKit / Swift)
  model management + inference + separation + training + ckpt tools
                 |
                 | stdin/stdout JSON Lines via Process pipes (no HTTP/WebUI)
                 v
RVC-Core (Python)
  request validation + cancellable task orchestration
       |                                  |
       v                                  v
  HuBERT -> RMVPE -> RVC             bundled PyMSS
  inference/training (strict MPS)     separation (MLX full model)

  CPU utility work only: decode/encode, DSP slicing, FAISS indexing
```

## Why MPS first, not MLX first

The existing RVC checkpoints are PyTorch `.pth` files and the pipeline combines a
Transformers HuBERT model, RMVPE, and four RVC generator variants. PyTorch MPS preserves
these model definitions and weight compatibility. A complete MLX backend would require
independent MLX implementations and numerical parity tests for all three networks; merely
copying tensors between PyTorch and MLX would add synchronization/copy overhead without
making the whole inference graph MLX-native.

PyMSS is bundled as the separation implementation: it exposes `mps` and `mlx` as explicit
backends, ports supported model forward passes to MLX, caches compiled functions, and
uses float16 compute. Its model weights are downloaded individually into the user's
application data directory rather than shipped in the app. For RVC, MLX should only become the
default after a full-network port passes checkpoint compatibility, audio parity, and
end-to-end benchmark gates on representative M1/M2/M3/M4 hardware.

## Acceptance gates for an MLX phase

1. HuBERT, RMVPE, and all RVC v1/v2 F0/non-F0 generators run without PyTorch model ops.
2. Existing `.pth` checkpoints convert automatically and reproducibly.
3. Output similarity passes a documented tolerance against float32 MPS.
4. Warm and cold end-to-end latency and peak unified memory beat MPS on real audio.
5. No CPU model fallback is permitted; backend selection remains visible in the UI.
