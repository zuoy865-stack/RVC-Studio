# RVC Studio for macOS

Native SwiftUI + AppKit frontend for `RVC-Core`. The default build no longer
depends on Qt, CMake, or `macdeployqt`. On macOS 26 it uses the system Liquid
Glass compositor; macOS 15 and newer use a native Material fallback.

The app embeds no browser, Gradio, HTTP server, RVC implementation, or PyMSS
implementation. Swift launches the core as a child process and communicates over
private newline-delimited JSON pipes.

The UI includes:

- RVC model refresh, inspection, speaker selection, unload and GPU-cache release;
- single-file conversion, native output playback, and recursive batch conversion;
- pitch, RMVPE/Praat, FAISS index ratio, volume envelope, consonant protection,
  resampling, and batch output format;
- bundled PyMSS/MLX separation code with five selectable models, explicit
  per-model download/delete state, and separately routed primary/secondary stems;
- dataset preprocessing, MPS RMVPE and HuBERT extraction, v1/v2 model training,
  single/multiple speakers, index training, and one-click sequencing;
- checkpoint inspection, metadata editing, small-model extraction, and merging;
- cancellable background jobs and streamed logs.

Development build:

```zsh
chmod +x build_macos.sh
./build_macos.sh
open build/RVC-Studio.app
```

Requirements: Xcode 26 or newer and macOS 15 or newer. The Liquid Glass treatment
appears automatically on macOS 26.

`package_macos.sh` additionally packages `RVC-Core`, a portable arm64 Python
runtime, and Python dependencies under `RVC-Core/runtime`. It expects
`RVC_PORTABLE_PYTHON` to point to an extracted python-build-standalone directory;
the resulting app does not require Python or PyMSS to be installed on the target Mac.

PyMSS checkpoints are deliberately not shipped inside the app. The download and
delete buttons manage individual models under
`~/Library/Application Support/RVC-Studio/models/pymss`.

`build_macos.sh` and `Package.swift` are the supported build entry points. The
former Qt/CMake frontend and its generated build artifacts have been removed.
