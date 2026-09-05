# RVC Studio

面向 Apple Silicon macOS 的原生 RVC Studio：包含 SwiftUI 桌面界面、RVC-Core 训练/推理流程、MPS 训练支持，以及自动最佳模型选择和 Early Stopping。

## 已包含的运行依赖

`RVC-Core/runtime/ffmpeg/` 包含 arm64 版 `ffmpeg` 与 `ffprobe`，供应用直接调用。

## 未包含的内容

仓库只提交代码与可再发布的运行依赖，不提交任何模型权重、索引、训练数据、训练日志、音频、checkpoint 或 Python 虚拟环境。首次使用时，请按 RVC-Core 的说明自行下载所需模型。

## 构建 macOS App

```zsh
cd RVC-Studio
./build_macos.sh
```

生成的应用位于 `RVC-Studio/build/RVC-Studio.app`。
