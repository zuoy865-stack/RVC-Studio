# RVC Studio(目前还处于实验测试阶段)

面向 Apple Silicon macOS 的原生 RVC Studio：包含 SwiftUI 桌面界面、RVC-Core 训练/推理流程、MPS 训练支持，以及自动最佳模型选择和 Early Stopping。

## 已包含的运行依赖

`RVC-Core/runtime/ffmpeg/` 包含 arm64 版 `ffmpeg` 与 `ffprobe`，供应用直接调用。



## 构建 macOS App

```zsh
cd RVC-Studio
./build_macos.sh
```

生成的应用位于 `RVC-Studio/build/RVC-Studio.app`。


音频分离模型算法参考的 https://github.com/pymss-project/pymss-studio
