import SwiftUI

struct SettingsView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        PageColumn {
            StudioCard("运行环境", subtitle: "RVC 推理和训练严格使用 MPS；CPU fallback 默认关闭。修改路径后重新连接即可生效。") {
                PathField(title: "RVC-Core", path: $model.corePath, kind: .directory)
                PathField(title: "Python", path: $model.pythonPath, kind: .anyFile)
                HStack {
                    FieldLabel("计算精度")
                    Picker("计算精度", selection: $model.precision) {
                        Text("Float32 · 稳定").tag("float32")
                        Text("Float16 · 实验").tag("float16")
                    }
                    .labelsHidden()
                    .frame(width: 180)
                    Spacer()
                    Button(model.isReady ? "重新连接后端" : "启动并验证 GPU",
                           systemImage: "bolt.horizontal.circle.fill") {
                        model.startBackend()
                    }
                    .primaryGlassButton()
                    .disabled(model.taskBusy || model.backendState == .connecting)
                }
            }

            StudioCard("原生架构") {
                VStack(alignment: .leading, spacing: 12) {
                    architectureRow("SwiftUI", detail: "导航、表单、状态与 Liquid Glass", symbol: "swift")
                    Divider()
                    architectureRow("AppKit", detail: "NSWindow、系统文件面板与 macOS 行为", symbol: "macwindow")
                    Divider()
                    architectureRow("RVC-Core", detail: "PyTorch / MPS 推理训练与 PyMSS / MLX 分离", symbol: "terminal")
                }
            }

            StudioCard("系统外观") {
                HStack(spacing: 12) {
                    Image(systemName: "circle.lefthalf.filled")
                        .font(.title2)
                        .foregroundStyle(.tint)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("跟随 macOS 外观").fontWeight(.medium)
                        Text("在 macOS 26 使用系统 Liquid Glass；旧版本自动使用原生 Material。")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func architectureRow(_ title: String, detail: String, symbol: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: symbol).frame(width: 26).foregroundStyle(.tint)
            Text(title).fontWeight(.medium).frame(width: 90, alignment: .leading)
            Text(detail).foregroundStyle(.secondary)
        }
    }
}
