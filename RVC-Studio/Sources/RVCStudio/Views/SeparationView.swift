import SwiftUI

struct SeparationView: View {
    @ObservedObject var model: AppModel
    @State private var confirmsDeletion = false

    var body: some View {
        PageColumn {
            InfoBanner(symbol: "sparkles", text: "PyMSS 已内置在 RVC-Core。此页面使用 MLX 全模型后端，权重下载后可完全离线运行。")

            StudioCard("分离模型") {
                LabeledField("模型") {
                    Picker("模型", selection: $model.separationModelID) {
                        ForEach(model.separationModels) { item in
                            Text(item.name).tag(item.id)
                        }
                    }
                    .labelsHidden()
                    .onChange(of: model.separationModelID) { _, _ in model.refreshSeparationModels() }
                }
                HStack {
                    Label(statusText, systemImage: model.selectedSeparationModel.installed
                          ? "checkmark.circle.fill" : "arrow.down.circle")
                        .font(.subheadline)
                        .foregroundStyle(model.selectedSeparationModel.installed ? .green : .secondary)
                        .help(model.separationModelDirectory)
                    Spacer()
                    Button("下载模型", systemImage: "arrow.down.circle") { model.downloadSeparationModel() }
                        .disabled(model.selectedSeparationModel.installed || model.taskBusy || !model.isReady)
                    Button("删除模型", systemImage: "trash", role: .destructive) { confirmsDeletion = true }
                        .disabled(!model.selectedSeparationModel.installed || model.taskBusy)
                }
            }

            StudioCard("输入与输出") {
                PathField(title: "输入音频 / 文件夹", path: $model.separationInputPath, kind: .audioOrDirectory)
                PathField(title: "主音轨输出", path: $model.separationPrimaryPath, kind: .directory)
                PathField(title: "次音轨输出", path: $model.separationSecondaryPath, kind: .directory)
                HStack {
                    FieldLabel("输出格式")
                    Picker("输出格式", selection: $model.separationFormat) {
                        ForEach(["wav", "flac", "mp3", "m4a"], id: \.self) { Text($0).tag($0) }
                    }
                    .labelsHidden()
                    .frame(width: 120)
                    Spacer()
                    Button("开始 MLX 分离", systemImage: "square.split.2x1.fill") { model.separate() }
                        .primaryGlassButton()
                        .disabled(!model.selectedSeparationModel.installed || model.taskBusy || !model.isReady)
                }
            }
        }
        .confirmationDialog("删除已下载的分离模型？", isPresented: $confirmsDeletion) {
            Button("删除模型", role: .destructive) { model.deleteSeparationModel() }
        } message: {
            Text("之后可以重新下载，不会影响 RVC 音色模型。")
        }
    }

    private var statusText: String {
        let item = model.selectedSeparationModel
        let mib = item.sizeBytes / 1_048_576
        return "\(item.installed ? "已下载，可离线使用" : "未下载") · \(mib) MiB"
    }
}
