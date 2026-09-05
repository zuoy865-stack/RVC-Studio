import SwiftUI

struct CheckpointView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        PageColumn {
            StudioCard("模型检查点工具") {
                LabeledField("操作") {
                    Picker("操作", selection: $model.checkpointAction) {
                        Text("查看模型信息").tag("inspect")
                        Text("修改模型信息").tag("edit")
                        Text("提取小模型").tag("extract")
                        Text("融合两个模型").tag("merge")
                    }.labelsHidden()
                }
                PathField(title: "模型 / 检查点 1", path: $model.checkpointPath1, kind: .pytorch)
                PathField(title: "模型 2（融合）", path: $model.checkpointPath2, kind: .pytorch)
                    .disabled(model.checkpointAction != "merge")
            }

            StudioCard("输出与元数据") {
                Grid(horizontalSpacing: 18, verticalSpacing: 14) {
                    GridRow {
                        LabeledField("输出名称（无需 .pth）") {
                            TextField("模型名称", text: $model.checkpointName).textFieldStyle(.roundedBorder)
                        }
                        LabeledField("模型说明") {
                            TextField("说明", text: $model.checkpointInfo).textFieldStyle(.roundedBorder)
                        }
                    }
                    GridRow {
                        LabeledField("模型 1 权重占比") {
                            RatioControl(value: $model.checkpointAlpha)
                        }
                        LabeledField("采样率") {
                            Picker("采样率", selection: $model.checkpointRate) {
                                ForEach(["32k", "40k", "48k"], id: \.self) { Text($0).tag($0) }
                            }.labelsHidden()
                        }
                    }
                    GridRow {
                        LabeledField("版本") {
                            Picker("版本", selection: $model.checkpointVersion) {
                                Text("v2").tag("v2")
                                Text("v1").tag("v1")
                            }.labelsHidden()
                        }
                        Toggle("带 F0", isOn: $model.checkpointUsesF0)
                    }
                }
                HStack {
                    Spacer()
                    Button("执行操作", systemImage: "play.fill") { model.runCheckpointTool() }
                        .primaryGlassButton()
                        .disabled(model.taskBusy || !model.isReady || model.checkpointPath1.isEmpty)
                }
            }
        }
    }
}
