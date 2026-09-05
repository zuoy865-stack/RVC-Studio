import SwiftUI

struct InferenceView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        PageColumn {
            StudioCard("音色模型", subtitle: "选择 .pth 音色，索引可选；模型信息会自动读取。") {
                PathMenuField(title: "RVC 模型", path: $model.modelPath,
                              choices: model.availableModels, kind: .model,
                              onCommit: model.inspectModel)
                HStack(spacing: 12) {
                    Text(model.modelInfo)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Spacer()
                    FieldLabel("说话人 ID")
                    Stepper(value: $model.speakerID, in: 0...max(0, model.speakerMaximum)) {
                        Text("\(model.speakerID)").monospacedDigit().frame(minWidth: 24)
                    }
                    Button("刷新", systemImage: "arrow.clockwise") { model.refreshModels() }
                    Button("释放显存", systemImage: "memorychip") { model.unloadModel() }
                }
                PathMenuField(title: "FAISS 特征索引（可选）", path: $model.indexPath,
                              choices: model.availableIndices, kind: .index)
            }

            StudioCard("推理参数") {
                Grid(horizontalSpacing: 18, verticalSpacing: 14) {
                    GridRow {
                        LabeledField("推理模式") {
                            Picker("推理模式", selection: $model.inferenceMode) {
                                Text("说话").tag("speech")
                                Text("唱歌").tag("singing")
                            }.labelsHidden()
                        }
                        LabeledField("变调", hint: "男转女通常 +12，女转男通常 -12") {
                            Stepper(value: $model.pitch, in: -48...48) {
                                Text("\(model.pitch) 半音").monospacedDigit()
                            }
                        }
                    }
                    GridRow {
                        LabeledField("音高提取") {
                            Picker("音高提取", selection: $model.f0Method) {
                                Text("RMVPE · MPS GPU").tag("rmvpe")
                                Text("Praat PM · CPU DSP").tag("pm")
                            }.labelsHidden()
                        }
                        LabeledField("检索特征占比") {
                            RatioControl(value: $model.indexRate)
                        }
                    }
                    GridRow {
                        LabeledField("音量包络") {
                            RatioControl(value: $model.rmsMixRate)
                        }
                        LabeledField("清辅音保护") {
                            RatioControl(value: $model.protect, range: 0...0.5)
                        }
                    }
                    GridRow {
                        LabeledField("输出重采样率（0 = 模型原采样率）") {
                            TextField("0", value: $model.resampleRate, format: .number)
                                .textFieldStyle(.roundedBorder)
                        }
                        Color.clear.frame(height: 1)
                    }
                }
            }

            StudioCard("转换") {
                Picker("转换模式", selection: $model.conversionMode) {
                    Text("单次推理").tag("single")
                    Text("批量推理").tag("batch")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 330)

                if model.conversionMode == "single" {
                    PathField(title: "输入音频", path: $model.inputPath, kind: .audio,
                              onPick: model.pickedInputAudio)
                    PathField(title: "输出音频", path: $model.outputPath, kind: .saveAudio)
                    HStack {
                        Spacer()
                        Button("播放输出", systemImage: "play.fill") { model.playOutput() }
                            .disabled(model.outputPath.isEmpty)
                        Button("开始转换", systemImage: "waveform.badge.plus") { model.convert() }
                            .primaryGlassButton()
                            .disabled(model.taskBusy || !model.isReady)
                    }
                } else {
                    PathField(title: "输入文件夹", path: $model.batchInputPath, kind: .directory)
                    PathField(title: "输出文件夹", path: $model.batchOutputPath, kind: .directory)
                    HStack {
                        FieldLabel("输出格式")
                        Picker("输出格式", selection: $model.batchFormat) {
                            ForEach(["wav", "flac", "mp3", "m4a"], id: \.self) { Text($0).tag($0) }
                        }
                        .labelsHidden()
                        .frame(width: 120)
                        Spacer()
                        Button("批量转换", systemImage: "square.stack.3d.up.fill") { model.convert() }
                            .primaryGlassButton()
                            .disabled(model.taskBusy || !model.isReady)
                    }
                }
            }
        }
    }
}
