import SwiftUI

struct TrainingView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        PageColumn {
            InfoBanner(symbol: "cpu", text: "RMVPE、HuBERT 和神经网络训练走 MPS；音频切片与 FAISS 索引属于 CPU 数据处理。多说话人目录格式：名称_ID_重复次数。")

            StudioCard("1  ·  实验配置") {
                Grid(horizontalSpacing: 18, verticalSpacing: 14) {
                    GridRow {
                        LabeledField("实验名") {
                            TextField("my-voice", text: $model.experiment).textFieldStyle(.roundedBorder)
                        }
                        LabeledField("目标采样率") {
                            Picker("目标采样率", selection: $model.sampleRate) {
                                ForEach(validSampleRates, id: \.self) { Text($0).tag($0) }
                            }.labelsHidden()
                        }
                    }
                    GridRow {
                        LabeledField("模型版本") {
                            Picker("模型版本", selection: $model.modelVersion) {
                                Text("v2").tag("v2")
                                Text("v1").tag("v1")
                            }.labelsHidden()
                        }
                        VStack(alignment: .leading, spacing: 9) {
                            Toggle("模型带音高指导", isOn: $model.useF0)
                            Toggle("多说话人训练", isOn: $model.multiSpeaker)
                        }
                    }
                }
                .onChange(of: model.modelVersion) { _, value in
                    if value == "v1", model.sampleRate == "32k" { model.sampleRate = "40k" }
                }
            }

            StudioCard("2  ·  数据处理与特征提取") {
                PathField(title: "训练集文件夹", path: $model.datasetPath, kind: .directory)
                Grid(horizontalSpacing: 18, verticalSpacing: 14) {
                    GridRow {
                        LabeledField("单说话人 ID") {
                            Stepper(value: $model.trainingSpeakerID, in: 0...109) {
                                Text("\(model.trainingSpeakerID)").monospacedDigit()
                            }.disabled(model.multiSpeaker)
                        }
                        LabeledField("数据处理进程数") {
                            Stepper(value: $model.workers, in: 1...64) {
                                Text("\(model.workers)").monospacedDigit()
                            }
                        }
                    }
                    GridRow {
                        LabeledField("F0 提取算法") {
                            Picker("F0 提取算法", selection: $model.trainingF0Method) {
                                Text("RMVPE · MPS GPU").tag("rmvpe")
                                Text("Praat PM · CPU DSP").tag("pm")
                            }.labelsHidden().disabled(!model.useF0)
                        }
                        Color.clear.frame(height: 1)
                    }
                }
            }

            StudioCard("3  ·  模型训练设置") {
                Grid(horizontalSpacing: 18, verticalSpacing: 14) {
                    GridRow {
                        numberStepper("每 N 轮保存", value: $model.saveEvery, range: 1...10_000)
                        numberStepper("总轮数", value: $model.epochs, range: 1...100_000)
                        numberStepper("批大小", value: $model.batchSize, range: 1...128)
                    }
                }
                Toggle("只保留最新 G/D 检查点", isOn: $model.saveLatest)
                Toggle("将训练集缓存到统一内存 / GPU（小数据集）", isOn: $model.cacheGPU)
                Toggle("每次保存可推理的小模型", isOn: $model.saveWeights)
                Divider()
                HStack(spacing: 24) {
                    Toggle("自动选择最佳模型", isOn: $model.autoBestModel)
                    Toggle("自动停止过拟合", isOn: $model.earlyStopping)
                }
                Grid(horizontalSpacing: 18, verticalSpacing: 14) {
                    GridRow {
                        doubleStepper(
                            "验证集比例",
                            value: $model.validationSplit,
                            range: 0.01...0.49,
                            step: 0.01,
                            format: "%.0f%%",
                            multiplier: 100
                        )
                        numberStepper("验证间隔（Epoch）", value: $model.validationInterval, range: 1...10_000)
                        numberStepper("patience（验证次数）", value: $model.earlyStoppingPatience, range: 1...1_000)
                            .disabled(!model.earlyStopping)
                    }
                    GridRow {
                        doubleStepper(
                            "min_delta",
                            value: $model.earlyStoppingMinDelta,
                            range: 0...1,
                            step: 0.001,
                            format: "%.3f"
                        )
                        .disabled(!model.earlyStopping)
                        numberStepper("Top-K 数量", value: $model.bestTopK, range: 1...20)
                            .disabled(!model.autoBestModel)
                        Color.clear.frame(height: 1)
                    }
                }
                .disabled(!model.autoBestModel && !model.earlyStopping)

                PathField(
                    title: "额外的非目标音色验证集（可选）",
                    path: $model.conversionValidationPath,
                    kind: .directory
                )
                .disabled(!model.autoBestModel && !model.earlyStopping)
                Text("始终包含内置低音、中音、高音和长音探针；可在此追加真人非目标歌手音频。")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack(spacing: 18) {
                    Label("Epoch \(model.trainingCurrentEpoch)/\(model.trainingMaxEpoch > 0 ? model.trainingMaxEpoch : model.epochs)", systemImage: "chart.line.uptrend.xyaxis")
                    Text("当前 Validation：\(scoreLabel(model.currentValidationScore))")
                    Text("最佳 Epoch：\(model.bestValidationEpoch.map(String.init) ?? "—")")
                    Text("最佳 Score：\(scoreLabel(model.bestValidationScore))")
                    if model.earlyStopping {
                        Text("bad_count：\(model.validationBadCount)/\(model.earlyStoppingPatience)")
                    }
                }
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                HStack(spacing: 18) {
                    Text("G loss：\(scoreLabel(model.currentGeneratorLoss))")
                    Text("D loss：\(scoreLabel(model.currentDiscriminatorLoss))")
                    Label(
                        model.checkpointEligible ? "健康检查通过" : "禁止进入 Top-K",
                        systemImage: model.checkpointEligible ? "checkmark.shield" : "exclamationmark.triangle"
                    )
                    if !model.trainingHealthReason.isEmpty {
                        Text(model.trainingHealthReason)
                    }
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(model.checkpointEligible ? Color.secondary : Color.orange)
            }

            StudioCard("预训练模型", subtitle: "可留空，使用 RVC-Core 默认预训练权重。") {
                PathField(title: "预训练 G", path: $model.pretrainedG, kind: .pytorch)
                PathField(title: "预训练 D", path: $model.pretrainedD, kind: .pytorch)
            }

            StudioCard("训练流程") {
                HStack(spacing: 8) {
                    trainingButton("处理数据", symbol: "waveform.path", operation: "preprocess")
                    trainingButton("提取 F0 + 特征", symbol: "point.3.connected.trianglepath.dotted", operation: "extract")
                    trainingButton("训练模型", symbol: "cpu", operation: "train")
                    trainingButton("训练索引", symbol: "text.magnifyingglass", operation: "index")
                    Spacer()
                    Button("一键训练", systemImage: "bolt.fill") { model.runTraining("one_click") }
                        .primaryGlassButton()
                        .disabled(model.taskBusy || !model.isReady)
                }
            }
        }
    }

    private var validSampleRates: [String] { model.modelVersion == "v1" ? ["40k", "48k"] : ["32k", "40k", "48k"] }

    private func numberStepper(_ title: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
        LabeledField(title) {
            Stepper(value: value, in: range) { Text("\(value.wrappedValue)").monospacedDigit() }
        }
    }

    private func doubleStepper(
        _ title: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        format: String,
        multiplier: Double = 1
    ) -> some View {
        LabeledField(title) {
            Stepper(value: value, in: range, step: step) {
                Text(String(format: format, value.wrappedValue * multiplier)).monospacedDigit()
            }
        }
    }

    private func scoreLabel(_ score: Double?) -> String {
        guard let score else { return "—" }
        return String(format: "%.6f", score)
    }

    private func trainingButton(_ title: String, symbol: String, operation: String) -> some View {
        Button(title, systemImage: symbol) { model.runTraining(operation) }
            .disabled(model.taskBusy || !model.isReady)
    }
}
