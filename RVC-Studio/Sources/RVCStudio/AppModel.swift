import Combine
import Foundation

@MainActor
final class AppModel: ObservableObject {
    @Published var selection: AppPage? = .inference
    @Published var backendState: BackendState = .disconnected
    @Published var taskBusy = false
    @Published var progress: Double?
    @Published var taskMessage = "空闲"
    @Published var logs: [String] = []
    @Published var showsLog = false
    @Published var alert: AppAlert?

    @Published var modelPath = ""
    @Published var indexPath = ""
    @Published var availableModels: [String] = []
    @Published var availableIndices: [String] = []
    @Published var modelInfo = "选择模型后显示版本、采样率和说话人数"
    @Published var speakerID = 0
    @Published var speakerMaximum = 4096
    @Published var inferenceMode = "speech"
    @Published var pitch = 0
    @Published var f0Method = "rmvpe"
    @Published var indexRate = 0.75
    @Published var rmsMixRate = 1.0
    @Published var protect = 0.33
    @Published var resampleRate = 0
    @Published var conversionMode = "single"
    @Published var inputPath = ""
    @Published var outputPath = ""
    @Published var batchInputPath = ""
    @Published var batchOutputPath = ""
    @Published var batchFormat = "wav"

    @Published var separationModels = SeparationModel.defaults
    @Published var separationModelID = "vocals"
    @Published var separationModelDirectory = ""
    @Published var separationInputPath = ""
    @Published var separationPrimaryPath = ""
    @Published var separationSecondaryPath = ""
    @Published var separationFormat = "wav"

    @Published var experiment = "my-voice"
    @Published var datasetPath = ""
    @Published var multiSpeaker = false
    @Published var sampleRate = "40k"
    @Published var modelVersion = "v2"
    @Published var useF0 = true
    @Published var trainingF0Method = "rmvpe"
    @Published var trainingSpeakerID = 0
    @Published var workers = min(8, ProcessInfo.processInfo.activeProcessorCount)
    @Published var saveEvery = 10
    @Published var epochs = 200
    @Published var batchSize = 4
    @Published var saveLatest = false
    @Published var cacheGPU = false
    @Published var saveWeights = true
    @Published var autoBestModel = false
    @Published var earlyStopping = false
    @Published var validationSplit = 0.10
    @Published var validationInterval = 10
    @Published var earlyStoppingPatience = 6
    @Published var earlyStoppingMinDelta = 0.002
    @Published var bestTopK = 3
    @Published var conversionValidationPath = ""
    @Published var trainingCurrentEpoch = 0
    @Published var trainingMaxEpoch = 0
    @Published var currentValidationScore: Double?
    @Published var bestValidationScore: Double?
    @Published var bestValidationEpoch: Int?
    @Published var validationBadCount = 0
    @Published var checkpointEligible = true
    @Published var currentGeneratorLoss: Double?
    @Published var currentDiscriminatorLoss: Double?
    @Published var trainingHealthReason = ""
    @Published var pretrainedG = ""
    @Published var pretrainedD = ""

    @Published var checkpointAction = "inspect"
    @Published var checkpointPath1 = ""
    @Published var checkpointPath2 = ""
    @Published var checkpointName = ""
    @Published var checkpointInfo = ""
    @Published var checkpointAlpha = 0.5
    @Published var checkpointRate = "40k"
    @Published var checkpointVersion = "v2"
    @Published var checkpointUsesF0 = true

    @Published var corePath: String
    @Published var pythonPath: String
    @Published var precision = "float32"

    private let backend = BackendClient()
    private let audio = AudioPlayback()
    private var pendingCommands: [String: String] = [:]
    private var didAttemptAutomaticStart = false

    init() {
        let core = Self.defaultCorePath()
        corePath = core
        pythonPath = Self.defaultPythonPath(corePath: core)

        backend.onJSON = { [weak self] object in self?.handle(object) }
        backend.onLog = { [weak self] line in self?.appendLog(line) }
        backend.onStopped = { [weak self] reason in
            guard let self else { return }
            self.backendState = .failed
            self.taskBusy = false
            self.progress = nil
            self.appendLog(reason)
        }
    }

    var isReady: Bool {
        if case .ready = backendState { return true }
        return false
    }

    var selectedSeparationModel: SeparationModel {
        separationModels.first(where: { $0.id == separationModelID })
            ?? SeparationModel.defaults[2]
    }

    func startAutomaticallyOnce() {
        guard !didAttemptAutomaticStart else { return }
        didAttemptAutomaticStart = true
        startBackend()
    }

    func startBackend() {
        guard FileManager.default.fileExists(atPath: corePath) else {
            fail("找不到 RVC-Core", "请在“设置”中选择正确的 RVC-Core 文件夹。")
            return
        }
        backendState = .connecting
        taskMessage = "正在启动并验证 MPS…"
        appendLog("启动严格 MPS 后端…")
        do {
            try backend.start(pythonPath: pythonPath, corePath: corePath, precision: precision)
        } catch {
            backendState = .failed
            fail("后端启动失败", error.localizedDescription)
        }
    }

    func refreshModels() {
        send("list_models")
    }

    func inspectModel() {
        guard isReady, !modelPath.isEmpty else { return }
        send("inspect_model", ["path": modelPath])
    }

    func unloadModel() {
        send("unload_model")
    }

    func refreshSeparationModels() {
        send("separation_models")
    }

    func downloadSeparationModel() {
        guardReady()
        guard isReady else { return }
        if send("download_separation_model", ["model": separationModelID]) != nil {
            beginTask("正在下载分离模型…")
        }
    }

    func deleteSeparationModel() {
        send("delete_separation_model", ["model": separationModelID])
    }

    func convert() {
        guardReady()
        guard isReady else { return }
        var request = inferenceParameters
        let command: String
        if conversionMode == "batch" {
            guard !batchInputPath.isEmpty, !batchOutputPath.isEmpty else {
                fail("缺少路径", "请选择批量输入与输出文件夹。")
                return
            }
            command = "convert_batch"
            request["input"] = batchInputPath
            request["output_dir"] = batchOutputPath
            request["format"] = batchFormat
        } else {
            guard !inputPath.isEmpty, !outputPath.isEmpty else {
                fail("缺少路径", "请选择输入和输出音频。")
                return
            }
            command = "convert"
            request["input"] = inputPath
            request["output"] = outputPath
        }
        if send(command, request) != nil { beginTask("正在准备转换…") }
    }

    func playOutput() {
        guard FileManager.default.fileExists(atPath: outputPath) else {
            fail("无法播放", "输出文件尚不存在。")
            return
        }
        do { try audio.play(path: outputPath) }
        catch { fail("无法播放", error.localizedDescription) }
    }

    func runTraining(_ operation: String) {
        guardReady()
        guard isReady else { return }
        var request: [String: Any] = [
            "experiment": experiment, "dataset": datasetPath,
            "multi_speaker": multiSpeaker, "sample_rate": sampleRate,
            "version": modelVersion, "use_f0": useF0,
            "f0_method": trainingF0Method, "speaker_id": trainingSpeakerID,
            "workers": workers, "save_every": saveEvery, "epochs": epochs,
            "batch_size": batchSize, "save_latest": saveLatest,
            "cache_gpu": cacheGPU, "save_weights": saveWeights,
            "auto_best_model": autoBestModel, "early_stopping": earlyStopping,
            "validation_split": validationSplit, "validation_interval": validationInterval,
            "patience": earlyStoppingPatience, "min_delta": earlyStoppingMinDelta,
            "top_k": bestTopK, "conversion_validation_dir": conversionValidationPath,
            "pretrained_g": pretrainedG, "pretrained_d": pretrainedD
        ]
        request["operation"] = operation
        if send("train_task", request) != nil {
            if operation == "train" || operation == "one_click" {
                trainingCurrentEpoch = 0
                trainingMaxEpoch = epochs
                currentValidationScore = nil
                bestValidationScore = nil
                bestValidationEpoch = nil
                validationBadCount = 0
                checkpointEligible = true
                currentGeneratorLoss = nil
                currentDiscriminatorLoss = nil
                trainingHealthReason = ""
            }
            beginTask("正在准备训练任务…")
        }
    }

    func separate() {
        guardReady()
        guard isReady else { return }
        guard !separationInputPath.isEmpty,
              !separationPrimaryPath.isEmpty,
              !separationSecondaryPath.isEmpty else {
            fail("缺少路径", "请选择输入以及两个音轨的输出文件夹。")
            return
        }
        let request: [String: Any] = [
            "model": separationModelID, "input": separationInputPath,
            "output": separationPrimaryPath, "secondary_output": separationSecondaryPath,
            "format": separationFormat
        ]
        if send("separate", request) != nil { beginTask("正在准备 MLX 分离…") }
    }

    func runCheckpointTool() {
        guardReady()
        guard isReady else { return }
        let request: [String: Any] = [
            "action": checkpointAction, "path": checkpointPath1,
            "path1": checkpointPath1, "path2": checkpointPath2,
            "name": checkpointName, "info": checkpointInfo,
            "alpha": checkpointAlpha, "sample_rate": checkpointRate,
            "version": checkpointVersion, "use_f0": checkpointUsesF0
        ]
        _ = send("ckpt", request)
    }

    func cancelTask() {
        _ = send("cancel_task")
        taskMessage = "正在停止任务…"
    }

    func pickedInputAudio(_ path: String) {
        inputPath = path
        if outputPath.isEmpty {
            let url = URL(fileURLWithPath: path)
            outputPath = url.deletingPathExtension().path + "_rvc.wav"
        }
    }

    private var inferenceParameters: [String: Any] {
        ["model": modelPath, "index": indexPath, "mode": inferenceMode,
         "speaker_id": speakerID, "pitch": pitch, "f0_method": f0Method,
         "index_rate": indexRate, "rms_mix_rate": rmsMixRate,
         "protect": protect, "resample_sr": resampleRate, "overwrite": true]
    }

    @discardableResult
    private func send(_ command: String, _ arguments: [String: Any] = [:]) -> String? {
        guard let id = backend.send(command, arguments: arguments) else {
            if command != "list_models" && command != "separation_models" {
                fail("后端未启动", "请先在“设置”中启动并验证 GPU。")
            }
            return nil
        }
        pendingCommands[id] = command
        return id
    }

    private func guardReady() {
        if !isReady { fail("后端未启动", "请先在“设置”中启动并验证 GPU。") }
    }

    private func beginTask(_ message: String) {
        taskBusy = true
        progress = nil
        taskMessage = message
    }

    private func handle(_ object: [String: Any]) {
        switch object["event"] as? String {
        case "ready":
            let data = object["data"] as? [String: Any] ?? [:]
            let dtype = data["dtype"] as? String ?? precision
            let memory = data["memory"] as? [String: Any] ?? [:]
            let bytes = int64(memory["driver_allocated_bytes"])
            backendState = .ready(dtype: dtype, memory: Self.bytesLabel(bytes))
            taskBusy = false
            progress = nil
            taskMessage = "GPU 已就绪"
            appendLog("严格 GPU 探针通过：\(data["device_name"] as? String ?? "Apple MPS")；CPU fallback=\(bool(data["cpu_fallback"]) ? "开启" : "关闭")")
            refreshModels()
            refreshSeparationModels()
        case "progress":
            appendLog(object["message"] as? String ?? "")
        case "response":
            handleResponse(object)
        case "task":
            handleTask(object)
        case "fatal":
            backendState = .failed
            taskBusy = false
            let message = object["error"] as? String ?? "后端发生未知错误"
            appendLog(message)
            fail("RVC Core 错误", message)
        default:
            appendLog("收到未知后端事件。")
        }
    }

    private func handleResponse(_ object: [String: Any]) {
        let id = String(describing: object["id"] ?? "")
        let command = pendingCommands.removeValue(forKey: id) ?? ""
        guard bool(object["ok"]) else {
            taskBusy = false
            progress = nil
            let message = object["error"] as? String ?? "未知错误"
            appendLog("错误：\(message)")
            showsLog = true
            fail("RVC Core 错误", message)
            return
        }
        let data = object["data"] as? [String: Any] ?? [:]
        switch command {
        case "list_models":
            availableModels = data["models"] as? [String] ?? []
            availableIndices = data["indices"] as? [String] ?? []
            if modelPath.isEmpty { modelPath = availableModels.first ?? "" }
            if indexPath.isEmpty { indexPath = availableIndices.first ?? "" }
            inspectModel()
        case "inspect_model":
            let version = data["version"] as? String ?? "–"
            let rate = int(data["sample_rate"])
            let speakers = max(1, int(data["speaker_count"]))
            modelInfo = "\(version) · \(rate) Hz · \(speakers) 位说话人 · F0 \(bool(data["uses_f0"]) ? "开启" : "关闭")"
            speakerMaximum = max(0, speakers - 1)
            speakerID = min(speakerID, speakerMaximum)
        case "ckpt":
            appendLog(data["result"] as? String ?? "操作完成")
            refreshModels()
        case "unload_model":
            appendLog("音色模型已卸载，GPU 缓存已释放。")
        case "separation_models", "delete_separation_model":
            updateSeparationModels(data)
        default:
            break
        }
    }

    private func handleTask(_ object: [String: Any]) {
        let state = object["state"] as? String ?? ""
        let message = object["message"] as? String ?? ""
        taskMessage = message
        appendLog(message)
        if bool(object["training_status"]) {
            trainingCurrentEpoch = int(object["current_epoch"])
            trainingMaxEpoch = int(object["max_epoch"])
            if let score = numeric(object["validation_score"]) {
                currentValidationScore = score
            }
            if let score = numeric(object["best_score"]) {
                bestValidationScore = score
            }
            let epoch = int(object["best_epoch"])
            bestValidationEpoch = epoch > 0 ? epoch : nil
            validationBadCount = int(object["bad_count"])
            checkpointEligible = bool(object["checkpoint_eligible"])
            currentGeneratorLoss = numeric(object["generator_loss"])
            currentDiscriminatorLoss = numeric(object["discriminator_loss"])
            trainingHealthReason = object["health_reason"] as? String ?? ""
        }
        if let total = numeric(object["total"]), total > 0,
           let current = numeric(object["current"]) {
            progress = min(1, max(0, current / total))
        } else if state == "running" {
            taskBusy = true
            progress = nil
        }
        guard ["done", "failed", "cancelled"].contains(state) else { return }
        taskBusy = false
        progress = state == "done" ? 1 : nil
        if let result = object["result"] as? [String: Any],
           let output = result["output"] as? String,
           !output.isEmpty,
           !FileManager.default.isDirectory(atPath: output) {
            outputPath = output
        }
        if state == "failed" {
            showsLog = true
            fail("任务失败", message)
        } else if state == "done" {
            refreshModels()
            refreshSeparationModels()
        }
    }

    private func updateSeparationModels(_ data: [String: Any]) {
        separationModelDirectory = data["model_dir"] as? String ?? separationModelDirectory
        guard let rows = data["models"] as? [[String: Any]] else { return }
        separationModels = SeparationModel.defaults.map { item in
            guard let row = rows.first(where: { ($0["key"] as? String) == item.id }) else { return item }
            return SeparationModel(id: item.id, name: item.name,
                                   installed: bool(row["installed"]),
                                   sizeBytes: int64(row["size_bytes"]))
        }
    }

    private func appendLog(_ value: String) {
        let lines = value.split(whereSeparator: \.isNewline).map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        logs.append(contentsOf: lines)
        if logs.count > 3_000 { logs.removeFirst(logs.count - 3_000) }
    }

    private func fail(_ title: String, _ message: String) {
        alert = AppAlert(title: title, message: message)
    }

    private static func defaultCorePath() -> String {
        let fm = FileManager.default
        var candidates: [URL] = []
        if let configured = ProcessInfo.processInfo.environment["RVC_CORE_DEV_PATH"] {
            candidates.append(URL(fileURLWithPath: configured))
        }
        if let resources = Bundle.main.resourceURL {
            candidates.append(resources.appendingPathComponent("RVC-Core", isDirectory: true))
        }
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
        candidates.append(cwd.appendingPathComponent("../RVC-Core", isDirectory: true).standardizedFileURL)
        candidates.append(cwd.appendingPathComponent("RVC-Core", isDirectory: true))
        let bundle = Bundle.main.bundleURL
        candidates.append(bundle.deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("RVC-Core", isDirectory: true))
        return candidates.first(where: { fm.fileExists(atPath: $0.path) })?.path
            ?? candidates.first?.path ?? ""
    }

    private static func defaultPythonPath(corePath: String) -> String {
        let fm = FileManager.default
        let core = URL(fileURLWithPath: corePath, isDirectory: true)
        let candidates = [
            core.appendingPathComponent("runtime/python/bin/python3").path,
            core.appendingPathComponent("runtime/python-env/bin/python").path
        ]
        return candidates.first(where: { fm.isExecutableFile(atPath: $0) }) ?? "python3"
    }

    private static func bytesLabel(_ bytes: Int64) -> String {
        guard bytes > 0 else { return "0 MB" }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter.string(fromByteCount: bytes)
    }
}

private func bool(_ value: Any?) -> Bool {
    if let value = value as? Bool { return value }
    if let value = value as? NSNumber { return value.boolValue }
    return false
}

private func int(_ value: Any?) -> Int {
    if let value = value as? Int { return value }
    if let value = value as? NSNumber { return value.intValue }
    return 0
}

private func int64(_ value: Any?) -> Int64 {
    if let value = value as? Int64 { return value }
    if let value = value as? NSNumber { return value.int64Value }
    return 0
}

private func numeric(_ value: Any?) -> Double? {
    if let value = value as? Double { return value }
    if let value = value as? NSNumber { return value.doubleValue }
    return nil
}

private extension FileManager {
    func isDirectory(atPath path: String) -> Bool {
        var flag: ObjCBool = false
        return fileExists(atPath: path, isDirectory: &flag) && flag.boolValue
    }
}
