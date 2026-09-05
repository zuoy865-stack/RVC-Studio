import Foundation

final class BackendClient: @unchecked Sendable {
    var onJSON: (([String: Any]) -> Void)?
    var onLog: ((String) -> Void)?
    var onStopped: ((String) -> Void)?

    private var process: Process?
    private var input: FileHandle?
    private var stdoutBuffer = Data()
    private let parseQueue = DispatchQueue(label: "studio.rvc.backend.protocol")
    private var sequence: UInt64 = 0

    var isRunning: Bool { process?.isRunning == true }

    func start(pythonPath: String, corePath: String, precision: String) throws {
        stop(gracefully: false)

        let task = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        let stdin = Pipe()
        let trimmedPython = pythonPath.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmedPython.contains("/") {
            task.executableURL = URL(fileURLWithPath: trimmedPython)
            task.arguments = ["-u", "-m", "rvc_core.server", "--precision", precision]
        } else {
            task.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            task.arguments = [trimmedPython.isEmpty ? "python3" : trimmedPython,
                              "-u", "-m", "rvc_core.server", "--precision", precision]
        }

        task.currentDirectoryURL = URL(fileURLWithPath: corePath, isDirectory: true)
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        environment["PYTORCH_MPS_PREFER_METAL"] = "1"
        let packagedSite = URL(fileURLWithPath: corePath).appendingPathComponent("runtime/site-packages").path
        var pythonSearchPath = corePath
        if FileManager.default.fileExists(atPath: packagedSite) {
            pythonSearchPath += ":\(packagedSite)"
        }
        if let existing = environment["PYTHONPATH"], !existing.isEmpty {
            pythonSearchPath += ":\(existing)"
        }
        environment["PYTHONPATH"] = pythonSearchPath
        task.environment = environment
        task.standardInput = stdin
        task.standardOutput = stdout
        task.standardError = stderr

        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            self?.parseQueue.async { self?.consumeStdout(data) }
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async { self?.onLog?(text) }
        }
        task.terminationHandler = { [weak self, weak task] finished in
            DispatchQueue.main.async {
                guard self?.process === task else { return }
                self?.process = nil
                self?.input = nil
                self?.onStopped?("后端已退出（代码 \(finished.terminationStatus)）")
            }
        }

        try task.run()
        process = task
        input = stdin.fileHandleForWriting
    }

    @discardableResult
    func send(_ command: String, arguments: [String: Any] = [:]) -> String? {
        guard isRunning, let input else { return nil }
        sequence += 1
        let id = String(sequence)
        var request = arguments
        request["id"] = id
        request["command"] = command
        guard JSONSerialization.isValidJSONObject(request),
              var data = try? JSONSerialization.data(withJSONObject: request) else { return nil }
        data.append(0x0A)
        do {
            try input.write(contentsOf: data)
            return id
        } catch {
            onLog?("无法向后端发送命令：\(error.localizedDescription)")
            return nil
        }
    }

    func stop(gracefully: Bool = true) {
        guard let process else { return }
        if gracefully { _ = send("shutdown") }
        if process.isRunning { process.terminate() }
        self.process = nil
        input = nil
    }

    private func consumeStdout(_ data: Data) {
        stdoutBuffer.append(data)
        while let newline = stdoutBuffer.firstIndex(of: 0x0A) {
            let line = stdoutBuffer[..<newline]
            stdoutBuffer.removeSubrange(...newline)
            guard !line.isEmpty else { continue }
            do {
                let value = try JSONSerialization.jsonObject(with: Data(line))
                guard let object = value as? [String: Any] else { throw ProtocolError.invalidFrame }
                DispatchQueue.main.async { [weak self] in self?.onJSON?(object) }
            } catch {
                let raw = String(data: line, encoding: .utf8) ?? "<binary>"
                DispatchQueue.main.async { [weak self] in self?.onLog?("后端协议错误：\(raw)") }
            }
        }
    }

    private enum ProtocolError: Error { case invalidFrame }
}
