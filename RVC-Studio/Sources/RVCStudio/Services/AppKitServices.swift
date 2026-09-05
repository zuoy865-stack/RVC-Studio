import AppKit
import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class AudioPlayback {
    private var player: AVAudioPlayer?

    func play(path: String) throws {
        player = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: path))
        player?.prepareToPlay()
        player?.play()
    }
}

enum NativePanel {
    @MainActor
    static func choose(_ kind: PickerKind, startingAt path: String = "") -> String? {
        if kind == .saveAudio {
            let panel = NSSavePanel()
            panel.title = "选择输出音频"
            panel.allowedContentTypes = [.wav, .audio]
            panel.canCreateDirectories = true
            if !path.isEmpty { panel.directoryURL = URL(fileURLWithPath: path).deletingLastPathComponent() }
            return panel.runModal() == .OK ? panel.url?.path : nil
        }

        let panel = NSOpenPanel()
        panel.canChooseDirectories = kind == .directory || kind == .audioOrDirectory
        panel.canChooseFiles = kind != .directory
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        if !path.isEmpty { panel.directoryURL = URL(fileURLWithPath: path).deletingLastPathComponent() }
        switch kind {
        case .audio, .audioOrDirectory:
            panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio]
        case .model, .pytorch:
            panel.allowedContentTypes = [UTType(filenameExtension: "pth")].compactMap { $0 }
        case .index:
            panel.allowedContentTypes = [UTType(filenameExtension: "index")].compactMap { $0 }
        case .anyFile, .directory, .saveAudio:
            break
        }
        return panel.runModal() == .OK ? panel.url?.path : nil
    }
}

struct WindowConfigurator: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { configure(view.window) }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async { configure(nsView.window) }
    }

    private func configure(_ window: NSWindow?) {
        guard let window else { return }
        window.title = "RVC Studio"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.styleMask.insert(.fullSizeContentView)
        window.toolbarStyle = .unifiedCompact
        window.isMovableByWindowBackground = false
        window.isOpaque = true
        window.hasShadow = true
        window.backgroundColor = .windowBackgroundColor
    }
}
