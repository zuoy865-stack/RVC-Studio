import SwiftUI

@main
struct RVCStudioApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            MainShellView(model: model)
                .frame(minWidth: 1_020, minHeight: 700)
                .background(WindowConfigurator())
        }
        .defaultSize(width: 1_260, height: 840)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(after: .appInfo) {
                Button("重新连接 RVC Core") { model.startBackend() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
            }
        }
    }
}
