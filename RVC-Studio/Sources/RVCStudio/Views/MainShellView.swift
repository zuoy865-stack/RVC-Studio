import SwiftUI

struct MainShellView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 190, ideal: 215, max: 250)
        } detail: {
            VStack(spacing: 0) {
                header
                page
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                taskFooter
            }
            .background {
                LinearGradient(
                    colors: [.accentColor.opacity(0.055), .clear, .purple.opacity(0.035)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
            }
        }
        .navigationSplitViewStyle(.balanced)
        .alert(item: $model.alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message), dismissButton: .default(Text("好")))
        }
        .task { model.startAutomaticallyOnce() }
    }

    private var sidebar: some View {
        List(selection: $model.selection) {
            Section("工作区") {
                ForEach(AppPage.allCases.filter { $0 != .settings }) { page in
                    Label(page.title, systemImage: page.symbol).tag(page)
                }
            }
            Section {
                Label(AppPage.settings.title, systemImage: AppPage.settings.symbol)
                    .tag(AppPage.settings)
            }
        }
        .safeAreaInset(edge: .top) {
            HStack(spacing: 11) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.accentColor.gradient)
                    Image(systemName: "waveform.path.ecg")
                        .font(.system(size: 17, weight: .bold))
                        .foregroundStyle(.white)
                }
                .frame(width: 38, height: 38)
                VStack(alignment: .leading, spacing: 1) {
                    Text("RVC Studio").font(.headline)
                    Text("Apple Silicon").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.top, 10)
            .padding(.bottom, 14)
        }
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading, spacing: 3) {
                Label("RVC · MPS", systemImage: "cpu")
                Label("PyMSS · MLX", systemImage: "sparkles")
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
        }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text(currentPage.title)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                Text(currentPage.subtitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            StatusPill(state: model.backendState)
        }
        .padding(.horizontal, 24)
        .padding(.top, 18)
        .padding(.bottom, 14)
    }

    @ViewBuilder private var page: some View {
        switch currentPage {
        case .inference: InferenceView(model: model)
        case .separation: SeparationView(model: model)
        case .training: TrainingView(model: model)
        case .checkpoint: CheckpointView(model: model)
        case .settings: SettingsView(model: model)
        }
    }

    private var taskFooter: some View {
        VStack(spacing: 8) {
            if model.taskBusy {
                if let progress = model.progress {
                    ProgressView(value: progress)
                } else {
                    ProgressView().controlSize(.small)
                }
            } else if model.progress == 1 {
                ProgressView(value: 1).tint(.green)
            }

            HStack(spacing: 10) {
                Image(systemName: model.taskBusy ? "bolt.horizontal.circle" : "checkmark.circle")
                    .foregroundStyle(model.taskBusy ? .orange : .secondary)
                Text(model.taskMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                Button(model.showsLog ? "隐藏日志" : "显示日志") {
                    withAnimation(.snappy) { model.showsLog.toggle() }
                }
                if model.taskBusy {
                    Button("停止任务", role: .destructive) { model.cancelTask() }
                }
            }
            if model.showsLog { logPanel.transition(.move(edge: .bottom).combined(with: .opacity)) }
        }
        .padding(.horizontal, 22)
        .padding(.bottom, 14)
    }

    private var logPanel: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(model.logs.enumerated()), id: \.offset) { index, line in
                        Text(line)
                            .font(.system(size: 11, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(index)
                    }
                }
                .padding(12)
            }
            .frame(height: 150)
            .background(.black.opacity(0.28), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .onChange(of: model.logs.count) { _, count in
                if count > 0 { proxy.scrollTo(count - 1, anchor: .bottom) }
            }
        }
    }

    private var currentPage: AppPage { model.selection ?? .inference }
}
