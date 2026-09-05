import SwiftUI

struct StudioCard<Content: View>: View {
    let title: String
    var subtitle: String?
    @ViewBuilder let content: Content

    init(_ title: String, subtitle: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.subtitle = subtitle
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .fontWeight(.semibold)
                if let subtitle {
                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            content
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .liquidGlass(cornerRadius: 22)
    }
}

struct InfoBanner: View {
    let symbol: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: symbol)
                .foregroundStyle(.tint)
            Text(text)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 15)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(.tint.opacity(0.14), lineWidth: 1)
        }
    }
}

struct FieldLabel: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text)
            .font(.caption)
            .fontWeight(.medium)
            .foregroundStyle(.secondary)
    }
}

struct LabeledField<Content: View>: View {
    let title: String
    var hint: String?
    @ViewBuilder let content: Content

    init(_ title: String, hint: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.hint = hint
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            FieldLabel(title)
            content
            if let hint {
                Text(hint).font(.caption).foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct PathField: View {
    let title: String
    @Binding var path: String
    let kind: PickerKind
    var onPick: ((String) -> Void)?

    var body: some View {
        HStack(spacing: 12) {
            FieldLabel(title)
                .frame(width: 126, alignment: .leading)
            TextField("路径", text: $path)
                .textFieldStyle(.roundedBorder)
            Button("选择…") {
                if let selected = NativePanel.choose(kind, startingAt: path) {
                    path = selected
                    onPick?(selected)
                }
            }
        }
    }
}

struct PathMenuField: View {
    let title: String
    @Binding var path: String
    let choices: [String]
    let kind: PickerKind
    var onCommit: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            FieldLabel(title)
            HStack(spacing: 8) {
                TextField("路径", text: $path, onCommit: { onCommit?() })
                    .textFieldStyle(.roundedBorder)
                if !choices.isEmpty {
                    Menu {
                        ForEach(choices, id: \.self) { item in
                            Button(URL(fileURLWithPath: item).lastPathComponent) {
                                path = item
                                onCommit?()
                            }
                        }
                    } label: {
                        Image(systemName: "chevron.down")
                    }
                    .menuStyle(.borderlessButton)
                    .frame(width: 24)
                }
                Button("选择…") {
                    if let selected = NativePanel.choose(kind, startingAt: path) {
                        path = selected
                        onCommit?()
                    }
                }
            }
        }
    }
}

struct RatioControl: View {
    @Binding var value: Double
    var range: ClosedRange<Double> = 0...1

    var body: some View {
        HStack(spacing: 12) {
            Slider(value: $value, in: range)
            Text(value, format: .number.precision(.fractionLength(2)))
                .font(.system(.body, design: .rounded).monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 42, alignment: .trailing)
        }
    }
}

struct StatusPill: View {
    let state: BackendState

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: state.symbol)
                .symbolEffect(.pulse, isActive: state == .connecting)
            Text(state.label)
                .lineLimit(1)
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(color)
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(color.opacity(0.10), in: Capsule())
        .overlay { Capsule().stroke(color.opacity(0.17), lineWidth: 1) }
    }

    private var color: Color {
        switch state {
        case .ready: .green
        case .failed: .red
        case .connecting: .orange
        case .disconnected: .secondary
        }
    }
}

struct PageColumn<Content: View>: View {
    @ViewBuilder let content: Content

    init(@ViewBuilder content: () -> Content) { self.content = content() }

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 14) { content }
                .padding(.horizontal, 22)
                .padding(.top, 8)
                .padding(.bottom, 28)
        }
        .scrollIndicators(.hidden)
    }
}

extension View {
    @ViewBuilder
    func liquidGlass(cornerRadius: CGFloat = 20) -> some View {
        if #available(macOS 26.0, *) {
            self.glassEffect(.regular, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        } else {
            self
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .stroke(.white.opacity(0.12), lineWidth: 1)
                }
        }
    }

    @ViewBuilder
    func primaryGlassButton() -> some View {
        if #available(macOS 26.0, *) {
            self.buttonStyle(.glassProminent)
        } else {
            self.buttonStyle(.borderedProminent)
        }
    }
}
