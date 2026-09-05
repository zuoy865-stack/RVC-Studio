import Foundation

enum AppPage: String, CaseIterable, Identifiable {
    case inference, separation, training, checkpoint, settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .inference: "模型推理"
        case .separation: "人声分离"
        case .training: "训练"
        case .checkpoint: "ckpt 工具"
        case .settings: "设置"
        }
    }

    var subtitle: String {
        switch self {
        case .inference: "加载 RVC 音色并使用 Apple Silicon MPS 进行转换"
        case .separation: "使用 PyMSS / MLX 在本机分离人声、伴奏与混响"
        case .training: "数据处理、特征提取、MPS 训练与 FAISS 索引"
        case .checkpoint: "检查、编辑、提取或融合 RVC 检查点"
        case .settings: "RVC-Core、Python、精度与 GPU 后端"
        }
    }

    var symbol: String {
        switch self {
        case .inference: "waveform"
        case .separation: "square.split.2x1"
        case .training: "cpu"
        case .checkpoint: "shippingbox"
        case .settings: "gearshape"
        }
    }
}

struct AppAlert: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

struct SeparationModel: Identifiable, Hashable {
    let id: String
    let name: String
    var installed = false
    var sizeBytes: Int64 = 0

    static let defaults = [
        SeparationModel(id: "dereverb", name: "去混响"),
        SeparationModel(id: "dereverb_aggressive", name: "去混响（激进）"),
        SeparationModel(id: "vocals", name: "去伴奏 / 提取人声"),
        SeparationModel(id: "vocals_aggressive", name: "去伴奏（激进）"),
        SeparationModel(id: "lead_vocal", name: "提取主旋律")
    ]
}

enum PickerKind {
    case audio, audioOrDirectory, model, index, pytorch, anyFile, directory, saveAudio
}

enum BackendState: Equatable {
    case disconnected
    case connecting
    case ready(dtype: String, memory: String)
    case failed

    var label: String {
        switch self {
        case .disconnected: "MPS 未连接"
        case .connecting: "正在验证 MPS"
        case let .ready(dtype, memory): "MPS · \(dtype) · \(memory)"
        case .failed: "后端不可用"
        }
    }

    var symbol: String {
        switch self {
        case .ready: "checkmark.circle.fill"
        case .connecting: "arrow.triangle.2.circlepath"
        case .disconnected: "circle.dotted"
        case .failed: "exclamationmark.triangle.fill"
        }
    }
}
