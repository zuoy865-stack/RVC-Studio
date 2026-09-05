import numpy as np


SPEECH_MODE = "speech"
SINGING_MODE = "singing"
SINGING_NOISE_SCALE = 0.15


def normalize_inference_mode(mode):
    """统一推理模式名称，默认保持原版说话模式。"""
    value = str(mode or SPEECH_MODE).strip().lower()
    if value not in (SPEECH_MODE, SINGING_MODE):
        raise ValueError("Unsupported inference mode: %s" % mode)
    return value


def _nearest_octave(candidate, reference):
    """把候选 F0 按八度折叠到最接近参考 F0 的位置。"""
    if candidate <= 0 or reference <= 0:
        return candidate
    ratio = candidate / reference
    while ratio > np.sqrt(2.0):
        candidate *= 0.5
        ratio *= 0.5
    while ratio < 1.0 / np.sqrt(2.0):
        candidate *= 2.0
        ratio *= 2.0
    return candidate


def correct_octave_errors(f0, confidence=None):
    """保守修正持续音中的单帧/短时八度跳变，不处理真实的大跨度换音。"""
    f0 = np.asarray(f0, dtype=np.float64).copy()
    if f0.size < 5:
        return f0

    conf = None
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float64)
        if conf.shape[0] != f0.shape[0]:
            conf = None

    voiced = f0 > 0
    logf = np.zeros_like(f0)
    logf[voiced] = np.log2(f0[voiced])

    corrected = f0.copy()
    for i in range(2, len(f0) - 2):
        if not voiced[i]:
            continue

        neighbor_idx = [i - 2, i - 1, i + 1, i + 2]
        neighbor_vals = [f0[j] for j in neighbor_idx if voiced[j]]
        if len(neighbor_vals) < 3:
            continue

        neighbor_logs = np.log2(neighbor_vals)
        # 邻域本身必须稳定，避免把真正的换音误判成八度错误。
        if np.max(neighbor_logs) - np.min(neighbor_logs) > 0.18:
            continue

        reference = float(np.median(neighbor_vals))
        current_delta = abs(logf[i] - np.log2(reference))
        if current_delta < 0.62:
            continue

        folded = _nearest_octave(float(f0[i]), reference)
        folded_delta = abs(np.log2(folded / reference))
        if folded_delta > 0.22:
            continue

        # 低置信度更倾向修正；高置信度仅在邻域非常稳定时修正。
        if conf is not None and conf[i] > 0.45:
            if np.max(neighbor_logs) - np.min(neighbor_logs) > 0.10:
                continue

        corrected[i] = folded

    return corrected


def smooth_singing_f0(f0, confidence=None):
    """对稳定持续音做轻量自适应平滑，保留滑音、颤音和真实换音。"""
    f0 = np.asarray(f0, dtype=np.float64).copy()
    if f0.size < 5:
        return f0

    conf = None
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float64)
        if conf.shape[0] != f0.shape[0]:
            conf = None

    voiced = f0 > 0
    logf = np.zeros_like(f0)
    logf[voiced] = np.log2(f0[voiced])
    output = f0.copy()
    weights = np.asarray([1.0, 2.0, 4.0, 2.0, 1.0], dtype=np.float64)

    for i in range(2, len(f0) - 2):
        if not voiced[i] or not np.all(voiced[i - 2 : i + 3]):
            continue

        local = logf[i - 2 : i + 3]
        # 有明显换音/快速滑音时不做平滑。
        if np.max(local) - np.min(local) > 0.16:
            continue
        if max(abs(logf[i] - logf[i - 1]), abs(logf[i + 1] - logf[i])) > 0.10:
            continue

        smoothed = float(np.sum(local * weights) / np.sum(weights))
        blend = 0.18
        if conf is not None:
            # RMVPE 置信度较低时增加一点平滑量。
            blend += 0.20 * float(np.clip((0.35 - conf[i]) / 0.35, 0.0, 1.0))
        if f0[i] >= 700.0:
            blend += 0.08
        blend = min(blend, 0.42)
        output[i] = 2.0 ** ((1.0 - blend) * logf[i] + blend * smoothed)

    return output


def process_singing_f0(f0, confidence=None):
    """唱歌模式 F0 后处理：先修八度错误，再轻量稳定持续音。"""
    f0 = np.asarray(f0, dtype=np.float64)
    if f0.size == 0:
        return f0
    f0 = correct_octave_errors(f0, confidence)
    f0 = smooth_singing_f0(f0, confidence)
    return f0


def smooth_retrieval_features(features, f0, amount=0.35):
    """仅在稳定有声音高区间平滑检索特征，减少长音 timbre jumping。"""
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] < 3 or f0 is None:
        return features

    pitch = np.asarray(f0, dtype=np.float64).reshape(-1)
    if pitch.size < 3:
        return features

    # HuBERT 检索特征约为 50 Hz，而连续 F0 为 100 Hz；统一到特征帧数。
    src_x = np.linspace(0.0, 1.0, num=pitch.size, endpoint=True)
    dst_x = np.linspace(0.0, 1.0, num=features.shape[0], endpoint=True)
    pitch = np.interp(dst_x, src_x, pitch)

    output = features.copy()
    logf = np.zeros_like(pitch)
    voiced = pitch > 0
    logf[voiced] = np.log2(pitch[voiced])

    for i in range(1, features.shape[0] - 1):
        if not (voiced[i - 1] and voiced[i] and voiced[i + 1]):
            continue
        local = logf[i - 1 : i + 2]
        # 只处理持续音，辅音附近和明显换音处保持原检索结果。
        if np.max(local) - np.min(local) > (0.8 / 12.0):
            continue
        smoothed = (
            features[i - 1] * 0.25
            + features[i] * 0.50
            + features[i + 1] * 0.25
        )
        output[i] = features[i] * (1.0 - amount) + smoothed * amount

    return output
