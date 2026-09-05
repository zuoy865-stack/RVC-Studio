"""训练验证集拆分、评分与 Early Stopping 状态管理。"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from infer.hubert import hubert_audio_requires_normalization


@contextmanager
def fixed_validation_seed(seed: int):
    """固定验证随机数并在结束后恢复训练随机状态，保证各轮分数可比较。"""

    cpu_state = torch.random.get_rng_state()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    torch.manual_seed(seed)
    if mps_state is not None:
        torch.mps.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


class DatasetView(Dataset):
    """只暴露原数据集的指定样本，并保留分桶采样需要的长度。"""

    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = list(indices)
        self.lengths = [dataset.lengths[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def split_train_validation(
    dataset: Dataset,
    validation_split: float,
    seed: int,
    speaker_index: int | None = None,
) -> tuple[DatasetView, DatasetView]:
    """按音频文件分组并确定性拆分，防止重复样本同时落入训练集和验证集。"""

    if not 0 < validation_split < 1:
        raise ValueError("验证集比例必须大于 0 且小于 1")
    records = getattr(dataset, "audiopaths_and_text", None)
    if not records or len(records) < 2:
        raise ValueError("启用验证时至少需要 2 个有效训练样本")

    grouped: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        speaker = str(record[speaker_index]) if speaker_index is not None else "0"
        grouped.setdefault((speaker, str(record[0])), []).append(index)
    if len(grouped) < 2:
        raise ValueError("启用验证时至少需要 2 个不同的音频文件")

    by_speaker: dict[str, list[tuple[str, str]]] = {}
    for key in grouped:
        by_speaker.setdefault(key[0], []).append(key)
    validation_keys: set[tuple[str, str]] = set()
    for speaker, speaker_keys in sorted(by_speaker.items()):
        keys = sorted(
            speaker_keys,
            key=lambda value: hashlib.sha256(
                f"{seed}:{speaker}:{value[1]}".encode("utf-8")
            ).digest(),
        )
        if len(keys) < 2:
            continue
        validation_count = max(1, int(round(len(keys) * validation_split)))
        validation_count = min(validation_count, len(keys) - 1)
        validation_keys.update(keys[:validation_count])
    if not validation_keys:
        keys = sorted(
            grouped,
            key=lambda value: hashlib.sha256(
                f"{seed}:{value[0]}:{value[1]}".encode("utf-8")
            ).digest(),
        )
        validation_keys.add(keys[0])
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for key, indices in grouped.items():
        target = validation_indices if key in validation_keys else train_indices
        target.extend(indices)
    return DatasetView(dataset, train_indices), DatasetView(dataset, validation_indices)


def _resample_wave(wave: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return wave
    target_length = max(1, int(round(wave.shape[-1] * target_rate / source_rate)))
    return F.interpolate(
        wave.reshape(1, 1, -1), size=target_length, mode="linear", align_corners=False
    ).reshape(1, -1)


def prepare_hubert_audio(
    wave: torch.Tensor, sample_rate: int, device: torch.device
) -> torch.Tensor:
    """把单声道音频转换为 HuBERT 所需的 16 kHz MPS 张量。"""

    audio = wave.float().reshape(1, -1).to(device)
    audio = _resample_wave(audio, sample_rate, 16000)
    if hubert_audio_requires_normalization():
        audio = F.layer_norm(audio, audio.shape)
    return audio


def extract_validation_embeddings(model, audio_16k: torch.Tensor, version: str):
    """一次前向提取内容特征，并融合频谱包络与浅层声学说话人表征。"""

    outputs = model(
        input_values=audio_16k,
        attention_mask=None,
        output_hidden_states=True,
        return_dict=True,
    )
    if version == "v1":
        content = model.final_proj(outputs.hidden_states[9])
    else:
        content = outputs.last_hidden_state
    shallow = outputs.hidden_states[3].float()
    shallow_speaker = torch.cat(
        (shallow.mean(dim=1), shallow.std(dim=1, unbiased=False)), dim=-1
    )
    shallow_speaker = F.normalize(shallow_speaker, dim=-1, eps=1e-8)

    window = torch.hann_window(1024, device=audio_16k.device, dtype=torch.float32)
    spectrum = torch.stft(
        audio_16k.float(),
        n_fft=1024,
        hop_length=160,
        win_length=1024,
        window=window,
        return_complex=True,
    ).abs()
    log_spectrum = torch.log1p(spectrum)
    envelope = log_spectrum.mean(dim=-1)
    variation = log_spectrum.std(dim=-1, unbiased=False)
    acoustic = torch.cat(
        (
            F.interpolate(envelope.unsqueeze(1), size=96, mode="linear").squeeze(1),
            F.interpolate(variation.unsqueeze(1), size=96, mode="linear").squeeze(1),
        ),
        dim=-1,
    )
    acoustic = acoustic - acoustic.mean(dim=-1, keepdim=True)
    acoustic = F.normalize(acoustic, dim=-1, eps=1e-8)
    speaker = F.normalize(
        torch.cat((acoustic * 0.8, shallow_speaker * 0.2), dim=-1),
        dim=-1,
        eps=1e-8,
    )
    return content.float(), speaker


def content_preservation_score(
    source_features: torch.Tensor, converted_features: torch.Tensor
) -> torch.Tensor:
    """比较转换前后的 HuBERT 帧级内容，允许输出时长存在轻微变化。"""

    frames = min(source_features.shape[1], converted_features.shape[1])
    frames = max(1, frames)
    source = F.interpolate(
        source_features.transpose(1, 2), size=frames, mode="linear", align_corners=False
    ).transpose(1, 2)
    converted = F.interpolate(
        converted_features.transpose(1, 2), size=frames, mode="linear", align_corners=False
    ).transpose(1, 2)
    return _bounded_cosine(source, converted, dim=-1).mean()


def f0_preservation_score(
    source_wave: torch.Tensor,
    source_rate: int,
    converted_wave: torch.Tensor,
    converted_rate: int,
) -> torch.Tensor:
    """比较转换前后的 F0 轮廓和音高稳定性，不比较音量或音色。"""

    source_f0 = estimate_f0_track(source_wave.reshape(1, -1), source_rate).clamp_min(1.0)
    converted_f0 = estimate_f0_track(converted_wave.reshape(1, -1), converted_rate).clamp_min(1.0)
    frames = max(1, min(source_f0.shape[-1], converted_f0.shape[-1]))
    source_f0 = F.interpolate(source_f0.unsqueeze(1), size=frames, mode="linear").squeeze(1)
    converted_f0 = F.interpolate(
        converted_f0.unsqueeze(1), size=frames, mode="linear"
    ).squeeze(1)
    source_log = torch.log2(source_f0)
    converted_log = torch.log2(converted_f0)
    contour = torch.exp(-1.5 * (source_log - converted_log).abs().mean())
    if frames > 1:
        source_delta = source_log.diff(dim=-1).std(unbiased=False)
        converted_delta = converted_log.diff(dim=-1).std(unbiased=False)
        stability = torch.exp(-2.0 * (source_delta - converted_delta).abs())
    else:
        stability = contour.new_tensor(1.0)
    return (0.8 * contour + 0.2 * stability).clamp(0.0, 1.0)


def spectral_artifact_score(
    converted_wave: torch.Tensor,
    converted_rate: int,
    source_wave: torch.Tensor,
    source_rate: int,
) -> torch.Tensor:
    """检测高频噪声、宽带金属感、削波和异常瞬态。"""

    converted = converted_wave.float().reshape(-1)
    source = source_wave.float().reshape(-1)

    def descriptors(wave: torch.Tensor, sample_rate: int):
        frame = min(2048, wave.shape[-1])
        frame = max(256, 2 ** int(math.floor(math.log2(max(256, frame)))))
        if wave.shape[-1] < frame:
            wave = F.pad(wave, (0, frame - wave.shape[-1]))
        hop = max(64, frame // 4)
        window = torch.hann_window(frame, device=wave.device, dtype=torch.float32)
        spectrum = torch.stft(
            wave,
            n_fft=frame,
            hop_length=hop,
            win_length=frame,
            window=window,
            return_complex=True,
        ).abs().clamp_min(1e-7)
        split = max(1, int(spectrum.shape[0] * min(0.95, 6000.0 / (sample_rate / 2))))
        high_ratio = spectrum[split:].sum() / spectrum.sum().clamp_min(1e-7)
        flatness = torch.exp(torch.log(spectrum).mean(dim=0)) / spectrum.mean(dim=0)
        flux = spectrum.diff(dim=-1).abs().mean() / spectrum.mean().clamp_min(1e-7)
        return high_ratio, flatness.mean(), flux

    converted_high, converted_flatness, converted_flux = descriptors(
        converted, converted_rate
    )
    source_high, source_flatness, source_flux = descriptors(source, source_rate)
    high_penalty = F.relu(converted_high - source_high * 1.8 - 0.01) * 20.0
    flatness_penalty = F.relu(converted_flatness - source_flatness * 1.5 - 0.03) * 8.0
    flux_penalty = F.relu(converted_flux - source_flux * 2.0 - 0.1) * 2.0
    clipping = (converted.abs() >= 0.995).float().mean() * 50.0
    return torch.exp(
        -(high_penalty + flatness_penalty + flux_penalty + clipping)
    ).clamp(0.0, 1.0)


def speaker_similarity_with_leakage_penalty(
    converted_speaker: torch.Tensor,
    target_speaker: torch.Tensor,
    source_speaker: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """以目标相似度为主体，并用输出仍接近源音色的程度施加泄漏惩罚。"""

    target_similarity = _bounded_cosine(converted_speaker, target_speaker, dim=-1).mean()
    source_similarity = _bounded_cosine(converted_speaker, source_speaker, dim=-1).mean()
    target_margin = ((target_similarity - source_similarity + 1.0) * 0.5).clamp(0.0, 1.0)
    score = target_similarity * (0.5 + 0.5 * target_margin)
    return score.clamp(0.0, 1.0), target_similarity, source_similarity


def _vocal_probe(
    frequencies: list[float],
    duration: float,
    formants: tuple[float, float],
    seed: int,
) -> np.ndarray:
    """生成带不同共振峰的确定性类人声唱歌探针。"""

    sample_rate = 16000
    count = int(round(duration * sample_rate))
    time = np.arange(count, dtype=np.float64) / sample_rate
    note_length = duration / len(frequencies)
    fundamental = np.asarray(
        [frequencies[min(len(frequencies) - 1, int(value / note_length))] for value in time]
    )
    phase = 2 * np.pi * np.cumsum(fundamental) / sample_rate
    rng = np.random.default_rng(seed)
    audio = np.zeros_like(time)
    for harmonic in range(1, 25):
        harmonic_frequency = fundamental * harmonic
        envelope = sum(
            np.exp(-0.5 * ((harmonic_frequency - formant) / (180 + formant * 0.08)) ** 2)
            for formant in formants
        )
        audio += envelope * np.sin(harmonic * phase + rng.uniform(-np.pi, np.pi)) / harmonic
    attack = np.minimum(1.0, time / 0.04)
    release = np.minimum(1.0, (duration - time) / 0.06)
    audio *= np.clip(attack * release, 0.0, 1.0)
    audio /= max(1e-6, np.max(np.abs(audio)))
    return (audio * 0.7).astype(np.float32)


def ensure_conversion_probes(model_dir: str | Path) -> list[Path]:
    """创建低音、中音、高音和长音四种固定非目标音色探针。"""

    probe_dir = Path(model_dir) / "conversion_validation_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("builtin_bass.wav", [92.5, 110.0, 123.5, 98.0], 2.4, (450.0, 1050.0), 11),
        ("builtin_mid.wav", [174.6, 220.0, 261.6, 196.0], 2.4, (700.0, 1550.0), 23),
        ("builtin_high.wav", [329.6, 392.0, 523.3, 440.0], 2.4, (900.0, 2450.0), 37),
        ("builtin_sustain.wav", [220.0], 3.2, (560.0, 1750.0), 41),
    )
    paths: list[Path] = []
    for name, frequencies, duration, formants, seed in definitions:
        path = probe_dir / name
        if not path.is_file():
            sf.write(path, _vocal_probe(frequencies, duration, formants, seed), 16000)
        paths.append(path)
    return paths


def load_probe_audio(paths: list[Path], maximum_custom: int = 8):
    """读取自带或用户提供的转换验证音频并统一为单声道浮点格式。"""

    result = []
    for path in paths[:maximum_custom]:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.size < max(512, sample_rate // 2):
            continue
        audio = audio[: sample_rate * 8]
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak
        result.append((path.name, torch.from_numpy(np.asarray(audio, dtype=np.float32)), int(sample_rate)))
    return result


def _bounded_cosine(left: torch.Tensor, right: torch.Tensor, dim: int) -> torch.Tensor:
    similarity = F.cosine_similarity(left.float(), right.float(), dim=dim, eps=1e-8)
    return ((similarity + 1.0) * 0.5).clamp(0.0, 1.0)


def estimate_f0_track(wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """用短时频谱主峰估计 F0 轨迹，避免验证阶段引入额外模型。"""

    frame_length = min(wave.shape[-1], max(256, int(sample_rate * 0.04)))
    if frame_length < 64:
        return wave.new_zeros((wave.shape[0], 1), dtype=torch.float32)
    hop_length = max(64, int(sample_rate * 0.01))
    if wave.shape[-1] < frame_length:
        wave = F.pad(wave, (0, frame_length - wave.shape[-1]))
    frames = wave.float().unfold(-1, frame_length, hop_length)
    window = torch.hann_window(frame_length, device=wave.device, dtype=torch.float32)
    magnitude = torch.fft.rfft(frames * window, dim=-1).abs()
    minimum_bin = max(1, int(math.ceil(50.0 * frame_length / sample_rate)))
    maximum_bin = min(magnitude.shape[-1] - 1, int(1100.0 * frame_length / sample_rate))
    if maximum_bin <= minimum_bin:
        return wave.new_zeros((wave.shape[0], frames.shape[1]), dtype=torch.float32)
    peak = magnitude[..., minimum_bin : maximum_bin + 1].argmax(dim=-1) + minimum_bin
    return peak.float() * (float(sample_rate) / frame_length)


def validation_components(
    target_mel: torch.Tensor,
    generated_mel: torch.Tensor,
    target_wave: torch.Tensor,
    generated_wave: torch.Tensor,
    sample_rate: int,
) -> dict[str, torch.Tensor]:
    """计算四个归一化验证分量，所有分量的范围均为 0 到 1。"""

    mel_frames = min(target_mel.shape[-1], generated_mel.shape[-1])
    wave_samples = min(target_wave.shape[-1], generated_wave.shape[-1])
    target_mel = target_mel[..., :mel_frames].float()
    generated_mel = generated_mel[..., :mel_frames].float()
    target_wave = target_wave[..., :wave_samples].float()
    generated_wave = generated_wave[..., :wave_samples].float()

    target_timbre = torch.cat(
        (target_mel.mean(dim=-1), target_mel.std(dim=-1, unbiased=False)), dim=1
    )
    generated_timbre = torch.cat(
        (generated_mel.mean(dim=-1), generated_mel.std(dim=-1, unbiased=False)), dim=1
    )
    timbre = _bounded_cosine(target_timbre, generated_timbre, dim=1).mean()

    target_f0 = estimate_f0_track(target_wave.squeeze(1), sample_rate).clamp_min(1.0)
    generated_f0 = estimate_f0_track(generated_wave.squeeze(1), sample_rate).clamp_min(1.0)
    f0_frames = min(target_f0.shape[-1], generated_f0.shape[-1])
    target_log_f0 = torch.log2(target_f0[..., :f0_frames])
    generated_log_f0 = torch.log2(generated_f0[..., :f0_frames])
    f0_consistency = torch.exp(
        -1.5 * (target_log_f0 - generated_log_f0).abs().mean(dim=-1)
    )
    if f0_frames > 1:
        target_stability = target_log_f0.diff(dim=-1).std(dim=-1, unbiased=False)
        generated_stability = generated_log_f0.diff(dim=-1).std(dim=-1, unbiased=False)
    else:
        target_stability = target_log_f0.new_zeros(target_log_f0.shape[0])
        generated_stability = generated_log_f0.new_zeros(generated_log_f0.shape[0])
    f0_stability = torch.exp(-2.0 * (target_stability - generated_stability).abs())
    f0 = (0.75 * f0_consistency + 0.25 * f0_stability).clamp(0.0, 1.0).mean()

    target_energy = target_mel.mean(dim=1)
    generated_energy = generated_mel.mean(dim=1)
    target_flux = target_mel.diff(dim=-1).abs().mean(dim=1)
    generated_flux = generated_mel.diff(dim=-1).abs().mean(dim=1)
    content_frames = min(target_flux.shape[-1], generated_flux.shape[-1])
    target_content = torch.stack(
        (target_energy[..., :content_frames], target_flux[..., :content_frames]), dim=1
    )
    generated_content = torch.stack(
        (
            generated_energy[..., :content_frames],
            generated_flux[..., :content_frames],
        ),
        dim=1,
    )
    content = _bounded_cosine(
        target_content.flatten(1),
        generated_content.flatten(1),
        dim=1,
    ).mean()

    spectral_scale = target_mel.abs().mean().clamp_min(1e-4)
    spectral_error = (target_mel - generated_mel).abs().mean() / spectral_scale
    spectral = torch.exp(-spectral_error).clamp(0.0, 1.0)
    return {
        "timbre": timbre,
        "f0": f0,
        "content": content,
        "spectral": spectral,
    }


def weighted_validation_score(components: dict[str, float]) -> float:
    """按首版约定权重合成 Validation Score。"""

    speaker = components.get("speaker", components.get("timbre"))
    if speaker is None:
        raise KeyError("Validation Score 缺少 speaker 分量")
    return (
        0.40 * speaker
        + 0.25 * components["f0"]
        + 0.20 * components["content"]
        + 0.15 * components["spectral"]
    )


def summarize_training_health(
    generator_losses: list[float], discriminator_losses: list[float]
) -> dict[str, Any]:
    """仅用 G/D loss 判断数值健康，不把它们混入模型排名分数。"""

    if not generator_losses or not discriminator_losses:
        return {
            "healthy": False,
            "generator_loss": None,
            "discriminator_loss": None,
            "reason": "当前 Epoch 没有可用的 G/D loss",
        }
    generator = float(np.mean(generator_losses))
    discriminator = float(np.mean(discriminator_losses))
    if not math.isfinite(generator) or not math.isfinite(discriminator):
        reason = "G/D loss 出现 NaN 或 Inf"
    elif max(map(abs, generator_losses)) > 10000 or max(
        map(abs, discriminator_losses)
    ) > 10000:
        reason = "G/D loss 数值爆炸"
    elif abs(discriminator) < 1e-6:
        reason = "Discriminator loss 接近零，疑似 GAN 塌缩"
    else:
        ratio = max(
            abs(generator) / max(abs(discriminator), 1e-6),
            abs(discriminator) / max(abs(generator), 1e-6),
        )
        reason = "G/D 严重失衡" if ratio > 500 else ""
    return {
        "healthy": not reason,
        "generator_loss": generator,
        "discriminator_loss": discriminator,
        "reason": reason,
    }


@dataclass(frozen=True)
class ValidationDecision:
    is_best: bool
    significant_improvement: bool
    should_stop: bool
    checkpoint_eligible: bool


class ValidationTracker:
    """追踪达到 min_delta 的最佳分数、最佳轮次与连续未提升次数。"""

    SCORE_VERSION = 2

    def __init__(self, state_path: str | Path, patience: int, min_delta: float):
        self.state_path = Path(state_path)
        self.patience = max(1, int(patience))
        self.min_delta = max(0.0, float(min_delta))
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.patience_score = float("-inf")
        self.bad_count = 0
        self.top_models: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("score_version") != self.SCORE_VERSION:
                return
            if state.get("best_score") is not None:
                self.best_score = float(state["best_score"])
            self.best_epoch = int(state.get("best_epoch", 0))
            patience_score = state.get("patience_score")
            self.patience_score = (
                float(patience_score) if patience_score is not None else self.best_score
            )
            self.bad_count = int(state.get("bad_count", 0))
            self.top_models = list(state.get("top_models", []))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.best_score = float("-inf")
            self.best_epoch = 0
            self.patience_score = float("-inf")
            self.bad_count = 0
            self.top_models = []

    def observe(
        self, score: float, epoch: int, checkpoint_eligible: bool = True
    ) -> ValidationDecision:
        significant = checkpoint_eligible and (
            not math.isfinite(self.patience_score)
            or score >= self.patience_score + self.min_delta
        )
        if significant:
            self.best_score = score
            self.best_epoch = epoch
            self.patience_score = score
            self.bad_count = 0
        else:
            self.bad_count += 1
        return ValidationDecision(
            is_best=significant,
            significant_improvement=significant,
            should_stop=self.bad_count >= self.patience,
            checkpoint_eligible=checkpoint_eligible,
        )

    def save(self) -> None:
        state = {
            "score_version": self.SCORE_VERSION,
            "best_score": self.best_score if math.isfinite(self.best_score) else None,
            "best_epoch": self.best_epoch,
            "patience_score": (
                self.patience_score if math.isfinite(self.patience_score) else None
            ),
            "bad_count": self.bad_count,
            "top_models": self.top_models,
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.state_path)
