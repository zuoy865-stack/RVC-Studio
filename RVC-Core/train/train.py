import os
import gc
import json
import logging
import math
import shutil
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")

warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.GradScaler.*is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.autocast.*is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Grad strides do not match bucket view strides.*",
    category=UserWarning,
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

import datetime

from train import utils

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

from configs.config import get_training_dtype
from rvc_core.device import require_mps_module, select_mps, synchronize
from i18n.i18n import I18nAuto

i18n = I18nAuto()

training_dtype = get_training_dtype()
training_is_half = training_dtype == torch.float16
training_device = select_mps(precision="float32", strict=True).device

from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.module import commons
from train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.module.models import MultiPeriodDiscriminator
    from infer.module.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.module.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from infer.module.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from train.process_ckpt import savee
from infer.hubert import load_hubert_model
from train.validation import (
    ValidationTracker,
    content_preservation_score,
    ensure_conversion_probes,
    estimate_f0_track,
    extract_validation_embeddings,
    f0_preservation_score,
    fixed_validation_seed,
    load_probe_audio,
    prepare_hubert_audio,
    speaker_similarity_with_leakage_penalty,
    spectral_artifact_score,
    split_train_validation,
    summarize_training_health,
    validation_components,
    weighted_validation_score,
)

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def load_pretrained_generator(model, path):
    target = model.module if hasattr(model, "module") else model
    saved_state = torch.load(path, map_location="cpu")["model"]
    current_state = target.state_dict()
    embedding_key = "emb_g.weight"
    if embedding_key in saved_state and embedding_key in current_state:
        saved_embedding = saved_state[embedding_key]
        current_embedding = current_state[embedding_key]
        if saved_embedding.shape != current_embedding.shape:
            compatible = (
                saved_embedding.dim() == current_embedding.dim()
                and saved_embedding.shape[1:] == current_embedding.shape[1:]
            )
            if compatible:
                expanded = current_embedding.clone()
                rows = min(saved_embedding.shape[0], current_embedding.shape[0])
                expanded[:rows].copy_(saved_embedding[:rows])
                saved_state[embedding_key] = expanded
    return target.load_state_dict(saved_state)


def main():
    n_gpus = 1
    logger = utils.get_logger(hps.model_dir)
    logger.info("严格 Apple GPU 训练：device=%s dtype=%s CPU fallback=disabled", training_device, training_dtype)
    run(0, 1, hps, logger, False)


def run(rank, n_gpus, hps, logger, use_ddp):
    global global_step
    if rank == 0:
        # logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    if use_ddp:
        dist.init_process_group(
            backend="gloo", init_method="env://?use_libuv=False", world_size=n_gpus, rank=rank
        )
    torch.manual_seed(hps.train.seed)
    if hps.if_f0 == 1:
        complete_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        complete_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    validation_enabled = bool(getattr(hps, "validation_enabled", False))
    eval_dataset = None
    if validation_enabled:
        if not 0 <= hps.validation_speaker_id < hps.model.spk_embed_dim:
            raise ValueError(
                f"Conversion Validation 目标说话人 ID 超出范围：{hps.validation_speaker_id}"
            )
        train_dataset, eval_dataset = split_train_validation(
            complete_dataset,
            hps.validation_split,
            hps.train.seed,
            speaker_index=4 if hps.if_f0 == 1 else 2,
        )
        logger.info(
            "验证集拆分完成：训练样本=%s，验证样本=%s，比例=%.3f",
            len(train_dataset),
            len(eval_dataset),
            hps.validation_split,
        )
    else:
        train_dataset = complete_dataset
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=min(2, max(1, os.cpu_count() or 1)),
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=2,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=hps.train.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn,
        )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=training_is_half,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=training_is_half,
        )
    net_g = net_g.to(device=training_device, dtype=training_dtype)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    net_d = net_d.to(device=training_device, dtype=training_dtype)
    require_mps_module(net_g, "RVC training generator")
    require_mps_module(net_d, "RVC training discriminator")
    synchronize(training_device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if use_ddp:
        if torch.cuda.is_available():
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)

    resume_d_pattern = "D_[0-9]*.pth" if validation_enabled else "D_*.pth"
    resume_g_pattern = "G_[0-9]*.pth" if validation_enabled else "G_*.pth"
    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, resume_d_pattern), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info(i18n("已恢复判别器检查点"))
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, resume_g_pattern), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except Exception:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info(i18n("已加载生成器预训练模型：%s") % hps.pretrainG)
            logger.info(load_pretrained_generator(net_g, hps.pretrainG))
        if hps.pretrainD != "":
            if rank == 0:
                logger.info(i18n("已加载判别器预训练模型：%s") % hps.pretrainD)
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=training_is_half)

    cache = []
    tracker = None
    if validation_enabled and rank == 0:
        tracker = ValidationTracker(
            Path(hps.model_dir) / "validation_state.json",
            hps.early_stopping_patience,
            hps.early_stopping_min_delta,
        )
    last_validation_score = None
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            epoch_health = train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            epoch_health = train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, eval_loader],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()
        if not validation_enabled or rank != 0:
            continue

        should_validate = (
            epoch % hps.validation_interval == 0 or epoch >= hps.total_epoch
        )
        validation_score = last_validation_score
        checkpoint_eligible = bool(epoch_health and epoch_health["healthy"])
        checkpoint_reason = epoch_health["reason"] if epoch_health else "缺少 G/D 健康状态"
        if should_validate:
            target_components = evaluate_validation(net_g, eval_loader, hps)
            conversion_components = evaluate_conversion_validation(
                net_g, eval_loader, hps
            )
            target_reconstruction_score = weighted_validation_score(
                target_components
            )
            components = {
                "speaker": conversion_components["speaker"],
                "f0": conversion_components["f0"],
                "content": conversion_components["content"],
                "spectral": 0.7 * conversion_components["spectral"]
                + 0.3 * target_components["spectral"],
            }
            validation_score = weighted_validation_score(components)
            score_is_finite = math.isfinite(validation_score)
            last_validation_score = validation_score if score_is_finite else None
            checkpoint_eligible = bool(epoch_health and epoch_health["healthy"])
            checkpoint_eligible = checkpoint_eligible and bool(
                conversion_components["finite"]
            )
            checkpoint_eligible = checkpoint_eligible and score_is_finite
            checkpoint_eligible = checkpoint_eligible and models_are_finite(
                net_g, net_d
            )
            if not checkpoint_eligible and not checkpoint_reason:
                checkpoint_reason = (
                    "模型参数、转换输出或 Validation Score 出现 NaN/Inf"
                )
            decision = tracker.observe(
                validation_score if score_is_finite else float("-inf"),
                epoch,
                checkpoint_eligible=checkpoint_eligible,
            )
            if hps.auto_best_model and checkpoint_eligible:
                if decision.significant_improvement:
                    save_best_training_checkpoints(
                        net_g, net_d, optim_g, optim_d, hps, epoch, global_step
                    )
                update_top_k_models(
                    net_g, hps, tracker, validation_score, epoch, global_step
                )
            tracker.save()
            utils.summarize(
                writer=writer_eval,
                global_step=global_step,
                scalars={
                    "validation/score": validation_score,
                    "validation/speaker": components["speaker"],
                    "validation/f0": components["f0"],
                    "validation/content": components["content"],
                    "validation/spectral": components["spectral"],
                    "validation/speaker_target_similarity": conversion_components[
                        "speaker_target_similarity"
                    ],
                    "validation/source_leakage_similarity": conversion_components[
                        "source_leakage_similarity"
                    ],
                    "validation/target_reconstruction_spectral": target_components[
                        "spectral"
                    ],
                    "validation/target_reconstruction_score": target_reconstruction_score,
                    "validation/target_reconstruction_timbre": target_components[
                        "timbre"
                    ],
                    "health/generator_loss": epoch_health["generator_loss"],
                    "health/discriminator_loss": epoch_health[
                        "discriminator_loss"
                    ],
                },
            )
            logger.info(
                "Validation Score=%.6f（目标说话人=%.6f，F0=%.6f，HuBERT内容=%.6f，频谱=%.6f）；"
                "target_similarity=%.6f，source_leakage=%.6f，探针=%s；"
                "best=%.6f@epoch %s；bad_count=%s/%s；checkpoint=%s",
                validation_score,
                components["speaker"],
                components["f0"],
                components["content"],
                components["spectral"],
                conversion_components["speaker_target_similarity"],
                conversion_components["source_leakage_similarity"],
                conversion_components["probe_count"],
                tracker.best_score,
                tracker.best_epoch,
                tracker.bad_count,
                tracker.patience,
                "允许进入 Top-K" if checkpoint_eligible else "健康门禁拒绝",
            )
            if not checkpoint_eligible:
                logger.warning(
                    "当前 checkpoint 禁止进入 Top-K：%s",
                    checkpoint_reason,
                )
            if not score_is_finite:
                validation_score = None
        emit_training_status(
            epoch,
            hps,
            validation_score,
            tracker,
            epoch_health,
            checkpoint_eligible,
            checkpoint_reason,
        )

        if should_validate and hps.early_stopping and decision.should_stop:
            logger.info(
                "Early Stopping：连续 %s 次验证未达到 min_delta=%.6f，训练在 epoch %s 自动停止；"
                "最佳模型保持为 epoch %s（score=%.6f）",
                tracker.patience,
                tracker.min_delta,
                epoch,
                tracker.best_epoch,
                tracker.best_score,
            )
            break
        if epoch >= hps.total_epoch:
            save_final_model(net_g, hps, epoch, logger)
            break

    if validation_enabled and rank == 0:
        writer.close()
        writer_eval.close()


def evaluate_validation(net_g, eval_loader, hps):
    """在 MPS 上执行只读验证，验证集不会参与梯度计算。"""

    if eval_loader is None or len(eval_loader.dataset) == 0:
        raise RuntimeError("验证集为空，无法计算 Validation Score")
    totals = {"timbre": 0.0, "f0": 0.0, "content": 0.0, "spectral": 0.0}
    sample_count = 0
    net_g.eval()
    with fixed_validation_seed(hps.train.seed), torch.no_grad():
        for info in eval_loader:
            if hps.if_f0 == 1:
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info
                pitch = pitch.to(training_device)
                pitchf = pitchf.to(training_device)
            else:
                (
                    phone,
                    phone_lengths,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info
            phone = phone.to(training_device)
            phone_lengths = phone_lengths.to(training_device)
            spec = spec.to(training_device)
            spec_lengths = spec_lengths.to(training_device)
            wave = wave.to(training_device)
            sid = sid.to(training_device)

            with autocast(enabled=training_is_half):
                if hps.if_f0 == 1:
                    y_hat, ids_slice, *_ = net_g(
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        sid,
                    )
                else:
                    y_hat, ids_slice, *_ = net_g(
                        phone, phone_lengths, spec, spec_lengths, sid
                    )
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                target_mel = commons.slice_segments(
                    mel,
                    ids_slice,
                    hps.train.segment_size // hps.data.hop_length,
                )
            with autocast(enabled=False):
                generated_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            target_wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )
            batch_components = validation_components(
                target_mel,
                generated_mel,
                target_wave,
                y_hat,
                hps.data.sampling_rate,
            )
            batch_size = phone.shape[0]
            for key, value in batch_components.items():
                totals[key] += float(value.detach().cpu()) * batch_size
            sample_count += batch_size
    net_g.train()
    return {key: value / sample_count for key, value in totals.items()}


def _validation_reference_paths(eval_loader, hps):
    """优先从目标说话人的 Target Validation 中选择参考音频。"""

    view = eval_loader.dataset
    dataset = getattr(view, "dataset", view)
    indices = getattr(view, "indices", range(len(dataset)))
    speaker_index = 4 if hps.if_f0 == 1 else 2
    selected = []
    for index in indices:
        record = dataset.audiopaths_and_text[index]
        path = Path(record[0])
        if int(record[speaker_index]) == hps.validation_speaker_id:
            selected.append(path)
    if not selected:
        for record in dataset.audiopaths_and_text:
            if int(record[speaker_index]) == hps.validation_speaker_id:
                selected.append(Path(record[0]))
    return selected[:6]


def _target_speaker_reference(hubert_model, eval_loader, hps):
    references = load_probe_audio(_validation_reference_paths(eval_loader, hps), 6)
    if not references:
        raise RuntimeError("找不到可用于 Speaker Similarity 的目标说话人参考音频")
    embeddings = []
    with torch.no_grad():
        for _, wave, sample_rate in references:
            audio = prepare_hubert_audio(wave, sample_rate, training_device)
            _, speaker = extract_validation_embeddings(
                hubert_model, audio, hps.version
            )
            embeddings.append(speaker)
    return F.normalize(torch.stack(embeddings).mean(dim=0), dim=-1, eps=1e-8)


def _conversion_probe_paths(hps):
    paths = ensure_conversion_probes(hps.model_dir)
    custom_dir = str(getattr(hps, "conversion_validation_dir", "")).strip()
    if custom_dir:
        supported = {".wav", ".flac", ".aiff", ".aif", ".ogg"}
        paths.extend(
            path
            for path in sorted(Path(custom_dir).iterdir())
            if path.is_file() and path.suffix.lower() in supported
        )
    return paths


def _resample_for_validation(wave, source_rate, target_rate):
    wave = wave.float().reshape(1, 1, -1).to(training_device)
    if source_rate == target_rate:
        return wave.reshape(1, -1)
    length = max(1, int(round(wave.shape[-1] * target_rate / source_rate)))
    return F.interpolate(
        wave, size=length, mode="linear", align_corners=False
    ).reshape(1, -1)


def _f0_inputs_for_generator(source_16k, frames):
    continuous = estimate_f0_track(source_16k, 16000)
    continuous = F.interpolate(
        continuous.unsqueeze(1), size=frames, mode="linear", align_corners=False
    ).squeeze(1)
    mel = 1127.0 * torch.log1p(continuous / 700.0)
    mel_min = 1127.0 * math.log1p(50.0 / 700.0)
    mel_max = 1127.0 * math.log1p(1100.0 / 700.0)
    coarse = torch.round((mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0)
    return coarse.clamp(1, 255).long(), continuous.float()


def _convert_validation_probe(net_g, source_content, source_wave, source_rate, hps):
    phone = F.interpolate(
        source_content.transpose(1, 2), scale_factor=2, mode="linear", align_corners=False
    ).transpose(1, 2)
    frames = phone.shape[1]
    lengths = torch.tensor([frames], device=training_device, dtype=torch.long)
    speaker = torch.tensor(
        [hps.validation_speaker_id], device=training_device, dtype=torch.long
    )
    if hps.if_f0 == 1:
        source_16k = _resample_for_validation(source_wave, source_rate, 16000)
        pitch, pitchf = _f0_inputs_for_generator(source_16k, frames)
        converted = net_g.infer(
            phone,
            lengths,
            pitch,
            pitchf,
            speaker,
            noise_scale=0.35,
        )[0]
    else:
        converted = net_g.infer(
            phone, lengths, speaker, noise_scale=0.35
        )[0]
    return converted[0, 0].float()


def evaluate_conversion_validation(net_g, eval_loader, hps):
    """用非目标音色探针执行真实转换并计算四项排名指标。"""

    hubert_model = load_hubert_model(training_device, is_half=False)
    net_g.eval()
    try:
        probes = load_probe_audio(_conversion_probe_paths(hps), 12)
        if len(probes) < 4:
            raise RuntimeError("Conversion Validation 至少需要 4 个不同音色/音域的探针")
        target_speaker = _target_speaker_reference(hubert_model, eval_loader, hps)
        totals = {
            "speaker": 0.0,
            "speaker_target_similarity": 0.0,
            "source_leakage_similarity": 0.0,
            "f0": 0.0,
            "content": 0.0,
            "spectral": 0.0,
        }
        finite = True
        valid_count = 0
        with fixed_validation_seed(hps.train.seed + 991), torch.no_grad():
            for _, source_wave_cpu, source_rate in probes:
                source_audio = prepare_hubert_audio(
                    source_wave_cpu, source_rate, training_device
                )
                source_content, source_speaker = extract_validation_embeddings(
                    hubert_model, source_audio, hps.version
                )
                converted = _convert_validation_probe(
                    net_g,
                    source_content,
                    source_wave_cpu,
                    source_rate,
                    hps,
                )
                if not bool(torch.isfinite(converted).all().item()):
                    finite = False
                    continue
                valid_count += 1
                converted_audio = prepare_hubert_audio(
                    converted, hps.data.sampling_rate, training_device
                )
                converted_content, converted_speaker = extract_validation_embeddings(
                    hubert_model, converted_audio, hps.version
                )
                speaker, target_similarity, source_similarity = (
                    speaker_similarity_with_leakage_penalty(
                        converted_speaker, target_speaker, source_speaker
                    )
                )
                values = {
                    "speaker": speaker,
                    "speaker_target_similarity": target_similarity,
                    "source_leakage_similarity": source_similarity,
                    "f0": f0_preservation_score(
                        source_wave_cpu.to(training_device),
                        source_rate,
                        converted,
                        hps.data.sampling_rate,
                    ),
                    "content": content_preservation_score(
                        source_content, converted_content
                    ),
                    "spectral": spectral_artifact_score(
                        converted,
                        hps.data.sampling_rate,
                        source_wave_cpu.to(training_device),
                        source_rate,
                    ),
                }
                for key, value in values.items():
                    totals[key] += float(value.detach().cpu())
        valid_count = max(1, valid_count)
        result = {key: value / valid_count for key, value in totals.items()}
        result["finite"] = finite
        result["probe_count"] = len(probes)
        return result
    finally:
        net_g.train()
        del hubert_model
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _generator_state(net_g):
    return net_g.module.state_dict() if hasattr(net_g, "module") else net_g.state_dict()


def models_are_finite(*models):
    """在保存前检查 G/D 的全部参数，阻止非有限 checkpoint 入选。"""

    with torch.no_grad():
        for model in models:
            target = model.module if hasattr(model, "module") else model
            for parameter in target.parameters():
                if not bool(torch.isfinite(parameter).all().item()):
                    return False
    return True


def save_best_training_checkpoints(
    net_g, net_d, optim_g, optim_d, hps, epoch, step
):
    """保存可恢复训练的最佳 G/D 检查点及其稳定别名。"""

    model_dir = Path(hps.model_dir)
    generator_tagged = model_dir / f"G_best_e{epoch}_s{step}.pth"
    discriminator_tagged = model_dir / f"D_best_e{epoch}_s{step}.pth"
    utils.save_checkpoint(
        net_g, optim_g, hps.train.learning_rate, epoch, str(generator_tagged)
    )
    utils.save_checkpoint(
        net_d, optim_d, hps.train.learning_rate, epoch, str(discriminator_tagged)
    )
    shutil.copy2(generator_tagged, model_dir / "G_best.pth")
    shutil.copy2(discriminator_tagged, model_dir / "D_best.pth")


def update_top_k_models(net_g, hps, tracker, score, epoch, step):
    """把当前候选模型加入 Top-K，并让 best 别名始终指向第一名。"""

    weights_dir = Path("assets/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        item
        for item in tracker.top_models
        if Path(str(item.get("path", ""))).is_file()
    ]
    existing.sort(key=lambda item: float(item["score"]), reverse=True)
    top_k = max(1, int(hps.best_top_k))
    qualifies = len(existing) < top_k or score > float(existing[-1]["score"])
    if not qualifies:
        for item in existing[top_k:]:
            path = Path(str(item["path"]))
            if path.is_file():
                path.unlink()
        tracker.top_models = existing[:top_k]
        return

    candidate_name = f"{hps.name}_best_candidate_e{epoch}_s{step}"
    result = savee(
        _generator_state(net_g),
        hps.sample_rate,
        hps.if_f0,
        candidate_name,
        epoch,
        hps.version,
        hps,
    )
    candidate_path = weights_dir / f"{candidate_name}.pth"
    if not candidate_path.is_file():
        raise RuntimeError(f"导出最佳推理模型失败：{result}")
    existing = [
        item
        for item in existing
        if not (int(item["epoch"]) == epoch and int(item["step"]) == step)
    ]
    existing.append(
        {"score": float(score), "epoch": int(epoch), "step": int(step), "path": str(candidate_path)}
    )
    existing.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = existing[:top_k]
    dropped = existing[top_k:]

    temporary_paths: list[Path] = []
    for rank, item in enumerate(selected, start=1):
        source = Path(str(item["path"]))
        temporary = weights_dir / f".{hps.name}_top_{rank}_e{epoch}_s{step}.tmp"
        if temporary.exists():
            temporary.unlink()
        source.replace(temporary)
        temporary_paths.append(temporary)
    for item in dropped:
        path = Path(str(item["path"]))
        if path.is_file():
            path.unlink()

    final_paths: set[Path] = set()
    for rank, (item, temporary) in enumerate(zip(selected, temporary_paths), start=1):
        destination = weights_dir / (
            f"{hps.name}_best_{rank}_e{int(item['epoch'])}_s{int(item['step'])}.pth"
        )
        if destination.exists():
            destination.unlink()
        temporary.replace(destination)
        item["path"] = str(destination)
        final_paths.add(destination)
    for old_path in weights_dir.glob(f"{hps.name}_best_[0-9]*_e*_s*.pth"):
        if old_path not in final_paths:
            old_path.unlink()
    shutil.copy2(Path(str(selected[0]["path"])), weights_dir / f"{hps.name}_best.pth")
    tracker.top_models = selected


def emit_training_status(
    epoch,
    hps,
    validation_score,
    tracker,
    health,
    checkpoint_eligible,
    checkpoint_reason,
):
    """输出可被 RVC-Core 任务层识别的结构化训练状态。"""

    payload = {
        "current_epoch": epoch,
        "max_epoch": hps.total_epoch,
        "validation_score": validation_score,
        "best_score": tracker.best_score if tracker.best_epoch else None,
        "best_epoch": tracker.best_epoch or None,
        "bad_count": tracker.bad_count,
        "patience": tracker.patience,
        "checkpoint_eligible": checkpoint_eligible,
        "generator_loss": health.get("generator_loss") if health else None,
        "discriminator_loss": health.get("discriminator_loss") if health else None,
        "health_reason": checkpoint_reason,
    }
    print("RVC_TRAIN_STATUS " + json.dumps(payload, ensure_ascii=False), flush=True)


def save_final_model(net_g, hps, epoch, logger):
    """达到最大 Epoch 时保留原有最终小模型导出行为。"""

    logger.info(i18n("训练已完成，正在保存最终模型"))
    logger.info(
        i18n("正在保存最终检查点：%s")
        % savee(
            _generator_state(net_g),
            hps.sample_rate,
            hps.if_f0,
            hps.name,
            epoch,
            hps.version,
            hps,
        )
    )


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    health_enabled = bool(getattr(hps, "validation_enabled", False))
    generator_health_losses = []
    discriminator_health_losses = []

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                phone = phone.to(training_device)
                phone_lengths = phone_lengths.to(training_device)
                if hps.if_f0 == 1:
                    pitch = pitch.to(training_device)
                    pitchf = pitchf.to(training_device)
                sid = sid.to(training_device)
                spec = spec.to(training_device)
                spec_lengths = spec_lengths.to(training_device)
                wave = wave.to(training_device)
                wave_lengths = wave_lengths.to(training_device)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Neural tensors always move to Apple GPU; CPU fallback is forbidden.
        if hps.if_cache_data_in_gpu == False:
            phone = phone.to(training_device)
            phone_lengths = phone_lengths.to(training_device)
            if hps.if_f0 == 1:
                pitch = pitch.to(training_device)
                pitchf = pitchf.to(training_device)
            sid = sid.to(training_device)
            spec = spec.to(training_device)
            spec_lengths = spec_lengths.to(training_device)
            wave = wave.to(training_device)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=training_is_half):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if training_is_half:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        if health_enabled and not bool(torch.isfinite(loss_disc).all().item()):
            raise FloatingPointError("Discriminator loss 出现 NaN 或 Inf，已停止训练")
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=training_is_half):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        if health_enabled and not bool(torch.isfinite(loss_gen_all).all().item()):
            raise FloatingPointError("Generator loss 出现 NaN 或 Inf，已停止训练")
        if health_enabled:
            discriminator_health_losses.append(float(loss_disc.detach().cpu()))
            generator_health_losses.append(float(loss_gen_all.detach().cpu()))
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    i18n("训练轮次：{} [{:.0f}%]").format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
    # /Run steps

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
            )
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                i18n("正在保存检查点 %s_e%s：%s")
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info(i18n("====> 轮次：{} {}").format(epoch, epoch_recorder.record()))
    if (
        epoch >= hps.total_epoch
        and rank == 0
        and not bool(getattr(hps, "validation_enabled", False))
    ):
        logger.info(i18n("训练已完成，正在保存最终模型"))

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            i18n("正在保存最终检查点：%s")
            % (
                savee(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        sleep(1)
        os._exit(0)
    if health_enabled:
        return summarize_training_health(
            generator_health_losses, discriminator_health_losses
        )
    return None


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
