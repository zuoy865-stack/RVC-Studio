"""Long-running native-UI workflows and training command construction."""

from __future__ import annotations

import copy
import json
import os
import random
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from configs.config import Config
from tools.multispeaker import build_manifest_from_root, write_manifest, load_manifest

CORE_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = CORE_ROOT / "logs"
MODEL_HOME = Path(os.environ.get("RVC_MODEL_HOME", "~/Library/Application Support/RVC-Studio/models")).expanduser()
PYMSS_MODEL_DIR = MODEL_HOME / "pymss"


class TaskManager:
    """Runs one cancellable task and streams its output as protocol events."""

    def __init__(self, emit: Callable[[dict[str, Any]], None]):
        self.emit = emit
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._cancelled = False
        self._task_id = ""
        self._name = ""
        self._status_detail: dict[str, Any] = {}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": bool(self._thread and self._thread.is_alive()),
                "id": self._task_id,
                "name": self._name,
                **self._status_detail,
            }

    def start(self, request_id: str, name: str, body: Callable[["TaskManager"], dict[str, Any] | None]):
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError(f"任务 {self._name} 正在运行，请先停止")
            self._cancelled = False
            self._task_id = request_id
            self._name = name
            self._status_detail = {}
            self._thread = threading.Thread(target=self._run, args=(body,), daemon=True, name=f"rvc-{name}")
            self._thread.start()
        return {"started": True, "task": name, "task_id": request_id}

    def _run(self, body):
        self.event("running", f"{self._name} 已启动")
        try:
            result = body(self) or {}
            if self._cancelled:
                self.event("cancelled", f"{self._name} 已停止")
            else:
                self.event("done", f"{self._name} 已完成", result=result)
        except Exception as exc:
            state = "cancelled" if self._cancelled else "failed"
            self.event(state, str(exc))
        finally:
            with self._lock:
                self._process = None

    def event(self, state: str, message: str, **extra):
        payload = {"event": "task", "id": self._task_id, "task": self._name, "state": state, "message": str(message)}
        payload.update(extra)
        if extra.get("training_status"):
            with self._lock:
                self._status_detail = {
                    key: extra.get(key)
                    for key in (
                        "current_epoch",
                        "max_epoch",
                        "validation_score",
                        "best_score",
                        "best_epoch",
                        "bad_count",
                        "patience",
                        "checkpoint_eligible",
                        "generator_loss",
                        "discriminator_loss",
                        "health_reason",
                    )
                }
        self.emit(payload)

    def run_command(self, command: list[str], step: str):
        if self._cancelled:
            raise RuntimeError("任务已停止")
        self.event("running", f"[{step}] {shlex.join(command)}", step=step)
        environment = os.environ.copy()
        environment.update({"PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTORCH_MPS_PREFER_METAL": "1"})
        process = subprocess.Popen(
            command, cwd=CORE_ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
        )
        with self._lock:
            self._process = process
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip()
            if stripped.startswith("RVC_TRAIN_STATUS "):
                try:
                    status = json.loads(stripped.removeprefix("RVC_TRAIN_STATUS "))
                    score = status.get("validation_score")
                    score_label = "等待验证" if score is None else f"{float(score):.6f}"
                    best = status.get("best_score")
                    best_label = "—" if best is None else f"{float(best):.6f}"
                    message = (
                        f"Epoch {status['current_epoch']}/{status['max_epoch']} · "
                        f"Validation {score_label} · Best {best_label}@{status.get('best_epoch') or '—'} · "
                        f"bad_count {status['bad_count']}/{status['patience']} · "
                        f"checkpoint {'健康' if status.get('checkpoint_eligible') else '禁止入选'}"
                    )
                    self.event("running", message, step=step, training_status=True, **status)
                    continue
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            self.event("running", stripped, step=step)
        code = process.wait()
        with self._lock:
            self._process = None
        if self._cancelled:
            raise RuntimeError("任务已停止")
        if code:
            raise RuntimeError(f"{step} 失败（退出代码 {code}）")

    def cancel(self) -> bool:
        with self._lock:
            active = bool(self._thread and self._thread.is_alive())
            self._cancelled = True
            process = self._process
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return active


def _experiment(params: dict[str, Any]) -> tuple[str, Path]:
    name = str(params.get("experiment") or "").strip()
    if not name or any(char in name for char in "/\\|\n\r"):
        raise ValueError("实验名不能为空，且不能包含路径分隔符")
    return name, LOG_ROOT / name


def _sample_rate(params: dict[str, Any]) -> tuple[str, int]:
    label = str(params.get("sample_rate", "40k"))
    rates = {"32k": 32000, "40k": 40000, "48k": 48000}
    if label not in rates:
        raise ValueError("采样率必须是 32k、40k 或 48k")
    return label, rates[label]


def _prepare_manifest(params: dict[str, Any], exp_dir: Path) -> str:
    if not bool(params.get("multi_speaker", False)):
        return ""
    manifest = build_manifest_from_root(str(params.get("dataset") or ""))
    return write_manifest(str(exp_dir), manifest)


def preprocess_commands(params: dict[str, Any]) -> list[tuple[str, list[str]]]:
    _, exp_dir = _experiment(params)
    _, rate = _sample_rate(params)
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest = _prepare_manifest(params, exp_dir)
    dataset = Path(str(params.get("dataset") or "")).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"训练集文件夹不存在：{dataset}")
    workers = max(1, int(params.get("workers", min(8, os.cpu_count() or 1))))
    cmd = [sys.executable, "-m", "train.preprocess", str(dataset), str(rate), str(workers), str(exp_dir), "False", "3.0"]
    if manifest:
        cmd.append(manifest)
    return [("数据处理", cmd)]


def feature_commands(params: dict[str, Any]) -> list[tuple[str, list[str]]]:
    _, exp_dir = _experiment(params)
    if not (exp_dir / "1_16k_wavs").is_dir():
        raise RuntimeError("请先处理训练数据")
    commands: list[tuple[str, list[str]]] = []
    if bool(params.get("use_f0", True)):
        method = str(params.get("f0_method", "rmvpe"))
        if method == "rmvpe":
            commands.append(("RMVPE 音高提取（MPS）", [sys.executable, "-m", "train.dataset.extract_f0", "mps", str(exp_dir)]))
        elif method == "pm":
            workers = max(1, int(params.get("workers", min(8, os.cpu_count() or 1))))
            commands.append(("Praat PM 音高提取（DSP）", [sys.executable, "-m", "train.dataset.extract_f0", "cpu", str(exp_dir), str(workers), "pm"]))
        else:
            raise ValueError("训练 F0 算法仅支持 rmvpe 或 pm")
    version = str(params.get("version", "v2"))
    commands.append(("HuBERT 特征提取（MPS）", [sys.executable, "-m", "train.dataset.extract_hubert_feature", "mps", "1", "0", str(exp_dir), version, "False"]))
    return commands


def prepare_training(params: dict[str, Any]) -> list[str]:
    name, exp_dir = _experiment(params)
    sr, _ = _sample_rate(params)
    version = str(params.get("version", "v2"))
    use_f0 = bool(params.get("use_f0", True))
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    gt_dir = exp_dir / "0_gt_wavs"
    required = [gt_dir, feature_dir]
    f0_dir, f0nsf_dir = exp_dir / "2a_f0", exp_dir / "2b-f0nsf"
    if use_f0:
        required += [f0_dir, f0nsf_dir]
    if any(not path.is_dir() for path in required):
        raise RuntimeError("训练数据或特征不完整，请先完成数据处理与特征提取")
    stems = {path.stem for path in gt_dir.glob("*.wav")} & {path.stem for path in feature_dir.glob("*.npy")}
    if use_f0:
        stems &= {path.name.removesuffix(".wav.npy") for path in f0_dir.glob("*.wav.npy")}
        stems &= {path.name.removesuffix(".wav.npy") for path in f0nsf_dir.glob("*.wav.npy")}
    multi = bool(params.get("multi_speaker", False))
    manifest_by_key = {}
    speakers = {}
    if multi:
        manifest = load_manifest(str(exp_dir))
        manifest_by_key = {entry["output_key"]: entry for entry in manifest["entries"]}
        stems = {stem for stem in stems if stem.rsplit("_", 1)[0] in manifest_by_key}
        speakers = {int(item["id"]): item["name"] for item in manifest["speakers"]}
    if not stems:
        raise RuntimeError("没有可用于训练的有效音频")
    lines = []
    for stem in sorted(stems):
        entry = manifest_by_key.get(stem.rsplit("_", 1)[0]) if multi else None
        speaker_id = int(entry["speaker_id"]) if entry else int(params.get("speaker_id", 0))
        repeat = int(entry["repeat"]) if entry else 1
        fields = [str(gt_dir / f"{stem}.wav"), str(feature_dir / f"{stem}.npy")]
        if use_f0:
            fields += [str(f0_dir / f"{stem}.wav.npy"), str(f0nsf_dir / f"{stem}.wav.npy")]
        fields.append(str(speaker_id))
        if multi:
            fields.append(str(entry["speaker_name"]))
        lines.extend(["|".join(fields)] * repeat)
    random.shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")
    config_key = f"{'v1' if version == 'v1' or sr == '40k' else 'v2'}/{sr}.json"
    template = Config(precision="float32", strict=True).json_config[config_key]
    config = copy.deepcopy(template)
    if multi:
        config["model"]["spk_embed_dim"] = 110
        config["speaker_info"] = [{"id": sid, "name": speakers[sid]} for sid in sorted(speakers)]
    (exp_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-m", "train.train", "-e", name, "-sr", sr, "-f0", "1" if use_f0 else "0", "-bs", str(max(1, int(params.get("batch_size", 4)))), "-te", str(max(1, int(params.get("epochs", 200)))), "-se", str(max(1, int(params.get("save_every", 10)))), "-l", "1" if params.get("save_latest") else "0", "-c", "1" if params.get("cache_gpu") else "0", "-sw", "1" if params.get("save_weights", True) else "0", "-v", version]
    auto_best = bool(params.get("auto_best_model", False))
    early_stopping = bool(params.get("early_stopping", False))
    if auto_best or early_stopping:
        validation_split = float(params.get("validation_split", 0.1))
        validation_interval = int(params.get("validation_interval", 10))
        patience = int(params.get("patience", 6))
        min_delta = float(params.get("min_delta", 0.002))
        top_k = int(params.get("top_k", 3))
        validation_speaker_id = int(params.get("speaker_id", 0))
        conversion_validation_dir = str(params.get("conversion_validation_dir") or "").strip()
        if not 0 < validation_split < 0.5:
            raise ValueError("验证集比例必须大于 0 且小于 0.5")
        if validation_interval < 1:
            raise ValueError("验证间隔必须至少为 1 个 Epoch")
        if patience < 1:
            raise ValueError("patience 必须至少为 1")
        if min_delta < 0:
            raise ValueError("min_delta 不能为负数")
        if not 1 <= top_k <= 20:
            raise ValueError("Top-K 数量必须在 1 到 20 之间")
        if validation_speaker_id < 0:
            raise ValueError("Conversion Validation 目标说话人 ID 不能为负数")
        if conversion_validation_dir:
            conversion_validation_path = Path(conversion_validation_dir).expanduser().resolve()
            if not conversion_validation_path.is_dir():
                raise FileNotFoundError(
                    f"转换验证音频目录不存在：{conversion_validation_path}"
                )
            conversion_validation_dir = str(conversion_validation_path)
        command += [
            "--auto_best_model", "1" if auto_best else "0",
            "--early_stopping", "1" if early_stopping else "0",
            "--validation_split", str(validation_split),
            "--validation_interval", str(validation_interval),
            "--early_stopping_patience", str(patience),
            "--early_stopping_min_delta", str(min_delta),
            "--best_top_k", str(top_k),
            "--conversion_validation_dir", conversion_validation_dir,
            "--validation_speaker_id", str(validation_speaker_id),
        ]
    for flag, key in (("-pg", "pretrained_g"), ("-pd", "pretrained_d")):
        value = str(params.get(key) or "").strip()
        if value:
            command += [flag, value]
    return command


def index_command(params: dict[str, Any]) -> list[str]:
    name, _ = _experiment(params)
    return [sys.executable, "-m", "train.train_index", name, str(params.get("version", "v2")), str(CORE_ROOT / "assets" / "indices"), str(max(1, int(params.get("workers", 4)))), "auto"]


def run_training_workflow(manager: TaskManager, operation: str, params: dict[str, Any]):
    if operation in {"preprocess", "one_click"}:
        for step, cmd in preprocess_commands(params): manager.run_command(cmd, step)
    if operation in {"extract", "one_click"}:
        for step, cmd in feature_commands(params): manager.run_command(cmd, step)
    if operation in {"train", "one_click"}:
        manager.run_command(prepare_training(params), "模型训练（严格 MPS）")
    if operation in {"index", "one_click"}:
        manager.run_command(index_command(params), "FAISS 特征索引（CPU）")
    return {"operation": operation, "experiment": params.get("experiment")}


SEPARATION_MODELS = {
    "dereverb": "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
    "dereverb_aggressive": "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "vocals": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "vocals_aggressive": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "lead_vocal": "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
}


def separation_command(params: dict[str, Any]) -> list[str]:
    model = SEPARATION_MODELS.get(str(params.get("model", "vocals")))
    if not model: raise ValueError("未知分离模型")
    input_path = Path(str(params.get("input") or "")).expanduser().resolve()
    output = Path(str(params.get("output") or "")).expanduser().resolve()
    secondary = Path(str(params.get("secondary_output") or output)).expanduser().resolve()
    if not input_path.exists(): raise FileNotFoundError(f"输入不存在：{input_path}")
    output.mkdir(parents=True, exist_ok=True)
    secondary.mkdir(parents=True, exist_ok=True)
    return [sys.executable, "-m", "rvc_core.separation_worker", model, "--model-dir", str(PYMSS_MODEL_DIR), "--input", str(input_path), "--primary-output", str(output), "--secondary-output", str(secondary), "--format", str(params.get("format", "wav"))]


def separation_download_command(model_key: str) -> list[str]:
    model = SEPARATION_MODELS.get(model_key)
    if not model:
        raise ValueError("未知分离模型")
    PYMSS_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable,
        "-m",
        "rvc_core.model_download_worker",
        model,
        "--model-dir",
        str(PYMSS_MODEL_DIR),
    ]
