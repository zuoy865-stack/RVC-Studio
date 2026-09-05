"""Short, synchronous service operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from train.process_ckpt import change_info, extract_small_model, merge, show_info
from rvc_core.workflows import CORE_ROOT
from rvc_core.workflows import PYMSS_MODEL_DIR, SEPARATION_MODELS
from pymss.model_registry import resolve_model


def list_models() -> dict[str, Any]:
    weights = CORE_ROOT / "assets" / "weights"
    indices = CORE_ROOT / "assets" / "indices"
    models = sorted(str(path.resolve()) for path in weights.glob("*.pth"))
    index_files = sorted({str(path.resolve()) for root in (indices, CORE_ROOT / "logs") for path in root.rglob("*.index")})
    return {"models": models, "indices": index_files}


def ckpt_operation(request: dict[str, Any]) -> dict[str, Any]:
    action = str(request.get("action") or "")
    if action == "inspect":
        result = show_info(str(request["path"]))
    elif action == "edit":
        result = change_info(str(request["path"]), str(request.get("info", "")), str(request.get("name", "")))
    elif action == "extract":
        result = extract_small_model(str(request["path"]), str(request["name"]), str(request.get("sample_rate", "40k")), bool(request.get("use_f0", True)), str(request.get("info", "")), str(request.get("version", "v2")))
    elif action == "merge":
        result = merge(str(request["path1"]), str(request["path2"]), float(request.get("alpha", 0.5)), str(request.get("sample_rate", "40k")), "是" if request.get("use_f0", True) else "否", str(request.get("info", "")), str(request["name"]), str(request.get("version", "v2")))
    else:
        raise ValueError("未知 ckpt 操作")
    if action != "inspect" and result != "成功":
        raise RuntimeError(result)
    return {"action": action, "result": result, "weights_dir": str(CORE_ROOT / "assets" / "weights")}


def separation_models() -> dict[str, Any]:
    result = []
    for key, name in SEPARATION_MODELS.items():
        resolved = resolve_model(name, model_dir=PYMSS_MODEL_DIR, require_supported=True, require_exists=False)
        paths = [Path(resolved["model_path"])]
        if resolved.get("config_path"):
            paths.append(Path(resolved["config_path"]))
        entry = resolved["entry"]
        result.append({"key": key, "name": name, "installed": all(path.is_file() for path in paths), "size_bytes": int(entry.size_bytes or 0)})
    return {"models": result, "model_dir": str(PYMSS_MODEL_DIR)}


def delete_separation_model(model_key: str) -> dict[str, Any]:
    name = SEPARATION_MODELS.get(model_key)
    if not name:
        raise ValueError("未知分离模型")
    resolved = resolve_model(name, model_dir=PYMSS_MODEL_DIR, require_supported=True, require_exists=False)
    removed = []
    for value in (resolved.get("model_path"), resolved.get("config_path")):
        if value and Path(value).is_file():
            Path(value).unlink()
            removed.append(str(value))
    return {"removed": removed, **separation_models()}
