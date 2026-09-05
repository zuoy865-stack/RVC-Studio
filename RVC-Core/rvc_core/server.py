"""Newline-delimited JSON service used by the native Qt application."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import threading
import traceback
from typing import Any

from rvc_core.engine import RVCEngine
from rvc_core.operations import ckpt_operation, delete_separation_model, list_models, separation_models
from rvc_core.workflows import TaskManager, run_training_workflow, separation_command, separation_download_command


_emit_lock = threading.Lock()


def emit(payload: dict[str, Any]) -> None:
    with _emit_lock:
        sys.__stdout__.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.__stdout__.flush()


def run(precision: str) -> int:
    try:
        engine = RVCEngine(precision=precision)
    except Exception as exc:
        emit({"event": "fatal", "error": str(exc), "trace": traceback.format_exc()})
        return 2

    # Reserve the real stdout exclusively for JSON protocol frames. Legacy RVC
    # modules are chatty and may print from worker threads.
    sys.stdout = sys.stderr
    emit({"event": "ready", "data": engine.status()})
    tasks = TaskManager(emit)
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            command = request.get("command")
            with contextlib.redirect_stdout(sys.stderr):
                if command == "status":
                    data = engine.status()
                elif command == "inspect_model":
                    data = engine.inspect_model(str(request["path"]))
                elif command == "load_model":
                    data = engine.load_model(str(request["path"]))
                elif command == "unload_model":
                    data = engine.unload_model()
                elif command == "list_models":
                    data = list_models()
                elif command == "convert":
                    params = dict(request)
                    data = tasks.start(str(request_id), "单文件转换（MPS）", lambda manager: engine.convert(params))
                elif command == "convert_batch":
                    params = dict(request)
                    def batch_body(manager):
                        return engine.convert_batch(params, lambda i, n, name: manager.event("running", f"批量转换 {i}/{n}：{name}", current=i, total=n))
                    data = tasks.start(str(request_id), "批量转换", batch_body)
                elif command == "train_task":
                    operation = str(request.get("operation") or "")
                    if operation not in {"preprocess", "extract", "train", "index", "one_click"}:
                        raise ValueError("未知训练操作")
                    params = dict(request)
                    data = tasks.start(str(request_id), {"preprocess":"数据处理", "extract":"特征提取", "train":"模型训练", "index":"索引训练", "one_click":"一键训练"}[operation], lambda manager: run_training_workflow(manager, operation, params))
                elif command == "separate":
                    params = dict(request)
                    cmd = separation_command(params)
                    data = tasks.start(str(request_id), "人声/伴奏分离（MLX）", lambda manager: (manager.run_command(cmd, "PyMSS MLX 推理") or {"output": params.get("output")}))
                elif command == "separation_models":
                    data = separation_models()
                elif command == "download_separation_model":
                    model_key = str(request.get("model") or "")
                    cmd = separation_download_command(model_key)
                    data = tasks.start(str(request_id), "下载分离模型", lambda manager: (manager.run_command(cmd, "模型下载") or separation_models()))
                elif command == "delete_separation_model":
                    data = delete_separation_model(str(request.get("model") or ""))
                elif command == "task_status":
                    data = tasks.status()
                elif command == "cancel_task":
                    data = {"cancelled": tasks.cancel()}
                elif command == "ckpt":
                    data = ckpt_operation(request)
                elif command == "shutdown":
                    emit({"event": "response", "id": request_id, "ok": True, "data": {}})
                    return 0
                else:
                    raise ValueError(f"unknown command: {command}")
            emit({"event": "response", "id": request_id, "ok": True, "data": data})
        except Exception as exc:
            emit({
                "event": "response",
                "id": request_id,
                "ok": False,
                "error": str(exc),
                "trace": traceback.format_exc(),
            })
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("float32", "float16"), default="float32")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    return run(args.precision)


if __name__ == "__main__":
    raise SystemExit(main())
