import json
import os
import subprocess
import sys
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parents[1]


def test_server_starts_on_strict_mps():
    environment = os.environ.copy()
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "rvc_core.server"],
        cwd=CORE_ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    first = json.loads(process.stdout.readline())
    assert first["event"] in {"ready", "fatal"}
    if first["event"] == "ready":
        assert first["data"]["device"] == "mps"
        assert first["data"]["cpu_fallback"] is False
        assert process.stdin is not None
        process.stdin.write('{"id":"1","command":"shutdown"}\n')
        process.stdin.flush()
    process.wait(timeout=10)
