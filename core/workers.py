"""GPU/media workers run as subprocesses: JSON request file in -> JSON response file out.

Why subprocesses: each media model (SDXL, Kokoro, Whisper, ...) has conflicting dependencies (Kokoro needs
Python 3.12 on Kaggle) and must release VRAM completely before the next one loads. Killing the whole
process group guarantees that.

Worker contract (workers/*.py and renderer/*.mjs):
    python workers/<name>.py --in request.json --out response.json
    request:  {"task": "...", ...task fields..., "seed": int, "out_dir": "..."}
    response: {"status": "ok"|"error", "error": "", "outputs": {...}, "units": {...},
               "model": "...", "model_revision": "...", "licence": "..."}
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from core.context import ROOT, RunContext, load_yaml
from core.logging import RunLogger


class WorkerError(RuntimeError):
    pass


_counter = {"n": 0}
_counter_lock = threading.Lock()


def _interpreter(worker: str) -> list[str]:
    if worker.endswith(".mjs"):
        return ["node"]
    if worker == "tts_kokoro.py":
        voice_py = load_yaml("config/models.yaml")["voice"]["python"]
        if Path(ROOT / voice_py).exists():
            return [str(ROOT / voice_py)]
    return [sys.executable]


def run_worker(ctx: RunContext, log: RunLogger, *, agent: str, worker: str, request: dict,
               gpu: str | None = "1", timeout_s: int = 1800) -> dict:
    """Run one worker call; returns response["outputs"]. Logs one run_log line."""
    with _counter_lock:
        _counter["n"] += 1
        n = _counter["n"]
    wdir = ctx.out_dir / "workers"
    wdir.mkdir(exist_ok=True)
    stem = f"{n:03d}_{Path(worker).stem}_{request.get('task', 'run')}"
    req_path, resp_path = wdir / f"{stem}.in.json", wdir / f"{stem}.out.json"
    request = {"out_dir": str(ctx.assets_dir), **request}
    req_path.write_text(json.dumps(request, indent=1, default=str), encoding="utf-8")

    script = ROOT / ("renderer" if worker.endswith(".mjs") else "workers") / worker
    cmd = [*_interpreter(worker), str(script), "--in", str(req_path), "--out", str(resp_path)]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    popen_kw: dict = {"cwd": str(ROOT), "env": env, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
    if os.name == "posix":
        popen_kw["start_new_session"] = True  # own process group -> kill frees all VRAM
    with log.call(agent, f"worker:{Path(worker).stem}", seed=request.get("seed"), gpu_id=gpu or "",
                  units={"task": request.get("task", "")}) as rec:
        t0 = time.time()
        proc = subprocess.Popen(cmd, **popen_kw)
        try:
            stdout, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill(proc)
            raise WorkerError(f"{worker} timed out after {timeout_s}s")
        (wdir / f"{stem}.log").write_text(stdout or "", encoding="utf-8")
        if not resp_path.exists():
            raise WorkerError(f"{worker} exited {proc.returncode} without response; see {stem}.log:\n{(stdout or '')[-1500:]}")
        resp = json.loads(resp_path.read_text(encoding="utf-8"))
        rec["model"] = resp.get("model", "")
        rec["model_revision"] = resp.get("model_revision", "")
        rec["units"] = {**rec.get("units", {}), **resp.get("units", {}), "wall_s": round(time.time() - t0, 2)}
        rec["output_path"] = str(resp_path)
        if resp.get("status") != "ok":
            rec["status"] = "error"
            raise WorkerError(f"{worker}: {resp.get('error', 'unknown error')}")
        outputs = resp.get("outputs", {})
        outputs.setdefault("meta", {"model": resp.get("model", ""), "model_revision": resp.get("model_revision", ""),
                                    "licence": resp.get("licence", "")})
        return outputs


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass


def worker_main(handler) -> None:
    """Boilerplate for workers: `if __name__ == "__main__": worker_main(handle)` where handle(req)->dict
    returns {"outputs":..., "units":..., "model":..., "model_revision":..., "licence":...}."""
    import argparse
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()
    req = json.loads(Path(a.inp).read_text(encoding="utf-8"))
    try:
        resp = {"status": "ok", "error": "", **handler(req)}
    except Exception as e:  # noqa: BLE001
        resp = {"status": "error", "error": f"{e.__class__.__name__}: {e}\n{traceback.format_exc()[-3000:]}",
                "outputs": {}, "units": {}}
    Path(a.out).write_text(json.dumps(resp, indent=1, default=str), encoding="utf-8")
