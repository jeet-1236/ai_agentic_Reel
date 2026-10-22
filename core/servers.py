"""Local model servers (llama.cpp `llama-server`) + model/binary discovery. Platform-owned.

    ensure_llm(ctx)   text LLM (config llm.text) on GPU0, stays up for the whole run; idempotent
    start_vlm(ctx)    vision LLM (config llm.vision, --mmproj) on GPU1, started by the critic on demand
    stop_vlm(ctx)     kills the VLM's process group so GPU1 VRAM is free for the next worker
    health(url)       True when GET <root>/health answers 200

Every function is a no-op when `ctx.mock_llm`. Servers bind 127.0.0.1 only (Kaggle AUP: no public servers).
The server is detached (own session, stdout to logs/servers/<alias>.log, pid in logs/servers/<alias>.pid),
so it survives the `python -c` that `scripts/setup_kaggle.sh` uses to start it, and a later process can
stop it from the pid file.

Discovery (no hard-coded Kaggle mount paths: they changed in Oct 2026):
    binary: $AGENTREEL_LLAMA_BIN -> <repo>/bin/**/llama-server -> PATH -> /kaggle/input/** (depth-limited walk)
    model : $AGENTREEL_LLM_MODEL / $AGENTREEL_VLM_MODEL / $AGENTREEL_VLM_MMPROJ (files)
            -> $AGENTREEL_MODELS_DIR -> <repo>/models -> /kaggle/input/** by file name
            -> huggingface_hub download into <repo>/models (Internet ON; logged)
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from core.context import ROOT, RunContext, load_yaml
from core.logging import RunLogger

KAGGLE_INPUT = Path(os.environ.get("AGENTREEL_KAGGLE_INPUT", "/kaggle/input"))
MODELS_DIR = ROOT / "models"
BIN_DIR = ROOT / "bin"
SERVERS_DIR = ROOT / "logs" / "servers"
WALK_DEPTH = 7
_PROCS: dict[str, subprocess.Popen] = {}


class ServerError(RuntimeError):
    pass


# ---------------------------------------------------------------- small helpers
def _root_url(url: str) -> str:
    """http://127.0.0.1:8001/v1 -> http://127.0.0.1:8001"""
    p = urlsplit(url)
    return f"{p.scheme or 'http'}://{p.netloc}"


def health(url: str, timeout: float = 2.0) -> bool:
    """True if llama-server at `url` (base or /v1 url) reports ready (HTTP 200 on /health)."""
    try:
        with urllib.request.urlopen(_root_url(url) + "/health", timeout=timeout) as r:  # noqa: S310 (localhost)
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def served_models(url: str, timeout: float = 3.0) -> list[str]:
    """Model ids from GET /v1/models ([] on error)."""
    try:
        with urllib.request.urlopen(_root_url(url) + "/v1/models", timeout=timeout) as r:  # noqa: S310
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("id", "") for m in data.get("data", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []


def _walk_find(base: Path, names: set[str], *, max_depth: int = WALK_DEPTH, want_dir: bool = False) -> Path | None:
    """Depth-limited search for a file (or dir) whose name is in `names`."""
    if not base.exists():
        return None
    base_depth = len(base.parts)
    for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
        depth = len(Path(dirpath).parts) - base_depth
        pool = dirnames if want_dir else filenames
        for n in pool:
            if n in names:
                return Path(dirpath) / n
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
    return None


def _ensure_executable(path: Path) -> Path:
    """Kaggle datasets are read-only and may drop the exec bit: copy into <repo>/bin and chmod."""
    if os.name != "posix" or os.access(path, os.X_OK):
        return path
    dst_dir = BIN_DIR / "llama"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in path.parent.iterdir():           # bring shared libs (libggml*.so, libllama.so) along
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)
    dst = dst_dir / path.name
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dst


def find_llama_server() -> Path:
    """Locate the llama-server binary (see module docstring for the search order)."""
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    env = os.environ.get("AGENTREEL_LLAMA_BIN")
    if env and Path(env).is_file():
        return _ensure_executable(Path(env))
    hit = _walk_find(BIN_DIR, {exe}, max_depth=5)
    if hit:
        return _ensure_executable(hit)
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    hit = _walk_find(KAGGLE_INPUT, {exe})
    if hit:
        return _ensure_executable(hit)
    raise ServerError("llama-server not found: set AGENTREEL_LLAMA_BIN, run scripts/setup_kaggle.sh, "
                      "or attach the agentreel-bin dataset")


def _repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def resolve_model(repo: str, filename: str | None = None, *, revision: str = "", env_var: str = "",
                  download: bool = True) -> Path:
    """Path to a model file (`filename` given) or snapshot dir (no filename).

    Order: $env_var -> $AGENTREEL_MODELS_DIR -> <repo>/models -> /kaggle/input walk -> HF download.
    Layout written by scripts/prefetch_models.py: models/<org>__<name>/<files...>.
    """
    if env_var and os.environ.get(env_var) and Path(os.environ[env_var]).exists():
        return Path(os.environ[env_var])
    roots = [Path(p) for p in os.environ.get("AGENTREEL_MODELS_DIR", "").split(os.pathsep) if p]
    roots += [MODELS_DIR, KAGGLE_INPUT]
    want = {filename} if filename else {_repo_dir_name(repo)}
    for r in roots:
        direct = r / _repo_dir_name(repo) / filename if filename else r / _repo_dir_name(repo)
        if direct.exists():
            return direct
        hit = _walk_find(r, want, want_dir=not filename)
        if hit:
            return hit
    if not download:
        raise ServerError(f"model {repo}/{filename or ''} not found locally")
    from huggingface_hub import hf_hub_download, snapshot_download  # lazy: heavy-ish, not needed in tests

    local = MODELS_DIR / _repo_dir_name(repo)
    kw = {"revision": revision or None, "local_dir": str(local)}
    if filename:
        return Path(hf_hub_download(repo, filename, **kw))
    return Path(snapshot_download(repo, **kw))


def llama_version(binary: Path) -> str:
    """`llama-server --version` (llama.cpp prints it on stderr)."""
    try:
        p = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30,
                           env=_server_env(binary, None))
        return " | ".join(line.strip() for line in (p.stdout + p.stderr).splitlines() if line.strip())[:500]
    except (OSError, subprocess.SubprocessError) as e:
        return f"unavailable: {e.__class__.__name__}"


def _server_env(binary: Path, gpu: str | None) -> dict:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if os.name == "posix":
        env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (str(binary.parent), env.get("LD_LIBRARY_PATH", "")) if p)
    return env


# ---------------------------------------------------------------- pid files
def _pid_file(alias: str) -> Path:
    return SERVERS_DIR / f"{alias}.pid"


def _read_pid(alias: str) -> int | None:
    f = _pid_file(alias)
    try:
        return int(f.read_text().strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name != "posix":
        proc = _PROCS.get(str(pid))
        return proc is not None and proc.poll() is None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # zombie check
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True


def _kill_group(pid: int, timeout_s: float = 20.0) -> None:
    """SIGTERM the process group, then SIGKILL after `timeout_s`."""
    if os.name != "posix":
        proc = _PROCS.pop(str(pid), None)
        if proc:
            proc.kill()
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    for sig, wait in ((signal.SIGTERM, timeout_s), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        t0 = time.time()
        while time.time() - t0 < wait:
            if not _alive(pid):
                return
            time.sleep(0.25)


# ---------------------------------------------------------------- launch
def build_command(binary: Path, model: Path, cfg: dict, seed: int, mmproj: Path | None = None) -> list[str]:
    """llama-server argv for one config block of config/models.yaml (llm.text / llm.vision)."""
    s = cfg["server"]
    cmd = [str(binary), "-m", str(model), "--alias", cfg["alias"], "--host", "127.0.0.1",
           "--port", str(s["port"]), "-c", str(s.get("ctx", 8192)), "-np", str(s.get("parallel", 1)),
           "-ngl", str(s.get("ngl", 999)), "--seed", str(seed)]
    if mmproj is not None:
        cmd += ["--mmproj", str(mmproj)]
    cmd += [str(x) for x in s.get("extra", [])]
    return cmd


def _launch(ctx: RunContext, log: RunLogger | None, kind: str, timeout_s: float) -> str:
    """Start (or reuse) the llama-server for llm.<kind>; returns its OpenAI base url."""
    cfg = load_yaml("config/models.yaml")["llm"][kind]
    s = cfg["server"]
    url = f"http://127.0.0.1:{s['port']}/v1"
    log = log or RunLogger(ctx)
    alias = cfg["alias"]

    if health(url):
        ids = served_models(url)
        log.record(agent="platform", tool="server", model=alias, model_revision=cfg.get("revision", ""),
                   gpu_id=str(s["gpu"]), units={"action": "reuse", "kind": kind, "url": url, "served": ids,
                                                "pid": _read_pid(alias)})
        if ids and alias not in ids:
            raise ServerError(f"port {s['port']} serves {ids}, expected {alias}; stop it first")
        return url

    pid = _read_pid(alias)
    with log.call("platform", "server", model=alias, model_revision=cfg.get("revision", ""),
                  gpu_id=str(s["gpu"]), seed=ctx.seed) as rec:
        if _alive(pid):                    # already launched (e.g. by setup) and still loading
            rec["units"] = {"action": "wait", "kind": kind, "pid": pid, "url": url}
        else:
            binary = find_llama_server()
            model = resolve_model(cfg["repo"], cfg["file"], revision=cfg.get("revision", ""),
                                  env_var="AGENTREEL_LLM_MODEL" if kind == "text" else "AGENTREEL_VLM_MODEL")
            mmproj = None
            if cfg.get("mmproj"):
                mmproj = resolve_model(cfg["repo"], cfg["mmproj"], revision=cfg.get("revision", ""),
                                       env_var="AGENTREEL_VLM_MMPROJ")
            cmd = build_command(binary, model, cfg, ctx.seed, mmproj)
            SERVERS_DIR.mkdir(parents=True, exist_ok=True)
            out = open(SERVERS_DIR / f"{alias}.log", "ab")  # noqa: SIM115 (handle owned by the child)
            kw: dict = {"stdout": out, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                        "env": _server_env(binary, str(s["gpu"])), "cwd": str(ROOT)}
            if os.name == "posix":
                kw["start_new_session"] = True   # own process group: survives the launcher, killpg frees VRAM
            else:
                kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            proc = subprocess.Popen(cmd, **kw)
            out.close()
            pid = proc.pid
            _PROCS[str(pid)] = proc
            _pid_file(alias).write_text(str(pid))
            rec["units"] = {"action": "start", "kind": kind, "flags": cmd[1:], "pid": pid, "url": url,
                            "binary": str(binary), "version": llama_version(binary),
                            "model_path": str(model), "mmproj": str(mmproj or "")}
            rec["output_path"] = str(SERVERS_DIR / f"{alias}.log")
        t0 = time.time()
        while not health(url):
            if not _alive(pid):
                tail = _tail(SERVERS_DIR / f"{alias}.log")
                raise ServerError(f"llama-server ({alias}) exited during load:\n{tail}")
            if time.time() - t0 > timeout_s:
                raise ServerError(f"llama-server ({alias}) not healthy after {timeout_s:.0f}s")
            time.sleep(1.0)
        rec["units"]["load_s"] = round(time.time() - t0, 1)
    return url


def _tail(path: Path, n: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-n:]
    except OSError:
        return ""


# ---------------------------------------------------------------- public API
def ensure_llm(ctx: RunContext, log: RunLogger | None = None, timeout_s: float = 900) -> str:
    """Make sure the text llama-server (GPU0) is up and healthy. Returns ctx.llm_base_url."""
    if ctx.mock_llm:
        return ctx.llm_base_url
    return _launch(ctx, log, "text", timeout_s)


def start_vlm(ctx: RunContext, log: RunLogger | None = None, timeout_s: float = 600) -> str:
    """Start the vision llama-server (GPU1, --mmproj). Returns ctx.vlm_base_url."""
    if ctx.mock_llm:
        return ctx.vlm_base_url
    return _launch(ctx, log, "vision", timeout_s)


def _stop(ctx: RunContext, log: RunLogger | None, kind: str) -> None:
    cfg = load_yaml("config/models.yaml")["llm"][kind]
    alias = cfg["alias"]
    pid = _read_pid(alias)
    log = log or RunLogger(ctx)
    with log.call("platform", "server", model=alias, gpu_id=str(cfg["server"]["gpu"])) as rec:
        rec["units"] = {"action": "stop", "kind": kind, "pid": pid, "was_alive": _alive(pid)}
        if pid:
            _kill_group(pid)
        _pid_file(alias).unlink(missing_ok=True)


def stop_vlm(ctx: RunContext, log: RunLogger | None = None) -> None:
    """Kill the VLM server's process group (frees GPU1)."""
    if ctx.mock_llm:
        return
    _stop(ctx, log, "vision")


def stop_llm(ctx: RunContext, log: RunLogger | None = None) -> None:
    """Kill the text server (end of session / switching models)."""
    if ctx.mock_llm:
        return
    _stop(ctx, log, "text")


def platform_context(seed: int = 42, run_id: str = "platform") -> RunContext:
    """RunContext for setup-time calls (setup_kaggle.sh, notebook cells)."""
    return RunContext(run_id=run_id, seed=seed, brief_path=ROOT / "briefs" / "P00.yaml")


def main(argv: list[str] | None = None) -> int:
    """python -m core.servers {start-llm|start-vlm|stop-vlm|stop-llm|status} [--seed N]"""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("action", choices=["start-llm", "start-vlm", "stop-vlm", "stop-llm", "status"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    ctx = platform_context(a.seed)
    if a.action == "status":
        cfg = load_yaml("config/models.yaml")["llm"]
        for kind in ("text", "vision"):
            url = f"http://127.0.0.1:{cfg[kind]['server']['port']}/v1"
            print(f"{kind:7s} {url} healthy={health(url)} pid={_read_pid(cfg[kind]['alias'])}")
        return 0
    fn = {"start-llm": ensure_llm, "start-vlm": start_vlm, "stop-vlm": stop_vlm, "stop-llm": stop_llm}[a.action]
    out = fn(ctx)
    print(out or "ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
