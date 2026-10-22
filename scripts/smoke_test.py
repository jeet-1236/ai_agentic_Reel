"""Platform smoke test: one LLM structured call, one image, one TTS line, a 2 s render, ffmpeg filter checks.

    python scripts/smoke_test.py            # mock LLM unless a llama-server is healthy; mock image/TTS workers
    python scripts/smoke_test.py --real     # real llama-server (started if needed), SDXL + Kokoro on GPU1
    python scripts/smoke_test.py --strict   # SKIP counts as failure

Writes to outputs/smoke-<ts>/ (its own logs/ subfolder, so logs/run_log.jsonl of real runs stays clean).
Exit code 0 = no FAIL.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel  # noqa: E402

from core.context import RunContext, load_yaml  # noqa: E402

RESULTS: list[tuple[str, str, float, str]] = []   # (check, PASS|FAIL|SKIP, seconds, detail)


@dataclass
class SmokeContext(RunContext):
    """Logs into outputs/<run_id>/logs instead of the shared logs/ folder."""

    @property
    def logs_dir(self) -> Path:
        p = self.out_dir / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p


class SmokeOut(BaseModel):
    headline: str
    bullets: list[str]
    score: int


def check(name: str):
    """Decorator: run fn() -> detail str; record PASS/FAIL/SKIP + timing."""
    def deco(fn):
        def run(*a, **kw):
            t0 = time.perf_counter()
            try:
                detail = fn(*a, **kw)
                status = "SKIP" if isinstance(detail, str) and detail.startswith("SKIP") else "PASS"
            except Exception as e:  # noqa: BLE001
                status, detail = "FAIL", f"{e.__class__.__name__}: {str(e)[:300]}"
            dt = time.perf_counter() - t0
            RESULTS.append((name, status, dt, str(detail)))
            print(f"[{status}] {name:22s} {dt:7.2f}s  {detail}", flush=True)
            return status
        return run
    return deco


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    try:
        from core import ffmpeg_utils

        return ffmpeg_utils.ffmpeg_bin() if name == "ffmpeg" else ffmpeg_utils.ffprobe_bin()
    except (ImportError, AttributeError):
        pass
    found = shutil.which(name)
    if found:
        return found
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
        return shutil.which(name) or name
    except ImportError:
        return name


@check("ffmpeg filters")
def check_ffmpeg() -> str:
    ff = ffmpeg_bin()
    filters = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=60).stdout
    enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60).stdout
    need_f = ["loudnorm", "ebur128", "subtitles", "ass", "silencedetect", "blackdetect", "freezedetect", "zoompan"]
    missing = [f for f in need_f if f" {f} " not in filters]
    missing += [e for e in ("libx264", "aac") if e not in enc]
    if missing:
        raise RuntimeError(f"{ff} missing: {missing}")
    return f"{ff}: {', '.join(need_f)} + libx264/aac"


@check("render 2s (ffmpeg)")
def check_render(out_dir: Path) -> str:
    """2 s 1080x1920 test card with libass captions + loudnorm -> h264 High/aac 48k, verified by ffprobe."""
    ff, fp = ffmpeg_bin(), ffmpeg_bin("ffprobe")
    srt = out_dir / "smoke.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nAgentReel smoke test\n", encoding="utf-8")
    mp4 = out_dir / "smoke_2s.mp4"
    sub = srt.name  # relative path avoids Windows drive-colon escaping in the filter graph
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30:duration=2",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
           "-vf", f"subtitles={sub}:force_style='FontSize=18,Outline=2'",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "2",
           "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", "30", "-fps_mode", "cfr",
           "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", mp4.name]
    r = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])
    probe = json.loads(subprocess.run([fp, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(mp4)],
                                      capture_output=True, text=True, timeout=60).stdout)
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    ok = (v["codec_name"], v.get("profile"), v["width"], v["height"], a["codec_name"], a["sample_rate"]) == \
         ("h264", "High", 1080, 1920, "aac", "48000")
    if not ok:
        raise RuntimeError(f"unexpected streams: {v['codec_name']}/{v.get('profile')} {v['width']}x{v['height']} "
                           f"{a['codec_name']}/{a['sample_rate']}")
    return f"{mp4.name} {float(probe['format']['duration']):.2f}s h264 High 1080x1920 aac 48k"


@check("hyperframes")
def check_hyperframes(ctx, log, real: bool) -> str:
    """Render the renderer owner's smoke fixture if present, else run `hyperframes doctor`."""
    if not shutil.which("node"):
        return "SKIP node not on PATH (renderer falls back to --renderer ffmpeg)"
    if not (ROOT / "renderer" / "node_modules" / "hyperframes").exists():
        return "SKIP renderer/node_modules missing (run setup_kaggle.sh)"
    fixture = ROOT / "renderer" / "fixtures" / "smoke_scene.json"
    worker = ROOT / "renderer" / "render_hyperframes.mjs"
    if real and fixture.exists() and worker.exists():
        from core.workers import run_worker

        out = run_worker(ctx, log, agent="smoke", worker="render_hyperframes.mjs", gpu=None,
                         request={"task": "render", "scene_path": str(fixture),
                                  "out_path": str(ctx.out_dir / "smoke_hf.mp4"), "mock": False})
        return f"rendered {out.get('video_path')}"
    npx = shutil.which("npx") or "npx"
    r = subprocess.run([npx, "--no-install", "hyperframes", "doctor"], cwd=ROOT / "renderer",
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[-400:])
    return "hyperframes doctor ok" + ("" if fixture.exists() else " (no renderer/fixtures/smoke_scene.json yet)")


@check("llm structured")
def check_llm(ctx, log, use_real: bool) -> str:
    from core.llm import LLM
    from core.seeds import stage_seed

    if use_real:
        from core.servers import ensure_llm

        ensure_llm(ctx, log)
    llm = LLM(ctx, log, mock=not use_real)
    t0 = time.perf_counter()
    out = llm.structured("smoke", "platform_smoke_v1", SmokeOut, seed=stage_seed(ctx.seed, "smoke"),
                         topic="using AI tools responsibly in college assignments")
    dt = time.perf_counter() - t0
    tail = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()][-1]
    toks = tail.get("output_tokens") or 0
    rate = f", {toks / dt:.1f} tok/s" if toks and use_real else ""
    return f"{'real' if use_real else 'mock'}: '{out.headline}' ({len(out.bullets)} bullets, {toks} out tokens{rate})"


def _worker_exists(name: str) -> bool:
    return (ROOT / "workers" / name).exists()


@check("image (sdxl)")
def check_image(ctx, log, real: bool) -> str:
    if not _worker_exists("image_sdxl.py"):
        return "SKIP workers/image_sdxl.py not written yet"
    from core.workers import run_worker

    w, h = load_yaml("config/models.yaml")["image"]["size_9x16"]
    out = run_worker(ctx, log, agent="smoke", worker="image_sdxl.py", gpu="1", request={
        "task": "generate", "prompt": "flat vector illustration of a laptop on a desk, pastel colours, no people",
        "negative": "photo, realistic, person, face, text", "width": w, "height": h, "seed": ctx.seed, "n": 1,
        "out_prefix": "smoke_img", "mock": not real})
    return f"{'real' if real else 'mock'}: {out.get('paths')}"


@check("tts (kokoro)")
def check_tts(ctx, log, real: bool) -> str:
    if not _worker_exists("tts_kokoro.py"):
        return "SKIP workers/tts_kokoro.py not written yet"
    from core.workers import run_worker

    v = load_yaml("config/models.yaml")["voice"]["tts"]
    out = run_worker(ctx, log, agent="smoke", worker="tts_kokoro.py", gpu="1", request={
        "task": "synth", "sentences": ["This is the AgentReel smoke test."], "voice": v.get("voice", "af_heart"),
        "speed": v.get("speed", 1.0), "seed": ctx.seed, "lexicon": {}, "mock": not real})
    return f"{'real' if real else 'mock'}: {out.get('wav_path')} ({len(out.get('tokens', []))} tokens)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="use the GPUs: real llama-server, SDXL, Kokoro")
    ap.add_argument("--strict", action="store_true", help="treat SKIP as failure")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    from core.logging import RunLogger
    from core.servers import health

    run_id = f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    probe_ctx = RunContext(run_id=run_id, seed=a.seed, brief_path=ROOT / "briefs" / "P00.yaml")
    use_real_llm = a.real or health(probe_ctx.llm_base_url)
    ctx = SmokeContext(run_id=run_id, seed=a.seed, brief_path=ROOT / "briefs" / "P00.yaml",
                       mock_llm=not use_real_llm, mock_media=not a.real)
    log = RunLogger(ctx)
    print(f"AgentReel smoke test -> {ctx.out_dir} (llm={'real' if use_real_llm else 'mock'}, "
          f"media={'real' if a.real else 'mock'})", flush=True)

    t0 = time.perf_counter()
    check_ffmpeg()
    check_llm(ctx, log, use_real_llm)
    check_image(ctx, log, a.real)
    check_tts(ctx, log, a.real)
    check_render(ctx.out_dir)
    check_hyperframes(ctx, log, a.real)
    total = time.perf_counter() - t0

    print("\n  check                   status    seconds  detail")
    for name, status, dt, detail in RESULTS:
        print(f"  {name:22s}  {status:6s} {dt:9.2f}  {detail[:90]}")
    print(f"  {'TOTAL':22s}  {'':6s} {total:9.2f}")
    (ctx.out_dir / "smoke_results.json").write_text(json.dumps(
        [{"check": n, "status": s, "seconds": round(d, 2), "detail": x} for n, s, d, x in RESULTS], indent=2),
        encoding="utf-8")
    bad = [r for r in RESULTS if r[1] == "FAIL" or (a.strict and r[1] == "SKIP")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
