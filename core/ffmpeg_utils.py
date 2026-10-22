"""ffmpeg/ffprobe helpers: binary resolution, probe, EBU R128 loudness, two-pass loudnorm, final delivery encode.

Binary resolution order: $AGENTREEL_FFMPEG / $AGENTREEL_FFPROBE -> system PATH (Kaggle) -> static_ffmpeg (dev).
Every call can be logged through `RunLogger.call(agent, "ffmpeg", ...)` by passing `log=`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

# Delivery spec (PLAN §6.7 / §9)
TARGET_I, TARGET_TP, TARGET_LRA = -14.0, -1.5, 11.0
SAMPLE_RATE, CHANNELS = 48000, 2


class FFmpegError(RuntimeError):
    pass


@lru_cache(maxsize=None)
def _resolve(name: str) -> str:
    env = os.environ.get(f"AGENTREEL_{name.upper()}")
    if env and Path(env).exists():
        return env
    found = shutil.which(name)
    if found:
        return found
    try:
        import static_ffmpeg  # dev only (pip static-ffmpeg)

        static_ffmpeg.add_paths(weak=True)
    except Exception:  # noqa: BLE001
        pass
    found = shutil.which(name)
    if not found:
        raise FFmpegError(f"{name} not found (install ffmpeg or `pip install static-ffmpeg`)")
    return found


def ffmpeg_bin() -> str:
    return _resolve("ffmpeg")


def ffprobe_bin() -> str:
    return _resolve("ffprobe")


def run(args: list[str], *, log=None, agent: str = "editor", tool: str = "ffmpeg", output_path: str | Path = "",
        timeout: int = 1800, units: dict | None = None) -> subprocess.CompletedProcess:
    """Run ffmpeg/ffprobe (args without the binary for tool in {ffmpeg, ffprobe}); raises FFmpegError on failure."""
    binary = ffprobe_bin() if tool == "ffprobe" else ffmpeg_bin()
    cmd = [binary, *[str(a) for a in args]]
    ctx = log.call(agent, tool, units={"args": " ".join(cmd[1:])[:1500], **(units or {})}) if log else nullcontext({})
    with ctx as rec:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        if output_path:
            rec["output_path"] = str(output_path)
        if p.returncode != 0:
            raise FFmpegError(f"{tool} exited {p.returncode}: {p.stderr[-2000:]}")
    return p


# ---------------------------------------------------------------- probe
def probe(path: str | Path, *, log=None, agent: str = "editor") -> dict[str, Any]:
    """ffprobe JSON (format + streams) plus a flat `summary` used by tests and the critic."""
    p = run(["-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], log=log, agent=agent, tool="ffprobe")
    data = json.loads(p.stdout)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = data.get("format", {})
    data["summary"] = {
        "duration": float(fmt.get("duration", 0) or 0),
        "size_bytes": int(fmt.get("size", 0) or 0),
        "v_codec": v.get("codec_name", ""), "v_profile": v.get("profile", ""),
        "width": v.get("width", 0), "height": v.get("height", 0), "pix_fmt": v.get("pix_fmt", ""),
        "r_frame_rate": v.get("r_frame_rate", ""), "avg_frame_rate": v.get("avg_frame_rate", ""),
        "nb_frames": int(v.get("nb_frames", 0) or 0),
        "a_codec": a.get("codec_name", ""), "sample_rate": int(a.get("sample_rate", 0) or 0),
        "channels": a.get("channels", 0), "a_bit_rate": int(a.get("bit_rate", 0) or 0),
    }
    return data


def duration_of(path: str | Path) -> float:
    return probe(path)["summary"]["duration"]


# ---------------------------------------------------------------- loudness
def measure_loudness(path: str | Path, *, log=None, agent: str = "editor") -> dict[str, float]:
    """EBU R128 via `ebur128=peak=true`: {"I": LUFS, "LRA": LU, "TP": dBTP} (-inf -> -99)."""
    p = run(["-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0", "-af", "ebur128=peak=true",
             "-f", "null", "-"], log=log, agent=agent, units={"measure": "ebur128"})
    summary = p.stderr[p.stderr.rfind("Summary:"):]

    def grab(pattern: str) -> float:
        m = re.search(pattern, summary, re.S)
        if not m:
            return -99.0
        s = m.group(1)
        return -99.0 if "inf" in s else float(s)

    return {"I": grab(r"Integrated loudness:.*?I:\s+(-?[\d.]+|-inf)"),
            "LRA": grab(r"Loudness range:.*?LRA:\s+(-?[\d.]+|-inf)"),
            "TP": grab(r"True peak:.*?Peak:\s+(-?[\d.]+|-inf)")}


def _loudnorm_args(i: float, tp: float, lra: float) -> str:
    return f"loudnorm=I={i}:TP={tp}:LRA={lra}"


def loudnorm_measure(path: str | Path, *, log=None, agent: str = "editor",
                     i: float = TARGET_I, tp: float = TARGET_TP, lra: float = TARGET_LRA) -> dict[str, str]:
    """loudnorm pass 1 -> measured input_i / input_tp / input_lra / input_thresh / target_offset (strings)."""
    p = run(["-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
             "-af", f"{_loudnorm_args(i, tp, lra)}:print_format=json", "-f", "null", "-"],
            log=log, agent=agent, units={"pass": 1})
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        raise FFmpegError(f"loudnorm pass 1 printed no JSON: {p.stderr[-800:]}")
    return json.loads(m[-1])


def loudnorm_two_pass(in_audio: str | Path, out_wav: str | Path, *, duration: float | None = None, log=None,
                      agent: str = "editor", i: float = TARGET_I, tp: float = TARGET_TP,
                      lra: float = TARGET_LRA) -> dict[str, Any]:
    """Two-pass loudnorm (linear=true, measured values) -> 48 kHz stereo WAV, padded/trimmed to `duration`.

    Speech is peaky: reaching I=-14 by pure gain often pushes true peak above -1.5 dBTP, and loudnorm then
    silently switches to dynamic mode (which undershoots on short clips). So while linear mode is infeasible,
    we pre-condition: gain to target + 4x-oversampled limiter at TP-1 dB, re-measure (<= 4 rounds). The final
    pass is always loudnorm linear=true with the measured values. Silent input is only resampled."""
    src = Path(in_audio)
    measured = loudnorm_measure(src, log=log, agent=agent, i=i, tp=tp, lra=lra)
    silent = "inf" in str(measured.get("input_i", "")).lower() or float(measured.get("input_i", -99)) < -70
    rounds: list[dict] = []
    i0 = float(measured["input_i"]) if not silent else 0.0
    gain = i - i0
    for k in range(3):   # secant search on the gain applied to the ORIGINAL (no cascaded limiting)
        if silent or float(measured["input_tp"]) + (i - float(measured["input_i"])) <= tp - 0.3:
            break                       # linear mode feasible
        pre = Path(out_wav).with_name(f"{Path(out_wav).stem}_pre{k}.wav")
        limit = 10 ** ((tp - 1.0) / 20)
        run(["-y", "-hide_banner", "-nostats", "-i", str(in_audio), "-map", "0:a:0", "-af",
             f"aresample=192000,volume={gain:.2f}dB,alimiter=limit={limit:.4f}:attack=2:release=60:level=disabled,"
             f"aresample={SAMPLE_RATE}", "-c:a", "pcm_f32le", str(pre)],
            log=log, agent=agent, output_path=pre, units={"pass": f"precondition{k}", "gain_db": round(gain, 2)})
        src = pre
        measured = loudnorm_measure(src, log=log, agent=agent, i=i, tp=tp, lra=lra)
        got = float(measured["input_i"])
        rounds.append({"gain_db": round(gain, 2), "input_i": measured["input_i"], "input_tp": measured["input_tp"]})
        if abs(got - i) < 0.3:
            break
        eff = max(0.25, (got - i0) / gain) if gain else 1.0
        gain = min(i - i0 + 9.0, gain + (i - got) / eff)
    in_audio = src
    chain = []
    if not silent:
        chain.append(
            f"{_loudnorm_args(i, tp, lra)}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true:print_format=json")
    chain.append(f"aresample={SAMPLE_RATE}")
    if duration:
        chain.append("apad")
    args = ["-y", "-hide_banner", "-nostats", "-i", str(in_audio), "-map", "0:a:0", "-af", ",".join(chain),
            "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-c:a", "pcm_s16le"]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    p = run([*args, str(out_wav)], log=log, agent=agent, output_path=out_wav, units={"pass": 2, "silent": silent})
    report: dict[str, Any] = {}
    m = re.findall(r"\{[^{}]*\"normalization_type\"[^{}]*\}", p.stderr, re.S)
    if m:
        report = json.loads(m[-1])
    return {"measured": measured, "precondition": rounds, "pass2": report, "silent": silent, "out_wav": str(out_wav)}


# ---------------------------------------------------------------- delivery encode
AAC_TARGET_BPS, AAC_LO_BPS, AAC_HI_BPS = 168_000, 150_000, 184_000


def encode_aac(wav: str | Path, out_m4a: str | Path, *, duration: float | None = None, log=None,
               agent: str = "editor") -> dict[str, int]:
    """AAC-LC 48 kHz stereo whose MEASURED bitrate lands in 150-184 kb/s (QA gate 128-192). ffmpeg's native
    encoder undershoots on sparse speech (160k nominal -> ~120 kb/s), so the nominal rate is corrected from
    the measured one (<= 4 deterministic tries)."""
    nominal, actual = 192_000, 0
    for _ in range(4):
        args = ["-y", "-hide_banner", "-nostats", "-i", str(wav), "-map", "0:a:0", "-c:a", "aac", "-b:a", str(nominal),
                "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS)]
        if duration:
            args += ["-t", f"{duration:.3f}"]
        run([*args, str(out_m4a)], log=log, agent=agent, output_path=out_m4a, units={"stage": "aac", "nominal_bps": nominal})
        actual = probe(out_m4a)["summary"]["a_bit_rate"]
        if AAC_LO_BPS <= actual <= AAC_HI_BPS or not actual:
            break
        nominal = int(min(320_000, max(96_000, nominal * AAC_TARGET_BPS / actual)) // 1000 * 1000)
    return {"nominal_bps": nominal, "measured_bps": actual}


def final_encode(video_in: str | Path, audio_in: str | Path | None, out_path: str | Path, *, duration: float | None = None,
                 fps: int = 30, crf: int = 20, preset: str = "medium", log=None, agent: str = "editor") -> Path:
    """Delivery encode: H.264 High yuv420p CFR 30 (crf 20, preset medium) + AAC 48 kHz stereo (bitrate via
    `encode_aac`), +faststart. Audio from `audio_in` (normalized WAV) or, when None, the input video's track."""
    out_path = Path(out_path)
    m4a = out_path.with_name(out_path.stem + "_audio.m4a")
    encode_aac(audio_in or video_in, m4a, duration=duration, log=log, agent=agent)
    args = ["-y", "-hide_banner", "-nostats", "-i", str(video_in), "-i", str(m4a), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", str(fps), "-fps_mode", "cfr",
            "-crf", str(crf), "-preset", preset, "-c:a", "copy", "-movflags", "+faststart"]
    if duration:
        args += ["-t", f"{duration:.3f}"]
    run([*args, str(out_path)], log=log, agent=agent, output_path=out_path, units={"stage": "final_encode"})
    m4a.unlink(missing_ok=True)
    return out_path
