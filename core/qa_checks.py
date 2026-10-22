"""QA hard gate (PLAN §9): pure check functions, each returning a `QACheck`.

Layout:
- media measurement (`ffprobe_json`, `measure_media`, `parse_*`) shells out to ffprobe/ffmpeg once per video;
- `check_*` functions are pure: they take measurements / state slices and return QACheck(s);
- small helpers (WCAG contrast, contact regexes, fuzzy match, OCR, logo match) are reusable.

Binaries resolve via core.ffmpeg_utils (when present) -> PATH -> static_ffmpeg (dev). Heavy imports
(numpy, PIL, cv2, rapidocr) stay inside functions.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

from core.schemas import QACheck

# ---------------------------------------------------------------- thresholds (PLAN §9)
DUR_MIN, DUR_MAX, DUR_TOL = 30.0, 60.0, 2.0
AUDIO_RATE, AUDIO_CH, ABR_MIN, ABR_MAX = 48000, 2, 128_000, 192_000
LUFS_TARGET, LUFS_TOL, TP_MAX = -14.0, 1.0, -1.0
SIZE_MAX = 100 * 1024 * 1024
SAFE_Y_9x16 = (250.0, 1570.0)                 # handout band for a 1920-px-high frame
CAPTION_MIN_PX = 48.0
CAP_MAX_LINES, CAP_MAX_CHARS, CAP_MAX_CPS, CAP_MIN_DUR = 2, 32, 17.0, 0.83
WCAG_MIN = 4.5
HOOK_T = 3.0
FUZZY_MIN = 0.9
ONSCREEN_MAX_WORDS = 7
END_CARD_S = 4.0
DEFAULT_LICENCES = {
    "apache-2.0", "mit", "bsd-3-clause", "ofl-1.1", "ofl", "creativeml-openrail++-m", "cc0-1.0", "cc0",
    "cc-by-4.0", "standard no-charge license", "official", "renderer", "generated",
}


def qc(check_id: str, passed: bool, agent: str, *, value: Any = "", threshold: Any = "", evidence: str = "",
       severity: str = "hard") -> QACheck:
    """Shorthand constructor (stringifies value/threshold)."""
    return QACheck(check_id=check_id, passed=bool(passed), value=str(value), threshold=str(threshold),
                   evidence=evidence[:1500], responsible_agent=agent, severity=severity)  # type: ignore[arg-type]


# ================================================================ binaries / measurement
def _resolve(name: str) -> str:
    try:
        from core import ffmpeg_utils  # renderer engineer's module (may not exist yet)

        fn = getattr(ffmpeg_utils, f"{name}_bin", None)
        if fn:
            return str(fn())
    except Exception:  # noqa: BLE001
        pass
    found = shutil.which(name)
    if found:
        return found
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except Exception:  # noqa: BLE001
        pass
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} not found (install ffmpeg or static_ffmpeg)")
    return found


def ffmpeg_path() -> str:
    return _resolve("ffmpeg")


def ffprobe_path() -> str:
    return _resolve("ffprobe")


def ffprobe_json(path: str | Path) -> dict:
    """ffprobe -show_streams -show_format as a dict."""
    out = subprocess.run([ffprobe_path(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr[-500:]}")
    return json.loads(out.stdout or "{}")


def measure_media(path: str | Path, *, freeze_d: float = 3.0, timeout: int = 900) -> dict:
    """One decode pass: vfrdet/blackdetect/freezedetect on video, ebur128/astats/silencedetect on audio.

    Returns {"probe": dict, "size": int, "stderr": str, **parsed} with parsed keys from `parse_filter_log`.
    """
    path = Path(path)
    probe = ffprobe_json(path)
    streams = probe.get("streams", [])
    has_v = any(s.get("codec_type") == "video" for s in streams)
    has_a = any(s.get("codec_type") == "audio" for s in streams)
    cmd = [ffmpeg_path(), "-hide_banner", "-nostats", "-i", str(path)]
    if has_v:
        cmd += ["-map", "0:v:0", "-vf", f"vfrdet,blackdetect=d=0.5:pix_th=0.10,freezedetect=n=-60dB:d={freeze_d}"]
    if has_a:
        cmd += ["-map", "0:a:0", "-af", "ebur128=peak=true,astats=metadata=0,silencedetect=n=-50dB:d=1.5"]
    cmd += ["-f", "null", "-"]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    parsed = parse_filter_log(run.stderr)
    return {"probe": probe, "size": path.stat().st_size, "returncode": run.returncode,
            "stderr_tail": run.stderr[-3000:], "has_video": has_v, "has_audio": has_a, **parsed}


def parse_filter_log(text: str) -> dict:
    """Parse ffmpeg stderr from the measurement pass."""
    out: dict[str, Any] = {"vfr": None, "integrated_lufs": None, "true_peak_dbtp": None, "astats_nan": False,
                           "silences": [], "blacks": [], "freezes": []}
    m = re.findall(r"VFR:\s*([\d.]+)\s*\((\d+)/(\d+)\)", text)
    if m:
        out["vfr"] = {"ratio": float(m[-1][0]), "vfr": int(m[-1][1]), "cfr": int(m[-1][2])}
    summary = text[text.rfind("Summary:"):] if "Summary:" in text else ""
    mi = re.search(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d.]+|-inf)\s*LUFS", summary)
    if mi:
        out["integrated_lufs"] = float(mi.group(1)) if mi.group(1) != "-inf" else float("-inf")
    mp = re.search(r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+|-inf)\s*dBFS", summary)
    if mp:
        out["true_peak_dbtp"] = float(mp.group(1)) if mp.group(1) != "-inf" else float("-inf")
    astats = [ln for ln in text.splitlines() if "Parsed_astats" in ln]
    out["astats_nan"] = any(re.search(r":\s*-?nan\b", ln, flags=re.I) for ln in astats)
    out["astats_lines"] = len(astats)
    out["silences"] = _intervals(text, r"silence_start:\s*(-?[\d.]+)", r"silence_end:\s*(-?[\d.]+)")
    out["blacks"] = [(float(a), float(b)) for a, b in re.findall(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)", text)]
    out["freezes"] = _intervals(text, r"freeze_start:\s*([\d.]+)", r"freeze_end:\s*([\d.]+)")
    return out


def _intervals(text: str, start_re: str, end_re: str) -> list[tuple[float, float | None]]:
    events = sorted([(m.start(), "s", float(m.group(1))) for m in re.finditer(start_re, text)]
                    + [(m.start(), "e", float(m.group(1))) for m in re.finditer(end_re, text)])
    res: list[tuple[float, float | None]] = []
    cur: float | None = None
    for _, kind, t in events:
        if kind == "s":
            cur = max(0.0, t)
        elif cur is not None:
            res.append((cur, t))
            cur = None
    if cur is not None:
        res.append((cur, None))      # still open at EOF
    return res


def extract_frame(video: str | Path, t: float, out_png: str | Path) -> Path:
    """Grab one frame at time t (seconds) as PNG."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(0.0, t):.3f}",
                    "-i", str(video), "-frames:v", "1", str(out_png)], check=True, capture_output=True, timeout=120)
    return out_png


# ================================================================ spec checks (Editor)
def _stream(probe: dict, kind: str) -> dict:
    return next((s for s in probe.get("streams", []) if s.get("codec_type") == kind), {})


def duration_of(m: dict) -> float:
    fmt = m.get("probe", {}).get("format", {})
    try:
        return float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        return 0.0


def check_duration(m: dict, target_s: float, voice_duration_s: float = 0.0) -> list[QACheck]:
    """30-60 s and brief target ±2 s. Long/short voice -> Scriptwriter (word budget), else Editor."""
    d = duration_of(m)
    agent = "scriptwriter" if voice_duration_s and abs(d - voice_duration_s) <= END_CARD_S + 2.0 else "editor"
    ev = f"ffprobe format.duration={d:.2f}s; voice={voice_duration_s:.2f}s"
    return [
        qc("duration_range", DUR_MIN <= d <= DUR_MAX, agent, value=f"{d:.2f}", threshold=f"[{DUR_MIN:g},{DUR_MAX:g}] s", evidence=ev),
        qc("duration_target", abs(d - target_s) <= DUR_TOL, agent, value=f"{d:.2f}",
           threshold=f"{target_s:g}±{DUR_TOL:g} s", evidence=ev),
    ]


def check_video_stream(m: dict, width: int, height: int, fps: int = 30) -> list[QACheck]:
    v = _stream(m.get("probe", {}), "video")
    if not v:
        return [qc("video_stream", False, "editor", value="missing", threshold="h264 stream", evidence="no video stream")]
    size = f"{v.get('width')}x{v.get('height')}"
    rfr, afr = v.get("r_frame_rate", ""), v.get("avg_frame_rate", "")
    vfr = m.get("vfr") or {}
    cfr_ok = rfr == afr == f"{fps}/1" and vfr.get("vfr") in (0, None)   # None: vfrdet not reported
    return [
        qc("video_codec", v.get("codec_name") == "h264", "editor", value=v.get("codec_name"), threshold="h264"),
        qc("video_profile", str(v.get("profile", "")).lower() == "high", "editor", value=v.get("profile"), threshold="High"),
        qc("video_size", size == f"{width}x{height}", "editor", value=size, threshold=f"{width}x{height}"),
        qc("video_pix_fmt", v.get("pix_fmt") == "yuv420p", "editor", value=v.get("pix_fmt"), threshold="yuv420p"),
        qc("video_cfr", cfr_ok, "editor", value=f"r={rfr} avg={afr} vfrdet={vfr or 'n/a'}",
           threshold=f"r=avg={fps}/1, VFR=0", evidence="ffprobe r_frame_rate/avg_frame_rate + vfrdet"),
    ]


def check_audio_stream(m: dict) -> list[QACheck]:
    a = _stream(m.get("probe", {}), "audio")
    if not a:
        return [qc("audio_stream", False, "editor", value="missing", threshold="aac stream", evidence="no audio stream")]
    try:
        br = int(a.get("bit_rate") or 0)
    except ValueError:
        br = 0
    return [
        qc("audio_codec", a.get("codec_name") == "aac", "editor", value=a.get("codec_name"), threshold="aac"),
        qc("audio_sample_rate", str(a.get("sample_rate")) == str(AUDIO_RATE), "editor", value=a.get("sample_rate"),
           threshold=AUDIO_RATE),
        qc("audio_channels", a.get("channels") == AUDIO_CH, "editor", value=a.get("channels"), threshold=AUDIO_CH),
        qc("audio_bitrate", ABR_MIN <= br <= ABR_MAX, "editor", value=br, threshold=f"[{ABR_MIN},{ABR_MAX}] b/s"),
    ]


def check_loudness(m: dict) -> list[QACheck]:
    i, tp = m.get("integrated_lufs"), m.get("true_peak_dbtp")
    ev = "ebur128=peak=true summary"
    return [
        qc("loudness_integrated", i is not None and abs(i - LUFS_TARGET) <= LUFS_TOL, "editor", value=i,
           threshold=f"{LUFS_TARGET:g}±{LUFS_TOL:g} LUFS", evidence=ev),
        qc("loudness_true_peak", tp is not None and tp <= TP_MAX, "editor", value=tp, threshold=f"<= {TP_MAX:g} dBTP",
           evidence=ev),
        qc("audio_nan", not m.get("astats_nan", False) and m.get("astats_lines", 1) > 0, "editor",
           value="NaN found" if m.get("astats_nan") else "ok", threshold="no NaN", evidence="astats"),
    ]


def check_size(m: dict) -> QACheck:
    s = int(m.get("size", 0))
    return qc("file_size", 0 < s <= SIZE_MAX, "editor", value=f"{s / 1e6:.1f} MB", threshold="<= 100 MB")


def _covered(iv: tuple[float, float], allowed: Iterable[tuple[float, float]], slack: float = 0.25) -> bool:
    return any(a - slack <= iv[0] and iv[1] <= b + slack for a, b in allowed)


def allowed_windows(duration: float, shot_bounds: Iterable[float] = (), end_card_s: float = END_CARD_S,
                    transition_pad: float = 0.6) -> list[tuple[float, float]]:
    """Planned non-content windows: start 0.5 s, end card, ±pad around each shot boundary."""
    win = [(0.0, 0.5), (max(0.0, duration - end_card_s - 0.5), duration + 1.0)]
    win += [(max(0.0, t - transition_pad), t + transition_pad) for t in shot_bounds]
    return win


def check_dead_air(m: dict, allowed: list[tuple[float, float]]) -> list[QACheck]:
    """silencedetect / blackdetect / freezedetect intervals must lie inside planned windows."""
    d = duration_of(m)
    res = []
    for cid, key, what in (("dead_air", "silences", "silence"), ("black_frames", "blacks", "black"),
                           ("frozen_frames", "freezes", "freeze")):
        bad = [(a, b if b is not None else d) for a, b in m.get(key, [])]
        bad = [iv for iv in bad if not _covered(iv, allowed)]
        res.append(qc(cid, not bad, "editor", value=f"{len(bad)} unplanned {what} interval(s)",
                      threshold="none outside planned shots",
                      evidence="; ".join(f"{a:.2f}-{b:.2f}s" for a, b in bad[:8])))
    return res


# ================================================================ layout / captions
def _box_xywh(b: dict) -> tuple[float, float, float, float]:
    if "bbox" in b:
        x, y, w, h = b["bbox"]
        return float(x), float(y), float(w), float(h)
    if "x0" in b:
        return float(b["x0"]), float(b["y0"]), float(b["x1"]) - float(b["x0"]), float(b["y1"]) - float(b["y0"])
    return float(b.get("x", 0)), float(b.get("y", 0)), float(b.get("w", b.get("width", 0))), float(b.get("h", b.get("height", 0)))


def layout_boxes(layout: Any) -> list[dict]:
    """Flatten layout.json (list of boxes | {"boxes": [...]} | {"shots": [{"shot_id", "boxes": [...]}]})."""
    if isinstance(layout, list):
        return [b for b in layout if isinstance(b, dict)]
    if not isinstance(layout, dict):
        return []
    out = [dict(b) for b in layout.get("boxes", []) if isinstance(b, dict)]
    for s in layout.get("shots", []) or []:
        for b in s.get("boxes", []) or []:
            out.append({"shot_id": s.get("shot_id"), **b})
    return out


def safe_band(width: int, height: int) -> tuple[float, float]:
    """Vertical text safe band: core.scene.safe_zones().text when available, else [250,1570]·height/1920."""
    try:
        from core.scene import safe_zones

        t = safe_zones(width, height).text
        return float(t.y0), float(t.y1)
    except Exception:  # noqa: BLE001
        s = height / 1920.0
        return SAFE_Y_9x16[0] * s, SAFE_Y_9x16[1] * s


def check_caption_layout(layout: Any, width: int, height: int, *, source: str = "layout.json") -> list[QACheck]:
    """All text boxes inside y∈[250,1570] (scaled by height/1920); caption text height ≥ 48 px (scaled)."""
    boxes = layout_boxes(layout)
    if not boxes:
        return [qc("caption_safe_zone", True, "storyboarder", value="unverified", threshold="y in [250,1570]",
                   evidence=f"no text boxes from {source}", severity="soft")]
    y0, y1 = safe_band(width, height)
    outside, small = [], []
    min_px = CAPTION_MIN_PX      # absolute rule in both aspects
    for b in boxes:
        x, y, w, h = _box_xywh(b)
        if b.get("kind", b.get("type", "text")) in ("logo", "image"):
            continue
        tag = f"shot {b.get('shot_id', '?')} '{str(b.get('text', ''))[:30]}' y={y:.0f}-{y + h:.0f}"
        if y < y0 - 0.5 or y + h > y1 + 0.5:
            outside.append(tag)
        if str(b.get("kind", b.get("type", ""))).startswith("caption"):
            lines = max(1, int(b.get("lines", 1) or 1))
            px = float(b.get("font_px") or b.get("font_size") or (h / lines))
            if px < min_px:
                small.append(f"{tag} font={px:.0f}px")
    return [
        qc("caption_safe_zone", not outside, "storyboarder", value=f"{len(outside)} box(es) outside",
           threshold=f"y in [{y0:.0f},{y1:.0f}]", evidence=f"{source}: " + "; ".join(outside[:8])),
        qc("caption_height", not small, "editor", value=f"{len(small)} small", threshold=f">= {min_px:.0f} px",
           evidence=f"{source}: " + "; ".join(small[:8])),
    ]


def parse_srt(path: str | Path) -> list[dict]:
    """Minimal SRT -> [{index,start,end,lines}]."""
    def ts(s: str) -> float:
        h, m, rest = s.strip().split(":")
        sec, ms = rest.replace(".", ",").split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000

    pages = []
    for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8-sig").strip()):
        rows = [r for r in block.splitlines() if r.strip()]
        if len(rows) >= 2 and "-->" in rows[1]:
            a, b = rows[1].split("-->")
            pages.append({"index": int(rows[0]) if rows[0].strip().isdigit() else len(pages),
                          "start": ts(a), "end": ts(b), "lines": rows[2:]})
    return pages


def check_caption_readability(pages: list[dict]) -> list[QACheck]:
    """≤2 lines, ≤32 chars/line (hard); ≤17 CPS, ≥0.83 s (soft) per page."""
    if not pages:
        return [qc("caption_present", False, "asset_generator", value="0 pages", threshold=">= 1 caption page",
                   evidence="no caption track")]
    lines_bad, chars_bad, cps_bad, dur_bad = [], [], [], []
    for p in pages:
        lines = [ln for ln in p.get("lines", [])]
        dur = float(p.get("end", 0)) - float(p.get("start", 0))
        n = sum(len(ln) for ln in lines)
        tag = f"#{p.get('index')}@{float(p.get('start', 0)):.2f}s"
        if len(lines) > CAP_MAX_LINES:
            lines_bad.append(f"{tag} {len(lines)} lines")
        for ln in lines:
            if len(ln) > CAP_MAX_CHARS:
                chars_bad.append(f"{tag} '{ln}' ({len(ln)})")
        if dur > 0 and n / dur > CAP_MAX_CPS:
            cps_bad.append(f"{tag} {n / dur:.1f} cps")
        if dur < CAP_MIN_DUR:
            dur_bad.append(f"{tag} {dur:.2f}s")
    a = "asset_generator"
    return [
        qc("caption_lines", not lines_bad, a, value=len(lines_bad), threshold=f"<= {CAP_MAX_LINES} lines", evidence="; ".join(lines_bad[:6])),
        qc("caption_chars_per_line", not chars_bad, a, value=len(chars_bad), threshold=f"<= {CAP_MAX_CHARS}", evidence="; ".join(chars_bad[:6])),
        qc("caption_cps", not cps_bad, a, value=len(cps_bad), threshold=f"<= {CAP_MAX_CPS:g} cps", evidence="; ".join(cps_bad[:6]), severity="soft"),
        qc("caption_min_duration", not dur_bad, a, value=len(dur_bad), threshold=f">= {CAP_MIN_DUR} s", evidence="; ".join(dur_bad[:6]), severity="soft"),
    ]


# ================================================================ contrast (WCAG 2.x)
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def relative_luminance(rgb: Iterable[float]) -> float:
    def lin(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(float(c)) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(c1: Iterable[float], c2: Iterable[float]) -> float:
    l1, l2 = sorted((relative_luminance(c1), relative_luminance(c2)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def sample_background(frame_path: str | Path, box: tuple[float, float, float, float], text_rgb: tuple[int, int, int],
                      stroke_rgb: tuple[int, int, int] | None = None) -> tuple[int, int, int]:
    """Worst-case background colour in a region: excludes glyph/stroke pixels, takes the pixel whose luminance
    is closest to the text colour at the 90th percentile."""
    import numpy as np
    from PIL import Image

    img = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32)
    x, y, w, h = (int(round(v)) for v in box)
    reg = img[max(0, y):max(0, y) + max(1, h), max(0, x):max(0, x) + max(1, w)].reshape(-1, 3)
    if reg.size == 0:
        reg = img.reshape(-1, 3)
    keep = np.linalg.norm(reg - np.array(text_rgb, dtype=np.float32), axis=1) > 60
    if stroke_rgb is not None:
        keep &= np.linalg.norm(reg - np.array(stroke_rgb, dtype=np.float32), axis=1) > 60
    bg = reg[keep] if keep.sum() > 50 else reg
    lum = bg @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    text_light = relative_luminance(text_rgb) > 0.5
    target = np.percentile(lum, 90 if text_light else 10)
    idx = int(np.argmin(np.abs(lum - target)))
    return tuple(int(v) for v in bg[idx])  # type: ignore[return-value]


def check_contrast(samples: list[dict]) -> QACheck:
    """samples: [{frame, text_rgb, bg_rgb, stroke_rgb?, stroke_px?}] -> min WCAG ratio ≥ 4.5 (outline ≥3 px counts)."""
    if not samples:
        return qc("caption_contrast", True, "editor", value="unverified", threshold=f">= {WCAG_MIN}:1",
                  evidence="no frames sampled", severity="soft")
    worst, ev = 99.0, []
    for s in samples:
        r = contrast_ratio(s["text_rgb"], s["bg_rgb"])
        if s.get("stroke_rgb") is not None and float(s.get("stroke_px", 0)) >= 3:
            r = max(r, contrast_ratio(s["text_rgb"], s["stroke_rgb"]))
        worst = min(worst, r)
        ev.append(f"{Path(str(s.get('frame', '?'))).name}:{r:.2f}")
    return qc("caption_contrast", worst >= WCAG_MIN, "editor", value=f"{worst:.2f}", threshold=f">= {WCAG_MIN}:1",
              evidence="; ".join(ev[:12]))


# ================================================================ text helpers
def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"[‘’`´]", "'", s)
    s = re.sub(r"[^a-z0-9@.'/+\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", norm_text(s))
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks]


STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are", "be", "with", "what", "who",
        "how", "your", "you", "it", "this", "that", "by", "at", "as", "from", "if", "no", "not", "any", "do", "doe"}


def fuzzy_contains(needle: str, hay: str) -> float:
    """Best partial-match similarity in [0,1] of needle inside hay (rapidfuzz if present, else difflib)."""
    n, h = norm_text(needle), norm_text(hay)
    if not n:
        return 1.0
    if not h:
        return 0.0
    try:
        from rapidfuzz import fuzz

        return fuzz.partial_ratio(n, h) / 100.0
    except ImportError:
        from difflib import SequenceMatcher

        if len(h) <= len(n):
            return SequenceMatcher(None, n, h).ratio()
        best = 0.0
        for i in range(0, len(h) - len(n) + 1, max(1, len(n) // 4)):
            best = max(best, SequenceMatcher(None, n, h[i:i + len(n)]).ratio())
        return best


# ================================================================ hook
def check_hook(keywords: list[str], words: list[dict], onscreen: list[str], t_max: float = HOOK_T) -> QACheck:
    """≥1 plan.topic_keyword spoken (word.start < 3 s) or shown on screen in 0-3 s."""
    spoken = " ".join(w.get("word", "") for w in words if float(w.get("start", 99)) < t_max)
    hay_tokens = set(_tokens(spoken + " " + " ".join(onscreen)))
    hay = " " + " ".join(_tokens(spoken + " " + " ".join(onscreen))) + " "
    hits = []
    for kw in keywords:
        kt = [t for t in _tokens(kw) if t not in STOP] or _tokens(kw)
        if kt and (f" {' '.join(_tokens(kw))} " in hay or all(t in hay_tokens for t in kt)):
            hits.append(kw)
    return qc("hook_topic_3s", bool(hits), "scriptwriter", value=hits or "none", threshold=f">=1 keyword by {t_max:g}s",
              evidence=f"spoken<{t_max:g}s: '{spoken[:120]}'; on-screen: {onscreen[:4]}; keywords: {keywords[:8]}")


# ================================================================ end card
def check_end_card(ocr_text: str, disclosure: str, cta_texts: list[str], *, logo: tuple[bool, float] | None = None,
                   declared_blob: str = "", declared_logo: bool = False, ocr_available: bool = True) -> list[QACheck]:
    """Disclosure + CTA fuzzy ≥0.9 in OCR of last 4 s; logo via template match. Falls back to scene.json."""
    use_ocr = ocr_available and bool(ocr_text.strip())
    hay = ocr_text if use_ocr else declared_blob
    src = "ocr" if use_ocr else "declared"
    d = fuzzy_contains(disclosure, hay)
    res = [qc("end_card_disclosure", d >= FUZZY_MIN, "editor", value=f"{d:.2f}", threshold=f">= {FUZZY_MIN}",
              evidence=f"{src}: expected '{disclosure}'")]
    ctas = [c for c in cta_texts if c.strip()]
    scores = [fuzzy_contains(c, hay) for c in ctas]
    worst = min(scores) if scores else 0.0
    res.append(qc("end_card_cta", bool(ctas) and worst >= FUZZY_MIN, "editor", value=f"{worst:.2f}",
                  threshold=f">= {FUZZY_MIN}", evidence=f"{src}: expected {ctas[:3]}"))
    if logo is not None:
        res.append(qc("end_card_logo", logo[0], "editor", value=f"{logo[1]:.2f}", threshold="matchTemplate >= 0.7",
                      evidence="cv2.matchTemplate on last-4 s frames"))
    else:
        res.append(qc("end_card_logo", declared_logo, "editor", value="declared" if declared_logo else "absent",
                      threshold="logo present", evidence="declared: scene.json logo reference (no raster template)"))
    return res


def match_logo(frames: list[str | Path], logo_path: str | Path, threshold: float = 0.7) -> tuple[bool, float] | None:
    """Multi-scale cv2.matchTemplate; None when cv2 or a raster logo is unavailable."""
    try:
        import cv2
    except ImportError:
        return None
    lp = Path(logo_path)
    if lp.suffix.lower() == ".svg":
        lp = next((p for p in (lp.with_suffix(".png"), lp.with_suffix(".jpg")) if p.exists()), lp)
    if not lp.exists() or lp.suffix.lower() == ".svg":
        return None
    tpl0 = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
    if tpl0 is None:
        return None
    best = 0.0
    for f in frames:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for scale in (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0):
            w = int(img.shape[1] * scale)
            h = int(tpl0.shape[0] * w / max(1, tpl0.shape[1]))
            if w < 16 or h < 8 or w > img.shape[1] or h > img.shape[0]:
                continue
            tpl = cv2.resize(tpl0, (w, h), interpolation=cv2.INTER_AREA)
            best = max(best, float(cv2.minMaxLoc(cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED))[1]))
    return best >= threshold, best


_OCR_ENGINE: dict[str, Any] = {}


def ocr_available() -> bool:
    try:
        _ocr_engine()
        return True
    except Exception:  # noqa: BLE001
        return False


def _ocr_engine():
    if "e" not in _OCR_ENGINE:
        try:
            from rapidocr import RapidOCR  # rapidocr >= 2
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[no-redef]
        _OCR_ENGINE["e"] = RapidOCR()
    return _OCR_ENGINE["e"]


def ocr_image(path: str | Path) -> list[dict]:
    """[{text, box:[x,y,w,h], score}] via RapidOCR (v1 tuple API or v2+ result object)."""
    eng = _ocr_engine()
    res = eng(str(path))
    items: list[dict] = []
    if isinstance(res, tuple):           # rapidocr_onnxruntime: (result, elapse)
        res = res[0] or []
        for quad, text, score in res:
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            items.append({"text": text, "box": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "score": float(score)})
        return items
    boxes, txts, scores = getattr(res, "boxes", None), getattr(res, "txts", None), getattr(res, "scores", None)
    if boxes is None or txts is None:
        return items
    for quad, text, score in zip(boxes, txts, scores or [1.0] * len(txts)):
        xs, ys = [float(p[0]) for p in quad], [float(p[1]) for p in quad]
        items.append({"text": text, "box": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)], "score": float(score)})
    return items


# ================================================================ facts / coverage / avoid
FACTUAL_HINT = re.compile(
    r"\d|%|₹|\b(must|allowed|not allowed|permitted|prohibited|banned|penalt\w*|punish\w*|rules?|polic(y|ies)|laws?|"
    r"sections?|act|committees?|helpline|according|circular|regulations?|guidelines?|deadline|fees?|"
    r"unfair means|ugc|aicte|mandatory|compulsory|illegal)\b", re.I)


def is_factual(sentence: str, topic_phrase: str = "") -> bool:
    """Heuristic: normative / numeric statements are claims; questions are not. Naming the topic itself
    ("these AI policy facts") is not a claim, so the topic phrase is removed before matching."""
    s = (sentence or "").strip()
    if topic_phrase:
        s = re.sub(re.escape(topic_phrase), " ", s, flags=re.I)
    return bool(s.strip()) and not s.rstrip().endswith("?") and bool(FACTUAL_HINT.search(s))


def json_strings(obj: Any) -> str:
    """All string values of a JSON object joined by newlines (scene.json text for 'declared' checks)."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return "\n".join(json_strings(v) for v in obj.values())
    if isinstance(obj, list):
        return "\n".join(json_strings(v) for v in obj)
    return ""


def cites_whitelisted_contact(sentence: str, contacts: list[dict] | None) -> bool:
    """True if the sentence contains a whitelisted contact (itself extracted verbatim from an official source)."""
    low = (sentence or "").lower()
    digits = re.sub(r"\D", "", sentence or "")
    for c in contacts or []:
        if not c.get("source_ids"):
            continue
        kind, val = c.get("kind"), str(c.get("value") or c.get("display") or "").lower()
        if kind == "url":
            host = re.sub(r"^https?://", "", val).split("/")[0].removeprefix("www.")
            if host and host in low:
                return True
        elif kind == "email" and val and val in low:
            return True
        elif kind == "phone":
            d = re.sub(r"\D", "", val)
            if len(d) >= 3 and (d in digits or d.removeprefix("91") in digits):
                return True
    return False


def check_facts(script: dict, facts: list[dict], storyboard: dict | None = None,
                contacts: list[dict] | None = None, topic_phrase: str = "") -> list[QACheck]:
    """Every factual sentence carries fact_ids (or cites a whitelisted contact); all ids exist; every cited
    fact has a quote + url."""
    from core.schemas import Script

    sents = Script.model_validate(script).sentences()
    by_id = {f["fact_id"]: f for f in facts}
    untagged, unknown, noquote = [], [], []
    for i, s in enumerate(sents):
        if not s.fact_ids and is_factual(s.display, topic_phrase) and \
                not cites_whitelisted_contact(s.display, contacts):
            untagged.append(f"s{i}: '{s.display[:70]}'")
        for fid in s.fact_ids:
            if fid not in by_id:
                unknown.append(f"s{i}:{fid}")
            elif not str(by_id[fid].get("quote", "")).strip() or not by_id[fid].get("url"):
                noquote.append(fid)
    for sh in (storyboard or {}).get("shots", []):
        unknown += [f"shot{sh.get('shot_id')}:{fid}" for fid in sh.get("fact_ids", []) if fid not in by_id]
    res = [
        qc("facts_tagged", not untagged, "scriptwriter", value=f"{len(untagged)} untagged factual sentence(s)",
           threshold="100% tagged", evidence="; ".join(untagged[:6]) + (f" | allowed fact ids: {sorted(by_id)[:30]}" if untagged else "")),
        qc("facts_known_ids", not unknown, "scriptwriter", value=len(unknown), threshold="0 unknown ids",
           evidence="; ".join(unknown[:10]) + (f" | allowed: {sorted(by_id)[:30]}" if unknown else "")),
        qc("facts_quotes", not noquote, "researcher", value=len(noquote), threshold="every cited fact has quote+url",
           evidence=", ".join(sorted(set(noquote))[:10])),
    ]
    if not facts and any(is_factual(s.display) for s in sents):
        res.append(qc("facts_evidence", False, "researcher", value="0 facts", threshold=">= 1 sourced fact",
                      evidence="script makes factual statements but no facts were curated"))
    return res


def check_must_include(brief: dict, coverage: list[dict], script: dict) -> list[QACheck]:
    """Covered items must be cited (fact_ids) or mentioned lexically; MISSING items must be softened."""
    from core.schemas import Script

    sents = Script.model_validate(script).sentences()
    text = " ".join(s.display + " " + s.spoken for s in sents)
    toks = set(_tokens(text))
    cited = {f for s in sents for f in s.fact_ids}
    cov = {c.get("item", ""): c for c in coverage}
    res = []
    for item in brief.get("must_include", []):
        c = cov.get(item, {})
        kt = [t for t in _tokens(item) if t not in STOP]
        overlap = sum(t in toks for t in kt) / max(1, len(kt))
        if c.get("status") == "MISSING":
            soft = bool(re.search(r"official|check|ask|confirm", text, re.I))
            res.append(qc(f"must_include:{item[:40]}", soft, "scriptwriter", value="MISSING in sources",
                          threshold="softened ('check the official page')", severity="soft",
                          evidence="coverage MISSING; do not invent it; point viewers to the official page"))
            continue
        hit = bool(set(c.get("fact_ids", [])) & cited) or overlap >= 0.5
        res.append(qc(f"must_include:{item[:40]}", hit, "scriptwriter", value=f"lexical {overlap:.2f}",
                      threshold="cited fact or >=50% keyword overlap",
                      evidence=f"item '{item}' facts {c.get('fact_ids', [])} not cited / not mentioned" if not hit else ""))
    return res


def check_must_avoid(brief: dict, script: dict, storyboard: dict | None = None) -> list[QACheck]:
    """Short must_avoid phrases (≤4 words) must not appear lexically; longer ones go to the LLM soft review."""
    from core.schemas import Script

    sents = Script.model_validate(script).sentences()
    shots = (storyboard or {}).get("shots", [])
    text = norm_text(" ".join(s.display + " " + s.spoken for s in sents) + " "
                     + " ".join(t for sh in shots for t in sh.get("on_screen_text", []))
                     + " " + " ".join(sh.get("visual_prompt", "") for sh in shots))
    res = []
    for item in brief.get("must_avoid", []):
        n = norm_text(item)
        if len(n.split()) > 4:
            res.append(qc(f"must_avoid:{item[:40]}", True, "scriptwriter", value="delegated",
                          threshold="absent", evidence="semantic item: checked by LLM soft review", severity="soft"))
            continue
        found = re.search(rf"\b{re.escape(n)}\b", text) is not None
        res.append(qc(f"must_avoid:{item[:40]}", not found, "scriptwriter", value="present" if found else "absent",
                      threshold="absent", evidence=f"'{item}' found in script/storyboard text" if found else ""))
    return res


def check_script_basics(script: dict, plan: dict, storyboard: dict | None = None) -> list[QACheck]:
    """Word budget, CTA present (Scriptwriter); ≤7 words per on-screen slot (Storyboarder)."""
    wc = int(script.get("word_count", 0))
    lo, hi = int(plan.get("word_budget_min", 0)), int(plan.get("word_budget_max", 10_000))
    cta = (script.get("cta") or {}).get("display", "").strip()
    wordy = [f"shot{sh.get('shot_id')}: '{t}'" for sh in (storyboard or {}).get("shots", [])
             for t in sh.get("on_screen_text", []) if len(t.split()) > ONSCREEN_MAX_WORDS]
    return [
        qc("word_budget", lo <= wc <= hi, "scriptwriter", value=wc, threshold=f"[{lo},{hi}]", severity="soft",
           evidence="" if lo <= wc <= hi else f"script has {wc} words; target {lo}-{hi}"),
        qc("cta_present", bool(cta), "scriptwriter", value=cta[:60] or "missing", threshold="non-empty CTA"),
        qc("onscreen_words", not wordy, "storyboarder", value=len(wordy), threshold=f"<= {ONSCREEN_MAX_WORDS} words/slot",
           evidence="; ".join(wordy[:6])),
    ]


# ================================================================ contacts
PHONE_LONG = re.compile(r"(?<![\w/])\+?\d[\d\s\-()]{6,}\d(?![\w/])")
PHONE_SHORT = re.compile(r"\b(?:call|dial|helpline|phone|tel|toll[- ]free|number)\b[:\s]*(?:no\.?\s*)?(\d{3,5})\b(?![\s\-]*\d)", re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL = re.compile(r"(?:https?://|www\.)[^\s,;)\"']+|\b[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.(?:gov\.in|nic\.in|edu\.in|ac\.in|org\.in|co\.in|in|edu|org|com|net)\b(?:/[^\s,;)\"']*)?", re.I)
YEAR = re.compile(r"^(19|20)\d{2}$")
YEAR_RANGE = re.compile(r"^(19|20)\d{2}\s*[-–]\s*((19|20)?\d{2})$")


def norm_phone(s: str) -> str:
    d = re.sub(r"\D", "", s)
    if len(d) > 10 and d.startswith("91"):
        d = d[2:]
    return d.lstrip("0") or d


def norm_email(s: str) -> str:
    return s.strip().lower().rstrip(".")


def norm_url(s: str) -> str:
    s = s.strip().lower().rstrip(".,;:/")
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    return s.rstrip("/")


def find_contacts(text: str) -> list[tuple[str, str, str]]:
    """[(kind, display, normalised)] for phones (long numbers or 'call 1930'), emails, URLs/domains."""
    found: list[tuple[str, str, str]] = []
    emails = EMAIL.findall(text)
    for e in emails:
        found.append(("email", e, norm_email(e)))
    stripped = EMAIL.sub(" ", text)
    for u in URL.findall(stripped):
        found.append(("url", u, norm_url(u)))
    no_url = URL.sub(" ", stripped)
    for p in PHONE_LONG.findall(no_url):
        if len(re.sub(r"\D", "", p)) >= 7 and not YEAR_RANGE.match(p.strip()):
            found.append(("phone", p.strip(), norm_phone(p)))
    for p in PHONE_SHORT.findall(PHONE_LONG.sub(" ", no_url)):
        if not YEAR.match(p):
            found.append(("phone", p, norm_phone(p)))
    seen, out = set(), []
    for k in found:
        if (k[0], k[2]) not in seen:
            seen.add((k[0], k[2]))
            out.append(k)
    return out


def _whitelist_values(contacts: list[dict]) -> dict[str, set[str]]:
    wl: dict[str, set[str]] = {"phone": set(), "email": set(), "url": set()}
    for c in contacts:
        k = c.get("kind")
        v = c.get("value", "") or c.get("display", "")
        if k == "phone":
            wl["phone"].add(norm_phone(v))
            wl["phone"].add(norm_phone(c.get("display", v)))
        elif k == "email":
            wl["email"].add(norm_email(v))
        elif k == "url":
            wl["url"].add(norm_url(v))
    return wl


def contact_allowed(kind: str, value: str, wl: dict[str, set[str]]) -> bool:
    """Exact normalised match; a displayed URL may be a shorter form (prefix) of a whitelisted one."""
    if kind == "url":
        return any(value == w or w.startswith(value + "/") for w in wl["url"])
    return value in wl.get(kind, set())


def check_contacts(texts: dict[str, str], contacts: list[dict], source_urls: Iterable[str] = ()) -> QACheck:
    """Every phone/email/URL in script display + OCR text must be in the whitelist (normalised).
    URLs of snapshotted official sources (`source_urls`) also count as verified URLs."""
    wl = _whitelist_values(contacts)
    wl["url"] |= {norm_url(u) for u in source_urls if u}
    bad, seen = [], 0
    for src, text in texts.items():
        for kind, disp, val in find_contacts(text or ""):
            seen += 1
            if not contact_allowed(kind, val, wl):
                bad.append(f"{src}: {kind} '{disp}'")
    allowed = sorted(c.get("display", c.get("value", "")) for c in contacts)
    return qc("contacts_whitelist", not bad, "scriptwriter", value=f"{len(bad)}/{seen} not whitelisted",
              threshold="100% in whitelist", evidence="; ".join(bad[:8]) + (f" | whitelist: {allowed[:12]}" if bad else ""))


# ================================================================ assets (AssetGenerator)
def check_shot_images(shot_assets: dict, nsfw_max: float = 0.2) -> list[QACheck]:
    """Per shot: no faces unless the VLM cleared them (vlm_ok), NSFW < threshold."""
    res = []
    for sid, a in sorted((shot_assets or {}).items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
        if not a.get("image_path"):
            continue
        fc, ok = int(a.get("face_count", 0)), bool(a.get("vlm_ok", True))
        res.append(qc(f"faces:{sid}", fc == 0 or ok, "asset_generator", value=fc, threshold="0 realistic faces",
                      evidence=f"YuNet faces={fc}; vlm_ok={ok} {a.get('vlm_note', '')}".strip()))
        ns = float(a.get("nsfw_score", 0.0))
        res.append(qc(f"nsfw:{sid}", ns < nsfw_max, "asset_generator", value=f"{ns:.3f}", threshold=f"< {nsfw_max}",
                      evidence=f"image {Path(a.get('image_path', '')).name}"))
    return res


def check_wer(voice: dict, wer_max: float = 0.08) -> QACheck:
    if not voice:
        return qc("voice_wer", False, "asset_generator", value="no voice", threshold=f"<= {wer_max}", evidence="voice asset missing")
    wer = float(voice.get("wer", 0.0))
    ev = f"ASR: '{str(voice.get('asr_text', ''))[:160]}'" if voice.get("asr_text") else "no ASR transcript (mock?)"
    return qc("voice_wer", wer <= wer_max, "asset_generator", value=f"{wer:.3f}", threshold=f"<= {wer_max}", evidence=ev)


def read_manifest_rows(path: str | Path) -> list[dict]:
    import csv

    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_licence_allowlist(root: Path) -> set[str]:
    p = root / "config" / "licences_allowlist.yaml"
    if not p.exists():
        return set(DEFAULT_LICENCES)
    import yaml

    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    items = data if isinstance(data, list) else (data.get("allowed") or data.get("licences") or data.get("allow") or [])
    vals = set()
    for it in items:
        vals.add(str(it.get("id", it.get("name", "")) if isinstance(it, dict) else it).strip().lower())
    return vals or set(DEFAULT_LICENCES)


def check_licences(rows: list[dict], allowlist: set[str] | Callable[[str], bool]) -> QACheck:
    """Every manifest row's licence is allowed (set of lowercase ids, or a predicate like core.manifest.licence_allowed)."""
    ok = allowlist if callable(allowlist) else (lambda lic: str(lic).strip().lower() in allowlist)
    bad = [f"{Path(r.get('path', '')).name}: '{r.get('licence', '')}'" for r in rows if not ok(str(r.get("licence", "")))]
    return qc("licences", not bad and bool(rows), "asset_generator", value=f"{len(bad)}/{len(rows)} not allowed",
              threshold="100% in allowlist", evidence="; ".join(bad[:8]) if bad else ("" if rows else "manifest empty/missing"),
              severity="hard" if rows else "soft")


# ================================================================ rule-pack code checks
RANKING = re.compile(r"\b(rank(ed|ing)?|nirf|naac|placement|package|lpa|award(ed|s)?|top \d+)\b", re.I)


def check_pack_rules(rule_ids: Iterable[str], script: dict, storyboard: dict, facts: list[dict]) -> list[QACheck]:
    """Category `check: code` rules not covered by the base checks (unknown ids are skipped)."""
    from core.schemas import Script

    ids = set(rule_ids)
    shots = (storyboard or {}).get("shots", [])
    sents = Script.model_validate(script).sentences() if script else []
    by_id = {f["fact_id"]: f for f in facts}
    res: list[QACheck] = []
    if "A.helpline_card" in ids:
        info = [s for s in shots if s.get("template") != "end_card"]
        ok = bool(info) and info[-1].get("template") == "contact_card"
        res.append(qc("rule:A.helpline_card", ok, "storyboarder", value=info[-1].get("template") if info else "none",
                      threshold="contact_card right before end_card", evidence="last informational shot"))
    if "B.notice_url" in ids:
        txt = [t for s in shots for t in s.get("on_screen_text", [])]
        has_num = any(re.search(r"\d", t) for t in txt)
        has_url = any(URL.search(t) for t in txt)
        res.append(qc("rule:B.notice_url", not has_num or has_url, "storyboarder", value=f"numbers={has_num} url={has_url}",
                      threshold="notice URL on screen when a number is shown"))
    if "C.allowed_items_sourced" in ids:
        bad = [str(s.get("shot_id")) for s in shots if s.get("template") == "do_dont_split" and not s.get("fact_ids")]
        res.append(qc("rule:C.allowed_items_sourced", not bad, "storyboarder", value=len(bad), threshold="fact_ids on do_dont_split",
                      evidence=f"shots without fact_ids: {bad}" if bad else ""))
    if "E.no_unquoted_rankings" in ids:
        bad = []
        for i, s in enumerate(sents):
            if RANKING.search(s.display):
                quotes = " ".join(by_id.get(f, {}).get("quote", "") for f in s.fact_ids)
                if not s.fact_ids or not re.search(r"\b(19|20)\d{2}\b", quotes):
                    bad.append(f"s{i}: '{s.display[:60]}'")
        res.append(qc("rule:E.no_unquoted_rankings", not bad, "scriptwriter", value=len(bad),
                      threshold="ranking claims quoted with year", evidence="; ".join(bad[:5])))
    if "P.real_activity" in ids:
        ok = any(s.fact_ids for s in sents[1:-1]) if len(sents) > 2 else False
        res.append(qc("rule:P.real_activity", ok, "scriptwriter", value=ok, threshold=">= 1 sourced body sentence"))
    return res
