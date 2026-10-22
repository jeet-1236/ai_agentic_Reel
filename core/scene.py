"""scene.json: the contract between the Editor and both renderers (HyperFrames + ffmpeg fallback).

All times are seconds on the final timeline (voice starts at 0). All pixel values are in output pixels.
Paths are repo-relative POSIX strings when the file lives under the repo, absolute otherwise; renderers
resolve them with `resolve_path()` (Python) or relative to the repo root (renderer/render_hyperframes.mjs).

Per-template slot schemas, durations, aspect support and text boxes live in
`renderer/templates/<name>/template.yaml` and are shared by both renderers.
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from core.context import ROOT
from core.seeds import stage_seed

SCENE_VERSION = 1
TEMPLATES_DIR = ROOT / "renderer" / "templates"
FONTS_DIR = ROOT / "assets" / "fonts"
PLACEHOLDER_LOGO = ROOT / "assets" / "official" / "logo_placeholder.svg"
BRAND_COLORS = {"power_red": "#B7202E", "vitality_red": "#ED1C24", "cool_gray": "#58595B", "dark_gray": "#231F20"}
CAPTION_STYLE = {"font_px": 64, "outline_px": 6, "line_height": 1.25, "weight": 700, "color": "#FFFFFF",
                 "outline_color": "#110F10", "highlight": "#B7202E"}


def disclosure_line() -> str:
    """Single source: config/settings.yaml `disclosure_line`."""
    try:
        s = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        s = {}
    return s.get("disclosure_line") or "Created with AI agents — LY Agentic AI Lab"


class Box(BaseModel):
    """Axis-aligned rectangle in output pixels (x1/y1 exclusive)."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def contains(self, b: "Box | dict", tol: int = 0) -> bool:
        d = b if isinstance(b, dict) else b.model_dump()
        return (d["x0"] >= self.x0 - tol and d["y0"] >= self.y0 - tol
                and d["x1"] <= self.x1 + tol and d["y1"] <= self.y1 + tol)


class SafeZones(BaseModel):
    """`text`: every text pixel must be inside (critic). `content`: template text while captions may show.
    `caption`: caption band. `end`: end-card text area (no captions there)."""
    text: Box
    content: Box
    caption: Box
    end: Box


def safe_zones(width: int, height: int) -> SafeZones:
    """Safe zones for 1080x1920 (handout band y in [250,1570], captions y in [1150,1500]) or 1920x1080."""
    if height > width:  # 9:16
        sx, sy = width / 1080, height / 1920
        b = lambda x0, y0, x1, y1: Box(x0=round(x0 * sx), y0=round(y0 * sy), x1=round(x1 * sx), y1=round(y1 * sy))  # noqa: E731
        return SafeZones(text=b(60, 250, 1020, 1570), content=b(90, 270, 990, 1110),
                         caption=b(90, 1150, 990, 1500), end=b(90, 270, 990, 1550))
    sx, sy = width / 1920, height / 1080
    b = lambda x0, y0, x1, y1: Box(x0=round(x0 * sx), y0=round(y0 * sy), x1=round(x1 * sx), y1=round(y1 * sy))  # noqa: E731
    return SafeZones(text=b(96, 54, 1824, 1026), content=b(160, 80, 1760, 770),
                     caption=b(160, 800, 1760, 1000), end=b(160, 80, 1760, 1000))


class KenBurns(BaseModel):
    """Zoom/pan of the background image over the shot. Pan values in [-1, 1] = fraction of the headroom."""
    zoom_from: float = 1.0
    zoom_to: float = 1.1
    pan_x_from: float = 0.0
    pan_x_to: float = 0.0
    pan_y_from: float = 0.0
    pan_y_to: float = 0.0


def kenburns_for(seed: int, shot_id: int) -> KenBurns:
    """Seeded (never random-random) Ken Burns move: zoom in or out with a gentle diagonal pan."""
    rng = random.Random(stage_seed(seed, "kenburns", shot_id))
    z_lo, z_hi = 1.0, round(rng.uniform(1.08, 1.16), 3)
    zoom_in = rng.random() < 0.6
    dx, dy = rng.choice([-1, 1]) * round(rng.uniform(0.3, 0.8), 3), rng.choice([-1, 1]) * round(rng.uniform(0.2, 0.6), 3)
    return KenBurns(zoom_from=z_lo if zoom_in else z_hi, zoom_to=z_hi if zoom_in else z_lo,
                    pan_x_from=-dx / 2, pan_x_to=dx / 2, pan_y_from=-dy / 2, pan_y_to=dy / 2)


class SceneWord(BaseModel):
    text: str
    start: float
    end: float


class SceneCaption(BaseModel):
    """One caption page: <= 2 lines; each line is a list of timed words (active-word highlight).
    `px` is the fitted font size (64 by default, shrunk to >= 48 when a line is too wide for the band)."""
    index: int
    start: float
    end: float
    lines: list[list[SceneWord]]
    px: int = 64

    def text_lines(self) -> list[str]:
        return [" ".join(w.text for w in line) for line in self.lines]


class SceneShot(BaseModel):
    shot_id: int
    template: str
    start: float
    end: float
    image: str = ""                       # background image (repo-relative or absolute); "" = brand gradient
    slots: dict[str, Any] = {}            # validated against template.yaml slots_schema
    kenburns: KenBurns | None = None
    zone: Literal["content", "end"] = "content"
    # Fitted text, filled by `fit_scene_text()`: mirrors `slots`, each string replaced by
    # {"px", "lh", "weight", "lines": [...], "overflow"}; renderers draw exactly these lines.
    text: dict[str, Any] = {}
    sentence_indices: list[int] = []
    fact_ids: list[str] = []

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


class SceneAudio(BaseModel):
    voice_path: str
    duration_s: float
    volume: float = 1.0


class SceneBrand(BaseModel):
    logo_path: str
    disclosure: str = Field(default_factory=disclosure_line)
    colors: dict[str, str] = Field(default_factory=lambda: dict(BRAND_COLORS))
    font_family: str = "Fira Sans"


class Scene(BaseModel):
    version: int = SCENE_VERSION
    run_id: str
    brief_id: str
    seed: int
    aspect: Literal["9x16", "16x9"]
    width: int
    height: int
    fps: int = 30
    duration_s: float
    safe: SafeZones
    shots: list[SceneShot]
    captions: list[SceneCaption] = []
    caption_style: dict[str, Any] = Field(default_factory=lambda: dict(CAPTION_STYLE))
    audio: SceneAudio | None = None
    brand: SceneBrand
    fonts_dir: str = "assets/fonts"

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @property
    def total_frames(self) -> int:
        return round(self.duration_s * self.fps)

    # ------------------------------------------------------------------ checks
    def validate_scene(self, check_files: bool = True) -> list[str]:
        """Hard problems (empty = renderable): timing gaps, aspect, slots vs schema, missing files."""
        return [e for e in self._problems(check_files) if not e.startswith("warn:")]

    def warnings(self) -> list[str]:
        """Soft problems: shot durations outside the template's [min, max]."""
        return [e[5:].strip() for e in self._problems(False) if e.startswith("warn:")]

    def _problems(self, check_files: bool) -> list[str]:
        import jsonschema

        errs: list[str] = []
        prev_end = 0.0
        for s in self.shots:
            try:
                spec = load_template_spec(s.template)
            except FileNotFoundError:
                errs.append(f"shot {s.shot_id}: unknown template {s.template}")
                continue
            if abs(s.start - prev_end) > 1e-3:
                errs.append(f"shot {s.shot_id}: starts at {s.start}, previous ends at {prev_end} (gap/overlap)")
            prev_end = s.end
            if s.duration < spec.get("min_duration_s", 0) - 1e-3 or s.duration > spec.get("max_duration_s", 1e9) + 1e-3:
                errs.append(f"warn: shot {s.shot_id} ({s.template}): duration {s.duration}s outside "
                            f"[{spec.get('min_duration_s')}, {spec.get('max_duration_s')}]")
            if self.aspect not in spec.get("aspects", ["9x16", "16x9"]):
                errs.append(f"shot {s.shot_id}: template {s.template} does not support {self.aspect}")
            for e in jsonschema.Draft202012Validator(spec["slots_schema"]).iter_errors(s.slots):
                errs.append(f"shot {s.shot_id} ({s.template}) slots: {e.message} at {list(e.absolute_path)}")
            if spec.get("image") == "required" and not s.image:
                errs.append(f"shot {s.shot_id} ({s.template}): image required")
            if check_files and s.image and not resolve_path(s.image).is_file():
                errs.append(f"shot {s.shot_id}: image missing {s.image}")
        if self.shots and abs(prev_end - self.duration_s) > 1e-3:
            errs.append(f"last shot ends at {prev_end}, scene duration is {self.duration_s}")
        for c in self.captions:
            if len(c.lines) > 2:
                errs.append(f"caption {c.index}: {len(c.lines)} lines > 2")
        if check_files:
            if self.audio and not resolve_path(self.audio.voice_path).is_file():
                errs.append(f"voice missing {self.audio.voice_path}")
            if not resolve_path(self.brand.logo_path).is_file():
                errs.append(f"logo missing {self.brand.logo_path}")
        return errs


# ---------------------------------------------------------------- helpers
def rel_path(p: str | Path) -> str:
    """Repo-relative POSIX path when under ROOT, else absolute POSIX path."""
    if not p:
        return ""
    pp = Path(p).resolve()
    try:
        return pp.relative_to(ROOT).as_posix()
    except ValueError:
        return pp.as_posix()


def resolve_path(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else ROOT / pp


@lru_cache(maxsize=None)
def load_template_spec(name: str) -> dict:
    """renderer/templates/<name>/template.yaml as a dict."""
    path = TEMPLATES_DIR / name / "template.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def template_names() -> list[str]:
    return sorted(p.parent.name for p in TEMPLATES_DIR.glob("*/template.yaml"))


# ---------------------------------------------------------------- text fitting (shared by both renderers)
FONT_FILES = {400: "FiraSans-Regular.ttf", 700: "FiraSans-Bold.ttf", 800: "FiraSans-ExtraBold.ttf"}
WIDTH_SAFETY = 0.97   # Pillow vs Chrome advance widths differ by < 1-2 %


@lru_cache(maxsize=512)
def _font(weight: int, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(str(FONTS_DIR / FONT_FILES.get(weight, FONT_FILES[700])), px)


def text_width(text: str, weight: int, px: int) -> float:
    return float(_font(weight, px).getlength(text))


def wrap_words(text: str, weight: int, px: int, max_w: float) -> list[str]:
    """Greedy word wrap; a single word wider than max_w gets its own (overflowing) line."""
    import re

    lines: list[str] = []
    cur = ""
    tokens: list[tuple[str, str]] = []          # (piece, joiner before it)
    for word in text.split():
        if text_width(word, weight, px) > max_w and re.search(r"[/\-.]", word[:-1]):
            parts = re.findall(r"[^/\-.]+[/\-.]?|[/\-.]", word)   # URLs/long words break after / - .
            tokens += [(p, " " if i == 0 else "") for i, p in enumerate(parts)]
        else:
            tokens.append((word, " "))
    for word, joiner in tokens:
        cand = f"{cur}{joiner}{word}" if cur else word
        if not cur or text_width(cand, weight, px) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def balance_lines(text: str, weight: int, px: int, max_w: float, n: int) -> list[str]:
    """Same line count as the greedy wrap, but the narrowest width that keeps it (no orphan last words)."""
    best = wrap_words(text, weight, px, max_w)
    if n < 2:
        return best
    lo, hi = max_w * 0.3, max_w
    for _ in range(14):
        mid = (lo + hi) / 2
        cand = wrap_words(text, weight, px, mid)
        if len(cand) <= n and max(text_width(ln, weight, px) for ln in cand) <= max_w:
            best, hi = cand, mid
        else:
            lo = mid
    return best


def fit_block(texts: list[str], weight: int, line_height: float, w: int, h: int, max_px: int, min_px: int,
              gap: int = 0, max_lines: int | None = None, pad_y: int = 0) -> list[dict]:
    """Largest common font size (step 2 px) so all `texts` stacked with `gap` fit in w x h."""
    sizes = list(range(max_px, min_px - 1, -2))
    if not sizes or sizes[-1] != min_px:
        sizes.append(min_px)
    maxw = w * WIDTH_SAFETY
    for px in sizes:
        wrapped = [wrap_words(t, weight, px, maxw) for t in texts]
        total = sum(len(ls) * px * line_height + pad_y for ls in wrapped) + gap * max(0, len(texts) - 1)
        widest = max((text_width(ln, weight, px) for ls in wrapped for ln in ls), default=0)
        if total <= h and widest <= maxw and (not max_lines or all(len(ls) <= max_lines for ls in wrapped)):
            wrapped = [balance_lines(t, weight, px, maxw, len(ls)) for t, ls in zip(texts, wrapped)]
            return [{"px": px, "lh": line_height, "weight": weight, "lines": ls, "overflow": False} for ls in wrapped]
    wrapped = [wrap_words(t, weight, min_px, maxw) for t in texts]
    return [{"px": min_px, "lh": line_height, "weight": weight, "lines": ls, "overflow": True} for ls in wrapped]


def _transform(text: str, spec: dict) -> str:
    text = " ".join(str(text).split())
    if spec.get("upper"):
        text = text.upper()
    if spec.get("wrap_quotes"):
        text = f"“{text.strip(chr(34) + chr(0x201c) + chr(0x201d))}”"
    return f"{spec.get('prefix', '')}{text}"


def fit_shot_text(template: str, slots: dict, aspect: str) -> dict:
    """Fit every text slot of one shot per template.yaml `text` specs -> SceneShot.text."""
    spec = load_template_spec(template)
    slots = {**(spec.get("defaults") or {}), **slots}
    out: dict[str, Any] = {}
    for key, ts in (spec.get("text") or {}).items():
        box = ts[aspect]
        args = dict(weight=ts.get("weight", 700), line_height=ts.get("line_height", 1.15), w=box["w"], h=box["h"],
                    max_px=box["max_px"], min_px=box["min_px"], gap=ts.get("gap", 0),
                    max_lines=ts.get("max_lines"), pad_y=ts.get("pad_y", 0))
        if "." in key:                                   # field of a list of objects, e.g. contacts.label
            parent, field = key.split(".", 1)
            items = slots.get(parent) or []
            dst = out.setdefault(parent, [{} for _ in items])
            for i, it in enumerate(items):
                if it.get(field):
                    dst[i][field] = fit_block([_transform(it[field], ts)], **args)[0]
            continue
        val = slots.get(key)
        if not val:
            continue
        if isinstance(val, list):
            texts = [_transform(v, ts) for v in val]
            if ts.get("per_item"):
                out[key] = [fit_block([t], **args)[0] for t in texts]
            else:
                out[key] = fit_block(texts, **args)
        else:
            out[key] = fit_block([_transform(val, ts)], **args)[0]
    if "logo_box" in spec:
        out["logo_box"] = spec["logo_box"][aspect]
    return out


def fit_caption_px(lines: list[str], max_w: int, max_px: int = 64, min_px: int = 48) -> int:
    """Caption font size: 64 px unless a line is wider than the band (then shrink, never below 48)."""
    for px in range(max_px, min_px - 1, -2):
        if all(text_width(ln, 700, px) + 2 * CAPTION_STYLE["outline_px"] <= max_w * WIDTH_SAFETY for ln in lines):
            return px
    return min_px


def fit_scene_text(scene: "Scene") -> "Scene":
    """Fill shot.text / shot.zone and caption.px for every shot/page (idempotent)."""
    for s in scene.shots:
        spec = load_template_spec(s.template)
        s.zone = spec.get("zone", "content")
        s.text = fit_shot_text(s.template, s.slots, scene.aspect)
    for c in scene.captions:
        c.px = fit_caption_px(c.text_lines(), scene.safe.caption.w, scene.caption_style.get("font_px", 64))
    return scene
