"""Fallback renderer: scene.json -> Pillow-drawn template cards + ffmpeg zoompan Ken Burns + caption overlay -> MP4.

    python workers/render_ffmpeg.py --in req.json --out resp.json
    req:  {scene_path, out_path, layout_path?, mock?, work_dir?, keep_work?}
    outputs: {video_path, frames, layout_path, render_engine}

Reads the same scene.json as renderer/render_hyperframes.mjs and draws the same pre-fitted text lines
(scene.shots[].text, fitted by core/scene.py), so both engines produce the same wording and similar layout.
Pipeline: per shot one segment (background: zoompan on the cover-cropped image or the brand gradient;
overlay: RGBA PNG with the template's text, faded in) -> concat -> one caption overlay stream (concat demuxer
of band-sized RGBA PNGs, one per active-word state) + voice track. Emits layout.json with every text box.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from core.ffmpeg_utils import ffmpeg_bin, run  # noqa: E402
from core.scene import Box, Scene, SceneCaption, SceneShot, _font, resolve_path, text_width  # noqa: E402

WHITE, DARK, RED, GREEN = (255, 255, 255, 255), (35, 31, 32, 255), (183, 32, 46, 255), (46, 158, 91, 255)
CARD = (24, 21, 22, 204)
UPSCALE = 2          # zoompan works on a 2x image -> sub-pixel smooth enough motion


def _hex(c: tuple | str) -> str:
    return c.upper() if isinstance(c, str) else "#{:02X}{:02X}{:02X}".format(*c[:3])


# ---------------------------------------------------------------- drawing primitives
class Painter:
    """Draws fitted text blocks onto an RGBA overlay and records their boxes (layout.json)."""

    def __init__(self, w: int, h: int):
        self.img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.boxes: list[dict] = []
        self.logos: list[dict] = []

    @staticmethod
    def block_h(t: dict) -> int:
        return round(len(t["lines"]) * t["px"] * t["lh"])

    @staticmethod
    def block_w(t: dict) -> int:
        return round(max((text_width(ln, t["weight"], t["px"]) for ln in t["lines"]), default=0))

    def rect(self, box: tuple[int, int, int, int], fill, radius: int = 0) -> None:
        if fill[3] < 255:   # alpha-composite translucent cards
            layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).rounded_rectangle(box, radius=radius, fill=fill)
            self.img.alpha_composite(layer)
        else:
            self.d.rounded_rectangle(box, radius=radius, fill=fill)

    def text(self, t: dict, x: int, y: int, *, align: str = "center", width: int = 0, color=WHITE, slot: str = "",
             shadow: bool = False) -> dict:
        """Draw lines of fitted block `t`; (x, y) = top-left of the row box of `width` (align inside it)."""
        px, lh_px = t["px"], t["px"] * t["lh"]
        font = _font(t["weight"], px)
        bw, bh = self.block_w(t), self.block_h(t)
        bx = x + (width - bw) // 2 if align == "center" and width else x
        if shadow:
            sh = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(sh)
            for i, ln in enumerate(t["lines"]):
                lw = text_width(ln, t["weight"], px)
                lx = x + (width - lw) / 2 if align == "center" and width else x
                sd.text((lx, y + 3 + i * lh_px + lh_px / 2), ln, font=font, fill=(0, 0, 0, 140), anchor="lm")
            self.img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(7)))
        for i, ln in enumerate(t["lines"]):
            lw = text_width(ln, t["weight"], px)
            lx = x + (width - lw) / 2 if align == "center" and width else x
            self.d.text((lx, y + i * lh_px + lh_px / 2), ln, font=font, fill=color, anchor="lm")
        b = {"kind": "text", "slot": slot, "text": " ".join(t["lines"]), "font_px": px, "line_height_px": round(lh_px),
             "lines": len(t["lines"]), "color": _hex(color), "stroke_color": "", "stroke_px": 0,
             "x0": int(bx), "y0": int(y), "x1": int(bx + bw), "y1": int(y + bh)}
        self.boxes.append(b)
        return b


def _stack_top(zone: Box, heights: list[int], gaps: list[int], valign: str = "center") -> int:
    total = sum(heights) + sum(gaps)
    return zone.y0 if valign == "top" else zone.y0 + max(0, (zone.h - total) // 2)


# ---------------------------------------------------------------- templates (mirror renderer/templates/*)
def paint_hook_title(p: Painter, s: SceneShot, zone: Box, aspect: str) -> None:
    t = s.text
    parts = [k for k in ("kicker", "title", "subtitle") if k in t]
    hs = {k: p.block_h(t[k]) for k in parts}
    heights = [hs.get("kicker", 0) + (18 if "kicker" in t else 0), hs["title"], 12 + 56, hs.get("subtitle", 0)]
    y = _stack_top(zone, heights, [0, 0, 0])
    if "kicker" in t:
        kw = p.block_w(t["kicker"])
        x0 = zone.x0 + (zone.w - kw) // 2
        p.rect((x0 - 18, y, x0 + kw + 18, y + hs["kicker"]), WHITE, 8)
        p.text(t["kicker"], zone.x0, y, width=zone.w, color=RED, slot="kicker")
        y += hs["kicker"] + 18
    p.text(t["title"], zone.x0, y, width=zone.w, slot="title", shadow=True)
    y += hs["title"] + 28
    p.rect((zone.x0 + zone.w // 2 - 90, y, zone.x0 + zone.w // 2 + 90, y + 12), RED, 6)
    y += 12 + 28
    if "subtitle" in t:
        p.text(t["subtitle"], zone.x0, y, width=zone.w, slot="subtitle", shadow=True)


def paint_illustration_kenburns(p: Painter, s: SceneShot, zone: Box, aspect: str) -> None:
    t = s.text.get("headline")
    if not t:
        return
    h = p.block_h(t)
    p.rect((zone.x0, zone.y0, zone.x1, zone.y0 + h + 52), CARD, 18)
    p.rect((zone.x0, zone.y0, zone.x0 + 12, zone.y0 + h + 52), RED, 0)
    p.text(t, zone.x0 + 42, zone.y0 + 26, width=zone.w - 72, slot="headline")


def _icon(p: Painter, x: int, y: int, ok: bool) -> None:
    p.d.ellipse((x, y, x + 64, y + 64), fill=GREEN if ok else RED)
    if ok:
        p.d.line([(x + 16, y + 34), (x + 28, y + 46), (x + 49, y + 20)], fill=WHITE, width=8, joint="curve")
    else:
        p.d.line([(x + 19, y + 19), (x + 45, y + 45)], fill=WHITE, width=8)
        p.d.line([(x + 45, y + 19), (x + 19, y + 45)], fill=WHITE, width=8)


def paint_do_dont_split(p: Painter, s: SceneShot, zone: Box, aspect: str) -> None:
    t = s.text
    gap = 36
    if aspect == "16x9":
        pw = (zone.w - gap) // 2
        panels = [(zone.x0, zone.y0, zone.x0 + pw, zone.y1), (zone.x0 + pw + gap, zone.y0, zone.x1, zone.y1)]
    else:
        ph = (zone.h - gap) // 2
        panels = [(zone.x0, zone.y0, zone.x1, zone.y0 + ph), (zone.x0, zone.y0 + ph + gap, zone.x1, zone.y1)]
    for (x0, y0, x1, y1), key, ok in ((panels[0], "do", True), (panels[1], "dont", False)):
        p.rect((x0, y0, x1, y1), CARD, 22)
        p.rect((x0, y0, x1, y0 + 10), GREEN if ok else RED, 0)
        y = y0 + 10 + 22
        _icon(p, x0 + 36, y, ok)
        lab = t[f"{key}_label"]
        p.text(lab, x0 + 36 + 64 + 22, y + (64 - p.block_h(lab)) // 2, align="left", slot=f"{key}_label")
        y += 64 + 14
        for it in t.get(f"{key}_items", []):
            p.d.ellipse((x0 + 36, y + 22, x0 + 50, y + 36), fill=(255, 255, 255, 217))
            p.text(it, x0 + 36 + 14 + 18, y, align="left", slot=f"{key}_items")
            y += p.block_h(it) + 14


def paint_step_list(p: Painter, s: SceneShot, zone: Box, aspect: str) -> None:
    t = s.text
    steps = t.get("steps", [])
    rows = [max(76, p.block_h(st)) + 32 for st in steps]
    th = p.block_h(t["title"]) + 28 if "title" in t else 0
    y = _stack_top(zone, [th, *rows], [22] * max(0, len(rows) - 1))
    if "title" in t:
        p.rect((zone.x0, y, zone.x0 + 12, y + p.block_h(t["title"])), RED, 0)
        p.text(t["title"], zone.x0 + 36, y, align="left", slot="title", shadow=True)
        y += th
    num_font = {"px": 48, "lh": 1.0, "weight": 800}
    for i, (st, rh) in enumerate(zip(steps, rows)):
        p.rect((zone.x0, y, zone.x1, y + rh), CARD, 18)
        cy = y + rh // 2
        p.d.ellipse((zone.x0 + 22, cy - 38, zone.x0 + 98, cy + 38), fill=RED)
        p.text({**num_font, "lines": [str(i + 1)]}, zone.x0 + 22, cy - 24, width=76, slot="number")
        p.text(st, zone.x0 + 22 + 76 + 28, cy - p.block_h(st) // 2, align="left", slot="steps")
        y += rh + 22


def paint_stat_callout(p: Painter, s: SceneShot, zone: Box, aspect: str) -> None:
    t = s.text
    parts = [k for k in ("stat", "label", "source") if k in t]
    hs = [p.block_h(t[k]) for k in parts]
    y = _stack_top(zone, hs, [26] * (len(hs) - 1))
    for k, h in zip(parts, hs):
        bw = p.block_w(t[k])
        x0 = zone.x0 + (zone.w - bw) // 2
        if k == "stat":
            p.rect((x0 - 36, y, x0 + bw + 36, y + h), RED, 28)
        elif k == "source":
            p.rect((x0 - 16, y, x0 + bw + 16, y + h), (0, 0, 0, 90), 10)
        p.text(t[k], zone.x0, y, width=zone.w, slot=k, shadow=(k == "label"))
        y += h + 26


def paint_contact_card(p: Painter, s: SceneShot, zone: Box, aspect: str) -> None:
    t = s.text
    items = t.get("contacts", [])
    rows = [(p.block_h(c["label"]) if "label" in c else 0) + p.block_h(c["value"]) + 36 for c in items]
    th = p.block_h(t["title"]) if "title" in t else 0
    y = _stack_top(zone, [th, *rows], [30] + [24] * max(0, len(rows) - 1))
    if "title" in t:
        p.text(t["title"], zone.x0, y, width=zone.w, slot="title", shadow=True)
        y += th + 30
    for c, rh in zip(items, rows):
        p.rect((zone.x0, y, zone.x1, y + rh), WHITE, 22)
        p.rect((zone.x0, y, zone.x0 + 14, y + rh), RED, 0)
        yy = y + 18
        inner_x, inner_w = zone.x0 + 54, zone.w - 94
        if "label" in c:
            p.text(c["label"], inner_x, yy, width=inner_w, color=DARK, slot="contact_label")
            yy += p.block_h(c["label"])
        p.text(c["value"], inner_x, yy, width=inner_w, color=RED, slot="contact_value")
        y += rh + 24


def _quote_mark(p: Painter, x: int, y: int) -> None:
    for dx in (0, 62):
        p.d.rectangle((x + dx, y + 48, x + dx + 44, y + 96), fill=RED)
        p.d.polygon([(x + dx, y + 58), (x + dx + 8, y + 22), (x + dx + 40, y), (x + dx + 46, y + 12),
                     (x + dx + 30, y + 26), (x + dx + 26, y + 50)], fill=RED)


def paint_quote_card(p: Painter, s: SceneShot, zone: Box, aspect: str) -> None:
    t = s.text
    qh = p.block_h(t["quote"])
    ah = p.block_h(t["attribution"]) + 26 if "attribution" in t else 0
    ch = 34 + 96 + 12 + qh + ah + 40
    y = _stack_top(zone, [ch + 12], [])
    p.rect((zone.x0, y, zone.x1, y + ch + 12), CARD, 26)
    p.rect((zone.x0, y + ch, zone.x1, y + ch + 12), RED, 0)
    _quote_mark(p, zone.x0 + 60, y + 34)
    yy = y + 34 + 96 + 12
    p.text(t["quote"], zone.x0 + 60, yy, align="left", slot="quote")
    if "attribution" in t:
        p.text(t["attribution"], zone.x0 + 60, yy + qh + 26, align="left", slot="attribution")


def _logo_image(path: Path, box_w: int, box_h: int) -> Image.Image | None:
    """Raster logo (PNG/JPG/WebP); for SVG use a sibling .png with the same stem (Pillow can't read SVG)."""
    cands = [path] if path.suffix.lower() != ".svg" else [path.with_suffix(".png")]
    for c in cands:
        if c.is_file():
            im = Image.open(c).convert("RGBA")
            im.thumbnail((box_w, box_h), Image.LANCZOS)
            return im
    return None


def paint_end_card(p: Painter, s: SceneShot, zone: Box, aspect: str, logo_path: Path | None = None) -> None:
    t = s.text
    lb = t.get("logo_box", {"w": 460, "h": 240})
    gap = 36 if aspect == "16x9" else 56
    parts = [k for k in ("cta", "url", "disclosure") if k in t]
    hs = [p.block_h(t[k]) + (8 if k == "url" else 0) for k in parts]
    y = _stack_top(zone, [lb["h"], *hs], [gap] * len(hs))
    logo = _logo_image(logo_path, lb["w"], lb["h"]) if logo_path else None
    lx0 = zone.x0 + (zone.w - lb["w"]) // 2
    if logo is not None:
        ox, oy = lx0 + (lb["w"] - logo.width) // 2, y + (lb["h"] - logo.height) // 2
        p.img.alpha_composite(logo, (ox, oy))
        p.logos.append({"kind": "logo", "x0": ox, "y0": oy, "x1": ox + logo.width, "y1": oy + logo.height})
    else:   # no raster available: neutral box so the layout still holds (logged as a warning)
        p.d.rounded_rectangle((lx0, y, lx0 + lb["w"], y + lb["h"]), radius=18, outline=RED, width=10)
        p.logos.append({"kind": "logo", "x0": lx0, "y0": y, "x1": lx0 + lb["w"], "y1": y + lb["h"], "missing_raster": True})
    y += lb["h"] + gap
    for k, h in zip(parts, hs):
        if k == "url":
            bw = p.block_w(t[k])
            x0 = zone.x0 + (zone.w - bw) // 2
            p.rect((x0 - 22, y, x0 + bw + 22, y + h), RED, 12)
            p.text(t[k], zone.x0, y + 4, width=zone.w, color=WHITE, slot="url")
        else:
            p.text(t[k], zone.x0, y, width=zone.w, color=DARK if k == "cta" else (58, 54, 55, 255), slot=k)
        y += h + gap


PAINTERS = {
    "hook_title": paint_hook_title, "illustration_kenburns": paint_illustration_kenburns,
    "do_dont_split": paint_do_dont_split, "step_list": paint_step_list, "stat_callout": paint_stat_callout,
    "contact_card": paint_contact_card, "quote_card": paint_quote_card, "end_card": paint_end_card,
}


# ---------------------------------------------------------------- backgrounds
def gradient_bg(w: int, h: int, end_card: bool = False) -> Image.Image:
    """Brand gradient (dark gray -> deep red) or the light end-card background; mirrors base.css."""
    import numpy as np

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if end_card:
        u = yy / max(1, h - 1)
        c0, c1 = np.array([255, 255, 255]), np.array([244, 241, 240])
        arr = c0[None, None, :] * (1 - u[..., None]) + c1[None, None, :] * u[..., None]
        return Image.fromarray(arr.astype("uint8"), "RGB")
    u = (xx * 0.34 + yy * 0.94) / (w * 0.34 + h * 0.94)
    stops = [(0.0, (61, 56, 57)), (0.55, (44, 40, 41)), (1.0, (75, 36, 39))]
    arr = np.zeros((h, w, 3), np.float32)
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        m = (u >= a) & (u <= b)
        f = ((u - a) / (b - a))[..., None]
        arr[m] = (np.array(ca) * (1 - f) + np.array(cb) * f)[m]
    img = Image.fromarray(arr.astype("uint8"), "RGB").convert("RGBA")
    band = Image.new("RGBA", (int(w * 1.4), int(h * 0.18)), (183, 32, 46, 46)).rotate(8, expand=True)
    img.alpha_composite(band, (int(-w * 0.2), int(h * 0.74 - band.height * 0.3)))
    return img.convert("RGB")


def cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """object-fit: cover (centre crop) like the HTML <img>."""
    r = max(w / img.width, h / img.height)
    im = img.resize((max(w, round(img.width * r)), max(h, round(img.height * r))), Image.LANCZOS)
    x0, y0 = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x0, y0, x0 + w, y0 + h))


def scrim(w: int, h: int, template: str) -> Image.Image:
    """Image darkening layer (mirrors .ar-scrim in base.css)."""
    import numpy as np

    u = np.linspace(0, 1, h, dtype=np.float32)
    if template == "illustration_kenburns":
        a = np.interp(u, [0, 0.4, 0.6, 1], [0.35, 0, 0, 0.35])
    else:
        a = np.interp(u, [0, 0.45, 1], [0.55, 0.35, 0.55])
    alpha = np.repeat((a * 255).astype("uint8")[:, None], w, axis=1)
    layer = Image.new("RGBA", (w, h), (14, 12, 13, 0))
    layer.putalpha(Image.fromarray(alpha, "L"))
    return layer


# ---------------------------------------------------------------- captions
def caption_png(c: SceneCaption, active: int, band: Box, style: dict) -> tuple[Image.Image, dict]:
    """Band-sized RGBA image of one caption page with word `active` highlighted (-1 = none)."""
    w, h = band.w, band.h
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    px, lh = c.px, c.px * style.get("line_height", 1.25)
    font = _font(700, px)
    outline = style.get("outline_px", 6)
    space = text_width(" ", 700, px)
    k = 0
    bx0, bx1, by1 = w, 0, 0
    for li, line in enumerate(c.lines):
        words = [wd.text for wd in line]
        lw = text_width(" ".join(words), 700, px)
        x = (w - lw) / 2
        y_mid = li * lh + lh / 2
        bx0, bx1, by1 = min(bx0, x), max(bx1, x + lw), (li + 1) * lh
        for wd in words:
            ww = text_width(wd, 700, px)
            if k == active:
                d.rounded_rectangle((x - 0.14 * px, y_mid - lh / 2 + 0.06 * px + (lh - px * 1.2) / 2,
                                     x + ww + 0.14 * px, y_mid + lh / 2 - (lh - px * 1.2) / 2),
                                    radius=int(0.18 * px), fill=style.get("highlight", "#B7202E"))
            d.text((x, y_mid), wd, font=font, fill=style.get("color", "#FFFFFF"), anchor="lm", stroke_width=outline,
                   stroke_fill=style.get("outline_color", "#110F10"))
            x += ww + space
            k += 1
    box = {"x0": int(band.x0 + bx0 - outline), "y0": int(band.y0), "x1": int(band.x0 + bx1 + outline), "y1": int(band.y0 + by1)}
    return img, box


# ---------------------------------------------------------------- pipeline
def _kb_filter(s: SceneShot, scene: Scene, n: int) -> str:
    """zoompan expression reproducing the HTML transform: scale around centre + translate (output px)."""
    W, H, k = scene.width, scene.height, UPSCALE
    kb = s.kenburns
    zf, zt = kb.zoom_from, kb.zoom_to
    xf, xt = kb.pan_x_from * (zf - 1) * W / 2, kb.pan_x_to * (zt - 1) * W / 2
    yf, yt = kb.pan_y_from * (zf - 1) * H / 2, kb.pan_y_to * (zt - 1) * H / 2
    u = f"(on/{max(1, n - 1)})"
    z = f"({zf}+({zt - zf})*{u})"
    tx, ty = f"({xf}+({xt - xf})*{u})", f"({yf}+({yt - yf})*{u})"
    x = f"iw/2-{tx}*{k}/{z}-iw/zoom/2"
    y = f"ih/2-{ty}*{k}/{z}-ih/zoom/2"
    return f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={W}x{H}:fps={scene.fps}"


def render_segment(s: SceneShot, scene: Scene, work: Path, log_logo: Path | None) -> tuple[Path, dict]:
    W, H, fps = scene.width, scene.height, scene.fps
    n = round(s.end * fps) - round(s.start * fps)
    zone = scene.safe.end if s.zone == "end" else scene.safe.content
    p = Painter(W, H)
    has_img = bool(s.image) and resolve_path(s.image).is_file()
    if has_img and s.template != "end_card":
        p.img.alpha_composite(scrim(W, H, s.template))
    painter = PAINTERS[s.template]
    if s.template == "end_card":
        painter(p, s, zone, scene.aspect, log_logo)
    else:
        painter(p, s, zone, scene.aspect)
    ov = work / f"ov_{s.shot_id}.png"
    p.img.save(ov)
    seg = work / f"seg_{s.shot_id:03d}.mp4"
    if has_img and s.kenburns is not None and s.template != "end_card":
        bg = work / f"bg_{s.shot_id}.png"
        cover(Image.open(resolve_path(s.image)).convert("RGB"), W * UPSCALE, H * UPSCALE).save(bg)
        bg_chain = f"[0:v]{_kb_filter(s, scene, n)},format=yuv420p[bg]"
        bg_in = ["-loop", "1", "-framerate", str(fps), "-i", str(bg)]
    else:
        bg = work / f"bg_{s.shot_id}.png"
        if has_img and s.template != "end_card":
            cover(Image.open(resolve_path(s.image)).convert("RGB"), W, H).save(bg)
        else:
            gradient_bg(W, H, end_card=s.template == "end_card").save(bg)
        bg_chain = "[0:v]format=yuv420p[bg]"
        bg_in = ["-loop", "1", "-framerate", str(fps), "-i", str(bg)]
    fc = (f"{bg_chain};[1:v]format=rgba,fade=t=in:st=0.08:d=0.35:alpha=1[ov];"
          f"[bg][ov]overlay=0:0:format=auto,format=yuv420p[v]")
    run([*bg_in, "-loop", "1", "-framerate", str(fps), "-i", str(ov), "-filter_complex", fc, "-map", "[v]",
         "-frames:v", str(n), "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
         "-pix_fmt", "yuv420p", "-y", "-hide_banner", "-loglevel", "error", str(seg)])
    layout = {"shot_id": s.shot_id, "template": s.template, "start": s.start, "end": s.end,
              "items": p.boxes, "logos": p.logos}
    return seg, layout


def write_caption_stream(scene: Scene, work: Path) -> tuple[Path | None, list[dict]]:
    """Concat-demuxer list of caption PNGs (one per active-word state, blanks in gaps) covering the timeline."""
    if not scene.captions:
        return None, []
    band = scene.safe.caption
    blank = work / "cap_blank.png"
    Image.new("RGBA", (band.w, band.h), (0, 0, 0, 0)).save(blank)
    entries: list[tuple[Path, float]] = []
    layout: list[dict] = []
    t = 0.0
    for c in sorted(scene.captions, key=lambda c: c.start):
        if c.start > t + 1e-3:
            entries.append((blank, c.start - t))
        words = [w for line in c.lines for w in line]
        starts = [max(c.start, w.start) for w in words] or [c.start]
        starts[0] = c.start
        for j in range(len(starts)):
            a, b = starts[j], (starts[j + 1] if j + 1 < len(starts) else c.end)
            if b - a <= 1e-3:
                continue
            img, box = caption_png(c, j if words else -1, band, scene.caption_style)
            f = work / f"cap_{c.index:03d}_{j:02d}.png"
            img.save(f)
            entries.append((f, b - a))
            if j == 0:
                st = scene.caption_style
                layout.append({"index": c.index, "start": c.start, "end": c.end, "text": c.text_lines(),
                               "font_px": c.px, "lines": len(c.lines), "color": st.get("color", "#FFFFFF"),
                               "stroke_color": st.get("outline_color", "#110F10"), "stroke_px": st.get("outline_px", 6), **box})
        t = max(t, c.end)
    if scene.duration_s > t + 1e-3:
        entries.append((blank, scene.duration_s - t))
    lst = work / "captions.ffconcat"
    lines = ["ffconcat version 1.0"]
    for f, dur in entries:
        lines += [f"file '{f.name}'", f"duration {dur:.4f}"]
    lines.append(f"file '{entries[-1][0].name}'")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lst, layout


def flat_boxes(shots: list[dict], caps: list[dict]) -> list[dict]:
    """Flat list for the critic: {shot_id, kind: text|caption|logo, text, x0..y1, font_px, lines, color, stroke_*}."""
    keys = ("text", "x0", "y0", "x1", "y1", "font_px", "lines", "color", "stroke_color", "stroke_px")
    out = [{"shot_id": s["shot_id"], "kind": "text", **{k: b[k] for k in keys}, "slot": b.get("slot", "")}
           for s in shots for b in s["items"]]
    out += [{"shot_id": s["shot_id"], "kind": "logo", "text": "", **{k: b[k] for k in ("x0", "y0", "x1", "y1")},
             "font_px": 0, "lines": 0, "color": "", "stroke_color": "", "stroke_px": 0} for s in shots for b in s["logos"]]
    for c in caps:
        mid = (c["start"] + c["end"]) / 2
        sh = next((s for s in shots if s["start"] <= mid < s["end"]), shots[-1] if shots else {"shot_id": -1})
        out.append({"shot_id": sh["shot_id"], "kind": "caption", **{k: c[k] for k in keys if k != "text"},
                    "text": " / ".join(c["text"]), "caption_index": c["index"], "start": c["start"], "end": c["end"]})
    return out


def mock_render(scene: Scene, out: Path) -> None:
    args = ["-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", f"color=c=0x2b2728:s={scene.width}x{scene.height}:r={scene.fps}:d={scene.duration_s}"]
    voice = resolve_path(scene.audio.voice_path) if scene.audio else None
    if voice and voice.is_file():
        args += ["-i", str(voice), "-af", "apad", "-t", f"{scene.duration_s:.3f}", "-c:a", "aac", "-ar", "48000", "-ac", "2"]
    run([*args, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)])


def handle(req: dict) -> dict:
    scene = Scene.load(req["scene_path"])
    out = Path(req["out_path"]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    layout_path = Path(req.get("layout_path") or out.parent / "layout.json")
    frames = scene.total_frames
    base_layout = {"engine": "ffmpeg", "width": scene.width, "height": scene.height, "aspect": scene.aspect,
                   "fps": scene.fps, "safe": scene.safe.model_dump()}
    meta = {"model": "ffmpeg+pillow", "model_revision": "", "licence": "LGPL/GPL (ffmpeg), HPND (Pillow)"}
    if req.get("mock"):
        mock_render(scene, out)
        layout_path.write_text(json.dumps({**base_layout, "mock": True, "boxes": [], "shots": [], "captions": []}, indent=1), encoding="utf-8")
        return {**meta, "outputs": {"video_path": str(out), "frames": frames, "layout_path": str(layout_path),
                                    "render_engine": "ffmpeg-mock"}, "units": {"frames": frames, "mock": True}}
    if not any(s.text for s in scene.shots):
        from core.scene import fit_scene_text
        fit_scene_text(scene)
    t0 = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix="agentreel-ff-", dir=req.get("work_dir")))
    try:
        logo = resolve_path(scene.brand.logo_path)
        segs, shots_layout = [], []
        for s in scene.shots:
            seg, lay = render_segment(s, scene, work, logo)
            segs.append(seg)
            shots_layout.append(lay)
        concat = work / "segments.ffconcat"
        concat.write_text("ffconcat version 1.0\n" + "".join(f"file '{p.name}'\n" for p in segs), encoding="utf-8")
        base = work / "base.mp4"
        run(["-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(base)])
        caps, caps_layout = write_caption_stream(scene, work)
        args = ["-y", "-hide_banner", "-loglevel", "error", "-i", str(base)]
        fc, vmap = "", "0:v"
        idx = 1
        if caps:
            args += ["-f", "concat", "-safe", "0", "-i", str(caps)]
            band = scene.safe.caption
            fc = f"[1:v]fps={scene.fps},format=rgba[c];[0:v][c]overlay={band.x0}:{band.y0}:format=auto:eof_action=pass,format=yuv420p[v]"
            vmap, idx = "[v]", 2
        voice = resolve_path(scene.audio.voice_path) if scene.audio else None
        if voice and voice.is_file():
            args += ["-i", str(voice)]
        if fc:
            args += ["-filter_complex", fc]
        args += ["-map", vmap]
        if voice and voice.is_file():
            args += ["-map", f"{idx}:a:0", "-af", "apad", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
        args += ["-frames:v", str(frames), "-t", f"{scene.duration_s:.3f}", "-r", str(scene.fps), "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", str(out)]
        run(args)
        layout = {**base_layout, "mock": False, "boxes": flat_boxes(shots_layout, caps_layout),
                  "shots": shots_layout, "captions": caps_layout}
        layout_path.write_text(json.dumps(layout, indent=1), encoding="utf-8")
        secs = time.perf_counter() - t0
        return {**meta, "outputs": {"video_path": str(out), "frames": frames, "layout_path": str(layout_path),
                                    "render_engine": "ffmpeg"},
                "units": {"frames": frames, "render_s": round(secs, 3), "render_fps": round(frames / secs, 2),
                          "ffmpeg": ffmpeg_bin()}}
    finally:
        if not req.get("keep_work"):
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    from core.workers import worker_main

    worker_main(handle)
