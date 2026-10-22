"""Editor: state -> scene.json -> render (HyperFrames, auto-fallback to ffmpeg) -> loudnorm + delivery encode.

Shot timing comes from the voice sentence spans (shot.sentence_indices); the end card (3-5 s, 4 s nominal,
stretched/shrunk towards the brief duration) is appended after the voice-over. Slots for each template are
derived from the storyboard's on_screen_text with the rules documented in renderer/templates/*/template.yaml.
Returns {"scene_path", "video_path", "extra": {"layout_path", "render_engine", ...}}.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import jsonschema

from core import ffmpeg_utils as ff
from core.context import ROOT, RunContext, load_yaml
from core.logging import RunLogger
from core.scene import (PLACEHOLDER_LOGO, Scene, SceneAudio, SceneBrand, SceneCaption, SceneShot,
                        SceneWord, disclosure_line, fit_scene_text, kenburns_for, load_template_spec, rel_path, safe_zones)
from core.schemas import TEMPLATES
from core.state import AgentState
from core.workers import WorkerError, run_worker

AGENT = "editor"
END_CARD_NOMINAL_S, END_CARD_MIN_S, END_CARD_MAX_S = 4.0, 3.0, 5.0
VO_TAIL_S = 0.3              # breath after the last word before the end card
MIN_SHOT_S = 1.0
WORD_RE = re.compile(r"[^\w']+", re.UNICODE)


# ---------------------------------------------------------------- small helpers
def _q(t: float, fps: int) -> float:
    """Quantise to the frame grid."""
    return round(round(t * fps) / fps, 4)


def _norm(word: str) -> str:
    return WORD_RE.sub("", word.lower())


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _host(url: str) -> str:
    try:
        h = urlparse(url if "//" in url else f"https://{url}").netloc
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


def _short_url(url: str) -> str:
    u = re.sub(r"^https?://", "", url.strip()).rstrip("/")
    return u[4:] if u.startswith("www.") else u


def _words(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")


def load_settings() -> dict:
    try:
        return load_yaml("config/settings.yaml")
    except FileNotFoundError:
        return {}


# ---------------------------------------------------------------- timing
def shot_times(shots: list[dict], spans: list[tuple[float, float]], content_end: float, fps: int = 30) -> list[tuple[float, float]]:
    """Contiguous [start, end) per shot from sentence spans; unanchored shots are interpolated; min 1 s each."""
    n = len(shots)
    if n == 0:
        return []
    anchors: list[float | None] = []
    for s in shots:
        idx = [i for i in s.get("sentence_indices", []) if 0 <= i < len(spans)]
        anchors.append(min(spans[i][0] for i in idx) if idx else None)
    anchors[0] = 0.0
    known = [i for i, a in enumerate(anchors) if a is not None]
    for i in range(n):
        if anchors[i] is None:
            p = max(k for k in known if k < i)
            nxt = [k for k in known if k > i]
            q, aq = (nxt[0], anchors[nxt[0]]) if nxt else (n, content_end)
            anchors[i] = anchors[p] + (aq - anchors[p]) * (i - p) / (q - p)  # type: ignore[operator]
    a = [float(x) for x in anchors]  # type: ignore[arg-type]
    for i in range(1, n):
        a[i] = max(a[i], a[i - 1] + MIN_SHOT_S)
    if content_end - a[-1] < MIN_SHOT_S:  # too many/late shots: spread evenly
        a = [content_end * i / n for i in range(n)]
    starts = [_q(x, fps) for x in a]
    ends = starts[1:] + [_q(content_end, fps)]
    return list(zip(starts, ends))


def end_card_length(content_end: float, target_s: float | None) -> float:
    if not target_s:
        return END_CARD_NOMINAL_S
    return max(END_CARD_MIN_S, min(END_CARD_MAX_S, target_s - content_end))


# ---------------------------------------------------------------- captions
def scene_captions(pages: list[dict]) -> list[SceneCaption]:
    """CaptionPage dumps -> pages whose lines carry timed words (for the active-word highlight)."""
    out: list[SceneCaption] = []
    for pg in pages:
        words = list(pg.get("words") or [])
        tokens = [ln.split() for ln in pg.get("lines") or []]
        flat = [t for ln in tokens for t in ln]
        times: list[tuple[float, float]] = []
        if len(flat) == len(words):
            times = [(w["start"], w["end"]) for w in words]
        else:   # greedy match on normalised text; unmatched tokens inherit the previous end
            j, last = 0, pg["start"]
            for tok in flat:
                k = next((m for m in range(j, min(len(words), j + 4)) if _norm(words[m]["word"]) == _norm(tok)), None)
                if k is None:
                    times.append((last, last))
                else:
                    times.append((words[k]["start"], words[k]["end"]))
                    last, j = words[k]["end"], k + 1
        it = iter(times)
        lines = [[SceneWord(text=t, start=round(a, 3), end=round(b, 3)) for t, (a, b) in zip(ln, it)] for ln in tokens]
        out.append(SceneCaption(index=pg["index"], start=round(pg["start"], 3), end=round(pg["end"], 3), lines=lines))
    return out


# ---------------------------------------------------------------- slots per template
def _match_contacts(texts: list[str], contacts: list[dict]) -> list[dict]:
    found: list[dict] = []
    for c in contacts:
        val, disp = c.get("value", ""), c.get("display", "")
        for t in texts:
            tl = t.lower()
            hit = False
            if c.get("kind") == "phone":
                d = _digits(disp) or _digits(val)
                hit = bool(d) and (d in _digits(t) or _digits(val)[-10:] in _digits(t) and len(_digits(t)) >= 3)
            elif c.get("kind") in ("email", "url"):
                hit = _short_url(val.lower()) in tl or (disp and disp.lower() in tl)
            if hit:
                found.append({"label": c.get("label") or c.get("kind", "").title(), "value": disp or val,
                              "kind": c.get("kind", "phone")})
                break
    return found[:3]


def cta_url(script: dict, contacts: list[dict], facts_by_id: dict) -> str:
    """Whitelisted URL for the end card: mentioned in the CTA > from the CTA's sources > first official URL."""
    urls = [c for c in contacts if c.get("kind") == "url"]
    if not urls:
        return ""
    cta = (script.get("cta") or {})
    text = f"{cta.get('display', '')} {cta.get('spoken', '')}".lower()
    for c in urls:
        if _short_url(c["value"].lower()) in text:
            return _short_url(c.get("display") or c["value"])
    srcs = {facts_by_id[f]["source_id"] for f in cta.get("fact_ids", []) if f in facts_by_id}
    for c in urls:
        if srcs & set(c.get("source_ids", [])):
            return _short_url(c.get("display") or c["value"])
    ranked = sorted(urls, key=lambda c: {"somaiya": 0, "gov": 1}.get(c.get("authority", ""), 2))
    return _short_url(ranked[0].get("display") or ranked[0]["value"])


def slots_for(template: str, shot: dict, sentences: list[dict], facts_by_id: dict, contacts: list[dict],
              brand: SceneBrand, script: dict) -> dict:
    """Template slots from on_screen_text (+ narration/facts/contacts). Rules mirror template.yaml."""
    ost = [str(t).strip() for t in shot.get("on_screen_text") or [] if str(t).strip()]
    narr = [sentences[i]["display"] for i in shot.get("sentence_indices", []) if 0 <= i < len(sentences)]
    first_fact = next((facts_by_id[f] for f in shot.get("fact_ids", []) if f in facts_by_id), None)
    if template == "hook_title":
        title = ost[0] if ost else _words(narr[0] if narr else "", 8)
        return {"title": title, **({"subtitle": ost[1]} if len(ost) > 1 else {})}
    if template == "illustration_kenburns":
        return {"headline": ost[0]} if ost else {}
    if template == "do_dont_split":
        neg = re.compile(r"^\s*(don'?t|do not|never|avoid|no\s|✗|x\s)[:\-\s]*", re.I)
        pos = re.compile(r"^\s*(do|✓|yes)[:\-\s]+", re.I)
        do, dont = [], []
        for t in ost:
            if neg.match(t):
                rest = neg.sub("", t, count=1).strip()
                dont.append(rest[:1].upper() + rest[1:] if not t.lower().startswith(("never", "avoid", "no ")) else t.strip())
            else:
                rest = pos.sub("", t, count=1).strip()
                do.append(rest[:1].upper() + rest[1:])
        if not dont and len(do) >= 2:
            half = (len(do) + 1) // 2
            do, dont = do[:half], do[half:]
        return {"do_items": do[:3], "dont_items": dont[:3]}
    if template == "step_list":
        if ost and ost[0].endswith(":"):
            return {"title": ost[0].rstrip(":").strip(), "steps": ost[1:6]}
        return {"steps": ost[:5]}
    if template == "stat_callout":
        out = {"stat": ost[0] if ost else ""}
        if len(ost) > 1:
            out["label"] = ost[1]
        if first_fact and _host(first_fact.get("url", "")):
            out["source"] = _host(first_fact["url"])
        return out
    if template == "contact_card":
        found = _match_contacts(ost, contacts) or _match_contacts(narr, contacts)
        title = next((t for t in ost if not _match_contacts([t], contacts)), "")
        return {**({"title": title} if title else {}), "contacts": found}
    if template == "quote_card":
        out = {"quote": ost[0] if ost else (first_fact or {}).get("quote", "")[:150]}
        if first_fact:
            out["attribution"] = _host(first_fact.get("url", ""))
        return out
    if template == "end_card":
        cta = ost[0] if ost else (script.get("cta") or {}).get("display", "")
        url = cta_url(script, contacts, facts_by_id)
        return {"cta": _words(cta, 16), **({"url": url} if url else {}), "disclosure": brand.disclosure,
                "logo": brand.logo_path}
    raise ValueError(f"unknown template {template}")


def _valid(template: str, slots: dict) -> list[str]:
    schema = load_template_spec(template)["slots_schema"]
    return [e.message for e in jsonschema.Draft202012Validator(schema).iter_errors(slots)]


def _fallback_slots(shot: dict, sentences: list[dict], has_image: bool) -> tuple[str, dict]:
    """When a template's slots can't be filled validly: image -> illustration, else a hook_title card."""
    ost = [t for t in shot.get("on_screen_text") or [] if str(t).strip()]
    narr = [sentences[i]["display"] for i in shot.get("sentence_indices", []) if 0 <= i < len(sentences)]
    text = ost[0] if ost else _words(narr[0] if narr else " ", 8)
    if has_image:
        return "illustration_kenburns", ({"headline": text} if text.strip() else {})
    return "hook_title", {"title": text.strip() or "…"}


# ---------------------------------------------------------------- scene builder
def build_scene(ctx: RunContext, state: AgentState, settings: dict | None = None) -> tuple[Scene, list[str]]:
    """Pure function of state (+ settings): returns the validated Scene and a list of warnings."""
    settings = settings if settings is not None else load_settings()
    warnings: list[str] = []
    plan, brief = state.get("plan") or {}, state.get("brief") or {}
    aspect = plan.get("aspect") or ("16x9" if brief.get("format") == "video_16x9" else "9x16")
    width = plan.get("width") or (1920 if aspect == "16x9" else 1080)
    height = plan.get("height") or (1080 if aspect == "16x9" else 1920)
    fps = plan.get("fps", 30)
    target = plan.get("duration_s") or brief.get("duration_s")
    script = state.get("script") or {}
    sentences = [{"display": script["hook"]["display"], "fact_ids": script["hook"].get("fact_ids", [])}] if script.get("hook") else []
    sentences += list(script.get("body") or []) + ([script["cta"]] if script.get("cta") else [])
    facts_by_id = {f["fact_id"]: f for f in state.get("facts") or []}
    contacts = list(state.get("contacts") or [])
    voice = state.get("voice") or {}
    spans = [tuple(s) for s in voice.get("sentence_spans") or []]
    vo_dur = float(voice.get("duration_s") or (spans[-1][1] if spans else 0.0))

    logo = settings.get("logo_path") or ""
    logo_p = (ROOT / logo) if logo and not Path(logo).is_absolute() else Path(logo) if logo else None
    if not logo_p or not logo_p.is_file():
        warnings.append(f"logo {logo or '(unset)'} not found -> placeholder {rel_path(PLACEHOLDER_LOGO)}")
        logo_p = PLACEHOLDER_LOGO
    brand = SceneBrand(logo_path=rel_path(logo_p), disclosure=settings.get("disclosure_line") or disclosure_line())

    sb_shots = list((state.get("storyboard") or {}).get("shots") or [])
    end_src = next((s for s in sb_shots if s.get("template") == "end_card"), None)
    body = [s for s in sb_shots if s.get("template") != "end_card"]
    if not vo_dur:
        vo_dur = max(MIN_SHOT_S * max(1, len(body)), float(target or 45) - END_CARD_NOMINAL_S - VO_TAIL_S)
        warnings.append(f"no voice timings -> shots spread over {vo_dur:.1f}s")
    content_end = _q(vo_dur + VO_TAIL_S, fps)
    end_len = end_card_length(content_end, target)
    total = _q(content_end + end_len, fps)
    if target and abs(total - target) > 2:
        warnings.append(f"duration {total:.2f}s vs brief {target}s (VO {vo_dur:.2f}s): outside ±2 s")

    assets = state.get("shot_assets") or {}
    shots: list[SceneShot] = []
    for (start, end), sb in zip(shot_times(body, spans, content_end, fps), body):
        sid = int(sb["shot_id"])
        asset = assets.get(str(sid)) or {}
        # only an image generated for THIS shot's current prompt (repair loops reuse shot ids)
        img = asset.get("image_path", "") if asset.get("visual_prompt", "") == sb.get("visual_prompt", "") else ""
        if asset.get("image_path") and not img:
            warnings.append(f"shot {sid}: stale image (made for a different prompt) ignored")
        if img and not Path(img).is_file() and not (ROOT / img).is_file():
            warnings.append(f"shot {sid}: image {img} missing -> gradient")
            img = ""
        tpl = sb.get("template") if sb.get("template") in TEMPLATES else "hook_title"
        slots = slots_for(tpl, sb, sentences, facts_by_id, contacts, brand, script)
        errs = _valid(tpl, slots) or ([] if img or load_template_spec(tpl).get("image") != "required" else ["image required"])
        if errs:
            new_tpl, slots = _fallback_slots(sb, sentences, bool(img))
            warnings.append(f"shot {sid}: {tpl} slots invalid ({'; '.join(errs)[:200]}) -> {new_tpl}")
            tpl = new_tpl
        shots.append(SceneShot(shot_id=sid, template=tpl, start=start, end=end, image=rel_path(img) if img else "",
                               slots=slots, kenburns=kenburns_for(ctx.seed, sid) if img else None,
                               sentence_indices=list(sb.get("sentence_indices") or []),
                               fact_ids=list(sb.get("fact_ids") or [])))
    end_id = int(end_src["shot_id"]) if end_src else (max([s.shot_id for s in shots], default=0) + 1)
    end_sb = end_src or {"shot_id": end_id, "on_screen_text": [], "fact_ids": (script.get("cta") or {}).get("fact_ids", [])}
    shots.append(SceneShot(shot_id=end_id, template="end_card", start=content_end, end=total,
                           slots=slots_for("end_card", end_sb, sentences, facts_by_id, contacts, brand, script),
                           fact_ids=list(end_sb.get("fact_ids") or [])))

    audio = None
    if voice.get("wav_path"):
        audio = SceneAudio(voice_path=rel_path(voice["wav_path"]), duration_s=round(vo_dur, 3))
    scene = Scene(run_id=ctx.run_id, brief_id=brief.get("id", plan.get("brief_id", "")), seed=ctx.seed, aspect=aspect,
                  width=width, height=height, fps=fps, duration_s=total, safe=safe_zones(width, height), shots=shots,
                  captions=[c for c in scene_captions(state.get("captions") or []) if c.start < content_end],
                  audio=audio, brand=brand)
    fit_scene_text(scene)
    for s in scene.shots:
        for path, blk in _iter_blocks(s.text):
            if blk.get("overflow"):
                warnings.append(f"shot {s.shot_id} ({s.template}): text '{path}' does not fit its box at {blk['px']}px")
    warnings += scene.warnings()
    return scene, warnings


def _iter_blocks(node: Any, path: str = ""):
    if isinstance(node, dict) and "lines" in node:
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _iter_blocks(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _iter_blocks(v, f"{path}[{i}]")


# ---------------------------------------------------------------- render + post
def render(ctx: RunContext, log: RunLogger, scene_path: Path, scene: Scene, engine: str) -> dict:
    """Run one renderer worker; returns its outputs (+ engine)."""
    rdir = ctx.out_dir / "render"
    rdir.mkdir(exist_ok=True)
    req = {"task": "render", "scene_path": str(scene_path), "out_path": str(rdir / f"render_{engine}.mp4"),
           "layout_path": str(ctx.out_dir / "layout.json"), "mock": ctx.mock_media, "seed": scene.seed}
    if engine == "hyperframes":
        req.update({"ffmpeg_path": ff.ffmpeg_bin(), "ffprobe_path": ff.ffprobe_bin()})
        return run_worker(ctx, log, agent=AGENT, worker="render_hyperframes.mjs", request=req, gpu=None, timeout_s=1800)
    return run_worker(ctx, log, agent=AGENT, worker="render_ffmpeg.py", request=req, gpu=None, timeout_s=1800)


def render_with_fallback(ctx: RunContext, log: RunLogger, scene_path: Path, scene: Scene) -> dict:
    engines = ["hyperframes", "ffmpeg"] if ctx.renderer != "ffmpeg" else ["ffmpeg"]
    last: Exception | None = None
    for i, eng in enumerate(engines):
        try:
            out = render(ctx, log, scene_path, scene, eng)
            out.setdefault("render_engine", eng)
            if i > 0:
                out["fallback_reason"] = f"{last.__class__.__name__}: {str(last)[:300]}"
            return out
        except (WorkerError, OSError) as e:
            last = e
            log.record(agent=AGENT, tool=f"render:{eng}", status="error", error=str(e)[:2000],
                       units={"fallback_to": engines[i + 1] if i + 1 < len(engines) else ""})
    raise WorkerError(f"all renderers failed: {last}")


def post_process(ctx: RunContext, log: RunLogger, scene: Scene, video_in: str, final: Path) -> dict:
    """Two-pass loudnorm on the voice (padded to the video length) + delivery encode + loudness check."""
    norm = ctx.out_dir / "render" / "voice_norm.wav"
    src = None
    if scene.audio:
        p = Path(scene.audio.voice_path)
        src = p if p.is_absolute() else ROOT / p
    if src is None or not src.is_file():
        src = Path(video_in)   # audio track of the render
    ln = ff.loudnorm_two_pass(src, norm, duration=scene.duration_s, log=log, agent=AGENT)
    ff.final_encode(video_in, norm, final, duration=scene.duration_s, fps=scene.fps, log=log, agent=AGENT)
    return {"loudnorm": ln, "final_loudness": ff.measure_loudness(final, log=log, agent=AGENT),
            "probe": ff.probe(final, log=log, agent=AGENT)["summary"]}


def record_assets(ctx: RunContext, scene: Scene, final: Path, engine: str, settings: dict) -> None:
    """Manifest rows (core/manifest.py) for the fonts, the logo and the rendered video."""
    from core.logging import sha256_file
    from core.manifest import add_asset
    from core.schemas import AssetRecord
    from core.scene import FONTS_DIR, resolve_path

    for f in sorted(FONTS_DIR.glob("*.ttf")):
        add_asset(ctx, AssetRecord(path=str(f), kind="font", generator="official", licence="OFL-1.1",
                                   sha256=sha256_file(f), source_url="https://github.com/google/fonts/tree/main/ofl"))
    logo = resolve_path(scene.brand.logo_path)
    placeholder = logo.resolve() == PLACEHOLDER_LOGO.resolve()
    add_asset(ctx, AssetRecord(path=str(logo), kind="logo", generator="placeholder" if placeholder else "official",
                               licence="own-work" if placeholder else "official", sha256=sha256_file(logo),
                               source_url="" if placeholder else settings.get("logo_source_url", "")))
    add_asset(ctx, AssetRecord(path=str(final), kind="render", generator=engine, seed=scene.seed, licence="own-work",
                               sha256=sha256_file(final)))


# ---------------------------------------------------------------- node
def make_node(ctx: RunContext, log: RunLogger, llm) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        fb = (state.get("feedback") or {}).get(AGENT, [])
        if fb:
            log.record(agent=AGENT, tool="feedback", units={"feedback": fb[:10]})
        scene, warns = build_scene(ctx, state)
        errs = scene.validate_scene()
        scene_path = ctx.artifact("scene.json")
        scene.save(scene_path)
        log.record(agent=AGENT, tool="scene", output_path=str(scene_path),
                   status="ok" if not errs else "error", error="; ".join(errs)[:2000],
                   units={"shots": len(scene.shots), "captions": len(scene.captions), "duration_s": scene.duration_s,
                          "warnings": warns[:30]})
        if errs:
            raise ValueError(f"scene.json invalid: {errs}")
        out = render_with_fallback(ctx, log, scene_path, scene)
        brief_id = scene.brief_id or (state.get("brief") or {}).get("id", "BRIEF")
        final = ctx.out_dir / f"{ctx.final_basename(brief_id)}.mp4"
        post = post_process(ctx, log, scene, out["video_path"], final)
        try:
            record_assets(ctx, scene, final, out.get("render_engine", ""), load_settings())
        except Exception as e:  # noqa: BLE001 - manifest problems must not lose a finished render
            log.record(agent=AGENT, tool="manifest", status="error", error=str(e)[:500])
        srt = state.get("srt_path")
        if srt and Path(srt).is_file():
            dst = final.with_suffix(".srt")
            if Path(srt).resolve() != dst.resolve():
                shutil.copyfile(srt, dst)
        return {
            "scene_path": str(scene_path),
            "video_path": str(final),
            "extra": {"layout_path": out.get("layout_path", ""), "render_engine": out.get("render_engine", ""),
                      "render_fallback_reason": out.get("fallback_reason", ""), "scene_warnings": warns,
                      "final_loudness": post["final_loudness"], "final_probe": post["probe"]},
        }

    return node
