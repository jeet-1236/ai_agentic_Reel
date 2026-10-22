"""Critic: QA hard gate (code) first, then soft checks (LLM rule review, VLM frame judge, NLI), then routing.

Writes qa_report_loop<N>.json + qa_report.json. Routing (PLAN §3):
- pass -> human_review (plan.mandatory_review) | export
- fail, loop < cap -> the most-upstream responsible agent (researcher > scriptwriter > storyboarder >
  asset_generator > editor) with feedback for every failing agent, redo_shots, extra.redo_voice, loop+1
- regression guard: a check failing again on the same artifact sha after its agent was asked to fix it
  -> escalate (stop looping)
- loop >= cap or escalated -> human_review (A/C, or any hard failure left) | export
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from core.context import ROOT, RunContext, load_yaml
from core.llm import LLM
from core.logging import RunLogger, sha256_file
from core.schemas import FrameJudgeOut, QACheck, QAReport, Script, SoftReviewOut
from core.seeds import stage_seed
from core.state import AgentState
from core import qa_checks as q

AGENT_PRIORITY = ["researcher", "scriptwriter", "storyboarder", "asset_generator", "editor"]
SOFT_AGENTS = set(AGENT_PRIORITY)
IMAGE_CHECKS = ("faces:", "nsfw:", "frame_judge:")
VOICE_CHECKS = ("voice_wer",)
# soft checks that are reported but never trigger a repair loop on their own
COSMETIC_SOFT = {"caption_cps", "caption_min_duration", "word_budget", "duration_target", "end_card_logo",
                 "caption_contrast", "on_screen_words"}

FIX_HINTS: dict[str, str] = {
    "duration": "Fit the brief target: adjust the word budget (scriptwriter) or the timeline/end-card length (editor).",
    "video_": "Re-encode: -c:v libx264 -profile:v high -pix_fmt yuv420p -r 30 -fps_mode cfr at the plan size.",
    "audio_": "Re-encode audio: -c:a aac -b:a 160k -ar 48000 -ac 2.",
    "loudness": "Two-pass loudnorm I=-14:TP=-1.5:LRA=11 with measured values, linear=true.",
    "file_size": "Raise CRF / lower bitrate to stay under 100 MB.",
    "dead_air": "Remove silence > 1.5 s outside the end card (tighten sentence gaps).",
    "black_frames": "Remove unplanned black frames (check missing images / transitions).",
    "frozen_frames": "Add motion (Ken Burns) to static shots; no frozen frames > 3 s.",
    "caption_safe_zone": "Keep every text box inside the safe band; shorten or move on-screen text.",
    "caption_height": "Caption font must be >= 48 px.",
    "caption_contrast": "Increase caption contrast (outline >= 4 px or darker band) to >= 4.5:1.",
    "caption_": "Re-chunk captions: <= 2 lines, <= 32 chars/line, <= 17 CPS, >= 0.83 s per page.",
    "hook_topic_3s": "Make the hook name the topic (a plan keyword) within the first 3 s, spoken and on screen.",
    "end_card": "End card must show the official logo unaltered, the disclosure line verbatim and the CTA.",
    "facts_tagged": "Tag every factual sentence with the fact_ids that support it, or rewrite it as non-factual.",
    "facts_known_ids": "Use only fact ids from facts.json.",
    "facts_quotes": "Re-fetch/curate evidence: every cited fact needs a verbatim quote and URL.",
    "facts_evidence": "Run one more search round for official evidence.",
    "nli:": "Rewrite the sentence so it is entailed by the quoted fact (quote it or label the paraphrase).",
    "contacts_whitelist": "Use only contacts from the whitelist (exact numbers/emails/URLs).",
    "must_include": "Cover this must_include item using its facts; if MISSING, say 'check the official page'.",
    "must_avoid": "Remove the must_avoid content.",
    "word_budget": "Adjust the script length to the plan word budget.",
    "cta_present": "Add a CTA sentence taken from the sources.",
    "onscreen_words": "Limit each on-screen text slot to 7 words.",
    "faces:": "Regenerate this shot: flat vector, no realistic people (silhouettes/objects).",
    "nsfw:": "Regenerate this shot with a safer prompt.",
    "frame_judge:": "Regenerate this shot to match intent: no realistic faces, no garbled text, no fear imagery.",
    "voice_wer": "Regenerate the voice; add lexicon entries for mispronounced words.",
    "licences": "Replace assets whose licence is not on the allowlist.",
    "soft_review": "Apply the reviewer's fix instructions.",
    "render_output": "Render failed or produced no video; re-run the renderer (fallback: ffmpeg).",
}


def _hint(check_id: str) -> str:
    for k, v in FIX_HINTS.items():
        if check_id.startswith(k):
            return v
    return "Fix the failing check."


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _file_sha(p: str | None) -> str:
    try:
        return sha256_file(p)[:16] if p and Path(p).is_file() else ""
    except OSError:
        return ""


def artifact_sha(state: AgentState, check: QACheck, agent: str) -> str:
    """Hash of the artifact the responsible agent produced for this check (regression guard key)."""
    cid = check.check_id
    if cid.startswith(IMAGE_CHECKS):
        sid = cid.split(":", 1)[1]
        a = (state.get("shot_assets") or {}).get(sid, {})
        return _file_sha(a.get("image_path")) or _sha_json(a)
    if cid in VOICE_CHECKS:
        v = state.get("voice") or {}
        return _file_sha(v.get("wav_path")) or _sha_json(v)
    if agent == "researcher":
        return _sha_json([state.get("facts"), [s.get("sha256") for s in state.get("sources") or []]])
    if agent == "scriptwriter":
        return _sha_json(state.get("script"))
    if agent == "storyboarder":
        return _sha_json(state.get("storyboard"))
    if agent == "asset_generator":
        return _sha_json([state.get("shot_assets"), state.get("voice"), state.get("captions")])
    return _file_sha(state.get("video_path")) or _sha_json(state.get("scene_path"))


# ================================================================ routing (pure)
def decide(state: AgentState, report: QAReport, *, max_loops: int,
           sha_fn: Callable[[AgentState, QACheck, str], str] = artifact_sha) -> dict:
    """Pure routing decision; returns the state update (without qa_report)."""
    loop = int(state.get("loop", 0) or 0)
    mandatory = bool((state.get("plan") or {}).get("mandatory_review", False))
    fails = report.failures()
    cleared = {a: [] for a in AGENT_PRIORITY}
    summary: dict[str, Any] = {"loop": loop, "passed": report.passed, "failures": [c.check_id for c in fails],
                               "hard_failures": [c.check_id for c in fails if c.severity == "hard"]}
    if not fails:
        return {"route": "human_review" if mandatory else "export", "feedback": cleared, "redo_shots": [],
                "extra": {"redo_voice": False, "qa_summary": {**summary, "reason": "all checks passed"}}}

    # Soft findings never justify an expensive repair loop on their own when they are cosmetic (a loop re-runs
    # LLM + media stages: minutes on a T4), and other soft findings get at most one loop. They stay in the
    # QA report and the review packet either way.
    soft_only = not summary["hard_failures"]
    if soft_only and (all(c.check_id in COSMETIC_SOFT for c in fails) or loop >= 1):
        return {"route": "human_review" if mandatory else "export", "feedback": cleared, "redo_shots": [],
                "extra": {"redo_voice": False,
                          "qa_summary": {**summary, "reason": "only soft findings: reported, not repaired"}}}

    history = list(state.get("history") or [])
    researcher_used = any(h.get("routed_to") == "researcher" for h in history)

    def agent_of(c: QACheck) -> str:
        a = c.responsible_agent
        if a == "researcher" and researcher_used:     # one extra search round only
            return "scriptwriter"
        return a if a in AGENT_PRIORITY else "editor"

    entries = [{"loop": loop, "check_id": c.check_id, "artifact_sha": sha_fn(state, c, agent_of(c)), "agent": agent_of(c)}
               for c in fails]
    regress = [e for e in entries for h in history
               if h.get("check_id") == e["check_id"] and h.get("artifact_sha") == e["artifact_sha"]
               and h.get("routed_to") == e["agent"] and int(h.get("loop", -1)) < loop]
    target = min({e["agent"] for e in entries}, key=AGENT_PRIORITY.index)

    feedback = dict(cleared)
    for c in fails:
        msg = f"[{c.check_id}] value={c.value} (threshold {c.threshold}). Evidence: {c.evidence}. Fix: {_hint(c.check_id)}"
        feedback.setdefault(agent_of(c), []).append(msg[:1200])
    redo = sorted({int(c.check_id.split(":", 1)[1]) for c in fails
                   if c.check_id.startswith(IMAGE_CHECKS) and c.check_id.split(":", 1)[1].isdigit()})
    redo_voice = any(c.check_id in VOICE_CHECKS for c in fails)

    if loop >= max_loops or regress:
        hard_left = bool(summary["hard_failures"])
        route = "human_review" if (mandatory or hard_left) else "export"
        reason = (f"regression: {sorted({e['check_id'] for e in regress})} reproduced the same artifact"
                  if regress else f"loop cap reached ({loop} >= {max_loops})")
        for e in entries:
            e["routed_to"] = route
        return {"route": route, "history": entries, "feedback": feedback,
                "extra": {"qa_summary": {**summary, "escalated": bool(regress), "reason": reason}}}

    for e in entries:
        e["routed_to"] = target
    return {"route": target, "loop": loop + 1, "history": entries, "feedback": feedback, "redo_shots": redo,
            "extra": {"redo_voice": redo_voice,
                      "qa_summary": {**summary, "escalated": False, "reason": f"repair loop {loop + 1} -> {target}"}}}


# ================================================================ helpers
def shot_windows(state: AgentState, duration: float) -> list[dict]:
    """[{shot_id, start, end, shot}] from shot_assets timings, else voice sentence spans, else equal split."""
    shots = (state.get("storyboard") or {}).get("shots", [])
    assets = state.get("shot_assets") or {}
    spans = (state.get("voice") or {}).get("sentence_spans") or []
    out = []
    for i, sh in enumerate(shots):
        a = assets.get(str(sh.get("shot_id")), {})
        s, e = float(a.get("start", 0) or 0), float(a.get("end", 0) or 0)
        if e <= s and spans and sh.get("sentence_indices"):
            idx = [j for j in sh["sentence_indices"] if 0 <= j < len(spans)]
            if idx:
                s, e = float(spans[min(idx)][0]), float(spans[max(idx)][1])
        if e <= s and duration > 0:
            step = duration / max(1, len(shots))
            s, e = i * step, (i + 1) * step
        out.append({"shot_id": sh.get("shot_id"), "start": s, "end": e, "shot": sh})
    return out


def _find_layout(ctx: RunContext, state: AgentState) -> Any:
    cands = []
    if state.get("scene_path"):
        cands.append(Path(state["scene_path"]).with_name("layout.json"))
    cands.append(ctx.artifact("layout.json"))
    for p in cands:
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8")), str(p)
            except Exception:  # noqa: BLE001
                pass
    return None, ""


def _manifest_rows(ctx: RunContext) -> list[dict]:
    try:
        from core import manifest  # type: ignore[attr-defined]

        for name in ("manifest_path", "path_for"):
            fn = getattr(manifest, name, None)
            if fn:
                return q.read_manifest_rows(fn(ctx))
    except Exception:  # noqa: BLE001
        pass
    for p in (ctx.assets_dir / "manifest.csv", ctx.out_dir / "manifest.csv", ROOT / "assets" / "manifest.csv"):
        rows = q.read_manifest_rows(p)
        if rows:
            return rows
    return []


def _load_scene(state: AgentState) -> dict:
    p = state.get("scene_path")
    if p and Path(p).is_file():
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _code_rule_ids(packs: list[str]) -> list[str]:
    try:
        from core.rules import code_rule_ids

        return list(code_rule_ids(list(packs)))
    except Exception:  # noqa: BLE001
        import yaml

        ids = []
        for pack in ["base", *packs]:
            p = ROOT / "config" / "rules" / f"{pack}.yaml"
            if p.exists():
                ids += [r["id"] for r in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("rules", [])
                        if r.get("check") == "code"]
        return ids


def _licence_predicate():
    try:
        from core.manifest import licence_allowed

        return licence_allowed
    except Exception:  # noqa: BLE001
        return q.load_licence_allowlist(ROOT)


def _settings() -> dict:
    try:
        return load_yaml("config/settings.yaml")
    except FileNotFoundError:
        return {}


def _models() -> dict:
    try:
        return load_yaml("config/models.yaml")
    except FileNotFoundError:
        return {}


def rules_text(packs: list[str], applies_to: tuple[str, ...] = ("script", "storyboard", "frames", "contacts")) -> str:
    """Rule-pack text for the soft review: core.rules.rules_text when available, else read the YAML packs."""
    try:
        from core.rules import load_packs  # creative engineer's module

        rules = [r for r in load_packs(list(packs))["rules"] if r.get("check") == "llm"]
        if rules:
            return "\n".join(f"- [{r['id']}] ({r.get('applies_to')} -> {r.get('responsible_agent')}) {r['text']}"
                             for r in rules)
    except Exception:  # noqa: BLE001
        try:
            from core.rules import rules_text as _rt

            txt = _rt(packs)
            if isinstance(txt, str) and txt.strip():
                return txt
        except Exception:  # noqa: BLE001
            pass
    import yaml

    lines = []
    for pack in packs:
        p = ROOT / "config" / "rules" / f"{pack}.yaml"
        if not p.exists():
            continue
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for r in data.get("rules", []) or []:
            if r.get("check", "llm") == "llm" and r.get("applies_to", "script") in applies_to:
                lines.append(f"- [{r.get('id')}] ({r.get('applies_to', 'script')} -> {r.get('responsible_agent', 'scriptwriter')}) {r.get('text')}")
    return "\n".join(lines) or "- No category-specific rules: check tone, accuracy, no blame, no fear imagery, safe messaging."


# ================================================================ hard checks
def run_hard_checks(ctx: RunContext, log: RunLogger, state: AgentState) -> tuple[list[QACheck], dict]:
    """All §9 code checks. Returns (checks, media) where media holds measurements, frames, OCR."""
    plan, brief = state.get("plan") or {}, state.get("brief") or {}
    script, storyboard = state.get("script") or {}, state.get("storyboard") or {}
    settings, models = _settings(), _models()
    loop = int(state.get("loop", 0) or 0)
    checks: list[QACheck] = []
    media: dict[str, Any] = {"frames": {}, "ocr": {}, "duration": 0.0}
    width, height = int(plan.get("width", 1080)), int(plan.get("height", 1920))

    video = state.get("video_path") or ""
    if not video or not Path(video).is_file():
        checks.append(q.qc("render_output", False, "editor", value="missing", threshold="mp4 exists",
                           evidence=f"video_path='{video}'"))
    else:
        try:
            with log.call("critic", "ffmpeg:qa_measure", output_path=video) as rec:
                m = q.measure_media(video)
                rec["units"] = {"duration_s": q.duration_of(m)}
            media["measure"] = {k: v for k, v in m.items() if k != "probe"}
            d = q.duration_of(m)
            media["duration"] = d
            voice_d = float((state.get("voice") or {}).get("duration_s", 0.0) or 0.0)
            checks += q.check_duration(m, float(brief.get("duration_s", plan.get("duration_s", 45))), voice_d)
            checks += q.check_video_stream(m, width, height, int(plan.get("fps", 30)))
            checks += q.check_audio_stream(m)
            checks += q.check_loudness(m)
            checks.append(q.check_size(m))
            wins = shot_windows(state, d)
            bounds = sorted({round(w["end"], 3) for w in wins[:-1]})
            vwords = (state.get("voice") or {}).get("words") or []
            vo_end = float(vwords[-1]["end"]) if vwords else float((state.get("voice") or {}).get("duration_s") or 0)
            checks += q.check_dead_air(m, q.allowed_windows(d, bounds, vo_end=vo_end or None))
            media["windows"] = wins
            _extract_frames(ctx, log, video, wins, d, loop, media)
        except Exception as e:  # noqa: BLE001
            checks.append(q.qc("render_output", False, "editor", value="unreadable", threshold="ffprobe/ffmpeg ok",
                               evidence=f"{e.__class__.__name__}: {e}"[:500]))

    # caption layout: renderer layout.json, else OCR boxes
    layout, lpath = _find_layout(ctx, state)
    if layout is None and media["ocr"]:
        layout = [{"text": it["text"], "box": it["box"], "bbox": it["box"], "kind": "text", "shot_id": k}
                  for k, items in media["ocr"].items() for it in items]
        lpath = "ocr"
    scene = _load_scene(state)
    media["scene"] = scene
    lay = q.check_caption_layout(layout, width, height, source=lpath or "layout.json")
    if not any(c.check_id == "caption_height" for c in lay) and scene.get("caption_style"):
        px = float(scene["caption_style"].get("font_px", 0) or 0)
        lay.append(q.qc("caption_height", px >= q.CAPTION_MIN_PX, "editor", value=f"{px:.0f}px",
                        threshold=f">= {q.CAPTION_MIN_PX:.0f} px", evidence="declared: scene.json caption_style.font_px"))
    checks += lay
    checks += _contrast(media, layout, width, height)

    pages = state.get("captions") or []
    if not pages and state.get("srt_path") and Path(state["srt_path"]).is_file():
        pages = q.parse_srt(state["srt_path"])
    checks += q.check_caption_readability(pages)

    if script:
        # hook: voice words before 3 s + on-screen text of shots starting before 3 s (+ OCR of the hook frame)
        wins = media.get("windows") or shot_windows(state, media["duration"] or float(plan.get("duration_s", 45)))
        onscreen = [t for w in wins if w["start"] < q.HOOK_T for t in w["shot"].get("on_screen_text", [])]
        onscreen += [it["text"] for it in media["ocr"].get("hook", [])]
        checks.append(q.check_hook(plan.get("topic_keywords", []), (state.get("voice") or {}).get("words", []), onscreen))
        from agents.scriptwriter import topic_phrase

        checks += q.check_facts(script, state.get("facts") or [], storyboard, state.get("contacts") or [],
                                topic_phrase((state.get("brief") or {}).get("topic", "")))
        checks += q.check_must_include(brief, state.get("coverage") or [], script)
        checks += q.check_must_avoid(brief, script, storyboard)
        checks += q.check_script_basics(script, plan, storyboard)
        disp = " \n".join(s.display for s in Script.model_validate(script).sentences())
        ocr_all = " \n".join(it["text"] for items in media["ocr"].values() for it in items)
        osd = " \n".join(t for sh in storyboard.get("shots", []) for t in sh.get("on_screen_text", []))
        src_urls = [u for x in (state.get("sources") or []) for u in (x.get("url"), x.get("final_url"))]
        src_urls += [f.get("url", "") for f in state.get("facts") or []]
        checks.append(q.check_contacts({"script": disp, "on_screen": osd, "ocr": ocr_all}, state.get("contacts") or [],
                                       src_urls))
        checks += q.check_pack_rules(_code_rule_ids(plan.get("rule_packs", [])), script, storyboard, state.get("facts") or [])
    else:
        checks.append(q.qc("script_present", False, "scriptwriter", value="missing", threshold="script"))

    if media.get("duration"):
        checks += _end_card(ctx, state, media, settings, script, storyboard)

    from core.graph import image_is_current

    shots_by_id = {str(s["shot_id"]): s for s in (state.get("storyboard") or {}).get("shots", [])}
    in_use = {sid: a for sid, a in (state.get("shot_assets") or {}).items()     # stale images are not in the video
              if sid in shots_by_id and image_is_current(shots_by_id[sid], a)}
    checks += q.check_shot_images(in_use, float(models.get("qa", {}).get("nsfw", {}).get("max_score", 0.2)))
    checks.append(q.check_wer(state.get("voice") or {}, float(models.get("voice", {}).get("wer_max", 0.08))))
    checks.append(q.check_licences(_manifest_rows(ctx), _licence_predicate()))
    return checks, media


def _extract_frames(ctx: RunContext, log: RunLogger, video: str, wins: list[dict], d: float, loop: int, media: dict) -> None:
    fdir = ctx.out_dir / "qa_frames" / f"loop{loop}"
    want = {f"shot{w['shot_id']}": (w["start"] + w["end"]) / 2 for w in wins}
    want.update({"hook": min(1.5, d / 2), "end_a": max(0.0, d - 3.0), "end_b": max(0.0, d - 1.0)})
    with log.call("critic", "ffmpeg:frames") as rec:
        for k, t in want.items():
            try:
                media["frames"][k] = str(q.extract_frame(video, t, fdir / f"{k}.png"))
            except Exception:  # noqa: BLE001
                pass
        rec["units"] = {"frames": len(media["frames"])}
        rec["output_path"] = str(fdir)
    if media["frames"] and q.ocr_available():
        with log.call("critic", "ocr:rapidocr") as rec:
            for k, p in media["frames"].items():
                try:
                    media["ocr"][k] = q.ocr_image(p)
                except Exception:  # noqa: BLE001
                    media["ocr"][k] = []
            rec["units"] = {"frames": len(media["ocr"])}
    media["ocr_available"] = bool(media["ocr"])


def _contrast(media: dict, layout: Any, width: int, height: int) -> list[QACheck]:
    boxes = [b for b in q.layout_boxes(layout) if str(b.get("kind", b.get("type", ""))).startswith("caption")]
    cap = boxes[0] if boxes else {}
    box = q._box_xywh(cap) if cap else (0, 1150 * height / 1920, width, 350 * height / 1920)
    style = (media.get("scene") or {}).get("caption_style") or {}
    text_rgb = q.hex_to_rgb(cap.get("color", style.get("color", "#FFFFFF")))
    stroke_hex = cap.get("stroke_color") or style.get("outline_color") or ("#000000" if style.get("outline_px") else "")
    stroke = q.hex_to_rgb(stroke_hex) if stroke_hex else None
    stroke_px = cap.get("stroke_px", style.get("outline_px", 0))
    samples = []
    for k, p in media.get("frames", {}).items():
        if not k.startswith("shot"):
            continue
        try:
            bg = q.sample_background(p, box, text_rgb, stroke)
            samples.append({"frame": p, "text_rgb": text_rgb, "bg_rgb": bg, "stroke_rgb": stroke,
                            "stroke_px": stroke_px})
        except Exception:  # noqa: BLE001
            continue
    return [q.check_contrast(samples)]


def _end_card(ctx: RunContext, state: AgentState, media: dict, settings: dict, script: dict, storyboard: dict) -> list[QACheck]:
    scene = media.get("scene") or {}
    brand = scene.get("brand") or {}
    disclosure = brand.get("disclosure") or settings.get("disclosure_line", "Created with AI agents — LY Agentic AI Lab")
    ec = [s for s in storyboard.get("shots", []) if s.get("template") == "end_card"]
    cta = [t for s in ec for t in s.get("on_screen_text", []) if q.fuzzy_contains(disclosure, t) < q.FUZZY_MIN]
    if not cta:
        cta = [(script.get("cta") or {}).get("display", "")]
    end_frames = [media["frames"][k] for k in ("end_a", "end_b") if k in media["frames"]]
    ocr_text = " \n".join(it["text"] for k in ("end_a", "end_b") for it in media["ocr"].get(k, []))
    # declared fallback: end-card shot slots + brand disclosure from scene.json
    end_shots = [s for s in scene.get("shots", []) if s.get("template") == "end_card"]
    blob = q.json_strings([s.get("slots", {}) for s in end_shots]) + "\n" + (brand.get("disclosure", "") if end_shots else "")
    logo_rel = brand.get("logo_path") or settings.get("logo_path", "assets/official/logo.svg")
    logo_p = Path(logo_rel) if Path(logo_rel).is_absolute() else ROOT / logo_rel
    declared_logo = bool(end_shots) and logo_p.is_file()
    logo = q.match_logo(end_frames, logo_p) if end_frames else None     # None: svg-only logo or no cv2
    placeholder = "placeholder" in str(brand.get("logo_path") or "") or not (ROOT / settings.get("logo_path", "")).is_file()
    return q.check_end_card(ocr_text, disclosure, cta, logo=logo, declared_blob=blob,
                            declared_logo=declared_logo, ocr_available=media.get("ocr_available", False),
                            placeholder_logo=placeholder)


# ================================================================ soft checks
def _default_soft_fixture(_vars: dict) -> dict:
    return {"passed": True, "violations": [], "responsible_agent": "scriptwriter", "fix_instructions": ""}


def run_soft_checks(ctx: RunContext, log: RunLogger, llm: LLM, state: AgentState, media: dict,
                    degrade: bool = False) -> list[QACheck]:
    loop = int(state.get("loop", 0) or 0)
    out: list[QACheck] = []
    out.append(_soft_review(ctx, llm, state, loop))
    out += _frame_judge(ctx, log, llm, state, media, loop, degrade)
    out += _nli(ctx, log, state)
    return out


def _soft_review(ctx: RunContext, llm: LLM, state: AgentState, loop: int) -> QACheck:
    plan = state.get("plan") or {}
    if llm.mock and "critic_soft_v1" not in llm.fixtures:
        llm.fixtures["critic_soft_v1"] = _default_soft_fixture
    script = state.get("script") or {}
    sents = [{"i": i, "display": s.display, "fact_ids": s.fact_ids}
             for i, s in enumerate(Script.model_validate(script).sentences())] if script else []
    facts = [{"fact_id": f["fact_id"], "claim": f.get("claim", ""), "quote": f.get("quote", ""), "url": f.get("url", "")}
             for f in state.get("facts") or []]
    try:
        r = llm.structured("critic", "critic_soft_v1", SoftReviewOut, creative=False,
                           seed=stage_seed(ctx.seed, "critic", "soft", loop),
                           brief=state.get("brief") or {}, rules=rules_text(plan.get("rule_packs", ["base"])),
                           script=sents, storyboard=(state.get("storyboard") or {}).get("shots", []), facts=facts,
                           coverage=state.get("coverage") or [])
    except Exception as e:  # noqa: BLE001  (infra failure must not loop the scriptwriter)
        return q.qc("soft_review", True, "scriptwriter", value="skipped", threshold="LLM verdict", severity="soft",
                    evidence=f"LLM unavailable: {e.__class__.__name__}: {e}"[:300])
    agent = r.responsible_agent if r.responsible_agent in SOFT_AGENTS else "scriptwriter"
    return q.qc("soft_review", r.passed and not r.violations, agent, value="; ".join(r.violations)[:600] or "ok",
                threshold="no rule violations", evidence=r.fix_instructions, severity="soft")


def _frame_judge(ctx: RunContext, log: RunLogger, llm: LLM, state: AgentState, media: dict, loop: int,
                 degrade: bool) -> list[QACheck]:
    frames = media.get("frames") or {}
    shots = [k for k in frames if k.startswith("shot")]
    skip = ""
    if llm.mock or ctx.mock_llm:
        skip = "mock LLM: VLM frame judge skipped"
    elif not shots:
        skip = "no frames extracted"
    elif degrade and loop > 0:
        skip = "time budget degrade: no VLM re-checks"
    if skip:
        return [q.qc("frame_judge", True, "asset_generator", value="skipped", threshold="VLM verdict per shot",
                     evidence=skip, severity="soft")]
    try:
        from core.servers import start_vlm, stop_vlm
    except ImportError as e:
        return [q.qc("frame_judge", True, "asset_generator", value="skipped", threshold="VLM verdict",
                     evidence=f"core.servers unavailable: {e}", severity="soft")]
    by_id = {str(w["shot_id"]): w["shot"] for w in media.get("windows", [])}
    out: list[QACheck] = []
    try:
        start_vlm(ctx, log)
        for k in [*shots, "end_b"]:
            if k not in frames:
                continue
            sid = k[4:] if k.startswith("shot") else "end_card"
            sh = by_id.get(sid, {"template": "end_card", "visual_prompt": "end card: official logo, disclosure line, CTA",
                                 "on_screen_text": []})
            try:
                v = llm.structured("critic", "frame_judge_v1", FrameJudgeOut, images=[frames[k]],
                                   seed=stage_seed(ctx.seed, "critic", f"frame{sid}", loop),
                                   template=sh.get("template", ""), intent=sh.get("visual_prompt", ""),
                                   on_screen_text=sh.get("on_screen_text", []))
            except Exception as e:  # noqa: BLE001
                out.append(q.qc(f"frame_judge:{sid}", True, "asset_generator", value="skipped", severity="soft",
                                evidence=f"VLM error {e.__class__.__name__}"))
                continue
            bad = [n for n, f in (("realistic face", v.realistic_human_face), ("garbled text", v.garbled_text),
                                  ("fear/victim imagery", v.fear_or_victim_imagery), ("off-intent", not v.matches_intent)) if f]
            agent = "editor" if sid == "end_card" else "asset_generator"
            hard = v.realistic_human_face or v.fear_or_victim_imagery
            out.append(q.qc(f"frame_judge:{sid}", not bad, agent, value=", ".join(bad) or "ok",
                            threshold="no face/garbled/fear; matches intent", evidence=f"{Path(frames[k]).name}: {v.note}",
                            severity="hard" if hard else "soft"))
    except Exception as e:  # noqa: BLE001
        out.append(q.qc("frame_judge", True, "asset_generator", value="skipped", severity="soft",
                        evidence=f"VLM server unavailable: {e.__class__.__name__}: {e}"[:300]))
    finally:
        try:
            stop_vlm(ctx, log)
        except Exception:  # noqa: BLE001
            pass
    return out


def _nli(ctx: RunContext, log: RunLogger, state: AgentState) -> list[QACheck]:
    script = state.get("script") or {}
    facts = {f["fact_id"]: f for f in state.get("facts") or []}
    if not script or not facts:
        return []
    pairs, idx = [], []
    for i, s in enumerate(Script.model_validate(script).sentences()):
        quotes = [facts[f]["quote"] for f in s.fact_ids if f in facts and facts[f].get("quote")]
        if quotes:
            pairs.append([" ".join(quotes), s.display])
            idx.append((i, s))
    if not pairs:
        return []
    thr = float(_models().get("retrieval", {}).get("nli", {}).get("threshold", 0.5))
    if not (ROOT / "workers" / "retrieval_models.py").exists():
        return [q.qc("nli", True, "scriptwriter", value="skipped", threshold=f">= {thr}", severity="soft",
                     evidence="workers/retrieval_models.py not available")]
    try:
        from core.workers import run_worker

        res = run_worker(ctx, log, agent="critic", worker="retrieval_models.py", gpu="1",
                         request={"task": "nli", "pairs": pairs, "mock": ctx.mock_media,
                                  "seed": stage_seed(ctx.seed, "critic", "nli", int(state.get("loop", 0) or 0))})
        scores = [float(x) for x in res.get("scores", [])]
    except Exception as e:  # noqa: BLE001
        return [q.qc("nli", True, "scriptwriter", value="skipped", threshold=f">= {thr}", severity="soft",
                     evidence=f"NLI worker error: {e.__class__.__name__}: {e}"[:300])]
    out = []
    for (i, s), sc in zip(idx, scores):
        ok = sc >= thr or ctx.mock_media
        out.append(q.qc(f"nli:s{i}", ok, "scriptwriter", value=f"{sc:.3f}", threshold=f">= {thr}",
                        evidence=f"'{s.display[:90]}' vs {s.fact_ids}" + (" (mock scores not enforced)" if ctx.mock_media else "")))
    return out


# ================================================================ mock mode
MOCK_UNENFORCED = ("caption_contrast", "end_card_", "dead_air", "black_frames", "frozen_frames", "loudness_",
                   "audio_nan", "voice_wer", "licences", "caption_safe_zone", "faces:", "nsfw:")


def relax_for_mock(checks: list[QACheck]) -> list[QACheck]:
    """mock_media: placeholder images/voice cannot meet content checks; record them but do not enforce.
    Container spec (codec/size/fps/audio format/duration), script, facts and contacts checks stay enforced."""
    out = []
    for c in checks:
        if not c.passed and c.check_id.startswith(MOCK_UNENFORCED):
            c = c.model_copy(update={"passed": True, "evidence": f"[mock media, not enforced: value={c.value}] {c.evidence}"[:1500]})
        out.append(c)
    return out


# ================================================================ node
def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    from core.budget import Budget
    from core.graph import MAX_LOOPS

    def node(state: AgentState) -> dict:
        loop = int(state.get("loop", 0) or 0)
        budget = Budget(ctx, state.get("plan"))
        degrade = budget.should_degrade()
        checks, media = run_hard_checks(ctx, log, state)
        if ctx.mock_media:
            checks = relax_for_mock(checks)
        if all(c.passed for c in checks if c.severity == "hard"):
            checks += run_soft_checks(ctx, log, llm, state, media, degrade)
        report = QAReport(run_id=ctx.run_id, loop=loop, passed=all(c.passed for c in checks), checks=checks)
        ctx.write_json(f"qa_report_loop{loop}.json", report)
        ctx.write_json("qa_report.json", report)
        cap = min(MAX_LOOPS, int(_settings().get("max_loops", MAX_LOOPS)))
        out = decide(state, report, max_loops=budget.loops_allowed(loop, cap))
        out["qa_report"] = report.model_dump(mode="json")
        out.setdefault("extra", {})["qa_summary"]["budget"] = budget.snapshot()
        log.record(agent="critic", tool="route", status="ok", output_path=str(ctx.artifact("qa_report.json")),
                   units={"loop": loop, "route": out["route"], "passed": report.passed,
                          "failures": [c.check_id for c in report.failures()][:40]})
        return out

    return node
