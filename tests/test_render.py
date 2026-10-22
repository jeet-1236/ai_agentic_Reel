"""Render tests: scene building (editor), both renderers (HyperFrames + ffmpeg fallback), delivery spec probe.

Fixtures are synthetic: placeholder illustrations (Pillow), a speech-like WAV (numpy), word timings and
caption pages. Real renders take ~10-60 s each; HyperFrames tests skip when its Chrome is not installed
(`cd renderer && npx hyperframes browser ensure`).
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from agents import editor
from core import ffmpeg_utils as ff
from core.context import ROOT
from core.scene import Scene, load_template_spec, safe_zones, template_names

HF_BIN = ROOT / "renderer" / "node_modules" / "hyperframes" / "bin" / "hyperframes.mjs"
RENDER_STATS: dict[str, dict] = {}


# ---------------------------------------------------------------- synthetic inputs
def speech_like_wav(path: Path, sentences: list[tuple[float, float]], total: float, sr: int = 24000) -> Path:
    """Mono 24 kHz 'voice': harmonic bursts (syllables at ~4 Hz) inside each sentence span, silence between."""
    t = np.arange(int(total * sr)) / sr
    sig = np.zeros_like(t)
    for a, b in sentences:
        m = (t >= a) & (t < b)
        tt = t[m] - a
        f0 = 140 + 25 * np.sin(2 * math.pi * 0.7 * tt)
        ph = 2 * math.pi * np.cumsum(f0) / sr
        tone = sum(np.sin(k * ph) / k for k in range(1, 6))
        env = np.clip(np.sin(2 * math.pi * 4.0 * tt) * 0.5 + 0.5, 0, 1) ** 1.5
        rng = np.random.default_rng(int(a * 1000))
        hiss = np.convolve(rng.standard_normal(m.sum()), np.ones(3) / 3, mode="same")  # fricative-ish noise
        sig[m] = tone * env + 0.35 * hiss * (1 - env)
    sig = 0.25 * sig / (np.abs(sig).max() or 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((sig * 32767).astype("<i2").tobytes())
    return path


def placeholder_image(path: Path, w: int, h: int, hue: int) -> Path:
    """Flat-vector-ish placeholder illustration (no people)."""
    img = Image.new("RGB", (w, h), (40 + hue % 60, 70, 110))
    d = ImageDraw.Draw(img)
    for i in range(9):
        r = (i * 97 + hue) % 360
        d.ellipse((w * (i % 3) / 3, h * (i // 3) / 3, w * (i % 3 + 1.4) / 3, h * (i // 3 + 1.2) / 3),
                  fill=(120 + r % 120, 80 + (r * 3) % 150, 90 + (r * 7) % 140))
    d.rectangle((w * 0.2, h * 0.55, w * 0.8, h * 0.62), fill=(240, 220, 120))
    img.save(path)
    return path


def make_state(tmp: Path, aspect: str = "9x16", shots: list[dict] | None = None, sentences: list[str] | None = None,
               target_s: int = 8) -> dict:
    """AgentState-like dict with everything the editor reads. Default: 2 shots (+ end card), ~8 s."""
    tmp.mkdir(parents=True, exist_ok=True)
    w, h = (1080, 1920) if aspect == "9x16" else (1920, 1080)
    sentences = sentences or ["AI in your exams?", "Only what the examiner allows.", "Check the official page."]
    # word timings: 0.32 s per word, 0.25 s pause between sentences
    words, spans, t = [], [], 0.1
    for si, s in enumerate(sentences):
        a = t
        for wd in s.split():
            words.append({"word": wd, "start": round(t, 3), "end": round(t + 0.28, 3), "sentence_index": si})
            t += 0.32
        spans.append((round(a, 3), round(t - 0.04, 3)))
        t += 0.25
    vo = round(t, 3)
    wav = speech_like_wav(tmp / "voice.wav", spans, vo)
    pages, cur = [], []
    for wd in words:   # <= 4 words per page, page per sentence
        if cur and (len(cur) == 4 or cur[-1]["sentence_index"] != wd["sentence_index"]):
            pages.append(cur)
            cur = []
        cur.append(wd)
    pages.append(cur)
    captions = [{"index": i, "start": p[0]["start"], "end": (pages[i + 1][0]["start"] if i + 1 < len(pages) else p[-1]["end"] + 0.2),
                 "lines": [" ".join(x["word"] for x in p)], "words": p} for i, p in enumerate(pages)]
    shots = shots or [
        {"shot_id": 1, "template": "hook_title", "sentence_indices": [0], "visual_prompt": "exam hall, flat vector",
         "on_screen_text": ["AI in exams?", "What counts as unfair means"], "fact_ids": []},
        {"shot_id": 2, "template": "step_list", "sentence_indices": [1, 2], "visual_prompt": "",
         "on_screen_text": ["Before you start:", "Check what's allowed", "Ask your examiner"], "fact_ids": ["F1"]},
    ]
    assets = {}
    for s in shots:
        if s.get("visual_prompt"):
            iw, ih = (768, 1344) if aspect == "9x16" else (1344, 768)
            assets[str(s["shot_id"])] = {"shot_id": s["shot_id"], "visual_prompt": s["visual_prompt"],
                                         "image_path": str(placeholder_image(tmp / f"img_{s['shot_id']}.png", iw, ih, s["shot_id"] * 40))}
    sent = [{"display": s, "spoken": s, "fact_ids": []} for s in sentences]
    return {
        "brief": {"id": "T01", "format": "video_16x9" if aspect == "16x9" else "video_9x16", "duration_s": 30},
        "plan": {"brief_id": "T01", "aspect": aspect, "width": w, "height": h, "fps": 30, "duration_s": target_s},
        "script": {"hook": {"style": "question", **sent[0]}, "hook_index": 0, "body": sent[1:-1], "cta": sent[-1],
                   "word_count": len(words)},
        "storyboard": {"shots": shots},
        "shot_assets": assets,
        "voice": {"wav_path": str(wav), "duration_s": vo, "voice": "test", "sentence_spans": spans, "words": words},
        "captions": captions,
        "facts": [{"fact_id": "F1", "claim": "c", "quote": "q", "chunk_id": "S1#c1", "source_id": "S1",
                   "url": "https://www.somaiya.edu/en/center-of-examination/"}],
        "contacts": [{"kind": "url", "value": "https://www.somaiya.edu/en/center-of-examination", "display": "somaiya.edu/en/center-of-examination",
                      "label": "CoE", "source_ids": ["S1"], "authority": "somaiya"},
                     {"kind": "phone", "value": "+9118001805522", "display": "1800-180-5522", "label": "Anti-Ragging Helpline",
                      "source_ids": ["S2"], "authority": "gov"}],
        "srt_path": "",
    }


ALL_TEMPLATE_SHOTS = [
    {"shot_id": 1, "template": "hook_title", "sentence_indices": [0], "visual_prompt": "v", "on_screen_text": ["AI in exams?", "What counts as unfair means"], "fact_ids": []},
    {"shot_id": 2, "template": "illustration_kenburns", "sentence_indices": [1], "visual_prompt": "v", "on_screen_text": ["Software tools prohibited by the examiner"], "fact_ids": []},
    {"shot_id": 3, "template": "do_dont_split", "sentence_indices": [2], "visual_prompt": "", "on_screen_text": ["Do: ask your instructor first", "Don't: use AI tools in exams"], "fact_ids": []},
    {"shot_id": 4, "template": "step_list", "sentence_indices": [3], "visual_prompt": "", "on_screen_text": ["If unsure:", "Read the circular", "Ask before you use", "Cite what you used"], "fact_ids": []},
    {"shot_id": 5, "template": "stat_callout", "sentence_indices": [4], "visual_prompt": "", "on_screen_text": ["2 committees", "CPUMC then UPUMC review cases"], "fact_ids": ["F1"]},
    {"shot_id": 6, "template": "contact_card", "sentence_indices": [5], "visual_prompt": "", "on_screen_text": ["Need help?", "Call 1800-180-5522"], "fact_ids": []},
    {"shot_id": 7, "template": "quote_card", "sentence_indices": [6], "visual_prompt": "", "on_screen_text": ["any type of evaluation theory, practical, oral"], "fact_ids": ["F1"]},
]
ALL_TEMPLATE_SENTENCES = ["Can you use AI in exams?", "Software tools need the examiner's okay.", "Ask first, never assume.",
                          "When unsure, follow three steps.", "Two committees review cases.", "Help is one call away.",
                          "The rules cover every evaluation.", "Check the official page."]


# ---------------------------------------------------------------- helpers
def hf_ready() -> bool:
    if not HF_BIN.exists() or not shutil.which("node"):
        return False
    r = subprocess.run(["node", str(HF_BIN), "browser", "path"], capture_output=True, text=True, timeout=60)
    lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    return bool(lines) and Path(lines[-1]).exists()


def check_layout(layout: dict, scene: Scene) -> None:
    safe = scene.safe
    assert layout["boxes"], "layout has no boxes"
    for b in layout["boxes"]:
        if b["kind"] == "logo":
            continue
        assert safe.text.contains(b, tol=1), f"{b['kind']} box outside safe zone: {b}"
        assert b["font_px"] >= 48, f"text below 48 px: {b}"
        if scene.aspect == "9x16":
            assert 250 <= b["y0"] and b["y1"] <= 1570, b
        if b["kind"] == "caption":
            assert safe.caption.contains(b, tol=1), f"caption outside band: {b}"
            assert b["lines"] <= 2 and b["stroke_px"] >= 6 and b["color"] == "#FFFFFF"
    kinds = {b["kind"] for b in layout["boxes"]}
    assert {"text", "caption", "logo"} <= kinds
    texts = " ".join(b["text"] for b in layout["boxes"])
    assert "LY Agentic AI Lab" in texts   # disclosure on the end card


def check_spec(video: Path, scene: Scene) -> dict:
    s = ff.probe(video)["summary"]
    assert (s["width"], s["height"]) == (scene.width, scene.height)
    assert s["v_codec"] == "h264" and s["v_profile"] == "High" and s["pix_fmt"] == "yuv420p"
    assert s["r_frame_rate"] == "30/1" and s["avg_frame_rate"] == "30/1"
    assert s["a_codec"] == "aac" and s["sample_rate"] == 48000 and s["channels"] == 2
    # QA gate is 128-192 kb/s; an 8 s test clip is ~35 % silent end card and AAC cannot spend bits on silence,
    # so the tiny clips only get a floor here (the gate itself is tested on 30 s in test_aac_bitrate_gate).
    assert 96_000 <= s["a_bit_rate"] <= 192_000
    assert abs(s["duration"] - scene.duration_s) < 0.1
    loud = ff.measure_loudness(video)
    assert abs(loud["I"] - (-14)) <= 1.0, loud
    assert loud["TP"] <= -1.0, loud
    return {**s, **{f"loud_{k}": v for k, v in loud.items()}}


def render_and_finish(ctx, log, state: dict, engine: str, tag: str) -> dict:
    ctx.renderer = engine
    scene, warns = editor.build_scene(ctx, state)
    assert not scene.validate_scene(), scene.validate_scene()
    scene_path = ctx.artifact("scene.json")
    scene.save(scene_path)
    out = editor.render(ctx, log, scene_path, scene, engine)
    final = ctx.out_dir / f"{tag}_final.mp4"
    post = editor.post_process(ctx, log, scene, out["video_path"], final)
    layout = json.loads(Path(out["layout_path"]).read_text(encoding="utf-8"))
    check_layout(layout, scene)
    spec = check_spec(final, scene)
    resp = sorted((ctx.out_dir / "workers").glob("*.out.json"))[-1]
    units = json.loads(resp.read_text(encoding="utf-8"))["units"]
    RENDER_STATS[tag] = {"frames": scene.total_frames, **{k: units.get(k) for k in ("render_s", "render_fps", "capture_s", "capture_fps")},
                         "probe": spec, "loudnorm": post["loudnorm"]["pass2"].get("normalization_type", "")}
    print(f"\n[{tag}] {json.dumps(RENDER_STATS[tag], default=str)}")
    return {"scene": scene, "layout": layout, "final": final, "warnings": warns}


# ---------------------------------------------------------------- fast tests
def test_templates_have_specs():
    names = template_names()
    assert set(names) == {"hook_title", "illustration_kenburns", "do_dont_split", "step_list", "stat_callout",
                          "contact_card", "quote_card", "end_card"}
    for n in names:
        spec = load_template_spec(n)
        assert {"9x16", "16x9"} <= set(spec["aspects"]) and spec["slots_schema"]["type"] == "object"
        d = ROOT / "renderer" / "templates" / n
        assert (d / "index.html").exists() and (d / "timeline.js").exists()
        assert "Math.random" not in (d / "timeline.js").read_text(encoding="utf-8")


def test_safe_zones_inside_handout_band():
    z = safe_zones(1080, 1920)
    assert z.text.y0 == 250 and z.text.y1 == 1570
    assert (z.caption.y0, z.caption.y1) == (1150, 1500)
    for b in (z.content, z.caption, z.end):
        assert z.text.contains(b)
    z2 = safe_zones(1920, 1080)
    for b in (z2.content, z2.caption, z2.end):
        assert z2.text.contains(b)


@pytest.mark.parametrize("aspect", ["9x16", "16x9"])
def test_build_scene_all_templates(ctx, tmp_path, aspect):
    state = make_state(tmp_path, aspect, shots=ALL_TEMPLATE_SHOTS, sentences=ALL_TEMPLATE_SENTENCES, target_s=30)
    scene, warns = editor.build_scene(ctx, state, settings={"logo_path": "", "disclosure_line": "Created with AI agents — LY Agentic AI Lab"})
    assert not scene.validate_scene(), scene.validate_scene()
    assert [s.template for s in scene.shots] == [s["template"] for s in ALL_TEMPLATE_SHOTS] + ["end_card"]
    assert not any("slots invalid" in w for w in warns), warns
    assert not any("does not fit" in w for w in warns), warns
    end = scene.shots[-1]
    assert 3.0 <= end.duration <= 5.0 and end.start >= state["voice"]["duration_s"]
    assert end.slots["disclosure"] == "Created with AI agents — LY Agentic AI Lab"
    assert end.slots.get("url") == "somaiya.edu/en/center-of-examination"
    cc = next(s for s in scene.shots if s.template == "contact_card")
    assert cc.slots["contacts"][0]["value"] == "1800-180-5522"
    dd = next(s for s in scene.shots if s.template == "do_dont_split")
    assert dd.slots["dont_items"] and dd.slots["do_items"]
    for a, b in zip(scene.shots, scene.shots[1:]):
        assert a.end == b.start
    assert all(s.kenburns for s in scene.shots if s.image)
    assert all(len(c.lines) <= 2 and 48 <= c.px <= 64 for c in scene.captions)


def test_aac_bitrate_gate(tmp_path):
    """30 s voice + 4 s end-card silence -> measured AAC bitrate inside the 128-192 kb/s gate, I = -14 LUFS."""
    spans = [(0.2 + 2.6 * k, 2.4 + 2.6 * k) for k in range(10)]
    wav = speech_like_wav(tmp_path / "v.wav", spans, 26.5)
    norm = tmp_path / "n.wav"
    ln = ff.loudnorm_two_pass(wav, norm, duration=30.5)
    rates = ff.encode_aac(norm, tmp_path / "a.m4a", duration=30.5)
    assert 128_000 <= rates["measured_bps"] <= 192_000, rates
    assert ln["pass2"]["normalization_type"] == "linear"
    assert abs(ff.measure_loudness(tmp_path / "a.m4a")["I"] + 14) <= 1.0


def test_shot_times_interpolates_and_keeps_min():
    shots = [{"sentence_indices": [0]}, {"sentence_indices": []}, {"sentence_indices": [2]}]
    t = editor.shot_times(shots, [(0, 1), (1, 2), (2.0, 3.0)], 6.0)
    assert t[0][0] == 0 and t[-1][1] == 6.0 and t[1][0] == pytest.approx(1.0, abs=0.04)
    assert all(b - a >= 0.99 for a, b in t)


# ---------------------------------------------------------------- renders
def test_render_ffmpeg_9x16(ctx, log, tmp_path):
    ctx.mock_media = False
    r = render_and_finish(ctx, log, make_state(tmp_path / "in"), "ffmpeg", "ffmpeg_9x16")
    assert 6.0 <= r["scene"].duration_s <= 8.5


def test_render_ffmpeg_16x9(ctx, log, tmp_path):
    ctx.mock_media = False
    render_and_finish(ctx, log, make_state(tmp_path / "in", "16x9"), "ffmpeg", "ffmpeg_16x9")


@pytest.mark.skipif(not hf_ready(), reason="HyperFrames Chrome not installed (npx hyperframes browser ensure)")
def test_render_hyperframes_9x16(ctx, log, tmp_path):
    ctx.mock_media = False
    r = render_and_finish(ctx, log, make_state(tmp_path / "in"), "hyperframes", "hyperframes_9x16")
    assert 6.0 <= r["scene"].duration_s <= 8.5


@pytest.mark.skipif(not hf_ready(), reason="HyperFrames Chrome not installed (npx hyperframes browser ensure)")
def test_render_hyperframes_16x9(ctx, log, tmp_path):
    ctx.mock_media = False
    render_and_finish(ctx, log, make_state(tmp_path / "in", "16x9"), "hyperframes", "hyperframes_16x9")


def test_editor_node_mock_media(ctx, log, llm, tmp_path, monkeypatch):
    """Full node in mock mode (placeholder render, no Chrome): scene.json + final mp4 + srt copy + spec."""
    state = make_state(tmp_path / "in")
    srt = tmp_path / "in" / "c.srt"
    srt.write_text("1\n00:00:00,100 --> 00:00:01,000\nCan you\n", encoding="utf-8")
    state["srt_path"] = str(srt)
    ctx.mock_media = True
    out = editor.make_node(ctx, log, llm)(state)
    final = Path(out["video_path"])
    assert final.name == f"{ctx.final_basename('T01')}.mp4" and final.with_suffix(".srt").exists()
    assert Path(out["scene_path"]).exists() and out["extra"]["render_engine"] == "hyperframes-mock"
    check_spec(final, Scene.load(out["scene_path"]))
    lines = [json.loads(x) for x in (ctx.logs_dir / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(x["tool"] == "ffmpeg" for x in lines) and any(x["tool"].startswith("worker:render") for x in lines)
