"""AssetGenerator / media workers / captions / manifest (all offline, mock media)."""
from __future__ import annotations

import threading
import wave
from pathlib import Path

import pytest
from PIL import Image

from core.captions import (align_display_words, check_pages, normalize_for_wer, paginate, srt_timestamp, to_srt,
                           wer, wrap_lines)
from core.manifest import add_asset, licence_allowed, licence_failures, read_manifest
from core.schemas import AssetRecord, HookOut, Script, ScriptSentenceOut, WordTiming
from core.workers import run_worker

PLAN = {"category": "C", "rule_packs": ["base", "C"], "aspect": "9x16", "width": 1080, "height": 1920,
        "topic_keywords": ["AI tools", "exams"], "duration_s": 40}


def _words(sentences: list[str], rate: float = 2.4, pause: float = 0.25) -> list[WordTiming]:
    out, t = [], 0.0
    for si, s in enumerate(sentences):
        for w in s.split():
            out.append(WordTiming(word=w, start=round(t, 3), end=round(t + 0.9 / rate, 3), sentence_index=si))
            t += 1 / rate
        t += pause
    return out


def _script() -> dict:
    return Script(
        hook=HookOut(style="question", display="Can you use AI in your KJSCE exams?",
                     spoken="Can you use AI in your K J S C E exams?", fact_ids=[]),
        hook_index=0,
        body=[ScriptSentenceOut(display="Unfair means include any software tool the examiner has not allowed.",
                                spoken="Unfair means include any software tool the examiner has not allowed.",
                                fact_ids=["F1"]),
              ScriptSentenceOut(display="Fines can reach ₹2,000 at Somaiya.",
                                spoken="Fines can reach two thousand rupees at Somaiya.", fact_ids=["F2"])],
        cta=ScriptSentenceOut(display="Ask your instructor before you use AI.",
                              spoken="Ask your instructor before you use AI.", fact_ids=[]),
        word_count=30,
    ).model_dump(mode="json")


# ---------------------------------------------------------------- captions
def test_wrap_lines_limits():
    assert wrap_lines(["short", "line"]) == ["short line"]
    lines = wrap_lines("Unfair means include any software tool the examiner".split())
    assert lines and len(lines) == 2 and all(len(x) <= 32 for x in lines)
    assert wrap_lines(["x" * 20] * 4) is None


def test_paginate_rules_and_sentence_boundaries():
    sents = ["Can you use AI tools in your college exams and assignments this semester?",
             "Short one.",
             "Anything the examiner has not allowed, a website, a file or a software tool, counts as unfair means."]
    words = _words(sents)
    pages = paginate(words, end_time=words[-1].end + 0.5)
    assert pages and check_pages(pages) == []
    for p in pages:
        assert len({w.sentence_index for w in p.words}) == 1
        assert len(p.lines) <= 2 and all(len(x) <= 32 for x in p.lines)
    # every word appears exactly once, in order
    assert [w.word for p in pages for w in p.words] == [w.word for w in words]


def test_srt_output():
    assert srt_timestamp(3725.5) == "01:02:05,500"
    pages = paginate(_words(["Hello there students.", "Second sentence here."]))
    srt = to_srt(pages)
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert srt.count(" --> ") == len(pages)


def test_align_display_words_handles_spoken_forms():
    display = ["Pay ₹2,000 at KJSCE today."]
    toks, t = [], 0.0
    for w in "Pay two thousand rupees at K J S C E today".split():
        toks.append({"text": w, "start": t, "end": t + 0.3, "sentence_index": 0})
        t += 0.4
    out = align_display_words(display, toks, [(0.0, t)])
    assert [w.word for w in out] == ["Pay", "₹2,000", "at", "KJSCE", "today."]
    money, kj = out[1], out[3]
    assert money.start == pytest.approx(0.4) and money.end == pytest.approx(1.5)   # two..rupees
    assert kj.start == pytest.approx(2.0) and kj.end == pytest.approx(3.9)         # K..E
    assert all(a.start <= b.start for a, b in zip(out, out[1:]))


def test_align_fallback_and_long_url():
    out = align_display_words(["Visit somaiya.edu/en/center-of-examination-and-evaluation now"], [], [(1.0, 4.0)])
    assert all(len(w.word) <= 32 for w in out)
    assert out[0].start == pytest.approx(1.0) and out[-1].end == pytest.approx(4.0)


def test_wer_normalisation():
    assert wer("Call one nine three zero now", "Call 1930 now.") == 0.0
    assert wer("Fines reach two thousand rupees", "Fines reach 2,000 rupees") == 0.0
    assert wer("Study at K J S C E", "Study at KJSCE.") == 0.0
    assert wer("a b c d", "") == 1.0
    assert 0 < wer("ask your instructor first", "ask your teacher first") < 0.5
    assert normalize_for_wer("50% & more") == ["fifty", "percent", "and", "more"]


# ---------------------------------------------------------------- manifest
def test_manifest_licence_check(ctx, tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(b"png")
    assert licence_allowed("Apache-2.0") and licence_allowed("openrail++")
    assert licence_allowed("CreativeML-OpenRAIL++-M AND BSD-3-Clause")
    assert not licence_allowed("CC-BY-NC-4.0") and not licence_allowed("")
    ok = add_asset(ctx, AssetRecord(path=str(f), kind="image", generator="g", licence="MIT", sha256="a"))
    bad = add_asset(ctx, AssetRecord(path=str(f), kind="image", generator="g", licence="CC-BY-NC-4.0", sha256="b"))
    add_asset(ctx, AssetRecord(path=str(f), kind="image", generator="g", licence="MIT", sha256="a"))  # duplicate
    assert ok and not bad
    rows = read_manifest(ctx)
    assert len(rows) == 2 and rows[0]["licence_ok"] == "True"
    assert [r["sha256"] for r in licence_failures(ctx)] == ["b"]


# ---------------------------------------------------------------- lexicon
def test_lexicon_markdown():
    from workers.tts_kokoro import apply_lexicon

    lex = {"respell": {"KJSCE": "K J S C E", "Tele-MANAS": "Tele Manas"},
           "words": {"Somaiya": "/sOmˈIjə/", "Manas": "mˈʌnəs"}}
    text, n = apply_lexicon("Somaiya's KJSCE students call Tele-MANAS. [Somaiya](/x/) stays.", lex)
    assert text == ("[Somaiya](/sOmˈIjə/)'s K J S C E students call Tele [Manas](/mˈʌnəs/). "
                    "[Somaiya](/x/) stays.")
    assert n == 4


# ---------------------------------------------------------------- workers (real subprocesses, mock mode)
def test_image_worker_mock_batch_deterministic(ctx, log):
    req = {"task": "generate", "mock": True, "width": 96, "height": 160,
           "jobs": [{"prompt": "a lamp", "negative": "", "seed": 7, "n": 2, "out_prefix": "a", "label": "shot 1"},
                    {"prompt": "a book", "negative": "", "seed": 9, "n": 1, "out_prefix": "b", "label": "shot 2"}]}
    out = run_worker(ctx, log, agent="t", worker="image_sdxl.py", request=req)
    assert [len(r["paths"]) for r in out["results"]] == [2, 1]
    assert out["results"][0]["seeds"] == [7, 8]
    first = Path(out["paths"][0]).read_bytes()
    assert Image.open(out["paths"][0]).size == (96, 160)
    out2 = run_worker(ctx, log, agent="t", worker="image_sdxl.py", request=req)
    assert Path(out2["paths"][0]).read_bytes() == first
    assert out["meta"]["licence"] == "own-work"


def test_checks_and_upscale_workers_mock(ctx, log):
    p = ctx.assets_dir / "src.png"
    Image.new("RGB", (60, 100), (10, 20, 30)).save(p)
    chk = run_worker(ctx, log, agent="t", worker="image_checks.py", request={"task": "check", "mock": True, "paths": [str(p)]})
    assert chk["results"][0]["face_count"] == 0 and chk["results"][0]["nsfw_score"] == 0.0
    up = run_worker(ctx, log, agent="t", worker="upscale.py",
                    request={"task": "upscale", "mock": True, "path": str(p), "width": 121, "height": 215})
    assert Image.open(up["path"]).size == (121, 215) and up["method"] == "pillow-lanczos"


def test_upscale_real_mode_falls_back_without_torch(ctx, log):
    try:
        import torch  # noqa: F401
        pytest.skip("torch installed: real path would run")
    except ImportError:
        pass
    p = ctx.assets_dir / "src.png"
    Image.new("RGB", (60, 100), (10, 20, 30)).save(p)
    up = run_worker(ctx, log, agent="t", worker="upscale.py",
                    request={"task": "upscale", "mock": False, "path": str(p), "width": 120, "height": 120})
    assert up["method"] == "pillow-lanczos" and "ModuleNotFoundError" in up["fallback_reason"]
    assert Image.open(up["path"]).size == (120, 120)


def test_haar_face_detector_real_cv2(ctx):
    from workers.image_checks import detect_faces

    p = ctx.assets_dir / "blank.png"
    Image.new("RGB", (200, 320), (240, 240, 240)).save(p)
    name, faces = detect_faces([str(p)], None)
    assert name in ("haar", "unavailable") and faces == [[]]


def test_tts_and_asr_workers_mock(ctx, log):
    t = run_worker(ctx, log, agent="t", worker="tts_kokoro.py", request={
        "task": "synth", "mock": True, "sentences": ["Hello Somaiya students.", "Second line here, okay?"],
        "voice": "af_heart", "speed": 1.0, "seed": 3, "lexicon": {"words": {"Somaiya": "sOmˈIjə"}}})
    with wave.open(t["wav_path"]) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1
        dur = w.getnframes() / 24000
    assert t["duration_s"] == pytest.approx(dur, abs=0.01)
    assert t["sentence_spans"][1][0] >= t["sentence_spans"][0][1] + 0.24
    assert [x["text"] for x in t["tokens"] if x["sentence_index"] == 0] == ["Hello", "Somaiya", "students"]
    assert t["lexicon_applied"] == 1
    a = run_worker(ctx, log, agent="t", worker="asr_whisper.py", request={
        "task": "transcribe", "mock": True, "wav_path": t["wav_path"], "expected_text": "Hello Somaiya students.",
        "expected_words": t["tokens"]})
    assert a["text"] == "Hello Somaiya students." and a["words"][0]["word"] == "Hello"


# ---------------------------------------------------------------- agent helpers
def test_prompt_feedback_and_selection():
    from agents.asset_generator import (build_prompt, best_effort, select_candidate, shot_feedback, target_size,
                                        voice_feedback)

    assert target_size({"width": 1080, "height": 1920}) == (1210, 2150)
    assert target_size({"width": 1920, "height": 1080}) == (2150, 1210)
    style = {"prefix": "flat vector,", "suffix": "minimal", "negative": "photo"}
    p, n = build_prompt("two students at a desk with a laptop", style, no_people=True)
    assert "students" not in p and "uninhabited" in p and "people" in n
    fb = {"asset_generator": ["[faces:3] value=1 (threshold 0). Evidence: x. Fix: Use objects only.",
                              "shot 4: brighter colours", "[voice_wer] value=0.2", {"shot_id": 3, "instruction": "no hands"},
                              {"lexicon": {"Vidyavihar": "vɪdjəvɪhˈɑɹ"}}]}
    assert shot_feedback(fb, 3) == ["value=1 (threshold 0). Evidence: x. Fix: Use objects only.", "no hands"]
    assert shot_feedback(fb, 4) == ["brighter colours"]
    notes, patch = voice_feedback(fb)
    assert len(notes) == 1 and patch == {"Vidyavihar": "vɪdjəvɪhˈɑɹ"}
    res = [{"face_count": 1, "nsfw_score": 0.0}, {"face_count": 0, "nsfw_score": 0.5}, {"face_count": 0, "nsfw_score": 0.1}]
    assert select_candidate(res, 0.2) == 2 and select_candidate(res[:2], 0.2) is None and best_effort(res[:2]) == 1


# ---------------------------------------------------------------- nodes
def test_image_shot_node_mock(ctx, log, llm):
    from agents.asset_generator import make_image_shot_node

    node = make_image_shot_node(ctx, log, llm)
    shot = {"shot_id": 2, "template": "illustration_kenburns", "sentence_indices": [1],
            "visual_prompt": "an exam hall desk with a laptop and a question paper", "on_screen_text": [], "fact_ids": []}
    out = node({"shot": shot, "plan": PLAN, "brief": {}, "loop": 0, "feedback": {}})
    a = out["shot_assets"]["2"]
    assert Image.open(a["image_path"]).size == (1210, 2150)
    assert len(a["candidates"]) == 2 and a["face_count"] == 0
    kinds = sorted(r["kind"] for r in read_manifest(ctx))
    assert kinds == ["image", "image_candidate", "image_candidate"]
    assert licence_failures(ctx) == []


def test_image_shot_retry_without_people(ctx, log, llm, monkeypatch):
    import agents.asset_generator as ag

    calls = []

    def fake(ctx_, log_, *, agent, worker, request, gpu="1", timeout_s=1800):
        calls.append((worker, request))
        if worker == "image_checks.py":
            first = len([c for c in calls if c[0] == "image_checks.py"]) == 1
            return {"results": [{"path": p, "face_count": 1 if first else 0, "face_scores": [], "nsfw_score": 0.0}
                                for p in request["paths"]]}
        return run_worker(ctx_, log_, agent=agent, worker=worker, request=request, gpu=gpu)

    monkeypatch.setattr(ag, "run_worker", fake)
    node = ag.make_image_shot_node(ctx, log, llm)
    shot = {"shot_id": 5, "template": "illustration_kenburns", "sentence_indices": [0],
            "visual_prompt": "a student holding a phone", "on_screen_text": [], "fact_ids": []}
    a = node({"shot": shot, "plan": PLAN, "brief": {}, "loop": 1,
              "feedback": {"asset_generator": ["[faces:5] Fix: remove the person"]}})["shot_assets"]["5"]
    gens = [r for w, r in calls if w == "image_sdxl.py"]
    assert len(gens) == 2 and "_r1" in gens[1]["jobs"][0]["out_prefix"]
    assert "student" not in gens[1]["jobs"][0]["prompt"]
    assert a["face_count"] == 0 and "_r1_" in Path(a["candidates"][-1]).name and len(a["candidates"]) == 4


def test_concurrent_shots_share_one_sdxl_load(ctx, log, llm, monkeypatch):
    import agents.asset_generator as ag

    workers = []

    def counting(ctx_, log_, **kw):
        workers.append(kw["worker"])
        return run_worker(ctx_, log_, **kw)

    monkeypatch.setattr(ag, "run_worker", counting)
    node = ag.make_image_shot_node(ctx, log, llm)
    node.__closure__  # noqa: B018 - node keeps its batcher
    results: dict = {}

    def go(i):
        shot = {"shot_id": i, "template": "illustration_kenburns", "sentence_indices": [0],
                "visual_prompt": f"icon number {i}", "on_screen_text": [], "fact_ids": []}
        results[i] = node({"shot": shot, "plan": PLAN, "brief": {}, "loop": 0, "feedback": {}})

    ts = [threading.Thread(target=go, args=(i,)) for i in (1, 2, 3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(results) == [1, 2, 3]
    assert all(str(i) in results[i]["shot_assets"] for i in results)
    assert workers.count("image_sdxl.py") == 1 and workers.count("upscale.py") == 1


def test_voice_node_mock(ctx, log, llm):
    from agents.asset_generator import make_voice_node

    state = {"script": _script(), "plan": PLAN, "loop": 0, "feedback": {}}
    out = make_voice_node(ctx, log, llm)(state)
    v = out["voice"]
    assert v["wer"] == 0.0 and Path(v["wav_path"]).exists() and v["duration_s"] > 5
    assert len(v["sentence_spans"]) == 4
    disp = [w["word"] for w in v["words"]]
    assert "₹2,000" in disp and "KJSCE" in disp and "exams?" in disp
    pages = out["captions"]
    assert pages and Path(out["srt_path"]).read_text(encoding="utf-8").count(" --> ") == len(pages)
    from core.schemas import CaptionPage

    hard = [i for i in check_pages([CaptionPage.model_validate(p) for p in pages]) if "lines" in i or "chars" in i
            or "crosses" in i or "overlap" in i]
    assert hard == []
    kinds = sorted(r["kind"] for r in read_manifest(ctx))
    assert kinds == ["caption", "voice"]


def test_join_node(ctx, log, llm, tmp_path):
    from agents.asset_generator import make_join_node

    img = tmp_path / "i.png"
    img.write_bytes(b"x")
    shots = [{"shot_id": 1, "visual_prompt": "x"}, {"shot_id": 2, "visual_prompt": ""}, {"shot_id": 3, "visual_prompt": "y"}]
    node = make_join_node(ctx, log, llm)
    st = {"storyboard": {"shots": shots}, "shot_assets": {"1": {"image_path": str(img)}, "3": {"image_path": str(img)}},
          "voice": {"wav_path": "v.wav"}, "loop": 0}
    out = node(st)
    assert out["extra"] == {"assets_ready": True, "redo_voice": False, "missing_shots": []} and out["redo_shots"] == []
    st["shot_assets"].pop("3")
    out = node(st)
    assert out["extra"]["assets_ready"] is False and out["extra"]["missing_shots"] == [3] and out["errors"]


def test_image_shot_faces_persist_sets_vlm_not_ok(ctx, log, llm, monkeypatch):
    import agents.asset_generator as ag

    def fake(ctx_, log_, *, agent, worker, request, gpu="1", timeout_s=1800):
        if worker == "image_checks.py":
            return {"face_detector": "yunet",
                    "results": [{"path": p, "face_count": 2, "face_scores": [0.91, 0.7], "nsfw_score": 0.0}
                                for p in request["paths"]]}
        return run_worker(ctx_, log_, agent=agent, worker=worker, request=request, gpu=gpu)

    monkeypatch.setattr(ag, "run_worker", fake)
    shot = {"shot_id": 6, "template": "illustration_kenburns", "sentence_indices": [0],
            "visual_prompt": "a crowded classroom", "on_screen_text": [], "fact_ids": []}
    a = ag.make_image_shot_node(ctx, log, llm)({"shot": shot, "plan": PLAN, "brief": {}, "loop": 0,
                                                 "feedback": {}})["shot_assets"]["6"]
    assert a["face_count"] == 2 and a["vlm_ok"] is False and "YuNet detected 2" in a["vlm_note"]
    assert Path(a["image_path"]).exists() and len(a["candidates"]) == 4


def test_assets_subgraph_with_real_dispatch(ctx, log, llm, monkeypatch):
    """asset_dispatch -> Send fan-out (image_shot x3 + voice) -> deferred assets_join, as wired in core/graph.py."""
    from langgraph.graph import START, StateGraph

    import agents.asset_generator as ag
    from core.graph import dispatch_assets
    from core.state import AgentState

    calls = []

    def counting(ctx_, log_, **kw):
        calls.append(kw["worker"])
        return run_worker(ctx_, log_, **kw)

    monkeypatch.setattr(ag, "run_worker", counting)
    g = StateGraph(AgentState)
    g.add_node("asset_dispatch", lambda s: {"extra": {"dispatch_loop": 0}})
    g.add_node("image_shot", ag.make_image_shot_node(ctx, log, llm))
    g.add_node("voice", ag.make_voice_node(ctx, log, llm))
    g.add_node("assets_join", ag.make_join_node(ctx, log, llm), defer=True)
    g.add_edge(START, "asset_dispatch")
    g.add_conditional_edges("asset_dispatch", dispatch_assets, ["image_shot", "voice", "assets_join"])
    g.add_edge("image_shot", "assets_join")
    g.add_edge("voice", "assets_join")
    shots = [{"shot_id": i, "template": "illustration_kenburns" if i else "hook_title", "sentence_indices": [0],
              "visual_prompt": f"icon {i}" if i != 2 else "", "on_screen_text": [], "fact_ids": []} for i in range(4)]
    out = g.compile().invoke({"storyboard": {"shots": shots}, "plan": PLAN, "brief": {}, "script": _script(),
                              "loop": 0, "feedback": {}})
    assert sorted(out["shot_assets"]) == ["0", "1", "3"]
    assert out["extra"]["assets_ready"] is True and out["redo_shots"] == [] and out["voice"]["wav_path"]
    assert calls.count("image_sdxl.py") == 1          # three shots coalesced into one SDXL load
