"""Offline tests for core/qa_checks.py, agents/critic.py, agents/human_review.py, agents/export.py, core/budget.py."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core import qa_checks as q
from core.schemas import QACheck, QAReport

SCRIPT = {
    "hook": {"style": "question", "display": "Can you use AI tools in your exams?", "spoken": "Can you use AI tools in your exams?",
             "fact_ids": []},
    "hook_index": 0,
    "body": [
        {"display": "Using a software tool the examiner prohibited counts as unfair means.",
         "spoken": "Using a software tool the examiner prohibited counts as unfair means.", "fact_ids": ["F1"]},
        {"display": "Cases go to the college committee first.", "spoken": "Cases go to the college committee first.",
         "fact_ids": ["F2"]},
        {"display": "So ask first.", "spoken": "So ask first.", "fact_ids": []},
    ],
    "cta": {"display": "Read the exam rules on somaiya.edu", "spoken": "Read the exam rules on somaiya dot edu",
            "fact_ids": ["F1"]},
    "word_count": 40,
}
FACTS = [
    {"fact_id": "F1", "claim": "c", "quote": "software tool which is prohibited by the examiner", "chunk_id": "S1#c1",
     "source_id": "S1", "url": "https://somaiya.edu/exam"},
    {"fact_id": "F2", "claim": "c", "quote": "College Prevention of Unfair Means Committee", "chunk_id": "S1#c2",
     "source_id": "S1", "url": "https://somaiya.edu/exam"},
]
PLAN = {"brief_id": "P00", "category": "C", "rule_packs": ["base", "C"], "aspect": "9x16", "width": 1080, "height": 1920,
        "fps": 30, "duration_s": 32, "word_budget_min": 30, "word_budget_max": 90, "shot_count_min": 6, "shot_count_max": 12,
        "mandatory_review": True, "queries": [], "audience_insight": "", "success_criteria": [],
        "topic_keywords": ["AI tools", "exams"], "budgets": []}


# ---------------------------------------------------------------- media fixtures
def _ff(*args: str) -> None:
    subprocess.run([q.ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True, timeout=180)


@pytest.fixture(scope="module")
def videos(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("media")
    ok, bad = d / "ok.mp4", d / "bad.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30:duration=32",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=32",
        "-af", "loudnorm=I=-14:TP=-2:LRA=11", "-ar", "48000", "-ac", "2",
        "-c:v", "libx264", "-profile:v", "high", "-preset", "veryfast", "-crf", "35", "-pix_fmt", "yuv420p",
        "-r", "30", "-fps_mode", "cfr", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(ok))
    _ff("-f", "lavfi", "-i", "testsrc2=size=540x960:rate=25:duration=31",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=31",
        "-ar", "44100", "-ac", "1", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", str(bad))
    return {"ok": ok, "bad": bad, "m_ok": q.measure_media(ok), "m_bad": q.measure_media(bad)}


def _by_id(checks: list[QACheck]) -> dict[str, QACheck]:
    return {c.check_id: c for c in checks}


def test_spec_checks_pass(videos):
    m = videos["m_ok"]
    checks = (q.check_duration(m, 32) + q.check_video_stream(m, 1080, 1920) + q.check_audio_stream(m)
              + q.check_loudness(m) + [q.check_size(m)] + q.check_dead_air(m, q.allowed_windows(q.duration_of(m))))
    failed = [(c.check_id, c.value) for c in checks if not c.passed]
    assert not failed, failed
    assert all(c.responsible_agent == "editor" for c in checks if c.check_id.startswith(("video", "audio", "loud")))


def test_spec_checks_fail_25fps_44k(videos):
    m = videos["m_bad"]
    c = _by_id(q.check_video_stream(m, 1080, 1920) + q.check_audio_stream(m) + q.check_duration(m, 45))
    for cid in ("video_size", "video_cfr", "audio_sample_rate", "audio_channels", "audio_bitrate", "duration_target"):
        assert not c[cid].passed, cid
    assert c["video_codec"].passed and c["duration_range"].passed
    assert "25/1" in c["video_cfr"].value


def test_parse_filter_log_intervals():
    text = ("[silencedetect @ 0x1] silence_start: 10.5\n[silencedetect @ 0x1] silence_end: 13.0 | silence_duration: 2.5\n"
            "[blackdetect @ 0x2] black_start:0 black_end:0.4 black_duration:0.4\n"
            "[freezedetect @ 0x3] lavfi.freezedetect.freeze_start: 40.0\n")
    m = q.parse_filter_log(text)
    assert m["silences"] == [(10.5, 13.0)] and m["blacks"] == [(0.0, 0.4)] and m["freezes"] == [(40.0, None)]
    m["probe"] = {"format": {"duration": "44.0"}}
    res = _by_id(q.check_dead_air(m, q.allowed_windows(44.0, [20.0])))
    assert not res["dead_air"].passed and res["black_frames"].passed and res["frozen_frames"].passed


# ---------------------------------------------------------------- contacts / captions / helpers
CONTACTS = [
    {"kind": "phone", "value": "18001805522", "display": "1800-180-5522", "source_ids": ["S1"], "authority": "gov"},
    {"kind": "email", "value": "helpline@antiragging.in", "display": "helpline@antiragging.in", "source_ids": ["S1"], "authority": "gov"},
    {"kind": "phone", "value": "1930", "display": "1930", "source_ids": ["S2"], "authority": "gov"},
    {"kind": "url", "value": "https://cybercrime.gov.in/", "display": "cybercrime.gov.in", "source_ids": ["S2"], "authority": "gov"},
]


def test_contacts_whitelist():
    good = ("Call 1800-180-5522 or email helpline@antiragging.in. Dial 1930 or visit cybercrime.gov.in. "
            "Circular 2022-2023 cites sections 20, 20A and 47.")
    assert q.check_contacts({"script": good}, CONTACTS).passed
    bad = q.check_contacts({"script": "Call 1800-180-5252 now", "ocr": "help@antiragging.in or call 112"}, CONTACTS)
    assert not bad.passed and bad.responsible_agent == "scriptwriter"
    assert "1800-180-5252" in bad.evidence and "help@antiragging.in" in bad.evidence and "112" in bad.evidence
    assert not q.check_contacts({"s": "see fake-helpline.com"}, CONTACTS).passed


def test_caption_readability():
    good = [{"index": 0, "start": 0.0, "end": 2.0, "lines": ["Can you use AI tools", "in your exams?"]}]
    assert all(c.passed for c in q.check_caption_readability(good))
    bad = [{"index": 0, "start": 0.0, "end": 0.5, "lines": ["a" * 40, "b", "c"]}]
    res = _by_id(q.check_caption_readability(bad))
    assert not res["caption_lines"].passed and not res["caption_chars_per_line"].passed
    assert not res["caption_cps"].passed and res["caption_cps"].severity == "soft"
    assert not res["caption_min_duration"].passed
    assert not q.check_caption_readability([])[0].passed


def test_srt_parse(tmp_path):
    p = tmp_path / "x.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:01,500\nHello there\nfriend\n\n2\n00:00:01,500 --> 00:00:03,000\nBye\n", encoding="utf-8")
    pages = q.parse_srt(p)
    assert len(pages) == 2 and pages[0]["lines"] == ["Hello there", "friend"] and pages[1]["end"] == 3.0


def test_contrast_and_layout(tmp_path):
    assert round(q.contrast_ratio((255, 255, 255), (0, 0, 0)), 1) == 21.0
    assert q.check_contrast([{"text_rgb": (255, 255, 255), "bg_rgb": (40, 40, 40)}]).passed
    low = q.check_contrast([{"text_rgb": (255, 255, 255), "bg_rgb": (230, 230, 230)}])
    assert not low.passed
    outlined = q.check_contrast([{"text_rgb": (255, 255, 255), "bg_rgb": (230, 230, 230), "stroke_rgb": (0, 0, 0), "stroke_px": 6}])
    assert outlined.passed
    from PIL import Image

    img = tmp_path / "f.png"
    Image.new("RGB", (100, 100), (20, 20, 20)).save(img)
    assert q.sample_background(img, (0, 0, 100, 100), (255, 255, 255)) == (20, 20, 20)

    layout = {"shots": [{"shot_id": 1, "boxes": [{"kind": "caption", "x": 40, "y": 1200, "w": 1000, "h": 140, "lines": 2, "font_px": 64},
                                                  {"kind": "text", "text": "Title", "x": 40, "y": 300, "w": 900, "h": 120}]}]}
    assert all(c.passed for c in q.check_caption_layout(layout, 1080, 1920))
    layout["shots"][0]["boxes"].append({"kind": "text", "text": "Footer", "x": 0, "y": 1700, "w": 500, "h": 80})
    layout["shots"][0]["boxes"][0]["font_px"] = 40
    res = _by_id(q.check_caption_layout(layout, 1080, 1920))
    assert not res["caption_safe_zone"].passed and res["caption_safe_zone"].responsible_agent == "storyboarder"
    assert not res["caption_height"].passed
    # 16:9 uses the renderer's band (core.scene.safe_zones) or the scaled 9:16 band
    y0, y1 = q.safe_band(1920, 1080)
    assert 0 < y0 < y1 <= 1080
    assert not q.check_caption_layout([{"kind": "text", "x": 0, "y": y1 - 10, "w": 10, "h": 50}], 1920, 1080)[0].passed


def test_hook_and_end_card():
    words = [{"word": "Can", "start": 0.1, "end": 0.3}, {"word": "you", "start": 0.3, "end": 0.5},
             {"word": "use", "start": 0.5, "end": 0.7}, {"word": "AI", "start": 0.7, "end": 0.9},
             {"word": "tools?", "start": 0.9, "end": 1.2}, {"word": "exams", "start": 3.5, "end": 3.9}]
    assert q.check_hook(["AI tools"], words, []).passed
    assert not q.check_hook(["unfair means"], words, []).passed
    assert q.check_hook(["unfair means"], words, ["What counts as unfair means?"]).passed

    disc = "Created with AI agents — LY Agentic AI Lab"
    ocr = "SOMAIYA\nRead the exam rules\nCreated with Al agents - LY Agentic Al Lab"
    res = _by_id(q.check_end_card(ocr, disc, ["Read the exam rules"], logo=(True, 0.85)))
    assert all(c.passed for c in res.values()), res
    res = _by_id(q.check_end_card("", disc, ["Read the exam rules"], declared_blob=q.json_strings({"logo": "logo.svg", "text": [disc]}),
                                  declared_logo=True, ocr_available=False))
    assert not res["end_card_cta"].passed and res["end_card_disclosure"].passed and "declared" in res["end_card_disclosure"].evidence


def test_fact_and_coverage_checks():
    c = _by_id(q.check_facts(SCRIPT, FACTS))
    assert c["facts_tagged"].passed and c["facts_known_ids"].passed and c["facts_quotes"].passed
    bad = json.loads(json.dumps(SCRIPT))
    bad["body"][0]["fact_ids"] = []
    bad["body"][1]["fact_ids"] = ["F9"]
    c = _by_id(q.check_facts(bad, FACTS))
    assert not c["facts_tagged"].passed and not c["facts_known_ids"].passed
    brief = {"must_include": ["what counts as unfair means in exams", "penalties for students"],
             "must_avoid": ["fear-based imagery", "cheating"]}
    cov = [{"item": "what counts as unfair means in exams", "fact_ids": ["F1"], "status": "covered"},
           {"item": "penalties for students", "fact_ids": [], "status": "MISSING"}]
    mi = q.check_must_include(brief, cov, SCRIPT)
    assert mi[0].passed and mi[1].value == "MISSING in sources"
    ma = _by_id(q.check_must_avoid(brief, SCRIPT))
    assert ma["must_avoid:fear-based imagery"].passed and ma["must_avoid:cheating"].passed
    assert not q.check_must_avoid({"must_avoid": ["unfair means"]}, SCRIPT)[0].passed


def test_shot_images_wer_licences():
    sa = {"1": {"shot_id": 1, "image_path": "a.png", "face_count": 1, "vlm_ok": False, "nsfw_score": 0.5},
          "2": {"shot_id": 2, "image_path": "b.png", "face_count": 0, "nsfw_score": 0.01}}
    c = _by_id(q.check_shot_images(sa))
    assert not c["faces:1"].passed and not c["nsfw:1"].passed and c["faces:2"].passed and c["nsfw:2"].passed
    assert not q.check_wer({"wav_path": "x", "duration_s": 1, "voice": "v", "sentence_spans": [], "words": [], "wer": 0.2}).passed
    rows = [{"path": "a.png", "licence": "CreativeML-OpenRAIL++-M"}, {"path": "f.ttf", "licence": "Proprietary"}]
    lic = q.check_licences(rows, q.DEFAULT_LICENCES)
    assert not lic.passed and "f.ttf" in lic.evidence


# ---------------------------------------------------------------- routing
def _report(*fails: tuple[str, str, str], loop: int = 0) -> QAReport:
    checks = [QACheck(check_id=i, passed=False, responsible_agent=a, severity=s, evidence="ev") for i, a, s in fails]
    checks.append(QACheck(check_id="ok", passed=True, responsible_agent="editor"))
    return QAReport(run_id="t", loop=loop, passed=not fails, checks=checks)


def _sha(_state, check, _agent):
    return "sha-" + check.check_id


def test_routing_priority_and_feedback():
    from agents.critic import decide

    state = {"plan": {"mandatory_review": False}, "loop": 0}
    rep = _report(("video_profile", "editor", "hard"), ("nsfw:3", "asset_generator", "hard"),
                  ("onscreen_words", "storyboarder", "hard"), ("voice_wer", "asset_generator", "hard"),
                  ("facts_tagged", "scriptwriter", "hard"))
    out = decide(state, rep, max_loops=3, sha_fn=_sha)
    assert out["route"] == "scriptwriter" and out["loop"] == 1
    assert out["redo_shots"] == [3] and out["extra"]["redo_voice"] is True
    assert out["feedback"]["scriptwriter"][0].startswith("[facts_tagged]") and out["feedback"]["researcher"] == []
    assert out["feedback"]["editor"] and {h["routed_to"] for h in out["history"]} == {"scriptwriter"}
    # researcher wins once, then demoted to scriptwriter
    rep = _report(("facts_quotes", "researcher", "hard"), ("video_size", "editor", "hard"))
    assert decide(state, rep, max_loops=3, sha_fn=_sha)["route"] == "researcher"
    state2 = {**state, "loop": 1, "history": [{"loop": 0, "check_id": "x", "artifact_sha": "y", "routed_to": "researcher"}]}
    assert decide(state2, rep, max_loops=3, sha_fn=_sha)["route"] == "scriptwriter"


def test_routing_pass_and_max_loops():
    from agents.critic import decide

    ok = _report()
    assert decide({"plan": {"mandatory_review": True}}, ok, max_loops=3)["route"] == "human_review"
    assert decide({"plan": {"mandatory_review": False}}, ok, max_loops=3)["route"] == "export"
    hard = _report(("video_size", "editor", "hard"))
    soft = _report(("caption_cps", "asset_generator", "soft"))
    s3 = {"plan": {"mandatory_review": False}, "loop": 3}
    assert decide(s3, hard, max_loops=3, sha_fn=_sha)["route"] == "human_review"
    out = decide(s3, soft, max_loops=3, sha_fn=_sha)
    assert out["route"] == "export" and "loop" not in out and "loop cap" in out["extra"]["qa_summary"]["reason"]
    assert decide({**s3, "plan": {"mandatory_review": True}}, soft, max_loops=3, sha_fn=_sha)["route"] == "human_review"


def test_regression_guard():
    from agents.critic import decide

    rep = _report(("video_size", "editor", "hard"), loop=1)
    hist = [{"loop": 0, "check_id": "video_size", "artifact_sha": "sha-video_size", "routed_to": "editor"}]
    out = decide({"plan": {"mandatory_review": False}, "loop": 1, "history": hist}, rep, max_loops=3, sha_fn=_sha)
    assert out["route"] == "human_review" and out["extra"]["qa_summary"]["escalated"] is True
    # same sha but the agent was never asked to fix it (routed elsewhere) -> keep looping
    hist2 = [{**hist[0], "routed_to": "scriptwriter"}]
    out = decide({"plan": {"mandatory_review": False}, "loop": 1, "history": hist2}, rep, max_loops=3, sha_fn=_sha)
    assert out["route"] == "editor" and out["loop"] == 2


# ---------------------------------------------------------------- critic node end-to-end (mock LLM)
def test_critic_node_writes_reports(ctx, log, llm, videos):
    from agents.critic import make_node

    voice_words = [{"word": w, "start": 0.2 * i, "end": 0.2 * i + 0.2, "sentence_index": 0}
                   for i, w in enumerate("Can you use AI tools in your exams".split())]
    state = {
        "brief": {"id": "P00", "duration_s": 32, "must_include": [], "must_avoid": []}, "plan": PLAN, "loop": 0,
        "script": SCRIPT, "facts": FACTS, "contacts": CONTACTS, "coverage": [],
        "storyboard": {"shots": [{"shot_id": 1, "template": "hook_title", "sentence_indices": [0], "visual_prompt": "",
                                  "on_screen_text": ["AI tools in exams?"], "fact_ids": []}]},
        "voice": {"wav_path": "", "duration_s": 30.0, "voice": "af", "sentence_spans": [[0, 30]], "words": voice_words, "wer": 0.0},
        "captions": [{"index": 0, "start": 0.0, "end": 2.0, "lines": ["Can you use AI tools", "in your exams?"], "words": []}],
        "video_path": str(videos["ok"]),
    }
    out = make_node(ctx, log, llm)(state)
    rep = QAReport.model_validate(out["qa_report"])
    assert ctx.artifact("qa_report_loop0.json").exists() and ctx.artifact("qa_report.json").exists()
    ids = _by_id(rep.checks)
    for cid in ("video_codec", "audio_codec", "loudness_integrated", "hook_topic_3s", "facts_tagged", "contacts_whitelist"):
        assert ids[cid].passed, (cid, ids[cid])
    assert ids["caption_contrast"].passed and "mock media" in ids["caption_contrast"].evidence   # relaxed in mock_media
    assert out["route"] in {"editor", "scriptwriter", "storyboarder", "asset_generator", "researcher", "human_review", "export"}
    if not rep.passed:
        assert out["loop"] == 1 and out["history"]


def test_relax_for_mock():
    from agents.critic import relax_for_mock

    cs = relax_for_mock([QACheck(check_id="caption_contrast", passed=False, value="1.1", responsible_agent="editor"),
                         QACheck(check_id="video_size", passed=False, responsible_agent="editor")])
    assert cs[0].passed and "mock media" in cs[0].evidence and not cs[1].passed


# ---------------------------------------------------------------- human review interrupt / resume
def _review_graph(ctx, log, llm):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from agents.human_review import make_node
    from core.state import AgentState

    g = StateGraph(AgentState)
    g.add_node("human_review", make_node(ctx, log, llm))
    g.add_edge(START, "human_review")
    g.add_edge("human_review", END)
    return g.compile(checkpointer=InMemorySaver())


def test_human_review_interrupt_and_resume(ctx, log, llm):
    from langgraph.types import Command

    graph = _review_graph(ctx, log, llm)
    state = {"plan": PLAN, "brief": {"id": "C01"}, "script": SCRIPT, "facts": FACTS, "contacts": CONTACTS,
             "coverage": [{"item": "penalties", "status": "MISSING", "fact_ids": []}],
             "qa_report": {"run_id": "t", "loop": 0, "passed": True, "checks": []}, "loop": 0, "video_path": "v.mp4"}
    cfg = {"configurable": {"thread_id": "r1"}}
    res = graph.invoke(state, cfg)
    assert "__interrupt__" in res
    summary = res["__interrupt__"][0].value
    assert summary["coverage_missing"] == ["penalties"] and Path(summary["packet_md"]).exists()
    packet = json.loads(Path(summary["packet_json"]).read_text(encoding="utf-8"))
    assert packet["claims"][0]["url"] == "https://somaiya.edu/exam"

    out = graph.invoke(Command(resume={"approved": False, "note": "soften line 2", "target_agent": "storyboarder"}), cfg)
    assert out["route"] == "storyboarder" and out["loop"] == 1 and out["feedback"]["storyboarder"] == ["[human_review] soften line 2"]
    lines = [json.loads(x) for x in log.path.read_text(encoding="utf-8").splitlines()]
    assert any(r["tool"] == "review_checkpoint" for r in lines) and not any(r["human_intervention"] for r in lines)

    cfg2 = {"configurable": {"thread_id": "r2"}}
    graph.invoke(state, cfg2)
    assert graph.invoke(Command(resume={"approved": True}), cfg2)["route"] == "export"


def test_human_review_skips_when_not_mandatory(ctx, log, llm):
    graph = _review_graph(ctx, log, llm)
    state = {"plan": {**PLAN, "mandatory_review": False}, "qa_report": {"passed": True, "checks": []}}
    out = graph.invoke(state, {"configurable": {"thread_id": "r3"}})
    assert out["route"] == "export" and "__interrupt__" not in out


# ---------------------------------------------------------------- export / budget
def test_export_node(ctx, log, llm, videos):
    from agents.export import make_node

    srt = ctx.out_dir / "c.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    ctx.write_json("qa_report.json", {"passed": True})
    log.record(agent="scriptwriter", tool="llm", model="m", input_tokens=100, output_tokens=50, latency_s=2.0)
    log.record(agent="asset_generator", tool="worker:image_sdxl", units={"images": 4})
    out = make_node(ctx, log, llm)({"brief": {"id": "C01"}, "video_path": str(videos["ok"]), "srt_path": str(srt),
                                    "facts": FACTS, "loop": 2, "stage_times": {"critic": 1.0}})
    f = out["final"]
    assert Path(f["video"]).name == "TXX_C01_final.mp4" and Path(f["srt"]).name == "TXX_C01_final.srt"
    summ = json.loads(Path(f["run_summary"]).read_text(encoding="utf-8"))
    assert summ["total_tokens"] == 150 and summ["loops"] == 2 and summ["interventions"] == 0
    cost = Path(f["cost_report"]).read_text(encoding="utf-8")
    assert "scriptwriter" in cost and "images" in cost


def test_budget(ctx):
    from datetime import datetime, timedelta, timezone

    from core.budget import Budget

    b = Budget(ctx, PLAN, settings={"time_limit_min": 180, "degrade_after_min": 120})
    now = b.started + timedelta(minutes=10)
    assert not b.should_degrade(now) and b.loops_allowed(0, 3, now) == 3
    late = b.started + timedelta(minutes=150)
    assert b.should_degrade(late) and b.loops_allowed(1, 3, late) == 1
    assert isinstance(b.snapshot(datetime.now(timezone.utc)), dict)
