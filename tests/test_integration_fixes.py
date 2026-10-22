"""Regressions found in the first mock end-to-end run (C01)."""
from agents.scriptwriter import rank_facts, topic_phrase
from core.qa_checks import check_facts, cites_whitelisted_contact
from core.textnorm import to_spoken


def test_rupee_slash_dash_not_spoken():
    s = to_spoken("Once a late fine reaches Rs.500/- and above")
    assert "five hundred rupees" in s and "/-" not in s and " - " not in s


def test_topic_phrase():
    assert topic_phrase("College AI policy explainer: what is allowed vs not allowed") == "AI policy"
    assert topic_phrase("Cyber safety: sextortion, digital arrest scams") == "Cyber safety"


def test_rank_facts_prefers_coverage_and_drops_irrelevant():
    facts = [
        {"fact_id": "F1", "claim": "Library facility is restricted to current students.", "quote": "x"},
        {"fact_id": "F2", "claim": "Unfair means cases go to the CPUMC.", "quote": "x"},
        {"fact_id": "F3", "claim": "Using a prohibited software tool in an exam is unfair means.", "quote": "x"},
    ]
    coverage = [{"item": "who decides on unfair-means cases", "fact_ids": ["F2"], "status": "covered"}]
    brief = {"topic": "College AI policy explainer: allowed vs not allowed in exams",
             "must_include": ["what counts as unfair means in exams"]}
    ids = [f["fact_id"] for f in rank_facts(facts, coverage, brief)]
    assert ids[0] == "F2" and "F3" in ids and "F1" not in ids


def test_cta_with_whitelisted_url_counts_as_sourced():
    contacts = [{"kind": "url", "value": "https://www.somaiya.edu/en/center-of-examination",
                 "display": "https://www.somaiya.edu/en/center-of-examination", "source_ids": ["S1"]}]
    assert cites_whitelisted_contact("Read the examination rules: somaiya.edu.", contacts)
    assert not cites_whitelisted_contact("Read the rules: example.com.", contacts)
    script = {"hook": {"style": "question", "display": "Unsure?", "spoken": "Unsure?", "fact_ids": []},
              "hook_index": 0, "body": [],
              "cta": {"display": "Read the examination rules on the official website: somaiya.edu.",
                      "spoken": "", "fact_ids": []}, "word_count": 10}
    checks = {c.check_id: c for c in check_facts(script, [], None, contacts)}
    assert checks["facts_tagged"].passed


# ---- found in the first real Kaggle run (2026-10-22)
def test_stale_image_not_reused_after_storyboard_change():
    from core.graph import dispatch_assets, image_is_current

    shot_old = {"shot_id": 3, "visual_prompt": "a college building"}
    shot_new = {"shot_id": 3, "visual_prompt": "an envelope icon"}
    asset = {"shot_id": 3, "image_path": "x.png", "visual_prompt": "a college building"}
    assert image_is_current(shot_old, asset) and not image_is_current(shot_new, asset)
    state = {"storyboard": {"shots": [shot_new, {"shot_id": 4, "visual_prompt": ""}]},
             "shot_assets": {"3": asset, "4": {"shot_id": 4, "image_path": "y.png", "visual_prompt": "old"}},
             "plan": {}, "brief": {}, "voice": {"wav_path": "v.wav"}}
    sends = dispatch_assets(state)
    assert [s.node for s in sends] == ["image_shot"] and sends[0].arg["shot"]["shot_id"] == 3


def test_short_urls():
    from core.textnorm import short_urls

    assert short_urls("Read https://www.somaiya.edu/en/center-of-examination/.") == "Read somaiya.edu/en/center-of-examination."
    assert short_urls("Email helpline@antiragging.in") == "Email helpline@antiragging.in"


def test_silence_after_voiceover_is_planned():
    from core.qa_checks import allowed_windows, check_dead_air

    m = {"silences": [(34.5, 40.28)], "blacks": [], "freezes": [], "probe": {"format": {"duration": "40.30"}}}
    without = {c.check_id: c for c in check_dead_air(m, allowed_windows(40.3, [10.0, 20.0]))}
    with_vo = {c.check_id: c for c in check_dead_air(m, allowed_windows(40.3, [10.0, 20.0], vo_end=34.98))}
    assert not without["dead_air"].passed and with_vo["dead_air"].passed
