"""Creative stack (offline, mock LLM): textnorm, rule packs, scriptwriter, hook judge, storyboarder."""
import pytest
import yaml

from agents import hook_judge, scriptwriter, storyboarder
from core.rules import available_packs, code_rule_ids, load_packs, rules_text, style_for, voice_for
from core.schemas import TEMPLATES, HookOut, Script, ScriptOut, ScriptSentenceOut, ShotOut, StoryboardOut
from core.textnorm import (cardinal, count_words, find_contacts, shorten, spoken_issues, strip_unlisted_contacts,
                           to_spoken, year_words)
from tests.conftest import ROOT

COE = "https://www.somaiya.edu/en/center-of-examination/"
FACTS = [
    {"fact_id": "F1", "claim": "Using a software tool prohibited by the examiner during an on-screen exam counts as unfair means",
     "quote": "q1", "chunk_id": "S1#c1", "source_id": "S1", "url": COE},
    {"fact_id": "F2", "claim": "Unfair means cases go to the College Prevention of Unfair Means Committee first",
     "quote": "q2", "chunk_id": "S2#c1", "source_id": "S2", "url": COE},
    {"fact_id": "F3", "claim": "An examination means any evaluation, including theory, practical, oral, seminar and presentation",
     "quote": "q3", "chunk_id": "S1#c2", "source_id": "S1", "url": COE},
    {"fact_id": "F4", "claim": "Carrying electronic gadgets like a mobile phone or smart watch into an exam is not allowed",
     "quote": "q4", "chunk_id": "S1#c3", "source_id": "S1", "url": COE},
]
CONTACTS = [
    {"kind": "url", "value": COE, "display": "somaiya.edu/en/center-of-examination", "label": "CoE page",
     "source_ids": ["S1"], "authority": "somaiya"},
    {"kind": "phone", "value": "1930", "display": "1930", "label": "Cybercrime helpline",
     "source_ids": ["S3"], "authority": "gov"},
]
COVERAGE = [
    {"item": "what counts as unfair means in exams", "fact_ids": ["F1", "F4"], "status": "covered"},
    {"item": "who decides on unfair-means cases", "fact_ids": ["F2"], "status": "covered"},
]
PLAN = {"category": "C", "rule_packs": ["base", "C"], "word_budget_min": 97, "word_budget_max": 119,
        "shot_count_min": 6, "shot_count_max": 12, "duration_s": 45,
        "topic_keywords": ["AI tools", "unfair means", "exams"], "audience_insight": "Students worry about cheating by accident"}


@pytest.fixture
def state():
    brief = yaml.safe_load((ROOT / "briefs" / "C01.yaml").read_text(encoding="utf-8"))
    return {"brief": brief, "plan": dict(PLAN), "facts": [dict(f) for f in FACTS], "contacts": list(CONTACTS),
            "coverage": list(COVERAGE), "loop": 0}


def s(display, ids=()):
    return {"display": display, "spoken": display, "fact_ids": list(ids)}


# ---------------------------------------------------------------- textnorm
@pytest.mark.parametrize("display, spoken", [
    ("Dial 1930 to report cyber fraud.", "Dial one nine three zero to report cyber fraud."),
    ("1930 is the national cybercrime helpline.", "one nine three zero is the national cybercrime helpline."),
    ("Call 1800-180-5522 any time.", "Call one eight zero zero, one eight zero, five five two two any time."),
    ("The fee is ₹2,000.", "The fee is two thousand rupees."),
    ("A ₹1,50,000 grant", "A one lakh fifty thousand rupees grant"),
    ("KJSCE and UGC", "K J S C E and U G C"),
    ("The POSH Act and APAAR ID", "The posh Act and apaar I D"),
    ("NEP and DPDP", "N E P and D P D P"),
    ("Visit https://www.somaiya.edu/en/center-of-examination/ today.", "Visit somaiya dot E D U today."),
    ("Email helpline@antiragging.in", "Email helpline at antiragging dot in"),
    ("In 2026 the rules changed.", "In twenty twenty-six the rules changed."),
    ("The DPDP Act 2023 applies.", "The D P D P Act twenty twenty-three applies."),
    ("It is NOT allowed.", "It is not allowed."),
    ("50% of 2000 students", "fifty percent of two thousand students"),
    ("Session 2024-25 starts", "Session twenty twenty-four to twenty-five starts"),
])
def test_to_spoken_cases(display, spoken):
    assert to_spoken(display) == spoken


def test_to_spoken_whitelisted_number_digits_without_context():
    assert to_spoken("Remember 1930.", CONTACTS) == "Remember one nine three zero."
    assert to_spoken("Remember 181.", [{"kind": "phone", "value": "181"}]) == "Remember one eight one."


def test_url_spoken_lowercase_form():
    assert to_spoken("somaiya.edu").lower() == "somaiya dot e d u"


def test_spoken_output_is_tts_safe():
    for text in ["Call 1800-180-5522 or 14416, visit cybercrime.gov.in & read COE/134/2022 dated 06/01/2023.",
                 "₹2,000 fee + 18% GST for 3rd-year B.Tech students at 10:30 am #KJSCE"]:
        assert spoken_issues(to_spoken(text)) == []


def test_number_words():
    assert cardinal(1_500_000, indian=True) == "fifteen lakh"
    assert cardinal(2_500_000) == "two million five hundred thousand"
    assert year_words(2005) == "two thousand five" and year_words(1930) == "nineteen thirty"


def test_count_and_shorten():
    assert count_words("Call 1800-180-5522 now.") == 3
    out = shorten("Read the examination rules on the official Somaiya website today")
    assert count_words(out) <= 7 and out.startswith("Read")


def test_contacts_whitelist_strip():
    text, removed = strip_unlisted_contacts("Call 1930 or 1800-180-5522, or visit fake.com.", CONTACTS)
    assert removed == ["1800-180-5522", "fake.com"]
    assert "1930" in text and "official helpline" in text and "official website" in text
    assert not find_contacts("In 2026 you can call us.", CONTACTS)      # a year is not a phone


# ---------------------------------------------------------------- rules / styles
def test_rule_packs_load_and_merge():
    assert set(available_packs()) >= {"base", "A", "B", "C", "D", "E", "P"}
    for name in ["A", "B", "C", "D", "E", "P"]:
        packs = load_packs(["base", name])
        assert packs["packs"] == ["base", name]
        for r in packs["rules"]:
            assert r["check"] in ("llm", "code")
            assert r["applies_to"] in ("script", "storyboard", "frames", "contacts")
            assert r["responsible_agent"] in ("scriptwriter", "storyboarder", "asset_generator", "editor", "researcher")
    c = load_packs(["base", "C"])
    assert c["mandatory_review"] and c["style_preset"] == "flat_vector"
    assert load_packs(["base", "A"])["style_preset"] == "silhouette_symbolic"
    assert not load_packs(["base", "B"])["mandatory_review"]
    txt = rules_text(c, "script")
    assert "[C.policy_quoted]" in txt and "[base.on_screen_words]" not in txt
    assert "base.contacts_whitelist" in code_rule_ids(c) and "C.policy_quoted" not in code_rule_ids(c)


def test_styles():
    a = style_for("A")
    assert a["sensitive"] and a["name"] == "silhouette_symbolic" and "photorealistic" in a["negative"]
    assert not style_for("B")["sensitive"]
    assert voice_for("A")["voice"] == "bf_emma" and voice_for("Z")["voice"] == "af_heart"


# ---------------------------------------------------------------- scriptwriter
def test_scriptwriter_mock_draft(ctx, log, llm, state):
    out = scriptwriter.make_node(ctx, log, llm)(state)
    d = out["extra"]["script_draft"]
    draft = ScriptOut.model_validate(d["draft"])
    assert [h.style for h in draft.hooks] == list(scriptwriter.HOOK_STYLES)
    ids = {f["fact_id"] for f in FACTS}
    assert all(set(x.fact_ids) <= ids for x in [*draft.hooks, *draft.body, draft.cta])
    assert d["budget"]["ok"], d["budget"]
    assert all(spoken_issues(x.spoken) == [] for x in [*draft.hooks, *draft.body, draft.cta])
    assert "script" not in out and (ctx.out_dir / "script_draft.json").exists()


def test_scriptwriter_word_budget_repair(ctx, log, llm, state):
    calls = []
    good = scriptwriter.draft_from_facts(state["brief"], FACTS, CONTACTS, COVERAGE, 97, 119, PLAN["topic_keywords"])
    short = ScriptOut(hooks=good.hooks, body=good.body[:2], cta=good.cta)

    def fixture(v):
        calls.append(v["feedback"])
        return (short if len(calls) == 1 else good).model_dump()

    llm.fixtures["scriptwriter_v1"] = fixture
    d = scriptwriter.make_node(ctx, log, llm)(state)["extra"]["script_draft"]
    assert len(calls) == 2
    assert any("Too short" in f for f in calls[1])          # explicit counts reach the repair prompt
    assert d["source"] == "llm_repair" and d["budget"]["ok"]


def test_scriptwriter_trims_and_sanitises(ctx, log, llm, state):
    long_body = [s(f"Here is connective line number {i} with a few extra words.") for i in range(20)]
    bad = {"hooks": [s("Confused about AI tools?") | {"style": "question"},
                     s("Call 1800-123-4567 for exam help.", ["F1"]) | {"style": "surprising_fact"},
                     s("You should know these AI tools facts.") | {"style": "direct_address"}],
           "body": [s(FACTS[0]["claim"] + ".", ["F1", "F99"]), *long_body],
           "cta": s("Read the rules at somaiya.edu.", ["F1"])}
    llm.fixtures["scriptwriter_v1"] = lambda v: bad
    d = scriptwriter.make_node(ctx, log, llm)(state)["extra"]["script_draft"]
    draft = ScriptOut.model_validate(d["draft"])
    assert "1800-123-4567" not in draft.hooks[1].display and "official helpline" in draft.hooks[1].display
    assert draft.body[0].fact_ids == ["F1"]                       # unknown F99 stripped
    assert max(d["budget"]["totals"]) <= 119
    assert any("trimmed" in i for i in d["issues"])


# ---------------------------------------------------------------- hook judge
def _draft_state(state, hooks):
    body = [ScriptSentenceOut(**s(FACTS[1]["claim"] + ".", ["F2"]))]
    cta = ScriptSentenceOut(**s("Read the rules at somaiya.edu.", ["F1"]))
    state["extra"] = {"script_draft": {"draft": ScriptOut(hooks=hooks, body=body, cta=cta).model_dump()}}
    return state


def test_hook_judge_order_swap_averaging_and_tiebreak(ctx, log, llm, state):
    hooks = [HookOut(style="question", display="Are AI tools allowed in your exams this year?", spoken="", fact_ids=[]),
             HookOut(style="surprising_fact", display="Exams include orals.", spoken="", fact_ids=["F3"]),
             HookOut(style="direct_address", display="You and AI tools: know this.", spoken="", fact_ids=[])]
    for h in hooks:
        h.spoken = to_spoken(h.display)
    seen = []

    def biased(v):      # position 0 always gets 9, others 5 -> pure position bias
        seen.append([h["display"] for h in v["hooks"]])
        return {"scores": [{"hook_index": h["index"], "topic_clarity": 9 if h["index"] == 0 else 5,
                            "audience_fit": 9 if h["index"] == 0 else 5, "tone_fit": 9 if h["index"] == 0 else 5,
                            "unsourced_claim": False, "reason": "x"} for h in v["hooks"]]}

    llm.fixtures["hook_judge_v1"] = biased
    out = hook_judge.make_node(ctx, log, llm)(_draft_state(state, hooks))
    ab = out["hook_ab"]
    assert seen[0] == list(reversed(seen[1]))                  # second pass reverses the order
    means = [r["mean"]["topic_clarity"] for r in ab["hooks"]]
    assert means == [7.0, 5.0, 7.0]                            # (9+5)/2 for the two ends
    # hooks 0 and 2 tie on total (both have the keyword early) -> fewer spoken words wins (hook 2)
    assert ab["hooks"][0]["total"] == ab["hooks"][2]["total"]
    assert ab["winner"] == 2 and out["script"]["hook_index"] == 2
    script = Script.model_validate(out["script"])
    assert script.word_count == sum(count_words(x.spoken) for x in script.sentences())   # spoken words set VO length
    for f in ("hook_ab.json", "script.json", "script.md"):
        assert (ctx.out_dir / f).exists()
    assert "F2" in (ctx.out_dir / "script.md").read_text(encoding="utf-8")


def test_hook_judge_disqualifies(ctx, log, llm, state):
    hooks = [HookOut(style="question", display="Are AI tools allowed in exams?", spoken="x", fact_ids=["F42"]),
             HookOut(style="surprising_fact", display="90% of students use AI tools.", spoken="x", fact_ids=[]),
             HookOut(style="direct_address", display="You can ask before using AI tools.", spoken="x", fact_ids=[])]
    llm.fixtures["hook_judge_v1"] = lambda v: {"scores": [
        {"hook_index": h["index"], "topic_clarity": 9 if h["display"].startswith(("Are", "90")) else 3,
         "audience_fit": 9, "tone_fit": 9, "unsourced_claim": False, "reason": ""} for h in v["hooks"]]}
    ab = hook_judge.make_node(ctx, log, llm)(_draft_state(state, hooks))["hook_ab"]
    assert ab["hooks"][0]["disqualified"] and ab["hooks"][1]["disqualified"]   # unknown id / number unsourced
    assert ab["winner"] == 2


def test_pick_winner_index_tiebreak():
    rows = [{"index": i, "total": 7.0, "spoken_words": 5, "disqualified": False, "flags": []} for i in range(3)]
    assert hook_judge.pick_winner(rows)[0] == 0


# ---------------------------------------------------------------- storyboarder
def _script_state(state, ctx, log, llm):
    """Run scriptwriter + hook_judge (mock fixtures) so state has a final script."""
    state["extra"] = scriptwriter.make_node(ctx, log, llm)(state)["extra"]
    state.update(hook_judge.make_node(ctx, log, llm)(state))
    return state


def _check_storyboard(sb: dict, n: int, contacts=CONTACTS):
    shots = [ShotOut.model_validate(x) for x in sb["shots"]]
    assert 6 <= len(shots) <= 12 or n + 1 < 6
    assert [i for x in shots for i in x.sentence_indices] == list(range(n))
    assert shots[0].template == "hook_title" and shots[-1].template == "end_card"
    assert all(x.template in TEMPLATES for x in shots)
    assert all(len(t.split()) <= 7 for x in shots for t in x.on_screen_text)
    return shots


def test_storyboard_mock_valid(ctx, log, llm, state):
    _script_state(state, ctx, log, llm)
    sb = storyboarder.make_node(ctx, log, llm)(state)["storyboard"]
    n = len(Script.model_validate(state["script"]).sentences())
    shots = _check_storyboard(sb, n)
    assert sb["remaining_issues"] == []
    assert any(x.template == "do_dont_split" for x in shots)           # C01 allowed vs not allowed
    for x in shots:
        if x.visual_prompt:
            assert x.visual_prompt.endswith("no people, no faces") and "flat vector" not in x.visual_prompt
    dd = [x for x in shots if x.template == "do_dont_split"][0]
    assert any(t.startswith("Don't: ") for t in dd.on_screen_text) and dd.on_screen_text[0].startswith("Do: ")
    assert sb["negative_prompt"] and (ctx.out_dir / "storyboard.json").exists()


def test_storyboard_repair_then_fallback(ctx, log, llm, state):
    _script_state(state, ctx, log, llm)
    calls = []

    def broken(v):
        calls.append(v["feedback"])
        return {"shots": [{"shot_id": 1, "template": "illustration_kenburns", "sentence_indices": [3, 1],
                           "visual_prompt": "x", "on_screen_text": ["a b c d e f g h i"], "fact_ids": []}]}

    llm.fixtures["storyboarder_v1"] = broken
    sb = storyboarder.make_node(ctx, log, llm)(state)["storyboard"]
    assert len(calls) == 2 and any("sentence_indices" in f for f in calls[1])   # retry got the error list
    assert sb["source"] == "deterministic_fallback"
    _check_storyboard(sb, len(Script.model_validate(state["script"]).sentences()))


def test_storyboard_contact_card_whitelist(ctx, log, llm, state):
    _script_state(state, ctx, log, llm)
    n = len(Script.model_validate(state["script"]).sentences())
    shots = [{"shot_id": 1, "template": "hook_title", "sentence_indices": [0], "visual_prompt": "exam hall with students",
              "on_screen_text": ["AI tools in exams"], "fact_ids": []}]
    mid = list(range(1, n - 1))
    size = -(-len(mid) // 4)
    chunks = [mid[i:i + size] for i in range(0, len(mid), size)]
    for k, c in enumerate(chunks):
        shots.append({"shot_id": k + 2, "template": "contact_card" if k == len(chunks) - 1 else "illustration_kenburns",
                      "sentence_indices": c, "visual_prompt": "desk", "fact_ids": [],
                      "on_screen_text": ["Helpline: 1800-000-1111", "Cyber: 1930"] if k == len(chunks) - 1 else ["Exam rules"]})
    shots.append({"shot_id": len(shots) + 1, "template": "end_card", "sentence_indices": [n - 1], "visual_prompt": "",
                  "on_screen_text": ["Read the rules"], "fact_ids": []})
    llm.fixtures["storyboarder_v1"] = lambda v: {"shots": shots}
    sb = storyboarder.make_node(ctx, log, llm)(state)["storyboard"]
    out = _check_storyboard(sb, n)
    card = [x for x in out if x.template == "contact_card"][0]
    assert card.on_screen_text == ["Cyber: 1930"]                       # unlisted number dropped
    hook = out[0]
    assert "student" not in hook.visual_prompt and "no people, no faces" in hook.visual_prompt
    assert all(x.visual_prompt == "" for x in out if x.template not in ("hook_title", "illustration_kenburns"))


def test_fallback_storyboard_shapes():
    sents = [{"index": i, "display": f"Sentence {i} about exams.", "fact_ids": [f"F{i}"] if i % 2 else []}
             for i in range(12)]
    sb = storyboarder.fallback_storyboard(sents, [], 6, 12, do_dont=False)
    assert 6 <= len(sb.shots) <= 12
    assert [i for x in sb.shots for i in x.sentence_indices] == list(range(12))
    assert sb.shots[0].template == "hook_title" and sb.shots[-1].sentence_indices == [11]
    tiny = storyboarder.fallback_storyboard(sents[:3], [], 6, 12, do_dont=False)
    assert [i for x in tiny.shots for i in x.sentence_indices] == [0, 1, 2]
    assert tiny.shots[-1].template == "end_card"
    assert isinstance(StoryboardOut.model_validate(tiny.model_dump()), StoryboardOut)
    assert storyboarder.compose_visual("a desk with text and a logo", style_for("B")) == "a desk with and a"
    assert storyboarder.compose_visual("two students at a desk", style_for("A")) ==         "two simple silhouette at a desk, no people, no faces"
    assert storyboarder.normalize_do_dont(["+ Ask first", "- Use phones", "✗ Share files", "x", "y", "z", "w"]) ==         ["Do: Ask first", "Do: X", "Do: Y", "Don't: Use phones", "Don't: Share files"]
    assert storyboarder.normalize_do_dont(["- Use phones"])[0] == storyboarder.GENERIC_DO
