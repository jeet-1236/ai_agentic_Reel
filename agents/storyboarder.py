"""Storyboarder: Script -> 6-12 template shots (StoryboardOut). Never fails.

LLM draft -> validate -> one LLM retry with the error list -> deterministic fallback for structural
errors (count, coverage/order of sentence indices, first/last shot). Then code finalisation on every path:
  * on_screen_text items <= 7 words, contact_card values only from the whitelist,
  * shot fact_ids = union of the narrated sentences' fact_ids,
  * visual_prompt = cleaned image subject (+ "no people, no faces" for A/C); the AssetGenerator adds the
    category style preset (prefix/palette/suffix) and the negative prompt, both recorded in storyboard.json
    as style_preset / negative_prompt; only hook_title / illustration_kenburns get an image prompt,
  * allowed-vs-not-allowed content gets a do_dont_split shot; category A gets a helpline contact_card.
Slot conventions (match renderer/templates/*/template.yaml): do_dont_split items "Do: ..." / "Don't: ..."
(1-3 each); contact_card items "Label: value"; end_card = [short CTA, url]; hook_title = [title].
"""
from __future__ import annotations

import re
from typing import Callable

from agents.scriptwriter import plan_view
from core.context import RunContext
from core.llm import LLM, LLMError
from core.logging import RunLogger
from core.rules import load_packs, rules_text, style_for
from core.schemas import TEMPLATES, Script, ShotOut, StoryboardOut
from core.seeds import stage_seed
from core.state import AgentState
from core.textnorm import contact_allowed, find_contacts, shorten

AGENT = "storyboarder"
PROMPT_ID = "storyboarder_v1"
MAX_WORDS = 7
IMAGE_TEMPLATES = {"hook_title", "illustration_kenburns"}
HARD_MIN, HARD_MAX = 6, 12

DONT_RX = re.compile(r"(?i)\b(not allowed|not permitted|prohibited|banned|forbidden|cannot|can't|don't|do not|"
                     r"never|not okay|isn't allowed|is not allowed)\b")
DO_RX = re.compile(r"(?i)\b(allowed|permitted|is fine|okay to|with permission)\b")
PERSON_RX = re.compile(r"(?i)\b(students?|persons?|people|man|men|woman|women|boys?|girls?|teachers?|professors?|"
                       r"victims?|child(?:ren)?|friends?|examiners?|invigilators?|faces?|portraits?|crowds?)\b")
BANNED_IMG_RX = re.compile(r"(?i)\b(text|words?|letters?|captions?|logos?|photo(?:graph)?s?|photorealistic|"
                           r"realistic|signage|typography|watermark|brand)\b")
STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "be", "you", "your",
        "it", "this", "that", "what", "here", "there", "do", "does", "can", "will", "at", "by", "from", "as",
        "if", "so", "but", "not", "about", "check", "know", "need", "s", "here's", "what's", "let's", "it's",
        "should", "these", "those", "means", "say", "says", "official", "sources", "facts"}


# ---------------------------------------------------------------- helpers
def screen_words(text: str) -> int:
    """On-screen word count as the critic counts it (every whitespace token)."""
    return len(text.split())


def shot_range(pv: dict, n_sentences: int) -> tuple[int, int]:
    lo = max(HARD_MIN, pv["shot_min"])
    hi = min(HARD_MAX, pv["shot_max"])
    lo = min(lo, hi)
    return min(lo, n_sentences + 1), hi        # a very short script cannot fill 6 shots


def needs_do_dont(topic: str, sentences: list[dict]) -> bool:
    text = " ".join(s["display"] for s in sentences)
    topic_hit = bool(re.search(r"(?i)allowed|do'?s and don'?ts|dos and don'?ts|can and can'?t", topic))
    return (topic_hit or bool(DO_RX.search(text))) and bool(DONT_RX.search(text))


def subject_from(text: str, n: int = 8) -> str:
    words = [w.strip(".,:;!?\"'()").lower() for w in text.split()]
    content = [w for w in words if w and w not in STOP and not re.search(r"\d|@|\.", w)]
    keys = ", ".join(dict.fromkeys(content[:n]))
    return f"a symbolic scene about {keys}" if keys else "an open notebook on a desk"


SENSITIVE_GUARD = "no people, no faces"


def compose_visual(subject: str, style: dict) -> str:
    """Clean image subject for the AssetGenerator, which adds the preset prefix/palette/suffix and negative
    prompt itself (agents/asset_generator.build_prompt). Here: strip text/logo/photo requests, and for
    sensitive categories swap people for silhouettes and add the explicit guard right after the subject
    (CLIP truncates at 77 tokens, so the guard must not come last)."""
    subj = re.sub(r"(?i)\b(?:no|without)\s+(?:realistic\s+)?(?:people|persons?|faces?|humans?)\b", "", subject)
    subj = BANNED_IMG_RX.sub("", subj)
    for part in (style.get("prefix", ""), style.get("suffix", ""), style.get("palette_hint", "")):
        if part:
            subj = subj.replace(part.strip(" ,"), "")
    if style.get("sensitive"):
        subj = PERSON_RX.sub("simple silhouette", subj)
    subj = re.sub(r"\s*,(\s*,)+", ",", re.sub(r"\s{2,}", " ", subj)).strip(" ,.;")
    subj = subj or "an open notebook on a desk"
    return f"{subj}, {SENSITIVE_GUARD}" if style.get("sensitive") else subj


def contact_lines(contacts: list[dict], kinds: tuple[str, ...] = ("phone", "email", "url"), limit: int = 3) -> list[str]:
    out = []
    for c in contacts:
        if c.get("kind") in kinds and (c.get("display") or c.get("value")):
            label = shorten(c.get("label") or c["kind"].capitalize(), 4)
            out.append(f"{label}: {c.get('display') or c.get('value')}")
        if len(out) >= limit:
            break
    return out


def _contact_line(found: dict, contacts: list[dict]) -> str:
    """'<whitelist label>: <value as written>' for a contact found in a sentence."""
    digits = re.sub(r"\D", "", found["text"])
    for c in contacts:
        vals = {c.get("value", ""), c.get("display", "")}
        same = (found["kind"] == "phone" and bool(digits) and digits in {re.sub(r"\D", "", v) for v in vals}) or \
            found["text"].lower() in {v.lower() for v in vals}
        if same and c.get("label"):
            return f"{shorten(c['label'], 4)}: {found['text']}"
    return f"{found['kind'].capitalize()}: {found['text']}"


DO_PREFIX, DONT_PREFIX = "Do: ", "Don't: "
GENERIC_DO = "Do: Check the official rules first"
_PREFIX_RX = re.compile(r"(?i)^\s*(?:(do\s*:|\+|✓)|(don'?t\s*:?|do not\s*:?|-|✗|never\s*:))\s*")


def do_dont_items(sentences: list[dict]) -> list[str]:
    """Renderer convention (renderer/templates/do_dont_split): "Do: ..." / "Don't: ..." items, 1-3 each."""
    do, dont = [], []
    for s in sentences:
        if not s["fact_ids"]:          # C.allowed_items_sourced: only sourced lines become items
            continue
        body = shorten(s["display"], MAX_WORDS - 1)
        (dont if DONT_RX.search(s["display"]) else do).append(body)
    return normalize_do_dont([DO_PREFIX + t for t in do] + [DONT_PREFIX + t for t in dont])


def normalize_do_dont(items: list[str]) -> list[str]:
    """Map +/-/✓/✗ prefixes to Do:/Don't:, keep <= 3 per side, guarantee one Do item."""
    do, dont = [], []
    for t in items:
        m = _PREFIX_RX.match(t)
        body = t[m.end():].strip() if m else t.strip()
        if not body:
            continue
        is_dont = bool(m and m.group(2)) or (not m and bool(DONT_RX.search(t)))
        if is_dont:
            dont.append(DONT_PREFIX + shorten(body, MAX_WORDS - 1))
        else:
            do.append(DO_PREFIX + shorten(body, MAX_WORDS - 1))
    if dont and not do:
        do = [GENERIC_DO]
    return do[:3] + dont[:3]


# ---------------------------------------------------------------- validation
def validate(sb: StoryboardOut, n: int, lo: int, hi: int, contacts: list[dict], urls: list[str],
             do_dont: bool, category: str) -> tuple[list[str], list[str]]:
    """Returns (structural errors -> fallback if unfixed, soft errors -> fixed in code)."""
    hard, soft = [], []
    shots = sb.shots
    if not lo <= len(shots) <= hi:
        hard.append(f"Use between {lo} and {hi} shots; got {len(shots)}.")
    seq = [i for s in shots for i in s.sentence_indices]
    if seq != list(range(n)):
        hard.append(f"sentence_indices across shots must be exactly 0..{n - 1} once each in order; got {seq}.")
    if shots:
        if shots[0].template not in ("hook_title", "illustration_kenburns") or 0 not in shots[0].sentence_indices:
            hard.append("Shot 1 must be hook_title narrating sentence 0.")
        if shots[-1].template != "end_card":
            hard.append("The last shot must be end_card.")
        if shots[-1].sentence_indices and shots[-1].sentence_indices != [n - 1]:
            hard.append(f"end_card may narrate only the CTA (sentence {n - 1}) or nothing.")
    for k, s in enumerate(shots[:-1] if shots else []):
        if not s.sentence_indices:
            hard.append(f"shot {k + 1} narrates no sentence; only end_card may be silent.")
    for k, s in enumerate(shots):
        long = [t for t in s.on_screen_text if screen_words(t) > MAX_WORDS]
        if long:
            soft.append(f"shot {k + 1}: on_screen_text over {MAX_WORDS} words: {long}")
        for t in s.on_screen_text:
            bad = [c["text"] for c in find_contacts(t, contacts) if not contact_allowed(c, contacts, urls)]
            if bad:
                soft.append(f"shot {k + 1}: contact values not in whitelist: {bad}")
        if s.template == "contact_card" and not any(find_contacts(t, contacts) for t in s.on_screen_text):
            soft.append(f"shot {k + 1}: contact_card shows no whitelisted contact")
    if do_dont and not any(s.template == "do_dont_split" for s in shots):
        soft.append("Script contrasts allowed vs not allowed: use one do_dont_split shot.")
    if category == "A" and any(c.get("kind") == "phone" for c in contacts) and \
            not any(s.template == "contact_card" for s in shots):
        soft.append("Safety topic: add a contact_card with the verified helpline before the end_card.")
    return hard, soft


# ---------------------------------------------------------------- deterministic storyboard
def _partition(items: list[int], k: int, sentences: list[dict] | None = None) -> list[list[int]]:
    """Split consecutive sentence indices into k groups. Connective lines (no fact_ids) ride along with
    the next factual sentence; groups are then merged / split to reach k."""
    if not items:
        return []
    k = max(1, min(k, len(items)))
    units: list[list[int]] = []
    pending: list[int] = []
    for i in items:
        pending.append(i)
        if sentences is None or sentences[i]["fact_ids"]:
            units.append(pending)
            pending = []
    if pending:
        if units:
            units[-1].extend(pending)
        else:
            units.append(pending)
    while len(units) > k:                                   # merge the smallest adjacent pair
        j = min(range(len(units) - 1), key=lambda j: len(units[j]) + len(units[j + 1]))
        units[j:j + 2] = [units[j] + units[j + 1]]
    while len(units) < k:                                   # split the largest group
        j = max(range(len(units)), key=lambda j: len(units[j]))
        if len(units[j]) < 2:
            break
        half = len(units[j]) // 2
        units[j:j + 1] = [units[j][:half], units[j][half:]]
    return units


def fallback_storyboard(sentences: list[dict], contacts: list[dict], lo: int, hi: int, do_dont: bool) -> StoryboardOut:
    """Hook title, illustration shots grouping consecutive sentences, CTA on the end card."""
    n = len(sentences)
    if n == 0:
        return StoryboardOut(shots=[ShotOut(shot_id=1, template="end_card", sentence_indices=[], visual_prompt="",
                                            on_screen_text=[], fact_ids=[])])
    if n == 1:
        return StoryboardOut(shots=[
            ShotOut(shot_id=1, template="hook_title", sentence_indices=[0], visual_prompt=subject_from(sentences[0]["display"]),
                    on_screen_text=[shorten(sentences[0]["display"])], fact_ids=sentences[0]["fact_ids"]),
            ShotOut(shot_id=2, template="end_card", sentence_indices=[], visual_prompt="", on_screen_text=[], fact_ids=[])])
    middle = list(range(1, n - 1))
    factual = sum(1 for i in middle if sentences[i]["fact_ids"])
    target = max(lo, min(hi, factual + 2))
    cta_on_end = True
    if target - 2 > len(middle):                       # not enough body sentences: CTA gets its own shot
        middle, cta_on_end = list(range(1, n)), False
    groups = _partition(middle, target - 2, sentences) if middle else []
    shots: list[ShotOut] = [ShotOut(shot_id=1, template="hook_title", sentence_indices=[0],
                                    visual_prompt=subject_from(sentences[0]["display"]),
                                    on_screen_text=[shorten(sentences[0]["display"])],
                                    fact_ids=list(sentences[0]["fact_ids"]))]
    dd_used = False
    for g in groups:
        ss = [sentences[i] for i in g]
        text = " ".join(s["display"] for s in ss)
        fids = list(dict.fromkeys(f for s in ss for f in s["fact_ids"]))
        listed = [c for c in find_contacts(text, contacts) if contact_allowed(c, contacts)]
        if listed:
            shot = ShotOut(shot_id=0, template="contact_card", sentence_indices=g, visual_prompt="",
                           on_screen_text=[_contact_line(c, contacts) for c in listed][:3], fact_ids=fids)
        elif do_dont and not dd_used and DONT_RX.search(text):
            dd_used = True
            shot = ShotOut(shot_id=0, template="do_dont_split", sentence_indices=g, visual_prompt="",
                           on_screen_text=do_dont_items(ss), fact_ids=fids)
        else:
            factual = [s for s in ss if s["fact_ids"]]
            shot = ShotOut(shot_id=0, template="illustration_kenburns", sentence_indices=g,
                           visual_prompt=subject_from(text),
                           on_screen_text=[shorten(factual[0]["display"])] if factual else [], fact_ids=fids)
        shots.append(shot)
    end_idx = [n - 1] if cta_on_end else []
    cta = sentences[-1]["display"]
    links = [c["text"] for c in find_contacts(cta, contacts) if c["kind"] == "url"]
    cta_text = shorten(re.sub(r"[:\-–]?\s*" + re.escape(links[0]), "", cta)) if links else shorten(cta)
    shots.append(ShotOut(shot_id=0, template="end_card", sentence_indices=end_idx, visual_prompt="",
                         on_screen_text=[t for t in (cta_text, *links[:1]) if t],
                         fact_ids=list(sentences[-1]["fact_ids"]) if cta_on_end else []))
    for k, s in enumerate(shots, start=1):
        s.shot_id = k
    return StoryboardOut(shots=shots)


# ---------------------------------------------------------------- finalisation (all paths)
def finalize(sb: StoryboardOut, sentences: list[dict], contacts: list[dict], urls: list[str], style: dict,
             do_dont: bool, category: str) -> tuple[StoryboardOut, list[str]]:
    notes: list[str] = []
    shots = [s.model_copy(deep=True) for s in sb.shots]
    n = len(sentences)
    # do/don't contrast without a do_dont_split shot -> convert the shot with the most "don't" lines
    if do_dont and not any(s.template == "do_dont_split" for s in shots) and len(shots) > 2:
        cands = [(sum(bool(DONT_RX.search(sentences[i]["display"])) for i in s.sentence_indices if i < n), k)
                 for k, s in enumerate(shots[1:-1], start=1)]
        best = max(cands, default=(0, 0))
        if best[0] > 0:
            k = best[1]
            shots[k].template = "do_dont_split"
            shots[k].on_screen_text = do_dont_items([sentences[i] for i in shots[k].sentence_indices if i < n])
            notes.append(f"shot {k + 1} -> do_dont_split")
    # safety topics: helpline card right before the end card
    phones = contact_lines(contacts, ("phone",))
    if category == "A" and phones and not any(s.template == "contact_card" for s in shots) and len(shots) > 2:
        shots[-2].template = "contact_card"
        shots[-2].on_screen_text = phones
        notes.append(f"shot {len(shots) - 1} -> contact_card (helpline)")
    for k, s in enumerate(shots, start=1):
        s.shot_id = k
        s.sentence_indices = [i for i in s.sentence_indices if 0 <= i < n]
        s.fact_ids = list(dict.fromkeys(f for i in s.sentence_indices for f in sentences[i]["fact_ids"]))
        texts = []
        for t in s.on_screen_text:
            bad = [c for c in find_contacts(t, contacts) if not contact_allowed(c, contacts, urls)]
            if bad:
                notes.append(f"shot {k}: dropped unlisted contact text {t!r}")
                continue
            if screen_words(t) > MAX_WORDS and s.template != "do_dont_split":
                notes.append(f"shot {k}: shortened {t!r}")
                t = shorten(t, MAX_WORDS)
            if t.strip():
                texts.append(t.strip())
        if s.template == "do_dont_split":
            texts = normalize_do_dont(texts)
        s.on_screen_text = texts
        if s.template == "contact_card" and not any(find_contacts(t, contacts) for t in texts):
            lines = contact_lines(contacts)
            if lines:
                s.on_screen_text = lines
            else:
                s.template = "illustration_kenburns"
                s.visual_prompt = s.visual_prompt or subject_from(" ".join(sentences[i]["display"] for i in s.sentence_indices))
                notes.append(f"shot {k}: contact_card without whitelisted contacts -> illustration")
        if s.template in IMAGE_TEMPLATES:
            subj = s.visual_prompt or subject_from(" ".join(sentences[i]["display"] for i in s.sentence_indices))
            s.visual_prompt = compose_visual(subj, style)
        else:
            s.visual_prompt = ""
    return StoryboardOut(shots=shots), notes


# ---------------------------------------------------------------- mock fixture
def register_fixtures(llm: LLM, state: AgentState | None = None) -> None:
    def fixture(v: dict) -> dict:
        sb = fallback_storyboard(v["sentences"], v.get("_contacts_raw", []), int(v["shot_min"]), int(v["shot_max"]),
                                 bool(v.get("_do_dont")))
        return sb.model_dump()

    llm.fixtures.setdefault(PROMPT_ID, fixture)


# ---------------------------------------------------------------- node
def build_storyboard(ctx: RunContext, log: RunLogger, llm: LLM, state: AgentState) -> dict:
    brief = state.get("brief", {}) or {}
    pv = plan_view(state)
    script = Script.model_validate(state["script"])
    sentences = [{"index": i, "display": s.display, "fact_ids": list(s.fact_ids)}
                 for i, s in enumerate(script.sentences())]
    n = len(sentences)
    contacts = state.get("contacts", []) or []
    urls = [f.get("url", "") for f in state.get("facts", []) or []]
    category = pv["category"]
    packs = load_packs(pv["rule_packs"])
    style = style_for(category, packs.get("style_preset") or None)
    lo, hi = shot_range(pv, n)
    do_dont = needs_do_dont(brief.get("topic", ""), sentences)
    loop = int(state.get("loop", 0) or 0)
    feedback = list((state.get("feedback") or {}).get(AGENT, []) or [])
    if ctx.mock_llm:
        register_fixtures(llm, state)

    style_desc = f"{style['name']}: {style['prefix']} ... {style['suffix']}"
    rules = rules_text(packs, "storyboard")
    cview = [{"kind": c.get("kind"), "display": c.get("display") or c.get("value"), "label": c.get("label", "")}
             for c in contacts]
    variables = dict(topic=brief.get("topic", ""), style=style_desc, sensitive=style["sensitive"], rules=rules,
                     sentences=sentences, contacts=cview, shot_min=lo, shot_max=hi,
                     _contacts_raw=contacts, _do_dont=do_dont, category=category)
    attempts: list[dict] = []
    sb: StoryboardOut | None = None
    source = "llm"
    for attempt in range(2):
        fb = feedback + (attempts[-1]["hard"] + attempts[-1]["soft"] if attempts else [])
        try:
            cand = llm.structured(AGENT, PROMPT_ID, StoryboardOut, creative=False,
                                  seed=stage_seed(ctx.seed, AGENT, attempt, loop), feedback=fb, **variables)
        except LLMError as e:
            attempts.append({"attempt": attempt, "hard": [f"llm failed: {e}"], "soft": []})
            continue
        hard, soft = validate(cand, n, lo, hi, contacts, urls, do_dont, category)
        attempts.append({"attempt": attempt, "hard": hard, "soft": soft})
        if not hard:                 # structurally valid; a retry is only spent on remaining soft errors
            sb, source = cand, ("llm" if attempt == 0 else "llm_repair")
            if not soft:
                break
    if sb is None:
        sb, source = fallback_storyboard(sentences, contacts, lo, hi, do_dont), "deterministic_fallback"
        log.record(agent=AGENT, tool="fallback", status="degraded",
                   units={"errors": attempts[-1]["hard"][:5] if attempts else []})

    final, notes = finalize(sb, sentences, contacts, urls, style, do_dont, category)
    hard, soft = validate(final, n, lo, hi, contacts, urls, do_dont, category)
    out = final.model_dump(mode="json")
    out.update({"style_preset": style["name"], "negative_prompt": style["negative"], "source": source,
                "notes": notes, "remaining_issues": hard + soft, "attempts": attempts})
    p = ctx.write_json("storyboard.json", out)
    log.record(agent=AGENT, tool="finalize", output_path=str(p),
               units={"source": source, "shots": len(final.shots), "notes": len(notes), "issues": len(hard + soft)})
    return {"storyboard": out}


def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        return build_storyboard(ctx, log, llm, state)

    return node


__all__ = ["make_node", "build_storyboard", "fallback_storyboard", "validate", "finalize", "compose_visual",
           "register_fixtures", "TEMPLATES"]
