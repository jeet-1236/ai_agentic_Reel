"""Scriptwriter: 3 hooks (question / surprising_fact / direct_address) + body + CTA from verified facts.

Flow: LLM draft -> code validation (hook styles, fact ids, contacts whitelist, word budget) -> at most one
LLM repair with explicit counts/errors -> deterministic finalisation:
  * unlisted contacts replaced by "the official helpline/website" (zero tolerance),
  * `spoken` = textnorm.to_spoken(display) for every sentence (never trusted from the LLM),
  * over-budget scripts trimmed (connective lines first).
If the LLM fails entirely, a deterministic draft is built from the facts (also the mock-mode fixture).
Writes extra.script_draft (all 3 hooks) + script_draft.json. hook_judge picks the hook and sets `script`.
"""
from __future__ import annotations

import re
from typing import Callable

from core.context import RunContext
from core.llm import LLM, LLMError
from core.logging import RunLogger
from core.rules import load_packs, rules_text
from core.schemas import HookOut, ScriptOut, ScriptSentenceOut
from core.seeds import stage_seed
from core.state import AgentState
from core.textnorm import count_words, short_urls, strip_unlisted_contacts, to_spoken

AGENT = "scriptwriter"
PROMPT_ID = "scriptwriter_v1"
HOOK_STYLES = ("question", "surprising_fact", "direct_address")
WORDS_PER_S = 2.1   # measured on Kaggle 2026-10-22: Kokoro af_heart, 72 words in 34.98 s incl. sentence pauses

CONNECTIVES = [
    "Here's what the official sources say.",
    "Here's what that means for you.",
    "Keep this in mind before you submit your next piece of work.",
    "When in doubt, ask before you act.",
    "Save this so you have it when you need it.",
    "Rules can differ, so always confirm first.",
    "A quick check today keeps you on track.",
    "Share it with a friend who needs it.",
]


# ---------------------------------------------------------------- inputs
def plan_view(state: AgentState) -> dict:
    """Plan fields the creative agents need, with brief-derived defaults when the plan is missing."""
    brief = state.get("brief", {}) or {}
    plan = state.get("plan", {}) or {}
    dur = int(plan.get("duration_s") or brief.get("duration_s") or 45)
    words = dur * WORDS_PER_S
    first = (brief.get("id") or "P")[:1].upper()
    cat = plan.get("category") or (first if first in "ABCDE" else "P")
    return {
        "category": cat,
        "rule_packs": plan.get("rule_packs") or ["base", cat],
        "word_min": int(plan.get("word_budget_min") or round(words * 0.9)),
        "word_max": int(plan.get("word_budget_max") or round(words * 1.1)),
        "shot_min": int(plan.get("shot_count_min") or 6),
        "shot_max": int(plan.get("shot_count_max") or 12),
        "audience_insight": plan.get("audience_insight", "") or brief.get("audience", ""),
        "topic_keywords": plan.get("topic_keywords") or _keywords_from_topic(brief.get("topic", "")),
    }


_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "what", "is", "are", "vs",
         "at", "by", "from", "how", "why", "your", "you", "do", "does", "can", "be", "this", "that", "it"}


def _keywords_from_topic(topic: str) -> list[str]:
    words = [w.strip(":,.—-\"'").lower() for w in topic.split()]
    return [w for w in words if len(w) > 2 and w not in _STOP][:5]


_GENERIC = {"college", "campus", "explainer", "awareness", "guide", "video", "basics", "students", "student"}


def topic_phrase(topic: str) -> str:
    """Short natural subject for template hooks: 'College AI policy explainer: ...' -> 'AI policy'."""
    title = re.split(r"[:—–]", topic or "", maxsplit=1)[0]
    words = [w.strip(",.\"'()") for w in title.split()]
    keep = [w for w in words if w and w.lower() not in _STOP and w.lower() not in _GENERIC]
    return " ".join(keep[:3])


def _relevance(fact: dict, terms: set[str]) -> int:
    toks = set(re.findall(r"[a-z0-9]+", (fact.get("claim", "") + " " + fact.get("quote", "")).lower()))
    return len(toks & terms)


def rank_facts(facts: list[dict], coverage: list[dict], brief: dict) -> list[dict]:
    """Facts covering must_include first (coverage order), then the rest by overlap with topic + must_include.
    Facts with no overlap at all are dropped when relevant ones exist (e.g. library rules for an exam topic)."""
    by_id = {f["fact_id"]: f for f in facts if f.get("claim")}
    terms = {t for t in re.findall(r"[a-z0-9]+", " ".join([brief.get("topic", ""), *brief.get("must_include", [])]).lower())
             if len(t) > 2 and t not in _STOP and t not in _GENERIC}
    ordered: list[dict] = []
    for c in coverage or []:
        for fid in c.get("fact_ids", []):
            if fid in by_id and by_id[fid] not in ordered:
                ordered.append(by_id[fid])
    rest = sorted((f for f in by_id.values() if f not in ordered), key=lambda f: -_relevance(f, terms))
    relevant = [f for f in rest if _relevance(f, terms) > 0]
    return ordered + (relevant if (ordered or relevant) else rest)


def facts_view(facts: list[dict]) -> list[dict]:
    return [{"id": f["fact_id"], "claim": f.get("claim", ""), "quote": f.get("quote", ""), "url": f.get("url", "")}
            for f in facts or []]


def contacts_view(contacts: list[dict]) -> list[dict]:
    return [{"kind": c.get("kind"), "display": c.get("display") or c.get("value"), "label": c.get("label", "")}
            for c in contacts or []]


def coverage_view(coverage: list[dict]) -> list[dict]:
    return [{"item": c.get("item"), "status": c.get("status", "MISSING"), "fact_ids": c.get("fact_ids", [])}
            for c in coverage or []]


# ---------------------------------------------------------------- budget
def budget_report(out: ScriptOut, wmin: int, wmax: int) -> dict:
    """Spoken-word totals for each hook (hook + body + CTA) against [wmin, wmax].

    Spoken words set the voice-over duration (a helpline read digit by digit is many words)."""
    def sw(text: str) -> int:
        return count_words(to_spoken(text))

    hook_w = [sw(h.display) for h in out.hooks]
    body_w = sum(sw(s.display) for s in out.body)
    cta_w = sw(out.cta.display)
    totals = [h + body_w + cta_w for h in hook_w] or [body_w + cta_w]
    ok = all(wmin <= t <= wmax for t in totals)
    msg = ""
    if not ok:
        lo, hi = min(totals), max(totals)
        if hi > wmax:
            msg = (f"Too long: totals per hook are {totals} words (hooks {hook_w}, body {body_w}, CTA {cta_w}). "
                   f"Target {wmin}-{wmax}. Cut about {hi - wmax + 3} words from the body.")
        else:
            msg = (f"Too short: totals per hook are {totals} words (hooks {hook_w}, body {body_w}, CTA {cta_w}). "
                   f"Target {wmin}-{wmax}. Add about {wmin - lo + 3} words of sourced body sentences "
                   f"(or connective lines without new claims).")
    return {"hook_words": hook_w, "body_words": body_w, "cta_words": cta_w, "totals": totals,
            "min": wmin, "max": wmax, "ok": ok, "message": msg}


def _distance(rep: dict) -> int:
    return sum(max(0, rep["min"] - t) + max(0, t - rep["max"]) for t in rep["totals"])


# ---------------------------------------------------------------- validation
def validate_draft(out: ScriptOut, facts: list[dict], contacts: list[dict], wmin: int, wmax: int) -> list[str]:
    """Problems worth one LLM repair. Empty list = good."""
    errs: list[str] = []
    ids = {f["fact_id"] for f in facts or []}
    styles = [h.style for h in out.hooks]
    if len(out.hooks) != 3 or sorted(styles) != sorted(HOOK_STYLES):
        errs.append(f"Write exactly 3 hooks with styles {list(HOOK_STYLES)}; got {styles}.")
    if not out.body:
        errs.append("The body is empty.")
    urls = [f.get("url", "") for f in facts or []]
    for label, s in _labelled(out):
        bad = [i for i in s.fact_ids if i not in ids]
        if bad:
            errs.append(f"{label}: fact_ids {bad} do not exist; use only ids from the facts list.")
        _, removed = strip_unlisted_contacts(s.display, contacts, urls)
        if removed:
            errs.append(f"{label}: contacts {removed} are not in the whitelist; remove them or use a whitelisted one.")
    for h in out.hooks:
        if h.style == "surprising_fact" and not h.fact_ids:
            errs.append("The surprising_fact hook must cite fact_ids.")
    rep = budget_report(out, wmin, wmax)
    if not rep["ok"]:
        errs.append(rep["message"])
    return errs


def _labelled(out: ScriptOut):
    for i, h in enumerate(out.hooks):
        yield f"hook {i} ({h.style})", h
    for i, s in enumerate(out.body):
        yield f"body sentence {i}", s
    yield "CTA", out.cta


# ---------------------------------------------------------------- finalisation
def _fix_hooks(out: ScriptOut, fallback: ScriptOut) -> list[HookOut]:
    """Exactly 3 hooks in HOOK_STYLES order; missing styles come from the deterministic draft."""
    by_style: dict[str, HookOut] = {}
    used: set[int] = set()
    for i, h in enumerate(out.hooks):
        st = h.style.strip().lower().replace(" ", "_").replace("-", "_")
        if st in HOOK_STYLES and st not in by_style and h.display.strip():
            by_style[st] = h.model_copy(update={"style": st})
            used.add(i)
    spare = [h for i, h in enumerate(out.hooks) if i not in used and h.display.strip()]
    hooks = []
    for i, st in enumerate(HOOK_STYLES):
        if st in by_style:
            hooks.append(by_style[st])
        elif spare:
            hooks.append(spare.pop(0).model_copy(update={"style": st}))
        else:
            hooks.append(fallback.hooks[i])
    return hooks


def _sentence(s, contacts: list[dict], urls: list[str], issues: list[str], label: str):
    display, removed = strip_unlisted_contacts(s.display.strip(), contacts, urls)
    display = short_urls(display)
    if removed:
        issues.append(f"{label}: removed unlisted contacts {removed}")
    return s.model_copy(update={"display": display, "spoken": to_spoken(display, contacts),
                                "fact_ids": list(dict.fromkeys(s.fact_ids))})


def trim_to_budget(out: ScriptOut, wmax: int, keep_ids: set[str]) -> tuple[ScriptOut, list[str]]:
    """Drop body sentences until every hook total <= wmax: connective lines first, then factual lines
    whose facts are not needed for must_include coverage, from the end. Keeps >= 2 body sentences."""
    body = list(out.body)
    dropped: list[str] = []

    def total() -> int:
        return max(budget_report(out.model_copy(update={"body": body}), 0, wmax)["totals"])

    for pick in (lambda s: not s.fact_ids, lambda s: not (set(s.fact_ids) & keep_ids), lambda s: True):
        i = len(body) - 1
        while total() > wmax and len(body) > 2 and i >= 0:
            if i < len(body) and pick(body[i]):
                dropped.append(body.pop(i).display)
            i -= 1
    return out.model_copy(update={"body": body}), dropped


def finalize(out: ScriptOut, facts: list[dict], contacts: list[dict], coverage: list[dict],
             wmin: int, wmax: int, fallback: ScriptOut) -> tuple[ScriptOut, list[str]]:
    """Deterministic post-processing that does not depend on the LLM behaving."""
    issues: list[str] = []
    urls = [f.get("url", "") for f in facts or []]
    ids = {f["fact_id"] for f in facts or []}
    hooks = [_sentence(h, contacts, urls, issues, f"hook {i}") for i, h in enumerate(_fix_hooks(out, fallback))]
    body = [_sentence(s, contacts, urls, issues, f"body {i}") for i, s in enumerate(out.body) if s.display.strip()]
    cta = _sentence(out.cta if out.cta.display.strip() else fallback.cta, contacts, urls, issues, "cta")
    # unknown fact ids: keep on hooks (hook_judge disqualifies them), strip elsewhere and report
    for label, s in [*[(f"body {i}", s) for i, s in enumerate(body)], ("cta", cta)]:
        bad = [i for i in s.fact_ids if i not in ids]
        if bad:
            issues.append(f"{label}: removed unknown fact_ids {bad}")
            s.fact_ids = [i for i in s.fact_ids if i in ids]
    res = ScriptOut(hooks=hooks, body=body or fallback.body, cta=cta)
    keep = {i for c in coverage or [] for i in c.get("fact_ids", [])}
    if not budget_report(res, wmin, wmax)["ok"] and max(budget_report(res, wmin, wmax)["totals"]) > wmax:
        res, dropped = trim_to_budget(res, wmax, keep)
        if dropped:
            issues.append(f"trimmed {len(dropped)} body sentences to fit the word budget")
    rep = budget_report(res, wmin, wmax)
    if not rep["ok"]:
        issues.append(f"word budget still off: {rep['totals']} vs {wmin}-{wmax}")
    return res, issues


# ---------------------------------------------------------------- deterministic draft (fallback + mock)
def _short_host(url: str) -> str:
    host = re.sub(r"(?i)^https?://", "", url or "").split("/")[0]
    return re.sub(r"^www\.", "", host)


def draft_from_facts(brief: dict, facts: list[dict], contacts: list[dict], coverage: list[dict],
                     wmin: int, wmax: int, keywords: list[str]) -> ScriptOut:
    """A plain but valid script built only from fact claims (no LLM)."""
    kw = topic_phrase(brief.get("topic", "")) or \
        (keywords or _keywords_from_topic(brief.get("topic", "")) or ["this topic"])[0]
    fv = rank_facts(facts or [], coverage, brief)
    top = fv[:4]
    f0 = min(top, key=lambda f: count_words(f["claim"])) if top else None   # shortest relevant claim = best hook
    hooks = [
        HookOut(style="question", display=f"Confused about {kw}?",
                spoken="", fact_ids=[]),
        HookOut(style="surprising_fact",
                display=(f0["claim"].rstrip(".") + ".") if f0 else f"Most students never check the {kw} rules.",
                spoken="", fact_ids=[f0["fact_id"]] if f0 else []),
        HookOut(style="direct_address", display=f"You should know these {kw} facts.", spoken="", fact_ids=[]),
    ]
    if not f0:   # a "surprising fact" without a source would be an unsourced claim
        hooks[1] = HookOut(style="surprising_fact", display=f"Here is the one {kw} rule to check first.",
                           spoken="", fact_ids=[])
    # CTA: brief CTA + an official host from the whitelist or the first fact source
    url = next((c.get("display") or c.get("value") for c in contacts or [] if c.get("kind") == "url"), "")
    cta_ids: list[str] = []
    if not url and f0:
        url = f0.get("url", "")
        cta_ids = [f0["fact_id"]]
    host = _short_host(url)
    if host and not cta_ids:
        cta_ids = [f["fact_id"] for f in fv if _short_host(f.get("url", "")) == host][:1]
    cta_text = (brief.get("cta") or "Check the official page for details").rstrip(".")
    cta = ScriptSentenceOut(display=f"{cta_text}: {host}." if host else f"{cta_text}.", spoken="", fact_ids=cta_ids)

    # every hook total (hook + body + CTA) must land in [wmin, wmax]
    def sw(text: str) -> int:
        return count_words(to_spoken(text, contacts))

    hw = [sw(h.display) for h in hooks]
    fixed = sw(cta.display)
    body: list[ScriptSentenceOut] = []

    def fits(line: str) -> bool:
        return max(hw) + fixed + sum(sw(s.display) for s in body) + sw(line) <= wmax

    def short() -> bool:
        return min(hw) + fixed + sum(sw(s.display) for s in body) < wmin

    body.append(ScriptSentenceOut(display=CONNECTIVES[0], spoken="", fact_ids=[]))
    for f in [f for f in fv if f is not f0] + ([f0] if f0 else []):
        line = f["claim"].rstrip(".") + "."
        if fits(line):
            body.append(ScriptSentenceOut(display=line, spoken="", fact_ids=[f["fact_id"]]))
    for c in coverage or []:
        line = f"For details on {c.get('item', 'this')}, check the official page."
        if c.get("status") == "MISSING" and fits(line):
            body.append(ScriptSentenceOut(display=line, spoken="", fact_ids=[]))
    for line in CONNECTIVES[1:]:
        if not short():
            break
        if fits(line):
            body.append(ScriptSentenceOut(display=line, spoken="", fact_ids=[]))
    out = ScriptOut(hooks=hooks, body=body, cta=cta)
    for s in [*out.hooks, *out.body, out.cta]:
        s.spoken = to_spoken(s.display, contacts)
    return out


# ---------------------------------------------------------------- mock fixtures
def register_fixtures(llm: LLM, state: AgentState | None = None) -> None:
    """Mock-mode fixture for scriptwriter_v1 (built from the prompt variables; real fact ids)."""

    def fixture(v: dict) -> dict:
        facts = [{"fact_id": f["id"], "claim": f["claim"], "url": f.get("url", "")} for f in v.get("facts", [])]
        out = draft_from_facts(v.get("brief", {}), facts, v.get("_contacts_raw", []), v.get("coverage", []),
                               int(v.get("word_min", 90)), int(v.get("word_max", 110)), v.get("topic_keywords", []))
        return out.model_dump()

    llm.fixtures.setdefault(PROMPT_ID, fixture)


# ---------------------------------------------------------------- node
def write_script(ctx: RunContext, log: RunLogger, llm: LLM, state: AgentState) -> dict:
    """Core of the node (also used by tests)."""
    brief = state.get("brief", {}) or {}
    pv = plan_view(state)
    facts = state.get("facts", []) or []
    contacts = state.get("contacts", []) or []
    coverage = state.get("coverage", []) or []
    loop = int(state.get("loop", 0) or 0)
    feedback = list((state.get("feedback") or {}).get(AGENT, []) or [])
    prev = (state.get("extra") or {}).get("script_draft", {}).get("draft") if loop else None
    if ctx.mock_llm:
        register_fixtures(llm, state)
    packs = load_packs(pv["rule_packs"])
    rules = rules_text(packs, "script") + "\n" + rules_text(packs, "contacts")
    brief_view = {k: brief.get(k) for k in ("id", "category", "topic", "audience", "duration_s", "tone",
                                            "must_include", "must_avoid", "cta")}
    variables = dict(brief=brief_view, audience_insight=pv["audience_insight"], topic_keywords=pv["topic_keywords"],
                     word_min=pv["word_min"], word_max=pv["word_max"], rules=rules, facts=facts_view(facts),
                     contacts=contacts_view(contacts), coverage=coverage_view(coverage),
                     _contacts_raw=contacts)
    fallback = draft_from_facts(brief, facts, contacts, coverage, pv["word_min"], pv["word_max"], pv["topic_keywords"])

    attempts: list[dict] = []
    source = "llm"
    try:
        out = llm.structured(AGENT, PROMPT_ID, ScriptOut, creative=True,
                             seed=stage_seed(ctx.seed, AGENT, 0, loop),
                             feedback=feedback, previous_draft=prev or "", **variables)
        errs = validate_draft(out, facts, contacts, pv["word_min"], pv["word_max"])
        attempts.append({"attempt": 0, "errors": errs, "budget": budget_report(out, pv["word_min"], pv["word_max"])})
        if errs:
            try:
                fixed = llm.structured(AGENT, PROMPT_ID, ScriptOut, creative=True,
                                       seed=stage_seed(ctx.seed, AGENT, 1, loop),
                                       feedback=feedback + ["Fix these problems in the draft below:"] + errs,
                                       previous_draft=out.model_dump(), **variables)
                errs2 = validate_draft(fixed, facts, contacts, pv["word_min"], pv["word_max"])
                attempts.append({"attempt": 1, "errors": errs2,
                                 "budget": budget_report(fixed, pv["word_min"], pv["word_max"])})
                better = (len(errs2), _distance(attempts[1]["budget"])) <= (len(errs), _distance(attempts[0]["budget"]))
                if better:
                    out, source = fixed, "llm_repair"
            except LLMError as e:
                attempts.append({"attempt": 1, "errors": [f"repair failed: {e}"]})
    except LLMError as e:
        log.record(agent=AGENT, tool="fallback", status="degraded", error=str(e)[:500])
        out, source = fallback, "deterministic_fallback"
        attempts.append({"attempt": 0, "errors": [f"llm failed: {e}"]})

    final, issues = finalize(out, facts, contacts, coverage, pv["word_min"], pv["word_max"], fallback)
    budget = budget_report(final, pv["word_min"], pv["word_max"])
    draft = {"draft": final.model_dump(), "source": source, "loop": loop, "attempts": attempts,
             "issues": issues, "budget": budget, "word_min": pv["word_min"], "word_max": pv["word_max"]}
    path = ctx.write_json("script_draft.json", draft)
    log.record(agent=AGENT, tool="finalize", output_path=str(path),
               units={"source": source, "issues": len(issues), "totals": budget["totals"]})
    return {"extra": {"script_draft": draft}}


def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        return write_script(ctx, log, llm, state)

    return node
