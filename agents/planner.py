"""Planner: brief -> plan.json. Code decides format, budgets and rule packs; the LLM drafts queries/insight."""
from __future__ import annotations

import re
from typing import Callable

from core.context import RunContext
from core.llm import LLM
from core.logging import RunLogger
from core.schemas import Brief, Plan, PlanOut, StageBudget
from core.seeds import stage_seed
from core.sources import STOPWORDS
from core.state import AgentState

AGENT = "planner"
PROMPT_ID = "planner_v1"
WORDS_PER_S = 2.1   # measured on Kaggle 2026-10-22: Kokoro af_heart, 72 words in 34.98 s incl. sentence pauses
SIZES = {"9x16": (1080, 1920), "16x9": (1920, 1080)}
MANDATORY_REVIEW = {"A", "C"}
# PLAN §14 (minutes, retry caps)
BUDGETS = [("planner", 1, 1), ("researcher", 5, 1), ("fact_curator", 5, 1), ("scriptwriter", 3, 3),
           ("hook_judge", 1, 1), ("storyboarder", 2, 3), ("asset_generator", 8, 2), ("editor", 6, 2),
           ("critic", 5, 3), ("human_review", 10, 1), ("export", 15, 1)]
MIN_QUERIES, MAX_QUERIES = 8, 12
END_CARD_S, VO_TAIL_S = 4.0, 0.3    # keep in sync with agents/editor.py END_CARD_NOMINAL_S / VO_TAIL_S


def compute_plan_fields(brief: Brief) -> dict:
    """Everything in the plan that code (not the LLM) decides."""
    cat = brief.category_code
    w, h = SIZES[brief.aspect]
    # the voice-over runs before the ~4 s end card (plus a short tail), so size the words for that window
    speech_s = max(20.0, brief.duration_s - END_CARD_S - VO_TAIL_S)
    words = speech_s * WORDS_PER_S
    return {
        "brief_id": brief.id, "category": cat, "rule_packs": ["base", cat], "aspect": brief.aspect,
        "width": w, "height": h, "fps": 30, "duration_s": brief.duration_s,
        "word_budget_min": int(round(words * 0.9)), "word_budget_max": int(round(words * 1.1)),
        "shot_count_min": 6, "shot_count_max": max(6, min(12, round(brief.duration_s / 5))),
        "mandatory_review": cat in MANDATORY_REVIEW,
        "budgets": [StageBudget(stage=s, minutes=m, max_retries=r) for s, m, r in BUDGETS],
    }


def _clean(q: str) -> str:
    q = re.sub(r"\bsite:\S+", "", q).replace('"', " ")
    return re.sub(r"\s+", " ", q).strip(" .?")


def merge_queries(llm_queries: list[str], brief: Brief) -> list[str]:
    """must_include items + topic always searched; LLM queries fill up to MAX_QUERIES; deduped, ordered."""
    must = [brief.topic, *brief.must_include]
    extra = [brief.cta, f"{brief.topic} official circular", f"{brief.category} official guidelines students"]
    out, seen = [], set()
    for q in [*must, *llm_queries, *extra]:
        q = _clean(q or "")
        key = q.lower()
        if q and key not in seen and not key.startswith("mock "):
            seen.add(key)
            out.append(q)
    return out[:max(MAX_QUERIES, len(must))]


def topic_keywords(llm_kw: list[str], brief: Brief) -> list[str]:
    kws = [k.strip() for k in llm_kw if k.strip() and not k.lower().startswith("mock ")]
    for t in re.findall(r"[A-Za-z][A-Za-z-]{2,}", brief.topic):
        if t.lower() not in STOPWORDS and t.lower() not in {k.lower() for k in kws}:
            kws.append(t.lower())
    return kws[:8]


def mock_plan_fixture(variables: dict) -> dict:
    """Deterministic PlanOut for --mock-llm runs, derived only from the brief."""
    b = variables["brief"]
    items = list(b.get("must_include", []))
    return {
        "queries": [b["topic"], *items, f"{b['topic']} rules", f"{b['category']} official notice", b.get("cta", "")],
        "audience_insight": f"{b['audience']} want a short, clear answer about: {b['topic']}.",
        "success_criteria": [*(f"Covers: {i}" for i in items), *(f"Avoids: {a}" for a in b.get("must_avoid", [])),
                             f"Ends with CTA: {b.get('cta', '')}"],
        "topic_keywords": re.findall(r"[A-Za-z]{4,}", b["topic"])[:5],
    }


def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    llm.fixtures.setdefault(PROMPT_ID, mock_plan_fixture)

    def node(state: AgentState) -> dict:
        brief = Brief.model_validate(state["brief"])
        fields = compute_plan_fields(brief)
        out = llm.structured(AGENT, PROMPT_ID, PlanOut, seed=stage_seed(ctx.seed, AGENT, 0, state.get("loop", 0)),
                             brief=brief.model_dump(), rule_packs=fields["rule_packs"],
                             word_min=fields["word_budget_min"], word_max=fields["word_budget_max"],
                             duration_s=brief.duration_s, feedback=state.get("feedback", {}).get(AGENT, []))
        criteria = [c for c in out.success_criteria if c.strip() and not c.startswith("mock ")]
        if not criteria:
            criteria = [f"Covers: {i}" for i in brief.must_include] + [f"Avoids: {a}" for a in brief.must_avoid]
        plan = Plan(**fields, queries=merge_queries(out.queries, brief),
                    audience_insight=out.audience_insight, success_criteria=criteria,
                    topic_keywords=topic_keywords(out.topic_keywords, brief))
        ctx.write_json("plan.json", plan)
        return {"plan": plan.model_dump()}

    return node
