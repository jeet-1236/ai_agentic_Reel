"""HookJudge (P4 Auto A/B hooks): score the 3 draft hooks, pick one, build the final Script.

- Two LLM passes with the hook order reversed (position-bias removal); per-criterion scores are averaged.
- total = (2*topic_clarity + audience_fit + tone_fit) / 4, +0.5 when a topic keyword is in the first
  7 words (~3 s at 2.4 words/s, the critic's hook check).
- Disqualified: unsourced_claim in any pass, fact_ids not in facts, or a number with no fact_ids.
- Ranking: highest total (2 dp); ties -> fewest spoken words -> lowest index. If every hook is
  disqualified, the least-flagged one wins and `selection` says so.
Writes hook_ab.json, script.json, script.md; returns {"script", "hook_ab"}.
"""
from __future__ import annotations

import re
from typing import Callable

from agents.scriptwriter import plan_view
from core.context import RunContext
from core.llm import LLM, LLMError
from core.logging import RunLogger
from core.rules import load_packs, rules_text
from core.schemas import HookJudgeOut, HookOut, Script, ScriptOut
from core.seeds import stage_seed
from core.state import AgentState
from core.textnorm import count_words

AGENT = "hook_judge"
PROMPT_ID = "hook_judge_v1"
CRITERIA = ("topic_clarity", "audience_fit", "tone_fit")
KEYWORD_WINDOW = 7
KEYWORD_BONUS = 0.5
NEUTRAL = 5


# ---------------------------------------------------------------- scoring
def keyword_early(text: str, keywords: list[str], window: int = KEYWORD_WINDOW) -> bool:
    first = " ".join(re.findall(r"[a-z0-9]+", text.lower())[:window])
    for kw in keywords or []:
        toks = re.findall(r"[a-z0-9]+", kw.lower())
        if toks and re.search(r"\b" + r"\s+".join(map(re.escape, toks)), first):
            return True
    return False


def code_flags(hook: HookOut, fact_ids: set[str]) -> list[str]:
    flags = []
    bad = [i for i in hook.fact_ids if i not in fact_ids]
    if bad:
        flags.append(f"unknown fact_ids {bad}")
    if re.search(r"\d", hook.display) and not hook.fact_ids:
        flags.append("number without fact_ids")
    return flags


def aggregate(hooks: list[HookOut], passes: list[dict], fact_ids: set[str], keywords: list[str]) -> list[dict]:
    """passes: [{"order": [orig idx by position], "scores": [HookScoreOut dumps]}] -> per-hook rows."""
    rows = []
    for idx, h in enumerate(hooks):
        per_pass = []
        for p in passes:
            pos = p["order"].index(idx) if idx in p["order"] else -1
            sc = next((s for s in p["scores"] if s.get("hook_index") == pos), None)
            per_pass.append({"pass": p["pass"], "position": pos, "score": sc})
        got = [pp["score"] for pp in per_pass if pp["score"]]
        mean = {c: round(sum(_clamp(s[c]) for s in got) / len(got), 3) if got else float(NEUTRAL) for c in CRITERIA}
        unsourced_votes = sum(1 for s in got if s.get("unsourced_claim"))
        early = keyword_early(h.spoken or h.display, keywords) or keyword_early(h.display, keywords)
        total = (2 * mean["topic_clarity"] + mean["audience_fit"] + mean["tone_fit"]) / 4 + (KEYWORD_BONUS if early else 0)
        flags = code_flags(h, fact_ids) + (["unsourced_claim (judge)"] if unsourced_votes else [])
        rows.append({
            "index": idx, "style": h.style, "display": h.display, "spoken": h.spoken, "fact_ids": h.fact_ids,
            "spoken_words": count_words(h.spoken or h.display), "keyword_in_3s": early,
            "passes": per_pass, "mean": mean, "unsourced_votes": unsourced_votes,
            "total": round(total, 2), "disqualified": bool(flags), "flags": flags,
        })
    return rows


def _clamp(v) -> float:
    try:
        return float(min(10, max(1, int(v))))
    except (TypeError, ValueError):
        return float(NEUTRAL)


def pick_winner(rows: list[dict]) -> tuple[int, str]:
    ok = [r for r in rows if not r["disqualified"]]
    if ok:
        best = min(ok, key=lambda r: (-r["total"], r["spoken_words"], r["index"]))
        return best["index"], "best total among qualified hooks (ties: fewest spoken words, then index)"
    best = min(rows, key=lambda r: (len(r["flags"]), -r["total"], r["spoken_words"], r["index"]))
    return best["index"], "ALL hooks disqualified: least-flagged hook used; critic/review must check it"


# ---------------------------------------------------------------- mock fixture
def register_fixtures(llm: LLM, state: AgentState | None = None) -> None:
    """Deterministic heuristic judge for mock mode (keyword early -> clearer; shorter -> better fit)."""

    def fixture(v: dict) -> dict:
        kws = v.get("topic_keywords", [])
        ids = {f["id"] for f in v.get("facts", [])}
        scores = []
        for h in v.get("hooks", []):
            early = keyword_early(h["display"], kws)
            n = count_words(h["display"])
            scores.append({
                "hook_index": h["index"],
                "topic_clarity": 8 if early else 5,
                "audience_fit": 8 if h["style"] == "direct_address" else 7,
                "tone_fit": 7 if n <= 14 else 5,
                "unsourced_claim": bool(re.search(r"\d", h["display"])) and not (set(h["fact_ids"]) & ids),
                "reason": "mock heuristic",
            })
        return {"scores": scores}

    llm.fixtures.setdefault(PROMPT_ID, fixture)


# ---------------------------------------------------------------- outputs
def script_markdown(script: Script, facts: list[dict], brief: dict, hook_rows: list[dict]) -> str:
    by_id = {f["fact_id"]: f for f in facts}
    lines = [f"# Script: {brief.get('id', '')} {brief.get('topic', '')}", "",
             f"Words: {script.word_count} · winning hook #{script.hook_index} ({script.hook.style})", ""]
    lines += ["| # | Role | Display (captions) | Spoken (TTS) | Facts |", "|---|---|---|---|---|"]
    roles = ["hook"] + ["body"] * len(script.body) + ["cta"]
    for i, (role, s) in enumerate(zip(roles, script.sentences())):
        fids = ", ".join(s.fact_ids) or "—"
        lines.append(f"| {i} | {role} | {s.display.replace('|', '/')} | {s.spoken.replace('|', '/')} | {fids} |")
    used = list(dict.fromkeys(i for s in script.sentences() for i in s.fact_ids))
    lines += ["", "## Sources", ""]
    for fid in used:
        f = by_id.get(fid)
        if f:
            lines.append(f"- **{fid}** {f.get('claim', '')}  \n  > \"{f.get('quote', '')}\"  \n  "
                         f"[{f.get('url', '')}]({f.get('url', '')}) ({f.get('source_id', '')} {f.get('chunk_id', '')})")
        else:
            lines.append(f"- **{fid}** NOT FOUND in facts.json")
    lines += ["", "## Hook A/B", "", "| # | Style | Hook | Mean (clarity/audience/tone) | Total | Status |",
              "|---|---|---|---|---|---|"]
    for r in hook_rows:
        m = r["mean"]
        status = "WINNER" if r["index"] == script.hook_index else ("disqualified: " + "; ".join(r["flags"]) if r["disqualified"] else "")
        lines.append(f"| {r['index']} | {r['style']} | {r['display']} | {m['topic_clarity']}/{m['audience_fit']}/"
                     f"{m['tone_fit']} | {r['total']} | {status} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- node
def judge_hooks(ctx: RunContext, log: RunLogger, llm: LLM, state: AgentState) -> dict:
    brief = state.get("brief", {}) or {}
    pv = plan_view(state)
    facts = state.get("facts", []) or []
    fact_ids = {f["fact_id"] for f in facts}
    draft_info = (state.get("extra") or {}).get("script_draft") or {}
    if not draft_info.get("draft"):
        raise RuntimeError("hook_judge needs extra.script_draft from the scriptwriter")
    draft = ScriptOut.model_validate(draft_info["draft"])
    hooks = draft.hooks
    loop = int(state.get("loop", 0) or 0)
    if ctx.mock_llm:
        register_fixtures(llm, state)
    rules = rules_text(load_packs(pv["rule_packs"]), "script")
    fv = [{"id": f["fact_id"], "claim": f.get("claim", "")} for f in facts]

    n = len(hooks)
    passes: list[dict] = []
    for p, order in enumerate([list(range(n)), list(reversed(range(n)))]):
        payload = [{"index": pos, "style": hooks[i].style, "display": hooks[i].display, "fact_ids": hooks[i].fact_ids}
                   for pos, i in enumerate(order)]
        try:
            out = llm.structured(AGENT, PROMPT_ID, HookJudgeOut, creative=False,
                                 seed=stage_seed(ctx.seed, AGENT, p, loop),
                                 topic=brief.get("topic", ""), audience=brief.get("audience", ""),
                                 tone=brief.get("tone", ""), topic_keywords=pv["topic_keywords"],
                                 rules=rules, facts=fv, hooks=payload)
            scores, err = [s.model_dump() for s in out.scores], ""
        except LLMError as e:
            scores, err = [], str(e)[:300]
        passes.append({"pass": p, "order": order, "scores": scores, "error": err})

    rows = aggregate(hooks, passes, fact_ids, pv["topic_keywords"])
    win, why = pick_winner(rows)
    hook = hooks[win]
    if rows[win]["disqualified"]:                       # never ship a citation to a fact that doesn't exist
        hook = hook.model_copy(update={"fact_ids": [i for i in hook.fact_ids if i in fact_ids]})
    spoken_words = count_words(hook.spoken) + sum(count_words(s.spoken) for s in draft.body) + \
        count_words(draft.cta.spoken)
    # word_count = spoken words: they set the VO duration the plan's word budget is derived from
    script = Script(hook=hook, hook_index=win, body=draft.body, cta=draft.cta, word_count=spoken_words)

    hook_ab = {"hooks": rows, "passes": passes, "winner": win, "selection": why,
               "weights": {"topic_clarity": 2, "audience_fit": 1, "tone_fit": 1, "keyword_bonus": KEYWORD_BONUS},
               "display_word_count": sum(count_words(s.display) for s in script.sentences()),
               "spoken_word_count": spoken_words,
               "loop": loop}
    ctx.write_json("hook_ab.json", hook_ab)
    p = ctx.write_json("script.json", script)
    md = ctx.artifact("script.md")
    md.write_text(script_markdown(script, facts, brief, rows), encoding="utf-8")
    log.record(agent=AGENT, tool="select_hook", output_path=str(p),
               units={"winner": win, "totals": [r["total"] for r in rows],
                      "disqualified": [r["index"] for r in rows if r["disqualified"]]})
    return {"script": script.model_dump(mode="json"), "hook_ab": hook_ab}


def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        return judge_hooks(ctx, log, llm, state)

    return node
