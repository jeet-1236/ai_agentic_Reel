"""HumanReview (PLAN §6.9): mandatory for A/C, and the landing point for runs the critic could not fix.

Before `interrupt()` the node only does idempotent work (it re-runs on resume): it writes
outputs/<run>/review/packet.json + packet.md. The resume value is validated as ReviewDecision and logged as a
review checkpoint (NOT an intervention).
"""
from __future__ import annotations

import json
from typing import Any, Callable

from langgraph.types import interrupt

from core.context import RunContext
from core.llm import LLM
from core.logging import RunLogger
from core.schemas import ReviewDecision, Script
from core.state import AgentState

REJECT_TARGETS = {"researcher", "scriptwriter", "storyboarder", "asset_generator", "editor"}


def build_packet(ctx: RunContext, state: AgentState) -> dict:
    """Review packet: video, script with fact ids + URLs, claims table, contacts, coverage gaps, QA failures."""
    facts = {f["fact_id"]: f for f in state.get("facts") or []}
    qa = state.get("qa_report") or {}
    nli = {c["check_id"]: c for c in qa.get("checks", []) if c["check_id"].startswith("nli:")}
    script_rows, claims = [], []
    if state.get("script"):
        for i, s in enumerate(Script.model_validate(state["script"]).sentences()):
            urls = sorted({facts[f]["url"] for f in s.fact_ids if f in facts})
            script_rows.append({"i": i, "display": s.display, "spoken": s.spoken, "fact_ids": s.fact_ids, "urls": urls})
            for f in s.fact_ids:
                fact = facts.get(f, {})
                claims.append({"sentence": i, "text": s.display, "fact_id": f, "quote": fact.get("quote", "UNKNOWN FACT ID"),
                               "url": fact.get("url", ""), "source_id": fact.get("source_id", ""),
                               "nli": nli.get(f"nli:s{i}", {}).get("value", "")})
    contacts = [{"kind": c.get("kind"), "display": c.get("display"), "label": c.get("label", ""),
                 "authority": c.get("authority"), "sources": c.get("source_ids", []), "conflict": c.get("conflict_note", "")}
                for c in state.get("contacts") or []]
    failures = [c for c in qa.get("checks", []) if not c.get("passed")]
    return {
        "run_id": ctx.run_id,
        "brief_id": (state.get("brief") or {}).get("id", ""),
        "category": (state.get("plan") or {}).get("category", ""),
        "mandatory_review": bool((state.get("plan") or {}).get("mandatory_review", False)),
        "video_path": state.get("video_path", ""),
        "srt_path": state.get("srt_path", ""),
        "loop": state.get("loop", 0),
        "qa_passed": bool(qa.get("passed", False)),
        "qa_summary": (state.get("extra") or {}).get("qa_summary", {}),
        "qa_failures": failures,
        "script": script_rows,
        "claims": claims,
        "contacts": contacts,
        "coverage_missing": [c.get("item") for c in state.get("coverage") or [] if c.get("status") == "MISSING"],
        "sources": [{"source_id": s.get("source_id"), "url": s.get("final_url") or s.get("url"), "title": s.get("title", ""),
                     "authority": s.get("authority", "")} for s in state.get("sources") or []],
    }


def packet_markdown(p: dict) -> str:
    def esc(s: Any) -> str:
        return str(s).replace("|", "\\|").replace("\n", " ")

    L = [f"# Review packet: {p['run_id']} ({p['brief_id']}, category {p['category']})", "",
         f"- Video: `{p['video_path']}`", f"- Captions: `{p['srt_path']}`",
         f"- QA passed: **{p['qa_passed']}** (loop {p['loop']}) {esc(p['qa_summary'].get('reason', ''))}", "",
         "## Script", "", "| # | Caption text | Fact ids | Sources |", "|---|---|---|---|"]
    L += [f"| {r['i']} | {esc(r['display'])} | {', '.join(r['fact_ids'])} | {' '.join(f'[link]({u})' for u in r['urls'])} |"
          for r in p["script"]]
    L += ["", "## Claims", "", "| Sentence | Fact | Quote | Source | NLI |", "|---|---|---|---|---|"]
    L += [f"| {c['sentence']} | {c['fact_id']} | {esc(c['quote'])[:240]} | [{c['source_id']}]({c['url']}) | {c['nli']} |"
          for c in p["claims"]]
    L += ["", "## Contacts (whitelist, precedence applied)", "", "| Kind | Contact | Label | Authority | Sources | Conflict |",
          "|---|---|---|---|---|---|"]
    L += [f"| {c['kind']} | {esc(c['display'])} | {esc(c['label'])} | {c['authority']} | {', '.join(c['sources'])} | {esc(c['conflict'])} |"
          for c in p["contacts"]]
    L += ["", "## Coverage gaps (MISSING, never invented)", ""] + [f"- {m}" for m in p["coverage_missing"]] + (
        [] if p["coverage_missing"] else ["- none"])
    L += ["", "## QA failures", ""] + [f"- **{c['check_id']}** ({c.get('severity', 'hard')}, -> {c['responsible_agent']}): "
                                        f"{esc(c.get('value', ''))} vs {esc(c.get('threshold', ''))}. {esc(c.get('evidence', ''))[:300]}"
                                        for c in p["qa_failures"]] + ([] if p["qa_failures"] else ["- none"])
    L += ["", "## Decision", "", "Approve: `python run.py --resume <run_id> --approve`  ",
          "Reject: `python run.py --resume <run_id> --reject \"note\" --target scriptwriter`", ""]
    return "\n".join(L)


def write_packet(ctx: RunContext, packet: dict) -> dict:
    """Idempotent writes (safe to repeat on resume)."""
    d = ctx.out_dir / "review"
    d.mkdir(parents=True, exist_ok=True)
    pj, pm = d / "packet.json", d / "packet.md"
    pj.write_text(json.dumps(packet, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pm.write_text(packet_markdown(packet), encoding="utf-8")
    return {"packet_json": str(pj), "packet_md": str(pm)}


def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        plan, qa = state.get("plan") or {}, state.get("qa_report") or {}
        if not plan.get("mandatory_review") and qa.get("passed"):
            return {"route": "export"}
        packet = build_packet(ctx, state)
        paths = write_packet(ctx, packet)
        summary = {"run_id": ctx.run_id, **paths, "video_path": packet["video_path"], "qa_passed": packet["qa_passed"],
                   "mandatory_review": packet["mandatory_review"],
                   "qa_failures": [c["check_id"] for c in packet["qa_failures"]],
                   "coverage_missing": packet["coverage_missing"],
                   "respond_with": {"approved": "bool", "note": "str", "target_agent": sorted(REJECT_TARGETS)}}
        value = interrupt(summary)
        decision = ReviewDecision.model_validate(value if not isinstance(value, ReviewDecision) else value.model_dump())
        log.review_checkpoint(decision.model_dump())
        if decision.approved:
            return {"review": decision.model_dump(), "route": "export"}
        target = decision.target_agent if decision.target_agent in REJECT_TARGETS else "scriptwriter"
        note = decision.note or "Rejected at human review (no note)."
        return {"review": decision.model_dump(), "route": target, "feedback": {target: [f"[human_review] {note}"]},
                "loop": int(state.get("loop", 0) or 0) + 1}

    return node
