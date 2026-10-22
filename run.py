"""AgentReel: one-command entry point.

    python run.py --brief briefs/C01.yaml --seed 42
    python run.py --brief briefs/C01.yaml --seed 42 --only scriptwriter     # re-run one stage (viva)
    python run.py --brief briefs/C01.yaml --seed 42 --from-stage storyboarder
    python run.py --resume <run_id> --approve | --reject "note" [--target scriptwriter]
    python run.py --brief ... --offline        # replay data/sources snapshots, no network
    python run.py --brief ... --mock-llm       # local smoke test without llama-server
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import yaml

from core.context import ROOT, RunContext, now_iso
from core.graph import build_graph
from core.llm import LLM
from core.logging import RunLogger
from core.schemas import Brief, ReviewDecision
from core.seeds import seed_everything


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brief", type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--team-id", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--only", default=None, help="re-run exactly one stage, reusing cached upstream results")
    ap.add_argument("--from-stage", default=None, help="re-run this stage and everything after it")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--renderer", choices=["hyperframes", "ffmpeg"], default=None)
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--mock-media", action="store_true", help="placeholder media workers (no GPU models)")
    ap.add_argument("--resume", default=None, help="run_id waiting at human review")
    ap.add_argument("--approve", action="store_true")
    ap.add_argument("--reject", default=None, metavar="NOTE")
    ap.add_argument("--target", default="scriptwriter", help="agent that handles a rejection")
    return ap.parse_args(argv)


def load_settings() -> dict:
    p = ROOT / "config" / "settings.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def make_context(a, settings: dict) -> RunContext:
    if a.resume:
        meta = json.loads((ROOT / "outputs" / a.resume / "run_meta.json").read_text(encoding="utf-8"))
        return RunContext(run_id=a.resume, seed=meta["seed"], brief_path=Path(meta["brief_path"]),
                          team_id=meta["team_id"], offline=meta["offline"], renderer=meta["renderer"],
                          mock_llm=meta["mock_llm"], mock_media=meta.get("mock_media", False))
    brief = Brief.model_validate(yaml.safe_load(a.brief.read_text(encoding="utf-8")))
    run_id = a.run_id or f"{brief.id}-s{a.seed}"
    return RunContext(run_id=run_id, seed=a.seed, brief_path=a.brief,
                      team_id=a.team_id or settings.get("team_id", "TXX"), offline=a.offline,
                      only=a.only, from_stage=a.from_stage,
                      renderer=a.renderer or settings.get("renderer", "hyperframes"), mock_llm=a.mock_llm,
                      mock_media=a.mock_media)


def main(argv=None) -> int:
    a = parse_args(argv)
    if not a.brief and not a.resume:
        print("need --brief or --resume", file=sys.stderr)
        return 2
    settings = load_settings()
    ctx = make_context(a, settings)
    seed_everything(ctx.seed)
    log = RunLogger(ctx)
    llm = LLM(ctx, log)
    if not ctx.mock_llm:
        from core.servers import ensure_llm

        ensure_llm(ctx, log)

    from langgraph.cache.sqlite import SqliteCache
    from langgraph.checkpoint.sqlite import SqliteSaver

    ctx.cache_db.parent.mkdir(parents=True, exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(str(ctx.checkpoint_db), check_same_thread=False))
    cache = SqliteCache(path=str(ctx.cache_db))
    graph = build_graph(ctx, log, llm, checkpointer=saver, cache=cache)
    cfg = {"configurable": {"thread_id": ctx.run_id}, "recursion_limit": 200}

    if a.resume:
        from langgraph.types import Command

        decision = ReviewDecision(approved=a.approve and not a.reject, note=a.reject or "",
                                  target_agent="" if a.approve else a.target)
        result = graph.invoke(Command(resume=decision.model_dump()), cfg)
        return report(ctx, graph, cfg, result)

    (ctx.out_dir / "run_meta.json").write_text(json.dumps({
        "run_id": ctx.run_id, "seed": ctx.seed, "brief_path": str(ctx.brief_path), "team_id": ctx.team_id,
        "offline": ctx.offline, "renderer": ctx.renderer, "mock_llm": ctx.mock_llm, "mock_media": ctx.mock_media, "started_at": now_iso(),
        "argv": sys.argv,
    }, indent=2), encoding="utf-8")
    ctx.write_json("run_manifest.json", {"environment": ctx.environment(), "started_at": now_iso()})
    log.record(agent="run", tool="start", units={"argv": sys.argv, "seed": ctx.seed})

    brief = yaml.safe_load(ctx.brief_path.read_text(encoding="utf-8"))
    stage = a.only or a.from_stage
    if stage:
        graph.clear_cache(nodes=[stage])     # force this stage to re-execute; upstream comes from cache
    kwargs = {"interrupt_after": [stage]} if a.only else {}
    result = graph.invoke({"brief": brief, "loop": 0}, cfg, **kwargs)
    return report(ctx, graph, cfg, result)


def report(ctx: RunContext, graph, cfg, result) -> int:
    snap = graph.get_state(cfg)
    if snap.next:
        pending = [i.value for t in snap.tasks for i in (t.interrupts or [])]
        if pending:
            p = ctx.out_dir / "review" / "PENDING_REVIEW.json"
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(pending, indent=2, default=str), encoding="utf-8")
            print(f"\n== Waiting for human review: {p}\n   approve: python run.py --resume {ctx.run_id} --approve"
                  f"\n   reject : python run.py --resume {ctx.run_id} --reject \"note\" --target scriptwriter")
            return 3
        print(f"Stopped before: {snap.next} (--only run finished)")
        return 0
    final = (result or {}).get("final", {})
    print(json.dumps({"run_id": ctx.run_id, "final": final, "stage_times": (result or {}).get("stage_times")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
