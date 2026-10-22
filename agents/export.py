"""Export: copy deliverables into outputs/<run_id>/final/ with handout naming and write run_summary.json +
cost_report.csv (scripts/make_cost_report.py when present, else built here from logs/run_log.jsonl)."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.context import ROOT, RunContext, now_iso
from core.llm import LLM
from core.logging import RunLogger
from core.state import AgentState


def read_run_log(path: Path, run_id: str) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if r.get("run_id") == run_id:
            rows.append(r)
    return rows


def build_cost_rows(rows: list[dict]) -> list[dict]:
    """Group log lines by (agent, tool, model): calls, tokens, numeric units, cost_inr, wall seconds."""
    g: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("agent", ""), r.get("tool", ""), r.get("model", ""))
        a = g.setdefault(key, {"agent": key[0], "tool": key[1], "model": key[2], "calls": 0, "input_tokens": 0,
                               "output_tokens": 0, "units": defaultdict(float), "cost_inr": 0.0, "wall_s": 0.0})
        a["calls"] += 1
        a["input_tokens"] += int(r.get("input_tokens") or 0)
        a["output_tokens"] += int(r.get("output_tokens") or 0)
        a["cost_inr"] += float(r.get("cost_inr") or 0.0)
        a["wall_s"] += float(r.get("latency_s") or 0.0)
        units = r.get("units") or {}
        if isinstance(units, dict):
            for k, v in units.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("attempt",):
                    a["units"][k] += float(v)
    out = []
    for a in g.values():
        a["units"] = json.dumps({k: round(v, 3) for k, v in sorted(a["units"].items())})
        a["cost_inr"] = round(a["cost_inr"], 2)
        a["wall_s"] = round(a["wall_s"], 2)
        out.append(a)
    return sorted(out, key=lambda a: (a["agent"], a["tool"], a["model"]))


def write_cost_report(ctx: RunContext, out_path: Path) -> Path:
    script = ROOT / "scripts" / "make_cost_report.py"
    log_path = ctx.logs_dir / "run_log.jsonl"
    if script.exists():
        try:
            subprocess.run([sys.executable, str(script), "--log", str(log_path), "--run-id", ctx.run_id,
                            "--out", str(out_path)], cwd=str(ROOT), capture_output=True, timeout=300, check=True)
            if out_path.exists():
                return out_path
        except Exception:  # noqa: BLE001
            pass
    rows = build_cost_rows(read_run_log(log_path, ctx.run_id))
    cols = ["agent", "tool", "model", "calls", "input_tokens", "output_tokens", "units", "cost_inr", "wall_s"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return out_path


def _start_time(ctx: RunContext, rows: list[dict]) -> str:
    try:
        return json.loads((ctx.out_dir / "run_meta.json").read_text(encoding="utf-8"))["started_at"]
    except Exception:  # noqa: BLE001
        return rows[0]["ts"] if rows else ctx.started_at


def _copy(src: str | Path | None, dst: Path, final: dict, key: str, missing: list[str]) -> None:
    if src and Path(src).is_file():
        shutil.copy2(src, dst)
        final[key] = str(dst)
    else:
        missing.append(f"{key}: {src or 'not produced'}")


def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        brief_id = (state.get("brief") or {}).get("id", "BRIEF")
        base = ctx.final_basename(brief_id)
        fdir = ctx.out_dir / "final"
        fdir.mkdir(parents=True, exist_ok=True)
        final: dict[str, Any] = {}
        missing: list[str] = []
        with log.call("export", "copy") as rec:
            _copy(state.get("video_path"), fdir / f"{base}.mp4", final, "video", missing)
            _copy(state.get("srt_path"), fdir / f"{base}.srt", final, "srt", missing)
            _copy(ctx.artifact("qa_report.json"), fdir / "qa_report.json", final, "qa_report", missing)
            facts_src = ctx.artifact("facts.json")
            if not facts_src.exists() and state.get("facts") is not None:
                facts_src = ctx.write_json("facts.json", state.get("facts"))
            _copy(facts_src, fdir / "facts.json", final, "facts", missing)
            man = next((p for p in (ctx.out_dir / "sources_manifest.csv", ctx.sources_dir.parent / "sources_manifest.csv")
                        if p.exists()), None)
            _copy(man, fdir / "sources_manifest.csv", final, "sources_manifest", missing)
            rec["units"] = {"files": len(final), "missing": len(missing)}
            rec["output_path"] = str(fdir)

        rows = read_run_log(ctx.logs_dir / "run_log.jsonl", ctx.run_id)
        started = _start_time(ctx, rows)
        try:
            wall = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()
        except (TypeError, ValueError):
            wall = 0.0
        summary = {
            "run_id": ctx.run_id, "brief_id": brief_id, "team_id": ctx.team_id, "seed": ctx.seed,
            "started_at": started, "finished_at": now_iso(), "wall_time_s": round(wall, 1),
            "stage_times": state.get("stage_times") or {}, "loops": int(state.get("loop", 0) or 0),
            "interventions": sum(1 for r in rows if r.get("human_intervention")),
            "review_checkpoints": sum(1 for r in rows if r.get("tool") == "review_checkpoint"),
            "review": state.get("review") or {},
            "qa_passed": bool((state.get("qa_report") or {}).get("passed", False)),
            "total_input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows),
            "total_output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
            "missing": missing,
        }
        summary["total_tokens"] = summary["total_input_tokens"] + summary["total_output_tokens"]
        sp = fdir / "run_summary.json"
        sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        final["run_summary"] = str(sp)
        final["cost_report"] = str(write_cost_report(ctx, fdir / "cost_report.csv"))
        log.record(agent="export", tool="final", output_path=final.get("video", ""), units={"files": sorted(final)})
        for name in ("run_log.jsonl", "interventions.md"):     # after the last record so the copy is complete
            src = ctx.logs_dir / name
            if src.exists():
                shutil.copy2(src, fdir / name)
                final[name.split(".")[0]] = str(fdir / name)
        return {"final": final}

    return node
