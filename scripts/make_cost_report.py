"""logs/run_log.jsonl -> logs/cost_report.csv (actual cost is Rs 0: everything is local or free-tier).

    python scripts/make_cost_report.py                       # all runs, grouped by run/stage/provider/model
    python scripts/make_cost_report.py --run-id C01-s42 --per-call

Columns: run_id, stage, tool, provider (local|tavily), model, calls, input_tokens, output_tokens, units,
cost_inr (actual, from the log), wall_s, gpu_seconds, shadow_cost_inr.

Shadow cost = what the same GPU time would cost on a rented T4, plus Tavily credits at pay-as-you-go price.
It is NOT money spent. The rates below are ASSUMPTIONS stated for the report; edit them or pass flags.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- ASSUMPTIONS (shadow cost only; actual spend is Rs 0) -------------------------------------
T4_USD_PER_HOUR = 0.35        # ASSUMPTION: public on-demand cloud price of one NVIDIA T4, USD/hour
USD_TO_INR = 88.0             # ASSUMPTION: exchange rate, INR per USD
TAVILY_USD_PER_CREDIT = 0.008  # ASSUMPTION: Tavily pay-as-you-go price per API credit (free tier used)
# -----------------------------------------------------------------------------------------------

GPU_TOOLS = ("llm", "vlm", "worker:", "server")
FIELDS = ["run_id", "stage", "tool", "provider", "model", "calls", "input_tokens", "output_tokens", "units",
          "cost_inr", "wall_s", "gpu_seconds", "shadow_cost_inr"]


def provider_of(rec: dict) -> str:
    blob = f"{rec.get('tool', '')} {rec.get('model', '')}".lower()
    return "tavily" if "tavily" in blob else "local"


def gpu_seconds_of(rec: dict) -> float:
    """Explicit units.gpu_seconds if the caller logged it; else latency for GPU-bound tools with a gpu_id."""
    units = rec.get("units") or {}
    if isinstance(units, dict) and isinstance(units.get("gpu_seconds"), (int, float)):
        return float(units["gpu_seconds"])
    tool = str(rec.get("tool", ""))
    if rec.get("gpu_id") not in ("", None) and tool.startswith(GPU_TOOLS) and rec.get("model") != "mock":
        if tool == "server" and (units or {}).get("action") != "start":
            return 0.0
        return float(rec.get("latency_s") or 0.0)
    return 0.0


def shadow_inr(gpu_s: float, credits: float, t4_usd_h: float, usd_inr: float, tavily_usd: float) -> float:
    return round((gpu_s / 3600.0 * t4_usd_h + credits * tavily_usd) * usd_inr, 4)


def _num_units(units) -> dict:
    return {k: v for k, v in (units or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)} \
        if isinstance(units, dict) else {}


def load_records(path: Path, run_id: str | None = None) -> list[dict]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rec = json.loads(raw)
            if not run_id or rec.get("run_id") == run_id:
                out.append(rec)
    return out


def build_rows(records: list[dict], per_call: bool = False, t4_usd_h: float = T4_USD_PER_HOUR,
               usd_inr: float = USD_TO_INR, tavily_usd: float = TAVILY_USD_PER_CREDIT) -> list[dict]:
    """Aggregate log records into cost rows (+ a TOTAL row)."""
    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for i, rec in enumerate(records):
        tool = str(rec.get("tool", ""))
        key = (i,) if per_call else (rec.get("run_id", ""), rec.get("agent", ""), tool, provider_of(rec),
                                     rec.get("model", ""))
        g = groups.setdefault(key, {"run_id": rec.get("run_id", ""), "stage": rec.get("agent", ""), "tool": tool,
                                    "provider": provider_of(rec), "model": rec.get("model", ""), "calls": 0,
                                    "input_tokens": 0, "output_tokens": 0, "units": {}, "cost_inr": 0.0,
                                    "wall_s": 0.0, "gpu_seconds": 0.0, "_credits": 0.0})
        g["calls"] += 1
        g["input_tokens"] += int(rec.get("input_tokens") or 0)
        g["output_tokens"] += int(rec.get("output_tokens") or 0)
        g["cost_inr"] += float(rec.get("cost_inr") or 0.0)
        g["wall_s"] += float(rec.get("latency_s") or 0.0)
        g["gpu_seconds"] += gpu_seconds_of(rec)
        nums = _num_units(rec.get("units"))
        g["_credits"] += float(nums.get("credits", 0))
        for k, v in nums.items():
            if k in ("attempt", "pid", "seed"):
                continue
            g["units"][k] = round(g["units"].get(k, 0) + v, 3)
    rows = []
    tot = {"run_id": "TOTAL", "stage": "", "tool": "", "provider": "", "model": "", "calls": 0, "input_tokens": 0,
           "output_tokens": 0, "units": {}, "cost_inr": 0.0, "wall_s": 0.0, "gpu_seconds": 0.0, "shadow_cost_inr": 0.0}
    for g in groups.values():
        g["shadow_cost_inr"] = shadow_inr(g["gpu_seconds"], g.pop("_credits"), t4_usd_h, usd_inr, tavily_usd)
        for k in ("calls", "input_tokens", "output_tokens", "cost_inr", "wall_s", "gpu_seconds", "shadow_cost_inr"):
            tot[k] += g[k]
        for k in ("credits",):
            if k in g["units"]:
                tot["units"][k] = tot["units"].get(k, 0) + g["units"][k]
        rows.append(g)
    rows.append(tot)
    for r in rows:
        r["cost_inr"] = round(r["cost_inr"], 4)
        r["wall_s"] = round(r["wall_s"], 2)
        r["gpu_seconds"] = round(r["gpu_seconds"], 2)
        r["shadow_cost_inr"] = round(r["shadow_cost_inr"], 4)
    return rows


def write_csv(rows: list[dict], out: Path, t4_usd_h: float, usd_inr: float, tavily_usd: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "units": json.dumps(r["units"], sort_keys=True)})
        w.writerow({"run_id": "ASSUMPTIONS", "model": (
            f"shadow_cost_inr only (actual spend = cost_inr): T4 USD/hour={t4_usd_h}; INR per USD={usd_inr}; "
            f"Tavily USD/credit={tavily_usd}")})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "run_log.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "cost_report.csv")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--per-call", action="store_true")
    ap.add_argument("--t4-usd-per-hour", type=float, default=T4_USD_PER_HOUR)
    ap.add_argument("--usd-to-inr", type=float, default=USD_TO_INR)
    ap.add_argument("--tavily-usd-per-credit", type=float, default=TAVILY_USD_PER_CREDIT)
    a = ap.parse_args(argv)
    if not a.log.exists():
        print(f"no log at {a.log}")
        return 1
    rows = build_rows(load_records(a.log, a.run_id), a.per_call, a.t4_usd_per_hour, a.usd_to_inr,
                      a.tavily_usd_per_credit)
    write_csv(rows, a.out, a.t4_usd_per_hour, a.usd_to_inr, a.tavily_usd_per_credit)
    t = rows[-1]
    print(f"{a.out}: {len(rows) - 1} rows | calls={t['calls']} tokens in/out={t['input_tokens']}/{t['output_tokens']} "
          f"| actual Rs {t['cost_inr']:.2f} | GPU {t['gpu_seconds']:.0f}s | shadow Rs {t['shadow_cost_inr']:.2f} (assumed rates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
