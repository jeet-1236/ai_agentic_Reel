"""Verify the hash chain of logs/run_log.jsonl (any edited/deleted/reordered line breaks it).

    python scripts/verify_logs.py                      # logs/run_log.jsonl
    python scripts/verify_logs.py path/to/run_log.jsonl --run-id C01-s42

Exit 0 = chain intact, 1 = broken or unreadable. Also prints per-run line counts and human interventions
(handout limit: 5 per run; target 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.logging import verify_chain  # noqa: E402

MAX_INTERVENTIONS = 5


def summarize(path: Path, run_id: str | None = None) -> dict:
    """Counts per run: lines, errors, interventions, review checkpoints."""
    lines: Counter = Counter()
    errors: Counter = Counter()
    interventions: Counter = Counter()
    reviews: Counter = Counter()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        rec = json.loads(raw)
        rid = rec.get("run_id", "")
        if run_id and rid != run_id:
            continue
        lines[rid] += 1
        if rec.get("status") not in ("ok", None):
            errors[rid] += 1
        if rec.get("human_intervention"):
            interventions[rid] += 1
        if rec.get("tool") == "review_checkpoint":
            reviews[rid] += 1
    return {rid: {"lines": n, "non_ok": errors[rid], "interventions": interventions[rid],
                  "review_checkpoints": reviews[rid]} for rid, n in lines.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path, default=ROOT / "logs" / "run_log.jsonl")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)
    if not a.path.exists():
        print(f"no log at {a.path}")
        return 1
    try:
        ok, n, msg = verify_chain(a.path)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        ok, n, msg = False, 0, f"unreadable line: {e}"
    runs = summarize(a.path, a.run_id) if ok else {}
    if a.json:
        print(json.dumps({"ok": ok, "lines": n, "message": msg, "runs": runs}, indent=2))
    else:
        print(f"{'OK ' if ok else 'BROKEN'} {a.path} ({n} lines checked): {msg}")
        for rid, s in runs.items():
            flag = "  <-- over the limit" if s["interventions"] > MAX_INTERVENTIONS else ""
            print(f"  {rid or '(no run_id)':30s} lines={s['lines']:5d} non_ok={s['non_ok']:3d} "
                  f"interventions={s['interventions']} review_checkpoints={s['review_checkpoints']}{flag}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
