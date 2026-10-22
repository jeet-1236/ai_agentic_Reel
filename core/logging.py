"""Append-only, hash-chained run log (logs/run_log.jsonl) + interventions.md.

Every model / tool / worker / ffmpeg call goes through `RunLogger.call(...)` (context manager) or
`RunLogger.record(...)`. Each line carries `prev_hash` = sha256 of the previous raw line, so any edit
breaks the chain (`scripts/verify_logs.py`).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.context import RunContext, now_iso

GENESIS = "0" * 64


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CallRecord(dict):
    """Mutable record yielded by RunLogger.call(); fill tokens/units/output_path inside the block."""


class RunLogger:
    _lock = threading.Lock()

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.path: Path = ctx.logs_dir / "run_log.jsonl"
        self.interventions: Path = ctx.logs_dir / "interventions.md"
        if not self.interventions.exists():
            self.interventions.write_text(
                "# Human interventions\n\n| Time | Run | Stage / agent | What the human changed | Reason |\n|---|---|---|---|---|\n",
                encoding="utf-8",
            )

    # ------------------------------------------------------------------ core
    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        with open(self.path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            last = f.read().splitlines()[-1]
        return hashlib.sha256(last).hexdigest()

    def record(self, **fields: Any) -> dict:
        base = {
            "ts": now_iso(),
            "run_id": self.ctx.run_id,
            "agent": "",
            "tool": "",
            "model": "",
            "model_revision": "",
            "prompt_id": "",
            "prompt_sha256": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "units": {},
            "cost_inr": 0.0,
            "latency_s": 0.0,
            "gpu_id": "",
            "seed": None,
            "status": "ok",
            "error": "",
            "output_path": "",
            "output_sha256": "",
            "human_intervention": False,
        }
        base.update(fields)
        out = base.get("output_path")
        if out and not base.get("output_sha256") and Path(out).is_file():
            base["output_sha256"] = sha256_file(out)
        with self._lock:
            base["prev_hash"] = self._last_hash()
            line = json.dumps(base, ensure_ascii=False, default=str)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return base

    @contextmanager
    def call(self, agent: str, tool: str, **fields: Any) -> Iterator[CallRecord]:
        """with log.call("scriptwriter", "llm", model=..., prompt_id=...) as rec: rec["output_tokens"] = ..."""
        rec = CallRecord(agent=agent, tool=tool, **fields)
        t0 = time.perf_counter()
        try:
            yield rec
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = f"{e.__class__.__name__}: {e}"[:2000]
            raise
        finally:
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
            self.record(**rec)

    # ------------------------------------------------------------------ humans
    def intervention(self, stage: str, what: str, why: str) -> None:
        """The ONLY way to record a human intervention (counts toward the limit of 5)."""
        ts = now_iso()
        with open(self.interventions, "a", encoding="utf-8") as f:
            f.write(f"| {ts} | {self.ctx.run_id} | {stage} | {what} | {why} |\n")
        self.record(agent=stage, tool="human", status="ok", human_intervention=True,
                    units={"what": what, "why": why})

    def review_checkpoint(self, decision: dict) -> None:
        """Mandatory A/C review: logged, but NOT an intervention."""
        self.record(agent="human_review", tool="review_checkpoint", units=decision)


def verify_chain(path: str | Path) -> tuple[bool, int, str]:
    """Returns (ok, lines_checked, message)."""
    prev = GENESIS
    n = 0
    with open(path, "rb") as f:
        for n, raw in enumerate(f.read().splitlines(), start=1):
            rec = json.loads(raw)
            if rec.get("prev_hash") != prev:
                return False, n, f"chain broken at line {n}"
            prev = hashlib.sha256(raw).hexdigest()
    return True, n, "ok"
