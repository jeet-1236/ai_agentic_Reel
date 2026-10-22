"""Wall-clock budget for the 3 h Live Run (PLAN §14).

`Budget(ctx, plan)` measures elapsed time since the run started (run_meta.json survives --resume) against
`config/settings.yaml` (time_limit_min, degrade_after_min) and the plan's per-stage budgets. The critic uses
`loops_allowed()` to stop repairing when a further loop would not fit before the hard stop.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from core.context import RunContext, load_yaml, now_iso

REPAIR_STAGES = ("scriptwriter", "storyboarder", "asset_generator", "editor", "critic")
DEFAULT_LOOP_MIN = 13.0      # PLAN §14: ≤3 repair loops in ≤40 min
RESERVE_MIN = 25.0           # review (10) + export/upload (15)


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Budget:
    def __init__(self, ctx: RunContext, plan: Any = None, settings: dict | None = None):
        self.ctx = ctx
        if settings is None:
            try:
                settings = load_yaml("config/settings.yaml")
            except FileNotFoundError:
                settings = {}
        self.time_limit_min = float(settings.get("time_limit_min", 180))
        self.degrade_after_min = float(settings.get("degrade_after_min", 120))
        plan_d = plan.model_dump() if hasattr(plan, "model_dump") else (plan or {})
        self.stage_min: dict[str, float] = {b["stage"]: float(b["minutes"]) for b in plan_d.get("budgets", []) or []}
        self.started = _parse_ts(self._start_ts())

    def _start_ts(self) -> str:
        meta = self.ctx.out_dir / "run_meta.json"
        try:
            return json.loads(meta.read_text(encoding="utf-8"))["started_at"]
        except Exception:  # noqa: BLE001
            return self.ctx.started_at or now_iso()

    # ------------------------------------------------------------ queries
    def elapsed_min(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        return max(0.0, (now - self.started).total_seconds() / 60.0)

    def remaining_min(self, now: datetime | None = None) -> float:
        return self.time_limit_min - self.elapsed_min(now)

    def should_degrade(self, now: datetime | None = None) -> bool:
        """Past `degrade_after_min`: 1 image candidate, no VLM re-checks, ffmpeg renderer."""
        return self.elapsed_min(now) >= self.degrade_after_min

    def loop_cost_min(self) -> float:
        est = sum(self.stage_min.get(s, 0.0) for s in REPAIR_STAGES)
        return est if est > 0 else DEFAULT_LOOP_MIN

    def loops_allowed(self, loop: int, max_loops: int, now: datetime | None = None) -> int:
        """Effective loop cap: never above max_loops; `loop` (stop now) when another loop won't fit."""
        if self.remaining_min(now) - RESERVE_MIN < self.loop_cost_min():
            return min(loop, max_loops)
        return max_loops

    def overrun(self, stage: str, seconds: float) -> bool:
        b = self.stage_min.get(stage)
        return b is not None and seconds / 60.0 > b

    def snapshot(self, now: datetime | None = None) -> dict:
        return {"elapsed_min": round(self.elapsed_min(now), 2), "remaining_min": round(self.remaining_min(now), 2),
                "degrade": self.should_degrade(now), "loop_cost_min": round(self.loop_cost_min(), 1)}
