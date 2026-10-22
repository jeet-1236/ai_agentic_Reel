"""Asset manifest: one CSV row per generated/used file (outputs/<run_id>/assets/manifest.csv).

`add_asset(ctx, AssetRecord)` appends a row (thread-safe, de-duplicated on path+sha256) and returns whether the
licence is on `config/licences_allowlist.yaml`. The Critic reads the file back with `read_manifest` /
`licence_failures` for the "licences" hard check.
"""
from __future__ import annotations

import csv
import re
import threading
from functools import lru_cache
from pathlib import Path

from core.context import ROOT, RunContext, load_yaml
from core.schemas import AssetRecord

FIELDS = [*AssetRecord.model_fields.keys(), "licence_ok"]
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _allowlist() -> tuple[frozenset[str], dict[str, str]]:
    cfg = load_yaml("config/licences_allowlist.yaml")
    allowed = frozenset(str(x).strip().lower() for x in cfg.get("allowed", []))
    aliases = {str(k).strip().lower(): str(v).strip().lower() for k, v in (cfg.get("aliases") or {}).items()}
    return allowed, aliases


def licence_allowed(licence: str) -> bool:
    """True if every part of a (possibly compound: 'A AND B', 'A; B') licence is allowlisted."""
    allowed, aliases = _allowlist()
    parts = [p.strip().lower() for p in re.split(r"\s+AND\s+|;", licence or "", flags=re.I)]
    parts = [aliases.get(p, p) for p in parts if p]
    return bool(parts) and all(p in allowed for p in parts)


def manifest_path(ctx: RunContext) -> Path:
    return ctx.assets_dir / "manifest.csv"


def _rel(ctx: RunContext, path: str) -> str:
    """Store paths relative to the run dir (portable across machines) when possible."""
    p = Path(path)
    for base in (ctx.out_dir, ROOT):
        try:
            return p.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return p.as_posix()


def read_manifest(ctx: RunContext) -> list[dict]:
    p = manifest_path(ctx)
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def add_asset(ctx: RunContext, record: AssetRecord) -> bool:
    """Append one manifest row; returns licence_ok. Same path+sha256 twice is written once."""
    ok = licence_allowed(record.licence)
    row = record.model_dump(mode="json")
    row["path"] = _rel(ctx, row["path"])
    row["licence_ok"] = ok
    p = manifest_path(ctx)
    with _lock:
        existing = read_manifest(ctx)
        if any(r["path"] == row["path"] and r["sha256"] == row["sha256"] for r in existing):
            return ok
        new = not p.exists() or p.stat().st_size == 0
        with open(p, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    return ok


def licence_failures(ctx: RunContext) -> list[dict]:
    """Rows whose licence is not allowlisted (re-evaluated against the current allowlist)."""
    return [r for r in read_manifest(ctx) if not licence_allowed(r.get("licence", ""))]
