"""Run context: paths, seed, flags. One instance per run, passed to every node factory."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists()


def load_yaml(rel: str) -> dict:
    with open(ROOT / rel, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class RunContext:
    run_id: str
    seed: int
    brief_path: Path
    team_id: str = "TXX"
    offline: bool = False                 # replay data/sources snapshots, no network
    only: str | None = None               # re-run a single stage
    from_stage: str | None = None
    renderer: str = "hyperframes"         # hyperframes | ffmpeg
    llm_base_url: str = "http://127.0.0.1:8001/v1"
    vlm_base_url: str = "http://127.0.0.1:8002/v1"
    mock_llm: bool = False                # tests / local dev without a server
    mock_media: bool = False              # workers return placeholders without loading models
    started_at: str = field(default_factory=now_iso)

    # ---- directories
    @property
    def out_dir(self) -> Path:
        p = ROOT / "outputs" / self.run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_dir(self) -> Path:
        p = ROOT / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def sources_dir(self) -> Path:
        p = ROOT / "data" / "sources"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def index_dir(self) -> Path:
        p = self.sources_dir.parent / "index"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def assets_dir(self) -> Path:
        p = self.out_dir / "assets"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_db(self) -> Path:
        return ROOT / "data" / "cache.sqlite"

    @property
    def checkpoint_db(self) -> Path:
        return self.out_dir / "checkpoints.sqlite"

    def artifact(self, name: str) -> Path:
        return self.out_dir / name

    def write_json(self, name: str, obj) -> Path:
        p = self.artifact(name)
        data = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def final_basename(self, brief_id: str) -> str:
        return f"{self.team_id}_{brief_id}_final"

    # ---- environment fingerprint for run_manifest.json
    def environment(self) -> dict:
        def sh(cmd: list[str]) -> str:
            try:
                return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception as e:  # noqa: BLE001
                return f"unavailable: {e.__class__.__name__}"

        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "on_kaggle": on_kaggle(),
            "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""),
            "git_commit": sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "pip_freeze": sh([sys.executable, "-m", "pip", "freeze"]).splitlines(),
        }
        try:
            from core.servers import find_llama_server, llama_version

            env["llama_server"] = llama_version(find_llama_server())
        except Exception as e:  # noqa: BLE001 - binary absent on dev boxes
            env["llama_server"] = f"unavailable: {e.__class__.__name__}"
        for f in ("/etc/build_date", "/etc/git_commit"):
            if Path(f).exists():
                env[f] = Path(f).read_text().strip()
        return env
