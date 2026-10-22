"""Helpers shared by the media workers (stdlib + PyYAML only, so it also works in the Py3.12 voice venv).

Model resolution order for every weight file / repo:
  1. `path` key in config/models.yaml (absolute, or relative to the repo root)
  2. a file/dir with that name under $AGENTREEL_MODELS_DIR (os.pathsep-separated; set by setup_kaggle.sh to the
     mounted `agentreel-models` dataset) or <repo>/models
  3. the Hugging Face cache / Hub (`HF_HUB_OFFLINE=1` -> cache only, never the network)
"""
from __future__ import annotations

import os
from pathlib import Path

from core.context import ROOT, load_yaml


def models_cfg() -> dict:
    return load_yaml("config/models.yaml")


def hf_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def search_dirs() -> list[Path]:
    dirs = [Path(p) for p in os.environ.get("AGENTREEL_MODELS_DIR", "").split(os.pathsep) if p]
    return [d for d in (*dirs, ROOT / "models") if d.exists()]


def find_local(name: str) -> Path | None:
    """First file or directory called `name` (or a repo id's last part) under the model search dirs."""
    target = name.rstrip("/").split("/")[-1]
    for d in search_dirs():
        direct = d / name
        if direct.exists():
            return direct
        for dirpath, dirnames, filenames in os.walk(d):
            if target in filenames or target in dirnames:
                return Path(dirpath) / target
    return None


def configured_path(cfg: dict) -> Path | None:
    p = cfg.get("path")
    if not p:
        return None
    q = Path(p) if Path(p).is_absolute() else ROOT / p
    return q if q.exists() else None


def hf_file(repo: str, filename: str, revision: str = "", cfg: dict | None = None) -> str:
    """Local path of one model file (config path -> models dir -> HF cache/Hub)."""
    local = configured_path(cfg or {}) or find_local(filename)
    if local is not None and local.is_file():
        return str(local)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, revision=revision or None, local_files_only=hf_offline())


def repo_or_dir(repo: str, cfg: dict | None = None) -> str:
    """A local snapshot dir if one is staged, else the repo id (libraries resolve it via the HF cache)."""
    local = configured_path(cfg or {}) or find_local(repo)
    return str(local) if local is not None and local.is_dir() else repo


def torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def with_meta(resp: dict) -> dict:
    """Copy model/model_revision/licence into outputs["meta"]: core.workers.run_worker returns only `outputs`,
    and the asset manifest needs them."""
    resp.setdefault("outputs", {})["meta"] = {k: resp.get(k, "") for k in ("model", "model_revision", "licence")}
    return resp
