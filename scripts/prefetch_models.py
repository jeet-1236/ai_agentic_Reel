"""Download every model in config/models.yaml (+ YuNet, Real-ESRGAN x2, HHEM tokenizer, fonts) into folders
ready for `kaggle datasets create`, and write config/models.lock.yaml (resolved commit SHA + file sha256).

    python scripts/prefetch_models.py --out /tmp/agentreel-datasets            # everything
    python scripts/prefetch_models.py --out ... --only llm.text image.base       # subset (dotted keys)
    python scripts/prefetch_models.py --dry-run                                  # list what would be fetched
    python scripts/prefetch_models.py --out ... --include-fallbacks --fonts

Layout (what core.servers.resolve_model() and the workers expect):
    <out>/agentreel-models/<org>__<name>/<files...>     plain files (no HF cache symlinks: Kaggle uploads them badly)
    <out>/agentreel-models/opencv__face_detection_yunet/face_detection_yunet_2023mar.onnx
    <out>/agentreel-fonts/*.ttf + OFL.txt
    <out>/<dataset>/dataset-metadata.json                (id = $KAGGLE_USERNAME/<dataset>)

Needs `huggingface_hub` (pip install huggingface_hub==<pin>). Set HF_TOKEN for gated repos.
If a download fails on the Xet backend, it retries once with HF_HUB_DISABLE_XET=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOCK = ROOT / "config" / "models.lock.yaml"
# generic junk we never need on Kaggle (other frameworks / export formats)
IGNORE = ["*.onnx", "onnx/*", "*.msgpack", "*.h5", "tf_model*", "flax_model*", "rust_model*", "openvino/*",
          "*.ot", "*.mlmodel", "coreml/*", "*.tflite", ".gitattributes"]
# per-key allow lists to avoid pulling 10s of GB we don't use
ALLOW: dict[str, list[str]] = {
    "image.base": ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
                   "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
                   "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors",
                   "unet/config.json", "vae/config.json", "LICENSE*"],
    "image.vae": ["config.json", "diffusion_pytorch_model.safetensors", "LICENSE*", "README.md"],
    "voice.tts": ["config.json", "kokoro-v1_0.pth", "voices/*.pt", "LICENSE*", "README.md"],
    "extra.hhem_tokenizer": ["config.json", "tokenizer*", "spiece.model", "special_tokens_map.json"],
}
EXTRA_HF = {  # not in models.yaml but required at runtime
    "extra.hhem_tokenizer": {"repo": "google/flan-t5-base", "revision": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2", "licence": "Apache-2.0",
                             "why": "HHEM-2.1-Open loads the flan-t5-base tokenizer"},
}
URLS = {
    "qa.face_detector": {
        "dir": "opencv__face_detection_yunet", "file": "face_detection_yunet_2023mar.onnx", "licence": "MIT",
        "urls": ["https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx",
                 "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"]},
    "image.upscaler_github": {
        "dir": "xinntao__Real-ESRGAN", "file": "RealESRGAN_x2plus.pth", "licence": "BSD-3-Clause",
        "urls": ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"],
        "fallback_only": True},
}
FONTS = {
    "FiraSans-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/firasans/FiraSans-Regular.ttf",
    "FiraSans-Medium.ttf": "https://github.com/google/fonts/raw/main/ofl/firasans/FiraSans-Medium.ttf",
    "FiraSans-SemiBold.ttf": "https://github.com/google/fonts/raw/main/ofl/firasans/FiraSans-SemiBold.ttf",
    "FiraSans-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/firasans/FiraSans-Bold.ttf",
    "FiraSans-ExtraBold.ttf": "https://github.com/google/fonts/raw/main/ofl/firasans/FiraSans-ExtraBold.ttf",
    "OFL-FiraSans.txt": "https://github.com/google/fonts/raw/main/ofl/firasans/OFL.txt",
    "NotoSans-Variable.ttf": "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
    "OFL-NotoSans.txt": "https://github.com/google/fonts/raw/main/ofl/notosans/OFL.txt",
}


@dataclass
class Job:
    key: str                      # dotted path in models.yaml (or extra.*)
    repo: str
    files: list[str] = field(default_factory=list)   # explicit files; [] = snapshot with allow/ignore
    revision: str = ""
    licence: str = ""
    allow: list[str] | None = None


def plan_downloads(cfg: dict, include_fallbacks: bool = False) -> list[Job]:
    """Flatten config/models.yaml into download jobs (pure; unit-tested)."""
    jobs: list[Job] = []

    def visit(node, path: list[str]):
        if not isinstance(node, dict):
            return
        key = ".".join(path)
        if "repo" in node:
            if "fallback" in key and not include_fallbacks:
                return
            files = [node[k] for k in ("file", "mmproj") if node.get(k)]
            jobs.append(Job(key=key, repo=node["repo"], files=files, revision=str(node.get("revision") or ""),
                            licence=node.get("licence", ""), allow=ALLOW.get(key)))
            return
        for k, v in node.items():
            visit(v, [*path, k])

    visit(cfg, [])
    for key, spec in EXTRA_HF.items():
        jobs.append(Job(key=key, repo=spec["repo"], revision=spec.get("revision", ""), licence=spec["licence"],
                        allow=ALLOW.get(key)))
    return jobs


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _disable_xet() -> None:
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        import huggingface_hub.constants as c

        c.HF_HUB_DISABLE_XET = True
    except (ImportError, AttributeError):
        pass


def _with_xet_fallback(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("HF_HUB_DISABLE_XET") == "1":
            raise
        print(f"   download failed ({e.__class__.__name__}: {str(e)[:160]}); retrying with HF_HUB_DISABLE_XET=1")
        _disable_xet()
        return fn(*a, **kw)


def fetch_hf(job: Job, models_out: Path) -> dict:
    """Download one HF job; return its lock entry."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    rev = job.revision or None
    sha = api.model_info(job.repo, revision=rev).sha
    dest = models_out / job.repo.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    if job.files:
        for f in job.files:
            _with_xet_fallback(hf_hub_download, job.repo, f, revision=sha, local_dir=str(dest))
    else:
        _with_xet_fallback(snapshot_download, job.repo, revision=sha, local_dir=str(dest),
                           allow_patterns=job.allow, ignore_patterns=None if job.allow else IGNORE)
    files = {}
    for p in sorted(dest.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            files[p.relative_to(dest).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return {"repo": job.repo, "requested_revision": job.revision, "resolved_revision": sha,
            "licence": job.licence, "dir": dest.name, "files": files}


def fetch_url(key: str, spec: dict, models_out: Path) -> dict:
    """Download a non-HF file (YuNet, Real-ESRGAN github release) trying each mirror."""
    dest = models_out / spec["dir"]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / spec["file"]
    last = ""
    for url in spec["urls"]:
        if target.exists() and target.stat().st_size > 1024:
            break
        try:
            print(f"   GET {url}")
            urllib.request.urlretrieve(url, target)  # noqa: S310 (fixed https URLs)
            last = url
        except OSError as e:
            print(f"   failed: {e}")
            target.unlink(missing_ok=True)
    if not target.exists():
        raise RuntimeError(f"{key}: all mirrors failed")
    return {"url": last or spec["urls"][0], "licence": spec["licence"], "dir": dest.name,
            "files": {spec["file"]: {"sha256": sha256_file(target), "bytes": target.stat().st_size}}}


def fetch_fonts(fonts_out: Path) -> dict:
    fonts_out.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, url in FONTS.items():
        p = fonts_out / name
        if not p.exists():
            print(f"   GET {url}")
            urllib.request.urlretrieve(url, p)  # noqa: S310
        out[name] = {"url": url, "sha256": sha256_file(p), "licence": "OFL-1.1"}
    return out


def write_metadata(folder: Path, slug: str, user: str, title: str) -> None:
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}],
            "subtitle": "AgentReel pre-staged files (private)", "isPrivate": True}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--only", nargs="*", default=None, help="dotted keys, e.g. llm.text image.base qa.face_detector")
    ap.add_argument("--include-fallbacks", action="store_true")
    ap.add_argument("--fonts", action="store_true", help="also fetch Fira Sans + Noto Sans (agentreel-fonts)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--kaggle-user", default=os.environ.get("KAGGLE_USERNAME", "YOUR_KAGGLE_USERNAME"))
    ap.add_argument("--no-lock", action="store_true", help="don't write config/models.lock.yaml")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    jobs = plan_downloads(cfg, a.include_fallbacks)
    url_jobs = {k: v for k, v in URLS.items() if not v.get("fallback_only")}
    if a.only:
        jobs = [j for j in jobs if j.key in a.only]
        url_jobs = {k: v for k, v in URLS.items() if k in a.only}
    if a.dry_run:
        for j in jobs:
            print(f"hf   {j.key:28s} {j.repo:50s} rev={j.revision or 'main'} files={j.files or j.allow or 'snapshot'}")
        for k, v in url_jobs.items():
            print(f"url  {k:28s} {v['urls'][0]}")
        if a.fonts:
            print(f"font {len(FONTS)} files -> agentreel-fonts")
        return 0

    models_out = a.out / "agentreel-models"
    models_out.mkdir(parents=True, exist_ok=True)
    lock = yaml.safe_load(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    lock = lock or {}
    lock.setdefault("models", {})
    failures = []
    for j in jobs:
        print(f"== {j.key}: {j.repo}")
        try:
            lock["models"][j.key] = fetch_hf(j, models_out)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{j.key}: {e.__class__.__name__}: {e}")
            print(f"   FAILED {failures[-1][:300]}")
            if j.key == "image.upscaler":   # HF mirror gone -> official GitHub release
                try:
                    lock["models"]["image.upscaler"] = fetch_url(j.key, URLS["image.upscaler_github"], models_out)
                    failures.pop()
                except Exception as e2:  # noqa: BLE001
                    print(f"   github fallback failed too: {e2}")
    for k, spec in url_jobs.items():
        print(f"== {k}")
        try:
            lock["models"][k] = fetch_url(k, spec, models_out)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{k}: {e}")
    write_metadata(models_out, "agentreel-models", a.kaggle_user, "agentreel-models")
    if a.fonts:
        fonts_out = a.out / "agentreel-fonts"
        lock["fonts"] = fetch_fonts(fonts_out)
        write_metadata(fonts_out, "agentreel-fonts", a.kaggle_user, "agentreel-fonts")
    for slug in ("agentreel-bin", "agentreel-wheels"):
        (a.out / slug).mkdir(parents=True, exist_ok=True)
        write_metadata(a.out / slug, slug, a.kaggle_user, slug)

    if not a.no_lock:
        LOCK.write_text("# Written by scripts/prefetch_models.py. Do not edit by hand.\n"
                        + yaml.safe_dump(lock, sort_keys=True, allow_unicode=True), encoding="utf-8")
        print(f"lock -> {LOCK}")
    print("\nNext: copy resolved_revision / sha256 of the primary file into config/models.yaml "
          "(revision:, sha256:) and follow scripts/make_kaggle_datasets.md")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
