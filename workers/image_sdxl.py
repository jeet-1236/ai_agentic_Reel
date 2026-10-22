"""SDXL 1.0 + SDXL-Lightning 4-step UNet + fp16-fix VAE image worker (GPU, fp16; T4 has no bf16).

Request (task "generate"), either a single job at top level or a batch in `jobs` (one model load per batch):
    {"prompt", "negative", "seed", "n", "out_prefix", "label"?} | {"jobs": [{...same...}]},
    "width", "height", "mock", "out_dir"
Outputs: {"results": [{"out_prefix", "paths", "seeds"}], "paths": [every image]}
Candidate i of a job uses a CPU torch.Generator seeded with seed + i (reproducible across GPUs).
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.workers import worker_main  # noqa: E402
from workers.media_common import hf_file, hf_offline, models_cfg, repo_or_dir, with_meta  # noqa: E402

LICENCE = "CreativeML-OpenRAIL++-M"


def _jobs(req: dict) -> list[dict]:
    if req.get("jobs"):
        return list(req["jobs"])
    return [{k: req.get(k) for k in ("prompt", "negative", "seed", "n", "out_prefix", "label")}]


def _model_info(cfg: dict) -> tuple[str, str]:
    img = cfg["image"]
    model = f"{img['base']['repo']}+{img['lightning']['repo']}/{img['lightning']['file']}+{img['vae']['repo']}"
    rev = ",".join(str(img[k].get("revision", "") or "-") for k in ("base", "lightning", "vae"))
    return model, rev


# ---------------------------------------------------------------- mock
def _mock_image(path: Path, prompt: str, seed: int, label: str, width: int, height: int) -> None:
    """Deterministic placeholder: vertical gradient in prompt-hash colours + shot label (Pillow only)."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    c1 = (h[0], h[1], h[2])
    c2 = ((h[3] + seed) % 256, (h[4] + seed * 7) % 256, (h[5] + seed * 13) % 256)
    grad = Image.linear_gradient("L").resize((width, height))
    img = ImageOps.colorize(grad, black=c1, white=c2).convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(24, width // 12))
        small = ImageFont.load_default(size=max(14, width // 40))
    except TypeError:  # very old Pillow
        font = small = ImageFont.load_default()
    d.rectangle([width * 0.1, height * 0.4, width * 0.9, height * 0.6], fill=(255 - c1[0], 255 - c1[1], 255 - c1[2]))
    d.text((width * 0.15, height * 0.43), label or "image", fill=c1, font=font)
    d.text((width * 0.15, height * 0.53), f"{h.hex()[:12]}  seed {seed}", fill=c1, font=small)
    img.save(path, format="PNG")


# ---------------------------------------------------------------- real
def _load_pipeline(cfg: dict):
    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from safetensors.torch import load_file

    img = cfg["image"]
    off = hf_offline()
    base_cfg, light_cfg, vae_cfg = img["base"], img["lightning"], img["vae"]
    base = repo_or_dir(base_cfg["repo"], base_cfg)
    base_rev = (base_cfg.get("revision") or None) if base == base_cfg["repo"] else None
    vae_src = repo_or_dir(vae_cfg["repo"], vae_cfg)
    vae_rev = (vae_cfg.get("revision") or None) if vae_src == vae_cfg["repo"] else None

    vae = AutoencoderKL.from_pretrained(vae_src, revision=vae_rev, torch_dtype=torch.float16, local_files_only=off)
    # Documented SDXL-Lightning recipe: base UNet config + Lightning 4-step weights.
    unet_config = UNet2DConditionModel.load_config(base, subfolder="unet", revision=base_rev, local_files_only=off)
    unet = UNet2DConditionModel.from_config(unet_config).to("cuda", torch.float16)
    ckpt = hf_file(light_cfg["repo"], light_cfg["file"], light_cfg.get("revision", ""), light_cfg)
    unet.load_state_dict(load_file(ckpt, device="cuda"))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, unet=unet, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
        revision=base_rev, local_files_only=off,
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def handle(req: dict) -> dict:
    cfg = models_cfg()
    img_cfg = cfg["image"]
    width = int(req.get("width") or img_cfg["size_9x16"][0])
    height = int(req.get("height") or img_cfg["size_9x16"][1])
    out_dir = Path(req["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(req)
    default_n = int(img_cfg.get("candidates_per_shot", 2))
    model, revision = _model_info(cfg)
    t0 = time.perf_counter()

    pipe = None
    if not req.get("mock"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("image_sdxl needs CUDA (fp16 SDXL); use mock mode on CPU-only machines")
        pipe = _load_pipeline(cfg)
    load_s = time.perf_counter() - t0

    results, all_paths = [], []
    for j, job in enumerate(jobs):
        n = int(job.get("n") or default_n)
        seed = int(job.get("seed") or 0)
        prefix = job.get("out_prefix") or f"img_{seed}"
        paths, seeds = [], []
        for i in range(n):
            path = out_dir / f"{prefix}_c{i}.png"
            if pipe is None:
                _mock_image(path, job.get("prompt") or "", seed + i, job.get("label") or prefix, width, height)
            else:
                import torch

                gen = torch.Generator("cpu").manual_seed(seed + i)
                image = pipe(
                    prompt=job.get("prompt") or "",
                    negative_prompt=job.get("negative") or None,   # ignored by diffusers at guidance 0
                    width=width, height=height,
                    num_inference_steps=int(img_cfg.get("steps", 4)),
                    guidance_scale=float(img_cfg.get("guidance_scale", 0.0)),
                    generator=gen,
                ).images[0]
                image.save(path, format="PNG")
            paths.append(str(path))
            seeds.append(seed + i)
        results.append({"out_prefix": prefix, "paths": paths, "seeds": seeds})
        all_paths.extend(paths)

    return with_meta({
        "outputs": {"results": results, "paths": all_paths},
        "units": {"images": len(all_paths), "width": width, "height": height, "load_s": round(load_s, 2),
                  "gen_s": round(time.perf_counter() - t0 - load_s, 2), "mock": bool(req.get("mock"))},
        "model": "mock-gradient" if req.get("mock") else model,
        "model_revision": "" if req.get("mock") else revision,
        "licence": "own-work" if req.get("mock") else LICENCE,
    })


if __name__ == "__main__":
    worker_main(handle)
