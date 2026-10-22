"""Real-ESRGAN x2 (via spandrel, fp16 on GPU, tiled) + cover-crop to the target size; Pillow Lanczos fallback.

Request (task "upscale"): single {"path", "width", "height", "out_path"?} or batch {"items": [{"path",
"out_path"?}], "width", "height"}; "mock" uses Lanczos only.
Outputs: {"paths": [...], "path": first, "method": "realesrgan-x2"|"pillow-lanczos", "fallback_reason": ""}
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.workers import worker_main  # noqa: E402
from workers.media_common import hf_file, models_cfg, with_meta  # noqa: E402

TILE = 512
PAD = 16


def cover_crop(img, width: int, height: int):
    """Scale to cover (width, height) with Lanczos and centre-crop."""
    from PIL import Image

    iw, ih = img.size
    s = max(width / iw, height / ih)
    nw, nh = max(width, math.ceil(iw * s)), max(height, math.ceil(ih * s))
    if (nw, nh) != (iw, ih):
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - width) // 2, (nh - height) // 2
    return img.crop((left, top, left + width, top + height))


def _load_esrgan(cfg: dict):
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    up = cfg["image"]["upscaler"]
    path = hf_file(up["repo"], up["file"], up.get("revision", ""), up)
    desc = ModelLoader().load_from_file(path)
    if not isinstance(desc, ImageModelDescriptor):
        raise TypeError(f"{path} is not an image-to-image model")
    desc.to(torch.device("cuda")).eval()
    if desc.supports_half:
        desc.half()
    return desc


def _esrgan(desc, img):
    """Tiled inference (bounded VRAM on a T4)."""
    import numpy as np
    import torch
    from PIL import Image

    scale = desc.scale
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    dtype = torch.float16 if desc.supports_half else torch.float32
    with torch.inference_mode():
        for y in range(0, h, TILE):
            for x in range(0, w, TILE):
                y0, x0 = max(0, y - PAD), max(0, x - PAD)
                y1, x1 = min(h, y + TILE + PAD), min(w, x + TILE + PAD)
                t = torch.from_numpy(arr[y0:y1, x0:x1]).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype)
                r = desc(t).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                ty, tx = (y - y0) * scale, (x - x0) * scale
                th, tw = (min(y + TILE, h) - y) * scale, (min(x + TILE, w) - x) * scale
                out[y * scale:y * scale + th, x * scale:x * scale + tw] = r[ty:ty + th, tx:tx + tw]
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))


def handle(req: dict) -> dict:
    from PIL import Image

    width, height = int(req["width"]), int(req["height"])
    items = req.get("items") or [{"path": req["path"], "out_path": req.get("out_path")}]
    cfg = models_cfg()
    t0 = time.perf_counter()
    desc, reason = None, ""
    if req.get("mock"):
        reason = "mock"
    else:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("no CUDA device")
            desc = _load_esrgan(cfg)
        except Exception as e:  # noqa: BLE001 - documented fallback
            reason = f"{e.__class__.__name__}: {e}"[:300]
    method = "realesrgan-x2" if desc is not None else "pillow-lanczos"

    paths = []
    for it in items:
        src = Path(it["path"])
        dst = Path(it.get("out_path") or src.with_name(f"{src.stem}_up.png"))
        img = Image.open(src).convert("RGB")
        if desc is not None and (img.width < width or img.height < height):
            img = _esrgan(desc, img)
        cover_crop(img, width, height).save(dst, format="PNG")
        paths.append(str(dst))

    up = cfg["image"]["upscaler"]
    return with_meta({
        "outputs": {"paths": paths, "path": paths[0] if paths else "", "method": method, "fallback_reason": reason},
        "units": {"images": len(paths), "width": width, "height": height, "wall_s": round(time.perf_counter() - t0, 2)},
        "model": f"{up['repo']}/{up['file']}" if desc is not None else "pillow-lanczos",
        "model_revision": up.get("revision", "") if desc is not None else "",
        "licence": up.get("licence", "BSD-3-Clause") if desc is not None else "own-work",
    })


if __name__ == "__main__":
    worker_main(handle)
