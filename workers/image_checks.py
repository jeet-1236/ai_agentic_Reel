"""Image safety checks: YuNet face detection (OpenCV, CPU) + Falconsai NSFW classifier (transformers).

Request (task "check"): {"paths": [...], "mock"}
Outputs: {"results": [{path, face_count, face_scores, nsfw_score}], "face_detector": "yunet"|"haar"|"mock"}
If the YuNet onnx cannot be found (and cannot be downloaded) the bundled Haar cascade is used (OpenCV 4.x), or
on OpenCV 5 the detector is reported as "unavailable" (face_count 0) - the detector name is returned so the
AssetGenerator logs it as an error and the Critic's VLM frame check remains the second line of defence.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.context import ROOT  # noqa: E402
from core.workers import worker_main  # noqa: E402
from workers.media_common import configured_path, find_local, hf_offline, models_cfg, with_meta  # noqa: E402

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
MAX_SIDE = 640          # YuNet is trained on small inputs; downscale large images for speed


def yunet_path(cfg: dict) -> Path | None:
    fd = cfg.get("qa", {}).get("face_detector", {})
    name = fd.get("file", "face_detection_yunet_2023mar.onnx")
    p = configured_path(fd) or find_local(name)
    if p is not None and p.is_file():
        return p
    if hf_offline():
        return None
    try:
        dst = ROOT / "models" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, dst)  # noqa: S310 - fixed https URL
        return dst if dst.stat().st_size > 100_000 else None
    except Exception:  # noqa: BLE001
        return None


def detect_faces(paths: list[str], model_path: Path | None, score_threshold: float = 0.6) -> tuple[str, list[list[float]]]:
    """Returns (detector_name, per-image list of face confidence scores)."""
    import cv2

    out: list[list[float]] = []
    if model_path is not None:
        det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(p)
            h, w = img.shape[:2]
            s = min(1.0, MAX_SIDE / max(h, w))
            if s < 1.0:
                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = det.detect(img)
            out.append([] if faces is None else [round(float(f[-1]), 4) for f in faces])
        return "yunet", out
    if not hasattr(cv2, "CascadeClassifier"):        # OpenCV 5 moved Haar cascades out of the main module
        return "unavailable", [[] for _ in paths]
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(p)
        h, w = img.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6, minSize=(24, 24))
        out.append([1.0] * len(faces))
    return "haar", out


def nsfw_scores(paths: list[str], cfg: dict) -> list[float]:
    import torch
    from PIL import Image
    from transformers import pipeline

    n = cfg.get("qa", {}).get("nsfw", {})
    clf = pipeline("image-classification", model=n.get("repo", "Falconsai/nsfw_image_detection"),
                   revision=n.get("revision") or None, device=0 if torch.cuda.is_available() else -1)
    images = [Image.open(p).convert("RGB") for p in paths]
    preds = clf(images)
    if preds and isinstance(preds[0], dict):   # single image -> flat list
        preds = [preds]
    return [round(float(next((d["score"] for d in pr if str(d["label"]).lower() == "nsfw"), 0.0)), 4)
            for pr in preds]


def handle(req: dict) -> dict:
    paths = [str(p) for p in req.get("paths", [])]
    cfg = models_cfg()
    thr = float(cfg.get("qa", {}).get("face_detector", {}).get("score_threshold", 0.6))
    if req.get("mock"):
        results = [{"path": p, "face_count": 0, "face_scores": [], "nsfw_score": 0.0} for p in paths]
        return with_meta({"outputs": {"results": results, "face_detector": "mock"}, "units": {"images": len(paths)},
                "model": "mock", "model_revision": "", "licence": "own-work"})
    detector, faces = detect_faces(paths, yunet_path(cfg), thr)
    nsfw = nsfw_scores(paths, cfg) if paths else []
    results = [{"path": p, "face_count": len(f), "face_scores": f, "nsfw_score": s}
               for p, f, s in zip(paths, faces, nsfw)]
    nrepo = cfg.get("qa", {}).get("nsfw", {}).get("repo", "Falconsai/nsfw_image_detection")
    return with_meta({
        "outputs": {"results": results, "face_detector": detector},
        "units": {"images": len(paths)},
        "model": f"{detector}+{nrepo}",
        "model_revision": cfg.get("qa", {}).get("nsfw", {}).get("revision", ""),
        "licence": "MIT AND Apache-2.0",
    })


if __name__ == "__main__":
    worker_main(handle)
