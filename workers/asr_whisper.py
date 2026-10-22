"""faster-whisper large-v3-turbo ASR worker (VO round-trip check + word timings).

Request (task "transcribe"): {"wav_path", "initial_prompt", "language"="en", "mock",
                              "expected_text"?, "expected_words"?: [{text|word, start, end}]}  (mock only)
Outputs: {"text", "words": [{word, start, end, probability}], "language", "duration_s"}
Decoding is deterministic: beam_size 5, temperature 0 (no fallback sampling), no conditioning on previous text.
"""
from __future__ import annotations

import inspect
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.workers import worker_main  # noqa: E402
from workers.media_common import hf_offline, models_cfg, repo_or_dir, with_meta  # noqa: E402


def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return 0.0


def _mock(req: dict) -> dict:
    text = str(req.get("expected_text", ""))
    words = []
    if req.get("expected_words"):
        for w in req["expected_words"]:
            words.append({"word": str(w.get("word", w.get("text", ""))), "start": float(w["start"]),
                          "end": float(w["end"]), "probability": 1.0})
    else:
        t = 0.0
        for w in text.split():
            words.append({"word": w, "start": round(t, 3), "end": round(t + 0.38, 3), "probability": 1.0})
            t += 1 / 2.6
    return {"text": text, "words": words}


def load_wav_16k(path: str):
    """PCM WAV -> mono float32 at 16 kHz, without PyAV.

    faster-whisper decodes file paths with av.open(..., metadata_errors=...), which PyAV >= 19 rejects. Passing
    an array skips that decoder. Our TTS writes plain PCM WAVs, so the stdlib `wave` module is enough."""
    import numpy as np

    with wave.open(path, "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    x = (x - 128.0) / 128.0 if width == 1 else x / float(np.iinfo(dtype).max)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(sr, 16000)
            x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        except ImportError:
            t = np.arange(int(len(x) * 16000 / sr)) * (sr / 16000)
            x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def handle(req: dict) -> dict:
    cfg = models_cfg()
    asr = cfg.get("voice", {}).get("asr", {})
    wav = str(req["wav_path"])
    dur = _wav_duration(wav)
    t0 = time.perf_counter()
    if req.get("mock"):
        out = _mock(req)
        return with_meta({"outputs": {**out, "language": "en", "duration_s": round(dur, 3)},
                "units": {"audio_s": round(dur, 3), "mock": True}, "model": "mock-echo", "model_revision": "",
                "licence": "own-work"})

    import ctranslate2
    from faster_whisper import WhisperModel

    cuda = ctranslate2.get_cuda_device_count() > 0
    device = "cuda" if cuda else "cpu"
    compute = asr.get("compute_type", "float16") if cuda else "int8"
    repo = asr.get("repo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo")
    src = repo_or_dir(repo, asr)
    kw: dict = {"device": device, "compute_type": compute, "local_files_only": hf_offline()}
    if src == repo and asr.get("revision") and "revision" in inspect.signature(WhisperModel.__init__).parameters:
        kw["revision"] = asr["revision"]
    model = WhisperModel(src, **kw)
    segments, info = model.transcribe(
        load_wav_16k(wav),
        language=req.get("language", "en"),
        beam_size=5,
        temperature=0.0,
        word_timestamps=True,
        initial_prompt=req.get("initial_prompt") or None,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    texts, words = [], []
    for seg in segments:                      # generator: transcription runs while iterating
        texts.append(seg.text.strip())
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                          "probability": round(float(w.probability), 4)})
    return with_meta({
        "outputs": {"text": " ".join(t for t in texts if t), "words": words,
                    "language": getattr(info, "language", "en"), "duration_s": round(dur, 3)},
        "units": {"audio_s": round(dur, 3), "device": device, "compute_type": compute,
                  "wall_s": round(time.perf_counter() - t0, 2)},
        "model": repo,
        "model_revision": asr.get("revision", ""),
        "licence": asr.get("licence", "MIT"),
    })


if __name__ == "__main__":
    worker_main(handle)
