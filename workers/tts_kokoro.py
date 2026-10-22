"""Kokoro-82M TTS worker. Runs under the Python 3.12 voice venv on Kaggle (config voice.python), because the
`kokoro` package requires Python < 3.13. Imports kokoro/torch only inside the real path; mock mode is stdlib.

Request (task "synth"): {"sentences": [spoken...], "voice", "speed", "lang_code", "seed", "lexicon",
                         "pause_s"=0.25, "out_name"?, "mock"}
Outputs: {"wav_path", "duration_s", "sample_rate": 24000, "sentence_spans": [[start, end]...],
          "tokens": [{text, start, end, sentence_index}]}  (24 kHz mono 16-bit WAV; the editor resamples)

Determinism: Kokoro's vocoder samples noise with torch.rand, so torch.manual_seed(seed + chunk_index) is called
before EVERY generated chunk (chunk_index counts across all sentences).
"""
from __future__ import annotations

import math
import re
import sys
import time
import wave
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.workers import worker_main  # noqa: E402
from workers.media_common import hf_offline, models_cfg, repo_or_dir, with_meta  # noqa: E402

SR = 24000
LINK_RE = r"\[[^\]]*\]\([^)]*\)"


# ---------------------------------------------------------------- lexicon
def apply_lexicon(text: str, lexicon: dict | None) -> tuple[str, int]:
    """Respellings (plain text) first, then misaki markdown links [Word](/phonemes/). Returns (text, n_applied)."""
    lexicon = lexicon or {}
    n = 0
    for word, rep in sorted((lexicon.get("respell") or {}).items(), key=lambda kv: -len(kv[0])):
        text, k = re.subn(rf"(?<![\w\[/]){re.escape(str(word))}(?![\w\]])", str(rep), text)
        n += k
    words = {str(w).lower(): str(ph).strip().strip("/") for w, ph in (lexicon.get("words") or {}).items() if ph}
    if words:
        alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        pat = re.compile(rf"{LINK_RE}|(?<![\w\[/])({alts})(?![\w\]])", re.IGNORECASE)

        def sub(m: re.Match) -> str:
            nonlocal n
            if m.group(1) is None:          # already a markdown link: leave untouched
                return m.group(0)
            n += 1
            return f"[{m.group(1)}](/{words[m.group(1).lower()]}/)"

        text = pat.sub(sub, text)
    return text, n


def _strip_links(text: str) -> str:
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)


def write_wav(path: Path, samples: array) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())


# ---------------------------------------------------------------- mock
def _mock(sentences: list[str], seed: int, pause_s: float, path: Path) -> dict:
    """Low tone during 'words' at ~2.6 words/s (Kokoro's measured rate), silence between; synthetic per-word timings."""
    samples = array("h")
    tokens, spans, t = [], [], 0.0
    freq = 160.0 + seed % 40
    for si, text in enumerate(sentences):
        words = [w for w in _strip_links(text).split() if re.sub(r"\W", "", w)]
        dur = max(0.4, len(words) / 2.6)
        weights = [len(w) + 2 for w in words] or [1]
        total = float(sum(weights))
        s_start = t
        for w, wt in zip(words, weights):
            d = dur * wt / total
            gap = min(0.04, d * 0.2)
            n_on, n_off = int((d - gap) * SR), int(gap * SR)
            samples.extend(int(1200 * math.sin(2 * math.pi * freq * k / SR)) for k in range(n_on))
            samples.extend([0] * n_off)
            tokens.append({"text": re.sub(r"^[^\w]+|[^\w]+$", "", w), "start": round(t, 3),
                           "end": round(t + d - gap, 3), "sentence_index": si})
            t += (n_on + n_off) / SR
        if not words:
            samples.extend([0] * int(dur * SR))
            t += int(dur * SR) / SR
        spans.append([round(s_start, 3), round(t, 3)])
        if si < len(sentences) - 1:
            samples.extend([0] * int(pause_s * SR))
            t += int(pause_s * SR) / SR
    write_wav(path, samples)
    return {"tokens": tokens, "spans": spans, "duration": len(samples) / SR}


# ---------------------------------------------------------------- real
def _pipeline(tts: dict, voice: str, lang: str):
    """KPipeline (+ pinned snapshot when a revision or a staged local dir is configured). Returns (pipe, voice_arg)."""
    import torch
    from kokoro import KModel, KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = tts.get("repo", "hexgrad/Kokoro-82M")
    src = repo_or_dir(repo, tts)
    if src == repo and tts.get("revision"):
        from huggingface_hub import snapshot_download

        src = snapshot_download(repo, revision=tts["revision"], local_files_only=hf_offline(),
                                allow_patterns=["config.json", "*.pth", f"voices/{voice}.pt"])
    if src != repo and Path(src).is_dir():
        d = Path(src)
        weights = sorted(d.glob("*.pth"))
        model = KModel(repo_id=repo, config=str(d / "config.json"), model=str(weights[0])).to(device).eval()
        pipe = KPipeline(lang_code=lang, repo_id=repo, model=model)
        vp = d / "voices" / f"{voice}.pt"
        return pipe, (str(vp) if vp.exists() else voice), device
    return KPipeline(lang_code=lang, repo_id=repo, device=device), voice, device


def _real(sentences: list[str], voice: str, speed: float, lang: str, seed: int, pause_s: float,
          tts: dict, path: Path) -> dict:
    import numpy as np
    import torch

    pipe, voice_arg, device = _pipeline(tts, voice, lang)
    pieces: list = []
    tokens, spans = [], []
    t, chunk_idx = 0.0, 0
    for si, text in enumerate(sentences):
        s_start = t
        gen = pipe(text, voice=voice_arg, speed=speed, split_pattern=None)
        while True:
            torch.manual_seed(seed + chunk_idx)          # before EVERY chunk (vocoder uses torch.rand)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed + chunk_idx)
            try:
                res = next(gen)
            except StopIteration:
                break
            chunk_idx += 1
            audio = res.audio if hasattr(res, "audio") else (res.output.audio if res.output is not None else None)
            if audio is None:
                continue
            a = audio.detach().float().cpu().numpy().reshape(-1)
            prev_end = 0.0
            for tok in res.tokens or []:
                word = str(getattr(tok, "text", "")).strip()
                if not re.search(r"\w", word):
                    continue                          # punctuation tokens carry no caption word
                st = tok.start_ts if tok.start_ts is not None else prev_end
                en = tok.end_ts if tok.end_ts is not None else st
                prev_end = en
                tokens.append({"text": word, "start": round(t + float(st), 3), "end": round(t + float(en), 3),
                               "sentence_index": si})
            pieces.append(a)
            t += len(a) / SR
        spans.append([round(s_start, 3), round(t, 3)])
        if si < len(sentences) - 1:
            n = int(pause_s * SR)
            pieces.append(np.zeros(n, dtype=np.float32))
            t += n / SR
    audio = np.concatenate(pieces) if pieces else np.zeros(SR // 2, dtype=np.float32)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    write_wav(path, array("h", pcm.tobytes()))
    return {"tokens": tokens, "spans": spans, "duration": len(pcm) / SR}


def handle(req: dict) -> dict:
    cfg = models_cfg()
    tts = cfg.get("voice", {}).get("tts", {})
    sentences = [str(s) for s in req.get("sentences", [])]
    voice = req.get("voice") or tts.get("voice", "af_heart")
    speed = float(req.get("speed") or tts.get("speed", 1.0))
    lang = req.get("lang_code") or tts.get("lang_code", "a")
    seed = int(req.get("seed") or 0)
    pause_s = float(req.get("pause_s", 0.25))
    out_dir = Path(req["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (req.get("out_name") or f"voice_{seed}.wav")
    prepared, n_lex = [], 0
    for s in sentences:
        p, k = apply_lexicon(s, req.get("lexicon"))
        prepared.append(p)
        n_lex += k

    t0 = time.perf_counter()
    if req.get("mock"):
        r = _mock(prepared, seed, pause_s, path)
    else:
        r = _real(prepared, voice, speed, lang, seed, pause_s, tts, path)
    return with_meta({
        "outputs": {"wav_path": str(path), "duration_s": round(r["duration"], 3), "sample_rate": SR,
                    "sentence_spans": r["spans"], "tokens": r["tokens"], "prepared_text": prepared,
                    "lexicon_applied": n_lex, "voice": voice, "seed": seed},
        "units": {"audio_s": round(r["duration"], 3), "sentences": len(sentences),
                  "wall_s": round(time.perf_counter() - t0, 2), "mock": bool(req.get("mock"))},
        "model": "mock-tone" if req.get("mock") else tts.get("repo", "hexgrad/Kokoro-82M"),
        "model_revision": "" if req.get("mock") else tts.get("revision", ""),
        "licence": "own-work" if req.get("mock") else tts.get("licence", "Apache-2.0"),
    })


if __name__ == "__main__":
    worker_main(handle)
