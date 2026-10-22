"""Retrieval models worker (GPU1): embed | rerank | nli.

    embed : {texts, kind: "query"|"passage", out_path?}        -> {npy_path}
    rerank: {query, passages} or {items: [{query, passages}]}   -> {scores} (list, or list of lists for items)
    nli   : {pairs: [[premise, hypothesis], ...]}               -> {scores} (HHEM-2.1-Open, 0..1 consistent)

Models: Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, vectara/hallucination_evaluation_model (trust_remote_code).
`mock: true` loads nothing: hash-feature vectors and lexical-overlap scores (deterministic, no torch).
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.workers import worker_main  # noqa: E402

DIM_MOCK = 256
EMBED_INSTRUCT = "Given a question about official university or government rules, retrieve passages that answer it"
RERANK_INSTRUCT = "Judge whether the official document passage answers or supports the query"


def _cfg() -> dict:
    import yaml

    return yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))["retrieval"]


def _tokens(text: str) -> list[str]:
    return [t[:7] for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]


# ---------------------------------------------------------------- mock
def mock_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM_MOCK), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in _tokens(t):
            h = hashlib.sha256(tok.encode()).digest()
            out[i, int.from_bytes(h[:4], "little") % DIM_MOCK] += 1.0 if h[4] & 1 else -1.0
        n = np.linalg.norm(out[i])
        if n > 0:
            out[i] /= n
    return out


def mock_overlap(query: str, passage: str) -> float:
    q, p = set(_tokens(query)), set(_tokens(passage))
    return round(len(q & p) / len(q), 6) if q else 0.0


# ---------------------------------------------------------------- real models
def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_kw(spec: dict) -> dict:
    import torch

    kw = {"torch_dtype": torch.float16 if _device() == "cuda" else torch.float32}
    if spec.get("revision"):
        kw["revision"] = spec["revision"]
    return kw


def real_embed(texts: list[str], kind: str, spec: dict) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    repo = spec.get("path") or spec["repo"]
    tok = AutoTokenizer.from_pretrained(repo, padding_side="left", revision=spec.get("revision") or None)
    model = AutoModel.from_pretrained(repo, **_load_kw(spec)).to(_device()).eval()
    if kind == "query":
        texts = [f"Instruct: {EMBED_INSTRUCT}\nQuery:{t}" for t in texts]
    vecs = []
    with torch.inference_mode():
        for i in range(0, len(texts), 16):
            batch = tok(texts[i:i + 16], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(_device())
            hidden = model(**batch).last_hidden_state
            vecs.append(F.normalize(hidden[:, -1].float(), p=2, dim=1).cpu().numpy())
    return np.concatenate(vecs).astype(np.float32) if vecs else np.zeros((0, 1024), np.float32)


class _Reranker:
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
              "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, spec: dict):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = spec.get("path") or spec["repo"]
        self.tok = AutoTokenizer.from_pretrained(repo, padding_side="left", revision=spec.get("revision") or None)
        self.model = AutoModelForCausalLM.from_pretrained(repo, **_load_kw(spec)).to(_device()).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")

    def score(self, query: str, passages: list[str]) -> list[float]:
        import torch

        out: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(passages), 8):
                texts = [f"{self.PREFIX}<Instruct>: {RERANK_INSTRUCT}\n<Query>: {query}\n<Document>: {p[:6000]}{self.SUFFIX}"
                         for p in passages[i:i + 8]]
                batch = self.tok(texts, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(_device())
                logits = self.model(**batch).logits[:, -1, :]
                pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
                out += torch.softmax(pair, dim=1)[:, 1].cpu().tolist()
        return out


def real_nli(pairs: list[list[str]], spec: dict) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification

    repo = spec.get("path") or spec["repo"]
    kw = {"trust_remote_code": True}
    if spec.get("revision"):
        kw["revision"] = spec["revision"]
    model = AutoModelForSequenceClassification.from_pretrained(repo, **kw).to(_device()).eval()
    scores: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(pairs), 16):
            s = model.predict([(p, h) for p, h in pairs[i:i + 16]])
            scores += [float(x) for x in (s.tolist() if hasattr(s, "tolist") else s)]
    return scores


# ---------------------------------------------------------------- handler
def handle(req: dict) -> dict:
    task, mock = req["task"], bool(req.get("mock"))
    cfg = _cfg()
    if task == "embed":
        spec = {**cfg["embedding"], **({"path": req["model_path"]} if req.get("model_path") else {})}
        texts = list(req["texts"])
        vecs = mock_embed(texts) if mock else real_embed(texts, req.get("kind", "passage"), spec)
        out = Path(req.get("out_path") or Path(req["out_dir"]) / f"emb_{req.get('kind', 'passage')}.npy")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, vecs)
        return {"outputs": {"npy_path": str(out)}, "units": {"texts": len(texts), "dim": int(vecs.shape[1]) if vecs.size else 0},
                "model": "mock-hash" if mock else spec["repo"], "model_revision": spec.get("revision", ""),
                "licence": spec.get("licence", "")}
    if task == "rerank":
        spec = {**cfg["reranker"], **({"path": req["model_path"]} if req.get("model_path") else {})}
        items = req.get("items") or [{"query": req["query"], "passages": req["passages"]}]
        rr = None if mock else _Reranker(spec)
        scores = [[mock_overlap(it["query"], p) for p in it["passages"]] if mock else rr.score(it["query"], it["passages"])
                  for it in items]
        return {"outputs": {"scores": scores if "items" in req else scores[0]},
                "units": {"queries": len(items), "passages": sum(len(it["passages"]) for it in items)},
                "model": "mock-overlap" if mock else spec["repo"], "model_revision": spec.get("revision", ""),
                "licence": spec.get("licence", "")}
    if task == "nli":
        spec = {**cfg["nli"], **({"path": req["model_path"]} if req.get("model_path") else {})}
        pairs = req["pairs"]
        scores = [mock_overlap(h, p) for p, h in pairs] if mock else real_nli(pairs, spec)
        return {"outputs": {"scores": scores}, "units": {"pairs": len(pairs)},
                "model": "mock-overlap" if mock else spec["repo"], "model_revision": spec.get("revision", ""),
                "licence": spec.get("licence", "")}
    raise ValueError(f"unknown task {task}")


if __name__ == "__main__":
    worker_main(handle)
