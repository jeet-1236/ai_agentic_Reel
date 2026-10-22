"""Heading-aware chunking + hybrid retrieval (bm25s + dense embeddings, RRF k=60, reranker).

Dense embedding and reranking run in workers/retrieval_models.py (GPU1, or hash-based mock vectors when
ctx.mock_media). Chunks live in memory and in outputs/<run_id>/chunks.json; vectors are cached under
data/index/ (git-ignored) keyed by sha256(model, kind, texts).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np

from core.context import RunContext, load_yaml
from core.logging import RunLogger
from core.schemas import Chunk
from core.sources import terms

TARGET_WORDS = 300          # ~400 tokens
MAX_WORDS = 420
RRF_K = 60
POOL = 20                   # candidates sent to the reranker per query
TOP_K = 8

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_PLAIN_HEADING = re.compile(r"^(subject|circular|chapter|section|article|annexure|schedule|part)\b.{0,100}$", re.I)


def _is_heading(line: str) -> str:
    m = _MD_HEADING.match(line)
    if m:
        return m.group(1).strip()
    s = line.strip()
    if _PLAIN_HEADING.match(s) and len(s.split()) <= 14:
        return s
    if 2 <= len(s.split()) <= 10 and s.upper() == s and re.search(r"[A-Z]{3}", s) and not s.endswith((".", ",")):
        return s
    return ""


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])", text) if s.strip()]


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """[(heading, [paragraph, ...])] from markdown or extracted PDF text."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    para: list[str] = []

    def flush():
        if para:
            sections[-1][1].append("\n".join(para).strip())
            para.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        h = _is_heading(line)
        if h:
            flush()
            sections.append((h, []))
            continue
        para.append(line.strip())
    flush()
    return [(h, [p for p in ps if p]) for h, ps in sections if any(ps)]


def chunk_text(source_id: str, url: str, text: str, target_words: int = TARGET_WORDS,
               max_words: int = MAX_WORDS) -> list[Chunk]:
    """Heading-aware chunks of ~target_words words; ids S<n>#c<k> (k from 1)."""
    pieces: list[tuple[str, str]] = []
    for heading, paras in _sections(text):
        buf: list[str] = []
        n = 0
        units: list[str] = []
        for p in paras:
            units += [p] if len(p.split()) <= max_words else _sentences(p)
        for u in units:
            w = len(u.split())
            if buf and n + w > target_words:
                pieces.append((heading, "\n\n".join(buf)))
                buf, n = [], 0
            if w > max_words:  # a single over-long sentence: hard split by words
                words = u.split()
                for i in range(0, len(words), target_words):
                    pieces.append((heading, " ".join(words[i:i + target_words])))
                continue
            buf.append(u)
            n += w
        if buf:
            pieces.append((heading, "\n\n".join(buf)))
    # merge tiny trailing pieces into the previous piece of the same heading
    merged: list[tuple[str, str]] = []
    for h, t in pieces:
        if merged and merged[-1][0] == h and len(t.split()) < 40 and len(merged[-1][1].split()) + len(t.split()) <= max_words:
            merged[-1] = (h, merged[-1][1] + "\n\n" + t)
        else:
            merged.append((h, t))
    return [Chunk(chunk_id=f"{source_id}#c{k}", source_id=source_id, url=url, heading=h, text=t)
            for k, (h, t) in enumerate(merged, start=1)]


def passage_text(c: Chunk) -> str:
    return f"{c.heading}\n{c.text}" if c.heading else c.text


# ---------------------------------------------------------------- model backends (worker)
EmbedFn = Callable[[list[str], str], np.ndarray]
RerankFn = Callable[[list[tuple[str, list[str]]]], list[list[float]]]


def worker_backends(ctx: RunContext, log: RunLogger, agent: str) -> tuple[EmbedFn, RerankFn]:
    """Embed / rerank through workers/retrieval_models.py with an on-disk vector cache."""
    from core.workers import run_worker

    cfg = load_yaml("config/models.yaml")["retrieval"]
    mock = bool(ctx.mock_media)
    cache_dir = ctx.sources_dir.parent / "index"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def embed(texts: list[str], kind: str) -> np.ndarray:
        key = hashlib.sha256(json.dumps([cfg["embedding"]["repo"], cfg["embedding"].get("revision", ""),
                                         mock, kind, texts]).encode()).hexdigest()[:24]
        path = cache_dir / f"emb_{key}.npy"
        if path.exists():
            log.record(agent=agent, tool="embed_cache", units={"texts": len(texts), "kind": kind, "hit": True},
                       output_path=str(path))
            return np.load(path)
        out = run_worker(ctx, log, agent=agent, worker="retrieval_models.py", gpu="1",
                         request={"task": "embed", "texts": texts, "kind": kind, "mock": mock,
                                  "out_path": str(path)})
        return np.load(out["npy_path"])

    def rerank(items: list[tuple[str, list[str]]]) -> list[list[float]]:
        out = run_worker(ctx, log, agent=agent, worker="retrieval_models.py", gpu="1",
                         request={"task": "rerank", "mock": mock,
                                  "items": [{"query": q, "passages": ps} for q, ps in items]})
        return out["scores"]

    return embed, rerank


def nli_scores(ctx: RunContext, log: RunLogger, agent: str, pairs: list[tuple[str, str]]) -> list[float]:
    """HHEM consistency score per (premise, hypothesis) pair, via the retrieval worker."""
    if not pairs:
        return []
    from core.workers import run_worker

    out = run_worker(ctx, log, agent=agent, worker="retrieval_models.py", gpu="1",
                     request={"task": "nli", "pairs": [list(p) for p in pairs], "mock": bool(ctx.mock_media)})
    return [float(s) for s in out["scores"]]


# ---------------------------------------------------------------- retriever
def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embed: EmbedFn | None = None, rerank: RerankFn | None = None):
        import bm25s

        self.chunks = chunks
        self.embed, self.rerank = embed, rerank
        self.bm25 = bm25s.BM25()
        self.bm25.index([terms(passage_text(c)) or ["_"] for c in chunks], show_progress=False)
        self.vectors = embed([passage_text(c) for c in chunks], "passage") if (embed and chunks) else None

    def _bm25(self, q: str, n: int) -> list[int]:
        toks = terms(q)
        if not toks or not self.chunks:
            return []
        # backend "auto" picks JAX when it is installed (Kaggle preinstalls it) and XLA then grabs ~75% of a GPU
        docs, scores = self.bm25.retrieve([toks], k=min(n, len(self.chunks)), show_progress=False,
                                          backend_selection="numpy")
        return [int(d) for d, s in zip(docs[0], scores[0]) if s > 0]

    def search_many(self, queries: list[str], k: int = TOP_K, pool: int = POOL) -> dict[str, list[tuple[Chunk, float]]]:
        """Per query: bm25 top-50 + dense top-50 -> RRF -> top `pool` -> rerank -> top k."""
        if not self.chunks or not queries:
            return {q: [] for q in queries}
        qvecs = self.embed(queries, "query") if self.vectors is not None else None
        fused: dict[str, list[int]] = {}
        for i, q in enumerate(queries):
            rankings = [self._bm25(q, 50)]
            if qvecs is not None:
                sims = self.vectors @ qvecs[i]
                rankings.append([int(j) for j in np.argsort(-sims, kind="stable")[:50]])
            scores = rrf(rankings)
            fused[q] = sorted(scores, key=lambda j: (-scores[j], j))[:pool]
        if self.rerank:
            rr = self.rerank([(q, [passage_text(self.chunks[j]) for j in fused[q]]) for q in queries])
        else:
            rr = [[1.0 / (r + 1) for r in range(len(fused[q]))] for q in queries]
        out: dict[str, list[tuple[Chunk, float]]] = {}
        for q, scores in zip(queries, rr):
            ranked = sorted(zip(fused[q], scores), key=lambda t: (-t[1], t[0]))[:k]
            out[q] = [(self.chunks[j], float(s)) for j, s in ranked]
        return out


def save_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.write_text(json.dumps([c.model_dump() for c in chunks], indent=1, ensure_ascii=False), encoding="utf-8")


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate(c) for c in json.loads(Path(path).read_text(encoding="utf-8"))]
