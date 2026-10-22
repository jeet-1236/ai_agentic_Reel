"""FactCurator: retrieved chunks -> LLM fact proposals -> quote check (code) -> HHEM NLI -> facts.json + coverage.json.

A proposal survives only if its normalised quote is a substring of the cited chunk and HHEM scores the claim
as consistent with the chunk (>= config/models.yaml retrieval.nli.threshold). Coverage of each must_include
item is computed without the LLM (retrieval + lexical/embedding similarity); uncovered items are MISSING.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable


from core import sources as src
from core.context import RunContext, load_yaml
from core.llm import LLM
from core.logging import RunLogger
from core.retrieval import HybridRetriever, chunk_text, load_chunks, nli_scores, worker_backends
from core.schemas import Brief, Chunk, CoverageItem, Fact, FactProposalsOut
from core.seeds import stage_seed
from core.state import AgentState

AGENT = "fact_curator"
PROMPT_ID = "fact_curator_v1"
MAX_CONTEXT_CHUNKS = 18
MIN_QUOTE_WORDS = 5
MAX_FACTS = 20
GENERIC = {src.stem(w) for w in "somaiya kjsce university college student students official".split()}
_QWORDS = {src.stem(w) for w in "count counts decide decides decided".split()}

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'",
                         "“": '"', "”": '"', "„": '"', "″": '"',
                         "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
                         "−": "-", "­": None, " ": " ", "​": None})


def normalise(text: str) -> str:
    """Case/whitespace/quote/dash-insensitive form used for the verbatim-quote check."""
    t = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    t = re.sub(r"[*_#>|]+", " ", t)  # markdown emphasis / table pipes added by extraction
    return re.sub(r"\s+", " ", t).strip()


def find_quote(quote: str, chunk_text_: str) -> str:
    """Return the chunk's own span matching the quote (whitespace collapsed), or "" if it is not verbatim.

    Matching ignores case, quote/dash style and whitespace (PDF extraction often inserts or drops spaces)."""
    def squash(s: str) -> tuple[str, list[int]]:
        chars, idx = [], []
        for i, ch in enumerate(s):
            n = normalise(ch) if not ch.isspace() else ""
            for c in n.replace(" ", ""):
                chars.append(c)
                idx.append(i)
        return "".join(chars), idx

    q, _ = squash(quote.strip().strip('"').strip("'").rstrip("."))
    t, idx = squash(chunk_text_)
    if len(q) < 10:
        return ""
    pos = t.find(q)
    if pos < 0:
        return ""
    span = chunk_text_[idx[pos]: idx[pos + len(q) - 1] + 1]
    return re.sub(r"\s+", " ", span).strip()


def _chunks_from_state(ctx: RunContext, state: AgentState) -> list[Chunk]:
    p = ctx.artifact("chunks.json")
    if p.exists():
        return load_chunks(p)
    from agents.researcher import resolve_data_path

    chunks: list[Chunk] = []
    for s in state.get("sources", []):
        text = resolve_data_path(ctx, s["text_path"]).read_text(encoding="utf-8")
        chunks += chunk_text(s["source_id"], s["url"], text)
    return chunks


def mock_facts_fixture(variables: dict) -> dict:
    """Deterministic FactProposalsOut for --mock-llm: first full sentences of the top retrieved chunks."""
    facts = []
    for ch in variables["chunks"]:
        for sent in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", ch["text"])):
            words = sent.split()
            if 8 <= len(words) <= 45 and sent.endswith("."):
                facts.append({"claim": sent, "quote": sent, "chunk_id": ch["chunk_id"]})
                break
        if len(facts) >= 8:
            break
    return {"facts": facts}


def coverage_map(items: list[str], facts: list[Fact], item_hits: dict[str, list[str]],
                 embed: Callable | None) -> list[CoverageItem]:
    """must_include item -> fact ids (LLM-free): lexical overlap, or retrieved-chunk match + similarity."""
    if not items:
        return []
    fvec = ivec = None
    if embed and facts:
        fvec = embed([f"{f.claim}\n{f.quote}" for f in facts], "passage")
        ivec = embed(items, "query")
    out = []
    for i, item in enumerate(items):
        it = [t for t in src.terms(item) if t not in GENERIC and t not in _QWORDS] or src.terms(item)
        top = set(item_hits.get(item, [])[:3])
        ids = []
        for j, f in enumerate(facts):
            lex = src.overlap(it, src.terms(f"{f.claim} {f.quote}")) / max(1, len(it))
            cos = float(fvec[j] @ ivec[i]) if fvec is not None else 0.0
            if lex >= 0.5 or (f.chunk_id in top and lex >= 0.25 and cos >= 0.45):
                ids.append(f.fact_id)
        out.append(CoverageItem(item=item, fact_ids=ids, status="covered" if ids else "MISSING"))
    return out


def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    llm.fixtures.setdefault(PROMPT_ID, mock_facts_fixture)
    threshold = float(load_yaml("config/models.yaml")["retrieval"]["nli"].get("threshold", 0.5))

    def node(state: AgentState) -> dict:
        brief = Brief.model_validate(state["brief"])
        plan = state.get("plan") or {}
        loop = state.get("loop", 0)
        chunks = _chunks_from_state(ctx, state)
        by_id = {c.chunk_id: c for c in chunks}
        src_url = {s["source_id"]: s["url"] for s in state.get("sources", [])}
        embed, rerank = worker_backends(ctx, log, AGENT)
        retr = HybridRetriever(chunks, embed, rerank) if chunks else None

        queries = list(dict.fromkeys([*brief.must_include, brief.topic, *plan.get("queries", [])]))
        hits = retr.search_many(queries) if retr else {q: [] for q in queries}
        best: dict[str, float] = {}
        for q in queries:  # must_include first, so their chunks are never squeezed out
            for rank, (c, _s) in enumerate(hits[q]):
                best[c.chunk_id] = min(best.get(c.chunk_id, 1e9), rank + 0.01 * queries.index(q))
        context_ids = sorted(best, key=lambda cid: (best[cid], cid))[:MAX_CONTEXT_CHUNKS]
        context = [{"chunk_id": cid, "source": src_url.get(by_id[cid].source_id, by_id[cid].url),
                    "heading": by_id[cid].heading, "text": by_id[cid].text} for cid in context_ids]

        proposals = llm.structured(AGENT, PROMPT_ID, FactProposalsOut, seed=stage_seed(ctx.seed, AGENT, 0, loop),
                                   brief=brief.model_dump(), must_include=brief.must_include, chunks=context,
                                   feedback=state.get("feedback", {}).get(AGENT, [])).facts if context else []

        verified: list[tuple[Chunk, str, str]] = []
        rejected: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for p in proposals:
            chunk = by_id.get(p.chunk_id.strip())
            span = find_quote(p.quote, chunk.text) if chunk else ""
            if not chunk or not span or len(span.split()) < MIN_QUOTE_WORDS:
                rejected.append({"chunk_id": p.chunk_id, "quote": p.quote[:200], "reason": "quote_not_in_chunk"})
                continue
            key = (chunk.chunk_id, normalise(span), normalise(p.claim))
            claim = re.sub(r"^\s*(?:[-•*·▪●]|\d+[.)]|\(?[a-z]\))\s+", "", p.claim).strip()  # drop list bullets
            if key not in seen and claim:
                seen.add(key)
                verified.append((chunk, span, claim))
        scores = nli_scores(ctx, log, AGENT, [(c.text, claim) for c, _q, claim in verified])
        best_by_quote: dict[tuple[str, str], tuple[Chunk, str, str, float]] = {}
        for (chunk, span, claim), score in zip(verified, scores):
            if score < threshold:
                rejected.append({"chunk_id": chunk.chunk_id, "quote": span[:200], "reason": f"nli {score:.3f}"})
                continue
            k = (chunk.chunk_id, normalise(span))
            if k not in best_by_quote or score > best_by_quote[k][3]:
                best_by_quote[k] = (chunk, span, claim, score)
        kept = list(best_by_quote.values())

        def order(t):
            sid, cnum = t[0].chunk_id.split("#c")
            return int(sid.lstrip("S") or 0), int(cnum), normalise(t[0].text).find(normalise(t[1])), t[2]

        facts = [Fact(fact_id=f"F{i}", claim=claim, quote=span, chunk_id=chunk.chunk_id, source_id=chunk.source_id,
                      url=src_url.get(chunk.source_id, chunk.url), nli_score=round(score, 4))
                 for i, (chunk, span, claim, score) in enumerate(sorted(kept, key=order)[:MAX_FACTS], start=1)]
        item_hits = {q: [c.chunk_id for c, _ in hits.get(q, [])] for q in brief.must_include}
        coverage = coverage_map(brief.must_include, facts, item_hits, embed if chunks else None)

        ctx.write_json("facts.json", [f.model_dump() for f in facts])
        ctx.write_json("coverage.json", [c.model_dump() for c in coverage])
        ctx.write_json("fact_curation_trace.json", {"context_chunks": context_ids, "proposals": len(proposals),
                                                    "rejected": rejected, "threshold": threshold})
        log.record(agent=AGENT, tool="summary", units={"proposals": len(proposals), "facts": len(facts),
                                                       "rejected": len(rejected),
                                                       "missing": [c.item for c in coverage if c.status == "MISSING"]},
                   output_path=str(ctx.artifact("facts.json")))
        return {"facts": [f.model_dump() for f in facts], "coverage": [c.model_dump() for c in coverage]}

    return node
