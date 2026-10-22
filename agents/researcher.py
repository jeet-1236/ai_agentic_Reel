"""Researcher: live official-source discovery -> fetch + snapshot -> extract -> chunk -> contact whitelist.

Discovery (online): (a) Tavily restricted to the allowlist (<= 15 calls), (b) BM25 over the cached sitemap +
notice catalogue (free; notices are refreshed live), (c) ddgs with site: filters as a fallback. Every host is
re-checked against config/domains_allowlist.yaml. Relevant links on fetched official pages are followed;
off-allowlist document hosts (Google Drive) only from an official referrer, which is recorded.

--offline: no network at all. The run's rows in data/sources_manifest.csv + data/sources/<sha256>.<ext>
snapshots are replayed, so sources, chunks and contacts are rebuilt byte-identically.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Callable

from core import sources as src
from core.context import ROOT, RunContext, now_iso
from core.contacts import extract_contacts
from core.llm import LLM
from core.logging import RunLogger, sha256_file
from core.retrieval import chunk_text, save_chunks
from core.schemas import Brief, Chunk, SourceDoc
from core.state import AgentState

AGENT = "researcher"
MAX_TAVILY_CALLS = 15
MAX_DDGS_CALLS = 8
MAX_FETCHES = 40
MAX_SEEDS = 16
MAX_FOLLOW_PER_PAGE = 4
MAX_DEPTH = 2
MIN_TEXT_CHARS = 80
NOTICE_PAGES = 2
GENERIC = {src.stem(w) for w in """somaiya kjsce kjsse svu university college students student official
vidyavihar india mumbai website page link read explore know learn""".split()}
SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".mp4", ".mp3", ".zip", ".rar", ".css", ".js",
            ".xls", ".xlsx", ".ppt", ".pptx", ".doc")


@dataclass
class Candidate:
    url: str
    score: float = 0.0
    via: list[str] = field(default_factory=list)
    query: str = ""
    title: str = ""
    referrer: str = ""
    depth: int = 0
    offlist_hops: int = 0
    authority: str = "other"


@dataclass
class Doc:
    url: str
    final_url: str
    referrer: str
    fetched_at: str
    sha256: str
    path: Path
    ext: str
    content_type: str
    status: int
    tls_verified: bool
    authority: str
    via: str = ""
    query: str = ""
    hop: int = 0
    title: str = ""
    extracted: src.Extracted | None = None


# ---------------------------------------------------------------- queries
def research_queries(brief: Brief, plan: dict, feedback: list) -> list[str]:
    """must_include items first (they drive coverage), then the plan's queries, then critic feedback."""
    qs = [*brief.must_include, *(plan.get("queries") or [brief.topic])]
    for f in feedback:
        text = f if isinstance(f, str) else json.dumps(f)
        qs.append(" ".join(text.split()[:12]))
    return list(dict.fromkeys(q for q in qs if q.strip()))


def _qterms(queries: list[str]) -> list[list[str]]:
    return [[t for t in src.terms(q) if t not in GENERIC] for q in queries]


def link_score(url: str, anchor: str, qterms: list[list[str]]) -> int:
    lt = src.terms(anchor + " " + src.url_words(url))
    best = 0
    for qt in qterms:
        m = src.overlap(qt, lt)
        if qt and m >= min(2, len(qt)):
            best = max(best, m)
    return best


def _usable(url: str) -> bool:
    p = url.lower().split("?")[0]
    return url.startswith(("http://", "https://")) and not p.endswith(SKIP_EXT)


# ---------------------------------------------------------------- discovery (online)
def discover(ctx: RunContext, log: RunLogger, allow: src.Allowlist, queries: list[str]) -> tuple[list[Candidate], dict]:
    """Candidate seeds: per query, results interleaved across tavily/catalogue/ddgs; then round-robin over
    queries (must_include queries first) so every query contributes its best hits."""
    cands: dict[str, Candidate] = {}
    per_query: dict[str, dict[str, list[str]]] = {q: {"tavily": [], "catalogue": [], "ddgs": []} for q in queries}
    trace: dict = {"tavily": [], "catalogue": [], "ddgs": [], "rejected_hosts": []}

    def add(url: str, rank: int, via: str, query: str, title: str = "", referrer: str = ""):
        url = src.canonical(url)
        if not url or not _usable(url):
            return
        if not allow.is_allowed(url):
            trace["rejected_hosts"].append(src.host_of(url))
            return
        c = cands.setdefault(url, Candidate(url=url, query=query, title=title, referrer=referrer,
                                            authority=allow.authority(url)))
        if via not in c.via:
            c.via.append(via)
        per_query[query][via].append(url)

    key = src.tavily_key()
    if key:
        for q in queries[:MAX_TAVILY_CALLS]:
            try:
                with log.call(AGENT, "tavily", units={"query": q, "credits": 1, "search_depth": "basic"}) as rec:
                    rows = src.tavily_search(q, allow, key, max_results=10)
                    rec["units"] = {**rec["units"], "results": len(rows)}
            except Exception as e:  # noqa: BLE001
                trace["tavily"].append({"query": q, "error": str(e)[:200]})
                continue
            trace["tavily"].append({"query": q, "urls": [r["url"] for r in rows]})
            for i, r in enumerate(rows):
                add(r["url"], i, "tavily", q, r.get("title", ""))

    notices = live_notices(ctx, log, allow)
    with log.call(AGENT, "sitemap_catalogue", units={"queries": len(queries), "live_notices": len(notices)}) as rec:
        cat = src.Catalogue.load(extra=notices)
        for q in queries:
            hits = cat.search(q, k=5)
            trace["catalogue"].append({"query": q, "urls": [e["url"] for e, _ in hits]})
            for i, (e, _s) in enumerate(hits):
                add(e["url"], i, "catalogue", q, e.get("title", ""), e.get("referrer", ""))
        rec["units"] = {**rec["units"], "entries": len(cat.entries), "candidates": len(cands)}

    n_tavily = sum(1 for c in cands.values() if "tavily" in c.via)
    if n_tavily < 5:
        calls = 0
        for q in queries:
            kw = " ".join([w for w in re.findall(r"[A-Za-z0-9-]+", q) if w.lower() not in src.STOPWORDS][:8])
            for site in ("somaiya.edu", "gov.in"):
                if calls >= MAX_DDGS_CALLS:
                    break
                calls += 1
                try:
                    with log.call(AGENT, "ddgs", units={"query": f"{kw} site:{site}"}) as rec:
                        rows = src.ddgs_search(f"{kw} site:{site}", max_results=10)
                        rec["units"] = {**rec["units"], "results": len(rows)}
                except Exception as e:  # noqa: BLE001
                    trace["ddgs"].append({"query": q, "site": site, "error": str(e)[:200]})
                    continue
                trace["ddgs"].append({"query": q, "site": site, "urls": [r["url"] for r in rows]})
                for i, r in enumerate(rows):
                    add(r["url"], i, "ddgs", q, r.get("title", ""))
    order: list[str] = []
    merged = {q: [u for trio in zip_longest(v["tavily"], v["catalogue"], v["ddgs"]) for u in trio if u]
              for q, v in per_query.items()}
    for rank in range(max((len(m) for m in merged.values()), default=0)):
        for q in queries:
            if rank < len(merged[q]) and merged[q][rank] not in order:
                order.append(merged[q][rank])
    for i, u in enumerate(order):
        cands[u].score = round(1.0 / (1 + i), 4)
    trace["rejected_hosts"] = sorted(set(trace["rejected_hosts"]))
    return [cands[u] for u in order[:MAX_SEEDS]], trace


def live_notices(ctx: RunContext, log: RunLogger, allow: src.Allowlist) -> list[dict]:
    """Refresh the AJAX notice listings (first pages) so new notices are searchable."""
    out: list[dict] = []
    for nc in allow.cfg.get("notices", []):
        for page in range(0, NOTICE_PAGES * 10, 10):
            url, data = src.notices_request(nc["endpoint"], nc["institute_id"], page)
            try:
                with log.call(AGENT, "fetch", units={"url": url, "kind": "notices", "page": page}) as rec:
                    r = src.fetch_url(url, allow, method="POST", data=data, headers=src.NOTICE_HEADERS)
                    rows = src.parse_notices(r.content.decode("utf-8", "replace")) if r.status == 200 else []
                    rec["units"] = {**rec["units"], "status": r.status, "notices": len(rows)}
            except Exception:  # noqa: BLE001
                break
            out += [{**n, "referrer": nc["referrer"]} for n in rows]
            if not rows:
                break
    return out


# ---------------------------------------------------------------- fetch + follow links
def crawl(ctx: RunContext, log: RunLogger, allow: src.Allowlist, brief: Brief, seeds: list[Candidate],
          queries: list[str], batch: str) -> tuple[list[Doc], list[dict]]:
    qterms = _qterms(queries)
    queue = list(seeds)
    seen = {c.url for c in seeds} | {src.drive_direct(c.url) for c in seeds}
    docs: list[Doc] = []
    skipped: list[dict] = []
    fetches = 0
    session = src._session()
    while queue and fetches < MAX_FETCHES:
        c = queue.pop(0)
        fetches += 1
        try:
            with log.call(AGENT, "fetch", units={"url": c.url, "referrer": c.referrer, "via": ",".join(c.via),
                                                  "offlist_hops": c.offlist_hops}) as rec:
                r = src.fetch_url(c.url, allow, session=session)
                rec["units"] = {**rec["units"], "status": r.status, "bytes": len(r.content),
                                "tls_verified": r.tls_verified, "content_type": r.content_type}
                if r.status != 200 or not r.content:
                    rec["status"] = "http_error"
                    skipped.append({"url": c.url, "reason": f"status {r.status}"})
                    continue
                ext = src.ext_for(r.content_type, r.final_url, r.content, r.filename)
                sha, path = src.snapshot(ctx.sources_dir, r.content, ext)
                rec["output_path"] = str(path)
        except Exception as e:  # noqa: BLE001
            skipped.append({"url": c.url, "reason": f"{e.__class__.__name__}: {str(e)[:150]}"})
            continue
        if any(d.sha256 == sha for d in docs):
            continue
        doc = Doc(url=c.url, final_url=r.final_url, referrer=c.referrer, fetched_at=now_iso(), sha256=sha, path=path,
                  ext=ext, content_type=r.content_type, status=r.status, tls_verified=r.tls_verified,
                  authority=c.authority, via=",".join(c.via), query=c.query, hop=c.offlist_hops,
                  title=r.filename or c.title)
        doc.extracted = _extract(log, doc)
        docs.append(doc)
        src.append_manifest(ctx, _manifest_row(ctx, brief, doc, batch))
        if c.depth >= MAX_DEPTH:
            continue
        follow = []
        for url, anchor in doc.extracted.links:
            url = src.canonical(url)
            if not url or not _usable(url) or url in seen or src.drive_direct(url) in seen:
                continue
            official = allow.is_allowed(url)
            doc_host = allow.is_document_host(url)
            if not official and not (doc_host and c.offlist_hops < allow.max_offlist_hops
                                     and (c.offlist_hops == 0 or allow.is_document_host(c.url))):
                continue
            s = link_score(url, anchor, qterms)
            if s:
                follow.append((s, url, anchor, official))
        for s, url, anchor, official in sorted(follow, key=lambda t: -t[0])[:MAX_FOLLOW_PER_PAGE]:
            seen.update({url, src.drive_direct(url)})
            queue.append(Candidate(url=url, score=s, via=["link"], query=anchor[:120], title=anchor[:120],
                                   referrer=c.url, depth=c.depth + 1,
                                   offlist_hops=0 if official else c.offlist_hops + 1,
                                   authority=allow.authority(url) if official else c.authority))
    return docs, skipped


def _extract(log: RunLogger, doc: Doc) -> src.Extracted:
    with log.call(AGENT, "extract", units={"sha256": doc.sha256[:12], "ext": doc.ext}) as rec:
        try:
            ex = src.extract(doc.path, doc.ext, doc.final_url)
        except Exception as e:  # noqa: BLE001
            ex = src.Extracted("", "", [], f"error:{e.__class__.__name__}")
        rec["units"] = {**rec["units"], "method": ex.method, "chars": len(ex.text), "links": len(ex.links)}
        if ex.method in ("js_shell", "none") or ex.method.startswith("error"):
            rec["status"] = "skipped"
    if not doc.title:
        doc.title = ex.title
    return ex


def _manifest_row(ctx: RunContext, brief: Brief, d: Doc, batch: str) -> dict:
    return {"source_id": "", "url": d.url, "final_url": d.final_url, "referrer": d.referrer, "fetched_at": d.fetched_at,
            "sha256": d.sha256, "path": src.rel_path(d.path), "content_type": d.content_type, "status": d.status,
            "tls_verified": d.tls_verified, "title": d.title, "text_path": "", "authority": d.authority,
            "run_id": ctx.run_id, "batch": batch, "brief_id": brief.id, "hop": d.hop, "via": d.via, "query": d.query}


# ---------------------------------------------------------------- offline replay
def replay(ctx: RunContext, log: RunLogger, brief: Brief) -> tuple[list[Doc], str]:
    rows = src.read_manifest(ctx)
    mine = [r for r in rows if r.get("run_id") == ctx.run_id] or [r for r in rows if r.get("brief_id") == brief.id]
    if not mine:
        raise RuntimeError(f"--offline: no rows for run {ctx.run_id} / brief {brief.id} in {src.manifest_path(ctx)}")
    batch = mine[-1].get("batch", "")
    docs = []
    for r in (x for x in mine if x.get("batch", "") == batch):
        path = ctx.sources_dir / Path(r["path"]).name
        with log.call(AGENT, "replay", units={"url": r["url"], "sha256": r["sha256"][:12]}) as rec:
            if not path.exists() or sha256_file(path) != r["sha256"]:
                rec["status"] = "missing_snapshot"
                continue
            rec["output_path"] = str(path)
        doc = Doc(url=r["url"], final_url=r["final_url"], referrer=r.get("referrer", ""), fetched_at=r["fetched_at"],
                  sha256=r["sha256"], path=path, ext=path.suffix.lstrip("."), content_type=r["content_type"],
                  status=int(r["status"] or 0), tls_verified=str(r["tls_verified"]).lower() == "true",
                  authority=r.get("authority") or "other", via=r.get("via", ""), query=r.get("query", ""),
                  hop=int(r.get("hop") or 0), title=r.get("title", ""))
        doc.extracted = _extract(log, doc)
        docs.append(doc)
    return docs, batch


# ---------------------------------------------------------------- build outputs
def build_outputs(ctx: RunContext, docs: list[Doc]) -> tuple[list[SourceDoc], list[Chunk], list]:
    sources: list[SourceDoc] = []
    chunks: list[Chunk] = []
    texts: list[tuple[str, str, str]] = []
    for d in docs:
        ex = d.extracted
        if ex is None or len(ex.text.strip()) < MIN_TEXT_CHARS or ex.method == "js_shell":
            continue
        sid = f"S{len(sources) + 1}"
        text_file = ctx.sources_dir / f"{d.sha256}.md"
        if not text_file.exists() or text_file.read_text(encoding="utf-8") != ex.text:
            text_file.write_text(ex.text, encoding="utf-8")
        sources.append(SourceDoc(source_id=sid, url=d.url, final_url=d.final_url, referrer=d.referrer,
                                 fetched_at=d.fetched_at, sha256=d.sha256, path=src.rel_path(d.path),
                                 content_type=d.content_type, status=d.status, tls_verified=d.tls_verified,
                                 title=(d.title or ex.title)[:200], text_path=src.rel_path(text_file),
                                 authority=d.authority))  # type: ignore[arg-type]
        chunks += chunk_text(sid, d.url, ex.text)
        texts.append((sid, d.authority, ex.text))
    return sources, chunks, extract_contacts(texts)


def resolve_data_path(ctx: RunContext, p: str) -> Path:
    path = Path(p)
    if path.is_absolute() and path.exists():
        return path
    if (ROOT / path).exists():
        return ROOT / path
    return ctx.sources_dir / path.name


# ---------------------------------------------------------------- node
def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        brief = Brief.model_validate(state["brief"])
        plan = state.get("plan") or {}
        loop = state.get("loop", 0)
        queries = research_queries(brief, plan, state.get("feedback", {}).get(AGENT, []))
        allow = src.Allowlist()
        trace: dict = {"queries": queries, "offline": ctx.offline}
        if ctx.offline:
            docs, batch = replay(ctx, log, brief)
            trace["replayed_batch"] = batch
        else:
            batch = f"{ctx.run_id}@{ctx.started_at}#L{loop}"
            seeds, dtrace = discover(ctx, log, allow, queries)
            docs, skipped = crawl(ctx, log, allow, brief, seeds, queries, batch)
            trace.update(dtrace, batch=batch, seeds=[c.__dict__ for c in seeds], skipped=skipped)
        sources, chunks, contacts = build_outputs(ctx, docs)
        trace["fetched"] = [{"url": d.url, "final_url": d.final_url, "referrer": d.referrer, "via": d.via,
                             "authority": d.authority, "hop": d.hop, "method": d.extracted.method if d.extracted else "",
                             "chars": len(d.extracted.text) if d.extracted else 0} for d in docs]
        ctx.write_json("sources.json", [s.model_dump(mode="json") for s in sources])
        ctx.write_json("contacts_whitelist.json", [c.model_dump(mode="json") for c in contacts])
        ctx.write_json("research_trace.json", trace)
        save_chunks(ctx.artifact("chunks.json"), chunks)
        log.record(agent=AGENT, tool="summary", units={"sources": len(sources), "chunks": len(chunks),
                                                       "contacts": len(contacts), "offline": ctx.offline},
                   output_path=str(ctx.artifact("sources.json")))
        return {"sources": [s.model_dump(mode="json") for s in sources],
                "contacts": [c.model_dump(mode="json") for c in contacts]}

    return node
