"""Research stack (planner, researcher, fact_curator, sources, retrieval, contacts): offline + mock only."""
import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from core import sources as src
from core.contacts import extract_contacts
from core.retrieval import HybridRetriever, chunk_text, worker_backends
from core.schemas import Brief, Fact

FIX = Path(__file__).parent / "fixtures" / "research"
ROOT = Path(__file__).resolve().parent.parent

COE = "https://www.somaiya.edu/en/test-examination/"
RAG_SVU = "https://www.somaiya.edu/en/test-anti-ragging/"
RAG_GOV = "https://www.antiragging.in/test-helpline/"
PAGES = {
    COE: ("coe.html", "text/html"),
    RAG_SVU: ("ragging_somaiya.html", "text/html"),
    RAG_GOV: ("ragging_gov.html", "text/html"),
    "https://drive.google.com/uc?export=download&id=FIXTUREdoc11111": ("circular.pdf", "application/pdf"),
}
CATALOGUE = [
    {"url": COE, "title": "Centre of Examination | Somaiya Vidyavihar University", "description": "examination rules"},
    {"url": RAG_SVU, "title": "Anti-Ragging Committee | Somaiya", "description": "anti ragging helpline"},
    {"url": RAG_GOV, "title": "National Anti-Ragging Helpline", "description": "anti ragging helpline"},
    {"url": "https://evil-somaiya.edu/fake", "title": "Unfair means exam anti ragging", "description": ""},
]
BRIEF = {
    "id": "C99", "category": "Responsible AI Usage", "topic": "Exam unfair means and the anti-ragging helpline",
    "audience": "all students", "duration_s": 45,
    "must_include": ["what counts as unfair means in exams", "who decides on unfair-means cases",
                     "anti-ragging helpline number", "the college library opening hours"],
    "cta": "Read the examination rules on the official website",
}


@pytest.fixture
def online(monkeypatch):
    """Serve fixtures instead of the network; record every URL requested."""
    requested: list[str] = []

    def fake_fetch(url, allowlist, *, method="GET", data=None, headers=None, session=None):
        target = src.drive_direct(url)
        requested.append(target)
        if target not in PAGES:
            return src.FetchResult(url, target, 404, "text/html", b"not found")
        name, ctype = PAGES[target]
        return src.FetchResult(url, target, 200, ctype, (FIX / name).read_bytes())

    monkeypatch.setattr(src, "fetch_url", fake_fetch)
    monkeypatch.setattr(src, "tavily_key", lambda: "")
    monkeypatch.setattr(src, "ddgs_search", lambda q, max_results=10: [])
    monkeypatch.setattr(src.Catalogue, "load", classmethod(lambda cls, directory=None, extra=None: cls(CATALOGUE)))
    return requested


def _run_research(ctx, log, llm):
    from agents import fact_curator, planner, researcher

    state = {"brief": BRIEF, "loop": 0}
    state.update(planner.make_node(ctx, log, llm)(state))
    state.update(researcher.make_node(ctx, log, llm)(state))
    state.update(fact_curator.make_node(ctx, log, llm)(state))
    return state


# ---------------------------------------------------------------- unit tests
def test_allowlist_hosts_and_authority():
    a = src.Allowlist()
    assert a.is_allowed("https://kjsce.somaiya.edu/en/notices/")
    assert a.is_allowed("https://svu-files.somaiya.edu/coe/x.pdf")
    assert a.is_allowed("https://static.pib.gov.in/x")
    assert not a.is_allowed("https://evil-somaiya.edu/")
    assert not a.is_allowed("https://somaiya.edu.evil.com/")
    assert not a.is_allowed("https://drive.google.com/file/d/abc/view")
    assert a.is_document_host("https://drive.google.com/file/d/abc/view")
    assert a.authority("https://www.antiragging.in/") == "gov"
    assert a.authority("https://i4c.mha.gov.in/") == "gov"
    assert a.authority("https://kjsce.somaiya.edu/") == "somaiya"
    assert a.authority("https://example.com/") == "other"


def test_drive_link_conversion():
    assert src.drive_direct("https://drive.google.com/file/d/1nRpz5W10pNESA5sfxX1QU_A2TNEWxgft/view?usp=sharing") == \
        "https://drive.google.com/uc?export=download&id=1nRpz5W10pNESA5sfxX1QU_A2TNEWxgft"
    assert src.drive_direct("https://drive.google.com/open?id=abcdefghijkl") == \
        "https://drive.google.com/uc?export=download&id=abcdefghijkl"
    assert src.drive_direct("https://www.somaiya.edu/x") == "https://www.somaiya.edu/x"


def test_planner_code_fields(ctx, log, llm):
    from agents import planner

    c01 = yaml.safe_load((ROOT / "briefs" / "C01.yaml").read_text(encoding="utf-8"))
    plan = planner.make_node(ctx, log, llm)({"brief": c01, "loop": 0})["plan"]
    assert plan["rule_packs"] == ["base", "C"] and plan["mandatory_review"]
    assert (plan["width"], plan["height"]) == (1080, 1920)
    # voice-over window = 45 s - 4 s end card - 0.3 s tail = 40.7 s at 2.4 words/s = 97.7 words, ±10 %
    assert (plan["word_budget_min"], plan["word_budget_max"]) == (88, 107)
    assert plan["shot_count_max"] == 9
    assert 8 <= len(plan["queries"]) <= 12
    assert all(i in plan["queries"] for i in c01["must_include"])
    assert (ctx.out_dir / "plan.json").exists()
    p00 = yaml.safe_load((ROOT / "briefs" / "P00.yaml").read_text(encoding="utf-8"))
    p00["format"] = "video_16x9"
    plan = planner.make_node(ctx, log, llm)({"brief": p00, "loop": 0})["plan"]
    assert plan["category"] == "P" and not plan["mandatory_review"] and plan["width"] == 1920


def test_extract_chunk_and_hybrid_retrieval(ctx, log):
    html = src.extract(FIX / "coe.html", "html", COE)
    assert "unfair means" in html.text.lower() and html.method == "trafilatura"
    assert any("FIXTUREdoc11111" in u for u, _ in html.links)
    pdf = src.extract(FIX / "circular.pdf", "pdf", "https://drive.google.com/x")
    assert "Prevention of Unfair Means Committee" in pdf.text and pdf.method == "pypdf"
    assert pdf.links and "FIXTUREdoc22222" in pdf.links[0][0]
    chunks = chunk_text("S1", COE, html.text) + chunk_text("S2", "drive", pdf.text)
    assert chunks[0].chunk_id == "S1#c1" and all("#c" in c.chunk_id for c in chunks)
    embed, rerank = worker_backends(ctx, log, "test")
    hits = HybridRetriever(chunks, embed, rerank).search_many(["who decides unfair means cases committee"], k=3)
    top = hits["who decides unfair means cases committee"][0][0]
    assert top.source_id == "S2"


def test_chunker_respects_headings_and_size():
    text = "# Rules\n\n" + " ".join(f"word{i}." for i in range(700)) + "\n\n## Contacts\n\nCall the office."
    chunks = chunk_text("S4", "u", text)
    assert {c.heading for c in chunks} == {"Rules", "Contacts"}
    assert max(len(c.text.split()) for c in chunks) <= 420
    assert [c.chunk_id for c in chunks][:2] == ["S4#c1", "S4#c2"]


def test_contacts_precedence_and_conflicts():
    som = src.extract(FIX / "ragging_somaiya.html", "html", RAG_SVU).text
    gov = src.extract(FIX / "ragging_gov.html", "html", RAG_GOV).text
    contacts = {(c.kind, c.value): c for c in extract_contacts([("S1", "somaiya", som), ("S2", "gov", gov)])}
    toll = contacts[("phone", "18001805522")]
    assert toll.authority == "gov" and toll.display == "1800-180-5522" and toll.source_ids[0] == "S2"
    assert "l800-180-5522" in toll.conflict_note
    mail = contacts[("email", "helpline@antiragging.in")]
    assert mail.authority == "gov" and "helplin@antiragqins.in" in mail.conflict_note
    assert ("email", "helplin@antiragqins.in") not in contacts
    assert ("phone", "112") in contacts and ("phone", "1930") in contacts
    assert ("phone", "2009") not in contacts               # a year, not a short code
    assert contacts[("email", "antiragging.cell@somaiya.edu")].authority == "somaiya"


def test_find_quote_normalisation():
    from agents.fact_curator import find_quote

    chunk = "Referring to the website link, shared files/ material – downloaded files is “prohibited” by the examiner."
    assert find_quote('shared files/ material - downloaded files is "prohibited"', chunk)
    assert find_quote("SHARED   FILES/MATERIAL", chunk)               # PDF spacing / case differences
    assert find_quote("AI tools are prohibited by the examiner", chunk) == ""


# ---------------------------------------------------------------- end-to-end (mocked network)
def test_research_pipeline_online_then_offline_replay(ctx, log, llm, online, monkeypatch):
    on = dataclasses.replace(ctx, offline=False)
    state = _run_research(on, log, llm)

    by_url = {s["url"]: s for s in state["sources"]}
    drive = "https://drive.google.com/file/d/FIXTUREdoc11111/view?usp=sharing"
    assert COE in by_url and drive in by_url, by_url.keys()
    assert by_url[drive]["referrer"] == COE and by_url[drive]["authority"] == "somaiya"
    assert not any("example.com" in u or "evil-somaiya" in u for u in online)        # off-list never fetched
    assert not any("FIXTUREdoc99999" in u for u in online)                            # irrelevant link not followed
    for s in state["sources"]:
        assert (on.sources_dir / Path(s["path"]).name).exists()
    rows = src.read_manifest(on)
    assert {r["url"] for r in rows} >= {COE, drive} and all(r["sha256"] for r in rows)

    contacts = {(c["kind"], c["value"]): c for c in state["contacts"]}
    assert contacts[("phone", "18001805522")]["authority"] == "gov"
    assert json.loads((on.out_dir / "contacts_whitelist.json").read_text(encoding="utf-8"))

    facts = [Fact.model_validate(f) for f in state["facts"]]
    assert len(facts) >= 3
    assert [f.fact_id for f in facts] == [f"F{i}" for i in range(1, len(facts) + 1)]
    chunks = {c["chunk_id"]: c["text"] for c in json.loads((on.out_dir / "chunks.json").read_text(encoding="utf-8"))}
    from agents.fact_curator import normalise
    for f in facts:
        assert normalise(f.quote) in normalise(chunks[f.chunk_id])
    cov = {c["item"]: c for c in state["coverage"]}
    assert cov["who decides on unfair-means cases"]["status"] == "covered"
    assert cov["the college library opening hours"]["status"] == "MISSING"
    first = {n: (on.out_dir / n).read_text(encoding="utf-8")
             for n in ("sources.json", "chunks.json", "contacts_whitelist.json", "facts.json", "coverage.json")}

    # offline replay: no network, identical artifacts
    def no_net(*a, **k):
        raise AssertionError("network used in --offline")
    monkeypatch.setattr(src, "fetch_url", no_net)
    monkeypatch.setattr(src, "ddgs_search", no_net)
    monkeypatch.setattr(src, "tavily_search", no_net)
    off = dataclasses.replace(ctx, offline=True)
    _run_research(off, log, llm)
    for name, text in first.items():
        assert (off.out_dir / name).read_text(encoding="utf-8") == text, name


def test_fact_curator_drops_unverified_quotes(ctx, log, llm, online):
    from agents import fact_curator, planner, researcher

    on = dataclasses.replace(ctx, offline=False)
    state = {"brief": BRIEF, "loop": 0}
    state.update(planner.make_node(on, log, llm)(state))
    state.update(researcher.make_node(on, log, llm)(state))

    def fixture(v):
        cid = v["chunks"][0]["chunk_id"]
        real = v["chunks"][0]["text"].split(".")[0]
        return {"facts": [{"claim": "Students may use AI tools in all exams.", "quote": "AI tools are allowed in exams",
                           "chunk_id": cid},
                          {"claim": real, "quote": real, "chunk_id": cid},
                          {"claim": "Unrelated claim about fees and hostel rooms charged yearly.", "quote": real,
                           "chunk_id": cid}]}
    llm.fixtures["fact_curator_v1"] = fixture
    out = fact_curator.make_node(on, log, llm)(state)
    assert len(out["facts"]) == 1                     # fabricated quote dropped; unsupported claim fails NLI
    trace = json.loads((on.out_dir / "fact_curation_trace.json").read_text(encoding="utf-8"))
    reasons = {r["reason"].split()[0] for r in trace["rejected"]}
    assert reasons == {"quote_not_in_chunk", "nli"}


def test_brief_model_round_trip():
    assert Brief.model_validate(BRIEF).category_code == "C"
