"""Official-source discovery: allowlist, search (Tavily / sitemap catalogue / ddgs), fetch, snapshot, extract.

Functions here are side-effect free apart from network I/O and writing snapshots; logging of each call is
done by the caller (agents/researcher.py) with `log.call(...)`.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse, urlunparse

from core.context import ROOT, RunContext, load_yaml
from core.schemas import SourceDoc

UA = "Mozilla/5.0 (X11; Linux x86_64) AgentReel/1.0 (official-source research bot; Somaiya AgentReel Challenge)"
TIMEOUT = (10, 40)
MAX_BYTES = 30 * 1024 * 1024
SITEMAPS_DIR = ROOT / "data" / "sitemaps"
MANIFEST_FIELDS = [*SourceDoc.model_fields.keys(), "run_id", "batch", "brief_id", "hop", "via", "query"]


# ---------------------------------------------------------------- allowlist
def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def _suffix_match(host: str, domains: list[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


class Allowlist:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg if cfg is not None else load_yaml("config/domains_allowlist.yaml")
        self.somaiya: list[str] = [d.lower() for d in self.cfg.get("somaiya", [])]
        self.gov: list[str] = [d.lower() for d in self.cfg.get("gov", [])]
        self.gov_suffixes: list[str] = [d.lower() for d in self.cfg.get("gov_suffixes", ["gov.in", "nic.in"])]
        self.document_hosts: list[str] = [d.lower() for d in self.cfg.get("document_hosts", [])]
        self.max_offlist_hops: int = int(self.cfg.get("max_offlist_hops", 1))

    def is_allowed(self, url_or_host: str) -> bool:
        host = host_of(url_or_host) if "/" in url_or_host else url_or_host.lower()
        return bool(host) and _suffix_match(host, self.somaiya + self.gov)

    def is_document_host(self, url_or_host: str) -> bool:
        host = host_of(url_or_host) if "/" in url_or_host else url_or_host.lower()
        return _suffix_match(host, self.document_hosts)

    def authority(self, url_or_host: str) -> str:
        host = host_of(url_or_host) if "/" in url_or_host else url_or_host.lower()
        if _suffix_match(host, self.gov) or (self.is_allowed(host) and _suffix_match(host, self.gov_suffixes)):
            return "gov"
        if _suffix_match(host, self.somaiya):
            return "somaiya"
        return "other"

    def include_domains(self) -> list[str]:
        return self.somaiya + self.gov


# ---------------------------------------------------------------- text helpers
STOPWORDS = set("""a an and are as at be by can do does for from has have how i in is it its of on or our should
that the their this to was what when where which who why will with you your we us not vs about into than
them they these those any all also more most may must""".split())


def stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    for suf in ("ations", "ation", "ings", "ing", "ness", "ment", "ed", "es", "s"):
        if len(w) - len(suf) >= 3 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    return w[:7]


def terms(text: str) -> list[str]:
    """Lowercased, stop-word-free, lightly stemmed tokens (used for BM25 + link scoring)."""
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def _tok_match(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def overlap(query_terms: list[str], text_terms: list[str]) -> int:
    """Number of distinct query terms present in the text (prefix match for stems >= 4 chars)."""
    ts = set(text_terms)
    return sum(1 for q in set(query_terms) if q in ts or any(_tok_match(q, t) for t in ts))


def url_words(url: str) -> str:
    p = urlparse(url)
    return re.sub(r"[-_/+.%]+", " ", unquote(p.netloc.split(".")[0] + " " + p.path)).strip()


# ---------------------------------------------------------------- URL normalisation
_DRIVE_FILE = re.compile(r"/file/d/([A-Za-z0-9_-]{10,})")


def drive_direct(url: str) -> str:
    """drive.google.com/file/d/<id>/view (or open?id=<id>) -> the uc?export=download form."""
    host = host_of(url)
    if host not in ("drive.google.com", "docs.google.com"):
        return url
    m = _DRIVE_FILE.search(urlparse(url).path)
    fid = m.group(1) if m else (parse_qs(urlparse(url).query).get("id") or [""])[0]
    if not fid or "/document/" in url or "/spreadsheets/" in url or "/presentation/" in url:
        return url
    return f"https://drive.google.com/uc?export=download&id={fid}"


def canonical(url: str) -> str:
    url = unescape(url.strip()).replace(" ", "%20")
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return ""
    path = quote(unquote(p.path), safe="/%+:@&=,;~!$'()*") or "/"
    return urlunparse((p.scheme, p.netloc.lower(), path, "", p.query, ""))


# ---------------------------------------------------------------- fetch + snapshot
@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str
    content: bytes
    tls_verified: bool = True
    error: str = ""
    filename: str = ""


def _session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-IN,en;q=0.9"})
    return s


def _get(session, url: str, verify: bool, method: str = "GET", data: dict | None = None,
         headers: dict | None = None):
    r = session.request(method, url, timeout=TIMEOUT, verify=verify, allow_redirects=True, stream=True, data=data,
                        headers=headers)
    content = r.raw.read(MAX_BYTES + 1, decode_content=True)
    return r, content[:MAX_BYTES]


def fetch_url(url: str, allowlist: Allowlist, *, method: str = "GET", data: dict | None = None,
              headers: dict | None = None, session=None) -> FetchResult:
    """GET with TLS verification; retry verify=False only for allowlisted hosts (tls_verified=False)."""
    import requests
    import urllib3

    session = session or _session()
    target = drive_direct(url)
    tls = True
    try:
        r, content = _get(session, target, True, method, data, headers)
    except requests.exceptions.SSLError:
        if not allowlist.is_allowed(target):
            raise
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        tls = False
        r, content = _get(session, target, False, method, data, headers)
    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
    if allowlist.is_document_host(r.url) and ctype == "text/html" and b"download-form" in content:
        r, content = _drive_confirm(session, content)  # "can't scan for viruses" interstitial
        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
    fname = ""
    m = re.search(r'filename="?([^";]+)"?', r.headers.get("content-disposition", ""))
    if m:
        fname = m.group(1)
    return FetchResult(url=url, final_url=r.url, status=r.status_code, content_type=ctype, content=content,
                       tls_verified=tls, filename=fname)


def _drive_confirm(session, html: bytes):
    from lxml import html as lh

    doc = lh.fromstring(html)
    form = doc.get_element_by_id("download-form")
    params = {i.get("name"): i.get("value", "") for i in form.xpath(".//input[@type='hidden']")}
    return _get(session, form.get("action") + "?" + "&".join(f"{k}={quote(v)}" for k, v in params.items()), True)


def ext_for(content_type: str, url: str, content: bytes, filename: str = "") -> str:
    if content[:5] == b"%PDF-" or "pdf" in content_type:
        return "pdf"
    if "html" in content_type or content.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html")):
        return "html"
    for name in (filename, urlparse(url).path):
        suf = Path(name).suffix.lower().lstrip(".")
        if suf in ("txt", "json", "xml", "csv", "png", "jpg", "jpeg", "docx"):
            return suf
    return "html" if "text" in content_type else "bin"


def snapshot(sources_dir: Path, content: bytes, ext: str) -> tuple[str, Path]:
    sha = hashlib.sha256(content).hexdigest()
    path = sources_dir / f"{sha}.{ext}"
    if not path.exists():
        path.write_bytes(content)
    return sha, path


def manifest_path(ctx: RunContext) -> Path:
    return ctx.sources_dir.parent / "sources_manifest.csv"


def append_manifest(ctx: RunContext, row: dict) -> None:
    p = manifest_path(ctx)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def read_manifest(ctx: RunContext) -> list[dict]:
    p = manifest_path(ctx)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rel_path(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# ---------------------------------------------------------------- extraction
@dataclass
class Extracted:
    title: str
    text: str
    links: list[tuple[str, str]] = field(default_factory=list)   # (absolute url, anchor text)
    method: str = ""                                              # trafilatura | pypdf | ocr | js_shell | none


def extract(path: Path, ext: str, url: str) -> Extracted:
    data = Path(path).read_bytes()
    if ext == "pdf":
        return _extract_pdf(data, url)
    if ext in ("html", "xml", "txt"):
        if ext == "txt":
            return Extracted("", data.decode("utf-8", "replace"), [], "text")
        return _extract_html(data, url)
    return Extracted("", "", [], "none")


def _extract_html(data: bytes, url: str) -> Extracted:
    import trafilatura
    from lxml import html as lh

    html = data.decode("utf-8", "replace")
    md = trafilatura.extract(html, url=url, output_format="markdown", with_metadata=True, favor_recall=True,
                             include_tables=True, include_comments=False, include_links=False) or ""
    title, body = "", md
    m = re.match(r"^---\n(.*?)\n---\n?", md, flags=re.S)
    if m:
        body = md[m.end():]
        t = re.search(r"^title:\s*(.+)$", m.group(1), flags=re.M)
        title = t.group(1).strip().strip('"') if t else ""
    links: list[tuple[str, str]] = []
    try:
        doc = lh.fromstring(html)
        if not title:
            title = (doc.findtext(".//title") or "").strip()
        for a in doc.xpath("//a[@href]"):
            href = canonical(urljoin(url, a.get("href", "")))
            if href:
                anchor = re.sub(r"\s+", " ", a.text_content() or a.get("title", "")).strip()
                links.append((href, anchor[:200]))
        noscript = " ".join(doc.xpath("//noscript//text()")).lower()
    except Exception:  # noqa: BLE001
        noscript = ""
    method = "trafilatura"
    if len(body.strip()) < 200 and ("javascript" in noscript or "enable js" in noscript or "<app-root" in html
                                    or 'id="root"' in html or 'id="app"' in html):
        method = "js_shell"
    return Extracted(title, body.strip(), links, method)


def _extract_pdf(data: bytes, url: str) -> Extracted:
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(data))
    pages, links = [], []
    for page in reader.pages:
        frags: list[tuple[float, float, str]] = []

        def visit(text, cm, tm, _fd, _fs, frags=frags):
            if text.strip():
                frags.append((tm[4] * cm[1] + tm[5] * cm[3] + cm[5], tm[4] * cm[0] + tm[5] * cm[2] + cm[4], text.strip()))

        try:
            pages.append(page.extract_text(visitor_text=visit) or "")
        except Exception:  # noqa: BLE001
            pages.append("")
        for annot in page.get("/Annots") or []:
            try:
                o = annot.get_object()
                uri = (o.get("/A") or {}).get("/URI")
                if not uri:
                    continue
                y1, y2 = float(o["/Rect"][1]), float(o["/Rect"][3])
                same = sorted((f for f in frags if y1 - 4 <= f[0] <= y2 + 4), key=lambda f: f[1])
                links.append((canonical(str(uri)), " ".join(f[2] for f in same)[:200]))
            except Exception:  # noqa: BLE001
                continue
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    title = ""
    try:
        title = str((reader.metadata or {}).get("/Title") or "")
    except Exception:  # noqa: BLE001
        pass
    if len(text.strip()) < 50:
        ocr = _ocr_pdf(data)
        if ocr:
            return Extracted(title, ocr, [lk for lk in links if lk[0]], "ocr")
    return Extracted(title, text, [lk for lk in links if lk[0]], "pypdf")


def _ocr_pdf(data: bytes) -> str:
    """Tesseract OCR of a scanned PDF; returns "" when pytesseract/pypdfium2/tesseract are unavailable."""
    try:
        import pypdfium2
        import pytesseract
    except ImportError:
        return ""
    try:
        pdf = pypdfium2.PdfDocument(data)
        out = []
        for i in range(min(len(pdf), 30)):
            img = pdf[i].render(scale=2).to_pil()
            out.append(pytesseract.image_to_string(img, lang="eng"))
        return "\n\n".join(t.strip() for t in out if t.strip())
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------- search
def tavily_key() -> str:
    key = os.environ.get("TAVILY_API_KEY", "")
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret("TAVILY_API_KEY") or ""
    except Exception:  # noqa: BLE001
        return ""


def tavily_search(query: str, allowlist: Allowlist, key: str, max_results: int = 10) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=key)
    kw = dict(search_depth="basic", max_results=max_results, include_domains=allowlist.include_domains())
    try:
        res = client.search(query, include_domains_mode="restrict", **kw)
    except TypeError:
        res = client.search(query, **kw)
    return [{"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")[:300]}
            for r in res.get("results", [])]


def ddgs_search(query: str, max_results: int = 10) -> list[dict]:
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=max_results, region="in-en", safesearch="moderate") or []
    return [{"url": r.get("href", ""), "title": r.get("title", ""), "snippet": r.get("body", "")[:300]} for r in rows]


class Catalogue:
    """BM25 over the cached sitemap/notice catalogue (data/sitemaps/*.json): title + description + URL words."""

    def __init__(self, entries: list[dict]):
        import bm25s

        seen, self.entries = set(), []
        for e in entries:
            if e.get("url") and e["url"] not in seen:
                seen.add(e["url"])
                self.entries.append(e)
        docs = [terms(" ".join([e.get("title", ""), e.get("title", ""), e.get("description", ""),
                                e.get("h1", ""), url_words(e["url"])])) or ["_"] for e in self.entries]
        self.bm25 = bm25s.BM25()
        if docs:
            self.bm25.index(docs, show_progress=False)

    @classmethod
    def load(cls, directory: Path = SITEMAPS_DIR, extra: list[dict] | None = None) -> "Catalogue":
        entries: list[dict] = []
        for p in sorted(Path(directory).glob("*.json")):
            if p.name == "hosts_check.json":
                continue
            entries += json.loads(p.read_text(encoding="utf-8")).get("entries", [])
        return cls(entries + (extra or []))

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        q = terms(query)
        if not self.entries or not q:
            return []
        docs, scores = self.bm25.retrieve([q], k=min(k, len(self.entries)), show_progress=False)
        return [(self.entries[int(i)], float(s)) for i, s in zip(docs[0], scores[0]) if s > 0]


# ---------------------------------------------------------------- notices (AJAX listing)
NOTICE_HEADERS = {"X-Requested-With": "XMLHttpRequest"}


def notices_request(endpoint: str, institute_id: int, page: int) -> tuple[str, dict]:
    data = {"page_no": page, "category_id": "", "sub_category_id": "", "department_id": "", "academic_year": "",
            "institute_id": institute_id, "lang": "en"}
    return f"{endpoint}{page}", data


def parse_notices(html: str) -> list[dict]:
    """Notice listing HTML -> [{url, title, date}] (title/URL come from the share-by-email link)."""
    out, seen = [], set()
    for m in re.finditer(r"mailto:\?subject=([^&]*)&(?:amp;)?body=([^'\"]*)", html):
        title = unescape(m.group(1)).strip().strip('"').strip()
        u = re.search(r"https?://\S+", unquote(unescape(m.group(2))))
        url = canonical(u.group(0)) if u else ""
        if url and url not in seen:
            seen.add(url)
            out.append({"url": url, "title": title, "kind": "notice"})
    return out
