"""Build the free discovery catalogue in data/sitemaps/ (run in the Build phase; output is committed).

    python scripts/crawl_sitemaps.py                # sitemaps + page titles + notices + host probe
    python scripts/crawl_sitemaps.py --no-titles    # URLs only (fast)

Writes one JSON per sitemap host ({sitemap, crawled_at, entries: [{url, title, description, h1, lastmod,
kind}]}), notices_<host>.json from the AJAX notice listings, and hosts_check.json (which allowlisted hosts
respond; used to curate config/domains_allowlist.yaml).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.context import now_iso  # noqa: E402
from core.sources import (NOTICE_HEADERS, SITEMAPS_DIR, Allowlist, _session, canonical, fetch_url,  # noqa: E402
                          host_of, notices_request, parse_notices)


def parse_sitemap(xml: str) -> list[dict]:
    out = []
    for block in re.findall(r"<url>(.*?)</url>", xml, flags=re.S):
        loc = re.search(r"<loc>\s*(.*?)\s*</loc>", block, flags=re.S)
        lm = re.search(r"<lastmod>\s*(.*?)\s*</lastmod>", block, flags=re.S)
        if loc:
            out.append({"url": canonical(unescape(loc.group(1))), "lastmod": lm.group(1) if lm else ""})
    return [e for e in out if e["url"]]


def page_meta(url: str, allow: Allowlist) -> dict:
    from lxml import html as lh

    try:
        r = fetch_url(url, allow, session=_session())
        if r.status != 200 or "html" not in r.content_type:
            return {"status": r.status}
        doc = lh.fromstring(r.content.decode("utf-8", "replace"))
        desc = doc.xpath("//meta[@name='description']/@content")
        h1 = doc.xpath("//h1")
        return {"status": r.status, "title": (doc.findtext(".//title") or "").strip(),
                "description": (desc[0] if desc else "").strip()[:300],
                "h1": re.sub(r"\s+", " ", h1[0].text_content()).strip()[:200] if h1 else ""}
    except Exception as e:  # noqa: BLE001
        return {"status": 0, "error": e.__class__.__name__}


def crawl_sitemap(url: str, allow: Allowlist, titles: bool) -> dict:
    r = fetch_url(url, allow)
    entries = parse_sitemap(r.content.decode("utf-8", "replace"))
    if titles:
        with ThreadPoolExecutor(8) as pool:
            metas = list(pool.map(lambda e: page_meta(e["url"], allow), entries))
        for e, m in zip(entries, metas):
            e.update(m)
    for e in entries:
        e["kind"] = "page"
    print(f"{url}: {len(entries)} urls", flush=True)
    return {"sitemap": url, "crawled_at": now_iso(), "entries": entries}


def crawl_notices(cfg: dict, allow: Allowlist, max_pages: int = 30) -> dict:
    entries, seen = [], set()
    for page in range(0, max_pages * 10, 10):          # the page parameter is a row offset (10 per page)
        url, data = notices_request(cfg["endpoint"], cfg["institute_id"], page)
        r = fetch_url(url, allow, method="POST", data=data, headers=NOTICE_HEADERS)
        rows = [n for n in parse_notices(r.content.decode("utf-8", "replace")) if n["url"] not in seen]
        if r.status != 200 or not rows:
            break
        for n in rows:
            seen.add(n["url"])
            entries.append({**n, "referrer": cfg["referrer"]})
    print(f"{cfg['endpoint']}: {len(entries)} notices", flush=True)
    return {"endpoint": cfg["endpoint"], "institute_id": cfg["institute_id"], "crawled_at": now_iso(),
            "entries": entries}


def probe(host: str, allow: Allowlist) -> dict:
    try:
        r = fetch_url(f"https://{host}/", allow, session=_session())
        return {"host": host, "status": r.status, "tls_verified": r.tls_verified, "final_url": r.final_url,
                "allowed": allow.is_allowed(host), "authority": allow.authority(host)}
    except Exception as e:  # noqa: BLE001
        return {"host": host, "status": 0, "error": f"{e.__class__.__name__}: {str(e)[:120]}",
                "allowed": allow.is_allowed(host), "authority": allow.authority(host)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-titles", action="store_true")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args(argv)
    allow = Allowlist()
    SITEMAPS_DIR.mkdir(parents=True, exist_ok=True)
    for sm in allow.cfg.get("sitemaps", []):
        out = crawl_sitemap(sm, allow, titles=not a.no_titles)
        (SITEMAPS_DIR / f"{host_of(sm)}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for nc in allow.cfg.get("notices", []):
        out = crawl_notices(nc, allow)
        (SITEMAPS_DIR / f"notices_{host_of(nc['endpoint'])}.json").write_text(
            json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    if not a.no_probe:
        with ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(lambda h: probe(h, allow), allow.cfg.get("probe_hosts", [])))
        (SITEMAPS_DIR / "hosts_check.json").write_text(json.dumps({"checked_at": now_iso(), "hosts": rows}, indent=1),
                                                       encoding="utf-8")
        for r in rows:
            print(f"  {r['host']:32s} {r['status']} tls={r.get('tls_verified', '-')} {r.get('error', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
