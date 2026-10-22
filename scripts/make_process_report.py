"""Fill docs/process_report_template.md from a run's outputs + logs -> docs/process_report.md (+ .pdf).

    python scripts/make_process_report.py                    # latest run in outputs/
    python scripts/make_process_report.py --run-id C01-s42 --no-pdf

PDF path (HyperFrames-free): markdown -> HTML (markdown-it-py) -> PDF with headless Chrome, driven by
puppeteer-core from renderer/node_modules when Node is available, else Chrome's own --print-to-pdf.
If no Chrome is found the .md/.html are kept and instructions are printed.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_cost_report import build_rows, load_records  # noqa: E402

TEMPLATE = ROOT / "docs" / "process_report_template.md"
_PH = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")


# ---------------------------------------------------------------- helpers
def md_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "_none_"
    esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def latest_run(outputs: Path) -> str | None:
    runs = [d for d in outputs.glob("*") if d.is_dir() and (d / "run_meta.json").exists()]
    return max(runs, key=lambda d: d.stat().st_mtime).name if runs else None


def _ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- sections
def brief_section(out: Path, meta: dict) -> tuple[str, dict]:
    brief = {}
    bp = Path(meta.get("brief_path", ""))
    for cand in (bp, ROOT / bp):
        if str(bp) and cand.is_file():
            brief = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
            break
    plan = read_json(out / "plan.json", {}) or {}
    rows = [[k, ", ".join(map(str, v)) if isinstance(v, list) else v] for k, v in brief.items()]
    if plan:
        rows += [["rule packs (planner)", ", ".join(plan.get("rule_packs", []))],
                 ["word budget", f"{plan.get('word_budget_min')}-{plan.get('word_budget_max')}"],
                 ["shots", f"{plan.get('shot_count_min')}-{plan.get('shot_count_max')}"]]
    return md_table(["field", "value"], rows), brief


def stage_times(records: list[dict], out: Path) -> tuple[str, str]:
    st = read_json(out / "stage_times.json") or (read_json(out / "final.json", {}) or {}).get("stage_times")
    rows = []
    if st:
        rows = [[k, f"{v:.1f}"] for k, v in st.items()]
        hdr = ["stage", "seconds (graph timer)"]
    else:  # derive from the log: per agent sum of call latency + first/last timestamp
        agg: dict[str, dict] = {}
        for r in records:
            a = r.get("agent") or "?"
            g = agg.setdefault(a, {"calls": 0, "busy": 0.0, "first": r.get("ts"), "last": r.get("ts")})
            g["calls"] += 1
            g["busy"] += float(r.get("latency_s") or 0)
            g["last"] = r.get("ts")
        rows = [[a, g["calls"], f"{g['busy']:.1f}", g["first"], g["last"]] for a, g in agg.items()]
        hdr = ["agent", "logged calls", "busy s (sum of call latency)", "first", "last"]
    ts = [t for t in (_ts(r.get("ts")) for r in records) if t]
    wall = f"{(max(ts) - min(ts)).total_seconds() / 60:.1f} min" if len(ts) > 1 else "n/a"
    return md_table(hdr, rows), wall


def cost_section(records: list[dict]) -> str:
    rows = build_rows(records)
    stage_rows = {}
    for r in rows[:-1]:
        s = stage_rows.setdefault(r["stage"], [0, 0, 0, 0.0, 0.0, 0.0])
        s[0] += r["calls"]; s[1] += r["input_tokens"]; s[2] += r["output_tokens"]  # noqa: E702
        s[3] += r["wall_s"]; s[4] += r["gpu_seconds"]; s[5] += r["shadow_cost_inr"]  # noqa: E702
    t = rows[-1]
    table = [[k, v[0], v[1], v[2], f"{v[3]:.1f}", f"{v[4]:.1f}", f"{v[5]:.2f}"] for k, v in stage_rows.items()]
    table.append(["**TOTAL**", t["calls"], t["input_tokens"], t["output_tokens"], f"{t['wall_s']:.1f}",
                  f"{t['gpu_seconds']:.1f}", f"{t['shadow_cost_inr']:.2f}"])
    return md_table(["stage", "calls", "input tok", "output tok", "wall s", "GPU s", "shadow Rs"], table) + \
        f"\n\nActual cost: **Rs {t['cost_inr']:.2f}**."


def critic_section(out: Path, records: list[dict]) -> str:
    reports = sorted(out.glob("qa_report*.json"))
    parts = []
    loops = [r for r in records if r.get("agent") == "critic"]
    for p in reports:
        q = read_json(p, {}) or {}
        fails = [c for c in q.get("checks", []) if not c.get("passed")]
        parts.append(f"**{p.name}**: loop {q.get('loop', '?')}, passed={q.get('passed')}, "
                     f"{len(q.get('checks', []))} checks, {len(fails)} failed")
        if fails:
            parts.append(md_table(["check", "value", "threshold", "routed to", "severity"],
                                  [[c.get("check_id"), c.get("value"), c.get("threshold"),
                                    c.get("responsible_agent"), c.get("severity")] for c in fails]))
    hist = read_json(out / "history.json")
    if hist:
        parts.append(md_table(["loop", "check", "artifact sha"],
                              [[h.get("loop"), h.get("check_id"), str(h.get("artifact_sha", ""))[:12]] for h in hist]))
    parts.append(f"Critic log lines: {len(loops)}.")
    return "\n\n".join(parts) if reports else "_No qa_report.json in this run yet._ " + parts[-1]


def hooks_section(out: Path) -> str:
    hab = read_json(out / "hook_ab.json")
    if not hab:
        return "_hook_ab.json not found._"
    hooks = hab.get("hooks") or []
    scores = hab.get("scores") or []
    winner = hab.get("winner", hab.get("hook_index"))
    rows = []
    for i, h in enumerate(hooks):
        s = [x for x in scores if x.get("hook_index") == i]
        avg = lambda k: (sum(x.get(k, 0) for x in s) / len(s)) if s else ""  # noqa: E731
        rows.append([i, h.get("style", ""), h.get("display", ""), avg("topic_clarity"), avg("audience_fit"),
                     avg("tone_fit"), "yes" if i == winner else ""])
    return md_table(["#", "style", "hook", "topic clarity", "audience fit", "tone fit", "winner"], rows)


def interventions_section(run_id: str, records: list[dict]) -> str:
    rows = []
    p = ROOT / "logs" / "interventions.md"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5 and cells[1] == run_id:
                rows.append(cells)
    reviews = [r for r in records if r.get("tool") == "review_checkpoint"]
    rv = md_table(["time", "decision"], [[r.get("ts"), json.dumps(r.get("units", {}))[:160]] for r in reviews])
    return (f"Interventions: **{len(rows)}** (limit 5, target 0).\n\n"
            + md_table(["time", "run", "stage", "what changed", "why"], rows)
            + "\n\nMandatory review checkpoints (A/C; logged, not interventions):\n\n" + rv)


def sources_section(out: Path) -> str:
    facts = read_json(out / "facts.json") or []
    if isinstance(facts, dict):
        facts = facts.get("facts", [])
    man = ROOT / "data" / "sources_manifest.csv"
    n_src = 0
    if man.exists():
        with open(man, encoding="utf-8") as f:
            n_src = sum(1 for _ in csv.DictReader(f))
    rows = [[f.get("fact_id"), f.get("claim", "")[:90], f.get("source_id"), f.get("url", ""),
             f"{float(f.get('nli_score') or 0):.2f}"] for f in facts[:30]]
    return (f"Snapshotted sources (all runs): {n_src}. Facts used in this run: {len(facts)}.\n\n"
            + md_table(["fact", "claim", "source", "url", "NLI"], rows))


def environment_section(out: Path) -> str:
    env = (read_json(out / "run_manifest.json", {}) or {}).get("environment", {})
    keep = ["python", "platform", "on_kaggle", "kaggle_run_type", "git_commit", "nvidia_smi", "/etc/build_date"]
    rows = [[k, str(env.get(k, ""))[:120]] for k in keep if k in env]
    return md_table(["key", "value"], rows)


def outputs_section(out: Path) -> str:
    final = read_json(out / "final.json") or {}
    files = sorted(p for p in out.glob("*") if p.is_file() and p.suffix in (".mp4", ".srt", ".json", ".md", ".csv"))
    rows = [[p.name, f"{p.stat().st_size / 1e6:.2f} MB"] for p in files]
    return (md_table(["file", "size"], rows) + ("\n\nfinal: `" + json.dumps(final) + "`" if final else ""))


# ---------------------------------------------------------------- render
def build_report(run_id: str, template: Path = TEMPLATE, log_path: Path | None = None) -> str:
    out = ROOT / "outputs" / run_id
    meta = read_json(out / "run_meta.json", {}) or {}
    log_path = log_path or ROOT / "logs" / "run_log.jsonl"
    records = load_records(log_path, run_id) if log_path.exists() else []
    brief_md, brief = brief_section(out, meta)
    st_md, wall = stage_times(records, out)
    try:
        from core.logging import verify_chain

        ok, n, msg = verify_chain(log_path) if log_path.exists() else (False, 0, "no log")
        chain = f"{'OK' if ok else 'BROKEN'} ({n} lines: {msg})"
    except Exception as e:  # noqa: BLE001
        chain = f"not verified ({e})"
    code = str(brief.get("id", "P"))[:1].upper()
    values = {
        "team_id": meta.get("team_id", "TXX"), "brief_id": brief.get("id", run_id), "run_id": run_id,
        "seed": meta.get("seed", 42), "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "brief_table": brief_md, "stage_times_table": st_md, "wall_clock": wall,
        "autonomy": "L3 (mandatory human review before export)" if code in "AC" else "L4 (no human in the loop)",
        "cost_table": cost_section(records) if records else "_no log lines for this run_",
        "critic_loops": critic_section(out, records), "hooks": hooks_section(out),
        "interventions": interventions_section(run_id, records), "sources": sources_section(out),
        "environment": environment_section(out), "outputs": outputs_section(out), "log_chain": chain,
    }
    return _PH.sub(lambda m: str(values.get(m.group(1), m.group(0))), template.read_text(encoding="utf-8"))


def to_html(md: str, title: str) -> str:
    from markdown_it import MarkdownIt

    body = MarkdownIt("commonmark", {"html": False}).enable("table").render(md)
    css = ("body{font-family:'Fira Sans','Noto Sans',Arial,sans-serif;max-width:900px;margin:32px auto;"
           "color:#231F20;font-size:12px;line-height:1.45}h1{color:#B7202E}h2{color:#B7202E;border-bottom:1px solid "
           "#ddd;padding-bottom:4px}table{border-collapse:collapse;width:100%;margin:8px 0}td,th{border:1px solid "
           "#ccc;padding:3px 6px;text-align:left;vertical-align:top}th{background:#f4f4f4}img{max-width:100%}"
           "code{background:#f4f4f4;padding:1px 3px}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{css}</style></head><body>{body}</body></html>")


def find_chrome() -> str | None:
    from make_agent_graph import _chrome_path

    p = _chrome_path()
    if p:
        return p
    for c in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"):
        if Path(c).exists():
            return c
    return None


PUPPETEER_JS = """
import { createRequire } from 'module';
const [, , renderer, chrome, src, dst] = process.argv;
const require = createRequire(renderer + '/package.json');
const puppeteer = require('puppeteer-core');
const browser = await puppeteer.launch({ executablePath: chrome, headless: true,
  args: ['--no-sandbox', '--disable-gpu'] });
const page = await browser.newPage();
await page.goto('file://' + src, { waitUntil: 'load' });
await page.pdf({ path: dst, format: 'A4', printBackground: true,
  margin: { top: '14mm', bottom: '14mm', left: '12mm', right: '12mm' } });
await browser.close();
"""


def html_to_pdf(src: Path, dst: Path) -> str:
    """Returns the method used, or '' if no PDF could be made."""
    chrome = find_chrome()
    if not chrome:
        return ""
    renderer = ROOT / "renderer"
    if shutil.which("node") and (renderer / "node_modules" / "puppeteer-core").exists():
        with tempfile.TemporaryDirectory() as td:
            js = Path(td) / "pdf.mjs"
            js.write_text(PUPPETEER_JS, encoding="utf-8")
            r = subprocess.run(["node", str(js), renderer.as_posix(), chrome, src.resolve().as_posix(),
                                str(dst.resolve())], capture_output=True, text=True, timeout=180)
            if r.returncode == 0 and dst.exists():
                return "puppeteer-core"
            print(f"puppeteer-core failed: {(r.stderr or r.stdout)[-300:]}")
    r = subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={dst.resolve()}", src.resolve().as_uri()],
                       capture_output=True, text=True, timeout=180)
    return "chrome --print-to-pdf" if dst.exists() else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "process_report.md")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    run_id = a.run_id or latest_run(ROOT / "outputs")
    if not run_id:
        print("no runs in outputs/ (need outputs/<run_id>/run_meta.json)")
        return 1
    md = build_report(run_id)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    print(f"wrote {a.out} (run {run_id})")
    if a.no_pdf:
        return 0
    htm = a.out.with_suffix(".html")
    htm.write_text(to_html(md, f"AgentReel process report {run_id}"), encoding="utf-8")
    pdf = a.out.with_suffix(".pdf")
    method = html_to_pdf(htm, pdf)
    if method:
        print(f"wrote {pdf} via {method}")
    else:
        print(f"PDF not produced (no Chrome found). Open {htm} in a browser and Print -> Save as PDF, or run "
              "`bash scripts/setup_kaggle.sh` first so HyperFrames' Chrome headless shell exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
