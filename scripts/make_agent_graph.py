"""Compile the LangGraph with a mock context and write docs/agent_graph.mmd (+ .png, rendered offline).

    python scripts/make_agent_graph.py               # .mmd + .png
    python scripts/make_agent_graph.py --no-png

PNG: `npx -y @mermaid-js/mermaid-cli@<pin>` using the Chrome that HyperFrames already installed (no download
when found), else pygraphviz. Never calls mermaid.ink (draw_mermaid_png's default needs internet + a 3rd party).
Agent modules that are missing or fail to import are replaced by stub nodes (listed on stdout), so the
wiring diagram can be drawn before every agent exists.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.context import RunContext  # noqa: E402

MERMAID_CLI = "@mermaid-js/mermaid-cli@11.4.2"


@dataclass
class _TmpContext(RunContext):
    tmp: Path = Path(".")

    @property
    def out_dir(self) -> Path:
        p = self.tmp / "outputs" / self.run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_dir(self) -> Path:
        p = self.tmp / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def sources_dir(self) -> Path:
        p = self.tmp / "data" / "sources"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_db(self) -> Path:
        return self.tmp / "data" / "cache.sqlite"


def compile_graph(tmp: Path):
    """Build the real graph with mock LLM/media; returns (compiled_graph, stubbed_factories)."""
    import core.graph as g
    from core.llm import LLM
    from core.logging import RunLogger

    ctx = _TmpContext(run_id="agent-graph", seed=42, brief_path=ROOT / "briefs" / "P00.yaml",
                      mock_llm=True, mock_media=True, offline=True, tmp=tmp)
    log = RunLogger(ctx)
    llm = LLM(ctx, log, mock=True)
    stubs: list[str] = []
    real_make = g._make

    def safe_make(module, factory, *args):
        try:
            return real_make(module, factory, *args)
        except Exception as e:  # noqa: BLE001 (missing module, heavy import, factory needing real files)
            stubs.append(f"agents.{module}.{factory} ({e.__class__.__name__}: {str(e)[:80]})")
            return lambda state: {}

    g._make = safe_make
    try:
        graph = g.build_graph(ctx, log, llm)
    finally:
        g._make = real_make
    return graph, stubs


def mermaid_text(graph) -> str:
    return graph.get_graph().draw_mermaid()


def _chrome_path() -> str | None:
    cand = os.environ.get("HYPERFRAMES_BROWSER_PATH") or os.environ.get("PUPPETEER_EXECUTABLE_PATH")
    if cand and Path(cand).exists():
        return cand
    for base in (Path.home() / ".cache" / "hyperframes" / "chrome", Path.home() / ".cache" / "puppeteer"):
        if base.exists():
            for name in ("chrome-headless-shell", "chrome-headless-shell.exe", "chrome", "chrome.exe"):
                hits = sorted(p for p in base.rglob(name) if p.is_file())
                if hits:
                    return str(hits[-1])
    for name in ("google-chrome", "chromium", "chromium-browser"):
        if shutil.which(name):
            return shutil.which(name)
    return None


def render_png_mmdc(mmd: Path, png: Path, timeout: int = 300) -> bool:
    npx = shutil.which("npx")
    if not npx:
        return False
    env = dict(os.environ)
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "puppeteer.json"
        pcfg: dict = {"args": ["--no-sandbox", "--disable-gpu"]}
        chrome = _chrome_path()
        if chrome:
            pcfg["executablePath"] = chrome
            pcfg["headless"] = "shell" if "headless-shell" in chrome else True
            env["PUPPETEER_SKIP_DOWNLOAD"] = "1"
        cfg.write_text(json.dumps(pcfg), encoding="utf-8")
        cmd = [npx, "-y", MERMAID_CLI, "-i", str(mmd), "-o", str(png), "-w", "1800", "-b", "white", "-p", str(cfg)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"mermaid-cli failed: {e}")
            return False
    if r.returncode != 0 or not png.exists():
        print(f"mermaid-cli exit {r.returncode}: {(r.stderr or r.stdout)[-600:]}")
        return False
    return True


def render_png_graphviz(graph, png: Path) -> bool:
    try:
        graph.get_graph().draw_png(str(png))
        return png.exists()
    except Exception as e:  # noqa: BLE001 (pygraphviz missing or graphviz error)
        print(f"pygraphviz fallback failed: {e.__class__.__name__}: {e}")
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "agent_graph.mmd")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as td:
        graph, stubs = compile_graph(Path(td))
        text = mermaid_text(graph)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
        print(f"wrote {a.out} ({len(text.splitlines())} lines)")
        if stubs:
            print("stubbed (not importable yet):\n  " + "\n  ".join(stubs))
        if a.no_png:
            return 0
        png = a.out.with_suffix(".png")
        if render_png_mmdc(a.out, png) or render_png_graphviz(graph, png):
            print(f"wrote {png}")
            return 0
    print("PNG not rendered (needs Node + npx, or pygraphviz). The .mmd renders on GitHub as-is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
