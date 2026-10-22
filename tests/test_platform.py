"""Platform tests: notebook, cost report, log verification, servers (mock), agent graph, prefetch plan, scripts."""
from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------- notebook
def test_notebook_is_valid_nbformat_v4():
    nb = json.loads((ROOT / "notebooks" / "kaggle_live.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and isinstance(nb["cells"], list) and len(nb["cells"]) >= 9
    assert nb["cells"][0]["cell_type"] == "markdown"
    text = ""
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code") and isinstance(c["source"], list)
        src = "".join(c["source"])
        text += src
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            ast.parse("\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!")))
    for needle in ("setup_kaggle.sh", "UserSecretsClient", "run.py", "--resume", "--approve", "--reject",
                   "--only", "make_archive", "ipywidgets"):
        assert needle in text, needle
    assert "tvly-" not in text and "hf_" not in text.replace("hf_hub", "")   # no secrets


# ---------------------------------------------------------------- logs / cost
def _synthetic_log(log):
    log.record(agent="planner", tool="llm", model="qwen3.5-9b", input_tokens=800, output_tokens=200,
               latency_s=10.0, gpu_id="0")
    log.record(agent="researcher", tool="tavily_search", units={"credits": 1, "results": 10}, latency_s=1.5)
    log.record(agent="asset_generator", tool="worker:image_sdxl", model="sdxl-lightning", latency_s=36.0,
               gpu_id="1", units={"images": 4, "gpu_seconds": 30.0})
    log.record(agent="scriptwriter", tool="llm", model="mock", latency_s=0.01, gpu_id="0")
    return log.path


def test_cost_report_from_synthetic_log(log, tmp_path):
    import make_cost_report as mcr

    path = _synthetic_log(log)
    out = tmp_path / "cost_report.csv"
    assert mcr.main(["--log", str(path), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0].keys() >= {"stage", "provider", "model", "input_tokens", "output_tokens", "units", "cost_inr",
                              "wall_s", "gpu_seconds", "shadow_cost_inr"}
    by_stage = {r["stage"]: r for r in rows if r["run_id"] not in ("TOTAL", "ASSUMPTIONS")}
    assert by_stage["researcher"]["provider"] == "tavily"
    assert by_stage["planner"]["provider"] == "local"
    assert float(by_stage["planner"]["gpu_seconds"]) == 10.0          # latency on a GPU tool
    assert float(by_stage["asset_generator"]["gpu_seconds"]) == 30.0  # explicit units.gpu_seconds wins
    assert float(by_stage["scriptwriter"]["gpu_seconds"]) == 0.0      # mock calls cost nothing
    assert json.loads(by_stage["asset_generator"]["units"])["images"] == 4
    total = next(r for r in rows if r["run_id"] == "TOTAL")
    assert int(total["input_tokens"]) == 800 and float(total["cost_inr"]) == 0.0
    expected = (40.0 / 3600 * mcr.T4_USD_PER_HOUR + 1 * mcr.TAVILY_USD_PER_CREDIT) * mcr.USD_TO_INR
    assert float(total["shadow_cost_inr"]) == pytest.approx(expected, abs=1e-3)
    assert any(r["run_id"] == "ASSUMPTIONS" for r in rows)


def test_verify_logs_ok_then_tampered(log):
    import verify_logs

    path = _synthetic_log(log)
    log.intervention("scriptwriter", "edited hook", "test")
    assert verify_logs.main([str(path)]) == 0
    s = verify_logs.summarize(path)
    assert s["test-run"]["interventions"] == 1 and s["test-run"]["lines"] == 5
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[2])
    rec["latency_s"] = 1.0
    lines[2] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_logs.main([str(path)]) == 1


# ---------------------------------------------------------------- servers
def test_servers_noop_in_mock(ctx, log):
    from core import servers

    assert ctx.mock_llm
    assert servers.ensure_llm(ctx, log) == ctx.llm_base_url
    assert servers.start_vlm(ctx, log) == ctx.vlm_base_url
    assert servers.stop_vlm(ctx, log) is None
    assert not log.path.exists() or log.path.read_text(encoding="utf-8") == ""   # nothing launched / logged


def test_servers_health_and_command():
    from core import servers
    from core.context import load_yaml

    assert servers.health("http://127.0.0.1:9/v1", timeout=0.5) is False
    assert servers._root_url("http://127.0.0.1:8001/v1") == "http://127.0.0.1:8001"
    cfg = load_yaml("config/models.yaml")["llm"]
    cmd = servers.build_command(Path("llama-server"), Path("m.gguf"), cfg["text"], 42)
    joined = " ".join(cmd)
    for part in ("--alias " + cfg["text"]["alias"], "--host 127.0.0.1", f"--port {cfg['text']['server']['port']}",
                 "-np 1", "-ngl 999", "--seed 42", "-fa on", "--jinja"):
        assert part in joined, part
    vcmd = servers.build_command(Path("llama-server"), Path("v.gguf"), cfg["vision"], 7, Path("mm.gguf"))
    assert vcmd[vcmd.index("--mmproj") + 1] == "mm.gguf"


def test_resolve_model_from_models_dir(tmp_path, monkeypatch):
    from core import servers

    d = tmp_path / "Org__Repo"
    d.mkdir()
    (d / "w.gguf").write_bytes(b"x")
    monkeypatch.setenv("AGENTREEL_MODELS_DIR", str(tmp_path))
    assert servers.resolve_model("Org/Repo", "w.gguf", download=False) == d / "w.gguf"
    assert servers.resolve_model("Org/Repo", download=False) == d
    with pytest.raises(servers.ServerError):
        servers.resolve_model("Org/Missing", "nope.gguf", download=False)


# ---------------------------------------------------------------- agent graph
def test_make_agent_graph_writes_mermaid(tmp_path):
    pytest.importorskip("langgraph")
    import make_agent_graph

    out = tmp_path / "agent_graph.mmd"
    assert make_agent_graph.main(["--no-png", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "graph TD" in text
    for node in ("planner", "critic", "human_review", "export", "image_shot"):
        assert node in text
    assert "mermaid.ink" not in Path(make_agent_graph.__file__).read_text(encoding="utf-8").replace(
        "Never calls mermaid.ink", "")


# ---------------------------------------------------------------- prefetch / scripts
def test_prefetch_plan_covers_models_yaml():
    import yaml

    import prefetch_models as pm

    cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    jobs = {j.key: j for j in pm.plan_downloads(cfg)}
    assert cfg["llm"]["text"]["file"] in jobs["llm.text"].files
    assert cfg["llm"]["vision"]["mmproj"] in jobs["llm.vision"].files
    assert "image.base" in jobs and jobs["image.base"].allow
    assert not any("fallback" in k for k in jobs)
    assert "extra.hhem_tokenizer" in jobs
    assert any("fallback" in j.key for j in pm.plan_downloads(cfg, include_fallbacks=True))


@pytest.mark.parametrize("name", ["setup_kaggle.sh", "build_llamacpp_sm75.sh"])
def test_shell_scripts_are_bash_with_lf(name):
    raw = (ROOT / "scripts" / name).read_bytes()
    assert raw.startswith(b"#!/usr/bin/env bash\n")
    assert b"\r\n" not in raw


def test_notebook_code_cells_compile():
    """Every code cell of the Kaggle notebook must be valid Python (shell `!` lines excluded)."""
    import ast

    nb = json.loads((ROOT / "notebooks" / "kaggle_live.ipynb").read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            code = "\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!"))
            ast.parse(code, filename=f"cell {i}")
