import json

from pydantic import BaseModel

from core.llm import inline_refs
from core.logging import verify_chain
from core.schemas import Brief, ScriptOut, StoryboardOut
from core.seeds import stage_seed


def test_stage_seed_deterministic_and_loop_sensitive():
    assert stage_seed(42, "scriptwriter", 0, 0) == stage_seed(42, "scriptwriter", 0, 0)
    assert stage_seed(42, "scriptwriter", 0, 0) != stage_seed(42, "scriptwriter", 0, 1)


def test_log_hash_chain_detects_edit(ctx, log):
    for i in range(3):
        log.record(agent="t", tool="x", units={"i": i})
    ok, n, _ = verify_chain(log.path)
    assert ok and n == 3
    lines = log.path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[1])
    rec["cost_inr"] = 99
    lines[1] = json.dumps(rec)
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, n, _ = verify_chain(log.path)
    assert not ok and n == 3


def test_inline_refs_removes_refs():
    for schema in (ScriptOut, StoryboardOut):
        s = json.dumps(inline_refs(schema.model_json_schema()))
        assert "$ref" not in s and "$defs" not in s


def test_inline_refs_keeps_fields_named_title():
    class Inner(BaseModel):
        title: str

    class Outer(BaseModel):
        title: str
        items: list[Inner]

    s = inline_refs(Outer.model_json_schema())
    assert "title" in s["properties"]
    assert "title" in s["properties"]["items"]["items"]["properties"]


def test_mock_llm_returns_valid_schema(llm):
    class Out(BaseModel):
        a: str
        b: list[int]

    from core import prompts

    p = prompts.PROMPTS_DIR / "_test_v1.md"
    p.write_text("## system\nsys\n## user\nhello {{name}}\n", encoding="utf-8")
    try:
        out = llm.structured("t", "_test_v1", Out, name="x")
        assert isinstance(out, Out)
    finally:
        p.unlink()


def test_brief_examples_validate():
    import yaml

    from tests.conftest import ROOT

    for f in ("P00.yaml", "C01.yaml"):
        Brief.model_validate(yaml.safe_load((ROOT / "briefs" / f).read_text(encoding="utf-8")))
