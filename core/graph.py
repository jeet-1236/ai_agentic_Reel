"""The AgentReel LangGraph. Wiring only; agent logic lives in agents/*.py.

Each agent module exports `make_node(ctx, log, llm) -> Callable[[AgentState], dict]`.
agents/asset_generator.py additionally exports make_image_shot_node / make_voice_node.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import time
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, Send

from core.context import RunContext, load_yaml
from core.llm import LLM
from core.logging import RunLogger
from core.prompts import PROMPTS_DIR
from core.state import AgentState

MAX_LOOPS = 3
LINEAR = ["planner", "researcher", "fact_curator", "scriptwriter", "hook_judge", "storyboarder"]
ROUTABLE = {"researcher", "scriptwriter", "storyboarder", "asset_generator", "editor", "human_review", "export"}
NO_CACHE = {"human_review", "export", "critic"}


def _versions_fingerprint() -> str:
    """Prompts + pipeline code + configs: part of every cache key (the built-in key only sees node input,
    so a prompt or code fix would otherwise silently reuse stale stage outputs)."""
    h = hashlib.sha256()
    root = PROMPTS_DIR.parent
    files = [*PROMPTS_DIR.glob("*.md"), *(root / "agents").glob("*.py"), *(root / "core").glob("*.py"),
             *(root / "workers").glob("*.py"), *(root / "config").rglob("*.yaml"),
             *(root / "renderer").glob("*.mjs"), *(root / "renderer" / "templates").rglob("*.*")]
    for p in sorted(files):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _timed(name: str, fn: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
    def wrapper(state: AgentState) -> dict:
        t0 = time.perf_counter()
        out = fn(state) or {}
        out.setdefault("stage_times", {})
        out["stage_times"] = {**out["stage_times"], name: round(time.perf_counter() - t0, 2)}
        return out

    wrapper.__name__ = name
    return wrapper


def _make(module: str, factory: str, ctx, log, llm):
    mod = importlib.import_module(f"agents.{module}")
    return getattr(mod, factory)(ctx, log, llm)


# ---------------------------------------------------------------- routing
def image_is_current(shot: dict, asset: dict | None) -> bool:
    """An image belongs to a shot only while the shot's visual prompt is the one it was generated for.
    Shot ids are reused when a repair loop re-plans the storyboard, so the id alone is not enough."""
    return bool(asset and asset.get("image_path")) and asset.get("visual_prompt", "") == shot.get("visual_prompt", "")


def dispatch_assets(state: AgentState) -> list[Send]:
    """Fan out one image job per shot that needs an image, plus the voice job when needed."""
    shots = state["storyboard"]["shots"]
    have = state.get("shot_assets") or {}
    redo = set(state.get("redo_shots") or [])
    sends = [
        Send("image_shot", {"shot": s, "plan": state["plan"], "brief": state["brief"],
                            "loop": state.get("loop", 0), "feedback": state.get("feedback", {})})
        for s in shots
        if s.get("visual_prompt") and (not image_is_current(s, have.get(str(s["shot_id"]))) or s["shot_id"] in redo)
    ]
    if not state.get("voice") or (state.get("extra") or {}).get("redo_voice"):
        sends.append(Send("voice", state))
    return sends or [Send("assets_join", state)]


def route_after_critic(state: AgentState) -> str:
    r = state.get("route", "export")
    return r if r in ROUTABLE else "export"


# ---------------------------------------------------------------- build
def build_graph(ctx: RunContext, log: RunLogger, llm: LLM, checkpointer=None, cache=None):
    fp = _versions_fingerprint()

    def key_func(state) -> str:
        blob = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(f"{fp}|{ctx.seed}|{blob}".encode()).hexdigest()

    policy = CachePolicy(key_func=key_func)
    g = StateGraph(AgentState)

    def add(name: str, fn, **kw):
        g.add_node(name, _timed(name, fn), cache_policy=None if name in NO_CACHE else policy, **kw)

    for name in LINEAR:
        add(name, _make(name, "make_node", ctx, log, llm))
    add("asset_dispatch", lambda s: {"extra": {"dispatch_loop": s.get("loop", 0)}})
    add("image_shot", _make("asset_generator", "make_image_shot_node", ctx, log, llm))
    add("voice", _make("asset_generator", "make_voice_node", ctx, log, llm))
    add("assets_join", _make("asset_generator", "make_join_node", ctx, log, llm), defer=True)
    add("editor", _make("editor", "make_node", ctx, log, llm))
    add("critic", _make("critic", "make_node", ctx, log, llm))
    add("human_review", _make("human_review", "make_node", ctx, log, llm))
    add("export", _make("export", "make_node", ctx, log, llm))

    g.add_edge(START, "planner")
    for a, b in zip(LINEAR, LINEAR[1:]):
        g.add_edge(a, b)
    g.add_edge("storyboarder", "asset_dispatch")
    g.add_conditional_edges("asset_dispatch", dispatch_assets, ["image_shot", "voice", "assets_join"])
    g.add_edge("image_shot", "assets_join")
    g.add_edge("voice", "assets_join")
    g.add_edge("assets_join", "editor")
    g.add_edge("editor", "critic")
    # critic / human_review set state["route"]; "asset_generator" re-enters through the dispatcher
    targets = {r: ("asset_dispatch" if r == "asset_generator" else r) for r in ROUTABLE}
    g.add_conditional_edges("critic", route_after_critic, targets)
    g.add_conditional_edges("human_review", route_after_critic,
                            {k: v for k, v in targets.items() if k != "human_review"})
    g.add_edge("export", END)

    return g.compile(checkpointer=checkpointer, cache=cache)
