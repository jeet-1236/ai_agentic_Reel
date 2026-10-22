"""LangGraph state. Holds small structured data + artifact paths; large files live in outputs/<run_id>/."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


class AgentState(TypedDict, total=False):
    # inputs
    brief: dict                               # Brief.model_dump()
    plan: dict                                # Plan.model_dump()
    # research
    sources: list[dict]                       # SourceDoc dumps (manifest order)
    facts: list[dict]                         # Fact dumps
    contacts: list[dict]                      # Contact dumps (whitelist)
    coverage: list[dict]                      # CoverageItem dumps
    # creative
    script: dict                              # Script dump
    hook_ab: dict                             # all hooks + scores
    storyboard: dict                          # StoryboardOut dump
    # assets
    shot_assets: Annotated[dict, merge_dicts]  # {str(shot_id): ShotAsset dump}; Send fan-out writes here
    voice: dict                               # VoiceAsset dump
    captions: list[dict]                      # CaptionPage dumps
    srt_path: str
    # render
    scene_path: str
    video_path: str
    # QA / control
    qa_report: dict                           # QAReport dump
    loop: int                                 # repair loop counter (0 = first pass)
    route: str                                # next agent chosen by critic / review
    feedback: Annotated[dict, merge_dicts]    # {agent_name: [instructions...]} from critic/review
    redo_shots: list[int]                     # shots the AssetGenerator must regenerate
    history: Annotated[list, operator.add]    # [{loop, check_id, artifact_sha, agent, routed_to}]
    review: dict                              # ReviewDecision dump
    stage_times: Annotated[dict, merge_dicts]  # {stage: seconds}
    final: dict                               # exported file paths
    errors: Annotated[list, operator.add]
    extra: Annotated[dict, merge_dicts]       # free-form, for stage-specific bookkeeping
