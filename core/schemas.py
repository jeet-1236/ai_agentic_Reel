"""Pydantic contracts shared by every agent, worker and check.

Conventions (see docs/CONVENTIONS.md):
- Models used as LLM structured output are suffixed `Out` and kept flat: no Optional/anyOf,
  defaults instead of None, enums as plain strings validated in code. `core.llm` inlines $refs.
- Everything persisted to outputs/<run_id>/ is one of these models dumped with `model_dump(mode="json")`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["A", "B", "C", "D", "E", "P"]
Aspect = Literal["9x16", "16x9"]
AgentName = Literal[
    "planner", "researcher", "fact_curator", "scriptwriter", "hook_judge",
    "storyboarder", "asset_generator", "editor", "critic", "human_review", "export",
]


# ---------------------------------------------------------------- brief / plan
class Brief(BaseModel):
    id: str
    category: str                     # free text on the card, e.g. "Responsible AI Usage"
    topic: str
    audience: str
    format: Literal["video_9x16", "video_16x9"] = "video_9x16"
    duration_s: int = Field(45, ge=30, le=60)
    tone: str = "clear, friendly"
    language: str = "English"
    must_include: list[str] = []
    must_avoid: list[str] = []
    cta: str = ""

    @property
    def aspect(self) -> Aspect:
        return "16x9" if self.format == "video_16x9" else "9x16"

    @property
    def category_code(self) -> Category:
        c = self.id[:1].upper()
        return c if c in "ABCDE" else "P"  # type: ignore[return-value]


class StageBudget(BaseModel):
    stage: str
    minutes: float
    max_retries: int = 1


class Plan(BaseModel):
    brief_id: str
    category: Category
    rule_packs: list[str]             # e.g. ["base", "C"]
    aspect: Aspect
    width: int
    height: int
    fps: int = 30
    duration_s: int
    word_budget_min: int
    word_budget_max: int
    shot_count_min: int
    shot_count_max: int
    mandatory_review: bool
    queries: list[str]
    audience_insight: str
    success_criteria: list[str]
    topic_keywords: list[str]         # used by the hook check
    budgets: list[StageBudget]


class PlanOut(BaseModel):
    """LLM part of the plan."""
    queries: list[str]
    audience_insight: str
    success_criteria: list[str]
    topic_keywords: list[str]


# ---------------------------------------------------------------- sources / facts
class SourceDoc(BaseModel):
    source_id: str                    # S1, S2 ... (stable order = manifest order)
    url: str
    final_url: str
    referrer: str = ""
    fetched_at: str
    sha256: str
    path: str                         # data/sources/<sha256>.<ext>
    content_type: str
    status: int
    tls_verified: bool = True
    title: str = ""
    text_path: str = ""               # extracted markdown/text
    authority: Literal["gov", "somaiya", "other"] = "other"


class Chunk(BaseModel):
    chunk_id: str                     # S3#c7
    source_id: str
    url: str
    heading: str = ""
    text: str


class Fact(BaseModel):
    fact_id: str                      # F1, F2 ...
    claim: str
    quote: str                        # verbatim substring of the chunk (normalised whitespace)
    chunk_id: str
    source_id: str
    url: str
    nli_score: float = 0.0


class FactProposalOut(BaseModel):
    claim: str
    quote: str
    chunk_id: str


class FactProposalsOut(BaseModel):
    facts: list[FactProposalOut]


class Contact(BaseModel):
    kind: Literal["phone", "email", "url"]
    value: str                        # normalised (E.164-ish digits for phones, lowercase email/url)
    display: str                      # as written in the source
    label: str = ""                   # e.g. "National Anti-Ragging Helpline"
    source_ids: list[str]
    authority: Literal["gov", "somaiya", "other"]
    conflict_note: str = ""


class CoverageItem(BaseModel):
    item: str
    fact_ids: list[str] = []
    status: Literal["covered", "MISSING"] = "MISSING"


# ---------------------------------------------------------------- script
class ScriptSentenceOut(BaseModel):
    display: str                      # caption text (digits, symbols allowed)
    spoken: str                       # TTS text (numbers/acronyms spelled for speech)
    fact_ids: list[str]               # [] only for non-factual connective lines


class HookOut(BaseModel):
    style: str                        # question | surprising_fact | direct_address
    display: str
    spoken: str
    fact_ids: list[str]


class ScriptOut(BaseModel):
    hooks: list[HookOut]              # exactly 3
    body: list[ScriptSentenceOut]
    cta: ScriptSentenceOut


class HookScoreOut(BaseModel):
    hook_index: int
    topic_clarity: int                # 1-10: topic clear within 3 s
    audience_fit: int
    tone_fit: int
    unsourced_claim: bool
    reason: str


class HookJudgeOut(BaseModel):
    scores: list[HookScoreOut]


class Script(BaseModel):
    hook: HookOut
    hook_index: int
    body: list[ScriptSentenceOut]
    cta: ScriptSentenceOut
    word_count: int                   # SPOKEN words (VO length); compared with plan.word_budget_*

    def sentences(self) -> list[ScriptSentenceOut]:
        h = ScriptSentenceOut(display=self.hook.display, spoken=self.hook.spoken, fact_ids=self.hook.fact_ids)
        return [h, *self.body, self.cta]


# ---------------------------------------------------------------- storyboard
TEMPLATES = (
    "hook_title", "illustration_kenburns", "do_dont_split", "step_list",
    "stat_callout", "contact_card", "quote_card", "end_card",
)


class ShotOut(BaseModel):
    shot_id: int
    template: str                     # one of TEMPLATES
    sentence_indices: list[int]       # indices into Script.sentences() narrated over this shot
    visual_prompt: str                # "" when the template needs no generated image
    on_screen_text: list[str]         # template slots, each <= 7 words
    fact_ids: list[str]

    @field_validator("template")
    @classmethod
    def _known_template(cls, v: str) -> str:
        if v not in TEMPLATES:
            raise ValueError(f"template must be one of {TEMPLATES}")
        return v


class StoryboardOut(BaseModel):
    shots: list[ShotOut]
    style_preset: str = ""            # filled in code from the rule pack (config/styles.yaml key)
    negative_prompt: str = ""         # filled in code


# ---------------------------------------------------------------- assets
class WordTiming(BaseModel):
    word: str                         # display-form word
    start: float                      # seconds on the final timeline
    end: float
    sentence_index: int


class CaptionPage(BaseModel):
    index: int
    start: float
    end: float
    lines: list[str]                  # <= 2 lines
    words: list[WordTiming]


class ShotAsset(BaseModel):
    shot_id: int
    image_path: str = ""
    candidates: list[str] = []
    seed: int = 0
    face_count: int = 0
    nsfw_score: float = 0.0
    vlm_ok: bool = True
    vlm_note: str = ""
    visual_prompt: str = ""           # storyboard prompt this image was made for (stale if the storyboard changes)
    start: float = 0.0                # filled by the editor from voice timings
    end: float = 0.0


class VoiceAsset(BaseModel):
    wav_path: str
    duration_s: float
    voice: str
    sentence_spans: list[tuple[float, float]]
    words: list[WordTiming]
    asr_text: str = ""
    wer: float = 0.0


class AssetRecord(BaseModel):
    """One row of assets/manifest.csv."""
    path: str
    kind: str                         # image | voice | caption | logo | font | render
    generator: str                    # model id or "official" / "renderer"
    model_revision: str = ""
    seed: int | str = ""
    prompt_sha256: str = ""
    licence: str
    sha256: str
    source_url: str = ""


# ---------------------------------------------------------------- QA / review
class QACheck(BaseModel):
    check_id: str
    passed: bool
    value: str = ""
    threshold: str = ""
    evidence: str = ""
    responsible_agent: AgentName
    severity: Literal["hard", "soft"] = "hard"


class QAReport(BaseModel):
    run_id: str
    loop: int
    passed: bool
    checks: list[QACheck]

    def failures(self) -> list[QACheck]:
        return [c for c in self.checks if not c.passed]


class SoftReviewOut(BaseModel):
    """LLM critic for tone / safe messaging / rule packs."""
    passed: bool
    violations: list[str]
    responsible_agent: str            # scriptwriter | storyboarder | asset_generator
    fix_instructions: str


class FrameJudgeOut(BaseModel):
    """VLM verdict for one frame."""
    realistic_human_face: bool
    garbled_text: bool
    fear_or_victim_imagery: bool
    matches_intent: bool
    note: str


class ReviewDecision(BaseModel):
    approved: bool
    note: str = ""
    target_agent: str = ""            # where to send a rejection
