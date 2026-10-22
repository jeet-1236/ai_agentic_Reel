"""Versioned prompt templates: prompts/<prompt_id>.md with `## system` and `## user` sections.

Placeholders use {{name}}. Values that are not strings are JSON-encoded. The returned sha256 covers the
raw template file, so editing a prompt changes prompt_sha256 in logs and invalidates the stage cache.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.context import ROOT
from core.logging import sha256_text

PROMPTS_DIR = ROOT / "prompts"
_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_id: str
    sha256: str
    system: str
    user: str


def _section(text: str, name: str) -> str:
    m = re.search(rf"^##\s*{name}\s*$(.*?)(?=^##\s*\w+\s*$|\Z)", text, flags=re.M | re.S)
    return m.group(1).strip() if m else ""


def render(prompt_id: str, **variables) -> RenderedPrompt:
    raw = (PROMPTS_DIR / f"{prompt_id}.md").read_text(encoding="utf-8")

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in variables:
            raise KeyError(f"prompt {prompt_id} needs variable '{key}'")
        v = variables[key]
        return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1, default=str)

    system = _PLACEHOLDER.sub(sub, _section(raw, "system"))
    user = _PLACEHOLDER.sub(sub, _section(raw, "user"))
    return RenderedPrompt(prompt_id, sha256_text(raw), system, user)


def prompt_sha(prompt_id: str) -> str:
    return sha256_text((PROMPTS_DIR / f"{prompt_id}.md").read_text(encoding="utf-8"))
