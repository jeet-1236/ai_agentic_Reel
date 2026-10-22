"""Category rule packs (config/rules/<pack>.yaml) and style presets (config/styles.yaml).

    packs = load_packs(["base", "C"])           # merged dict
    rules_text(packs, "script")                 # bullet list for prompts
    code_rule_ids(packs)                        # ids the critic checks in code
    style_for("C"), voice_for("A"), negative_prompt("C"), is_sensitive("A")
"""
from __future__ import annotations

from functools import lru_cache

from core.context import ROOT, load_yaml

RULES_DIR = ROOT / "config" / "rules"
APPLIES_TO = ("script", "storyboard", "frames", "contacts")
REQUIRED_RULE_KEYS = ("id", "text", "applies_to", "check", "responsible_agent")


@lru_cache(maxsize=None)
def _load_pack(name: str) -> dict:
    data = load_yaml(f"config/rules/{name}.yaml")
    if data.get("pack") != name:
        raise ValueError(f"rule pack file {name}.yaml declares pack={data.get('pack')!r}")
    for r in data.get("rules", []):
        missing = [k for k in REQUIRED_RULE_KEYS if k not in r]
        if missing:
            raise ValueError(f"rule {r.get('id')} in pack {name} misses {missing}")
        if r["check"] not in ("llm", "code"):
            raise ValueError(f"rule {r['id']}: check must be llm|code")
    return data


def available_packs() -> list[str]:
    return sorted(p.stem for p in RULES_DIR.glob("*.yaml"))


def load_packs(names: list[str] | tuple[str, ...]) -> dict:
    """Merge packs in order ("base" is always included first).

    Returns {packs, descriptions, mandatory_review, style_preset, rules}; each rule gets a `pack` key.
    The style preset of the last pack that declares one wins (category over base).
    """
    order = ["base"] + [n for n in names if n != "base"]
    merged: dict = {"packs": [], "descriptions": {}, "mandatory_review": False, "style_preset": "", "rules": []}
    seen: set[str] = set()
    for name in order:
        if name in merged["packs"]:
            continue
        pack = _load_pack(name)
        merged["packs"].append(name)
        merged["descriptions"][name] = pack.get("description", "")
        merged["mandatory_review"] = merged["mandatory_review"] or bool(pack.get("mandatory_review"))
        if pack.get("style_preset"):
            merged["style_preset"] = pack["style_preset"]
        for r in pack.get("rules", []):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            merged["rules"].append({**r, "pack": name})
    if not merged["style_preset"]:
        merged["style_preset"] = load_styles().get("default_preset", "flat_vector")
    return merged


def _as_packs(packs: dict | list[str]) -> dict:
    return packs if isinstance(packs, dict) else load_packs(list(packs))


def _applies(rule: dict, applies_to: str | None) -> bool:
    if applies_to is None:
        return True
    at = rule["applies_to"]
    return applies_to in (at if isinstance(at, list) else [at])


def rules_text(packs: dict | list[str], applies_to: str | None = None) -> str:
    """'- [C.policy_quoted] Policy statements only as quoted ...' lines for a prompt."""
    rules = [r for r in _as_packs(packs)["rules"] if _applies(r, applies_to)]
    return "\n".join(f"- [{r['id']}] {r['text']}" for r in rules) or "- (no extra rules)"


def code_rule_ids(packs: dict | list[str]) -> list[str]:
    return [r["id"] for r in _as_packs(packs)["rules"] if r["check"] == "code"]


def rules_for_agent(packs: dict | list[str], agent: str) -> list[dict]:
    return [r for r in _as_packs(packs)["rules"] if r["responsible_agent"] == agent]


# ---------------------------------------------------------------- styles
@lru_cache(maxsize=1)
def load_styles() -> dict:
    return load_yaml("config/styles.yaml")


def is_sensitive(category: str) -> bool:
    return category in load_styles().get("sensitive_categories", [])


def style_for(category: str, preset: str | None = None) -> dict:
    """Resolved preset: {name, prefix, suffix, negative, palette_hint, sensitive, sensitive_suffix}."""
    st = load_styles()
    cat = st.get("categories", {}).get(category, {})
    name = preset or cat.get("preset") or st.get("default_preset", "flat_vector")
    p = st["presets"].get(name) or st["presets"][st.get("default_preset", "flat_vector")]
    return {
        "name": name,
        "prefix": p.get("prefix", ""),
        "suffix": p.get("suffix", ""),
        "negative": negative_prompt(category, name),
        "palette_hint": st.get("palette", {}).get("prompt_hint", ""),
        "sensitive": is_sensitive(category),
        "sensitive_suffix": st.get("sensitive_suffix", "no people, no faces"),
        "mood": cat.get("mood", ""),
    }


def negative_prompt(category: str, preset: str | None = None) -> str:
    st = load_styles()
    name = preset or st.get("categories", {}).get(category, {}).get("preset") or st.get("default_preset")
    extra = st.get("presets", {}).get(name, {}).get("negative_extra", "")
    neg = " ".join(st.get("negative_prompt", "").split())
    if is_sensitive(category):
        neg += ", people, person, human figure with face, crowd"
    return f"{neg}, {extra}" if extra else neg


def voice_for(category: str) -> dict:
    """{voice, speed} Kokoro preset for the category (default af_heart, 1.0)."""
    cat = load_styles().get("categories", {}).get(category, {})
    return {"voice": cat.get("voice", "af_heart"), "speed": float(cat.get("speed", 1.0))}


def caption_style() -> dict:
    return dict(load_styles().get("caption", {}))


def palette() -> dict:
    return {k: v for k, v in load_styles().get("palette", {}).items() if k != "prompt_hint"}
