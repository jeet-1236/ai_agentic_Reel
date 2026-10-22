"""Structured LLM / VLM calls against local llama-server (OpenAI-compatible), with logging and repair.

Usage inside an agent:
    out = llm.structured("scriptwriter", "scriptwriter_v1", ScriptOut, creative=True,
                         seed=stage_seed(ctx.seed, "scriptwriter", 0, loop), brief=..., facts=...)

- JSON-schema constrained decoding via `response_format` (llama.cpp converts it to a grammar).
  Schemas are $ref-inlined first (llama.cpp mishandles nested $ref).
- One repair retry with the validation error appended; then raises LLMError.
- `mock=True` returns deterministic schema-valid objects (tests / local dev without a GPU), or a
  registered fixture: `llm.fixtures["scriptwriter_v1"] = lambda vars: {...}`.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from core.context import RunContext, load_yaml
from core.logging import RunLogger
from core.prompts import render

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------- schema helpers
def inline_refs(schema: dict) -> dict:
    """Return a copy of a pydantic JSON schema with every $ref replaced by its $defs body."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any, in_properties: bool = False) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and not in_properties:
                name = node["$ref"].split("/")[-1]
                return walk(copy.deepcopy(defs[name]))
            if in_properties:  # keys are field names (a field may be called "title")
                return {k: walk(v) for k, v in node.items()}
            return {k: walk(v, in_properties=(k == "properties")) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def fake_instance(schema: dict, seed_text: str = "") -> Any:
    """Deterministic minimal instance of a JSON schema (mock mode)."""
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return fake_instance(schema["anyOf"][0], seed_text)
    if t == "object":
        return {k: fake_instance(v, f"{seed_text}.{k}") for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [fake_instance(schema.get("items", {"type": "string"}), seed_text + "[0]")]
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return False
    return f"mock {seed_text.strip('.')}"


# ---------------------------------------------------------------- client
class LLM:
    def __init__(self, ctx: RunContext, log: RunLogger, mock: bool | None = None):
        self.ctx = ctx
        self.log = log
        self.mock = ctx.mock_llm if mock is None else mock
        cfg = load_yaml("config/models.yaml")
        self.text_cfg = cfg["llm"]["text"]
        self.vision_cfg = cfg["llm"]["vision"]
        self.decoding = cfg["decoding"]
        self.fixtures: dict[str, Callable[[dict], dict]] = {}
        self._clients: dict[str, Any] = {}

    def _client(self, base_url: str, model: str, temperature: float, seed: int, max_tokens: int, thinking: bool):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=base_url,
            api_key="local",
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            timeout=600,
            max_retries=2,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}, "cache_prompt": False},
        )

    # ------------------------------------------------------------ public API
    def structured(self, agent: str, prompt_id: str, schema: type[T], *, creative: bool = False,
                   seed: int = 0, images: list[str] | None = None, **variables) -> T:
        rp = render(prompt_id, **variables)
        dec = self.decoding["creative_steps" if creative else "json_steps"]
        cfg = self.vision_cfg if images else self.text_cfg
        base_url = self.ctx.vlm_base_url if images else self.ctx.llm_base_url
        js = inline_refs(schema.model_json_schema())

        if self.mock:
            return self._mock(agent, prompt_id, rp.sha256, schema, js, seed, variables)

        messages: list[dict] = [{"role": "system", "content": rp.system}, {"role": "user", "content": self._content(rp.user, images)}]
        last_err = ""
        for attempt in range(2):
            if last_err:
                messages = messages + [{"role": "user", "content":
                    f"Your previous answer was invalid: {last_err[:800]}\nReturn ONLY valid JSON matching the schema."}]
            client = self._client(base_url, cfg["alias"], dec["temperature"], seed, dec["max_tokens"], dec["thinking"])
            bound = client.bind(response_format={"type": "json_schema",
                                                 "json_schema": {"name": schema.__name__, "schema": js, "strict": True}})
            with self.log.call(agent, "llm" if not images else "vlm", model=cfg["alias"],
                               model_revision=cfg.get("revision", ""), prompt_id=prompt_id,
                               prompt_sha256=rp.sha256, seed=seed, gpu_id=cfg["server"]["gpu"],
                               units={"attempt": attempt, "images": len(images or [])}) as rec:
                msg = bound.invoke(messages)
                usage = getattr(msg, "usage_metadata", None) or {}
                rec["input_tokens"] = usage.get("input_tokens", 0)
                rec["output_tokens"] = usage.get("output_tokens", 0)
                text = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                try:
                    obj = schema.model_validate_json(text)
                    rec["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                    return obj
                except ValidationError as e:
                    rec["status"] = "invalid_output"
                    rec["error"] = str(e)[:1500]
                    last_err = str(e)
                    messages = messages + [{"role": "assistant", "content": text}]
        raise LLMError(f"{agent}/{prompt_id}: invalid structured output after repair: {last_err[:300]}")

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _content(text: str, images: list[str] | None):
        if not images:
            return text
        parts: list[dict] = [{"type": "text", "text": text}]
        for p in images:
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            mime = "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        return parts

    def _mock(self, agent, prompt_id, sha, schema: type[T], js: dict, seed: int, variables: dict) -> T:
        data = self.fixtures[prompt_id](variables) if prompt_id in self.fixtures else fake_instance(js, schema.__name__)
        text = json.dumps(data, sort_keys=True)
        with self.log.call(agent, "llm", model="mock", prompt_id=prompt_id, prompt_sha256=sha, seed=seed,
                           units={"mock": True}) as rec:
            rec["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            return schema.model_validate(data)
