# AgentReel build conventions (read before writing code)

Plan: `PLAN.md`. Shared contracts (do NOT change signatures without updating every caller):
`core/schemas.py`, `core/state.py`, `core/context.py`, `core/logging.py`, `core/llm.py`, `core/prompts.py`,
`core/workers.py`, `core/graph.py`, `run.py`. Adding fields with defaults to schemas is fine.

## Environment
- Target: Kaggle, Python 3.13, Linux, 2× T4. Dev: Windows laptop, `.venv` (Python 3.13) at repo root.
  Run things with `.venv\Scripts\python.exe` (Windows) and install with
  `C:\Users\hp\AppData\Roaming\Python\Python314\Scripts\uv.exe pip install --python .venv\Scripts\python.exe <pkg>`.
  Add every new dependency to `requirements.txt` with an exact `==` pin.
- Heavy ML deps (torch, diffusers, kokoro, faster-whisper, sentence-transformers, onnxruntime-gpu) are NOT
  installed locally. Import them **inside** worker functions only, never at module top level of `core/` or `agents/`.
- Local ffmpeg/ffprobe: `import static_ffmpeg; static_ffmpeg.add_paths()` (dev only; Kaggle has system ffmpeg).
  Code must call `core.ffmpeg_utils.ffmpeg_bin()` / `ffprobe_bin()` which resolve either.
- Paths: use `pathlib`, forward-compatible with Linux. No hard-coded `/kaggle` paths outside `scripts/` and `core/servers.py`.

## Agent node contract
```python
def make_node(ctx: RunContext, log: RunLogger, llm: LLM) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:   # return ONLY the keys you change
        ...
    return node
```
- Pure function of state + files on disk; write artifacts with `ctx.write_json("<name>.json", obj)`.
- Seeds: `stage_seed(ctx.seed, "<agent>", item, state.get("loop", 0))`.
- LLM: `llm.structured(agent, prompt_id, OutSchema, creative=bool, seed=..., **vars)`. Prompt files
  `prompts/<prompt_id>.md` with `## system` / `## user` and `{{var}}` placeholders. Start every id at `_v1`.
- Every non-LLM tool call that matters (HTTP fetch, search, ffmpeg, OCR, model) is logged with
  `with log.call(agent, tool, ...) as rec:` and sets `rec["units"]`, `rec["output_path"]`.
- Must work with `ctx.mock_llm=True` (LLM returns schema-valid fakes or a fixture you register in
  `llm.fixtures[prompt_id]`) and `ctx.mock_media=True` (workers return placeholders) and `ctx.offline=True`.
- Read critic feedback for your agent from `state.get("feedback", {}).get("<agent>", [])` and include it in the prompt.

## Workers (subprocess, see core/workers.py)
`run_worker(ctx, log, agent=..., worker="image_sdxl.py", request={...}, gpu="1")` → `outputs` dict.
Each worker file ends with `if __name__ == "__main__": worker_main(handle)`; `handle(req)` must honour
`req.get("mock")` (no model load, deterministic placeholder output). Pass `"mock": ctx.mock_media`.

| Worker | task(s) | request | outputs |
|---|---|---|---|
| `retrieval_models.py` | embed / rerank / nli | `texts, kind, out_path?` / `query, passages` or batch `items: [{query, passages}]` / `pairs` | `npy_path` / `scores` / `scores` |
| `image_sdxl.py` | generate | `prompt, negative, width, height, seed, n, out_prefix` | `paths` |
| `image_checks.py` | check | `paths` | `results: [{path, face_count, face_scores, nsfw_score}]` |
| `upscale.py` | upscale | `path, width, height` | `path` |
| `tts_kokoro.py` | synth | `sentences: [spoken], voice, speed, seed, lexicon` | `wav_path, sentence_spans, tokens: [{text,start,end,sentence_index}]` |
| `asr_whisper.py` | transcribe | `wav_path, initial_prompt` | `text, words: [{word,start,end}]` |
| `render_hyperframes.mjs` | render | `scene_path, out_path` | `video_path, frames` |
| `render_ffmpeg.py` | render | `scene_path, out_path` | `video_path` |

## LLM servers
`core/servers.py` (platform owner): `ensure_llm(ctx)`, `start_vlm(ctx)`, `stop_vlm(ctx)`, `health(url)`.
No-ops when `ctx.mock_llm`. Servers bind `127.0.0.1` only.

## State keys written by whom
| Key | Writer |
|---|---|
| plan | planner |
| sources, contacts | researcher |
| facts, coverage | fact_curator |
| script, hook_ab | scriptwriter (script draft with 3 hooks in `extra.script_draft`), hook_judge (final `script`, `hook_ab`) |
| storyboard | storyboarder |
| shot_assets, voice, captions, srt_path | asset_generator (image_shot / voice / assets_join) |
| scene_path, video_path | editor |
| qa_report, route, feedback, redo_shots, history, loop, extra.redo_voice | critic |
| review, route | human_review |
| final | export |

Critic feedback items (`state["feedback"][agent]` is a list) may be plain strings or dicts:
`"[check_id] value/threshold/evidence -> fix"`, `{"shot_id": 3, "instruction": "..."}` (asset_generator),
`"[faces:3]"` / `"shot 3: ..."` (asset_generator), `{"lexicon": {"Somaiya": "/soʊˈmaɪjə/"}}` (voice patch).
`run_worker` returns the worker's `outputs` plus `outputs["meta"] = {model, model_revision, licence}`.
Final SRT: `outputs/<run_id>/captions.srt` -> exported as `<TeamID>_<BriefID>_final.srt`.

Critic routing values: `researcher | scriptwriter | storyboarder | asset_generator | editor | human_review | export`.
`loop` is incremented by the critic when it routes back. After `MAX_LOOPS` (3) it routes to `human_review`
(A/C or any remaining hard failure) or `export`.

## Rule packs (`config/rules/<pack>.yaml`, owner: creative)
```yaml
pack: C
description: Responsible AI
mandatory_review: true
style_preset: flat_vector           # key in config/styles.yaml
rules:
  - id: C.policy_quoted
    text: Policy statements only as quoted in sources; never imply a policy exists if none was found.
    applies_to: script               # script | storyboard | frames | contacts
    check: llm                       # llm (critic soft check) | code (critic hard check id)
    responsible_agent: scriptwriter
```
Packs: `base` (always), `A`..`E`, `P` (practice). Planner selects `["base", <category>]`.

## Tests
`tests/` with pytest; run `.venv\Scripts\python.exe -m pytest -q`. Each owner adds tests for their
modules that run offline in under a minute (mock modes). Don't write to `logs/` in tests: build a
`RunContext(run_id="test-...")` and a `RunLogger` against a tmp dir by monkeypatching `ROOT` or
`ctx.logs_dir` (see `tests/conftest.py`).

## Licences
Every generated/used asset gets an `AssetRecord` row via `core/manifest.py` (`add_asset(ctx, record)`).
Allowed licences: `config/licences_allowlist.yaml`. GSAP: "Standard No-Charge License". Fonts: OFL.
