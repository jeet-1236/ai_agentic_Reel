# AgentReel

A multi-agent pipeline that turns a one-card brief into a captioned, voiced 30-60 s short video (9:16 by
default, 16:9 supported). It finds and cites official sources itself. Everything runs locally on a Kaggle
notebook with 2x T4 GPUs, so the actual cost is **Rs 0**. No step is specific to one topic: the Live Run brief
is drawn on the day.

> **TEAM_ID and the official logo are still pending.** `config/settings.yaml` has `team_id: TXX` and a
> placeholder logo path. Final files are named `<TEAM_ID>_<BriefID>_final.mp4`, so set the real values before
> the Live Run.

## Autonomy level

| Brief category | Level | What a human does |
|---|---|---|
| B Admissions, D Fest, E Image | **L4** | Nothing. One command from brief to export. The critic loop fixes failures (at most 3 loops) |
| A Safety, C Responsible AI | **L3** | One **mandatory review** before export (LangGraph `interrupt()`): approve, or reject with a note and a target agent. This review is required by the rules and is logged as `review_checkpoint`; it is **not** counted as an intervention |

Any other human change goes through `RunLogger.intervention(stage, what, why)` and is written to
`logs/interventions.md` plus the hash-chained log. The limit is 5 and our target is 0.

## Architecture

![Agent graph](docs/agent_graph.png)

(`docs/agent_graph.mmd` is generated from the compiled graph by `scripts/make_agent_graph.py`, offline.)

```
brief.yaml -> Planner -> Researcher -> FactCurator -> Scriptwriter (3 hooks) -> HookJudge -> Storyboarder
  -> AssetGenerator (Send fan-out: one image job per shot + voice) -> Editor (HyperFrames | ffmpeg)
  -> Critic (hard checks -> soft checks) --fail, <=3 loops--> responsible agent
  -> HumanReview (A/C only) -> Export
```

- **LangGraph 1.2.16** state machine. SqliteSaver checkpoints, and a SqliteCache whose key includes the prompt
  files, `models.yaml` and the seed. This is what makes `--only <stage>` exact.
- **Pydantic contracts** for every LLM output (`core/schemas.py`). Decoding is JSON-schema constrained through
  llama.cpp, with one repair retry.
- **Model routing:** Qwen3.5-9B (text, GPU0, always loaded), Qwen3-VL-8B (frame QA, GPU1, on demand), and
  small specialist models (embedding, reranker, HHEM NLI, YuNet, NSFW) as subprocess workers on GPU1.
- **Grounding:** live Tavily search restricted to an allowlist of official domains, with a sitemap catalogue
  and `ddgs` as fallbacks. Every fetched page is snapshotted and hashed (`--offline` replays the snapshots).
  A fact is kept only if its quote is a verbatim substring of the source and HHEM scores it at least 0.5.
  Contacts come only from a whitelist built from the sources, ordered by source precedence.
- **Logging:** one JSONL line per model, tool, worker or ffmpeg call, hash-chained (`scripts/verify_logs.py`).
  `scripts/make_cost_report.py` builds `logs/cost_report.csv` from the log.

The full design is in `PLAN.md`, and the build conventions are in `docs/CONVENTIONS.md`.

## One-command run

```bash
python run.py --brief briefs/C01.yaml --seed 42
python run.py --brief briefs/C01.yaml --seed 42 --only scriptwriter     # re-run one stage from cache
python run.py --brief briefs/C01.yaml --seed 42 --offline              # replay source snapshots, no network
python run.py --resume C01-s42 --approve                                # A/C review
python run.py --resume C01-s42 --reject "hook overclaims" --target scriptwriter
python run.py --brief briefs/P00.yaml --mock-llm --mock-media           # laptop smoke run, no GPU
```

Afterwards:

```bash
python scripts/verify_logs.py            # hash chain + interventions per run
python scripts/make_cost_report.py       # logs/cost_report.csv (actual Rs 0 + labelled shadow cost)
python scripts/make_agent_graph.py       # docs/agent_graph.mmd/.png
python scripts/make_process_report.py    # docs/process_report.md/.pdf
```

## Kaggle setup

No datasets are needed: with Internet on, setup pulls everything from its official source at pinned versions.

| What | Source (pinned) |
|---|---|
| This repo | `git clone` of your GitHub repo (`REPO_URL` in notebook cell 1) |
| LLM / VLM | `unsloth/Qwen3.5-9B-GGUF` (Q8_0), `Qwen/Qwen3-VL-8B-Instruct-GGUF` (Q4_K_M + mmproj F16) |
| Images | `stabilityai/stable-diffusion-xl-base-1.0` (fp16 parts), `ByteDance/SDXL-Lightning` (4-step UNet), `madebyollin/sdxl-vae-fp16-fix`, `ai-forever/Real-ESRGAN` (x2) |
| Voice | `hexgrad/Kokoro-82M`, `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |
| Retrieval / QA | `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B`, `vectara/hallucination_evaluation_model` (+ `google/flan-t5-base` tokenizer), `Falconsai/nsfw_image_detection`, `opencv/face_detection_yunet` |
| llama.cpp | GitHub release `b11796`: `llama-…-bin-ubuntu-cuda-12.8-x64` + `cudart-llama-…` (CUDA 12.8 runtime) |
| Node / renderer | nodejs.org `v22.23.4`; `npm ci` (HyperFrames 0.8.183, GSAP 3.13.0); Chrome headless shell via `hyperframes browser ensure` |

Every Hugging Face download uses the commit SHA in `config/models.yaml` (verified 2026-10-21); file sha256s go
to `config/models.lock.yaml`. Total ≈ 28 GB, written to the scratch disk `/tmp/agentreel` because
`/kaggle/working` is capped at 20 GB.

1. The repo lives at https://github.com/jeet-1236/agentic_reel (private is fine: add a `GITHUB_TOKEN` Kaggle Secret); its URL is already in
   notebook cell 1 (`REPO_URL_DEFAULT`).
2. Notebook settings: **GPU T4 x2**, **Internet on**, **Persistence: Files**, Kaggle Secrets `TAVILY_API_KEY`
   (optional `HF_TOKEN`, `GITHUB_TOKEN`).
3. Open `notebooks/kaggle_live.ipynb` (File → Import Notebook) and run cells 1-4 **before** the brief draw.
   Cell 4 runs `bash scripts/setup_kaggle.sh` (idempotent; ~25-35 min cold, <1 min warm): pip packages, the
   Python 3.12 Kokoro venv, models from Hugging Face, Node 22 + Chrome + its system libraries, ffmpeg checks,
   llama.cpp, `llama-server` on GPU0, and `scripts/smoke_test.py --real`. It must end with **ALL GREEN**.
4. Live Run: fill the brief in cell 5, run cell 6 (output streams, so the session stays active), use the
   review panel in cell 7 for A/C, and export with cell 8.

Optional, to skip downloads on later sessions: the `agentreel-models` / `agentreel-bin` / `agentreel-wheels`
datasets from `scripts/make_kaggle_datasets.md` are used automatically when attached.

Fallbacks are built in. If Chrome can't start, use `--renderer ffmpeg`. If the Kokoro venv fails, use
`kokoro-onnx` on 3.13. If Tavily is unavailable, research falls back to the sitemap catalogue, then `ddgs`.

## Repository layout

```
run.py                 one-command entry point
agents/                planner researcher fact_curator scriptwriter hook_judge storyboarder
                       asset_generator editor critic human_review export
core/                  graph state schemas llm logging prompts seeds workers servers (+ qa_checks, sources, ...)
workers/               GPU/media subprocess workers (JSON in -> files out)
renderer/              HyperFrames templates + render driver (Node 22)
prompts/               versioned prompt templates <agent>_v<N>.md
config/                models.yaml (+ models.lock.yaml), rules/, allowlists, styles, lexicon, settings
briefs/                brief YAMLs (_template.yaml for Live Run day)
scripts/               setup_kaggle.sh, build_llamacpp_sm75.sh, prefetch_models.py, make_kaggle_datasets.md,
                       smoke_test.py, verify_logs.py, make_cost_report.py, make_agent_graph.py, make_process_report.py
notebooks/             kaggle_live.ipynb (setup, run, review panel, export, viva)
data/                  sitemaps/, sources/ (snapshots), sources_manifest.csv
outputs/<run_id>/      plan.json facts.json hook_ab.json storyboard.json scene.json qa_report.json review/ final files
logs/                  run_log.jsonl (hash-chained), cost_report.csv, interventions.md
docs/                  agent_graph.png, process_report.pdf, viva_notes.md, CONVENTIONS.md
tests/                 pytest, offline, mock modes
```

## Models and licences

| Role | Model / tool | Licence |
|---|---|---|
| Orchestration | LangGraph 1.2.16 | MIT |
| LLM serving | llama.cpp `llama-server` (tag b11793, built for sm_75) | MIT |
| Text LLM | Qwen3.5-9B Q8_0 GGUF (fallback Qwen3-14B Q4_K_M) | Apache-2.0 |
| VLM (frame QA) | Qwen3-VL-8B-Instruct Q4_K_M + mmproj | Apache-2.0 |
| Search | Tavily API (free tier); fallback `ddgs` | API terms / MIT |
| Extraction | trafilatura 2.3, pypdf, pypdfium2, Tesseract / RapidOCR | Apache-2.0 / BSD / Apache-2.0 |
| Retrieval | bm25s, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B (fallback bge-m3 / bge-reranker-v2-m3) | MIT / Apache-2.0 |
| Claim check | HHEM-2.1-Open (+ flan-t5-base tokenizer) | Apache-2.0 |
| Images | SDXL 1.0 base + SDXL-Lightning 4-step + sdxl-vae-fp16-fix | CreativeML OpenRAIL++-M / MIT |
| Upscale | Real-ESRGAN x2 via spandrel | BSD-3-Clause / MIT |
| TTS | Kokoro-82M (fallback kokoro-onnx) | Apache-2.0 |
| ASR / alignment | faster-whisper large-v3-turbo | MIT |
| Render | HyperFrames 0.8.183 (Node 22, headless Chrome) + GSAP 3.13; fallback ffmpeg + Pillow | Apache-2.0 / GSAP Standard No-Charge License |
| Audio / video | ffmpeg (2-pass loudnorm, libx264, libass) | LGPL/GPL |
| QA | YuNet face detector, Falconsai NSFW classifier, RapidOCR | MIT / Apache-2.0 / Apache-2.0 |
| Fonts | Fira Sans, Noto Sans | SIL OFL 1.1 |

Rejected on licence: XTTS-v2, F5-TTS, IndexTTS-2, DMD2, SDXL-Turbo, FLUX.1-dev, FLUX.2-klein-9B, Sana,
InsightFace models, MMS aligner, Bespoke-MiniCheck-7B, LTX-Video, SD3.5, Remotion. Every generated or used
asset gets a row in `outputs/<run>/assets/manifest.csv` with its generator, revision, seed, licence and
sha256. The Critic blocks any licence that is not in `config/licences_allowlist.yaml`.

## Credits

- **LangGraph** (LangChain, MIT): graph runtime, checkpoints, cache, interrupts.
- **llama.cpp** (ggml-org, MIT): local OpenAI-compatible serving with grammar-constrained JSON.
- **HyperFrames** (HeyGen, Apache-2.0): deterministic HTML-to-MP4 rendering.
- **GSAP** (GreenSock, Standard No-Charge License): template animation.
- **html-video** (nexu-io, Apache-2.0): we reuse the *idea* of a template manifest (each template declares its
  slot schema). No code is copied.
- **Kokoro-82M** (hexgrad, Apache-2.0), **SDXL-Lightning** (ByteDance, OpenRAIL++), **faster-whisper**
  (SYSTRAN, MIT), **trafilatura** (Apache-2.0), **bm25s** (MIT), **HHEM-2.1-Open** (Vectara, Apache-2.0),
  **YuNet** (OpenCV Zoo, MIT), **RapidOCR** (RapidAI, Apache-2.0), and the Qwen team (Apache-2.0).

**Borrowed-code rule:** any snippet adapted from another project gets a comment at the point of use
(`# Adapted from <url> (<licence>)`) and an entry in this section. We copy only from licences on the allowlist.

## Results

*Placeholder: fill this in after the dry runs and the Live Run.*

| Run | Brief | Duration | Loops | Interventions | Wall-clock | QA passed |
|---|---|---|---|---|---|---|
| | | | | | | |

## Tests

```bash
.venv\Scripts\python.exe -m pytest -q          # Windows dev
python -m pytest -q                            # Kaggle / Linux
```
