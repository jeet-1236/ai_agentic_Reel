# AgentReel — Implementation Plan (Kaggle 2×T4, fully local, brief-agnostic)

> Version 1.0 · 2026-10-21 · For: the 2-person team building the Somaiya AgentReel Challenge entry.
> Based on the handout, the team's answers, and web research done today (sources in §18). Items marked **VERIFY** are things the research could not confirm. Each one has a Day-1 smoke test.

---

## 0. Decisions

| # | Topic | Decision |
|---|---|---|
| D1 | Practice topic | **C01**, College AI policy explainer (allowed vs not allowed in assignments & exams). Audience: all students |
| D2 | Demo day | **A new, unknown topic is drawn**, so nothing in the pipeline may be topic-specific except data. Must be dynamic and reproducible |
| D3 | Format | 9:16 1080×1920 primary. 16:9 1920×1080 supported as a parameter. No digital-experience track |
| D4 | Language / audio | English VO + English burned-in captions + `.srt`. **Voice only, no music** |
| D5 | Sources | None provided. The pipeline **discovers, snapshots and cites** official Somaiya + Indian government pages at run time |
| D6 | C01 content | **Reframed from what exists.** No public Somaiya AI policy exists (§11). The video never claims one |
| D7 | Search | **Tavily** (team key, Kaggle Secret) restricted to an allowlist. Fallbacks: Somaiya sitemap catalogue (free), then `ddgs` |
| D8 | Framework | **LangGraph 1.2.x** |
| D9 | Runtime | Kaggle notebook, 2× T4 (sm_75: FP16 only, no BF16, no FlashAttention-2), 4 CPU, 29 GB RAM, Internet ON, **₹0** |
| D10 | Renderer | **HyperFrames** (HeyGen, Apache-2.0, HTML+GSAP → MP4, deterministic) with an **ffmpeg + Pillow fallback** reading the same scene JSON. Remotion rejected (licence, §1.4) |
| D11 | Voice | **Kokoro-82M**, US/UK English voice (Apache-2.0) |
| D12 | P4 creativity | **Auto A/B hooks**: 3 hooks, judged, winner used, all logged |
| D13 | Autonomy | **L4** for B/D/E briefs; **L3** for A/C (one mandatory review, not an intervention). Target: **0 interventions** |
| D14 | Timeline | No dates given: relative days (§15). Tag `v1.0-build` ≥ 24 h before the Live Run |

## 1. What the research changed (summary)

### 1.1 LLM serving
- **vLLM runs on T4 but is the wrong primary.**
  - On sm_75 it falls back to the TRITON_ATTN backend, which has slow prefill (Triton has no tensor cores below sm_80).
  - Gemma 4 crashes on it; gpt-oss-20b doesn't run.
  - Its batch-invariant determinism needs sm_80+.
- **Primary: llama.cpp `llama-server`**, pinned build:
  - OpenAI-compatible API, with JSON-schema→grammar constrained output
  - per-request `seed`
  - `usage` returned with every response
  - measured on T4: Qwen2.5-14B Q4 13 tok/s; Llama-2-7B Q4 46 tok/s; Qwen3-30B-A3B 22–43 tok/s
- **Model: Qwen3.5-9B** (Apache-2.0, IFEval 91.5), Q8_0 GGUF (9.5 GB) on GPU0. Thinking OFF for JSON/critic steps.

### 1.2 Images
- **SDXL 1.0 + SDXL-Lightning 4-step** (openrail++) with `madebyollin/sdxl-vae-fp16-fix` (MIT), about 8–10 s per image on T4.
- **Rejected:**
  - DMD2, SDXL-Turbo: non-commercial
  - Sana: research-only clause
  - Z-Image-Turbo: black images in fp16 on T4
  - FLUX.2-klein-9B: non-commercial
- **Optional upgrade:** FLUX.2 [klein] 4B (Apache-2.0). Untimed on T4 → Day-1 test.
- **Image-to-video is not viable in 3 h on T4** (extrapolated 45+ min/clip). Motion comes from Ken Burns/parallax + GSAP animation.

### 1.3 Voice
- **Kokoro** is Apache-2.0, has native per-token timestamps, and round-trip WER is about 3–5%.
- **Traps:**
  - The PyPI `kokoro` package requires Python < 3.13, and Kaggle is now on 3.13.
  - Output is non-deterministic unless `torch.manual_seed` is called before every chunk.
  - "1930" is read as a year.
  - "₹" is not handled.
  - ALL-CAPS words are spelled out.

### 1.4 Renderer
- **Remotion** is "source-available, not open source":
  - Free for individuals.
  - The upcoming v5 terms say educational institutions are not "non-profit" for the free licence, so a paid licence would be needed if the university runs it.
- **HyperFrames** has the same headless-Chrome approach under Apache-2.0, with no thresholds:
  - "same input, same frames, same output"
  - seek-based frame capture
  - built for agents
- The team's link `nexu-io/html-video` is a studio app on top of HyperFrames that drives external coding-agent CLIs. We reuse its *template-manifest idea* (each template declares a JSON schema of slots) but not the app.

### 1.5 Sources
- **No public Somaiya/KJSCE AI policy.**
- **Somaiya pages:**
  - Server-rendered, so plain `requests` + trafilatura works.
  - Sitemaps exist (193 + 366 URLs).
  - Somaiya's anti-ragging page contains **typos in the national helpline number and email**, so contacts need a source-precedence rule (§6.2).
- **Government sites:** several have expired/invalid TLS or are JS-only shells. PIB press releases are a static alternative.

### 1.6 Kaggle
- **Platform changes:**
  - Image moved to **Python 3.13** (28 Sep 2026).
  - P100 retired 15 Sep 2026; only T4×2 remains.
  - Some users report the T4 option greyed out since then → **test availability now**.
- **Limits:**
  - 12 h sessions
  - ~30 GPU-h/week
  - 20 GB persisted `/kaggle/working`
  - **interactive idle timeout 20 min** (docs conflict: 20 vs 60)
  - batch "commit" runs **cannot take input** (`input()` raises)
- **Missing from the image:** Node.js and some Chrome libraries.
- **Weights:** private Kaggle Datasets/Models up to 200 GB, read at 135–770 MB/s.

### 1.7 LangGraph
- Built-in node caching **ignores prompt edits** (its key covers only the node input), so the cache key must include the prompt version.
- `interrupt()` re-runs the node from its first line on resume.
- `draw_mermaid_png()` needs internet by default.
- Pins: `langgraph==1.2.16`, `langchain-core==1.6.12`, `langchain-openai==1.7.2`.

## 2. How the design maps to the rubric

| Rubric | Marks | Mechanism |
|---|---|---|
| O1 Clarity & actionability | 15 | A/B hooks judged on "topic clear in 3 s"; CTA and contact only from sources; contact end card for A/C |
| O2 Emotional impact | 10 | Tone presets per category; voice/pace presets; GSAP motion; hook variety (question / surprising fact / direct address) |
| O3 Narrative & quality | 10 | Word-timestamp captions; scene templates; spec gate; WER check on the VO |
| O4 Accuracy | 5 | Quote-verified facts, contact whitelist with source precedence, NLI claim check, mandatory review for A/C |
| P1 Autonomy | 20 | One command; critic routes to the responsible agent (max 3 loops); interventions logged; target 0 |
| P2 Workflow design | 15 | Explicit LangGraph graph, Pydantic schemas everywhere, live RAG grounding, model routing (9B text / 8B VLM / small NLI) |
| P3 Reproducibility | 15 | Pinned everything, seeds per stage, source snapshots, content-addressed cache, `--only <stage>` replay, hash-chained logs |
| P4 Creativity | 10 | Auto A/B hooks + live official-source discovery with snapshot replay + spoken/display text split for TTS |

Penalties designed out:

| Penalty | Mechanism |
|---|---|
| Fabricated fact | A claim without a verified quote is dropped |
| Wrong helpline | Exact-match whitelist + source precedence |
| Over/under length | Duration gate before render |
| Unlicensed asset | Manifest gate with licence allowlist |
| Real face | Illustration style + YuNet detector + VLM check |
| Edited logs | Hash chain |

## 3. Architecture

```
brief.yaml ─► Planner ─► Researcher ─► FactCurator ─► Scriptwriter ─► HookJudge ─► Storyboarder
                                                                                    │
            ┌───────────────────────────────────────────────────────────────────────┘
            ▼
      AssetGenerator ──(Send fan-out per shot)──► ImageWorker ×N
            │  ├─ Voice: lexicon → Kokoro → timestamps → ASR/WER check
            │  └─ Captions: chunk → .srt + caption track
            ▼
          Editor (scene JSON → HyperFrames render → ffmpeg loudnorm/encode)
            ▼
          Critic (hard checks → soft checks) ──fail (≤3 loops)──► responsible agent
            │ pass
            ▼
      HumanReview (A/C only, interrupt) ──reject+note──► responsible agent
            ▼
          Export (TeamID_BriefID_final.mp4, .srt, qa_report.json, logs, cost_report.csv)
```

**Critic routing:**

| Failure class | Routed to |
|---|---|
| Claim unsupported / quote mismatch / contact not in whitelist / must_include uncovered | Scriptwriter (offending sentence + allowed facts); Researcher if evidence is missing entirely (one extra search round) |
| Duration / word budget / hook-in-3 s / CTA missing / tone or safe-messaging violation | Scriptwriter |
| Too many words on screen / shot timing / text outside safe zone | Storyboarder |
| Face detected / garbled text / off-brief or fear imagery | AssetGenerator (that shot only; seed = f(seed, shot, loop)) |
| WER > 8 % / mispronunciation | AssetGenerator (voice), with a lexicon patch proposed by the LLM |
| Spec (codec/fps/size/loudness), render error | Editor |

State keeps `loop_count`, `history[]` (failure id + artifact hash). If a fix reproduces an earlier failing artifact hash, the loop stops. After 3 loops the best version that passes every hard check ships, or (A/C) goes to review flagged.

## 4. Kaggle runtime

### 4.1 Where things run

| Device | Resident | Notes |
|---|---|---|
| GPU0 (T4 15 GB) | `llama-server` Qwen3.5-9B Q8_0 (~9.5 GB + KV for 16k ctx) | Up the whole run, bound to **127.0.0.1** (Kaggle AUP: no public servers) |
| GPU1 (T4 15 GB) | Loaded in turn: embeddings+reranker → SDXL-Lightning → Kokoro → faster-whisper turbo → Qwen3-VL-8B (2nd llama-server) | Each stage frees VRAM before the next (`del`, `torch.cuda.empty_cache()`, kill subprocess **process group**) |
| CPU (4 cores) | LangGraph, BM25, trafilatura, HyperFrames/Chrome render, ffmpeg, OCR, YuNet, HHEM NLI | Rendering is the main CPU load |

Media models run as **subprocess workers** (`workers/*.py`, JSON in → files out) with `CUDA_VISIBLE_DEVICES=1`. This isolates dependency conflicts and guarantees VRAM is freed.

### 4.2 Python environments (Kaggle is Python 3.13)
- **System env (3.13, preinstalled torch):** orchestrator, LangGraph, diffusers/SDXL, faster-whisper, sentence-transformers, RapidOCR, OpenCV.
- **`uv venv --python 3.12` "voice" env:** `kokoro` + `misaki` + its own torch wheel, used only by `workers/tts.py`.
  - Fallback: `kokoro-onnx` 0.6.1 on 3.13, with timings from faster-whisper word timestamps.
- Every install is pinned. Wheels are pre-downloaded into a private Kaggle Dataset `agentreel-wheels`, so the Live Run does no PyPI resolution.

### 4.3 Pre-staged Kaggle Datasets/Models (built in the Build phase, versioned)

| Dataset | Contents |
|---|---|
| `agentreel-models` | Qwen3.5-9B Q8_0 GGUF, Qwen3-VL-8B Q4_K_M + mmproj, SDXL base fp16 + Lightning 4-step LoRA/UNet + fp16-fix VAE, Kokoro-82M + voices, faster-whisper large-v3-turbo, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B (or bge-reranker-v2-m3), HHEM-2.1-Open, YuNet onnx, Real-ESRGAN x2 |
| `agentreel-bin` | llama.cpp built **for sm_75** (`-DCMAKE_CUDA_ARCHITECTURES=75`; prebuilt tarballs only carry PTX and JIT-compile on first run), static ffmpeg (BtbN gpl, libass+x264) as a fallback, Node 22 tarball, `node_modules` + Chrome Headless Shell for HyperFrames |
| `agentreel-wheels` | All pip wheels incl. the 3.12 voice venv |
| `agentreel-fonts` | Fira Sans (Somaiya brand typeface, OFL) + Noto Sans (OFL) |

Paths are resolved with `kagglehub`/`os.walk`, never hard-coded (mount paths changed in Oct 2026). `config/models.yaml` records every model id + HF revision SHA + sha256 of the file.

### 4.4 Setup and run mode
- `scripts/setup_kaggle.sh` (≈10 min, run **before** the brief draw):
  1. Link the binaries.
  2. Install wheels.
  3. Check for missing Chrome libraries and install them with `apt-get` (VERIFY that runtime apt works; fallback: `.deb` files in `agentreel-bin`).
  4. Start `llama-server`.
  5. Run a smoke test: one LLM call, one image, one TTS line, a 2 s render, `ffmpeg -filters | grep -E "loudnorm|ass"`.
- **Live Run = interactive session** (commit runs can't take review input). Keep a cell active: the run streams progress, so the 20-min idle timer never trips.
- Each student uses **their own Kaggle account** (Terms forbid account sharing), giving two GPU quotas. The notebook is shared as a collaborator.
- At start-up the run logs `/etc/build_date`, `pip freeze`, `nvidia-smi` and llama.cpp `--version` into `run_manifest.json`. **"Pin to original environment"** is enabled on the notebook.

## 5. Tech stack

| Layer | Primary | Licence | Fallback |
|---|---|---|---|
| Orchestration | LangGraph 1.2.16, SqliteSaver checkpoints, SqliteCache | MIT | — |
| LLM serving | llama.cpp `llama-server` (pinned build, sm_75) | MIT | Ollama `qwen3.5:9b` (format=schema, seed); vLLM 0.30 + Qwen3-8B-AWQ (`--dtype float16 --enforce-eager --attention-backend TRITON_ATTN`) |
| Text LLM | Qwen3.5-9B Q8_0 | Apache-2.0 | Qwen3-14B Q4_K_M |
| VLM (frame QA) | Qwen3-VL-8B-Instruct Q4_K_M via llama-server `--mmproj` | Apache-2.0 | Qwen3.5-9B (natively multimodal) |
| Search | Tavily `basic`, `include_domains` restrict | API (free 1,000 credits/mo) | Sitemap catalogue → `ddgs` (MIT) with `site:` + host post-filter |
| Extraction | trafilatura 2.3 (HTML), pypdf/pypdfium2 (PDF), Tesseract/RapidOCR (scans) | Apache / BSD / Apache | — (no PyMuPDF: AGPL) |
| Retrieval | bm25s + Qwen3-Embedding-0.6B + FAISS, RRF fusion, Qwen3-Reranker-0.6B | MIT / Apache | bge-m3 + bge-reranker-v2-m3 |
| Claim check | HHEM-2.1-Open (NLI) + exact quote match + regex whitelist | Apache-2.0 | DeBERTa-v3-large NLI (MIT) |
| Images | SDXL 1.0 + SDXL-Lightning 4-step + fp16-fix VAE (+ optional style LoRA, licence-checked) | openrail++ / MIT | FLUX.2 [klein] 4B (Apache) if the Day-1 test passes |
| Upscale | Real-ESRGAN x2 via spandrel | BSD-3 / MIT | Lanczos |
| TTS | Kokoro-82M (`af_heart`/`bf_emma`; pick in dry runs) | Apache-2.0 | kokoro-onnx |
| ASR / alignment | faster-whisper large-v3-turbo (`word_timestamps`) | MIT | Kokoro native timings |
| Render | HyperFrames (Node 22, Puppeteer Chrome, ffmpeg) + GSAP | Apache-2.0 (GSAP: free standard licence, record it) | ffmpeg + Pillow scene renderer |
| Audio post | ffmpeg 2-pass `loudnorm` | LGPL/GPL | pyloudnorm measure |
| QA | ffprobe/ebur128/vfrdet, RapidOCR, YuNet face detector, Falconsai NSFW | — / Apache / MIT / Apache | Tesseract |
| Graph image | `graph.get_graph().draw_mermaid()` → mermaid-cli via the Node already installed (offline) | MIT | pygraphviz `draw_png` |

**Rejected on licence:** XTTS-v2, F5-TTS, IndexTTS-2, DMD2, SDXL-Turbo, FLUX.1-dev, FLUX.2-klein-9B, Sana, InsightFace models, MMS aligner, Bespoke-MiniCheck-7B, LTX-Video, SD3.5 (revenue threshold), Remotion (v5 terms for institutions).

## 6. Agents

Every LLM agent uses a Pydantic schema → `ChatOpenAI(...).with_structured_output(Schema, method="json_schema", include_raw=True)`.
- **Keep schemas flat.** llama.cpp's grammar converter silently drops unsupported keywords and breaks on nested `$ref`, and on `anyOf` mixed with `properties`.
- On a validation failure there is one repair retry with the error message, then the failure is logged.
- Prompts live in `prompts/<agent>_v<N>.md`; the `prompt_id` + sha256 go into every log line.

### 6.1 Planner
- **In:** `brief.yaml` (validated by schema).
- **Out:** `plan.json`.
- **Code-only parts:**
  - category → rule pack (§7)
  - word budget = `duration_s × 2.4` (±10 %)
  - shot count = clamp(round(duration/5), 6, 12)
  - per-stage time budget (§14)
  - retry caps
- **LLM parts:** search queries (8–12), audience insight, success criteria from `must_include`/`must_avoid`/`cta`.

### 6.2 Researcher (live official-source discovery)
1. **Candidate URLs:**
   - (a) Tavily `search(query, include_domains=ALLOWLIST, include_domains_mode="restrict", search_depth="basic", max_results=10)` for each query
   - (b) BM25 match of the query against the cached **sitemap catalogue** of somaiya.edu + kjsce.somaiya.edu (193 + 366 URLs, free)
   - (c) `ddgs` fallback

   Every host is re-checked against the allowlist in code. Off-allowlist links (e.g. Google Drive PDFs linked from an official CoE page) are allowed only **one hop from an official referrer**, and the referrer is logged.
2. **Fetch + snapshot:**
   - `requests` fetches with TLS verification. The retry with `verify=False` is allowed for allowlisted hosts only, and is recorded as `tls_verified=false`.
   - Raw bytes go to `data/sources/<sha256>.<ext>`.
   - A row is appended to `sources_manifest.csv` (url, final_url, referrer, fetched_at, sha256, status, content_type, tls_verified, title).
3. **Extract:** trafilatura markdown (with metadata) and pypdf. If ~0 chars come out, OCR (Tesseract, preinstalled). JS-shell pages are skipped and logged; facts for those come from PIB/static pages.
4. **Index:** heading-aware chunks of ~400 tokens; ids `S<n>#c<k>`; bm25s + dense + RRF(k=60) + rerank top 20 → top 8 per query.
5. **Contacts:** regex (phones via `phonenumbers` region IN, emails, URLs) over **source text only** → `contacts_whitelist.json` with source ids. **Precedence:** the national helpline's own government page > a PIB page > a Somaiya page. Conflicts are recorded and surfaced in review. The anti-ragging page typo is the motivating case.
6. **Replay:** `--offline` replays the manifest so a re-run's inputs are byte-identical.

### 6.3 FactCurator
- The LLM proposes atomic facts, each with a **verbatim quote** + chunk id.
- **Code checks** that the normalised quote is a substring of the chunk. Facts that fail are dropped.
- HHEM scores fact vs chunk; anything below 0.5 is dropped.
- **Output:** `facts.json` {fact_id, claim, quote, source_id, url, page/section, score}.
- **Coverage map:** each `must_include` item → fact_ids or `MISSING`. A MISSING item is never invented. It is reported, softened to "check the official page", and shown at review for A/C.

### 6.4 Scriptwriter + HookJudge (P4: Auto A/B hooks)
- **Scriptwriter** writes 3 hooks (question / surprising sourced fact / direct address), one body, one CTA.
  - Every sentence carries `fact_ids` (empty allowed only for connective, non-factual lines; the Critic checks this).
  - It writes two text forms per sentence (P4): `display` for captions, and `spoken` for TTS (e.g. "1930" → "one nine three zero", "₹2,000" → "two thousand rupees", "KJSCE" → "K J S C E").
- **HookJudge** scores each hook 1–10 on: topic clear in ≤3 s, audience fit, tone fit, no unsourced claim.
  - Each hook is scored twice with candidate order swapped (removes position bias).
  - Ties go to the shortest hook.
  - All scores are logged (`hook_ab.json`).

### 6.5 Storyboarder
- 6–12 shots: `template`, `duration_s`, `visual_prompt`, `on_screen_text` (≤ 7 words), `caption_band`, `fact_ids`, `asset_source`.
- **Templates (fixed library, §8):** `hook_title`, `illustration_kenburns`, `do_dont_split`, `step_list`, `stat_callout`, `contact_card`, `quote_card`, `end_card`.
- Visual prompts use the category style preset: flat-vector/isometric, **no realistic people**. Sensitive topics use silhouettes or objects only.

### 6.6 AssetGenerator
- **Images** (one Send fan-out item per shot):
  - SDXL-Lightning at 768×1344 (or 1344×768 for 16:9), seed = H(seed, shot, loop).
  - 2 candidates; the VLM picks one.
  - Real-ESRGAN x2 → cover-crop with Ken Burns headroom.
  - YuNet face check + NSFW check.
- **Voice:**
  - Kokoro on the `spoken` text, `torch.manual_seed(seed+chunk)` before each chunk, misaki lexicon `config/lexicon.yaml` for names.
  - Per-sentence WAVs + native timings; sentence offsets accumulated in code.
  - faster-whisper transcript → WER vs `spoken` (> 8 % → regenerate with a lexicon patch).
- **Captions:**
  - Word timings mapped back onto `display` text.
  - Pages of ≤ 2 lines, ≤ 32 chars/line, ≤ 17 CPS, ≥ 0.83 s each.
  - `.srt` exported.
- **Manifest:** every file gets a row in `assets/manifest.csv` (path, kind, generator, model+revision, seed, prompt hash, licence, sha256, source URL for the logo).

### 6.7 Editor
- Builds `scene.json` (shots, assets, word timings, captions, template slots) and validates it against each template's slot schema.
- `workers/render_hyperframes.mjs` fills HTML templates → `npx hyperframes render` at 30 fps.
- **ffmpeg post-processing:**
  - Two-pass `loudnorm=I=-14:TP=-1.5:LRA=11:linear=true` with measured values, `-ar 48000 -ac 2`.
  - Encode: `-c:v libx264 -profile:v high -pix_fmt yuv420p -r 30 -fps_mode cfr -crf 20 -c:a aac -b:a 160k -movflags +faststart`.
- **End card (4 s):** official logo (unaltered), CTA from facts, "Created with AI agents — LY Agentic AI Lab".
- **Fallback renderer:** same `scene.json` → Pillow frames + ffmpeg zoompan + libass captions.

### 6.8 Critic
**Hard checks (code) run first** (§9). Then the **soft checks**:
- claim entailment (HHEM)
- rule-pack tone/safe-messaging/no-blame review by the LLM, with sentence-level reasons
- VLM review of 1 frame per shot + the end card (realistic face? garbled text? fear imagery? matches shot intent?)

**Output:** `qa_report.json` [{check_id, pass, value, threshold, evidence, responsible_agent}].

### 6.9 HumanReview (A/C mandatory)
- LangGraph `interrupt()` with SqliteSaver.
- The review packet is the preview MP4 + script with clickable source URLs + claims table + contact list with source precedence + coverage map. It is rendered in an **ipywidgets panel** in the notebook (no tunnel needed). A Gradio share link is optional.
- The response is validated with `interrupt(..., response_schema=Review)` → approve / reject + note + target agent.
- Logged as `review_checkpoint` (not counted as an intervention).
- Code before `interrupt()` must be side-effect free (it re-runs on resume).
- In batch mode (`KAGGLE_KERNEL_RUN_TYPE` ≠ interactive, VERIFY the values), A/C runs **stop at review** and never auto-approve.

## 7. Category rule packs (`config/rules/*.yaml`)

| Pack | Rules (each maps to a Critic check) |
|---|---|
| base | no claim without fact_id; contacts ∈ whitelist; disclosure line; logo untouched; no realistic people; licence allowlist; spec; hook ≤ 3 s; CTA from sources |
| A Safety | safe-messaging (no methods, hopeful tone, help card last); helpline end card; no victim depiction (objects/silhouettes); no blame of any group; laws described only as quoted; mandatory review |
| B Admissions | no numbers/dates/fees/quotas unless quoted from an official notice, with the notice URL on screen; "only official channels" framing |
| C Responsible AI | policy statements only as quoted; every "allowed / not allowed" item needs a fact_id; no implication that a policy exists if none was found; mandatory review |
| D Fest | safe-fest message (passes, no substances, help desk) if sources support it |
| E Image | no rankings/placement figures/awards unless quoted from an official page with the year |

**Reference contacts** (from today's research). The pipeline must still find each of these at run time; this list is for **testing the whitelist logic only**, never as an input:

| Contact | Number |
|---|---|
| Anti-ragging | 1800-180-5522, helpline@antiragging.in |
| Tele-MANAS | 14416 / 1-800-891-4416 |
| Cybercrime | 1930, cybercrime.gov.in |
| Women helpline | 181 |
| Emergency | 112 |
| NCW | 14490 |
| MANAS (narcotics) | 1933 |

## 8. Renderer: HyperFrames templates

- `renderer/templates/<name>/` = `index.html` + GSAP timeline + `template.yaml` (slot JSON schema, supported aspect ratios, min/max duration). This follows html-video's manifest convention.
- **Layout:** each template reads `--w/--h` and safe-zone variables, so 9:16 and 16:9 share code. Snapshot tests render each template in both aspect ratios.
- **Captions:**
  - Fira Sans Bold 64 px (≥ 48 px rule) with a 6 px dark outline.
  - The active word is highlighted in Somaiya red `#B7202E`.
  - Caption band y ∈ [1150, 1500] in 9:16. This sits inside the handout's band [250, 1570] and also clears Meta's stricter bottom-35 % zone.
- **Determinism:**
  - No `Math.random`; any randomness comes from a seeded PRNG passed in the slots.
  - Fonts are local.
  - Assets are referenced by relative path.
- **Brand:**
  - Colours from brand.somaiya.edu.in (Power Red #B7202E, Vitality Red #ED1C24, Cool Gray #58595B, Dark Gray #231F20).
  - The logo is the official SVG from the website, URL + sha256 in the manifest (§16 open item: which logo).

## 9. QA hard gate

| Check | Method | Threshold | On fail |
|---|---|---|---|
| Duration | ffprobe `format=duration` | 30–60 s, and brief target ± 2 s | Scriptwriter (pre-render word budget) / Editor |
| Video | ffprobe stream: codec, profile, w×h, pix_fmt | h264 **High**, 1080×1920 (or 1920×1080), yuv420p | Editor |
| CFR | `r_frame_rate == avg_frame_rate` and `vfrdet` VFR=0 | 30/1 | Editor |
| Audio | ffprobe | aac, 48000 Hz, 2 ch, 128–192 kb/s | Editor |
| Loudness | `ebur128=peak=true` | I = −14 ± 1 LUFS, TP ≤ −1 dBTP; `astats` no NaN | Editor |
| Size | file size | ≤ 100 MB | Editor |
| Dead air / black / frozen | `silencedetect=n=-50dB:d=1.5`, `blackdetect`, `freezedetect` | none outside planned shots | Editor / AssetGenerator |
| Caption layout | layout boxes from the renderer + RapidOCR on 2 fps | all text y ∈ [250, 1570]; height ≥ 48 px | Storyboarder |
| Caption readability | parse caption track | ≤ 2 lines, ≤ 32 chars/line, ≤ 17 CPS, ≥ 0.83 s | AssetGenerator (captions) |
| Contrast | WCAG ratio, caption vs frame | ≥ 4.5:1 | Editor |
| Hook | ASR words + on-screen text in 0–3 s | ≥ 1 topic keyword from the plan | Scriptwriter |
| End card | OCR last 4 s + `cv2.matchTemplate` logo | disclosure line (fuzzy ≥ 0.9), CTA, logo present | Editor |
| Facts | every factual sentence has fact_ids; quote substring check; HHEM ≥ 0.5 | 100 % | Scriptwriter / Researcher |
| Contacts | regex over script + OCR text | 100 % ∈ whitelist (precedence applied) | Scriptwriter |
| Faces | YuNet on images + frames → VLM "photoreal?" | 0 realistic faces | AssetGenerator |
| NSFW | Falconsai classifier | p < 0.2 | AssetGenerator |
| VO fidelity | WER(spoken, ASR) | ≤ 8 % | AssetGenerator (voice) |
| Licences | manifest licence ∈ allowlist | 100 % | AssetGenerator |

## 10. Logging and cost
- **`core/logging.py`:** a decorator plus a LangChain callback handler. One JSONL line per LLM/model/tool/ffmpeg call, with fields:

  | Group | Fields |
  |---|---|
  | Identity | ts, run_id, agent, tool, model, model_revision, prompt_id, prompt_sha256 |
  | Usage | input_tokens, output_tokens (from llama.cpp `usage`) or units (images, audio_s, frames) |
  | Cost and timing | cost_inr (0.0 local; Tavily 0.0 + `credits`), latency_s, gpu_id, gpu_seconds |
  | Outcome | seed, status, output_path, output_sha256, human_intervention |
  | Integrity | **prev_hash**, the sha256 of the previous line |
- **`scripts/verify_logs.py`** checks the hash chain. Logs are append-only; nobody edits them by hand.
- **`cost_report.csv`** is generated from the JSONL: stage, provider, model, tokens/units, cost_inr, wall_s, plus a "shadow cost" column (GPU-seconds × a stated public T4 hourly rate) for the report.
- **`log_intervention(stage, what, why)`** is the only way to write `interventions.md`; it also emits a JSONL line with `human_intervention:true`.

## 11. C01 content strategy (decision D6)

**Findings:**
- No public Somaiya/KJSCE AI or generative-AI policy was found. Checked: code of conduct, CoE, exam cell, notices, IQAC (titles login-gated).
- The scanned Student Code of Conduct PDF is still unread → OCR it in the build phase.

**Usable official facts:**
- **CoE Circular/COE/134/2022, "Reporting Unfair Means Cases":**
  - "The examination means any type of evaluation theory, practical, oral, seminar, presentation, on screen etc."
  - Includes "Referring to the website link, shared files/ material, downloaded files/ material, software tool… which is prohibited by the examiner during the on screen examination".
  - Also lists "Carrying any electronic gadgets like mobile phone, smart watch…".
- **CoE Circular/COE/133/2022 (06/01/2023):**
  - Cases go to the College Prevention of Unfair Means Committee (CPUMC), then the University committee (UPUMC).
  - They are handled "as per the sections 20, 20A and 47 of exam manual".
- **UGC 2018 Academic Integrity Regulations:** the penalty levels apply to Masters/research, so **do not present them as UG assignment rules**.

**Resulting angle:** "In exams: anything the examiner hasn't allowed, a website, a file, a software tool (and that includes AI tools), counts as unfair means. In assignments: your instructor's rules decide; ask before you use AI."

- "includes AI tools" is an *interpretation*. The Critic must allow only the quoted wording or an explicitly-labelled paraphrase ("software tools prohibited by the examiner").
- The paraphrase is shown at review.
- **CTA:** the CoE page URL.

This is also the demo of how the pipeline behaves when a requested fact doesn't exist: it reports the gap instead of inventing a policy.

## 12. Reproducibility
- **One command:** `python run.py --brief briefs/<id>.yaml --seed 42`.
  - Flags: `--offline` (replay source snapshots), `--from-stage X`, `--only X`, `--resume <run_id>`, `--renderer hyperframes|ffmpeg`, `--aspect 9x16|16x9`.
- **Seeds:** `stage_seed = sha256(seed, stage, item, loop)[:8]`.
  - LLM: `seed` per request, `temperature=0` for JSON/critic steps, fixed temperature 0.7 for creative steps, `-np 1` and `cache_prompt:false` (prompt caching can change outputs).
  - Diffusion: CPU `torch.Generator`.
  - Kokoro: `torch.manual_seed` per chunk.
  - HyperFrames: deterministic by design; any randomness via a seeded PRNG.
- **Stage cache:** LangGraph `SqliteCache`, with `key_func` = sha256(stage inputs, prompt sha, model revision sha, seed), because the built-in key ignores prompt edits.
- **Pins:** pip `==`, npm lockfile + exact HyperFrames version, llama.cpp build tag + binary sha256, model revision SHAs, Kaggle image build date.
- **Re-running one stage at the viva:**
  - `python run.py --brief … --seed 42 --only scriptwriter` reuses cached upstream artifacts and re-executes only that node, writing a diff vs the previous output.
  - For LangGraph time travel, `get_state_history` → `invoke(None, snapshot.config, interrupt_after=[node])` is the alternative path (test it before the viva).
- **Honest limits** (say these at the viva): llama.cpp/CUDA outputs are reproducible on the same build + GPU + seed, but bit-identity across hardware isn't guaranteed. The snapshots + cache make the *pipeline* exactly replayable regardless.

## 13. Repository layout

```
agentreel-<teamID>/
├── README.md  run.py  requirements.txt  requirements-voice312.txt  .env.example
├── config/  models.yaml  domains_allowlist.yaml  lexicon.yaml  styles.yaml  rules/{base,A,B,C,D,E}.yaml  licences_allowlist.yaml
├── briefs/  P00.yaml  C01.yaml  T-A05.yaml  T-B01.yaml  (test briefs)  <live>.yaml
├── agents/  planner.py researcher.py fact_curator.py scriptwriter.py hook_judge.py storyboarder.py
│            asset_generator.py editor.py critic.py human_review.py
├── core/    graph.py state.py schemas.py llm.py logging.py cache.py seeds.py budget.py sources.py qa_checks.py
├── workers/ image_sdxl.py tts_kokoro.py asr_whisper.py vlm_judge.py render_hyperframes.mjs render_ffmpeg.py
├── renderer/templates/<template>/{index.html, timeline.js, template.yaml}
├── prompts/ <agent>_v<N>.md
├── data/    sitemaps/  sources/  sources_manifest.csv  index/
├── assets/  official/ (logo + manifest row)  fonts/  manifest.csv
├── outputs/<run_id>/  plan.json facts.json hook_ab.json script.md storyboard.json scene.json
│                      TeamID_BriefID_final.mp4 .srt qa_report.json run_manifest.json review/
├── logs/    run_log.jsonl  cost_report.csv  interventions.md
├── notebooks/kaggle_live.ipynb   (setup cell · run cell · review panel cell)
├── scripts/ setup_kaggle.sh prefetch_models.py build_llamacpp_sm75.sh crawl_sitemaps.py
│            verify_logs.py make_agent_graph.py make_cost_report.py make_process_report.py
├── tests/   test_schemas.py test_qa_checks.py test_whitelist.py test_templates_render.py
└── docs/    agent_graph.png  process_report.pdf  consent/ (unused: no self-shot footage)
```

## 14. Live Run budget (3 h hard stop)

| Phase | Target | Notes |
|---|---|---|
| Before the draw | — | Session up, `setup_kaggle.sh` done, smoke test green, llama-server warm, Tavily key loaded |
| T+0:00–0:08 | 8 min | Draw card → fill `briefs/<ID>.yaml` from the template → schema validation |
| Planner | 1 min | |
| Researcher + FactCurator | 10 min | ≤ 15 Tavily calls, ≤ 40 fetches, embed/rerank on GPU1 |
| Script + 3 hooks + judge + storyboard | 6 min | ~6–8k output tokens at ~20–30 tok/s (VERIFY Qwen3.5-9B on T4) |
| Images | 5 min | ≤ 12 shots × 2 candidates × ~9 s + upscale |
| Voice + ASR check + captions | 3 min | |
| Render + loudnorm/encode | 6 min | HyperFrames on 4 CPU (VERIFY, estimated 2–6 min for 60 s) |
| Critic (hard + VLM + NLI) | 5 min | |
| **One clean pass** | **≈ 45 min** | |
| Repair loops (≤ 3, partial re-runs) | ≤ 40 min | Only failing stages re-run (cache) |
| Review (A/C) | 10 min | |
| Export, copy logs, tag `v1.0-final`, upload | 15 min | |
| Buffer | ≥ 50 min | |

`core/budget.py` tracks wall-clock per stage against the plan and writes overruns to the log. Above 120 min elapsed it cuts back: 1 image candidate, no VLM re-checks, ffmpeg renderer. This is a safety net, not the P4 feature.

## 15. Build schedule (relative days; two people in parallel)

| Day | Person A (agents & data) | Person B (runtime & media) |
|---|---|---|
| 0 | **Kaggle gate:** both accounts phone-verified, T4×2 selectable, internet on; Tavily key in Secrets | **Smoke tests on Kaggle:** llama.cpp prebuilt + Qwen3.5-9B tok/s; SDXL-Lightning s/img; Kokoro in a 3.12 venv; HyperFrames render of a 5 s 1080×1920 test incl. apt libs; ffmpeg loudnorm/libass |
| 1 | Repo skeleton, schemas, `run.py`, logging + hash chain, seeds, cache | Build llama.cpp sm_75, upload `agentreel-bin`/`-models`/`-wheels`; `setup_kaggle.sh` |
| 2 | Sitemap catalogue, allowlist, Tavily/ddgs search, fetch+snapshot+manifest | GPU1 worker protocol (subprocess, VRAM release); image worker + face/NSFW checks |
| 3 | Extraction, chunking, hybrid retrieval, contacts whitelist + precedence | Voice worker: lexicon, spoken/display, timings, ASR/WER; caption pager + `.srt` |
| 4 | FactCurator (quote check + HHEM), coverage map | HyperFrames templates ×8 (9:16 first), slot schemas, caption track, end card |
| 5 | Planner, Scriptwriter (spoken/display), HookJudge A/B | Editor: scene.json → render → loudnorm/encode; ffmpeg fallback renderer |
| 6 | Storyboarder + rule packs A–E | Critic hard checks (§9) + unit tests |
| 7 | Critic soft checks + routing + loop guard | VLM judge worker; 16:9 layouts |
| 8 | HumanReview interrupt + ipywidgets panel; `--only/--from-stage/--offline` | Full dry run **P00** ×2, time each stage |
| 9 | Dry runs: **C01**, one random **A** brief, one random **B** brief (drawn blind) | Fix failures; tune prompts/thresholds; licence manifest audit |
| 10 | README, agent graph, process-report generator, viva answers | Clean-session rerun from scratch with `--offline` = identical artifacts; tag `v1.0-build` |
| Live | Run, review, submit | Monitor GPUs/logs, export, upload |

Gate rules:
- If Day-0 HyperFrames fails on Kaggle, the ffmpeg renderer becomes primary.
- If Qwen3.5-9B is slower than 15 tok/s, switch the creative steps to fewer tokens or Qwen3-8B.

## 16. Risks and open items

| Risk / item | Impact | Mitigation / owner |
|---|---|---|
| Kaggle T4 greyed out / queue | Can't run | Test Day 0 on both accounts; run early in the day; keep the session pre-warmed |
| Runtime `apt-get` blocked | HyperFrames can't start Chrome | `.deb` files in `agentreel-bin`; fallback renderer |
| Kokoro on Py 3.13 | No voice | 3.12 venv; kokoro-onnx fallback |
| Gov sites TLS/JS shells | Missing contacts | `verify=False` only for allowlisted hosts (logged); PIB static pages |
| Somaiya page typos/conflicts | Wrong number on screen = O4 zero | Precedence rule + conflict shown at review |
| Tavily credits (1,000/mo) | Search fails late in month | ≤ 15 calls/run; sitemap catalogue + ddgs fallback; count credits in logs |
| No source for a `must_include` | Accuracy vs completeness | Report MISSING; never invent; shown at review |
| GSAP licence | Asset licence question | Record GSAP's "Standard No-Charge" licence in README; templates also work with CSS-only animation |
| **Open:** which official logo (KJSCE "formerly" SVG vs KJSSE vs SVU) | Brand rule | Ask faculty; until then use the SVG from kjsce.somaiya.edu, URL logged |
| **Open:** Team ID, repo name | File naming | Config values |
| **Open:** Student Code of Conduct PDF (scanned) | May contain AI/academic-integrity text | OCR on Day 2 |

## 17. Viva kit
- `docs/agent_graph.png` (from the compiled graph, rendered offline).
- **Live demo script:**
  1. Play the output.
  2. Click a claim → `facts.json` → snapshot file → original URL.
  3. Open `qa_report.json` from a dry run where the Critic caught an error (e.g. a contact mismatch or caption out of the safe zone) and show the automatic fix in `run_log.jsonl`.
  4. `--only scriptwriter --seed 42` and show the diff.
  5. `verify_logs.py`.
- **Answers for the 8 viva questions** are kept in `docs/viva_notes.md`. Points to cover:
  - Most cost/time: rendering and LLM tokens.
  - Next autonomy level: remove the review by auto-approving when every check passes, which the rules forbid for A/C.
  - Hallucinated helplines are prevented by the whitelist + quote match.
  - Ethical risks: faces, wrong contacts, policy over-claiming (C01).

## 18. Key sources (fetched 2026-10-21)
- vLLM GPU install & T4 backend: docs.vllm.ai/en/stable/getting_started/installation/gpu/ ; github.com/vllm-project/vllm/issues/60821, /38918, /36802, /47549
- llama.cpp grammars/JSON schema limits: raw.githubusercontent.com/ggml-org/llama.cpp/master/grammars/README.md
- Qwen3.5-9B: huggingface.co/Qwen/Qwen3.5-9B
- kaggle-vllm (Kaggle 2×T4 validation): github.com/kaggle-vllm/kaggle-vllm
- Kaggle limits: kaggle.com/docs/notebooks ; kaggle.com/docs/efficient-gpu-usage ; P100 retirement: kaggle.com/discussions/product-announcements/735239 ; Python 3.13 image: github.com/Kaggle/docker-python/pull/1562
- SDXL-LCM T4 timing: huggingface.co/blog/lcm_lora ; SDXL fp16 VAE: huggingface.co/madebyollin/sdxl-vae-fp16-fix ; FLUX.2 klein: bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence
- Kokoro: huggingface.co/hexgrad/Kokoro-82M ; faster-whisper: github.com/SYSTRAN/faster-whisper
- Remotion licence: github.com/remotion-dev/remotion/blob/main/LICENSE.md ; remotion.dev/docs/terms
- HyperFrames: github.com/heygen-com/hyperframes ; html-video: github.com/nexu-io/html-video
- LangGraph: docs.langchain.com/oss/python/langgraph (graph-api, interrupts, persistence, streaming); pypi.org/project/langgraph
- Tavily: docs.tavily.com/documentation/api-reference/endpoint/search ; docs.tavily.com/documentation/api-credits
- HHEM-2.1-Open: huggingface.co/vectara/hallucination_evaluation_model ; RapidOCR: github.com/RapidAI/RapidOCR ; YuNet: github.com/opencv/opencv_zoo
- Somaiya: somaiya.edu/en/center-of-examination/ (CoE circulars 133/134), somaiya.edu/en/anti-ragging-committee/, somaiya.edu/en/emergency/, brand.somaiya.edu.in, kjsce.somaiya.edu/sitemap.xml
- UGC 2018 academic integrity: ugc.gov.in/pdfnews/7771545_academic-integrity-Regulation2018.pdf ; MeitY AI governance (PIB PRID 2186639)
- AGEWEC 2026 (inspiration; licence-declaration practice): agewec.com
