# Viva notes: prepared answers

These answers are grounded in this repo's design (`PLAN.md`) and in what the logs show. The questions are
**paraphrased** from the handout's viva list. Before the viva, check the numbering against the handout and swap
in real numbers from the Live Run (`logs/cost_report.csv`, `outputs/<run>/qa_report.json`, `docs/process_report.md`).

Have these open for the demo: the final MP4, `facts.json`, one snapshot in `data/sources/`, `qa_report.json`
from a dry run where the Critic caught something, and a terminal in the repo.

---

## Q1. Walk us through your agent architecture. Why these agents?

- It is a LangGraph state machine of 11 nodes: Planner -> Researcher -> FactCurator -> Scriptwriter ->
  HookJudge -> Storyboarder -> AssetGenerator (fan-out per shot + voice) -> Editor -> Critic ->
  (HumanReview for A/C) -> Export. Show `docs/agent_graph.png`. It is generated from the compiled graph, not
  drawn by hand.
- Each agent owns one artifact and one failure class. The Critic can therefore route a failure to the agent
  responsible for it: a contact not in the whitelist goes to Scriptwriter, a face goes to AssetGenerator for
  that shot only, loudness goes to Editor.
- Agents talk only through Pydantic schemas (`core/schemas.py`). LLM outputs are JSON-schema constrained in
  llama.cpp, with one repair retry.
- Deterministic work stays in code: word budget, shot count, quote-substring checks, ffprobe/ebur128 specs and
  caption geometry. The LLM handles only language and judgement.

## Q2. Which step costs the most time and money? How did you measure it?

- Actual money: **Rs 0**. All models are local on Kaggle's free T4s, and Tavily is on its free tier (at most
  15 calls per run, credits logged).
- Time: rendering (HyperFrames headless Chrome on 4 CPUs) and LLM tokens (script with 3 hooks, judge, critic)
  dominate. Images are about 9 s each with SDXL-Lightning 4-step. Read the real numbers from
  `logs/cost_report.csv` and the stage-time table in `docs/process_report.md`.
- `make_cost_report.py` also has a **shadow cost** column: GPU-seconds multiplied by an assumed public T4
  hourly rate and an INR rate. The assumptions are written in the CSV. This shows what a paid deployment would
  cost, without claiming we spent it.
- Measurement: every call is one JSONL line with latency, tokens (from llama.cpp `usage`) and units. Nothing
  is estimated after the fact.

## Q3. How reproducible is a run? Show us.

- One command with `--seed 42`. Per-stage seeds are `sha256(seed, stage, item, loop)`. JSON steps run at
  temperature 0, creative steps at a fixed 0.7, with `-np 1` and `cache_prompt:false` on llama.cpp. Kokoro
  gets `torch.manual_seed` before each chunk, and diffusion uses a CPU generator.
- Everything is pinned: pip `==`, npm lockfile, llama.cpp tag and binary sha256, HF revision SHAs and file
  sha256 in `config/models.lock.yaml`, the Kaggle image build date in `run_manifest.json`.
- Sources are snapshotted by sha256, so `--offline` replays byte-identical inputs.
- **Demo:** `python run.py --brief briefs/<id>.yaml --seed 42 --only scriptwriter` (notebook cell 8). The
  upstream stages come from the cache, only that node runs again, and the cell prints the diff.
- **Honest limit:** CUDA/llama.cpp output is reproducible on the same build, GPU and seed. Bit-identity across
  different hardware is not guaranteed. The snapshots and the stage cache make the *pipeline* exactly
  replayable anyway.

## Q4. How do you stop the system from hallucinating facts, helplines or policies?

- **Facts:** the LLM must give a verbatim quote and a chunk id for every fact. Code checks that the normalised
  quote is a substring of the snapshotted source and that HHEM scores it at least 0.5. Facts that fail are
  dropped. Every script sentence carries `fact_ids`, and the Critic rejects factual sentences without them.
- **Helplines:** contacts are extracted by regex from source text only, into a whitelist with source ids. When
  sources disagree, the national helpline's own government page beats PIB, which beats Somaiya. This is
  motivated by real typos on a Somaiya page. Any number or email in the script or OCR'd frames that is not in
  the whitelist is a hard failure.
- **Missing information is reported, not invented.** A `must_include` item with no source is marked MISSING
  and shown at review. C01 is the example: no public Somaiya AI policy exists, so the video quotes the CoE
  unfair-means circular instead of inventing a policy.
- **Demo:** click a claim, open `facts.json`, then the snapshot file, then the original URL.

## Q5. Show a case where the system corrected itself.

- Open `qa_report.json` from a dry run with a failure, for example a contact mismatch or a caption outside
  the safe zone. In `run_log.jsonl`, the next lines show the Critic setting `route`, the responsible agent
  re-running with `feedback`, and the next loop passing.
- Loop guard: at most 3 loops. `history[]` stores (check id, artifact hash). If a fix reproduces an earlier
  failing artifact, the loop stops. Then the best version that passes all hard checks ships, or goes to review
  flagged (A/C).

## Q6. What autonomy level do you claim? What would the next level take?

- **L4** for B/D/E: one command, no human. **L3** for A/C, because the rules require one review before
  export. The review is a LangGraph `interrupt()` with an approve/reject panel in the notebook, logged as
  `review_checkpoint` and not counted as an intervention.
- Interventions so far: show `python scripts/verify_logs.py` (per-run count; target 0, limit 5).
- Next level: auto-approve A/C when every hard and soft check passes and the coverage map has no MISSING
  items. Technically that is one routing change. The handout forbids it for safety and policy topics, and we
  agree, because the review is where a human catches tone and framing issues that metrics miss.

## Q7. What are the ethical risks? How did you handle them?

- **Wrong contact or wrong policy:** the whitelist with precedence, the quote check and mandatory review for
  A/C. For C01 we never imply that an AI policy exists.
- **Realistic people and deepfake risk:** only illustration style presets, plus the YuNet face detector and a
  VLM "photoreal?" check, routed back to that shot. No self-shot footage, so no consent forms are needed.
- **Harmful framing on safety topics (A):** safe-messaging rules (no methods, hopeful tone, help card last, no
  victim depiction, no blaming a group), checked by the LLM critic with sentence-level reasons.
- **Licences and attribution:** every asset has a manifest row with its licence, checked against an
  allowlist. Models with non-commercial licences were rejected (see README). An AI disclosure line is on the
  end card, and the official logo is unaltered.
- **Privacy and security:** the servers bind to 127.0.0.1 only (Kaggle AUP), keys live in Kaggle Secrets, and
  we fetch only public official pages. TLS bypass is allowed only for allowlisted hosts and is logged.

## Q8. What failed during development, and what would you improve?

*Fill in from the dry runs (examples of what to cover):*
- Kaggle moved to Python 3.13, which forced a separate 3.12 venv for Kokoro (fallback: kokoro-onnx).
- Chrome needed system libraries that the Ubuntu 24.04 image doesn't ship. Setup installs them, with `.deb`
  files as a fallback and the ffmpeg renderer as a last resort.
- LangGraph's built-in cache key ignores prompt edits, so we added the prompt and model fingerprint to the key.
- llama.cpp's grammar converter breaks on nested `$ref`, so schemas are kept flat and refs are inlined.
- Improvements: a faster render path (GPU encode, fewer Chrome frames), a learned hook judge calibrated
  against real audience feedback, multilingual VO and captions, and auto-approval once the rules allow it.

---

### Live demo script (5 min)

1. Play the final video.
2. Click a claim: `facts.json`, then `data/sources/<sha>.html`, then the URL.
3. Open `qa_report.json` from the caught-error dry run and the matching `run_log.jsonl` lines.
4. Run `python run.py --brief briefs/<id>.yaml --seed 42 --only scriptwriter` and show the diff.
5. Run `python scripts/verify_logs.py`: chain OK. Edit one byte in a copy, run it again: BROKEN.
