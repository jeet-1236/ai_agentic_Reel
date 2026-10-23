"""AssetGenerator: per-shot illustrations (image_shot, Send fan-out), voice-over + captions (voice), join.

GPU discipline: LangGraph runs the Send branches concurrently in threads, but GPU1 holds one model at a time.
- Every worker call from this module is serialised by `_GPU_LOCK`.
- image_shot nodes are coalesced by `_Batcher`: the first shot to arrive waits a short window, then runs ONE
  image_sdxl call (one SDXL load) for every shot that arrived, then one checks call and one upscale call.
  Each node still returns only its own {"shot_assets": {sid: ...}}.
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Callable

from core.captions import align_display_words, check_pages, paginate, wer, write_srt
from core.context import ROOT, RunContext, load_yaml
from core.graph import script_fingerprint
from core.logging import RunLogger, sha256_file, sha256_text
from core.manifest import add_asset
from core.schemas import AssetRecord, Script, ShotAsset, VoiceAsset
from core.seeds import stage_seed
from core.state import AgentState
from core.workers import run_worker

AGENT = "asset_generator"
GPU = "1"
KB_HEADROOM = 0.12            # Ken Burns zoom/pan room around the 1080x1920 frame
_GPU_LOCK = threading.RLock()
_T0: dict[tuple[str, int], float] = {}
_T0_LOCK = threading.Lock()

BUILTIN_STYLE = {
    "name": "builtin_flat_vector",
    "prefix": "flat vector illustration, clean geometric shapes, bold simple forms,",
    "suffix": "minimal uncluttered background, soft gradients, high contrast, empty space in the lower third",
    "negative": ("photorealistic, photograph, realistic face, person portrait, text, letters, watermark, logo, "
                 "gore, weapon, nsfw, distorted, blurry, low quality"),
    "palette_hint": "", "sensitive": False, "sensitive_suffix": "",
}
PEOPLE_RE = re.compile(
    r"\b(people|persons?|humans?|man|men|woman|women|boys?|girls?|child(?:ren)?|kids?|students?|teachers?|"
    r"professors?|faculty|crowds?|faces?|portraits?|selfies?|someone|person's|people's|student's|students')\b",
    re.IGNORECASE)
NO_PEOPLE = "uninhabited scene, only objects, icons and symbols, still life"
FACE_WORDS = re.compile(r"face|people|person|human|realistic", re.IGNORECASE)
SHOT_REF = re.compile(r"\[[a-z_]+:(\d+)\]|\bshot[\s_#:-]*(\d+)\b", re.IGNORECASE)
NON_IMAGE = re.compile(r"voice|\bwer\b|pronunc|lexicon|caption|audio|speech", re.IGNORECASE)


# ================================================================ helpers (pure)
def _even(x: float) -> int:
    return int(round(x / 2.0)) * 2


def target_size(plan: dict, headroom: float = KB_HEADROOM) -> tuple[int, int]:
    """Final image size: output frame + Ken Burns headroom (1080x1920 -> 1210x2150)."""
    return _even(int(plan.get("width", 1080)) * (1 + headroom)), _even(int(plan.get("height", 1920)) * (1 + headroom))


def gen_size(plan: dict, models: dict) -> tuple[int, int]:
    img = models.get("image", {})
    key = "size_16x9" if plan.get("aspect") == "16x9" else "size_9x16"
    w, h = img.get(key, [1344, 768] if key == "size_16x9" else [768, 1344])
    return int(w), int(h)


def load_style(plan: dict) -> dict:
    """Style preset via core.rules (creative owner) with the pack's style_preset; built-in default otherwise."""
    preset = None
    for pack in reversed(plan.get("rule_packs") or []):
        p = ROOT / "config" / "rules" / f"{pack}.yaml"
        if p.exists():
            preset = (load_yaml(f"config/rules/{pack}.yaml") or {}).get("style_preset")
            if preset:
                break
    try:
        from core.rules import style_for

        return {**BUILTIN_STYLE, **style_for(str(plan.get("category", "P")), preset)}
    except Exception:  # noqa: BLE001 - styles.yaml missing or malformed: keep generating
        return dict(BUILTIN_STYLE)


def shot_feedback(feedback: dict | None, shot_id: int) -> list[str]:
    """Critic/review instructions for this shot. Items: "[faces:3] ..." / "shot 3: ..." strings or
    {"shot_id": 3, "instruction": "..."} dicts. Items without a shot reference are not applied to images."""
    out = []
    for item in (feedback or {}).get(AGENT, []) or []:
        if isinstance(item, dict):
            sid = item.get("shot_id", item.get("shot"))
            text = str(item.get("instruction") or item.get("text") or item.get("note") or "")
            if sid is not None and int(sid) == int(shot_id) and text:
                out.append(text)
            continue
        text = str(item)
        ids = {int(a or b) for a, b in SHOT_REF.findall(text)}
        if int(shot_id) in ids and not NON_IMAGE.search(text):
            out.append(SHOT_REF.sub("", text).strip(" :-"))
    return out


def voice_feedback(feedback: dict | None) -> tuple[list[str], dict]:
    """(voice-related notes, lexicon patch {word: phonemes}) from feedback[asset_generator]."""
    notes, patch = [], {}
    for item in (feedback or {}).get(AGENT, []) or []:
        if isinstance(item, dict) and isinstance(item.get("lexicon"), dict):
            patch.update({str(k): str(v) for k, v in item["lexicon"].items()})
        elif isinstance(item, str) and NON_IMAGE.search(item):
            notes.append(item)
    return notes, patch


def _clean_feedback(text: str) -> str:
    """Keep the actionable part of a critic message for an SDXL prompt."""
    m = re.search(r"Fix:\s*(.+)$", text, re.DOTALL)
    return " ".join((m.group(1) if m else text).split())[:200]


def build_prompt(visual_prompt: str, style: dict, extra: list[str] | None = None,
                 no_people: bool = False) -> tuple[str, str]:
    """(prompt, negative). Style prefix first: CLIP truncates at 77 tokens, so the tail is least important."""
    vp = visual_prompt
    if no_people:
        vp = " ".join(PEOPLE_RE.sub("", vp).split()).replace(" ,", ",")
    parts = [style.get("prefix", ""), vp, *[_clean_feedback(e) for e in (extra or [])]]
    if no_people:
        parts.append(NO_PEOPLE)
    if style.get("sensitive"):
        parts.append(style.get("sensitive_suffix", ""))
    parts += [style.get("palette_hint", ""), style.get("suffix", "")]
    prompt = ", ".join(p.strip(" ,") for p in parts if p and p.strip(" ,"))
    neg = style.get("negative", "")
    if no_people:
        neg = f"{neg}, people, person, human, face, crowd, character"
    return prompt, neg


def select_candidate(results: list[dict], max_nsfw: float) -> int | None:
    """First candidate with no face and NSFW below threshold."""
    for i, r in enumerate(results):
        if int(r.get("face_count", 0)) == 0 and float(r.get("nsfw_score", 0.0)) < max_nsfw:
            return i
    return None


def best_effort(results: list[dict]) -> int:
    return min(range(len(results)), key=lambda i: (int(results[i].get("face_count", 0)),
                                                   float(results[i].get("nsfw_score", 0.0))))


def _mark_start(ctx: RunContext, loop: int) -> None:
    with _T0_LOCK:
        _T0.setdefault((ctx.run_id, loop), time.time())


# ================================================================ batching
class _Batcher:
    """Coalesce concurrent submissions into batches processed by one leader thread."""

    def __init__(self, runner: Callable[[list[dict]], list], window_s: float):
        self.runner, self.window = runner, window_s
        self.lock = threading.Lock()
        self.pending: list[tuple[dict, Future]] = []
        self.leader = False

    def submit(self, job: dict):
        fut: Future = Future()
        with self.lock:
            self.pending.append((job, fut))
            lead = not self.leader
            self.leader = self.leader or lead
        if lead:
            self._lead()
        return fut.result()

    def _lead(self) -> None:
        while True:
            seen = -1
            while True:                      # debounce: wait until no new job arrived for one window
                time.sleep(self.window)
                with self.lock:
                    if len(self.pending) == seen:
                        batch, self.pending = self.pending, []
                        break
                    seen = len(self.pending)
            try:
                results = self.runner([j for j, _ in batch])
                for (_, f), r in zip(batch, results):
                    f.set_result(r)
            except BaseException as e:  # noqa: BLE001 - propagate to every waiting shot
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)
            with self.lock:
                if not self.pending:
                    self.leader = False
                    return


# ================================================================ image pipeline (one batch)
def _process_shots(ctx: RunContext, log: RunLogger, jobs: list[dict]) -> list[dict]:
    """Generate -> check -> (retry without people) -> upscale/cover-crop -> manifest. Returns ShotAsset dumps."""
    models = load_yaml("config/models.yaml")
    max_nsfw = float(models.get("qa", {}).get("nsfw", {}).get("max_score", 0.2))
    plan = jobs[0]["plan"]
    gw, gh = gen_size(plan, models)
    tw, th = target_size(plan, float(models.get("image", {}).get("kenburns_headroom", KB_HEADROOM)))
    mock = ctx.mock_media

    def generate(js: list[dict]) -> tuple[dict, dict]:
        out = run_worker(ctx, log, agent=AGENT, worker="image_sdxl.py", gpu=GPU, request={
            "task": "generate", "mock": mock, "width": gw, "height": gh, "seed": js[0]["seed"],
            "jobs": [{k: j[k] for k in ("prompt", "negative", "seed", "n", "out_prefix", "label")} for j in js]})
        return {r["out_prefix"]: r for r in out["results"]}, out.get("meta", {})

    def check(paths: list[str]) -> dict:
        out = run_worker(ctx, log, agent=AGENT, worker="image_checks.py", gpu=GPU,
                         request={"task": "check", "mock": mock, "paths": paths})
        if out.get("face_detector") in ("unavailable", "haar"):
            log.record(agent=AGENT, tool="face_check", status="error" if out["face_detector"] == "unavailable" else "ok",
                       error="YuNet model not found; face check degraded", units={"detector": out["face_detector"]})
        return {r["path"]: r for r in out["results"]}

    chosen: dict[int, dict] = {}
    with _GPU_LOCK:
        gen, gen_meta = generate(jobs)
        checks = check([p for r in gen.values() for p in r["paths"]])
        retry = []
        for j in jobs:
            r = gen[j["out_prefix"]]
            res = [checks[p] for p in r["paths"]]
            j["candidates"] = list(r["paths"])
            idx = select_candidate(res, max_nsfw)
            if idx is None:
                retry.append(j)
            else:
                chosen[j["shot_id"]] = {"path": r["paths"][idx], "seed": r["seeds"][idx], "check": res[idx],
                                        "prompt": j["prompt"]}
        if retry:
            rjobs = []
            for j in retry:
                p, n = build_prompt(j["visual_prompt"], j["style"], j["extra"], no_people=True)
                rjobs.append({**j, "prompt": p, "negative": n, "out_prefix": f"{j['out_prefix']}_r1",
                              "seed": stage_seed(ctx.seed, AGENT, f"shot{j['shot_id']}-retry", j["loop"])})
            gen2, _ = generate(rjobs)
            checks2 = check([p for r in gen2.values() for p in r["paths"]])
            for j, rj in zip(retry, rjobs):
                r2 = gen2[rj["out_prefix"]]
                j["candidates"] += r2["paths"]
                pool = [(p, s, checks.get(p) or checks2[p], j["prompt"]) for p, s in
                        zip(gen[j["out_prefix"]]["paths"], gen[j["out_prefix"]]["seeds"])]
                pool += [(p, s, checks2[p], rj["prompt"]) for p, s in zip(r2["paths"], r2["seeds"])]
                res = [c for _, _, c, _ in pool]
                idx = select_candidate(res, max_nsfw)
                idx = best_effort(res) if idx is None else idx
                chosen[j["shot_id"]] = {"path": pool[idx][0], "seed": pool[idx][1], "check": pool[idx][2],
                                        "prompt": pool[idx][3], "retried": True}
        up = run_worker(ctx, log, agent=AGENT, worker="upscale.py", gpu=GPU, request={
            "task": "upscale", "mock": mock, "width": tw, "height": th,
            "items": [{"path": chosen[j["shot_id"]]["path"],
                       "out_path": str(ctx.assets_dir / f"{j['out_prefix']}_final.png")} for j in jobs]})

    out = []
    up_meta = up.get("meta", {})
    for j, final in zip(jobs, up["paths"]):
        c = chosen[j["shot_id"]]
        gen_model = gen_meta.get("model", "")
        for p in j["candidates"]:
            add_asset(ctx, AssetRecord(path=p, kind="image_candidate", generator=gen_model,
                                       model_revision=gen_meta.get("model_revision", ""), seed=j["seed"],
                                       prompt_sha256=sha256_text(j["prompt"]), licence=gen_meta.get("licence", ""),
                                       sha256=sha256_file(p)))
        lic = gen_meta.get("licence", "")
        if up.get("method") == "realesrgan-x2":
            lic = f"{lic} AND {up_meta.get('licence', 'BSD-3-Clause')}"
        add_asset(ctx, AssetRecord(path=final, kind="image", generator=f"{gen_model}+{up_meta.get('model', '')}",
                                   model_revision=gen_meta.get("model_revision", ""), seed=c["seed"],
                                   prompt_sha256=sha256_text(c["prompt"]), licence=lic, sha256=sha256_file(final)))
        faces = int(c["check"].get("face_count", 0))
        notes = ["regenerated without people"] if c.get("retried") else []
        if faces:   # no VLM in the asset stage: an uncleared YuNet detection fails the critic's faces check
            scores = ", ".join(f"{s:.2f}" for s in c["check"].get("face_scores", []))
            notes.append(f"YuNet detected {faces} face(s) [{scores}] on every candidate incl. no-people retry; "
                         "not cleared by a VLM")
        out.append(ShotAsset(shot_id=j["shot_id"], image_path=final, candidates=j["candidates"], seed=c["seed"],
                             face_count=faces, nsfw_score=float(c["check"].get("nsfw_score", 0.0)),
                             vlm_ok=faces == 0, vlm_note="; ".join(notes),
                             visual_prompt=j["visual_prompt"]).model_dump(mode="json"))
    log.record(agent=AGENT, tool="image_batch", seed=jobs[0]["seed"], units={
        "shots": len(jobs), "images": sum(len(j["candidates"]) for j in jobs), "retried": len(retry),
        "upscale": up.get("method", ""), "size": [tw, th]})
    return out


def make_image_shot_node(ctx: RunContext, log: RunLogger, llm) -> Callable[[dict], dict]:
    """Node for one Send payload {shot, plan, brief, loop, feedback}."""
    models = load_yaml("config/models.yaml")
    window = 0.05 if ctx.mock_media else float(models.get("image", {}).get("batch_window_s", 0.75))
    batcher = _Batcher(lambda jobs: _process_shots(ctx, log, jobs), window)
    n_cand = int(models.get("image", {}).get("candidates_per_shot", 2))

    def node(payload: dict) -> dict:
        shot, plan = payload["shot"], payload["plan"]
        loop = int(payload.get("loop", 0) or 0)
        _mark_start(ctx, loop)
        sid = int(shot["shot_id"])
        style = load_style(plan)
        extra = shot_feedback(payload.get("feedback"), sid)
        no_people = any(FACE_WORDS.search(e) for e in extra)
        prompt, negative = build_prompt(shot.get("visual_prompt", ""), style, extra, no_people)
        job = {"shot_id": sid, "plan": plan, "loop": loop, "style": style, "extra": extra,
               "visual_prompt": shot.get("visual_prompt", ""), "prompt": prompt, "negative": negative,
               "seed": stage_seed(ctx.seed, AGENT, f"shot{sid}", loop), "n": n_cand,
               "out_prefix": f"shot{sid:02d}_l{loop}", "label": f"shot {sid}"}
        try:
            asset = batcher.submit(job)
        except Exception as e:  # noqa: BLE001 - fail soft: join reports the missing shot, critic re-routes
            msg = f"image_shot {sid}: {e.__class__.__name__}: {e}"[:1500]
            log.record(agent=AGENT, tool="image_shot", status="error", error=msg, units={"shot_id": sid})
            return {"errors": [msg]}
        return {"shot_assets": {str(sid): asset}}

    return node


# ================================================================ voice
def _glossary(state: AgentState, lexicon: dict) -> str:
    names = list((lexicon.get("words") or {}).keys()) + list((lexicon.get("respell") or {}).keys())
    names += list((state.get("plan") or {}).get("topic_keywords", []))[:12]
    return ("Glossary: " + ", ".join(dict.fromkeys(str(n) for n in names)) + ".") if names else ""


def make_voice_node(ctx: RunContext, log: RunLogger, llm) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        loop = int(state.get("loop", 0) or 0)
        _mark_start(ctx, loop)
        models = load_yaml("config/models.yaml")
        vcfg = models.get("voice", {})
        tts = vcfg.get("tts", {})
        wer_max = float(vcfg.get("wer_max", 0.08))
        sentences = Script.model_validate(state["script"]).sentences()
        spoken = [s.spoken for s in sentences]
        display = [s.display for s in sentences]
        voice, speed = tts.get("voice", "af_heart"), float(tts.get("speed", 1.0))
        try:
            from core.rules import voice_for

            v = voice_for(str((state.get("plan") or {}).get("category", "P")))
            voice, speed = v.get("voice", voice), float(v.get("speed", speed))
        except Exception:  # noqa: BLE001
            pass
        lexicon = load_yaml("config/lexicon.yaml") if (ROOT / "config" / "lexicon.yaml").exists() else {}
        notes, patch = voice_feedback(state.get("feedback"))
        lexicon = {**lexicon, "words": {**(lexicon.get("words") or {}), **patch}}
        reference = " ".join(spoken)
        prompt = _glossary(state, lexicon)

        attempts = []
        seeds = [stage_seed(ctx.seed, AGENT, "voice", loop), stage_seed(ctx.seed, AGENT, "voice-retry", loop)]
        for k, seed in enumerate(seeds):
            with _GPU_LOCK:
                t = run_worker(ctx, log, agent=AGENT, worker="tts_kokoro.py", gpu=GPU, request={
                    "task": "synth", "mock": ctx.mock_media, "sentences": spoken, "voice": voice, "speed": speed,
                    "lang_code": tts.get("lang_code", "a"), "seed": seed, "lexicon": lexicon, "pause_s": 0.25,
                    "out_name": f"voice_l{loop}_a{k}.wav"})
                areq = {"task": "transcribe", "mock": ctx.mock_media, "wav_path": t["wav_path"],
                        "initial_prompt": prompt, "language": "en"}
                if ctx.mock_media:
                    areq |= {"expected_text": reference, "expected_words": t["tokens"]}
                a = run_worker(ctx, log, agent=AGENT, worker="asr_whisper.py", gpu=GPU, request=areq)
            w = wer(reference, a.get("text", ""))
            attempts.append({"tts": t, "asr": a, "wer": w, "seed": seed})
            log.record(agent=AGENT, tool="voice_wer", seed=seed, units={
                "wer": w, "audio_s": t.get("duration_s", 0.0), "attempt": k, "lexicon_applied": t.get("lexicon_applied", 0)})
            if w <= wer_max:
                break
        best = min(attempts, key=lambda x: x["wer"])
        t, a = best["tts"], best["asr"]

        words = align_display_words(display, t["tokens"], t["sentence_spans"], fallback_words=a.get("words"))
        pages = paginate(words, end_time=float(t["duration_s"]))
        srt = write_srt(pages, ctx.artifact("captions.srt"))
        va = VoiceAsset(wav_path=t["wav_path"], duration_s=float(t["duration_s"]), voice=voice,
                        sentence_spans=[(float(s), float(e)) for s, e in t["sentence_spans"]], words=words,
                        asr_text=a.get("text", ""), wer=float(best["wer"]),
                        script_sha=script_fingerprint(state["script"]))
        ctx.write_json("voice.json", {**va.model_dump(mode="json"), "seed": best["seed"], "attempts":
                                      [{"seed": x["seed"], "wer": x["wer"]} for x in attempts], "notes": notes})
        ctx.write_json("captions.json", [p.model_dump(mode="json") for p in pages])

        tm = t.get("meta", {})
        add_asset(ctx, AssetRecord(path=t["wav_path"], kind="voice", generator=tm.get("model", ""),
                                   model_revision=tm.get("model_revision", ""), seed=best["seed"],
                                   prompt_sha256=sha256_text(reference), licence=tm.get("licence", ""),
                                   sha256=sha256_file(t["wav_path"])))
        add_asset(ctx, AssetRecord(path=str(srt), kind="caption", generator="agentreel/core.captions",
                                   prompt_sha256=sha256_text(" ".join(display)), licence="own-work",
                                   sha256=sha256_file(srt)))
        issues = check_pages(pages)
        log.record(agent=AGENT, tool="captions", output_path=str(srt), units={
            "pages": len(pages), "issues": len(issues), "wer": best["wer"], "audio_s": va.duration_s,
            "attempts": len(attempts)})
        return {"voice": va.model_dump(mode="json"), "captions": [p.model_dump(mode="json") for p in pages],
                "srt_path": str(srt)}

    return node


# ================================================================ join
def make_join_node(ctx: RunContext, log: RunLogger, llm) -> Callable[[AgentState], dict]:
    def node(state: AgentState) -> dict:
        loop = int(state.get("loop", 0) or 0)
        shots = (state.get("storyboard") or {}).get("shots", [])
        have = state.get("shot_assets") or {}
        from core.graph import image_is_current, voice_is_current

        missing = [int(s["shot_id"]) for s in shots if s.get("visual_prompt")
                   and not (image_is_current(s, have.get(str(s["shot_id"])))
                            and Path(have[str(s["shot_id"])]["image_path"]).exists())]
        voice_ok = voice_is_current(state)
        ready = not missing and voice_ok
        with _T0_LOCK:
            t0 = _T0.pop((ctx.run_id, loop), None)
        wall = round(time.time() - t0, 2) if t0 else 0.0
        log.record(agent=AGENT, tool="assets_join", status="ok" if ready else "error",
                   units={"shots": len(shots), "images": len(have), "missing_shots": missing, "voice": voice_ok,
                          "assets_wall_s": wall, "loop": loop})
        out: dict = {"extra": {"assets_ready": ready, "redo_voice": False, "missing_shots": missing},
                     "redo_shots": [], "stage_times": {"assets": wall}}
        if not ready:
            out["errors"] = [f"assets_join: missing images for shots {missing}" if missing else "",
                             "" if voice_ok else "assets_join: voice missing"]
            out["errors"] = [e for e in out["errors"] if e]
        return out

    return node
