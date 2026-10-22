#!/usr/bin/env bash
# AgentReel Kaggle setup (idempotent, ~10 min cold, <1 min warm). Run BEFORE the brief draw:
#     bash scripts/setup_kaggle.sh                # full setup + llama-server + smoke test
#     SKIP_LLM=1 SKIP_SMOKE=1 bash scripts/setup_kaggle.sh
# Two ways to get binaries/models (both supported, datasets win when attached):
#   ONLINE (default, Internet ON): llama.cpp from its GitHub release, Node from nodejs.org, models from Hugging Face
#     at the revisions pinned in config/models.yaml, pip/npm from their registries.
#   DATASETS: attach agentreel-models / agentreel-bin / agentreel-wheels (scripts/make_kaggle_datasets.md).
# Env knobs:
#   AGENTREEL_MODELS_DS / AGENTREEL_BIN_DS / AGENTREEL_WHEELS_DS / AGENTREEL_FONTS_DS  dataset dirs (auto-found)
#   AGENTREEL_SCRATCH  big re-creatable files (default /tmp/agentreel on Kaggle: /kaggle/working is capped at 20 GB)
#   LLAMA_TAG          llama.cpp release (default b11796)   NODE_VERSION (default v22.23.4)
#   OFFLINE=1     never touch the internet (fail instead of falling back to PyPI/npm/HF)
#   SKIP_MODELS=1 don't download models (e.g. mock-only runs)   WITH_FALLBACK_MODELS=1 also fetch Qwen3-14B etc.
#   SMOKE_ARGS    args for scripts/smoke_test.py (default: --real when a GPU is visible)
# Writes .setup/env (KEY=VALUE lines) that the notebook loads into os.environ.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${AGENTREEL_KAGGLE_INPUT:-/kaggle/input}"
STAMP="$REPO/.setup"
LLAMA_TAG="${LLAMA_TAG:-b11796}"
NODE_VERSION="${NODE_VERSION:-v22.23.4}"
if [ -z "${AGENTREEL_SCRATCH:-}" ]; then
  if [ -d /kaggle ]; then AGENTREEL_SCRATCH=/tmp/agentreel; else AGENTREEL_SCRATCH="$REPO/.scratch"; fi
fi
SCRATCH="$AGENTREEL_SCRATCH"
mkdir -p "$SCRATCH/bin" "$SCRATCH/dl" "$STAMP" "$REPO/logs"
# heavy, re-creatable things live on scratch; the repo only holds symlinks to them
[ -e "$REPO/bin" ] || ln -sfn "$SCRATCH/bin" "$REPO/bin"
BIN="$REPO/bin"
ENVF="$STAMP/env"
: > "$ENVF"
cd "$REPO" || exit 1

G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; B=$'\033[1m'; N=$'\033[0m'
declare -a SUMMARY=()
FAILS=0
ok()   { SUMMARY+=("${G}PASS${N} $1"); echo "${G}[ok]${N} $1"; }
bad()  { SUMMARY+=("${R}FAIL${N} $1"); echo "${R}[FAIL]${N} $1"; FAILS=$((FAILS+1)); }
warn() { SUMMARY+=("${Y}WARN${N} $1"); echo "${Y}[warn]${N} $1"; }
step() { echo; echo "${B}== $1${N}"; }
have() { command -v "$1" >/dev/null 2>&1; }
setenv() { export "$1=$2"; echo "$1=$2" >> "$ENVF"; }
online() { [ "${OFFLINE:-0}" != "1" ]; }
load_env() { while IFS="=" read -r k v; do [ -n "$k" ] && export "$k=$v"; done < "$ENVF"; }

SUDO=""
if [ "$(id -u)" != "0" ] && have sudo; then SUDO="sudo"; fi

# ------------------------------------------------------------------ 0. platform
step "0. platform"
if [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ] || [ -d /kaggle ]; then
  ok "Kaggle detected (run type: ${KAGGLE_KERNEL_RUN_TYPE:-unknown})"
else
  warn "not on Kaggle; continuing (paths relative to $REPO)"
fi
PY="$(command -v python3 || command -v python)"
echo "python: $($PY --version 2>&1)  repo: $REPO"
GPUS=0
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
  GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
[ "$GPUS" -ge 2 ] && ok "$GPUS GPUs visible" || warn "$GPUS GPUs visible (need 2x T4 for --real runs)"
setenv HYPERFRAMES_NO_TELEMETRY 1
setenv PYTHONUNBUFFERED 1
setenv AGENTREEL_SCRATCH "$SCRATCH"
setenv HF_HOME "$SCRATCH/hf"                       # any runtime HF cache stays off /kaggle/working
# llama.cpp release builds carry sm_75 as PTX: the first load JIT-compiles kernels. Keep that cache big and in
# the repo dir so a saved notebook version (or a second run in the same session) reuses it.
setenv CUDA_CACHE_PATH "$REPO/.cache/nv"
setenv CUDA_CACHE_MAXSIZE 4294967296
echo "scratch: $SCRATCH ($(df -h "$SCRATCH" 2>/dev/null | awk 'NR==2 {print $4 " free"}'))"

# ------------------------------------------------------------------ 1. datasets
step "1. datasets"
find_ds() {  # $1 = dataset slug; mount layout changed in Oct 2026, so search instead of hard-coding
  local hit
  hit="$(find "$INPUT" -maxdepth 5 -type d -name "$1*" 2>/dev/null | head -n 1)"
  echo "$hit"
}
DS_MODELS="${AGENTREEL_MODELS_DS:-$(find_ds agentreel-models)}"
DS_BIN="${AGENTREEL_BIN_DS:-$(find_ds agentreel-bin)}"
DS_WHEELS="${AGENTREEL_WHEELS_DS:-$(find_ds agentreel-wheels)}"
DS_FONTS="${AGENTREEL_FONTS_DS:-$(find_ds agentreel-fonts)}"
for pair in "models:$DS_MODELS" "bin:$DS_BIN" "wheels:$DS_WHEELS" "fonts:$DS_FONTS"; do
  name="${pair%%:*}"; dir="${pair#*:}"
  if [ -n "$dir" ] && [ -d "$dir" ]; then ok "dataset agentreel-$name -> $dir"; else warn "dataset agentreel-$name not attached"; fi
done

if [ -n "$DS_MODELS" ]; then
  if [ -L "$REPO/models" ] || [ ! -e "$REPO/models" ]; then
    ln -sfn "$DS_MODELS" "$REPO/models"
  else
    warn "models/ is a real directory; leaving it, adding dataset via AGENTREEL_MODELS_DIR"
  fi
  setenv AGENTREEL_MODELS_DIR "$REPO/models:$DS_MODELS"
fi
if [ -n "$DS_FONTS" ]; then
  mkdir -p "$REPO/assets/fonts"
  find "$DS_FONTS" -type f \( -iname '*.ttf' -o -iname '*.otf' -o -iname '*.woff2' -o -iname 'OFL*.txt' \) \
    -exec cp -n {} "$REPO/assets/fonts/" \; 2>/dev/null
  mkdir -p "$HOME/.fonts" && cp -n "$REPO"/assets/fonts/*.[ot]tf "$HOME/.fonts/" 2>/dev/null
  have fc-cache && fc-cache -f >/dev/null 2>&1
  ok "fonts copied to assets/fonts ($(ls "$REPO/assets/fonts" | wc -l) files)"
fi

bin_file() {  # first file in agentreel-bin matching a glob
  [ -n "$DS_BIN" ] && find "$DS_BIN" -maxdepth 3 -type f -name "$1" 2>/dev/null | sort | tail -n 1
}

# llama.cpp (sm_75 build from scripts/build_llamacpp_sm75.sh)
if [ ! -x "$BIN/llama/llama-server" ]; then
  TGZ="$(bin_file 'llama-server-*sm75*.tar.gz')"
  if [ -n "$TGZ" ]; then
    mkdir -p "$BIN/llama" && tar -xzf "$TGZ" -C "$BIN/llama" --strip-components=1 && chmod +x "$BIN/llama/"llama-* 2>/dev/null
  fi
fi
if [ ! -x "$BIN/llama/llama-server" ] && online; then
  # official release: CUDA 12.8 build + its cudart/cuBLAS bundle, extracted into one dir (no system CUDA needed)
  REL="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG"
  mkdir -p "$BIN/llama"
  for t in "llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz" "cudart-llama-$LLAMA_TAG-bin-ubuntu-cuda-12.8-x64.tar.gz"; do
    [ -f "$SCRATCH/dl/$t" ] || curl -fsSL --retry 3 -o "$SCRATCH/dl/$t" "$REL/$t" || { rm -f "$SCRATCH/dl/$t"; continue; }
    tar -xzf "$SCRATCH/dl/$t" -C "$BIN/llama" --strip-components=1
  done
  chmod +x "$BIN/llama/"llama-* 2>/dev/null
  [ -x "$BIN/llama/llama-server" ] && warn "llama.cpp $LLAMA_TAG from GitHub release (sm_75 via PTX JIT: first load is slow)"
fi
if [ -x "$BIN/llama/llama-server" ]; then
  setenv AGENTREEL_LLAMA_BIN "$BIN/llama/llama-server"
  ok "llama-server: $(LD_LIBRARY_PATH="$BIN/llama" "$BIN/llama/llama-server" --version 2>&1 | head -n 1)"
elif have llama-server; then
  ok "llama-server on PATH: $(command -v llama-server)"
else
  bad "llama-server missing (attach agentreel-bin or run scripts/build_llamacpp_sm75.sh)"
fi

# ------------------------------------------------------------------ 2. python wheels (system 3.13)
step "2. python packages (system env)"
have uv || { online && $PY -m pip install -q uv; }
REQ="$REPO/requirements.txt"
REQ_SHA="$(cat "$REQ" "$REPO"/requirements/*.txt 2>/dev/null | sha256sum | cut -c1-16)"
uv_sys() { uv pip install --system "$@" || uv pip install --system --break-system-packages "$@"; }
if [ -f "$STAMP/req.$REQ_SHA" ]; then
  ok "requirements already installed ($REQ_SHA)"
else
  WH313=""
  [ -n "$DS_WHEELS" ] && WH313="$(find "$DS_WHEELS" -maxdepth 3 -type d -name 'py313' | head -n 1)"
  # merge requirements.txt + requirements/*.txt (owners' files; voice312 excluded). Identical duplicate pins are
  # fine for uv; conflicting pins fail loudly. Drop CPU onnxruntime if Kaggle already ships onnxruntime-gpu.
  MERGED="$STAMP/requirements.merged.txt"
  cat "$REQ" > "$MERGED"
  for f in "$REPO"/requirements/*.txt; do
    [ -f "$f" ] || continue
    case "$f" in *voice312*) continue;; esac
    printf '\n' >> "$MERGED"; cat "$f" >> "$MERGED"
  done
  if $PY -c "import importlib.metadata as m; m.version('onnxruntime-gpu')" >/dev/null 2>&1; then
    sed -i -E '/^onnxruntime==/d' "$MERGED"
  fi
  sed -i 's/\r$//' "$MERGED"
  if [ -n "$WH313" ] && uv_sys --no-index --find-links "$WH313" -r "$MERGED"; then
    touch "$STAMP/req.$REQ_SHA"; ok "wheels installed offline from $WH313"
  elif online && uv_sys -r "$MERGED"; then
    touch "$STAMP/req.$REQ_SHA"; warn "wheels installed from PyPI (agentreel-wheels missing or incomplete)"
  else
    bad "pip install failed"
  fi
fi
$PY -c "import langgraph, langchain_openai, trafilatura, bm25s" 2>/dev/null \
  && ok "core imports (langgraph, langchain_openai, trafilatura, bm25s)" || bad "core imports failed"

# ------------------------------------------------------------------ 2b. models from Hugging Face (no dataset)
step "2b. models"
if [ -n "$DS_MODELS" ]; then
  ok "models from dataset ($DS_MODELS)"
elif [ "${SKIP_MODELS:-0}" = "1" ]; then
  warn "SKIP_MODELS=1: no model weights (mock runs only)"
elif online; then
  MDIR="$SCRATCH/dl/agentreel-models"
  FREE_GB="$(df -BG "$SCRATCH" | awk 'NR==2 {gsub("G","",$4); print $4}')"
  NEED_GB=32; [ "${WITH_FALLBACK_MODELS:-0}" = "1" ] && NEED_GB=45
  if [ -f "$STAMP/models.done" ]; then
    ok "models already downloaded ($(du -sh "$MDIR" 2>/dev/null | cut -f1))"
  elif [ "${FREE_GB:-0}" -lt "$NEED_GB" ]; then
    bad "only ${FREE_GB} GB free on $SCRATCH; models need ~${NEED_GB} GB (set AGENTREEL_SCRATCH to a bigger disk)"
  else
    FB=""; [ "${WITH_FALLBACK_MODELS:-0}" = "1" ] && FB="--include-fallbacks"
    # pinned revisions from config/models.yaml; writes config/models.lock.yaml (file sha256s) for the run manifest
    if $PY scripts/prefetch_models.py --out "$SCRATCH/dl" $FB 2>&1 | tee "$REPO/logs/prefetch_models.log" | grep -E "^(==|   FAILED|FAILURES)"; [ "${PIPESTATUS[0]}" = "0" ]; then
      touch "$STAMP/models.done"; ok "models downloaded from Hugging Face ($(du -sh "$MDIR" | cut -f1))"
    else
      bad "model download failed: see logs/prefetch_models.log"
    fi
  fi
  setenv AGENTREEL_MODELS_DIR "$MDIR"
else
  bad "no agentreel-models dataset and OFFLINE=1"
fi

# ------------------------------------------------------------------ 3. voice venv (Python 3.12, kokoro)
step "3. voice venv (.venv-voice, Python 3.12)"
VPY="$REPO/.venv-voice/bin/python"
VREQ=""
for f in "$REPO/requirements-voice312.txt" "$REPO/requirements/voice312.txt"; do [ -f "$f" ] && VREQ="$f" && break; done
if [ -z "$VREQ" ]; then
  # default pins (kokoro needs Python <3.13); en_core_web_sm is fetched by misaki at runtime unless pre-installed
  VREQ="$STAMP/voice312.default.txt"
  printf '%s\n' "kokoro==0.9.4" "misaki[en]==0.9.4" "soundfile==0.13.1" "numpy==2.2.6" "torch==2.8.0" > "$VREQ"
fi
VREQ_SHA="$(sha256sum "$VREQ" | cut -c1-16)"
if [ -x "$VPY" ] && [ -f "$STAMP/voice.$VREQ_SHA" ]; then
  ok "voice venv ready"
else
  PY312=""
  PYTGZ="$(bin_file 'cpython-3.12*-x86_64-unknown-linux-gnu-install_only*.tar.gz')"
  if [ -n "$PYTGZ" ]; then
    mkdir -p "$BIN/python312" && tar -xzf "$PYTGZ" -C "$BIN/python312" && PY312="$BIN/python312/python/bin/python3.12"
  fi
  [ -z "$PY312" ] && have python3.12 && PY312="$(command -v python3.12)"
  [ -z "$PY312" ] && PY312="3.12"     # uv downloads a python-build-standalone 3.12 (needs internet)
  VENV_DIR="$REPO/.venv-voice"
  if [ -d /kaggle ] && [ ! -e "$VENV_DIR" ]; then   # ~3 GB (torch): keep it on scratch, symlink into the repo
    VENV_DIR="$SCRATCH/venv-voice"; ln -sfn "$VENV_DIR" "$REPO/.venv-voice"
  fi
  uv venv --python "$PY312" "$VENV_DIR" --allow-existing >/dev/null 2>&1 || uv venv --python "$PY312" "$VENV_DIR"
  WH312=""
  [ -n "$DS_WHEELS" ] && WH312="$(find "$DS_WHEELS" -maxdepth 3 -type d -name 'py312' | head -n 1)"
  if [ -n "$WH312" ] && uv pip install --python "$VPY" --no-index --find-links "$WH312" -r "$VREQ" \
       && { ls "$WH312"/en_core_web_sm-*.whl >/dev/null 2>&1 && uv pip install --python "$VPY" --no-index "$WH312"/en_core_web_sm-*.whl || true; }; then
    touch "$STAMP/voice.$VREQ_SHA"; ok "voice venv from wheels ($WH312)"
  elif online && uv pip install --python "$VPY" -r "$VREQ" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match; then
    uv pip install --python "$VPY" "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" || true
    touch "$STAMP/voice.$VREQ_SHA"; warn "voice venv from PyPI (agentreel-wheels/py312 missing)"
  else
    bad "voice venv failed -> fallback: pip install kokoro-onnx==0.6.1 into the system env and set voice.python to python3 (see README)"
  fi
fi
if [ -x "$VPY" ]; then
  # workers/tts_kokoro.py writes WAVs with the stdlib `wave` module, so soundfile is not required
  CUDA_VISIBLE_DEVICES=1 "$VPY" -c "import kokoro, misaki, torch; print('kokoro ok, cuda', torch.cuda.is_available(), torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '')" \
    && ok "kokoro imports in .venv-voice" || bad "kokoro import failed in .venv-voice"
fi

# ------------------------------------------------------------------ 4. ffmpeg
step "4. ffmpeg"
ff_ok() {
  have ffmpeg || return 1
  local f e
  f="$(ffmpeg -hide_banner -filters 2>/dev/null)"; e="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
  # here-strings, not `echo | grep -q`: grep -q exits early -> SIGPIPE on echo -> false failure under pipefail
  grep -qE ' loudnorm ' <<<"$f" && grep -qE ' subtitles ' <<<"$f" && grep -qE ' ass ' <<<"$f" \
    && grep -q libx264 <<<"$e" && grep -qE ' aac ' <<<"$e"
}
if ff_ok; then
  ok "system ffmpeg has loudnorm/subtitles/ass/libx264/aac ($(ffmpeg -version | head -n 1 | cut -c1-40))"
else
  FTX="$(bin_file 'ffmpeg-*linux64-gpl*.tar.xz')"
  if [ -z "$FTX" ] && online; then   # BtbN static GPL build (libass, x264, loudnorm)
    FTX="$SCRATCH/dl/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"
    [ -f "$FTX" ] || curl -fsSL --retry 3 -o "$FTX" \
      "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz" \
      || { rm -f "$FTX"; FTX=""; }
  fi
  if [ -n "$FTX" ]; then
    mkdir -p "$BIN/ffmpeg" && tar -xJf "$FTX" -C "$BIN/ffmpeg" --strip-components=1
    $SUDO ln -sf "$BIN/ffmpeg/bin/ffmpeg" /usr/local/bin/ffmpeg && $SUDO ln -sf "$BIN/ffmpeg/bin/ffprobe" /usr/local/bin/ffprobe
    hash -r
  fi
  ff_ok && ok "static ffmpeg from agentreel-bin linked to /usr/local/bin" || bad "ffmpeg lacks loudnorm/subtitles/libx264 (renderer QA will fail)"
fi

# ------------------------------------------------------------------ 5. Node 22 + renderer
step "5. Node 22 + HyperFrames renderer"
node_major() { node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/'; }
if [ "$(node_major)" -ge 22 ] 2>/dev/null; then
  ok "node $(node --version) already on PATH"
else
  NTX="$(bin_file 'node-v22*-linux-x64.tar.xz')"
  if [ -z "$NTX" ] && online; then   # official tarball from nodejs.org
    NTX="$SCRATCH/dl/node-$NODE_VERSION-linux-x64.tar.xz"
    [ -f "$NTX" ] || curl -fsSL --retry 3 -o "$NTX" "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz" \
      || { rm -f "$NTX"; NTX=""; }
  fi
  if [ -n "$NTX" ]; then
    mkdir -p "$BIN/node" && tar -xJf "$NTX" -C "$BIN/node" --strip-components=1
  elif online; then
    # fallback: Node shipped as a Python wheel
    uv_sys "nodejs-wheel-binaries==22.20.0" >/dev/null 2>&1 || true
    NW="$($PY -c 'import nodejs_wheel, pathlib; p = pathlib.Path(nodejs_wheel.__file__).parent; print(next(p.rglob("bin/node")))' 2>/dev/null)"
    [ -n "$NW" ] && mkdir -p "$BIN/node" && ln -sfn "$(dirname "$(dirname "$NW")")"/* "$BIN/node/" 2>/dev/null
  fi
  if [ -x "$BIN/node/bin/node" ]; then
    for t in node npm npx; do [ -e "$BIN/node/bin/$t" ] && $SUDO ln -sf "$BIN/node/bin/$t" "/usr/local/bin/$t"; done
    hash -r
  fi
  [ "$(node_major)" -ge 22 ] 2>/dev/null && ok "node $(node --version) installed" || bad "Node 22 not available (renderer falls back to ffmpeg: --renderer ffmpeg)"
fi

if have node; then
  if [ -f "$REPO/renderer/node_modules/hyperframes/package.json" ]; then
    ok "renderer/node_modules present"
  else
    NMT="$(bin_file 'renderer_node_modules*.tar.gz')"
    NPMC="$(bin_file 'npm-cache*.tar.gz')"
    if [ -n "$NMT" ]; then
      tar -xzf "$NMT" -C "$REPO/renderer" && ok "renderer/node_modules from agentreel-bin"
    elif [ -n "$NPMC" ]; then
      mkdir -p /tmp/agentreel-npm-cache && tar -xzf "$NPMC" -C /tmp/agentreel-npm-cache
      (cd "$REPO/renderer" && npm ci --offline --no-audit --no-fund --cache /tmp/agentreel-npm-cache) \
        && ok "npm ci --offline from cached tarballs" || bad "npm ci --offline failed"
    elif online; then
      (cd "$REPO/renderer" && npm ci --no-audit --no-fund) && warn "npm ci from registry (no cached node_modules)" || bad "npm ci failed"
    else
      bad "renderer node_modules unavailable"
    fi
  fi
  # Chrome headless shell: prefer the copy in agentreel-bin, else `hyperframes browser ensure` downloads it
  CHT="$(bin_file 'hyperframes-chrome*.tar.gz')"
  if [ -n "$CHT" ] && [ ! -d "$HOME/.cache/hyperframes/chrome" ]; then
    mkdir -p "$HOME/.cache/hyperframes" && tar -xzf "$CHT" -C "$HOME/.cache/hyperframes"
  fi
  (cd "$REPO/renderer" && npx --no-install hyperframes browser ensure >/dev/null 2>&1) || warn "hyperframes browser ensure failed"
  CHROME="$(cd "$REPO/renderer" && npx --no-install hyperframes browser path 2>/dev/null | tail -n 1)"
  if [ -n "$CHROME" ] && [ -x "$CHROME" ]; then
    setenv HYPERFRAMES_BROWSER_PATH "$CHROME"
    MISSING="$(ldd "$CHROME" 2>/dev/null | grep 'not found' | awk '{print $1}' | tr '\n' ' ')"
    if [ -n "$MISSING" ]; then
      echo "missing Chrome libs: $MISSING"
      LIBS="libatk1.0-0t64 libatk-bridge2.0-0t64 libxcomposite1 libxkbcommon0 libgbm1 libnss3 libasound2t64 libxrandr2 libxdamage1 libcups2t64 libpango-1.0-0 libcairo2"
      if online && { $SUDO apt-get update -qq >/dev/null 2>&1; $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $LIBS >/dev/null 2>&1; }; then
        ok "apt-get installed Chrome libs"
      else
        DEBDIR="$(find "${DS_BIN:-/nonexistent}" -maxdepth 3 -type d -name debs 2>/dev/null | head -n 1)"
        if [ -n "$DEBDIR" ]; then $SUDO dpkg -i "$DEBDIR"/*.deb >/dev/null 2>&1 || true; fi
      fi
      MISSING="$(ldd "$CHROME" 2>/dev/null | grep 'not found' | awk '{print $1}' | tr '\n' ' ')"
      [ -z "$MISSING" ] && ok "Chrome shared libs resolved" || bad "Chrome libs still missing: $MISSING (use --renderer ffmpeg)"
    else
      ok "Chrome headless shell ready ($CHROME)"
    fi
  else
    bad "Chrome headless shell not found (use --renderer ffmpeg)"
  fi
  if (cd "$REPO/renderer" && npx --no-install hyperframes doctor > "$REPO/logs/hyperframes_doctor.txt" 2>&1); then
    ok "hyperframes doctor (logs/hyperframes_doctor.txt)"
  else
    warn "hyperframes doctor reported problems: see logs/hyperframes_doctor.txt"
  fi
fi

# ------------------------------------------------------------------ 6. llama-server (GPU0)
step "6. llama-server"
if [ "${SKIP_LLM:-0}" = "1" ]; then
  warn "SKIP_LLM=1: llama-server not started"
else
  load_env
  if $PY -c "import sys; sys.path.insert(0, '$REPO'); from core.servers import ensure_llm, platform_context; print(ensure_llm(platform_context(42, 'platform-setup')))"; then
    ok "llama-server healthy ($($PY -m core.servers status 2>/dev/null | head -n 1))"
  else
    bad "llama-server failed to start: see logs/servers/*.log"
  fi
fi

# ------------------------------------------------------------------ 7. smoke test
step "7. smoke test"
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  warn "SKIP_SMOKE=1"
else
  load_env
  DEFAULT_SMOKE=""
  [ "$GPUS" -ge 2 ] && [ "${SKIP_LLM:-0}" != "1" ] && DEFAULT_SMOKE="--real"
  # shellcheck disable=SC2086
  if $PY scripts/smoke_test.py ${SMOKE_ARGS:-$DEFAULT_SMOKE}; then ok "smoke test"; else bad "smoke test (see output above)"; fi
fi

# ------------------------------------------------------------------ summary
echo; echo "${B}================ AgentReel setup summary ================${N}"
for line in "${SUMMARY[@]}"; do echo "  $line"; done
echo "  env for the notebook: $ENVF"
if [ "$FAILS" -eq 0 ]; then
  echo "${G}${B}ALL GREEN${N}"; exit 0
else
  echo "${R}${B}$FAILS FAILURE(S)${N}"; exit 1
fi
