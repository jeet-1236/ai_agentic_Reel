# Creating the private Kaggle Datasets (Build phase, once; new version on every change)

Four **private** datasets are attached to `notebooks/kaggle_live.ipynb`. They let the Live Run start without
PyPI/npm/HF downloads, and they pin every binary and weight file.

| Dataset | Contents | Built by | Approx. size |
|---|---|---|---|
| `agentreel-models` | GGUFs (Qwen3.5-9B Q8_0, Qwen3-VL-8B Q4_K_M + mmproj), SDXL base fp16 parts + Lightning UNet + fp16-fix VAE, Kokoro-82M, faster-whisper turbo, Qwen3 embedding/reranker, HHEM + flan-t5 tokenizer, Falconsai NSFW, YuNet, Real-ESRGAN x2 | `scripts/prefetch_models.py` | ~32 GB |
| `agentreel-bin` | `llama-server-b11793-sm75-linux-x64.tar.gz`, static ffmpeg, Node 22 tarball, `renderer_node_modules.tar.gz`, `hyperframes-chrome.tar.gz`, `debs/`, CPython 3.12 standalone | `scripts/build_llamacpp_sm75.sh` + commands below | ~1.5 GB |
| `agentreel-wheels` | `py313/` wheels for `requirements.txt`, `py312/` wheels for the Kokoro venv (incl. torch cu128 + `en_core_web_sm`) | commands below | ~4 GB |
| `agentreel-fonts` | Fira Sans + Noto Sans `.ttf` + OFL texts | `prefetch_models.py --fonts` | ~3 MB |

`setup_kaggle.sh` finds them by **name prefix** anywhere under `/kaggle/input` (depth 5). Kaggle changed mount
paths in Oct 2026, so nothing hard-codes `/kaggle/input/<slug>`. The file names below are the globs it looks for.

## 0. One-time: Kaggle CLI + credentials

```bash
pip install kaggle==1.7.4.5           # any 1.7.x works; pin what you used in requirements-dev notes
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json   # Account -> API -> Create token
export KAGGLE_USERNAME=<your-username>
kaggle datasets list --mine           # sanity check
```

To upload from inside a Kaggle notebook instead: store `KAGGLE_USERNAME` and `KAGGLE_KEY` as **Kaggle Secrets**
and export them in a cell. Build big folders under `/tmp`, because `/kaggle/working` is capped at 20 GB
(check free space with `df -h /tmp`).

## 1. Order of creation

1. **agentreel-bin**, built on a **Kaggle GPU session** so CUDA and the driver match the Live Run
2. **agentreel-wheels**, built on a Kaggle session (same Python 3.13 / glibc as the Live Run)
3. **agentreel-fonts** (tiny)
4. **agentreel-models** (largest; can upload in the background while you test 1-3)
5. Attach all four to `notebooks/kaggle_live.ipynb` (Add Input -> Your Datasets), run cell 2, and it must end **ALL GREEN**
6. Copy `resolved_revision` + `sha256` from `config/models.lock.yaml` into `config/models.yaml`, commit, tag `v1.0-build`

## 2. agentreel-bin

```bash
OUT=/tmp/agentreel-datasets/agentreel-bin
mkdir -p $OUT/debs
# (a) llama.cpp for sm_75, ~40 min
OUT=$OUT bash scripts/build_llamacpp_sm75.sh
# (b) static ffmpeg (BtbN GPL build: libx264 + libass) as a fallback for the system ffmpeg
curl -L -o $OUT/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz \
  https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz
# (c) Node 22 LTS tarball
curl -L -o $OUT/node-v22.20.0-linux-x64.tar.xz https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz
# (d) renderer node_modules + HyperFrames' Chrome headless shell (Linux build: do this ON Kaggle/Linux)
tar -xJf $OUT/node-v22.20.0-linux-x64.tar.xz -C /tmp && export PATH=/tmp/node-v22.20.0-linux-x64/bin:$PATH
(cd renderer && npm ci --no-audit --no-fund && npx hyperframes browser ensure)
tar -czf $OUT/renderer_node_modules.tar.gz -C renderer node_modules
tar -czf $OUT/hyperframes-chrome.tar.gz -C ~/.cache/hyperframes chrome
# (e) Chrome system libs as .deb (only used if runtime apt-get is blocked)
cd $OUT/debs && apt-get download libatk1.0-0t64 libatk-bridge2.0-0t64 libxcomposite1 libxkbcommon0 libgbm1 \
  libnss3 libasound2t64 libxrandr2 libxdamage1 libcups2t64 libpango-1.0-0 libcairo2 && cd -
# (f) CPython 3.12 for the Kokoro venv: python-build-standalone "install_only" build (unpacks to python/bin/python3.12).
#     Pick the newest cpython-3.12.*-x86_64-unknown-linux-gnu-install_only.tar.gz asset from
#     https://github.com/astral-sh/python-build-standalone/releases and keep its file name.
curl -L -o $OUT/<asset-name>.tar.gz https://github.com/astral-sh/python-build-standalone/releases/download/<tag>/<asset-name>.tar.gz
#     Without it, setup falls back to `uv venv --python 3.12` (downloads; Internet ON).
sha256sum $OUT/* > $OUT/SHA256SUMS.txt
python scripts/prefetch_models.py --out /tmp/agentreel-datasets --only none --no-lock   # writes dataset-metadata.json files
kaggle datasets create -p $OUT --dir-mode tar      # first time
# later: kaggle datasets version -p $OUT -m "llama b11793 + node 22.20" --dir-mode tar
```

## 3. agentreel-wheels

Run on Kaggle (CPU session is fine) so platform tags match:

```bash
OUT=/tmp/agentreel-datasets/agentreel-wheels
mkdir -p $OUT/py313 $OUT/py312
# system env (3.13): everything in requirements.txt (+ requirements/*.txt if present)
pip download -d $OUT/py313 -r requirements.txt $(for f in requirements/*.txt; do case $f in *voice312*) ;; *) echo -r $f;; esac; done)
# voice env (3.12): kokoro + misaki + torch cu128 (+ spaCy English model that misaki loads)
pip download -d $OUT/py312 --python-version 3.12 --only-binary=:all: --platform manylinux_2_28_x86_64 \
  --platform manylinux2014_x86_64 --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-voice312.txt          # or: kokoro==0.9.4 "misaki[en]==0.9.4" soundfile==0.13.1 numpy==2.2.6 torch==2.8.0
curl -L -o $OUT/py312/en_core_web_sm-3.8.0-py3-none-any.whl \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
# fallback TTS on 3.13 (no venv): kokoro-onnx
pip download -d $OUT/py313 kokoro-onnx==0.6.1
kaggle datasets create -p $OUT --dir-mode tar
```

If a source-only package has no wheel for the platform, `pip wheel` it on Kaggle and put the `.whl` in the same folder.

## 4. agentreel-fonts and agentreel-models

```bash
export KAGGLE_USERNAME=<your-username> HF_TOKEN=<read token, only for gated repos>
python scripts/prefetch_models.py --dry-run                     # review the list first
python scripts/prefetch_models.py --out /tmp/agentreel-datasets --fonts
#   -> /tmp/agentreel-datasets/agentreel-models/<org>__<name>/...   + config/models.lock.yaml
#   -> /tmp/agentreel-datasets/agentreel-fonts/*.ttf
# If downloads stall on the Xet backend the script retries with HF_HUB_DISABLE_XET=1 automatically.
kaggle datasets create -p /tmp/agentreel-datasets/agentreel-fonts
kaggle datasets create -p /tmp/agentreel-datasets/agentreel-models --dir-mode skip
#   --dir-mode skip uploads files individually (keeps <org>__<name>/ folders browsable; large GGUFs upload resumably)
kaggle datasets status $KAGGLE_USERNAME/agentreel-models       # wait for "ready" (processing can take 30+ min)
```

Alternative for the GGUFs: upload them as a private **Kaggle Model** and attach it. `core/servers.py` finds the
`.gguf` by file name anywhere under `/kaggle/input`, so either works.

## 5. Keep it private and versioned

- Every dataset is created with `"isPrivate": true` (in `dataset-metadata.json`). Share each one with your teammate's
  Kaggle account (Dataset -> Settings -> Sharing). The Kaggle Terms forbid sharing accounts.
- New file = `kaggle datasets version -p <dir> -m "<what changed>"`. Note the version number in `README.md`.
- Pin the dataset version in the notebook (Input -> version dropdown) before tagging `v1.0-build`.

## 6. Secrets (never in the repo or in a dataset)

Notebook -> Add-ons -> Secrets: `TAVILY_API_KEY` (required), `HF_TOKEN` (optional), `GITHUB_TOKEN` (only if the
repo is private and cloned in cell 1). Enable each secret for this notebook, on **both** team accounts.
