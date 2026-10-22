#!/usr/bin/env bash
# Build llama.cpp `llama-server` for Tesla T4 (sm_75) and package it for the agentreel-bin dataset.
# Takes ~40 min on a Kaggle 4-CPU session (CUDA kernels dominate). Run once in the Build phase, on a Kaggle
# GPU session (same CUDA/driver as the Live Run), then upload the tarball (scripts/make_kaggle_datasets.md).
#
#     bash scripts/build_llamacpp_sm75.sh                 # -> dist/agentreel-bin/llama-server-<tag>-sm75-linux-x64.tar.gz
#     LLAMA_TAG=b11793 JOBS=4 OUT=/tmp/agentreel-bin bash scripts/build_llamacpp_sm75.sh
#
# Why our own build: prebuilt CUDA tarballs ship PTX for newer archs only and JIT-compile on first start (slow,
# sometimes fails on sm_75). -DCMAKE_CUDA_ARCHITECTURES=75 gives native SASS for T4.
set -euo pipefail

TAG="${LLAMA_TAG:-b11793}"
JOBS="${JOBS:-$(nproc)}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$REPO/dist/agentreel-bin}"
WORK="${WORK:-/tmp/llama-build}"
mkdir -p "$OUT" "$WORK"

command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || { echo "nvcc not found (need the CUDA toolkit; Kaggle GPU image has it in /usr/local/cuda)"; exit 1; }
command -v cmake >/dev/null || pip install -q cmake
echo "nvcc: $(nvcc --version | tail -n 1)"

SRC="$WORK/llama.cpp"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
else
  git -C "$SRC" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" && git -C "$SRC" checkout -q "$TAG"
fi
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
echo "llama.cpp $TAG @ $COMMIT"

# Static ggml/llama libs -> one self-contained binary (only libcudart/libcublas from the system CUDA are dynamic).
cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON
time cmake --build "$SRC/build" --config Release -j "$JOBS" --target llama-server

PKG="llama-server-$TAG-sm75-linux-x64"
STAGE="$WORK/$PKG"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp "$SRC/build/bin/llama-server" "$STAGE/"
# any shared libs that were still produced (older trees ignore BUILD_SHARED_LIBS for ggml backends)
find "$SRC/build" -name '*.so*' -exec cp -P {} "$STAGE/" \; 2>/dev/null || true
cp "$SRC/LICENSE" "$STAGE/LICENSE-llama.cpp"
"$STAGE/llama-server" --version > "$STAGE/VERSION.txt" 2>&1 || true
cat > "$STAGE/BUILD_INFO.txt" <<EOF
tag=$TAG
commit=$COMMIT
cuda_arch=75
flags=-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF
nvcc=$(nvcc --version | tail -n 1)
built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
(cd "$STAGE" && sha256sum llama-server > SHA256SUMS)

tar -C "$WORK" -czf "$OUT/$PKG.tar.gz" "$PKG"
sha256sum "$OUT/$PKG.tar.gz" | tee "$OUT/$PKG.tar.gz.sha256"
echo "done: $OUT/$PKG.tar.gz  (setup_kaggle.sh extracts it to bin/llama/)"
echo "smoke: LD_LIBRARY_PATH=$STAGE $STAGE/llama-server --version"
