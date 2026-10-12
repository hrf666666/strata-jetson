#!/usr/bin/env bash
# strata-jetson: environment setup for Jetson Orin (JetPack 6.x, aarch64).
# Idempotent: safe to re-run. Network: huggingface via hf-mirror.com, GitHub via SSH-over-443.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/work"

echo "== 1. 功耗与频率锁定 (MAXN_SUPER + jetson_clocks) =="
sudo nvpmodel -m 0
sudo jetson_clocks --fan

echo "== 2. GitHub 直连 hosts 钉扎（可选，若 ssh.github.com 可达可跳过） =="
GIP=$(curl -s -m 8 'https://dns.alidns.com/resolve?name=github.com&type=A' | jq -r '[.Answer[]? | select(.type==1) | .data][0]')
if [ -n "$GIP" ] && ! grep -q github.com /etc/hosts; then
  echo "$GIP github.com" | sudo tee -a /etc/hosts >/dev/null
fi

echo "== 3. Python venv（cmake/ninja，aarch64 wheel 均有） =="
python3 -m venv "$WORK/svenv"
"$WORK/svenv/bin/pip" -q install cmake ninja

echo "== 4. hf 下载工具（新 CLI 叫 hf；_ENDPOINT 指向镜像） =="
PY310=${PY310:-$HOME/miniconda3/envs/py310/bin}
$PY310/pip -q install -U huggingface_hub

echo "== 5. aria2（大文件多连接下载） =="
command -v aria2c >/dev/null || sudo apt-get install -y aria2

echo "== 6. Strata 源码基线（GitHub archive 经 gh-proxy）+ 本仓库 patch 应用 =="
if [ ! -d "$WORK/Strata" ]; then
  mkdir -p "$WORK"
  curl -sL -o /tmp/strata.tar.gz https://gh-proxy.com/https://github.com/Niko1221/Strata/archive/refs/heads/main.tar.gz
  mkdir -p "$WORK/Strata" && tar xzf /tmp/strata.tar.gz -C "$WORK/Strata" --strip-components=1
  cd "$WORK/Strata" && git init -q -b main && git add -A && git -c user.email=port@local -c user.name=upstream commit -qm 'upstream snapshot'
  git apply "$ROOT"/patches/*.patch 2>/dev/null || true
  git add -A && git -c user.email=port@local -c user.name=strata-jetson-port commit -qm 'port(aarch64): Strata on Jetson'
fi

echo "== 7. 编译（CUDA sm_87） =="
cd "$WORK/Strata"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
cmake --build build -j 8
echo "DONE: $(ls build/program/ 2>/dev/null || find build -maxdepth 2 -type f -name 'strata*' | head -3)"
