#!/usr/bin/env bash
# strata-jetson: assemble the minimal edge deployment from a built Strata tree.
# Produces <out>/ with: engine binary, required serve modules, config + tokenizer + run script.
# Usage: edge_deploy.sh [strata_tree] [out_dir] [config_name]
set -euo pipefail
TREE="${1:-$(cd "$(dirname "$0")/../work/Strata" && pwd)}"
OUT="${2:-$(dirname "$3")/../edge-deploy}"
CFG="${3:-strata-q2_0.json}"

ENGINE="$TREE/engine-cuda12/strata"
[ -x "$ENGINE" ] || { echo "engine not built at $ENGINE"; exit 1; }

mkdir -p "$OUT/serve"
cp "$ENGINE" "$OUT/strata"
# server + its real dependencies (frontend/mcp/responses/prometheus are usable-but-optional;
# we keep server.py intact and ship them too - see docs/edge-slimming.md for the measured costs)
for f in server.py frontend.py mcp.py responses.py structured.py runconfig.py \
         logit_bias.py winjob.py session_save_retry.py telemetry.py keepawake.py \
         cache_drop.py cpu_affinity.py __init__.py chat_template.jinja; do
  [ -f "$TREE/serve/$f" ] && cp "$TREE/serve/$f" "$OUT/serve/"
done
cp "$TREE/$CFG" "$OUT/" 2>/dev/null || true
# tokenizer + expert profile live under the pack dir referenced by the config; copy by pointer
PACK=$(jq -r '.args[.args.index("--pack")+1] // empty' "$OUT/$CFG" 2>/dev/null || true)
if [ -n "$PACK" ] && [ -d "$PACK" ]; then
  mkdir -p "$OUT/pack-tokenizer"
  cp -r "$PACK/tokenizer" "$OUT/pack-tokenizer/" 2>/dev/null || true
fi
cat > "$OUT/README.txt" <<EOF
strata-jetson edge deployment
  engine:   ./strata ($(du -m "$ENGINE" | cut -f1) MB, sm_87)
  start:    python3 -m serve.server --engine ./strata --config $CFG --port 8080
  requires: python3.10+, numpy/jinja2/regex/pyyaml/tqdm/requests/pillow/psutil
  models:   the config references the GGUF shards + pack on the local disk
EOF
du -sh "$OUT"
echo "edge deployment assembled at $OUT"
