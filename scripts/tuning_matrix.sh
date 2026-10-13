#!/usr/bin/env bash
# strata-jetson: the "squeeze the potential" tuning matrix on an existing zoo model.
# llama-bench sweeps: threads x flash-attn x KV-cache quantization, full cross product,
# at two prompt lengths. Produces results/raw/tuning_matrix_<tag>.json
# Usage: tuning_matrix.sh [model.gguf] (default: hunyuan-7b)
set -uo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-$R/work/zoo/gguf/hunyuan-7b.gguf}"
[ -f "$M" ] || { echo "model missing: $M"; exit 1; }
LB="$R/work/llama-build/bin/llama-bench"
TAG=$(date +%m%d-%H%M)
OUT="$R/results/raw/tuning_matrix_$TAG.json"
echo "tuning matrix on $M"
"$LB" -m "$M" -p 128,2048 -n 128 -r 3 -o json \
  -t 4,6,8 -fa 0,1 -ctk f16,q8_0 -ctv f16,q8_0 > "$OUT" 2>"$R/results/raw/tuning_matrix_$TAG.log"
RC=$?
echo "llama-bench rc=$RC -> $OUT"
# human-readable top-5 by tg128
python3 - "$OUT" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("parse fail:", e); sys.exit(0)
rows = [r for r in d if r.get("test") == "tg128"]
rows.sort(key=lambda r: -(r.get("avg_ts") or 0))
print("top-5 decode configs (tg128):")
for r in rows[:5]:
    print(f"  threads={r.get('n_threads')} fa={r.get('flash_attn')} "
          f"kv={r.get('type_k')}/{r.get('type_v')} -> {r.get('avg_ts'):.1f} tok/s")
print(f"worst: {rows[-1].get('avg_ts'):.1f} tok/s  (spread {rows[0].get('avg_ts')/max(rows[-1].get('avg_ts'),0.01):.2f}x)")
EOF
