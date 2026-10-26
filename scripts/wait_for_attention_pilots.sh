#!/bin/bash
# Wait for the first 1-CPU/1-GPU allocation, then run both attention pilots
# sequentially. One watcher only; the preflight guard prevents duplicates.
# The pilots write unique output IDs and do not overwrite historical results.
set -uo pipefail
REPO="/home/jiit_992401210050/Desktop/ASR WaveTransformer/ASR_1bit"
cd "$REPO" || exit 1
LOG="logs/attention_pilots_watcher.log"
mkdir -p logs results
exec >> "$LOG" 2>&1

PY=".venv/bin/python"
PREFLIGHT="results/attention_pilot_preflight_wait.json"

preflight() {
  "$PY" scripts/check_cluster_resources.py \
    --job-name qat_attn_mixed_pilot --required-cpus 1 \
    --output "$PREFLIGHT" >/dev/null 2>&1
  "$PY" - "$PREFLIGHT" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if d.get("ready") else 1)
PY
}

run_pilot() {
  local id="$1" cfg="$2" jid="" short="" rc=0
  if [ -s "results/${id}_pilot.json" ]; then
    echo "$(date -Is) skip ${id}: result already exists"
    return 0
  fi
  echo "$(date -Is) submitting ${id} with ${cfg}"
  jid=$(qsub -v "CONFIG=$cfg,POLICY_ID=$id" cluster/qat_attention_mixed_pilot.pbs 2>&1) || rc=$?
  echo "qsub output: ${jid}"
  if [ "$rc" -ne 0 ] || [[ "$jid" != *.* ]]; then
    echo "$(date -Is) ERROR: qsub failed for ${id}"
    return 1
  fi
  short="${jid%%.*}"
  echo "$(date -Is) ${id} job ${jid}"
  while qstat "$short" >/dev/null 2>&1; do
    sleep 60
  done
  echo "$(date -Is) ${id} left the queue"
  return 0
}

echo "=== watcher start $(date -Is) ==="
while ! preflight; do
  echo "$(date -Is) waiting: no free CPU for 1-CPU/1-GPU pilot"
  sleep 120
done
echo "$(date -Is) ready: submitting Strategy A then B"

run_pilot attention_mixed_strategy_A configs/attention_v1_cpu16_category.yaml || exit 5
run_pilot attention_mixed_strategy_B configs/attention_v1_cpu16_sensitive24.yaml || exit 6

echo "=== watcher finished $(date -Is) ==="
