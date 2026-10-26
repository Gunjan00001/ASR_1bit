#!/bin/bash
# One-shot wait for a 1-CPU/1-GPU slot, then run the 2-bit full job and final
# test evaluation. The pilot (12923) already passed; no duplicate is submitted.
set -uo pipefail
REPO="/home/jiit_992401210050/Desktop/ASR WaveTransformer/ASR_1bit"
cd "$REPO" || exit 1
LOG="logs/wait_ffn_2bit_full.log"
mkdir -p logs results
exec >> "$LOG" 2>&1
PY=".venv/bin/python"
PRE="results/qat_ffn_2bit_full_preflight_wait.json"

preflight() {
  "$PY" scripts/check_cluster_resources.py --job-name qat_ffn_2bit_full \
    --required-cpus 1 --output "$PRE" >/dev/null 2>&1
  "$PY" - "$PRE" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if d.get("ready") else 1)
PY
}

run_job() {
  local script="$1" label="$2" jid short
  jid=$(qsub "$script" 2>&1) || { echo "$(date -Is) qsub failed: $jid"; return 1; }
  echo "$(date -Is) submitted $label: $jid"
  [[ "$jid" == *.* ]] || { echo "$(date -Is) invalid qsub output"; return 1; }
  short="${jid%%.*}"
  while qstat "$short" >/dev/null 2>&1; do sleep 60; done
  echo "$(date -Is) $label left the queue"
  return 0
}

echo "=== watcher start $(date -Is) ==="
while ! preflight; do
  echo "$(date -Is) waiting: no free CPU for 2-bit full"
  sleep 120
done
echo "$(date -Is) ready: submitting full 2-bit"
run_job cluster/qat_ffn_2bit_full.pbs qat_ffn_2bit_full || exit 5
echo "$(date -Is) full finished; submitting final eval"
run_job cluster/final_eval_2bit_full.pbs final_eval_2bit_full || exit 6
echo "=== watcher finished $(date -Is) ==="
