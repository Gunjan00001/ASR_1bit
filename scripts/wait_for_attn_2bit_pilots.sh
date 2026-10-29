#!/bin/bash
# One-shot A/B pilot watcher for the 2-bit attention policies.
set -uo pipefail
REPO="/home/jiit_992401210050/Desktop/ASR WaveTransformer/ASR_1bit"
cd "$REPO" || exit 1
LOG="logs/wait_attn_2bit_pilots.log"
mkdir -p logs results
exec >> "$LOG" 2>&1
PY=".venv/bin/python"
PRE="results/attn_2bit_pilot_preflight_wait.json"

preflight() {
  "$PY" scripts/check_cluster_resources.py --job-name attn_2bit_pilot \
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

run_pilot() {
  local id="$1" cfg="$2" jid short
  if [ -s "results/${id}_pilot.json" ]; then
    echo "$(date -Is) skip $id: result exists"
    return 0
  fi
  jid=$(qsub -v "CONFIG=$cfg,POLICY_ID=$id" cluster/qat_attention_mixed_pilot.pbs 2>&1) || {
    echo "$(date -Is) qsub failed: $jid"; return 1; }
  echo "$(date -Is) submitted $id: $jid"
  [[ "$jid" == *.* ]] || return 1
  short="${jid%%.*}"
  while qstat "$short" >/dev/null 2>&1; do sleep 60; done
  echo "$(date -Is) $id left the queue"
  return 0
}

echo "=== watcher start $(date -Is) ==="
while ! preflight; do
  echo "$(date -Is) waiting: no free CPU for A/B pilots"
  sleep 120
done
echo "$(date -Is) ready: A then B"
run_pilot attn_2bit_on_2bitffn_A configs/attn_2bit_on_2bitffn_A.yaml || exit 5
run_pilot attn_2bit_on_2bitffn_B configs/attn_2bit_on_2bitffn_B.yaml || exit 6
echo "=== watcher finished $(date -Is) ==="
