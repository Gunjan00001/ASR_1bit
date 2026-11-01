#!/bin/bash
set -uo pipefail
REPO="/home/jiit_992401210050/Desktop/ASR WaveTransformer/ASR_1bit"; cd "$REPO" || exit 1
LOG="logs/wait_attn_2bit_A_full.log"; mkdir -p logs results; exec >> "$LOG" 2>&1
PY=".venv/bin/python"; PRE="results/attn_2bit_A_full_preflight_wait.json"
ready(){ "$PY" scripts/check_cluster_resources.py --job-name attn_2bit_A_full --required-cpus 1 --output "$PRE" >/dev/null 2>&1; "$PY" - "$PRE" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
sys.exit(0 if d.get("ready") else 1)
PY
}
run(){ local jid short; jid=$(qsub "$1" 2>&1) || { echo "$(date -Is) qsub failed $jid"; return 1; }; echo "$(date -Is) submitted $1: $jid"; [[ "$jid" == *.* ]] || return 1; short=${jid%%.*}; while qstat "$short" >/dev/null 2>&1; do sleep 60; done; echo "$(date -Is) $1 left queue"; }
echo "=== watcher start $(date -Is) ==="
while ! ready; do echo "$(date -Is) waiting: no free CPU"; sleep 120; done
echo "$(date -Is) ready"; run cluster/attn_2bit_A_full.pbs || exit 5
run cluster/final_eval_attn_2bit_A_full.pbs || exit 6
echo "=== watcher finished $(date -Is) ==="
