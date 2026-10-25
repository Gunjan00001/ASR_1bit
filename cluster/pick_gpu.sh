#!/bin/bash
# Pick the least-busy GPU on the compute node and export CUDA_VISIBLE_DEVICES.
#
# PBS sets CUDA_VISIBLE_DEVICES to an EMPTY string (which hides every GPU), so
# the first step is always `unset`. All 8 A100s are visible to every job on
# this cluster and PBS does not pin them, so we choose explicitly by free
# memory (then utilization) instead of waiting for a "free" allocation.
#
# Two jobs submitted at the same moment would otherwise both see the same idle
# GPU before either allocates memory. A small shared reservation directory (on
# the shared home filesystem) serializes selection: each pick claims a GPU for
# 5 minutes, and a concurrent pick skips claimed GPUs. Stale claims expire.
set -uo pipefail

unset CUDA_VISIBLE_DEVICES

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[pick_gpu] nvidia-smi not found; leaving CUDA_VISIBLE_DEVICES unset"
    return 0 2>/dev/null || exit 0
fi

RES_DIR="${ONEBIT_GPU_RESERVATIONS:-$HOME/.onebit_gpu_reservations}"
mkdir -p "$RES_DIR" 2>/dev/null || true
now=$(date +%s)

# Candidates in preference order: most free memory first, then lowest
# utilization. `memory.used` is a better proxy for "not busy" than utilization,
# which fluctuates second to second.
candidates=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
                 --format=csv,noheader,nounits 2>/dev/null \
             | awk -F', *' '{printf "%s %s %s\n", $3+0, $2+0, $1+0}' \
             | sort -n -k1,1 -k2,2 | awk '{print $3}')

best=""
lock_dir="$RES_DIR/.lock"
acquired=0
for _ in $(seq 1 100); do
    if mkdir "$lock_dir" 2>/dev/null; then acquired=1; break; fi
    sleep 0.1
done

if [ "$acquired" = "1" ]; then
    # Re-check expiries while holding the lock, then choose the first
    # unreserved candidate.
    for f in "$RES_DIR"/gpu_*; do
        [ -e "$f" ] || continue
        ts=$(cat "$f" 2>/dev/null || echo 0)
        [ $((now - ts)) -gt 300 ] 2>/dev/null && rm -f "$f" 2>/dev/null || true
    done
    for idx in $candidates; do
        if [ ! -f "$RES_DIR/gpu_$idx" ]; then best="$idx"; break; fi
    done
    if [ -z "$best" ]; then
        best=$(echo "$candidates" | head -1)
    fi
    if [ -n "$best" ]; then
        echo "$now" > "$RES_DIR/gpu_$best"
    fi
    rmdir "$lock_dir" 2>/dev/null || true
else
    # Could not take the lock; fall back to the least-used GPU without a claim.
    best=$(echo "$candidates" | head -1)
    echo "[pick_gpu] WARNING: reservation lock busy; choosing GPU $best without a claim"
fi

if [ -n "$best" ]; then
    export CUDA_VISIBLE_DEVICES="$best"
    echo "[pick_gpu] selected GPU $best (reserved in $RES_DIR); snapshot:"
else
    echo "[pick_gpu] WARNING: could not query GPUs; leaving all visible"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/    /'
