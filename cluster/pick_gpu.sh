#!/bin/bash
# Pick the least-busy GPU on the compute node and export CUDA_VISIBLE_DEVICES.
#
# PBS sets CUDA_VISIBLE_DEVICES to an EMPTY string (which hides every GPU), so
# the first step is always `unset`. All 8 A100s are visible to every job on
# this cluster and PBS does not pin them, so we choose explicitly by
# (utilization, memory used) instead of waiting for a "free" allocation.
set -uo pipefail

unset CUDA_VISIBLE_DEVICES

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[pick_gpu] nvidia-smi not found; leaving CUDA_VISIBLE_DEVICES unset"
    return 0 2>/dev/null || exit 0
fi

best=""
if command -v nvidia-smi >/dev/null 2>&1; then
    best=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
             --format=csv,noheader,nounits 2>/dev/null \
           | awk -F', *' '{printf "%s %s %s\n", $2+0, $3+0, $1+0}' \
           | sort -n -k1,1 -k2,2 | head -1 | awk '{print $3}')
fi

if [ -n "$best" ]; then
    export CUDA_VISIBLE_DEVICES="$best"
    echo "[pick_gpu] selected GPU $best (of 8); snapshot:"
else
    echo "[pick_gpu] WARNING: could not query GPUs; leaving all visible"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/    /'
