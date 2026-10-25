# Cluster execution — PBS + 8×A100-80GB

This directory contains the **reproducible execution layer** for Phases 11–19 on
the local GPU cluster. It complements `kaggle/` (cloud T4) with a second
compute profile; the scientific configs in `configs/` stay pristine and
hardware-specific settings are applied through generated `effective` configs.

Cluster facts used here are documented in the workspace-level `gpu.md`:
8 × NVIDIA A100-SXM4-80GB on one GPU node, PBS scheduler, driver 525 with CUDA
12.6 PyTorch builds, shared network home, per-user 16-CPU quota.

## Environment

The project runs in `ASR_1bit/.venv` (created with `--system-site-packages`), so
it inherits the cluster's CUDA-enabled stack:

```
torch 2.11.0+cu126   transformers 4.57.6   datasets 5.0.1
torchvision 0.26.0+cu126   numpy 2.5.3   jiwer 4.0.0
```

This differs from the pinned Kaggle stack in `requirements.txt` (torch 2.14,
transformers 5.17). Every new results JSON records the actual versions, and the
old Kaggle results are kept separate — no cross-profile extrapolation.

Audio decoding uses `ONEBIT_AUDIO_DECODER=soundfile` because `torchcodec` is not
installed on the cluster; the code path is the same one the Kaggle runner falls
back to.

## GPU selection (`pick_gpu.sh`)

PBS sets `CUDA_VISIBLE_DEVICES` to an **empty string**, hiding every GPU, and it
does not pin GPUs to a job. `pick_gpu.sh`:

1. unsets `CUDA_VISIBLE_DEVICES`;
2. queries all 8 GPUs and orders them by **free memory first**, then utilization;
3. claims the chosen GPU for 5 minutes in `~/.onebit_gpu_reservations` (an
   atomic `mkdir` lock serializes concurrent picks);
4. exports the chosen index before the Python process starts.

The reservation prevents two jobs submitted at the same instant from both
choosing the same idle A100 before either has allocated memory. Claims expire
after 5 minutes, and a fallback picks the least-used GPU if the lock is busy.

This is why a job can start immediately even when some A100s are busy; it never
waits for a specific device. It does not migrate a job that starts on a GPU
which later becomes busy.

## Jobs

| PBS script | Purpose | CPUs | Output |
|---|---|---:|---|
| `qat_attn_ffn.pbs` | Phase 11: attention+FFN QAT (the decisive experiment) | 8 | `results/qat_attn_ffn.json`, `checkpoints/qat_attn_ffn*` |
| `qat_attn_a100.pbs` | Phase 12/14: attention-only QAT re-run for a like-for-like A100 arm | 4 | `results/qat_attn_a100.json`, `checkpoints/qat_attn_a100*` |
| `qat_ffn_a100.pbs` | Phase 15: FFN-only QAT sensitivity arm | 1 | `results/qat_ffn_a100.json`, `checkpoints/qat_ffn_a100*` |
| `distill_attn_ffn.pbs` | Phase 13: FP32 teacher → attention+FFN 1-bit student | 1 | `results/distill_attn_ffn.json`, `checkpoints/distill_attn_ffn*` |
| `distill_attn_ffn_kd01.pbs` | Phase 13/14: λ_kd=0.1 distillation follow-up | 1 | `results/distill_attn_ffn_kd01.json` |
| `screen_ffn_sensitivity.pbs` | research: one-layer FFN sensitivity screen (no training) | 4 | `results/layer_sensitivity_ffn.json` |
| `qat_ffn_pilot.pbs` | research: 500-step FFN control vs mixed-precision pilot | 4 | `results/qat_ffn_control_pilot.json`, `results/qat_ffn_mixed_pilot.json` |
| `final_eval_a100.pbs` | Phase 12/14/15/17/18/19: final comparison, packed and streaming benchmarks | 1 | `results/final_comparison_*.{json,md}`, `results/packed_inference_benchmark.json`, `results/streaming_demo_qat_attn_ffn.json` |

> **Node allocation:** `jiit-gpu01` has 96 CPUs and 8 A100s, but PBS assigns
> whole CPUs and GPU slots. When `resources_assigned.ncpus = 96`, new jobs stay
> queued even if `nvidia-smi` shows free GPU memory. The mixed-precision pilot
> was queued and canceled for this reason; rerun `qat_ffn_pilot.pbs` when a slot
> frees.

Each job writes a live log under `logs/` on the shared home filesystem (PBS
buffers its own `-o` output, so `tee` is used instead), runs `pack_qat_model.py`
after training, and records the measured artifact size.

Submit and monitor:

```bash
qsub cluster/qat_attn_ffn.pbs
qstat -u "$USER"
tail -f logs/qat_attn_ffn.log
```

## Final evaluation / benchmark

Once checkpoints exist, run Phase 12/14/15/17 in one process:

```bash
# dev-clean, 256 utterances, all available arms
.venv/bin/python scripts/run_final_eval.py --split dev --n 256

# test-clean subset
.venv/bin/python scripts/run_final_eval.py --split test --n 256
```

This writes per-arm JSON plus `results/final_comparison_<split>_n<N>.{json,md}`.
Missing checkpoints are skipped and recorded explicitly, so the script can run
before every job finishes.

Phase 18 (packed inference / binary-compute boundary):

```bash
.venv/bin/python scripts/benchmark_packed.py \
    --config configs/qat_attn_ffn.yaml --checkpoint checkpoints/qat_attn_ffn --n 4
```

Phase 19 (chunked streaming demo):

```bash
.venv/bin/python scripts/stream_infer.py --sample-index 0 --chunk 2.0 --overlap 0.5
```

## Reproducibility notes

- Generated `configs/cluster_effective_*.yaml` files are git-ignored; the
  pristine experiment definitions are committed.
- `checkpoints/` and `logs/` are git-ignored; committed `results/*.json` files
  are the durable measurements.
- The PBS walltime is 5–6 h per job; the attention+FFN run extrapolates to
  roughly 1.5–2 h on an A100 (the T4 probe was 4.8 h).
