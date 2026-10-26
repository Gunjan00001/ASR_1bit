# Changelog

All notable checkpoints of this repository. Versions are git tags `vX.Y.Z`
(`major.minor.patch`); see the versioning section of `README.md`.

The repository is maintained as uploaded code snapshots; each entry describes
an exact, reproducible repository state.

---

## Unreleased — preliminary attention sensitivity screen + policies

- Added `weight_reconstruction_error`, 4 tests, and a 3-test attention-policy
  config generator; **129 tests** pass.
- Ran a **preliminary CPU screen on 16 fixed dev utterances** (FP32 ref WER
  0.0256); all 96 attention layers measured with WER/CER, logit/KL, argmax, and
  weight reconstruction error. `attention_output` is least sensitive
  (mean ΔWER +0.0047); `attention_k` is most sensitive (+0.0069).
- Generated two configs without changing the 17 FFN exclusions:
  `attention_v1_cpu16_category` (24 `attention_output` layers quantized;
  estimated 0.992 GB) and `attention_v1_cpu16_sensitive24` (top-24 rel-L2
  attention FP32, 72 quantized; estimated 0.797 GB).
- The comparable 2k/500 pilot job is prepared (`cluster/qat_attention_mixed_pilot.pbs`).
  It was **not submitted**: 0 free PBS CPUs (8 GPUs free). No attention QAT/WER
  result exists; v1.6.0 results remain unchanged.

---

## v1.6.0 — Mixed-precision FFN full comparison (2026-10-26)

**Scope:** the full 10,000-utterance / 1,875-step controlled comparison and the
predefined test evaluation are complete. The 17-layer FP32 exclusion list was
fixed before the run and was not tuned. No further test tuning was performed.

### Added
- `per_output_channel_mean_abs` scaling (`alpha_i = mean_j |W[i,j]|`), with
  broadcasting, zero-row safety, gradient, and packed-artifact tests.
- `scripts/analyze_quantization_error.py` — per-tensor vs per-channel FFN
  reconstruction screen and gate.
- `scripts/screen_layer_sensitivity.py` — one-layer-at-a-time PTQ screen with
  logit/KL/argmax/WER metrics; explicitly `screening_only`, `not_qat`.
- Per-layer FP32/1-bit mixed precision: `exclude_layers` in `replace_linears`,
  QAT setup, manifests, loading, final evaluation, packing, and the artifact
  verifier.
- `configs/qat_ffn_per_channel.yaml` (screened out) and
  `configs/qat_ffn_mixed.yaml` (17 sensitive FFN layers FP32, 79 quantized).
- `src/onebit_asr/training/callbacks.py` — `DevWERCallback` for validation-based
  best-checkpoint selection (history, best step, restore).
- Artifact sidecar manifest + `scripts/verify_packed_artifact.py`; the exported
  model is now reloadable and numerically verifiable (max logit diff 0 on smoke).
- Checkpoint manifests now record `layers`, `exclude_layers`, `scale_mode`,
  `ste_clip`, granularity, config path, and init model; loaders verify them.
- `cluster/screen_ffn_sensitivity.pbs`, `cluster/qat_ffn_pilot.pbs`,
  `cluster/qat_ffn_full.pbs`, `cluster/final_eval_full_arms.pbs`.
- 19 new tests (126 total), all passing; includes resume, PBS preflight/job checks, and the sensitivity helper.

### Screened / measured
- Per-channel reconstruction gate: FFN-only QAT masters 0.6458→0.6306 (**2.28%**),
  FP32 0.6508→0.6343 (**2.43%**), gate 10% → **FAIL**. No per-channel QAT run.
- FFN layer screen (32 dev utts, FP32 ref WER 0.0287): `ffn_in` mean ΔWER
  **0.00342**, `ffn_out` **0.00183** (1.9×); top layers are middle/late (18, 17,
  19, 11, 8, …), selected from measurements rather than position.
- Mixed-precision smoke: 79/79 BitLinear gradients, 17 excluded FP32 layers,
  save/load verified, artifact reload **max |Δ logit| 0.0, argmax 1.0**.
- Mixed artifact: 79 BitLinear layers + 17 FP32 FFN layers; effective
  ~1.00 bit/quantized weight, ~14.7 bits/model parameter.
- **Controlled pilot (job 12772, 2026-10-26; 2k/500 steps; dev-128; same seed,
  train/eval IDs, batch, and LR):** FFN-only control **0.2303 WER / 0.8133 GB**;
  mixed FFN **0.0900 WER / 1.0896 GB**; mixed minus control **−14.04 pp**.
  Both checkpoints save/load and both packed artifacts reload with **max
  |Δ logit| 0, argmax 1.0, decoded 2/2**. This is a pilot, not a full-test
  result.

### Full-run hardening (prepared, no full WER yet)
- `DevWERCallback` now evaluates the final training step, persists the best-dev
  state/history for resume, and restores the selected state before evaluation.
- `training/resume.py` + `train_qat.py --resume` use the latest numeric Trainer
  checkpoint; output/checkpoint paths refuse silent overwrite.
- Full job `cluster/qat_ffn_full.pbs`: 10k utterances, 1,875 steps, seed 42,
  256-dev selection, fixed 17-layer exclusion list, unique `*_full` outputs,
  per-arm skip/resume, packed-artifact verification.
- Read-only PBS preflight `scripts/check_cluster_resources.py`; it reported
  ready at 2 free CPUs / 8 GPUs on 2026-10-26. The full run was later submitted
  as job `12848` (1 CPU) and completed; see **Full-run results**.
- 19 new tests; **126 total**, all passing.
- The full-run and final-test evaluation are complete; see **Full-run results**
  below. The pilot was only a pilot.

### Full-run results
| Arm | Dev-256 | Test-256 | Full test-clean | Artifact |
|---|---:|---:|---:|---:|
| FFN-only control (96 BitLinear) | 0.1163 | 0.1124 | 0.1132 | 813,346,080 B |
| Mixed FFN (79 BitLinear + 17 fixed FP32) | **0.0527** | **0.0516** | **0.0546** | **1,089,643,604 B** |

Full-test delta **−5.86 pp**; the mixed arm is **2.18× smaller than FP32** and
the control 2.92×. This is a new Pareto point, not a new absolute-WER best;
attention-only remains 4.49% at 1.983 GB. Selected checkpoints were step 1,875
and restored; save/load and packed-artifact reload passed.

### Blocked then resolved
- The 2026-10-25 pilot attempt was queued and canceled because the GPU node had
  all 96 CPUs assigned and no free slot. The 2026-10-26 rerun (job `12772`)
  completed on a free A100. Outputs: `results/qat_ffn_{control,mixed}_pilot.json`,
  `results/qat_ffn_{control,mixed}_pilot_size_report.json`, and
  `results/packed_verify_ffn_{control,mixed}_pilot.json`.
- The first full submission (job `12823`) aborted at the CUDA preflight due to
  `torch.device_count()`; fixed to `torch.cuda.device_count()` with a static
  regression test. The full run then completed as job `12848`, and the final
  test as job `12899`. No historical result was overwritten.

---

## v1.5.2 — Final status closure

**Scope:** status/documentation only. **No measurements changed.**

### Added
- README §32 “Completion status”: phase-by-phase audit, explicit
  future/optional items, and the final full test-clean WER table.
- `planning.md` §10.10 completion audit; §10.9 reframed as future/optional work.
- `EXPERIMENT_LOG.md` E23: project completion audit, blocked Kaggle re-run, and
  the final numbers.

### Changed
- README status row for the Kaggle clean reproduction is now “superseded by the
  A100 re-run + packing” (Kaggle CLI/credentials unavailable; quota exhausted
  historically), instead of “in progress”.
- `pyproject.toml` — version 1.5.1 → 1.5.2.

### Final WER (full test-clean, 2620)
FP32 **0.0193**; attention-only QAT **0.0449** at 1.983 GB (best 1-bit);
FFN-only QAT **0.1134** at 0.813 GB; attention+FFN QAT **0.2050** at 0.423 GB;
attention+FFN + KD λ_kd=0.1 **0.1998** at 0.423 GB.

---

## v1.5.1 — Documentation: deep-dive, experiment log, figures, debugging comments

**Scope:** documentation and code-comment checkpoint. **No new measurements**;
all numbers are the v1.5.0 artifacts. This release exists so a new engineer can
learn the project and debug it without reverse-engineering the history.

### Added
- `deepdive.md` — a from-scratch teaching guide: ASR/CTC, the Conformer
  parameter map, the 1-bit math (including the `alpha + 1/N` gradient), BitLinear
  line by line, layer selection, PTQ vs QAT, distillation, packing, benchmarking,
  streaming, reproducibility, a debugging cookbook, glossary, and exercises.
- `EXPERIMENT_LOG.md` — every experiment in the requested format: what we did,
  why, did it work, why not, what we changed instead, and what worked and why.
  Includes the infrastructure failure log and the decision log.
- `docs/figures/` — eight figures generated from the committed JSONs:
  accuracy vs deploy size, WER by arm, PTQ vs QAT, training curves, artifact
  composition, A100 RTF/latency, streaming WER, and the parameter map.
- `scripts/make_figures.py` — deterministic figure generator (reads `results/`,
  writes `docs/figures/`; no hand-entered numbers).
- README: “Documentation map”, “Results at a glance” (figures + captions),
  “What worked / what didn't”, and the parameter-map figure.

### Changed
- Debugging notes and invariants added throughout:
  `quantization/{ste,scaling,binarization,bitlinear,packed}.py`,
  `models/{conformer_utils,replace_layers}.py`,
  `training/{qat,distillation}.py`,
  `evaluation/{evaluate,latency,model_size}.py`, `streaming.py`, and the
  `train_qat`, `train_distillation`, and `run_final_eval` entry points.
- `pyproject.toml` — version 1.5.0 → 1.5.1.
- README test count / layout updated for 88 tests and the `docs/` tree.

### Notes
- The figures are derived from `results/*.json`; if a result changes, rerun
  `scripts/make_figures.py` and `scripts/summarize_results.py`.
- No result was regenerated or overwritten in this checkpoint.

---

## v1.5.0 — Phase 12–19: attention+FFN QAT, distillation, sensitivity, benchmarks

**Scope:** the full post-Phase-11 research program on the local 8×A100-80GB
PBS cluster, plus the execution/benchmarking/streaming infrastructure it
required. Every number below is a real measurement stored in `results/`; the
complete generated tables are in [`RESULTS.md`](RESULTS.md). The λ_kd=0.1
distillation follow-up completed at **0.2022 dev / 0.1998 full test-clean**, a
small improvement over QAT alone (0.2080 / 0.2050).

### Measured — 1-bit accuracy/size trade-off (A100, pinned protocol)

Fixed 256-utterance greedy-CTC subsets, seed 42. "Deploy size" is the measured
packed safetensors artifact (packed weights + FP32 scales + non-quantized FP
parameters), not a theoretical 1-bit payload.

| Arm | Dev WER | Dev CER | Test WER | Test CER | Deploy size |
|---|---:|---:|---:|---:|---:|
| FP32 baseline | 0.0202 | 0.0049 | 0.0162 | 0.0044 | 2.374 GB |
| PTQ attention-only | 0.4454 | 0.2180 | — | — | 1.983 GB |
| PTQ FFN-only | 1.0000 | 1.0000 | — | — | 0.813 GB |
| PTQ attention+FFN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.423 GB |
| PTQ all linears | 1.0000 | 1.0000 | — | — | 0.421 GB |
| **QAT attention-only** | **0.0421** | **0.0128** | **0.0439** | **0.0131** | 1.983 GB |
| **QAT FFN-only** | **0.1190** | **0.0338** | **0.1095** | **0.0318** | 0.813 GB |
| **QAT attention+FFN** | **0.2080** | **0.0640** | **0.2029** | **0.0648** | 0.423 GB |
| QAT + KD (λ_kd=0.5) | 0.2232 | 0.0712 | 0.2167 | 0.0706 | 0.423 GB |
| QAT + KD (λ_kd=0.1) | **0.2022** | **0.0641** | **0.1986** | **0.0647** | 0.423 GB |

**Full test-clean (2620 utterances), accuracy-only:** FP32 **0.0193 / 0.0050**;
attention-only **0.0449 / 0.0137**; FFN-only **0.1134 / 0.0325**;
attention+FFN **0.2050 / 0.0639**; KD λ_kd=0.5 **0.2218 / 0.0709**; KD λ_kd=0.1
**0.1998 / 0.0636**. The ranking is identical to the fixed 256-utterance
subsets; the subset numbers are slightly easier.

**Interpretation.** QAT recovers attention-only from 44.5% PTQ to 4.21% and
FFN-only from 100% to 11.90%. The combined attention+FFN arm recovers to 20.80%,
**worse than either isolated arm** — the two binarizations interact. The 5.6×
compression target is reachable but costs ~10× the FP32 WER on this recipe.
Distillation at λ_kd=0.5 did not help: the summed KL term dominated CTC by ~8×
(6107 vs 726). Lowering λ_kd to 0.1 recovers a small, consistent gain
(**20.22%**, −0.60 pp vs QAT alone) at the same artifact size; the teacher
signal remains a marginal knob, not a solution to the interaction.

### Measured — performance (A100, warmed-up benchmark)

- All arms: RTF ≈0.006–0.007 end-to-end, p50 ≈0.035–0.045 s, peak VRAM ≈4.4 GB
  (dev) / ≈6.0 GB (test). **Binarization does not accelerate inference** — the
  runtime expands `alpha·sign(W)` before `F.linear`; the difference between arms
  is <10% and tracks model size, not precision.
- Phase 18 packed path: 192 packed layers, 62.9 MB packed weights; BitLinear
  forward 1.657 s vs PackedBitLinear 1.826 s on CPU (**1.10× slower**). The
  XNOR+popcount reference is **2.3× faster** than a float matmul and exactly
  matches the ±1 dot product, but it requires 1-bit activations (not this model).
- Phase 19 streaming demo: 4.45 s sample, 2 s chunks / 0.5 s overlap, 3 chunks;
  chunked WER **0.6923** vs full-utterance **0.1538**, chunked RTF 0.175. This
  is pseudo-streaming, not a true streaming encoder.

### Measured — PTQ layer sensitivity (256 dev utterances)

| PTQ selection | WER | CER | Deploy size |
|---|---:|---:|---:|
| attention-only | 0.4454 | 0.2180 | 1.983 GB |
| FFN-only | 1.0000 | 1.0000 | 0.813 GB |
| attention+FFN | 1.0000 | 1.0000 | 0.423 GB |
| + feature_projection | 1.0000 | 1.0000 | 0.421 GB |
| + ctc_head (all linears) | 1.0000 | 1.0000 | 0.421 GB |

### Fixed (discovered during the A100 work)
- **Final-eval crash on an in-progress checkpoint directory.** `Trainer` creates
  `output_dir` at job start, so an `is_dir()` check let a concurrently training
  arm through and `from_pretrained` raised `OSError`, aborting the whole
  comparison. Fixed by requiring `model.safetensors`/`pytorch_model.bin` and
  catching per-arm errors so one bad arm cannot abort the table.
- **Distillation probe device mismatch.** The probe moved the batch to CUDA while
  the student was still on CPU. Fixed by deriving the device from the model and
  moving the student before the probe (same change in `train_qat.py`, which had
  silently run the probe on CPU).
- **Save/load false negatives.** The 1e-2 logit tolerance rejected functionally
  identical checkpoints (argmax agreement 1.000, max |Δ logit| 0.028–0.063).
  Criterion is now `atol/rtol=1e-1` + argmax agreement ≥0.99; the historical
  JSONs keep their boolean and record the measured agreement.
- **GPU selection race.** Two jobs submitted together both chose the same idle
  A100. `cluster/pick_gpu.sh` now orders by free memory, takes an atomic lock,
  and claims the GPU for 5 minutes in `~/.onebit_gpu_reservations`.
- **CPU quota was the real scheduler bottleneck,** not GPUs. The distillation and
  FFN-only jobs were resized to 1 CPU each so they could run concurrently with
  the attention+FFN job; the qat_ffn arm still completed 1875 steps in 50:48.

### Added — execution
- `cluster/` — PBS execution layer for the local 8×A100-80GB node:
  `pick_gpu.sh` (unsets the empty `CUDA_VISIBLE_DEVICES` and selects the
  least-busy GPU), `qat_attn_ffn.pbs`, `qat_attn_a100.pbs`,
  `distill_attn_ffn.pbs`, and `cluster/README.md`.
- `configs/qat_attn_ffn.yaml` — attention+FFN QAT (the configuration that
  collapsed under PTQ).
- `configs/distill_attn_ffn.yaml` — distillation spec (teacher, λ weights,
  temperature, teacher-logits cache).

### Added — Phase 13 (knowledge distillation)
- `src/onebit_asr/training/distillation.py` — frame-masked
  `λ_ctc·CTC + λ_kd·T²·KL(student/T ‖ teacher/T)` (KL **summed** over valid
  frames so λ weights are comparable to the summed CTC loss), one-shot teacher
  logits precomputation, cache load/save, `DistillationDataset`,
  `DistillationCollator`, `DistillationTrainer`.
- `scripts/train_distillation.py` — smoke/probe/full modes, checkpointing,
  save/load verification, pinned dev evaluation.
- `results/distillation_smoke.json` — CPU pipeline check (8 steps); WER 1.0000
  is reported as measured and is **not** an accuracy experiment.

### Added — Phase 12/14/15/16/17
- `src/onebit_asr/data/dataset.py` — `load_test_subset` (test-clean, same
  fixed-seed selection rule).
- `src/onebit_asr/evaluation/latency.py` — warmup + synchronized latency,
  p50/p95, RTF (end-to-end and model-only), throughput, peak RSS, CUDA peak
  allocated/reserved.
- `src/onebit_asr/evaluation/benchmark.py` — accuracy+performance orchestration
  and a Markdown comparison table.
- `scripts/run_final_eval.py` — evaluates/benchmarks FP32, four PTQ sensitivity
  variants, and every available QAT/distillation checkpoint under one protocol.
- `src/onebit_asr/evaluation/model_size.py` — `quick_size_report` (component
  bytes without writing a container file).

### Added — Phase 18 (binary compute investigation)
- `src/onebit_asr/quantization/packed.py` — packed 1-bit storage,
  `PackedBitLinear` (expands bits at forward time; explicitly **not** a compute
  accelerator), and an XNOR+popcount kernel reference for the 1-bit×1-bit case.
- `scripts/benchmark_packed.py` — storage/inference/kernel-boundary benchmark.

### Added — Phase 19 (streaming)
- `src/onebit_asr/streaming.py` — overlapping chunking, word-level overlap
  merge, and a `StreamingTranscriber` explicitly scoped as pseudo-streaming.
- `scripts/stream_infer.py` — demo with partial transcripts and WER vs a full
  utterance decode.

### Added — tests (88 total, up from 63)
- `tests/test_distillation.py` (9), `tests/test_packed.py` (5),
  `tests/test_streaming.py` (8), `tests/test_latency.py` (3).

### Changed
- `src/onebit_asr/training/qat.py` — `build_training_arguments` accepts
  `remove_unused_columns` (required for the teacher columns).
- `src/onebit_asr/training/__init__.py` — exports the distillation API.
- `.gitignore` — ignores generated cluster effective configs and cluster logs.
- `pyproject.toml` — version 1.3.0 → 1.5.0; the missing `v1.4.0` tag is added to
  the previous snapshot (the tag had not been pushed).

### Fixed
- `scripts/diagnose_ptq_collapse.py` — removed a stray duplicate
  `Wav2Vec2ConformerForCTC` import flagged `noqa: F401`.

### Environment note (important)
- Cluster runs use **torch 2.11.0+cu126 / transformers 4.57.6 / datasets 5.0.1**
  in a project venv with system site packages. This differs from the pinned
  Kaggle stack. Every result records its versions; no number is extrapolated
  across profiles, and the historical Kaggle artifacts are never overwritten.

---

## v1.4.0 — First real cloud-GPU QAT: attention-only 1-bit WER 3.92%

**Scope:** the first real GPU QAT training run (Kaggle Tesla T4) — attention-only
1-bit quantization — plus the entire Kaggle execution pipeline and the
infrastructure fixes it required. No FFN QAT, no distillation, no new scientific
arms. Full narrative and failure log: [`planning.md`](planning.md).

### Added — Kaggle execution pipeline
- `kaggle/kernel.py` — script kernel: env report → clone repo at a pinned SHA →
  install deps (preserving the CUDA 13 torch stack) → verify audio decode →
  verify **exactly 96** attention BitLinear replacements → run the GPU probe →
  enforce the ≥15% VRAM headroom gate (auto-retry once with gradient
  checkpointing) → full QAT → pack the artifact → collect outputs.
- `kaggle/verify_replacement.py` (96-layer gate), `kaggle/requirements-kaggle.txt`,
  `kaggle/constraints-kaggle.txt`, `kaggle/kernel-metadata.template.json`,
  `kaggle/README.md`.
- `scripts/kaggle_submit.ps1` / `kaggle_status.ps1` / `kaggle_output.ps1` —
  local CLI wrappers (submit pins the exact pushed commit; GitHub stays the
  source of truth).
- `configs/qat_attn.yaml` — pristine attention-only experiment spec.
- `scripts/pack_qat_model.py` + `model_size.build_packed_tensors` /
  `write_packed_artifact` — write a real packed 1-bit deploy artifact.
- `dataset.build_ctc_train_dataset` — streamed, bounded-memory feature build.
- `dataset._audio_decoder` / `_decode_audio_soundfile` — opt-in
  `ONEBIT_AUDIO_DECODER=soundfile` fallback (default unchanged).

### Measured — attention-only QAT (Kaggle Tesla T4)
- Replacement: **96/96** attention layers; FFN, feature_projection, ctc_head FP.
- Probe: attempt 1 (GC off) OOM → attempt 2 (GC on) **PASS**: 9.232 s/step,
  peak reserved **12.99 / 15.64 GB (16.9% free)**, extrapolated **4.81 h**.
- Training: train.100, 10,000 utts × 3 epochs, batch 4 × accum 4, fp16 + gradient
  checkpointing, 1,875 steps, final train loss 348.9.
- Evaluation (pinned 256-utt dev-clean): **WER 0.0392 / CER 0.0125**
  (vs FP32 0.0202; vs attn+FFN PTQ 1.0000). `results/qat_attn.json`.
- Packed deploy artifact (generated locally from the valid retrieved checkpoint):
  **1,983,557,856 B (~1.98 GB)** — dominated by the retained FP32 FFN
  (`results/qat_attn_size_report.json`).

### Fixed (each discovered by a real failure)
- torch/torchvision CUDA-stack mismatch (`4ff384c`).
- 10k-subset host-RAM OOM via streamed features (`7933a48`).
- GPU device placement in probe/eval/save-load (`bb0e50d`).
- Probe CUDA OOM now treated as a gate failure with a GC retry (`60e4535`).
- Kaggle CLI UTF-8 crash on the Windows console (`6b90094`).
- Trainer epoch checkpoints exhausting Kaggle's 20 GB quota (`d1927e4`).
- Save/load false negative: raw-logit `atol=1e-5` → argmax + tolerant
  `atol/rtol=1e-3` (`9fa7bbb`).
- `_config_path` provenance recorded the `extends` root instead of the leaf.
- Kaggle summary reported a null probe gate when attempt 1 had crashed.

### Honest note on the original run status
The v7 Kaggle job ended `ERROR` because the pre-fix save/load check returned exit
1 **after** training and evaluation succeeded, which skipped the Kaggle packing
stage. That is `FAILURE CAUSED BY INFRASTRUCTURE/BUG`, not a failed QAT
experiment. `results/qat_attn.json` is preserved with `save_load_verified: false`
and an explanatory note; the clean reproduction is recorded separately as
`results/qat_attn_repro.json` (+ `results/qat_attn_repro_size_report.json`).

### Clean reproduction (attempt 1 measured; final Kaggle run quota-blocked)
- `results/qat_attn_repro.json` — rerun on commit `20019c9`: **WER 0.0408 /
  CER 0.0124** vs the original 0.0392 (**+0.16 pp**; both kept, per policy).
- That rerun again ended `ERROR`: the first "corrected" save/load check still
  required exact per-frame argmax equality (a GPU false negative), so packing was
  skipped. Fixed in `d94331d` — tolerant logit closeness (`atol/rtol=1e-2`) plus
  argmax agreement ≥ 0.99 (both recorded) — and `kernel.py` now proceeds to
  packing whenever training + evaluation succeeded.
- The final clean Kaggle run (`COMPLETE` + Kaggle-side packing) was **blocked by
  the weekly GPU quota** (`kaggle quota`: 30.01 h / 30.00 h; refresh
  `2026-09-26`). It is queued for after the reset.
- `results/qat_attn_repro_size_report.json` — packed deploy artifact
  **1,983,557,856 B**, generated locally from attempt 1's checkpoint.

### Not included (explicitly deferred)
Attention+FFN QAT, FFN-only QAT, layer-sensitivity sweeps, knowledge
distillation, 2-bit/INT8, binary compute kernels, streaming ASR.

---

## v1.3.0 — QAT GPU probe tooling (no GPU run executed)

**Scope:** the mandatory pre-run probe from the Phase 11 plan §7b. Tooling only —
no GPU was available, so **no GPU probe or QAT training was run** in this
checkpoint.

### Added
- `--probe` mode in `scripts/train_qat.py`: runs a short (default 30-step)
  training pass using the *full run's* batch settings, then records measured
  seconds/optimizer-step, peak VRAM, and an extrapolation to the full run.
  Writes `results/qat_probe.json`. No eval and no checkpoint are produced.
- `src/onebit_asr/training/qat.py`: `gpu_memory_stats` (device, total, peak
  allocated/reserved, whole-GPU `nvidia-smi` usage), `reset_gpu_peak_memory`,
  `extrapolate_training_time`, `vram_headroom` (default 15% required).
- `configs/qat.yaml`: `qat.probe.steps` (default 30).
- `--probe-steps` CLI override.
- 5 new tests (61 total): extrapolation math, invalid step-time rejection,
  headroom pass/fail, CPU not-applicable, CPU-labeled memory stats.

### Probe result schema (honest labeling)
`results/qat_probe.json` records `is_real_gpu_probe: true|false`, the actual
`compute_profile`, measured memory/time, extrapolation, and a
`verdict.full_run_recommended` gated on ≥15% VRAM headroom. A CPU validation
run sets `is_real_gpu_probe: false` with an explicit notice, so a non-GPU
probe can never be mistaken for a GPU measurement.

### Verified in this checkpoint (local CPU only)
- `pytest tests/ -q` → 61 passed.
- `--probe --smoke --probe-steps 3 --output <temp>` completes end-to-end and
  writes a correctly-labeled (`is_real_gpu_probe: false`) probe JSON. The
  output was written outside `results/` and is intentionally not committed.
- No repo files under `results/` were added for the probe (a CPU probe is not
  a measurement of interest).

### Not included (explicitly deferred)
The real A10G probe, the full 1,000 × 3 QAT run, and any QAT WER — all gated
on GPU access and explicit approval. See README §31.

---

## v1.2.0 — Phase 11 QAT infrastructure + CPU smoke run

**Scope:** QAT infrastructure, tests, and a tiny CPU smoke/dry run only.
The **real cloud-GPU QAT training has NOT been run** (gated; requires GPU +
explicit approval). No distillation, no Phase 12/13, no CUDA kernels, no
hardware acceleration, no fabricated results.

### Added — QAT assembly (`src/onebit_asr/training/qat.py`)
- `setup_qat_model` — load FP32 fine-tuned checkpoint → `replace_linears` with
  the exact Phase 9/10 attn+FFN selection (FP masters preserved).
- `freeze_feature_extractor` / `freeze_report` — freeze only the CNN feature
  extractor; report trainable/frozen counts by group.
- `probe_gradient_flow` — one forward/backward to measure gradient flow
  honestly (Trainer zeroes grads after the last step).
- `master_weight_snapshot` / `master_weight_delta` — prove the optimizer
  updated the FP masters and that they remain FP (not binary).
- `build_training_arguments` / `build_trainer` — stock HF `Trainer`;
  `gradient_checkpointing` taken explicitly from config (never hard-coded);
  `warmup_ratio` resolved to `warmup_steps` (transformers 5.x API).
- `save_qat_checkpoint` / `load_qat_checkpoint` — FP master weights +
  `checkpoint_manifest.json` with `is_1bit_packed: false`.

### Added — data & CLI
- `src/onebit_asr/data/collator.py` — `DataCollatorCTCWithPadding` (−100 labels).
- `src/onebit_asr/data/dataset.py` — `load_train_subset` (train.100;
  leakage-guarded against validation/test) and `prepare_ctc_features`.
- `scripts/train_qat.py` — `--smoke` CPU dry run / full cloud-gpu mode.
- `configs/qat.yaml` — full training block (subset, lr, epochs, batch,
  grad-accum, warmup, max_grad_norm, optimizer, scheduler, fp16/bf16,
  `gradient_checkpointing`) + smoke overrides; inherits layers/quantization
  from `binary.yaml`.

### Added — tests (56 total, up from 40)
- `tests/test_qat.py` (15) — freezing, BitLinear grads, optimizer updates,
  FP-ness of masters, finite/decreasing loss, save/load round trip, collator,
  leakage guard, gradient-checkpointing flag, gradient probe, master delta,
  stock Trainer.
- `tests/test_evaluate.py` (1) — evaluator forces eval mode (and restores it).

### Fixed (found during Phase 11 work)
- **Broken clean checkout:** the `.gitignore` rule `data/` also matched the
  source package `src/onebit_asr/data/`, so `__init__.py`, `text.py`,
  `dataset.py` (and later `collator.py`) were never committed — a fresh clone
  could not import `onebit_asr.data.*`. The rule is now anchored (`/data/`)
  and the package is tracked.
- `transformers` 5.17 `TrainingArguments` has no `warmup_ratio`; the ratio is
  now resolved to `warmup_steps`.
- Evaluator now guarantees eval mode during measurement, so a training loop
  cannot leak dropout into the pinned protocol.
- Post-training gradient inspection is meaningless (grads are zeroed); replaced
  with an explicit pre-training probe.

### Measured this checkpoint (`results/qat_smoke.json`)
- Gradient flow: **192/192** BitLinear masters received gradients; frozen
  feature extractor gradient = zero.
- Master weights updated by the optimizer (digest changed, |W| sum +302.4) and
  remained FP (fraction exactly ±1 = 0.0).
- 589,166,752 trainable / 4,210,176 frozen params; 16 train.100 utterances,
  8 steps, batch 1 × grad-accum 2, AdamW lr 2e-5, local CPU.
- Train loss 907.30 → 1137.07 over 8 steps (pipeline check only; not an
  accuracy experiment).
- Smoke WER/CER on 8 dev utterances: **1.0000 / 1.0000** (reported as measured;
  the smoke run is not expected to recover accuracy).
- Checkpoint save/load verified (identical logits); checkpoint is FP32 master
  weights, **not** bit-packed.
- `results/baseline.json` unchanged.

### Not included (explicitly deferred)
Real cloud-GPU QAT run (see README §31), QAT evaluation (Phase 12),
knowledge distillation (Phase 13), layer sensitivity, 2-bit, activation
quantization, CUDA kernels, hardware acceleration, streaming ASR.

---

## v1.1.0 — CPU-safe checkpoint through Phase 10 (1-bit PTQ)

**Scope:** Phases 4–10, plus the Phase-0 audit/cleanup fixes. Local CPU
profile only. No training, no QAT, no distillation, no GPU.

### Added — quantization core (Phases 4–6)
- `src/onebit_asr/quantization/ste.py` — `BinaryQuantize` (sign forward,
  straight-through backward) with optional, explicit clipped-STE mode.
- `src/onebit_asr/quantization/scaling.py` — `alpha = mean(|W|)`, per-tensor,
  recomputed each forward, not a learned parameter.
- `src/onebit_asr/quantization/binarization.py` — `Wb = sign(W)`,
  `Wq = alpha · Wb` through the STE.
- `src/onebit_asr/quantization/bitlinear.py` — `BitLinear(nn.Linear)`,
  FP master weights, quantizes inside `forward()`, `from_linear()`,
  explicit config in `extra_repr()`.

### Added — layer replacement (Phase 8)
- `src/onebit_asr/models/conformer_utils.py` — shared taxonomy
  (`categorize`, `iter_linears`, `select_by_category`, `selected_categories`,
  `get_parent_module`); now the single source of truth for both inspection and
  replacement. Unknown selection keys raise instead of silently matching
  nothing.
- `src/onebit_asr/models/replace_layers.py` — `replace_linears`,
  `report_replacement`, `summarize_by_category`.

### Added — evaluation + PTQ (Phases 9–10)
- `src/onebit_asr/evaluation/evaluate.py` — model-agnostic `evaluate_model`
  runner (the Phase 3 protocol, reusable for any in-memory model).
- `scripts/quantize.py` — 1-bit PTQ pipeline (no retraining).
- `scripts/verify_eval_equivalence.py` — proves the refactored evaluator
  reproduces `results/baseline.json` predictions exactly.
- `scripts/smoke_ptq_pipeline.py` — replacement + forward + size smoke check.
- `scripts/diagnose_ptq_collapse.py` — control experiment isolating the cause
  of the PTQ collapse.
- `src/onebit_asr/evaluation/model_size.py` — `binary_model_size_report`
  (real `numpy.packbits` packing, scales, non-quantized FP params, measured
  safetensors container overhead) and dual checkpoint figures.
- `src/onebit_asr/config.py` — `load_config` with `extends` composition.

### Added — tests (Phase 7 gate)
- `tests/conftest.py`, `tests/test_ste.py`, `tests/test_binarization.py`,
  `tests/test_bitlinear.py`, `tests/test_layer_replacement.py`.
- 40 tests, all passing, using tiny synthetic tensors/models only.

### Changed — audit/cleanup fixes (Step 0)
- `.gitignore` rewritten cleanly (removed malformed UTF-16LE line; keeps
  `*.log`); stray root logs removed.
- `scripts/inspect_model.py` — fixed operator-precedence bug in
  `encoder_depth`; docstring/terminology corrected to `ffn_in` / `ffn_out`;
  uses the shared taxonomy.
- `configs/binary.yaml` — `ffn: true` replaced with `ffn_in` / `ffn_out`;
  `extends: baseline.yaml`.
- `configs/baseline.yaml` — removed dead `normalize_text` config key
  (normalization is unconditional under the pinned protocol).
- `configs/qat.yaml` / `configs/distillation.yaml` — now compose via
  `extends:` instead of prose comments.
- `scripts/eval_baseline.py` — refactored onto the shared evaluator; records
  both checkpoint size figures.

### Measured results (this checkpoint)
- FP32 baseline (unchanged reference): WER **0.0202**, CER **0.0049**,
  RTF 0.284 — `results/baseline.json` (immutable).
- 1-bit PTQ (attention + FFN): WER **1.0000**, CER **1.0000** — total
  collapse, as the plan predicted — `results/ptq_binary.json`.
- Control (no replacement) exactly reproduces the FP32 hypotheses;
  FFN-only binarization is the decisive cause of the collapse —
  `results/ptq_collapse_diagnostic.json`.
- Size: FP32 safetensors **2.374 GB**; packed binary artifact (packed weights
  + scales + FP params, real file) **0.423 GB** — `results/size_report.json`.

### Not included (explicitly deferred)
QAT, knowledge distillation, any GPU training, CUDA kernels, streaming ASR,
binary hardware acceleration.

---

## v1.0.0 — FP32 baseline milestone (Phases 0–3)

- Phase 0: environment, pinned dependencies, `scripts/check_env.py` → PASS.
- Phase 1: architectural map of `facebook/wav2vec2-conformer-rope-large-960h-ft`
  (593,376,928 params, 194 `nn.Linear`, 504.2M Linear params = 84.97%).
- Phase 2: inference smoke test (`scripts/infer.py`) — exact-match
  transcription on a real LibriSpeech sample.
- Phase 3: reproducible FP32 baseline on a fixed 256-utterance dev-clean
  subset (seed 42), greedy CTC, normalized text: WER **0.0202**, CER
  **0.0049**, RTF 0.284.
- Windows audio setup: `torchcodec` + FFmpeg shared DLLs
  (`scripts/setup_windows_ffmpeg.ps1`).
