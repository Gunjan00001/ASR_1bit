# Changelog

All notable checkpoints of this repository. Versions are git tags `vX.Y.Z`
(`major.minor.patch`); see the versioning section of `README.md`.

The repository is maintained as uploaded code snapshots; each entry describes
an exact, reproducible repository state.

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
