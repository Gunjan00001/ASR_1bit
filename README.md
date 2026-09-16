# One-Bit Wav2Vec2-Conformer ASR

Research project: progressively replace FP linear layers of a pretrained
**Wav2Vec2-Conformer CTC** model with custom 1-bit **BitLinear** layers, then
measure — honestly — what that costs in accuracy, size, and speed.

**Current checkpoint:** `v1.2.0` — CPU-safe work complete through **Phase 11
QAT infrastructure + tests + CPU smoke/dry run**. The FP32 baseline, the 1-bit
PTQ model, and the QAT pipeline are all measured/validated under the same
pinned protocol. The **real cloud-GPU QAT training has NOT been run** (gated;
see §31). Results are real measurements; nothing in this README is estimated.

> This README is written for an engineer joining at this checkpoint. It states
> what exists, what was measured, and — just as importantly — what has **not**
> been done yet.

---

## 1. Project objective

Build and benchmark a 1-bit ASR system by starting from a pretrained
Wav2Vec2-Conformer CTC model and progressively replacing suitable
floating-point linear layers with custom BitLinear layers.

Core question:

> Can a Wav2Vec2-Conformer ASR model be made substantially smaller using 1-bit
> weights while retaining competitive ASR accuracy?

The project reuses standard ASR infrastructure (HF Transformers, LibriSpeech
tooling, CTC, tokenizer) and implements only the low-bit-specific parts.
See `one_bit_asr_implementation_plan.md` for the full plan.

## 2. Current project status

| Phase | Description | Status |
|---|---|---|
| 0 | Environment | ✅ done |
| 1 | Model inspection | ✅ done |
| 2 | Baseline inference | ✅ done |
| 3 | FP32 baseline evaluation | ✅ done |
| 4 | Toy BitLinear | ✅ done |
| 5 | Binary quantization + scaling | ✅ done |
| 6 | Straight-Through Estimator | ✅ done |
| 7 | Test gate (40 tests) | ✅ green |
| 8 | Configurable layer replacement | ✅ done |
| 9 | First 1-bit experiment (PTQ) | ✅ done |
| 10 | 1-bit PTQ evaluation + size accounting | ✅ done |
| 11 | QAT (infrastructure + tests + CPU smoke) | ✅ done (local) |
| 11 | QAT (real cloud-GPU training run) | ⛔ not run (gated) |
| 12 | QAT evaluation | ⛔ not started |
| 13 | Knowledge distillation | ⛔ not started |
| 14–19 | Controlled experiments, sensitivity, packing, benchmarks, streaming | ⛔ not started |

## 3. Base model

`facebook/wav2vec2-conformer-rope-large-960h-ft` (`Wav2Vec2ConformerForCTC`),
fine-tuned on 960 h of LibriSpeech. Model-card reference (greedy CTC):
**WER 1.96 (test-clean) / 3.98 (test-other)**.

## 4. Model architecture summary (measured in Phase 1)

- Class: `Wav2Vec2ConformerForCTC`
- Total parameters: **593,376,928**; FP32 in-memory size ≈ **2,373.5 MB**
- Encoder: **24 × `Wav2Vec2ConformerEncoderLayer`**
- Linear layers: **194** `nn.Linear`, totaling **504,218,656 params (84.97%)**
- Convolutional feature extractor: 7 × `Wav2Vec2ConformerLayerNormConvLayer`
  (kept FP)
- Full map: `results/model_inspection.json`

### Linear census by category (measured)

| Category | Layers | Params | % of model | Shape |
|---|---:|---:|---:|---|
| `ffn_in` (`ffn{1,2}.intermediate_dense`) | 48 | 201,523,200 | 33.96% | 1024 → 4096 |
| `ffn_out` (`ffn{1,2}.output_dense`) | 48 | 201,375,744 | 33.94% | 4096 → 1024 |
| `attention_q` (`self_attn.linear_q`) | 24 | 25,190,400 | 4.25% | 1024 → 1024 |
| `attention_k` (`self_attn.linear_k`) | 24 | 25,190,400 | 4.25% | 1024 → 1024 |
| `attention_v` (`self_attn.linear_v`) | 24 | 25,190,400 | 4.25% | 1024 → 1024 |
| `attention_output` (`self_attn.linear_out`) | 24 | 25,190,400 | 4.25% | 1024 → 1024 |
| `feature_projection` (`feature_projection.projection`) | 1 | 525,312 | 0.09% | 512 → 1024 |
| `ctc_head` (`lm_head`) | 1 | 32,800 | 0.01% | 1024 → 32 |

## 5. Environment / compute profile

- **Profile:** `local` (CPU only). No CUDA GPU is available.
- Machine: AMD Ryzen AI 7 350 (8 cores), ~24 GB RAM, Radeon 860M iGPU
  (no usable PyTorch CUDA path).
- OS: Windows 11 (`Windows-11-10.0.26200-SP0`), Python **3.14.3**
- `torch.cuda.is_available() == False`; torch CPU build; 8 torch threads.
- Report: `results/env.json` (`python scripts/check_env.py` → PASS).

> Training-scale phases (QAT, distillation) are scoped to a future `cloud-gpu`
> profile. Nothing GPU-dependent has been run.

## 6. Exact dependency versions

Pinned in `requirements.txt` (verified installable on Python 3.14 / Windows):

```
torch==2.14.0            (CPU build: 2.14.0+cpu)
torchaudio==2.11.0
transformers==5.17.0
datasets==5.0.1
evaluate==0.4.6
jiwer==4.0.0
accelerate==1.15.0
numpy==2.5.3
soundfile==0.14.0
pyyaml==6.0.3
pytest==9.1.1
psutil==7.2.2
torchcodec==0.16.0
```

`torchcodec` is required for `datasets` audio decoding and needs FFmpeg
**shared** libraries on Windows — run `scripts/setup_windows_ffmpeg.ps1` once
after creating `.venv` (see §28).

## 7. Phases 0–3 accomplishments

- **Phase 0** — venv, pinned deps, `scripts/check_env.py` (PASS), seed
  discipline, environment report `results/env.json`.
- **Phase 1** — read-only architectural map: `scripts/inspect_model.py` →
  `results/model_inspection.json` (194 Linears, categorized; model untouched).
- **Phase 2** — inference pipeline: `scripts/infer.py` (audio → processor →
  conformer → logits → greedy decode → text). Smoke test transcribed a real
  LibriSpeech sample exactly (reference match).
- **Phase 3** — reproducible FP32 baseline on a fixed 256-utterance dev-clean
  subset (seed 42), greedy CTC, normalized text.

## 8. FP32 baseline result (immutable reference)

`results/baseline.json` — **never edited or regenerated** after this checkpoint.

| Metric | Value |
|---|---|
| WER | **0.0202** (2.02%) |
| CER | **0.0049** (0.49%) |
| RTF | 0.284 |
| Latency mean / p50 / p95 / max | 2.09 s / 1.60 s / 5.02 s / 9.75 s |
| Throughput | 0.479 utt/s (1.76 audio-s/s) |
| Peak RSS | 3,509.7 MB |
| Params | 593,376,928 |
| Checkpoint (as recorded then) | 4,747,815,835 B over `model.safetensors` + `pytorch_model.bin` |

The 2.02% WER matches the model-card 1.96% reference, confirming a sane
baseline. *(Historical note: `baseline.json` predates the dual-size schema and
records the full-download figure — 4.75 GB, which double-counts the duplicate
`.bin`. It is left byte-for-byte unchanged; newer runs record both figures. See
§24 and `CHANGELOG.md`.)*

## 9. Phase 4 — BitLinear design

`src/onebit_asr/quantization/bitlinear.py` — `BitLinear(nn.Linear)`:

- **Subclasses `nn.Linear`** → identical `state_dict` keys, drop-in swap, no
  HF `Trainer` changes required.
- **Retains FP master weights.** `self.weight` is always full precision.
- **Quantizes inside `forward()`**, producing a *derived* tensor each call.
- **Bias stays FP** (or `None`, if there is no bias).
- `BitLinear.from_linear(nn.Linear, scale_mode, ste_clip)` — numerically copies
  existing FP weights/bias into master weights (pre-quantization values
  preserved).
- `extra_repr()` reports `in_features/out_features/bias/scale_mode/ste_clip`.

Forward:

```
W_master (FP, never mutated)
      -> sign(W)             [STE backward]
      -> alpha * sign(W)     [alpha = mean(|W|), recomputed per forward]
      -> F.linear(input, Wq, bias)
```

## 10. Binary quantization equation

```
Wb = sign(W)
Wq = alpha * Wb        (alpha = mean(|W|))
```

Implemented in `binarization.py`; `sign` is routed through the STE so gradients
reach the master weights. `sign(0) = 0` (documented; measure-zero in trained
weights, covered by tests).

## 11. Scaling strategy

`scaling.py`: **per-tensor** `alpha = mean(|W|)` (XNOR-Net-style L1 scaling).

- Per-layer scalar, **not** per-channel.
- **Recomputed on every forward** from the current master weights.
- **Not a learned parameter.** Gradients still flow into the master weights,
  because `alpha` is a differentiable function of `W`.
- Configurable (`quantization.scale`); only `per_tensor_mean_abs` is
  implemented. An unknown mode raises.

## 12. STE behavior

`ste.py` — `BinaryQuantize(torch.autograd.Function)`:

- **Forward:** `sign(x)`
- **Backward:** straight-through — the upstream gradient passes unchanged to
  the FP master weight.
- **Optional clipped variant** (`ste_clip=<float>`): zeroes gradients where
  `|x| > clip_value`. This is opt-in and explicit — the active mode is always
  visible in `BitLinear.extra_repr()` ("no silent behavior change").
- Default is pure pass-through (`ste_clip: null` in `configs/binary.yaml`).

## 13. Why FP master weights are retained

- Binary weights cannot be updated directly by SGD (the sign function is
  non-informative outside the STE), so learning must happen in a full-precision
  shadow copy.
- Keeping `self.weight` FP means the checkpoint loads losslessly, `state_dict`
  keys are unchanged, and the same module supports QAT (Phase 11) without
  modification.
- The master weight is **never** overwritten by binary values — verified by
  `tests/test_bitlinear.py`.

## 14. Layer-replacement architecture

Two modules, one shared taxonomy:

- `models/conformer_utils.py` — `categorize`, `iter_linears`,
  `select_by_category`, `selected_categories`, `get_parent_module`.
  Unknown selection keys raise `ValueError` (prevents a typo silently
  exempting every intended layer).
- `models/replace_layers.py` — `replace_linears` (parent-module `setattr`,
  weight-preserving swap) and `report_replacement`.

Replacements are selected by **category**, never by hard-coded architecture
assumptions.

## 15. Exact Linear naming discovered in this HF checkpoint

Hugging Face naming differs from fairseq. For
`facebook/wav2vec2-conformer-rope-large-960h-ft`:

```
wav2vec2_conformer.feature_projection.projection          -> feature_projection
wav2vec2_conformer.encoder.layers.{i}.self_attn.linear_q  -> attention_q
wav2vec2_conformer.encoder.layers.{i}.self_attn.linear_k  -> attention_k
wav2vec2_conformer.encoder.layers.{i}.self_attn.linear_v  -> attention_v
wav2vec2_conformer.encoder.layers.{i}.self_attn.linear_out-> attention_output
wav2vec2_conformer.encoder.layers.{i}.ffn1.intermediate_dense -> ffn_in
wav2vec2_conformer.encoder.layers.{i}.ffn1.output_dense        -> ffn_out
wav2vec2_conformer.encoder.layers.{i}.ffn2.intermediate_dense -> ffn_in
wav2vec2_conformer.encoder.layers.{i}.ffn2.output_dense        -> ffn_out
lm_head                                                   -> ctc_head
```

Note: the taxonomy emits `ffn_in` / `ffn_out` (not a single `ffn` key).
`configs/binary.yaml` uses these exact keys.

## 16. Layers that are quantization candidates

Attention projections + FFN projections = **192 Linear layers, 503,316,480
params (~84.8% of the model)**. This is the Phase 9/10 configuration:

```yaml
attention_q/k/v/output: true
ffn_in: true
ffn_out: true
```

This is an experimental choice, not a permanent rule.

## 17. Layers that remain FP

- `feature_projection.projection` (`false`)
- `ctc_head` / `lm_head` (`false`)
- All convolutions, LayerNorm/GroupNorm, the convolutional feature extractor,
  positional-convolution modules, and all biases

## 18. Phase 7 test coverage

`pytest tests/ -q` → **40 passed** (tiny synthetic tensors/models only; the
593M checkpoint is never loaded for unit tests).

- `test_ste.py` — forward = sign; pure pass-through backward; upstream value
  preservation; clipped STE zeroing; invalid clip rejection.
- `test_binarization.py` — binary values are ±1; `alpha = mean(|W|)`;
  `Wq = alpha·sign(W)`; per-tensor scalar; unknown-mode rejection;
  `sign(0)=0`; differentiability.
- `test_bitlinear.py` — shape parity with `nn.Linear`; binary weights ±1 before
  scaling; scaling correctness; gradient reaches master weight; optimizer
  updates master weight; **master weight not overwritten**; bias FP; no-bias
  case; dtype/device preserved; `state_dict` key parity; save/load round-trip;
  `extra_repr` config; no-grad/eval forward; **tiny toy task reduces loss**;
  deterministic repeated forwards.
- `test_layer_replacement.py` — taxonomy matches HF naming; unknown key raises;
  category selection counts; only selected categories swapped; weights
  preserved numerically; forward still runs; attention-only / FFN-only
  selections; idempotent replacement; report rendering; parent lookup.

## 19. Phase 8 replacement behavior (measured on the real model)

Applying `configs/binary.yaml` to the checkpoint produced:

```
replaced: 192 layers, 503,316,480 params
retained:   2 layers,     558,112 params   (feature_projection + ctc_head)
```

Category breakdown of the swap is recorded in `results/ptq_binary.json`
(`replacement` section).

## 20. Phases 9/10 — PTQ methodology

`scripts/quantize.py`:

1. Load the original FP32 checkpoint unchanged.
2. Load `configs/binary.yaml`.
3. Replace configured attention + FFN `nn.Linear` layers with `BitLinear`
   (weights copied into FP master weights; `alpha` computed in forward).
4. **Do not retrain.**
5. **Do not modify the FP32 baseline checkpoint.**
6. Evaluate with the exact Phase 3 protocol.
7. Record WER/CER, delta vs FP32, replacement counts, size accounting,
   provenance, and the selected config.

## 21. Exact evaluation protocol (pinned)

Identical for the FP32 baseline and the 1-bit PTQ model:

- Dataset: `openslr/librispeech_asr`, config `clean`, split `validation`
- Subset: **256 utterances**, fixed seed **42** (sample IDs recorded in JSON)
- Decoding: **greedy CTC** (argmax), batch size 1, **no language model**
- Text normalization: **lowercase + strip punctuation** (unconditional)
- Metrics: WER/CER (jiwer), RTF = processing time / audio duration
- Per-utterance latency (mean/p50/p95/max) and peak RSS recorded
- Compute profile: `local`, CPU
- Evaluation order: as selected by the fixed seed

Equivalence check: `scripts/verify_eval_equivalence.py` confirms the refactored
evaluator reproduces `results/baseline.json` hypotheses exactly (0 mismatches)
— i.e. the protocol did not drift between Phase 3 and Phase 9.

## 22. Exact local CPU limitations

- No CUDA GPU: training-scale phases (QAT/distillation) are not runnable here
  at meaningful scale; they are deferred to a `cloud-gpu` profile.
- Evaluation is CPU-bound: ~0.28 RTF FP32 and ~0.40 RTF binarized; a full
  256-utterance pass takes ~11–13 minutes.
- Peak RSS ~3.3–3.5 GB (the FP32 model alone is ~2.37 GB).
- Binarization does **not** accelerate inference here: `BitLinear.forward()`
  reconstructs `alpha·sign(W)` as FP and calls `F.linear`, so the binarized
  run is *slower* than FP32 (0.284 → 0.400 RTF). This is expected and is not
  hidden (see §25, §26).

## 23. Actual PTQ results

`results/ptq_binary.json` (256 utterances, identical protocol):

| Model | WER | CER | WER Δ vs FP32 | RTF | p50 latency | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| FP32 baseline | **0.0202** | **0.0049** | — | 0.284 | 1.60 s | 3,509.7 MB |
| 1-bit PTQ (attention+FFN) | **1.0000** | **1.0000** | **+0.9798** | 0.400 | 2.47 s | 3,297.6 MB |

**The 1-bit PTQ model collapses completely** (WER 100%; hypotheses are empty).
The plan predicted severe degradation — this is a recorded result, not a
failure, and no tuning-around was performed.

### Verification that the collapse is genuine

`scripts/diagnose_ptq_collapse.py` (16-utterance diagnostic, same code path;
`results/ptq_collapse_diagnostic.json`):

| Variant | Replaced | WER | Non-empty hypotheses | Agreement with FP32 |
|---|---:|---:|---:|---:|
| `none` (control) | 0 | 0.0256 | 16/16 | **1.000** |
| `attention_only` | 96 | 0.4441 | 16/16 | 0.125 |
| `ffn_only` | 96 | 1.0000 | 0/16 | 0.000 |
| `full` | 192 | 1.0000 | 0/16 | 0.000 |

The control reproduces the FP32 hypotheses exactly, so the pipeline is sound;
the collapse is caused by binarization itself, and the **FFN projections are
the decisive sensitivity** (FFN-only already destroys output). Attention-only
degrades partially but still emits text. (Formal per-layer sensitivity sweeps
remain Phase 15.)

## 24. Actual model-size accounting

`results/size_report.json` — separates theoretical from actual storage. Nothing
here is labeled "1-bit model size" unless it is only the raw payload.

| Quantity | Bytes | Note |
|---|---:|---|
| FP32 checkpoint (`model.safetensors`) | 2,373,821,388 | **canonical** comparison basis |
| FP32 checkpoint (full download) | 4,747,815,835 | includes duplicate `pytorch_model.bin` (double-count) |
| Quantized weight count | 503,316,480 | attention + FFN weights |
| Theoretical payload (1 bit/weight) | 62,914,560 | raw payload only; excludes scales/params/overhead |
| Naive binary (1 byte/weight) | 503,316,480 | how `bool`/`int8` tensors serialize |
| **Packed payload** (`numpy.packbits`) | **62,914,560** | real packed bits |
| FP scales (1 FP32/layer × 192) | 768 | per-tensor scale |
| Non-quantized FP params | 360,241,792 | conv/norm/proj/head/biases (90,060,448 params) |
| **Packed artifact (measured)** | **423,289,296** | real safetensors file: packed + scales + FP params |
| Container overhead (measured) | 132,176 | safetensors headers/alignment |
| FP master weights (excluded) | 2,013,265,920 | QAT training state; not in the packed artifact |

Read this as: the deployed packed artifact is **≈0.423 GB vs 2.374 GB FP32
(≈5.6× smaller)**, but only ~15% of the artifact is the bit-packed weights —
the rest is the FP layers we chose not to quantize. Quantizing the remaining FP
parameters and removing FP master weights from deployment state are future work.

## 25. What has NOT been implemented yet

- **No trained 1-bit QAT model.** QAT *infrastructure* + a CPU smoke run exist
  (§27), but the real cloud-GPU QAT training has not been run, so **no accuracy
  recovery is claimed**.
- **No knowledge distillation** (Phase 13).
- **No CUDA kernels / custom bitwise ops** (Phases 17–18).
- **No actual binary hardware acceleration** — inference expands weights back
  to FP (`alpha·sign(W)`) before `F.linear`; the model is *smaller on disk*,
  not *faster to run* (it measured slower).
- **No GPU training** of any kind has been executed.
- No streaming ASR (Phase 19).
- `evaluate` is pinned but unused (metrics are computed with `jiwer`);
  `features/audio.py`, `decoding/decode.py`, `evaluation/latency.py`,
  `evaluation/benchmark.py`, `training/train.py`, `training/distillation.py`,
  and several `scripts/*` remain intentional placeholders.

## 26. Honest claims policy

Reused vs. ours, measured vs. theoretical vs. planned:

- **Reused (upstream):** HF `Wav2Vec2ConformerForCTC`, `Wav2Vec2Processor`,
  HF `Trainer`, LibriSpeech tooling (`datasets`), CTC loss, tokenizer/vocab,
  the pretrained checkpoint itself.
- **Our implementation:** `BitLinear`, binarization, scaling, STE, layer
  replacement + taxonomy, PTQ pipeline, evaluation runner, size/packing
  accounting, QAT assembly (freeze/probe/trainer wiring/checkpoint), tests.
- **Measured (real numbers in `results/`):** baseline WER/CER/RTF, PTQ
  WER/CER, replacement counts, packed artifact size, latency/RSS, QAT smoke
  gradient-flow/freeze/master-update checks and smoke WER.
- **Theoretical (labeled as such):** 1-bit payload = 62,914,560 B; FP32 weight
  bytes = params × 4.
- **Planned (not implemented):** the real cloud-GPU QAT run, distillation,
  binary compute kernels, hardware acceleration, streaming.

No performance claim in this README is unsupported by a file in `results/`.

## 27. Phase 11 — QAT infrastructure (implemented) and smoke run (measured)

> The real QAT training run is **cloud-GPU work and has not been executed**.
> See §31 for the gated procedure.

### Design (preserves every approved decision)

- **Student:** `Wav2Vec2ConformerForCTC` from the fine-tuned FP32 checkpoint,
  then `replace_linears` with the exact Phase 9/10 selection
  (attention_q/k/v/output + ffn_in/ffn_out → `BitLinear`; `feature_projection`
  and `ctc_head` stay FP). Master weights = the fine-tuned FP32 weights.
- **Quantization:** the same `BitLinear` — FP masters, quantization inside
  `forward()`, `per_tensor_mean_abs` (`alpha = mean(|W|)`), pass-through STE.
  No new quantization method.
- **Freezing:** only the CNN feature extractor (`wav2vec2_conformer.feature_extractor`)
  is frozen. Everything else — BitLinear masters, biases, LayerNorms,
  `feature_projection`, `ctc_head`/`lm_head` — stays trainable.
- **Trainer:** stock Hugging Face `Trainer`, unmodified (a `TrainerCallback`
  records loss; the data collator is custom).
- **Data:** `train.100` only, never the validation split (a leakage guard
  raises if the configured split is `validation`/`test`).
- **Gradient checkpointing:** explicit, configurable
  (`qat.training.gradient_checkpointing`, default `false`), passed straight to
  `TrainingArguments` — never hard-coded.

### Files

- `src/onebit_asr/training/qat.py` — `setup_qat_model`,
  `freeze_feature_extractor`, `freeze_report`, `probe_gradient_flow`,
  `master_weight_snapshot`/`master_weight_delta`, `build_training_arguments`,
  `build_trainer`, `save_qat_checkpoint`/`load_qat_checkpoint`, and the GPU
  probe helpers `gpu_memory_stats`, `extrapolate_training_time`,
  `vram_headroom`.
- `src/onebit_asr/data/collator.py` — `DataCollatorCTCWithPadding` (−100 labels).
- `src/onebit_asr/data/dataset.py` — `load_train_subset` (leakage-guarded),
  `prepare_ctc_features`.
- `scripts/train_qat.py` — `--smoke` (CPU dry run), `--probe` (timed/VRAM
  probe that gates the full run), or full (cloud-gpu) modes.
- `tests/test_qat.py`, `tests/test_evaluate.py` — 21 new tests.

### Two honest measurement notes

1. **Gradient flow is measured by a pre-training probe.** `Trainer` zeroes
   gradients after the final step, so inspecting grads after `train()` is
   meaningless (it reports 0). `probe_gradient_flow` runs one forward/backward
   on a batch and reports the real numbers.
2. **Evaluation forces eval mode.** `evaluate_model` guarantees
   `model.eval()` during measurement (and restores the previous mode), so the
   pinned protocol is not polluted by training-mode dropout.

### Measured smoke result (`results/qat_smoke.json`)

16 `train.100` utterances, batch 1, grad-accum 2, 8 steps, AdamW lr 2e-5,
local CPU, 8 dev utterances for eval:

| Check | Result |
|---|---|
| BitLinear masters receiving gradients | **192 / 192** |
| Frozen feature-extractor gradient | **zero** (frozen verified) |
| Master weights updated by optimizer | **yes** (digest changed, |W| sum +302.4) |
| Master weights remain FP (not binary) | **yes** (fraction exactly ±1 = 0.0) |
| Makeup of the trained params | 589,166,752 trainable / 4,210,176 frozen |
| Train loss (step 1 → 8) | 907.30 → 1137.07 |
| Smoke WER / CER (8 utts) | 1.0000 / 1.0000 |
| Checkpoint save/load verified | **yes** (identical logits) |
| Checkpoint is 1-bit packed | **no** (`fp32_master_weights`) |

The smoke run **does not** (and is not intended to) recover accuracy: 8 steps
at lr 2e-5 on 16 utterances cannot undo the PTQ collapse. It validates the
pipeline (gradients, freezing, optimizer updates, save/load, evaluation) so the
real cloud run is trustworthy. The measured smoke WER is reported as-is.

### What QAT has and has not demonstrated

- Demonstrated: the QAT machinery is correct and produces trainable 1-bit
  students with intact FP masters.
- Not demonstrated: any accuracy recovery. That requires the real cloud-GPU
  run in §31. **No QAT WER claim is made here.**

## 28. Reproducibility instructions

From a clean checkout:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
.venv\Scripts\python.exe scripts\check_env.py          # Phase 0 -> PASS
.\scripts\setup_windows_ffmpeg.ps1                     # Windows audio decoding
.venv\Scripts\python.exe -m pytest tests/ -q           # 61 passed
.venv\Scripts\python.exe scripts\inspect_model.py      # Phase 1 (read-only)
.venv\Scripts\python.exe scripts\infer.py --demo       # Phase 2 smoke test
.venv\Scripts\python.exe scripts\verify_eval_equivalence.py --n 8   # protocol intact
.venv\Scripts\python.exe scripts\eval_baseline.py      # Phase 3 (writes baseline.json)
.venv\Scripts\python.exe scripts\quantize.py           # Phases 9/10 (writes ptq_binary.json, size_report.json)
.venv\Scripts\python.exe scripts\train_qat.py --config configs/qat.yaml --smoke   # Phase 11 CPU dry run
```

Note: re-running `eval_baseline.py` overwrites `results/baseline.json`. The
committed `baseline.json` is the frozen reference — prefer not to overwrite it;
use `--output` to write elsewhere if you must re-run.

Determinism: fixed subset seed (42), recorded sample IDs, greedy decoding,
pinned dependencies, recorded provenance in every results JSON.

## 29. Commands used to verify this checkpoint

```powershell
.venv\Scripts\python.exe scripts\check_env.py                    # PASS
.venv\Scripts\python.exe -m pytest tests/ -q                     # 61 passed
.venv\Scripts\python.exe scripts\inspect_model.py                # 194 Linears, 593,376,928 params
.venv\Scripts\python.exe scripts\verify_eval_equivalence.py --n 8 # 0 mismatches
.venv\Scripts\python.exe scripts\quantize.py                     # WER 1.0000, delta +0.9798
.venv\Scripts\python.exe scripts\diagnose_ptq_collapse.py --n 16 # control agrees 1.000
.venv\Scripts\python.exe scripts\train_qat.py --config configs/qat.yaml --smoke
                                                                 # 192/192 grads, freeze ok, save/load ok
```

## 30. Known limitations / issues

- **WER collapse under PTQ** (expected; §23). QAT is the intended remedy.
- **QAT accuracy recovery is NOT yet measured** — only the smoke pipeline is;
  the real cloud-GPU run is gated (§31).
- **Smoke-run loss rose** (907 → 1137 over 8 steps): expected on 16 utterances
  at lr 2e-5 against an already-collapsed model; it is a pipeline check, not an
  accuracy experiment.
- **FFN sensitivity:** binarizing FFN alone already destroys output; the
  shipped config quantizes the whole candidate pool. Per-layer/per-component
  sensitivity is Phase 15.
- **Binarized inference is slower on CPU**, not faster (§22, §25).
- **Master weights dominate memory:** the PTQ/QAT model keeps ~2.01 GB of FP
  master weights that are not needed for pure inference; deployment-state
  accounting excludes them, but an in-memory model still holds them. QAT
  training peak RSS measured ~7.5 GB locally (optimizer state).
- **Local training is not viable at scale:** `local` CPU profile is for smoke
  runs only.
- **`results/baseline.json` uses the pre-`v1.1.0` size schema** (single
  `checkpoint_bytes` = full download, 4.75 GB). It is intentionally immutable;
  treat `checkpoint_bytes_safetensors` in newer files as canonical.
- **Single scale mode / single STE mode** implemented (per-tensor mean-abs,
  pass-through default). No per-channel scaling, no learned alpha yet.
- No LM decoding; WER figures are greedy-only and therefore not directly
  comparable to LM-assisted ASR leaderboards.

## 31. Gated next step — real cloud-GPU QAT (NOT run)

Requires GPU access **and** explicit approval. Not executed by this repository
checkpoint — no GPU exists on the development machine (CPU-only, see §5).

### Step 1 — mandatory probe (measures VRAM + time/step)

```powershell
# On a cloud GPU machine (e.g. A10G 24 GB), from a clean checkout of v1.3.0:
.venv\Scripts\python.exe scripts\train_qat.py --config configs/qat.yaml --probe
# writes results/qat_probe.json with peak VRAM, s/optimizer-step, and an
# extrapolation to the full run
```

The full run proceeds **only if** the probe reports ≥15% VRAM headroom
(`vram_headroom.passes == true`). Otherwise enable
`qat.training.gradient_checkpointing: true` (and/or `batch_size: 2` with
`gradient_accumulation_steps: 8`) and re-probe. On T4 also set
`fp16: true`, `bf16: false` (Turing has no bf16).

### Step 2 — full run (only after the probe passes)

```powershell
.venv\Scripts\python.exe scripts\train_qat.py --config configs/qat.yaml `
    --output results/qat.json --checkpoint-dir checkpoints/qat
```

- Full mode uses `training.*`: `train.100`, `subset_size` 1000 (raise for a
  stronger run), lr 2e-5, 3 epochs, then evaluates on the full pinned
  256-utterance dev subset.
- Outputs: `results/qat.json` (commit it) and `checkpoints/qat/` (FP32 masters
  + manifest; git-ignored).
- Then and only then: Phase 12 (QAT evaluation) / Phase 13 (distillation).
- **Acceptance is a measured WER**, reported exactly as measured.

## Layout

```
one-bit-asr/
├── configs/            baseline / binary / qat / distillation (extends-composed)
├── src/onebit_asr/
│   ├── config.py               extends-aware config loader
│   ├── data/                   text normalization, dev/train loaders, CTC collator
│   ├── features/               (placeholder) audio helpers
│   ├── models/                 conformer_utils (taxonomy), replace_layers
│   ├── quantization/           ste, scaling, binarization, bitlinear
│   ├── training/               qat (freeze/probe/trainer/checkpoint); train+distillation placeholders
│   ├── decoding/               (placeholder) decode
│   └── evaluation/             evaluate (runner), metrics, model_size
├── scripts/            check_env, inspect_model, infer, eval_baseline,
│                       verify_eval_equivalence, quantize, diagnose_ptq_collapse,
│                       smoke_ptq_pipeline, train_qat, setup_windows_ffmpeg
├── tests/              61 tests (STE, binarization, BitLinear, replacement, QAT, evaluate)
├── results/            env, model_inspection, baseline, ptq_binary,
│                       size_report, ptq_collapse_diagnostic, qat_smoke  (tracked)
├── checkpoints/        (git-ignored)
└── one_bit_asr_implementation_plan.md
```
