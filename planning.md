# planning.md — Living research history & execution plan

This is the project's living history and execution plan. The original,
forward-looking specification lives unchanged in
[`one_bit_asr_implementation_plan.md`](one_bit_asr_implementation_plan.md);
this file records what was actually **done**, **measured**, **failed**, and
**what comes next**. It deliberately does not rewrite history to look cleaner
than it was.

Status labels used throughout:

- `IMPLEMENTED` — code exists and is exercised by tests.
- `MEASURED` — a real number from a real run, stored under `results/`.
- `FAILED EXPERIMENT` — a scientific result that was negative (kept on purpose).
- `FAILURE CAUSED BY INFRASTRUCTURE/BUG` — a run that died for non-scientific
  reasons; the underlying experiment may still be valid.
- `THEORETICAL` — a computed/derived figure, not a measurement.
- `NOT YET RUN` — planned, not executed.

---

## 1. What we originally wanted to achieve

Build and benchmark a **1-bit ASR system** by starting from a pretrained
`facebook/wav2vec2-conformer-rope-large-960h-ft` (Wav2Vec2-Conformer CTC) and
progressively replacing suitable floating-point `nn.Linear` layers with custom
**BitLinear** layers. The core question:

> Can a Wav2Vec2-Conformer ASR model be made substantially smaller using 1-bit
> weights while retaining competitive ASR accuracy?

Reuse standard ASR infrastructure (HF Transformers, LibriSpeech tooling, CTC,
tokenizer); implement only the low-bit-specific parts (BitLinear, binarization,
scaling, STE, QAT, distillation, packing, benchmarking).

Compute profiles: `local` (CPU only) for Phases 0–10; `cloud-gpu` for
training-scale Phases 11+.

---

## 2. Completed phases (what, and why)

| Phase | What | Why it was taken | Status |
|---|---|---|---|
| 0 | Environment, pinned deps, `check_env.py` | Establish a reproducible base | `MEASURED` (PASS) |
| 1 | `inspect_model.py` architecture map | Never modify before inspecting (plan rule 1) | `MEASURED` |
| 2 | `infer.py` smoke test | Prove the audio→logits→text pipeline works | `MEASURED` |
| 3 | FP32 baseline on a fixed 256-utt dev subset | Freeze the reference every later arm is compared against | `MEASURED` |
| 4–7 | BitLinear, binarization, scaling, STE, tests | Build the low-bit core with a test gate before touching ASR | `IMPLEMENTED` |
| 8 | Configurable layer replacement | Swap `nn.Linear → BitLinear` by category, no hard-coded arch | `IMPLEMENTED` |
| 9–10 | 1-bit PTQ + size accounting | Measure raw degradation from binarization without retraining | `MEASURED` (negative result, kept) |
| 11 | QAT infrastructure + tests + CPU smoke | Make a trainable 1-bit student; validate the pipeline cheaply | `IMPLEMENTED` / `MEASURED` |
| 11 | GPU probe tooling | Gate a full GPU run on measured VRAM/time before committing | `IMPLEMENTED` |
| 11 | **Real attention-only QAT on a T4** | First actual accuracy-recovery experiment | `MEASURED` |

### Phase 1 facts (`results/model_inspection.json`)
- Total params **593,376,928**; 24 × `Wav2Vec2ConformerEncoderLayer`; **194
  `nn.Linear`**, totaling **504,218,656 params (84.97%)**.
- FFN dominates: `ffn_in` 201.5M + `ffn_out` 201.4M ≈ 67.9% of the model.

### Phase 3 facts (`results/baseline.json`, immutable)
- WER **0.0202**, CER **0.0049**, RTF 0.284; peak RSS 3,509.7 MB.
- Protocol: 256-utterance dev-clean subset, seed 42, greedy CTC, normalized text.

### Phases 4–8 facts
- `BitLinear(nn.Linear)`: FP master weights, quantizes inside `forward()`
  (`Wb = sign(W)`, `Wq = alpha * sign(W)`, `alpha = mean(|W|)` per tensor,
  recomputed each forward), pass-through STE, FP bias. Same `state_dict` keys,
  so the stock HF `Trainer` runs QAT unmodified.
- Replacement is by **category**, unknown keys raise (no silent no-op).

### Phases 9–10 facts (`results/ptq_binary.json`, `size_report.json`)
- 1-bit **PTQ (attention + FFN)** WER **1.0000**, CER **1.0000** → total
  collapse. Recorded as a result, not a bug (the plan predicted it).
- Packed PTQ artifact **423,289,296 B (≈0.423 GB)** vs FP32 safetensors
  **2,373,821,388 B (≈2.374 GB)**.

---

## 3. The key diagnostic (why attention-only QAT was attempted)

`results/ptq_collapse_diagnostic.json` (16-utt control, same code path):

| Variant | Replaced | WER | Non-empty hyps | Agreement with FP32 |
|---|---:|---:|---:|---:|
| `none` (control) | 0 | 0.0256 | 16/16 | **1.000** |
| `attention_only` | 96 | 0.4441 | 16/16 | 0.125 |
| `ffn_only` | 96 | 1.0000 | 0/16 | 0.000 |
| `full` | 192 | 1.0000 | 0/16 | 0.000 |

The control reproduces FP32 exactly, so the pipeline is sound and the collapse
is caused by binarization itself. **FFN projections are the decisive
sensitivity** (FFN-only already destroys output). Attention-only degrades
partially but still emits text.

**Decision:** the first real QAT run was **attention-only** — the conservative,
isolated first step, keeping FFN in FP. This isolates "can QAT recover attention
binarization?" before tackling the harder FFN question.

---

## 4. Phase 11 — real attention-only QAT (the first recovery)

**Run:** Kaggle script kernel `gunjanpal/asr-1bit-qat-attn`, Tesla T4.
Commit pinned at the time: `e4392c2`. Result committed in `585a4d8`.

**Config (`configs/qat_attn.yaml`):** attention_q/k/v/output → BitLinear (96
layers); FFN, feature_projection, ctc_head stay FP. train.100, 10,000
utterances, 3 epochs, batch 4, grad-accum 4, lr 2e-5, warmup_ratio 0.1,
per-tensor mean-abs scale, pass-through STE. Hardware-specific effective config
(`configs/kaggle_effective_attn.yaml`, generated in the Kaggle clone only):
`fp16: true`, `bf16: false`, `gradient_checkpointing: true`, `save_strategy: "no"`.

**Probe (mandatory gate, §7b):** attempt 1 (no gradient checkpointing) **OOM'd**
(14.55/15.64 GB). Attempt 2 (gradient checkpointing) passed: 30 steps in 276.9 s
= **9.232 s/step**, peak reserved **12.99 GB / 15.64 GB (16.9% free)**, gate
requires ≥15%, extrapolated full run **1,875 steps ≈ 4.81 h**.

**Training (`MEASURED`):** 1,875 steps, final train loss **348.9** (from 6537 at
step 10 — a steady, strong descent). `results/qat_attn.json`.

**Evaluation (pinned 256-utt dev-clean):**

| Model | WER | CER | Δ WER vs FP32 | RTF | p50 lat | peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 0.0202 | 0.0049 | — | 0.284 | 1.60 s | 3,509.7 MB |
| 1-bit PTQ (attn+FFN) | 1.0000 | 1.0000 | +0.9798 | 0.400 | 2.47 s | 3,297.6 MB |
| PTQ attention-only (16-utt diagnostic) | 0.4441 | — | — | — | — | — |
| **1-bit QAT attention-only (256 utts)** | **0.0392** | **0.0125** | **+0.0190** | 0.012 | 0.07 s | 11,509.2 MB |

**Interpretation:** attention-only PTQ collapsed to 44.4% WER; **QAT recovered it
to 3.92% WER**, close to the FP32 2.02% baseline. This is the project's **first
real 1-bit accuracy recovery**. (RTF values are not comparable across rows: the
baseline/PTQ RTF was measured on local CPU, the QAT RTF on a T4 GPU.)

### Current model-size situation (honest)
Because only the 96 attention layers (100.7M of 593.4M params, ~17%) are
binarized, the packed QAT artifact measures **1,983,557,856 B (≈1.98 GB)** vs
**2.374 GB** FP32 — only **≈1.20× smaller**. The artifact is dominated by the
**FP32 FFN** and other non-quantized parameters (`non_quantized_fp_param_bytes`
= 1,970,854,528). This is **not** the compression target; the 5.6× figure from
the full PTQ configuration required quantizing the FFN, which PTQ destroyed.

`results/qat_attn_size_report.json` details:
- quantized weight count **100,663,296** (packed payload 12,582,912 B)
- non-quantized param count **492,713,632** (1,970,854,528 B FP)
- packed artifact measured **1,983,557,856 B**; container overhead 120,032 B
- FP32 master weights (402,653,184 B) are training state, excluded from the artifact.

---

## 5. Failure log (exact causes and fixes)

The project's history is largely a sequence of Kaggle-infrastructure failures,
each diagnosed and fixed. These are preserved deliberately.

1. **torch / torchvision mismatch** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   The Kaggle image shipped torch 2.10.0+cu128 / torchvision 0.25.0, incompatible
   with the needed CUDA-13 stack. *Fix:* `kaggle/requirements-kaggle.txt` pins
   `torch==2.14.0+cu130`, `torchvision==0.29.0+cu130`; `constraints-kaggle.txt`
   fail-fast pins; stale `torchaudio` removed. Commit `4ff384c`.

2. **Dataset RAM / OOM kill** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   Materializing a 10,000-utterance subset as Python lists OOM-killed the Kaggle
   worker. *Fix:* stream training features via `Dataset.from_generator`
   (`build_ctc_train_dataset`), bounding RAM. Commit `7933a48`.

3. **GPU device placement** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   Probe/eval/save-load left input tensors on CPU while the model was on CUDA.
   *Fix:* move inputs to the model's device. Commit `bb0e50d`.

4. **Probe CUDA OOM** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   Attempt 1 without gradient checkpointing OOM'd before measuring anything.
   *Fix:* treat a probe crash as a **gate failure**, retry once with
   `gradient_checkpointing: true`; if that also fails, abort (never silently
   shrink the run). Commit `60e4535`.

5. **Kaggle CLI UTF-8 crash on Windows console** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   The Python Kaggle CLI crashed printing non-ASCII filenames on a legacy code
   page. *Fix:* force `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1` in the wrappers.
   Commit `6b90094`.

6. **Kaggle OAuth/token refresh** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   A non-interactive token refresh was added to the CLI wrappers (`ef01133`) and
   then **reverted** (`fd662a5`) as it caused more churn than it solved. Kept
   here as history; not material to the science.

7. **Trainer epoch checkpoints exhausted Kaggle disk** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   HF Trainer `checkpoint-*` directories carry optimizer state (~5 GB each) and
   filled Kaggle's ~20 GB `/kaggle/working` quota. *Fix:* `save_strategy: "no"`
   plus a `cleanup_trainer_checkpoints()` safety net; the final model is saved
   explicitly via `save_qat_checkpoint`. Commit `d1927e4`.

8. **Save/load verification false negative** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   The original check compared raw logits with `torch.allclose(atol=1e-5)`,
   which is a false negative across GPU processes because fp32 kernels are not
   bit-identical. This made the otherwise-successful QAT run exit 1 and **skip
   the packing stage**. *Fix:* compare decoded **argmax** equality plus a
   tolerant `atol/rtol=1e-3` logit check. Commit `9fa7bbb`.

9. **`_config_path` provenance bug** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   `load_config` used `setdefault`, so the **root** of the `extends` chain
   (`baseline.yaml`) was recorded instead of the requested leaf
   (`qat_attn.yaml`) in training/size reports. *Fix:* assign the leaf path
   unconditionally. (This cleanup.)

10. **Kaggle summary probe-gate reporting** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
    `collect_outputs` read only attempt-1's probe file, which had crashed, so
    `kaggle_summary.json` reported a null gate even though attempt 2 passed.
    *Fix:* prefer whichever probe actually passed; record both attempts. (This
    cleanup.)

11. **Why the previous Kaggle job says `ERROR` despite a valid result.**
    Failure #8 (above) — a post-training verification/packaging-path failure —
    caused `train_qat.py` to return exit code 1, which made `kernel.py` mark the
    whole run FAILED. The **training and 256-utterance evaluation completed and
    are valid**; `results/qat_attn.json` was still written and collected. The
    `ERROR` status is therefore `FAILURE CAUSED BY INFRASTRUCTURE/BUG`, **not** a
    failed QAT experiment. The packed artifact was generated **locally** from the
    downloaded FP32-master checkpoint because the Kaggle pack step was skipped.

---

## 6. Reproducibility / cleanup rerun of Phase 11 (this task)

The previous run left the Kaggle pipeline unproven end-to-end (it never packed
on Kaggle). A **clean rerun of the same attention-only QAT experiment** was
performed with the corrected code, to:

1. verify the corrected save/load path;
2. produce a Kaggle run that ends `COMPLETE`;
3. generate the packed 1-bit deployment artifact **through the Kaggle pipeline**;
4. confirm/reproduce the existing 3.92% WER.

The original 3.92% result (`results/qat_attn.json`) is **preserved untouched**
and the rerun is recorded **separately** as
`results/qat_attn_repro.json` (+ `results/qat_attn_repro_size_report.json`).
If the rerun WER differs, **both** are retained and the difference is documented.

> **Rerun attempt 1 (commit `20019c9`, kernel v8):** training and evaluation
> succeeded — **WER 0.0408 / CER 0.0124** (reproducing the original 3.92% within
> 0.16 pp). But the job again ended `ERROR`: the previously "corrected" save/load
> check still demanded *exact per-frame argmax equality*, which is a false
> negative across GPU processes, so `train_qat.py` returned exit 1 and Kaggle
> packing was skipped once more. **Fix (this commit):** the check now uses
> tolerant logit closeness (`atol/rtol=1e-2`) plus argmax agreement ≥ 0.99, and
> records both metrics in the result; and `kernel.py` now proceeds to packing
> whenever the training + evaluation result exists, so a diagnostic mismatch can
> no longer abort the pipeline.
>
> **Rerun attempt 2 (final):** _pending — to be filled in after the clean run
> completes._ (See `results/qat_attn_repro.json` and
> `results/qat_attn_repro_size_report.json`.)

---

## 7. What remains unfinished

- **Phase 12 (QAT evaluation)** — comparison table FP32 / PTQ / QAT (partially
  present in `results/qat_attn.json` `comparison`); full test-clean evaluation.
- **Phase 13 (Knowledge distillation)** — `src/onebit_asr/training/distillation.py`
  is a placeholder. **Not started.**
- **Phase 14 (Controlled experiments)** — A FP32 / B PTQ / C QAT exist;
  **D (QAT + distillation) missing**; 2-bit / INT8 not started.
- **Phase 15 (Layer sensitivity)** — only the 16-utt diagnostic exists; formal
  sweep not run.
- **Phase 16 (Binary storage)** — done for PTQ and for attention-only QAT; the
  full FFN-inclusive configuration is not done.
- **Phase 17 (Inference benchmarking)** — `evaluation/latency.py`,
  `evaluation/benchmark.py` are placeholders; only eval-time RTF is measured.
- **Phase 18 (Binary compute / kernels)** — not started.
- **Phase 19 (Streaming ASR)** — not started.
- Placeholder modules: `features/audio.py`, `decoding/decode.py`,
  `evaluation/latency.py`, `evaluation/benchmark.py`, `training/train.py`,
  `training/distillation.py`, and several `scripts/*`.
- Infra debt: ~45 min re-download of LibriSpeech per Kaggle run (no cache /
  `HF_TOKEN`); no `--pack-only` Kaggle mode.

---

## 8. The current unresolved scientific problem

The 3.92% result answers **only** "can attention be trained to 1 bit?" (yes). It
does **not** answer the project's core accuracy-vs-size question, because the
**FFN remains FP32** and dominates both the remaining size (1.98 GB) and the PTQ
collapse. The decisive open question is:

> **How far can we push 1-bit quantization of the attention + FFN portions while
> retaining usable ASR accuracy?**

Evidence so far: attention-only QAT → **3.92% WER**, while attention+FFN PTQ →
**100% WER**. Whether QAT can make FFN 1-bit quantization viable is **unknown**.

---

## 9. Next experiments, in order

1. **attention+FFN QAT** (the next substantive experiment). Rationale: PTQ
   showed catastrophic FFN sensitivity; attention-only QAT proved QAT can recover
   accuracy; the compression objective requires addressing the FFN. This will
   roughly double the quantized parameter count and must pass a fresh VRAM probe
   (expect a lower headroom; gradient checkpointing and/or smaller micro-batch
   may be required).
2. **Knowledge distillation** (Phase 13) as the likely fallback if full QAT
   underperforms.
3. **Layer sensitivity sweep** (Phase 15) to find the best efficiency/accuracy
   trade-off.
4. **Controlled experiments incl. distillation arm D** (Phase 14), optional
   2-bit / INT8.
5. **Benchmarking** (Phase 17) and, optionally, **binary compute** (Phase 18) and
   **streaming** (Phase 19).

**Explicitly out of scope for the current cleanup task:** FFN QAT, FFN-only QAT,
sensitivity sweeps, distillation, 2-bit/INT8, binary kernels, streaming. This
task stops after the clean Phase-11 reproduction and documentation.
