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

12. **Kaggle weekly GPU quota exhausted** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
    After the v5–v8 runs, `kaggle quota` reported **30.01 h / 30.00 h used**
    (`refreshAt 2026-09-26`), so the final clean reproduction push was rejected
    with `Kernel push error: Maximum weekly GPU quota of 30.00 hours reached`.
    The Kaggle `COMPLETE` reproduction is deferred until after the reset. (The
    submit wrapper previously ignored this error and printed "Submitted"; fixed.)

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
> **Rerun attempt 2 (final): BLOCKED by the Kaggle weekly GPU quota.** Pushing the
> corrected commit (`cd438a8`, kernel v9) was rejected with
> `Kernel push error: Maximum weekly GPU quota of 30.00 hours reached`
> (`kaggle quota`: **30.01 h used / 30.00 h**, `refreshAt 2026-09-26`). The clean
> Kaggle `COMPLETE` run and the Kaggle-side packing therefore remain **pending
> until the quota refreshes on 2026-09-26**. (The submit wrapper was also fixed:
> it now treats `Kernel push error` as a failure instead of falsely reporting
> success.)
>
> **Offline reproduction artifacts (from attempt 1's valid checkpoint, v8):**
> - `results/qat_attn_repro.json` — real measured rerun: **WER 0.0408 / CER 0.0124**
>   vs the original 0.0392 → difference **+0.0016 (+0.16 pp)**; both are kept.
> - `results/qat_attn_repro_size_report.json` — packed deploy artifact measured
>   **1,983,557,856 B**, generated **locally** from attempt 1's checkpoint (the
>   Kaggle-side pack step was skipped; the report's `platform` shows Windows).
> - `results/qat_attn_repro_probe_gc.json` — probe attempt 2 (GC) PASS,
>   **9.457 s/step**, 16.9% VRAM headroom.
>
> These are honest offline reproductions, **not** the requested Kaggle `COMPLETE`
> run; that run is queued for after the quota reset.

---

## 7. What remains unfinished

> **Historical snapshot (v1.4.0).** Superseded by §10: Phases 12–19 are now
> implemented and measured (full test-clean, distillation, sensitivity,
> benchmarking, packed inference, streaming). Kept unchanged for history.

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

---

## 10. v1.5.0 — Phase 12–19 execution on the A100 cluster (measured)

This section records the second execution profile: the local PBS cluster
(8 × A100-SXM4-80GB, one GPU node). The execution layer lives in `cluster/` and
is documented in `cluster/README.md`. Every number below is backed by a JSON
under `results/`; the generated tables are in `RESULTS.md`.

### 10.1 Why a second profile

The Kaggle T4 runs produced the first real QAT result (attention-only 3.92%),
but the project's decisive question — whether **FFN** binarization is recoverable
— needs more compute than a 16 GB T4 comfortably provides, and the Kaggle GPU
quota was exhausted. The A100 node gives 80 GB per GPU, so attention+FFN QAT and
distillation run without gradient checkpointing.

### 10.2 Environment (measured, not assumed)

- `torch 2.11.0+cu126`, `torchvision 0.26.0+cu126`, `transformers 4.57.6`,
  `datasets 5.0.1`, `numpy 2.5.3`, `jiwer 4.0.0`.
- Project venv: `ASR_1bit/.venv` with `--system-site-packages`; package installed
  editable (`pip install -e . --no-deps`).
- Audio: `ONEBIT_AUDIO_DECODER=soundfile` (torchcodec is not installed on the
  cluster; this is the same fallback the Kaggle runner uses).
- The pinned Kaggle stack (torch 2.14 / transformers 5.17) is **not** what runs
  here. Results record `versions` and stay profile-local.

### 10.3 GPU selection

PBS sets `CUDA_VISIBLE_DEVICES` to an **empty string** and does not pin GPUs.
`cluster/pick_gpu.sh` unsets it, orders the 8 GPUs by free memory then
utilization, claims the chosen device for 5 minutes in
`~/.onebit_gpu_reservations` (atomic `mkdir` lock), and exports it before Python
starts. The reservation was added after two simultaneous jobs both selected the
same idle A100; it prevents that class of co-location.

### 10.4 Jobs (final status)

| Job | Phase | CPUs | Result |
|---|---|---:|---|
| `qat_attn_ffn.pbs` | 11 — attention+FFN QAT | 8 | `0.2082 → 0.2080` dev WER, 1875 steps, packed 423.3 MB |
| `qat_attn_a100.pbs` | 12/14 — attention-only re-run | 4 | `0.0421` dev / `0.0439` test, packed 1,983.6 MB |
| `qat_ffn_a100.pbs` | 15 — FFN-only QAT | 1 | `0.1186 → 0.1190` dev / `0.1095` test, packed 813.3 MB |
| `distill_attn_ffn.pbs` | 13 — distillation, λ_kd=0.5 | 1 | `0.2232` dev / `0.2167` test, packed 423.3 MB |
| `distill_attn_ffn_kd01.pbs` | 13/14 — λ_kd=0.1 follow-up | 1 | `0.2022` dev (vs 0.2080 QAT), packed 423.3 MB |
| `final_eval_a100.pbs` | 12/14/15/17/18/19 | 1 | dev+test comparison, packed and streaming results |

The CPU quota (≈14–16 CPUs for this user, including a pre-existing 4-CPU job)
was the scheduler bottleneck, not GPU availability. Resizing the distillation and
FFN-only jobs to **1 CPU** let them run concurrently; the FFN-only job still
completed 1875 steps in 50:48.

### 10.5 Measured — accuracy/size trade-off

Pinned protocol: fixed 256-utterance greedy-CTC subsets, seed 42, normalized
text. The dev numbers below are the final-eval pass (the training-script eval
differs by ≤0.0004 from floating-point nondeterminism and is kept in its own
JSON). Test numbers are a fixed 256-utterance test-clean subset.

| Arm | Dev WER | Dev CER | Test WER | Test CER | Deploy size | vs FP32 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.0202 | 0.0049 | 0.0162 | 0.0044 | 2.374 GB | 1.0× |
| PTQ attention-only | 0.4454 | 0.2180 | — | — | 1.983 GB | 1.2× |
| PTQ FFN-only | 1.0000 | 1.0000 | — | — | 0.813 GB | 2.9× |
| PTQ attention+FFN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.423 GB | 5.6× |
| QAT attention-only | **0.0421** | **0.0128** | **0.0439** | **0.0131** | 1.983 GB | 1.2× |
| QAT FFN-only | **0.1190** | **0.0338** | **0.1095** | **0.0318** | 0.813 GB | 2.9× |
| QAT attention+FFN | **0.2080** | **0.0640** | **0.2029** | **0.0648** | 0.423 GB | 5.6× |
| QAT + KD (λ_kd=0.5) | 0.2232 | 0.0712 | 0.2167 | 0.0706 | 0.423 GB | 5.6× |
| QAT + KD (λ_kd=0.1) | **0.2022** | **0.0641** | **0.1986** | **0.0647** | 0.423 GB | 5.6× |

**Full test-clean (2620), accuracy-only:** FP32 **0.0193 / 0.0050**;
attention-only **0.0449 / 0.0137**; FFN-only **0.1134 / 0.0325**;
attention+FFN **0.2050 / 0.0639**; KD λ_kd=0.5 **0.2218 / 0.0709**; KD λ_kd=0.1
**0.1998 / 0.0636**. The ranking is identical to the 256-utterance subsets.

**The scientific result.** QAT recovers every isolated selection: attention-only
from 44.5% to 4.21%, FFN-only from 100% to 11.90%. The combined attention+FFN
arm recovers from 100% to 20.80%, which is **worse than either isolated arm** —
the two binarizations interfere. The 5.6× compression target is reachable, but
at ~10× the FP32 WER on this recipe. The best accuracy/size points measured are:

- 1.2× smaller at 4.21% WER (attention-only),
- 2.9× smaller at 11.90% WER (FFN-only),
- 5.6× smaller at 20.80% WER (attention+FFN).

Distillation at λ_kd=0.5 did **not** improve the combined arm (22.32%). The
logged loss components explain the failure mode: the summed KL term averaged
6107 vs CTC 726, so the student optimized almost entirely the teacher
distribution. Lowering λ_kd to 0.1 (same teacher/logits, same budget) improves
the combined arm to **20.22%** (−0.60 pp vs QAT alone) at the same 0.423 GB;
the KL term is still ~11× the CTC term (6385 vs 559), so the gain is marginal.

### 10.6 Measured — PTQ layer sensitivity (256 dev utterances)

| Selection | WER | CER | Deploy size |
|---|---:|---:|---:|
| attention-only | 0.4454 | 0.2180 | 1.983 GB |
| FFN-only | 1.0000 | 1.0000 | 0.813 GB |
| attention+FFN | 1.0000 | 1.0000 | 0.423 GB |
| + feature_projection | 1.0000 | 1.0000 | 0.421 GB |
| + ctc_head (all linears) | 1.0000 | 1.0000 | 0.421 GB |

This is the full 256-utterance confirmation of the earlier 16-utterance
diagnostic: FFN-only PTQ already destroys the output, and adding
feature_projection/ctc_head changes nothing (the model is already collapsed).

### 10.7 Measured — performance, packed inference, streaming

- **Performance (A100, warmed-up):** every arm has RTF ≈0.006–0.007 end-to-end,
  p50 ≈0.035–0.046 s, peak VRAM ≈4.4 GB (dev) / ≈6.0 GB (test). Binarization
  does not accelerate inference; the runtime expands `alpha·sign(W)` before
  `F.linear`. The small RTF differences track model/parameter count, not
  precision.
- **Phase 18:** 192 packed layers, 62.9 MB packed weights (vs 358.9 MB
  non-packed FP + 1.38 MB scales/biases). BitLinear CPU forward 1.657 s vs
  PackedBitLinear 1.826 s (**1.10× slower**). The XNOR+popcount reference
  exactly matches the ±1 dot product and is **2.3× faster than float** for
  1-bit×1-bit, but this model has FP activations, so it cannot use that kernel.
- **Phase 19:** 4.45 s sample, 2 s chunks / 0.5 s overlap, 3 chunks. Chunked
  WER **0.6923** / CER 0.5072 vs full-utterance **0.1538** / 0.0580; chunked
  RTF 0.175. The merge duplicates words when the overlap does not cover a word;
  this is an honest pseudo-streaming prototype, not a streaming encoder.

### 10.8 Failure log additions (A100 work)

1. **Final-eval crash on an in-progress checkpoint directory**
   (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`). `Trainer` creates `output_dir` when
   a job starts, so `is_dir()` treated a concurrently training arm as usable;
   `from_pretrained` raised `OSError` and aborted the whole dev comparison.
   *Fix:* require `model.safetensors`/`pytorch_model.bin`, and catch per-arm
   errors so one bad arm cannot abort the table.
2. **Distillation probe device mismatch** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`).
   The probe moved the batch to CUDA while the student was still on CPU.
   *Fix:* derive the device from the model and move the student before the probe.
   The same change makes the QAT probe run on GPU instead of silently on CPU.
3. **Save/load false negatives** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`). The
   1e-2 logit tolerance rejected functionally identical checkpoints: argmax
   agreement was 1.000 while max |Δ logit| was 0.028–0.063. *Fix:* 1e-1
   tolerance + argmax agreement ≥0.99. Historical JSONs keep their boolean and
   record the agreement; the checkpoints are functionally identical.
4. **GPU selection race** (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`). Two jobs
   submitted together both selected GPU 5 before either allocated memory.
   *Fix:* atomic reservation directory + free-memory-first ordering.
5. **CPU quota, not GPU availability, was the bottleneck**
   (`FAILURE CAUSED BY INFRASTRUCTURE/BUG`). Jobs queued with “Insufficient
   ncpus” while A100s were free. *Fix:* resize GPU-bound jobs to 1 CPU and run
   them concurrently.

### 10.9 Future/optional work (not required phases)

- A recipe that improves the attention+FFN arm beyond the marginal distillation
  gain (longer training, LR/early-stop sweep, mean-normalized KD, layer-wise
  selection).
- True binary compute (XNOR+popcount with 1-bit activations), 2-bit/INT8,
  per-channel/learned scaling.
- LM decoding.
- A true streaming encoder with cached state.

These are explicitly future research. The required phases (0–17, with 18/19
investigated/prototyped to the plan's optional scope) are complete.

### 10.10 Completion audit

| Phase | Final state |
|---|---|
| 0–10 | complete; tests + FP32/PTQ/size results |
| 11 | complete on T4 and A100 (attention, FFN, attention+FFN) |
| 12 | complete on dev 256, test 256, full test-clean 2620 |
| 13 | complete for λ_kd=0.5 and 0.1 |
| 14 | complete for FP32/PTQ/QAT/QAT+KD; 2-bit/INT8 optional, not run |
| 15 | complete for PTQ and QAT component selections |
| 16 | complete; measured packed artifacts for every trained arm |
| 17 | complete on A100 dev 256; full-test accuracy-only run is labelled |
| 18 | investigated: packed storage + XNOR boundary; integrated kernel is future work |
| 19 | prototype measured: chunked pseudo-streaming; true streaming encoder is future work |
| Kaggle clean repro | unavailable (no CLI/credentials; quota exhausted); superseded by the A100 re-run |

**Final full test-clean WER:** FP32 0.0193; attention-only 0.0449 (best 1-bit,
1.983 GB); FFN-only 0.1134 (0.813 GB); attention+FFN 0.2050 (0.423 GB);
attention+FFN + KD λ_kd=0.1 **0.1998** (0.423 GB).

### 10.11 v1.6 mixed-precision pilot (measured 2026-10-26)

The per-output-channel screen failed its 10% gate (2.3–2.4%), so no per-channel
QAT was run. The one-layer FFN sensitivity screen then selected 17 sensitive
layers (union of top 12 by relative L2 and top 8 by WER delta; 14 `ffn_in`,
3 `ffn_out`). The controlled pilot (`cluster/qat_ffn_pilot.pbs`, job `12772`)
compared per-tensor FFN-only QAT against the mixed policy:

| Arm | Dev-128 WER | CER | Best/final step | Artifact |
|---|---:|---:|---:|---:|
| FFN-only control (96 BitLinear) | 0.2303 | 0.0684 | 500 / 500 | 813,346,080 B |
| Mixed FFN (79 BitLinear + 17 FP32) | **0.0900** | 0.0284 | 500 / 500 | 1,089,643,604 B |

Both arms used the same 2,000 training IDs, 128 eval IDs, seed 42, batch 4 ×
accum 4, lr 2e-5, and 500 steps. Mixed minus control: **−14.04 pp**. Both
checkpoints restored the best state (which was step 500), and both packed
artifacts reload with max |Δ logit| 0. This was a **dev-128 pilot**, not a
full-test result.

### 10.12 v1.6 full mixed-precision result (measured 2026-10-26)

The full 10k/1,875-step controlled run (job `12848`, 1 CPU, A100) and the final
test evaluation (job `12899`) completed:

| Arm | Dev-256 | Test-256 | Full test-clean | Artifact |
|---|---:|---:|---:|---:|
| FFN-only control | 0.1163 | 0.1124 | 0.1132 | 0.8133 GB |
| Mixed FFN (79 1-bit + 17 fixed FP32) | **0.0527** | **0.0516** | **0.0546** | 1.0896 GB |

The mixed arm is a new Pareto point: 2.18× smaller than FP32 at 5.46% full-test
WER, versus the control's 2.92× at 11.32%. Attention-only remains the absolute
accuracy leader at 4.49% / 1.983 GB. The 17-layer list was fixed before the run;
test evaluation was one pass after selection. The full job's step-1,875 best
state was restored and both packed artifacts reload exactly.



