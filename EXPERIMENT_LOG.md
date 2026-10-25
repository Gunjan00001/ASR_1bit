# Experiment log

This is the project's engineering lab notebook. It answers four questions for
every experiment:

1. **What did we do?**
2. **Why did we do it?**
3. **Did it work?**
4. **If not, why — and what did we do instead? What worked, and why?**

The forward-looking plan is `one_bit_asr_implementation_plan.md`; the scientific
history is `planning.md`; the measured tables are `RESULTS.md`; the teaching guide
is `deepdive.md`. This file is the *reasoning* log that connects them.

Status labels:

- **WORKED** — the hypothesis was supported by the measured metric.
- **PARTIAL** — it worked, but not enough / not in the way we hoped.
- **DID NOT WORK** — the experiment produced a negative result (kept on purpose).
- **INFRA FAILURE** — the run died for infrastructure reasons; the experiment may
  still be valid.
- **OPEN** — not answered yet.

---

## Timeline

| ID | Date | Experiment | Status |
|---|---|---|---|
| E0 | 2026-09-12 | Environment + dependencies | WORKED |
| E1 | 2026-09-12 | Model inspection / parameter map | WORKED |
| E2 | 2026-09-12 | Inference smoke test | WORKED |
| E3 | 2026-09-14 | FP32 baseline (256 dev) | WORKED (reference) |
| E4–E6 | 2026-09-14 | BitLinear, binarization, scaling, STE | WORKED (tests) |
| E7 | 2026-09-14 | Configurable layer replacement | WORKED |
| E8 | 2026-09-14 | 1-bit PTQ | DID NOT WORK (expected) |
| E9 | 2026-09-14 | PTQ collapse diagnostic | WORKED (found FFN) |
| E10 | 2026-09-15 | QAT infrastructure + CPU smoke | WORKED (pipeline only) |
| E11 | 2026-09-20 | Attention-only QAT (Kaggle T4) | WORKED (3.92%) |
| E12 | 2026-09-21 | Attention-only QAT reproduction | WORKED (4.08%) |
| E13 | 2026-10-25 | Attention-only QAT (A100) | WORKED (4.21%) |
| E14 | 2026-10-25 | FFN-only QAT (A100) | WORKED (11.90%) |
| E15 | 2026-10-25 | Attention+FFN QAT (A100) | PARTIAL (20.80%) |
| E16 | 2026-10-25 | Distillation λ_kd=0.5 | DID NOT WORK (22.32%) |
| E17 | 2026-10-25 | Distillation λ_kd=0.1 | PARTIAL (20.22%) |
| E18 | 2026-10-25 | PTQ sensitivity sweep | WORKED (confirmed) |
| E19 | 2026-10-25 | Inference benchmark | WORKED (negative speed result) |
| E20 | 2026-10-25 | Packed 1-bit inference | PARTIAL (storage yes, speed no) |
| E21 | 2026-10-25 | Chunked streaming | PARTIAL (prototype) |
| E22 | 2026-10-25 | Full test-clean evaluation | WORKED (ranking confirmed) |
| E23 | 2026-10-25 | Completion audit (v1.5.2) | WORKED |
| E24 | 2026-10-25 | Per-output-channel reconstruction screen | DID NOT WORK (below gate) |
| E25 | 2026-10-25 | FFN per-layer sensitivity screen | WORKED (screening) |
| E26 | 2026-10-25 | Mixed-precision FFN implementation + pilot | PARTIAL (pilot blocked) |

---

## E0 — Environment and dependencies

**Question:** can a clean environment install everything, and can we detect the
hardware honestly?

**Why:** every later result depends on the versions and the compute profile. A
silent CPU/GPU mix-up would invalidate every benchmark.

**What:** created a venv, pinned `torch`, `transformers`, `datasets`, `jiwer`,
etc., wrote `scripts/check_env.py` to import everything and report versions/CUDA.

**Did it work?** **Yes.** `results/env.json` shows Python 3.14.3, torch 2.14.0
CPU, `cuda_available=false`, 8 threads.

**What we learned:** the local machine is CPU-only. Training-scale phases must be
cloud-GPU. This led to the two-profile (`local` / `cloud-gpu`) rule that every
result follows.

---

## E1 — Model inspection

**Question:** which linear layers exist, how big are they, and which are plausible
quantization candidates?

**Why:** the plan's first rule is "inspect before modifying." Guessing layer
names is how you quantize nothing and report a fake win.

**What:** `scripts/inspect_model.py` loaded the checkpoint read-only and produced
`results/model_inspection.json` + the shared taxonomy.

**Did it work?** **Yes.** 593,376,928 parameters; 24 Conformer layers; 194
`nn.Linear`; 504,218,656 linear parameters (84.97%). FFN = 402.9M (67.9%),
attention = 100.8M (17.0%).

**What we learned:** the FFN is the size prize. Attention-only compression cannot
shrink the model much. This directly motivated the later FFN experiments.

---

## E2 — Inference smoke test

**Question:** does the audio → processor → model → CTC decode pipeline work?

**Why:** a broken decode path would make every WER number meaningless.

**What:** `scripts/infer.py --demo` on a real LibriSpeech sample.

**Did it work?** **Yes.** The transcription matched the reference exactly.

---

## E3 — FP32 baseline

**Question:** what is the reference WER/CER/latency on a fixed subset?

**Why:** every 1-bit arm must be compared against a frozen, reproducible
reference. Without it, "3.9%" has no meaning.

**What:** 256-utterance dev-clean subset, seed 42, greedy CTC, lowercase/strip
punctuation, CPU.

**Did it work?** **Yes.** `results/baseline.json`: **WER 0.0202, CER 0.0049,
RTF 0.284**, peak RSS 3.5 GB. This matched the model card's 1.96% test-clean
reference closely, confirming a sane setup.

**What we learned:** the protocol is stable. The 256-utterance subset is the
reference for all local work; the full test-clean is the final check.

---

## E4–E6 — BitLinear, sign quantization, scaling, STE

**Question:** can we build a drop-in 1-bit layer that trains with normal
PyTorch?

**Why:** the custom low-bit parts are the project's actual contribution. Reusing
`nn.Linear` with `state_dict` compatibility lets the stock HF `Trainer` work.

**What:**

- `BinaryQuantize` custom autograd function: `sign` forward, straight-through
  backward.
- `alpha = mean(|W|)` per tensor, recomputed each forward, not learned.
- `Wq = alpha * sign(W)`.
- `BitLinear(nn.Linear)` keeps FP32 masters and quantizes inside `forward`.

**Did it work?** **Yes.** The test gate verifies ±1 weights, scaling, gradient
flow, optimizer updates, FP masters, save/load, and a toy task that reduces loss.
`sign(0)=0` is documented; the clipped STE is opt-in.

**What we learned:** keeping FP masters is the key design decision. It makes QAT,
checkpointing, and `from_pretrained` all work without custom Trainer code.

---

## E7 — Layer replacement

**Question:** can we swap `nn.Linear → BitLinear` by category without hard-coding
the architecture?

**Why:** the taxonomy must be shared by inspection, replacement, and reporting;
otherwise a typo silently quantizes nothing.

**What:** `conformer_utils.categorize` + `replace_linears`; unknown keys raise;
replacement is idempotent and weight-preserving.

**Did it work?** **Yes.** Applying the full selection swapped 192 layers /
503.3M params; the retained layers were feature projection + CTC head. Tests
cover naming, counts, weights, forward, and idempotency.

**What we learned:** `ffn_in`/`ffn_out` must be distinct keys. The old `ffn` key
would have silently matched nothing.

---

## E8 — 1-bit PTQ

**Question:** how much accuracy does binarization cost without retraining?

**Why:** this is the honest lower bound. The plan predicted severe degradation;
we needed to measure it, not tune around it.

**What:** loaded the FP32 checkpoint, replaced attention + FFN, evaluated the
pinned 256-utterance dev subset with no training.

**Did it work?** **No — and that was the expected result.** WER **1.0000**, CER
1.0000, hypotheses empty (`results/ptq_binary.json`).

**Why it failed:** the fine-tuned network was never adapted to ±alpha weights.
Every selected weight changes magnitude and sign; errors compound through 24
layers; the FFN is the most sensitive part. There is no mechanism in PTQ to
compensate.

**What we did instead:** the control experiment E9, then QAT.

---

## E9 — PTQ collapse diagnostic (the important control)

**Question:** is the 100% WER a real quantization effect or a code bug?

**Why:** a collapsed metric can hide a device/dtype/replacement bug. We needed a
control that must reproduce FP32 exactly.

**What:** same replacement + evaluation code path, four variants on 16 fixed
utterances:

| Variant | Replaced | WER | Non-empty | Agreement with FP32 |
|---|---:|---:|---:|---:|
| none (control) | 0 | 0.0256 | 16/16 | **1.000** |
| attention-only | 96 | 0.4441 | 16/16 | 0.125 |
| FFN-only | 96 | 1.0000 | 0/16 | 0.000 |
| full | 192 | 1.0000 | 0/16 | 0.000 |

**Did it work?** **Yes.** The control reproduces FP32 exactly, so the pipeline is
sound. The collapse tracks binarization, and **FFN-only is already fatal**.

**Why it matters:** it changed the next experiment. Instead of jumping to full
QAT, we first asked "can QAT recover attention?" (the less sensitive part).

---

## E10 — QAT infrastructure and CPU smoke

**Question:** can the QAT pipeline run end-to-end, and do gradients reach the
BitLinear masters?

**Why:** a full GPU run is expensive; a cheap pipeline check must come first.
The plan also required proving that the FP masters update and stay FP.

**What:** QAT assembly (`setup_qat_model`, freeze, gradient probe, stock
Trainer), CTC collator, leakage guard, FP32-master checkpointing. CPU smoke: 16
train.100 utterances, 8 steps.

**Did it work?** **Yes, as a pipeline check.** `192/192` BitLinear masters got
gradients; the frozen feature extractor got zero; the optimizer changed the
masters (digest changed); the masters stayed FP (fraction exactly ±1 = 0.0);
save/load verified; smoke WER 1.0000 (reported, not an accuracy experiment).

**What we learned:** the gradient probe must run **before** training, because the
Trainer zeroes gradients after each step. This became an explicit design note.

---

## E11 — Attention-only QAT on a Kaggle T4

**Question:** can QAT recover the attention-only PTQ collapse (44.4% → ?)? 

**Why:** attention-only was the conservative first step; it isolates the easier
half of the problem before touching the FFN.

**What:** 96 attention layers → BitLinear; FFN/heads FP; train.100, 10,000
utterances, 3 epochs, batch 4 × accum 4, lr 2e-5, fp16, 1875 steps. Mandatory
GPU probe with a 15% VRAM-headroom gate.

**Did it work?** **Yes.** WER **0.0392**, CER **0.0125** on 256 dev utts. The
first real 1-bit accuracy recovery: 44.4% → 3.92%, close to the 2.02% FP32.

**What went wrong along the way (infra):**

- Probe attempt 1 OOM'd at 14.55/15.64 GB. We treated an OOM as a gate failure
  and retried once with gradient checkpointing; attempt 2 passed at 16.9% free.
- The save/load check returned a false negative on GPU (exact logit equality),
  causing exit 1 and skipping the packing stage. The training/eval result was
  valid. This became a recurring lesson: **compare decoded output, not bits.**

**Size result:** 1.983 GB packed vs 2.374 GB FP32 (1.20×). Only 17% of the
parameters were binarized; the FP32 FFN dominates. This made the FFN experiment
the next priority.

---

## E12 — Attention-only QAT reproduction

**Question:** does the 3.92% result reproduce on a clean run?

**Why:** the first run had an infrastructure failure during packing; a clean
reproduction was needed to prove the result and the pipeline.

**What:** same config, same data, corrected save/load check.

**Did it work?** **Yes, mostly.** WER **0.0408**, CER 0.0124, +0.16 pp vs the
original. The run again ended `ERROR` because the first "corrected" check still
demanded exact per-frame argmax equality. The final clean Kaggle run was then
blocked by the weekly GPU quota (30.01/30.00 h).

**What we changed:** tolerant logit closeness + argmax agreement, and the runner
now proceeds to packing whenever training + evaluation produced a result. The
historical result is kept untouched; the reproduction is a separate file.

**What we learned:** packaging/verification failures should never invalidate a
scientific result. Separate "experiment failed" from "infrastructure failed" in
the result schema.

---

## E13 — Attention-only QAT on an A100

**Question:** can we reproduce the T4 result on a different GPU/profile?

**Why:** the Kaggle checkpoint was not in the repo; a like-for-like A100 arm was
needed for the final comparison, and the A100 gives room for FFN work.

**What:** `cluster/qat_attn_a100.pbs`, 4 CPUs, 1 GPU, fp16, no gradient
checkpointing, 1875 steps.

**Did it work?** **Yes.** Dev WER **0.0421**, full test-clean **0.0449**,
packed 1,983,557,856 B. The raw final loss (97.2) is much lower than the T4 run
(348.9) because the stacks differ; WER is the comparison.

**What we learned:** the conclusion is robust across T4/A100, but raw losses and
RTF are not comparable across profiles. Label them.

---

## E14 — FFN-only QAT

**Question:** can QAT recover the FFN-only PTQ collapse (100% → ?)? 

**Why:** the FFN is 68% of the parameters and the decisive PTQ sensitivity. The
project's compression target depends on it.

**What:** 96 FFN layers → BitLinear; attention/heads FP; same 10k/3-epoch
budget; 1 CPU because the per-user quota was the bottleneck.

**Did it work?** **Yes, surprisingly well.** Dev WER **0.1190**, full test-clean
**0.1134**, packed **0.813 GB** (2.9× smaller). Training loss 165.

**Why it worked:** the FFN is wide and over-parameterized; its FP masters have
enough redundancy to learn weights that survive sign quantization. The FFN-only
PTQ collapse was catastrophic because there was no adaptation, but QAT supplies
it.

**What it changed:** it made the combined attention+FFN result the key question,
because FFN-only alone is a useful 2.9× compression point.

---

## E15 — Attention+FFN QAT (the decisive experiment)

**Question:** can QAT recover the full 192-layer binarization that collapsed to
100% under PTQ?

**Why:** it reaches the 5.6× compression target and answers the project's core
question.

**What:** 192 layers → BitLinear; 10k train.100, 3 epochs, batch 4 × accum 4,
lr 2e-5, fp16, no gradient checkpointing, 1875 steps; packed deploy artifact.

**Did it work?** **Partially.** Dev WER **0.2080**, full test-clean **0.2050**,
packed **0.423 GB** (5.6×). It is a huge recovery from 100% PTQ, but ~10× worse
than FP32 and **worse than FFN-only** (0.1190/0.1134).

**Why it did not fully work:** the two binarizations interact. The combined
training loss (298.6) is not obviously worse than the isolated arms (97.2 /
165.4), so this is not simple underfitting. Hypotheses:

1. **Capacity:** removing both attention and FFN precision leaves too little.
2. **Conditioning:** the per-tensor scale/STE optimization is harder with 192
   quantized layers.
3. **Error compounding:** attention and FFN errors interact differently when both
   are present.
4. **Generalization/overfitting:** the train loss is lower but dev WER is worse.

We have **not** isolated the cause. That is the main open scientific question.

**What we did instead:** distillation (E16/E17), to see whether the FP32 teacher
could recover the lost accuracy.

---

## E16 — Distillation, λ_kd=0.5

**Question:** can an FP32 teacher improve the combined 1-bit student?

**Why:** distillation is the plan's fallback when QAT alone underperforms; the
teacher has exactly the information the student lost.

**What:** frozen FP32 teacher; one-shot precomputed teacher logits for the 10k
subset; student = attention+FFN 1-bit; loss
`0.5·CTC + 0.5·T²·KL(student/T ‖ teacher/T)`, T=2; same budget.

**Did it work?** **No.** Dev WER **0.2232**, full test-clean **0.2218** — slightly
worse than QAT alone (0.2080/0.2050).

**Why it failed:** the summed KL dominated the CTC term by ~8× (6107 vs 726). The
student optimized the teacher distribution almost exclusively. The teacher is
full-precision; the 1-bit student cannot match it, and the KL gradient is noisy.
The loss stayed high (~2500 at the end) instead of converging like QAT.

**What we did instead:** a controlled λ change (E17), reusing the cached teacher
logits.

---

## E17 — Distillation, λ_kd=0.1

**Question:** was λ_kd=0.5 simply too large? 

**Why:** a controlled experiment: change only λ_kd, reuse the same teacher
logits, data, layers, LR, and epochs. If the failure was the weight, this should
show it.

**What:** λ_kd 0.5 → 0.1; everything else identical.

**Did it work?** **Partially / yes.** Dev WER **0.2022**, full test-clean
**0.1998**, a consistent ~0.6 pp improvement over QAT alone at the same 0.423 GB.
Save/load verified (the new tolerance).

**Why it worked better:** the CTC term now has relatively more weight; the KL
still dominates (6385 vs 559), but less. The teacher acts as a regularizer rather
than the whole objective.

**Why it is still marginal:** the KL sum is still ~11× the CTC sum. Mean-normalized
KL or an even smaller λ might help more; not measured yet.

---

## E18 — PTQ sensitivity sweep (256 dev utterances)

**Question:** which components are sensitive under PTQ, on the full protocol?

**Why:** the E9 diagnostic used 16 utterances; we wanted the full 256-utterance
confirmation and extra variants.

**What:** PTQ with attention-only, FFN-only, attention+FFN, +feature_projection,
and all linears (including CTC head).

**Did it work?** **Yes, it confirmed the picture.**

| Selection | WER | CER | Packed size |
|---|---:|---:|---:|
| attention-only | 0.4454 | 0.2180 | 1.983 GB |
| FFN-only | 1.0000 | 1.0000 | 0.813 GB |
| attention+FFN | 1.0000 | 1.0000 | 0.423 GB |
| + feature_projection | 1.0000 | 1.0000 | 0.421 GB |
| + CTC head (all) | 1.0000 | 1.0000 | 0.421 GB |

**What we learned:** quantizing the tiny heads changes nothing once the model has
already collapsed; their size benefit is negligible. Attention is the only
component that survives PTQ partially.

---

## E19 — Inference benchmark

**Question:** does 1-bit storage make inference faster?

**Why:** the project must not claim a speedup it did not measure. The plan
explicitly warns that a runtime can expand weights back to FP.

**What:** warmed-up latency/RTF/throughput/VRAM on the A100, all arms, same
protocol. Separate from the WER evaluator.

**Did it work?** **Yes, and the answer is no speedup.** Every arm: RTF
≈0.006–0.007, p50 ≈0.035–0.045 s, VRAM ≈4.4 GB (dev 256). The difference between
arms is <10% and tracks model size, not precision.

**Why:** `BitLinear.forward` expands `alpha·sign(W)` to FP and calls `F.linear`.
The GPU still does FP matmuls. The model is smaller on disk, not faster in the
loop.

**What we did instead:** measured the packed path (E20) and the XNOR boundary.

---

## E20 — Packed 1-bit inference and the binary-compute boundary

**Question:** can we serve the model from packed 1-bit weights, and can a bitwise
kernel accelerate it?

**What:** `PackedBitLinear` stores `numpy.packbits(sign(W))` + alpha + bias; its
forward unpacks to ±1 floats and calls `F.linear`. A reference XNOR+popcount
kernel for 1-bit×1-bit matrices is benchmarked separately.

**Did it work?**

- **Storage: yes.** 192 packed layers, 62.9 MB packed weights for attention+FFN
  (vs 63.0 MB theoretical payload), and the artifact drops to 0.42 GB.
- **Speed: no.** Packed forward 1.826 s vs BitLinear 1.657 s on CPU (1.10×
  slower) because unpacking costs time.
- **XNOR kernel: yes, but scoped.** It exactly matches the ±1 dot product and is
  2.3× faster than float for 1-bit×1-bit. The ASR model has FP activations, so it
  cannot use that kernel without becoming a different model.

**What we learned:** 1-bit storage and 1-bit compute are separate engineering
problems. This repo solves the storage part and measures the boundary of the
compute part.

---

## E21 — Chunked streaming

**Question:** can we provide partial transcripts with a chunked interface?

**Why:** Phase 19 is optional but part of the plan; a real streaming encoder is a
much larger project.

**What:** overlapping 2 s windows (0.5 s overlap), full encoder per window,
greedy CTC, word-level suffix/prefix merge.

**Did it work?** **Partially.** On a 4.45 s sample: chunked WER **69.2%** vs
**15.4%** full-utterance, chunked RTF 0.175. The merge duplicates words when the
overlap does not cover a word.

**Why:** Wav2Vec2-Conformer is non-streaming; each window lacks the context of
the full utterance. The overlap merge is a heuristic, not an alignment model.

**What would work better:** cached convolution/attention state, a streaming
attention mask, and a proper incremental decoder. Not implemented.

---

## E22 — Full test-clean evaluation

**Question:** does the 256-utterance ranking hold on the full 2620-utterance
test-clean?

**Why:** the plan's final evaluation is full test-clean; a small subset can be
easier or biased.

**What:** accuracy-only evaluation of FP32, attention-only, FFN-only,
attention+FFN, KD 0.5, KD 0.1 on all 2620 test-clean utterances.

**Did it work?** **Yes.**

| Arm | Full test WER | CER |
|---|---:|---:|
| FP32 | 0.0193 | 0.0050 |
| QAT attention-only | 0.0449 | 0.0137 |
| QAT FFN-only | 0.1134 | 0.0325 |
| QAT attention+FFN | 0.2050 | 0.0639 |
| QAT + KD 0.5 | 0.2218 | 0.0709 |
| QAT + KD 0.1 | 0.1998 | 0.0636 |

**What we learned:** the ranking is identical; the 256 subsets are slightly
easier. The full test-clean is the number to quote.

---

## Infrastructure failure log

Each entry is a real failure that changed the code. This is deliberately kept.

| # | Symptom | Root cause | Fix | Commit |
|---|---|---|---|---|
| I1 | `torchvision::nms does not exist` on Kaggle | mismatched torch/torchvision CUDA wheels | pin matching `torch==2.14.0`, `torchvision==0.29.0`; remove stale torchaudio | `4ff384c` |
| I2 | Kaggle worker OOM-killed while building 10k features | materializing decoded audio as Python lists | stream features via `Dataset.from_generator` | `7933a48` |
| I3 | probe/eval crash on CUDA | inputs left on CPU | move inputs to the model device | `bb0e50d` |
| I4 | probe OOM before measuring | no gradient checkpointing | treat crash as gate failure; retry once with GC; abort otherwise | `60e4535` |
| I5 | Kaggle CLI crashes on Windows console | non-UTF-8 code page | force `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1` | `6b90094` |
| I6 | token-refresh churn in wrappers | non-interactive refresh | reverted | `fd662a5` |
| I7 | Kaggle disk exhausted | Trainer epoch checkpoints (~5 GB optimizer state) | `save_strategy: "no"` + cleanup | `d1927e4` |
| I8 | save/load false negative after a valid run | exact logit equality across GPU processes | argmax + tolerant logits | `9fa7bbb`, later 1e-1 |
| I9 | provenance recorded `baseline.yaml` instead of the leaf config | `setdefault` in `load_config` | assign the leaf path unconditionally | `20019c9` |
| I10 | Kaggle summary reported a null probe gate | read only attempt 1 (which crashed) | prefer whichever probe passed | `cd438a8` |
| I11 | final clean Kaggle run rejected | weekly GPU quota 30.01/30.00 h | deferred and documented | — |
| I12 | final-eval crashed on an in-progress checkpoint | `Trainer` creates `output_dir` before weights exist; `is_dir()` was too weak | require `model.safetensors`/`pytorch_model.bin`; catch per-arm errors | this checkpoint |
| I13 | distillation probe device mismatch | batch moved to CUDA, model still on CPU | derive device from the model; move the student before the probe | this checkpoint |
| I14 | two jobs selected the same A100 | simultaneous GPU selection race | atomic reservation directory + free-memory-first ordering | this checkpoint |
| I15 | jobs queued "Insufficient ncpus" while GPUs were idle | per-user CPU quota, not GPU availability | resize GPU-bound jobs to 1 CPU | this checkpoint |
| I16 | full test-clean with benchmark was too slow | two full passes per arm | `--no-benchmark` accuracy-only mode | this checkpoint |

---

## Decision log

| Decision | Alternatives considered | Why we chose it |
|---|---|---|
| `BitLinear` subclasses `nn.Linear` | wrapper module | preserves `state_dict` keys; stock HF Trainer works; lossless checkpoint load |
| FP32 master weights | binary parameters as state | SGD cannot update signs; masters accumulate small updates |
| `alpha = mean(|W|)` per tensor | per-channel, learned alpha | simple, differentiable, cheap, configurable; others are future work |
| Quantize inside `forward` | quantize once after loading | QAT needs current masters every step; Trainer needs no changes |
| Pure pass-through STE by default | clipped STE default | simple and standard; clipping is opt-in and visible |
| Shared taxonomy + raise unknown keys | free-form name matching | prevents a typo from silently matching nothing |
| PTQ first | QAT first | measures the raw cost; produces the negative result the plan predicted |
| Attention-only QAT first | full QAT first | conservative; FFN-only PTQ was fatal, so isolate the easier half |
| FFN-only and attention+FFN next | only one combined arm | completes the sensitivity matrix; found the interaction |
| Distillation as fallback | tune QAT longer first | teacher has the lost information; cheap because logits are precomputed |
| Profile-local results | merge T4/A100 numbers | hardware and library stacks differ; no cross-profile extrapolation |
| Accuracy-only full test | benchmark every arm on 2620 | accuracy is the question; latency is already measured on 256 |
| 1-CPU GPU jobs | 4–8 CPUs each | the CPU quota was the bottleneck; GPU-bound jobs run concurrently |
| Honest `n/a` / "not measured" | fill with estimates | the project's core rule: never fabricate |

---

## What worked, what didn't, and why

### Worked

- **Drop-in `BitLinear`:** state_dict compatibility + FP masters made QAT and
  checkpointing trivial. Tests caught every invariant.
- **QAT attention-only:** 44.5% PTQ → 4.21% dev / 4.49% full test.
- **QAT FFN-only:** 100% PTQ → 11.90% dev / 11.34% full test at 2.9× compression.
- **Bit packing and honest size accounting:** 0.423 GB attention+FFN artifact,
  measured from a real file.
- **Reproducibility:** fixed subsets, recorded IDs, config provenance, result
  JSONs with versions, 88 tests.

### Partial

- **QAT attention+FFN:** recovers to 20.80% dev / 20.50% full test — the 5.6×
  target, but ~10× the FP32 WER. The two binarizations interact.
- **Distillation λ_kd=0.1:** 20.22% dev / 19.98% full test; a real but marginal
  gain over QAT. KL still dominates CTC.
- **Packed inference:** storage win, but 1.10× slower on CPU because it expands
  bits. XNOR is fast only for 1-bit×1-bit.
- **Chunked streaming:** partial transcripts work, but chunked WER 69.2% vs
  15.4% full; it is not a streaming encoder.

### Did not work

- **1-bit PTQ:** 100% WER for attention+FFN. Expected and kept.
- **Distillation λ_kd=0.5:** 22.32% dev / 22.18% full test, worse than QAT. The
  KD term dominated.
- **Speedup from binarization:** measured; none. The runtime is FP.

### Open

- Why attention+FFN QAT is worse than FFN-only QAT.
- Whether layer-wise / sequential quantization fixes the interaction.
- True binary compute (1-bit activations + XNOR kernel).
- 2-bit / INT8 / per-channel / learned scaling.
- Full test-clean latency benchmark, LM decoding, true streaming encoder.

---

## E23 — Project completion audit (v1.5.2)

**Question:** are any required phases silently unrun, and what is the final
number to quote?

**Why:** a long research repository accumulates "in progress" markers. The
project should end with an explicit statement of what is complete, what is
optional, and what is blocked.

**What:** audited phases 0–19 against the original plan, the README status
table, `planning.md`, and the committed result JSONs.

**Did it work?** **Yes.** Every required phase has a measured artifact or an
explicitly scoped investigation:

- 0–10: complete (environment, inspection, inference, FP32 baseline, BitLinear,
  PTQ, size accounting).
- 11: complete on T4 and A100 (attention-only, FFN-only, attention+FFN).
- 12: complete on dev 256, test 256, and full test-clean 2620.
- 13: complete for λ_kd=0.5 and λ_kd=0.1.
- 14: complete for FP32/PTQ/QAT/QAT+KD; 2-bit/INT8 was optional and was not run.
- 15: complete for the PTQ and QAT component selections.
- 16: complete; measured packed artifacts for every trained arm.
- 17: complete on A100 dev 256 (warmed-up RTF/VRAM). The full-test run was
  accuracy-only and is labelled as such.
- 18: investigated (packed storage + XNOR/popcount boundary). A true integrated
  bitwise kernel is future work because the model has FP activations.
- 19: prototype measured (chunked pseudo-streaming). A true streaming encoder
  with cached state is future work.

**What is blocked:** the Kaggle T4 clean re-run. The Kaggle CLI/credentials are
not available in this environment, and the weekly GPU quota was exhausted
historically. The A100 re-run reproduces the attention-only result and the
packing pipeline, so the science is covered; the Kaggle artifact remains a
historical record.

**Final full test-clean WER (2620 utterances, greedy CTC):**

| Model | WER | CER | Deploy size |
|---|---:|---:|---:|
| FP32 reference | **0.0193** | 0.0050 | 2.374 GB |
| Best 1-bit accuracy: attention-only QAT | **0.0449** | 0.0137 | 1.983 GB |
| FFN-only QAT (2.9×) | 0.1134 | 0.0325 | 0.813 GB |
| Attention+FFN QAT (5.6×) | 0.2050 | 0.0639 | 0.423 GB |
| Attention+FFN + KD λ_kd=0.1 (5.6×) | **0.1998** | 0.0636 | 0.423 GB |

Dev-clean (256): 0.0202 / 0.0421 / 0.1190 / 0.2080 / 0.2022. The historical
Kaggle T4 attention-only run measured **0.0392** on the 256 dev subset; it is
kept as a separate result, but the final full-test number is 0.0449.

**Conclusion:** the project is complete at the scope of the original plan. The
open items above are explicitly future research, not missing required phases.

---

## E24 — Per-output-channel reconstruction screen

**Question:** does one scale per FFN output row (`alpha_i = mean_j |W[i,j]|`)
remove enough weight-reconstruction error to justify a QAT run?

**Why:** the current quantizer uses one scale per matrix; the FFN is 68% of the
parameters and the best compressed arm (11.34% full-test at 0.813 GB). This is
the cheapest test of whether the scale granularity is the bottleneck. A
pre-registered gate stopped the experiment if the improvement was below 10%
relative.

**What:** implemented `per_output_channel_mean_abs` and a CPU screen
(`scripts/analyze_quantization_error.py`) over all 96 FFN weight matrices of
(a) the FFN-only QAT masters and (b) the FP32 starting weights. Metrics:
relative Frobenius reconstruction error, row-norm CV, zero rows.

**Did it work?** **The screen worked; the quantizer did not clear the gate.**

| Source | Per-tensor error | Per-channel error | Improvement |
|---|---:|---:|---:|
| FFN-only QAT masters | 0.6458 | 0.6306 | **2.28%** |
| FP32 starting weights | 0.6508 | 0.6343 | **2.43%** |
| `ffn_in` (FP32) | 0.6404 | 0.6225 | 2.68% |
| `ffn_out` (FP32) | 0.6611 | 0.6460 | 2.18% |

Row CV was only 0.168–0.176: FFN rows have fairly similar magnitudes, so one
scale per row removes only ~2% of the reconstruction error. The gate required
≥10%.

**What we did instead:** did **not** run per-channel QAT (the planned gate
stopped it); pivoted to the per-layer sensitivity screen (E25).

**What worked, and why:** the cheap screen saved an expensive QAT run. The
per-channel plumbing itself is correct and validated: 96/96 BitLinear gradients,
save/load, and a packed-artifact reload with max logit diff **0**. Those
capabilities are kept for future work even though this particular quantizer was
rejected.

---

## E25 — FFN per-layer sensitivity screen

**Question:** which individual FFN layers are most damaged by isolated 1-bit
quantization?

**Why:** per-channel scaling failed the reconstruction gate; mixed precision
needs evidence about specific layers, not an assumption based on position.
The screen is cheap (no retraining) and distinguishes screening from QAT.

**What:** `scripts/screen_layer_sensitivity.py`, 32 fixed dev utterances, each of
the 96 FFN Linear layers replaced **one at a time** by BitLinear (per-tensor),
with the rest of the model FP32. Measured relative L2 logit change, mean
per-frame KL(reference‖variant), frame argmax agreement, and greedy WER/CER.

**Did it work?** **Yes, as a screening experiment.** FP32 reference WER on the
32 utterances was **0.0287**. The deliberately isolated single-layer damage is
small (≤1.28 pp WER), but the logit/KL ranking is informative:

| Layer | Category | Rel. L2 | KL | Argmax | ΔWER |
|---|---|---:|---:|---:|---:|
| `layers.18.ffn1.intermediate_dense` | ffn_in | 0.2823 | 0.2791 | 0.896 | +0.0128 |
| `layers.18.ffn1.output_dense` | ffn_out | 0.2388 | 0.2344 | 0.912 | +0.0080 |
| `layers.23.ffn1.intermediate_dense` | ffn_in | 0.2350 | 0.0361 | 0.969 | +0.0016 |
| `layers.17.ffn1.intermediate_dense` | ffn_in | 0.2325 | 0.2432 | 0.914 | −0.0016 |
| `layers.19.ffn1.intermediate_dense` | ffn_in | 0.2281 | 0.1840 | 0.926 | −0.0016 |
| `layers.17.ffn2.intermediate_dense` | ffn_in | 0.1815 | 0.1684 | 0.931 | +0.0096 |

Group summary: `ffn_in` mean ΔWER **0.00342**, `ffn_out` **0.00183** — the
intermediate (`ffn_in`) layers are ~1.9× more sensitive. The most sensitive
layers are middle/late (17–23), not simply the first or last.

**Why:** `ffn_in` is the 1024→4096 expansion; quantizing it changes a wider
activation basis. WER on 32 utterances is noisy (small deltas, some negative),
so the logit/KL metrics are the primary screening signal; WER is secondary.

**What we did instead:** encoded the measured policy as
`configs/qat_ffn_mixed.yaml`: keep **17** layers FP32 (union of top 12 by
relative L2 and top 8 by WER delta; 14 `ffn_in`, 3 `ffn_out`), quantize the
other 79. The 17 FP layers cost 285.2 MB, so the expected artifact is
~1.10 GB — still far below the 1.983 GB attention-only model.

**What worked, and why:** the screen identified specific layers from measured
output perturbation, not position, and did so in minutes. It is explicitly
labeled `screening_only: true`, `not_qat: true`; it is not a WER claim.

---

## E26 — Mixed-precision FFN implementation and pilot (pilot blocked)

**Question:** does keeping the 17 sensitive FFN layers FP32 while quantizing the
other 79 improve the FFN-only WER/size trade-off?

**Why:** E25 found that `ffn_in` is more sensitive and identified the specific
layers; the balanced gate wants <10% WER with materially smaller storage than
the 1.983 GB attention-only model.

**What:** added a per-layer exclusion policy to `replace_linears` (exact names,
validated; unknown names raise), plumbed it through QAT setup, checkpoint
manifests, loading, final evaluation, packing, and the artifact verifier.
`configs/qat_ffn_mixed.yaml` keeps 17 layers FP32. Added a dev-WER callback for
validation-based best-checkpoint selection; added an artifact sidecar manifest
and a loader that rebuilds from the artifact.

**Did it work?** **Implementation: yes. Training: not run.**

Validated on CPU and the real model:
- 79/79 BitLinear masters received gradients; the 17 excluded layers stayed FP32;
  the frozen feature extractor had zero gradient.
- Save/load verified; the packed artifact rebuilt with **max logit diff 0.0,
  argmax agreement 1.0, decoded match 1/1** (both per-channel and mixed smoke).
- 107 tests pass (19 new: per-channel scaling/broadcasting/zero rows/gradients,
  exclusion policy, manifest checks, artifact cases, callback selection).
- Mixed smoke WER 0.7105 (8 steps, 16 utterances) is a pipeline check, **not** an
  accuracy result.
- Artifact stats: 79 BitLinear layers, 17 FP FFN layers, ~1.10 GB expected
  (measured smoke artifact includes the FP layers), effective ~1.00 bit per
  quantized weight, ~14.7 bits per model parameter including all FP parameters.

**Why no pilot/WER:** the GPU node was fully allocated (`resources_assigned.ncpus
= 96`, all 8 GPU slots in use by 26 running jobs; PBS reported “not enough free
nodes”). The controlled pilot (control vs mixed, 2k utts × 500 steps, dev-eval
every 250) was queued and then canceled rather than left unattended. **No mixed
precision WER is claimed.**

**What to do next (exact command):**

```bash
qsub cluster/qat_ffn_pilot.pbs
```

It runs the per-tensor FFN control and the mixed arm sequentially, same seed,
budget, dev-eval cadence, and best-dev restoration. Promote to full only if the
mixed best dev WER ≤ control. If not, do not tune arbitrarily; move to the
attention-layer sensitivity screen (the combined attention+FFN arm is still worse
than FFN-only).

**What worked, and why:** the mixed policy is data-driven, fully testable, and
does not touch the evaluation protocol or historical results. The GPU pilot is
the only missing step; the node allocation, not the code, blocked it.

---

## How to add a new log entry

When you run a new experiment, append:

```markdown
## E23 — <short title>

**Question:** ...
**Why:** ...
**What:** config, data, steps, hardware.
**Did it work?** Yes / No / Partially. Quote the JSON path and the metric.
**If not, why:** the measured or hypothesized cause.
**What we did instead:** ...
**What worked, and why:** ...
```

Then update `RESULTS.md` (via `scripts/summarize_results.py`), `README.md` if the
headline changes, and tag a new version. Never delete a negative result.
