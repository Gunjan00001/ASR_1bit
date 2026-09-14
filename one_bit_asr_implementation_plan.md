# Implementation Plan — One-Bit Wav2Vec2-Conformer ASR

## 0. Project Objective

Build and benchmark a **1-bit Automatic Speech Recognition system** by starting from a pretrained **Wav2Vec2-Conformer CTC** model and progressively replacing suitable floating-point linear layers with custom **BitLinear** layers.

The core question is:

> Can a Wav2Vec2-Conformer ASR model be made substantially smaller and potentially more efficient using 1-bit weights while retaining competitive ASR accuracy?

## 1. Scope

**Do NOT build the entire ASR stack from scratch.**

### Reuse
- Wav2Vec2 audio feature extraction
- STFT/audio preprocessing
- LibriSpeech dataset tooling
- Wav2Vec2-Conformer architecture
- CTC loss
- tokenizer/vocabulary infrastructure
- standard training utilities

### Implement ourselves
- BitLinear
- binary weight quantization
- scaling strategy
- Straight-Through Estimator (STE)
- quantization-aware training (QAT)
- layer replacement mechanism
- knowledge distillation
- binary weight packing/accounting
- benchmarking and analysis

The project focuses on **low-bit ASR**, not reinventing standard ASR infrastructure.

## 2. Recommended Stack

Use:
- Python
- PyTorch
- Torchaudio
- Hugging Face Transformers
- Hugging Face Datasets
- Evaluate / JiWER
- Accelerate where useful
- NumPy
- YAML/JSON configuration

Pin tested dependency versions after compatibility is established.

## 2a. Compute Environment & Strategy

Detected local hardware (2026-09-12):

- CPU: AMD Ryzen AI 7 350 (8 cores)
- RAM: ~24 GB
- GPU: AMD Radeon 860M (integrated) — **no CUDA; treat the local machine as CPU-only**
- Python: 3.14.3 (torch 2.14.0 confirmed resolvable; verify torchaudio/transformers/datasets compatibility in Phase 0, then pin)

Two execution profiles share identical configs and pinned dependency versions:

| Profile | Where | Phases |
|---|---|---|
| `local` | This machine (CPU) | 0–10: environment, inspection, inference, FP32 baseline on subsets, BitLinear dev/tests, layer replacement, PTQ |
| `cloud-gpu` | Cloud GPU (e.g., Colab/RunPod) | 11–18: QAT, distillation, controlled experiments, layer sensitivity, final full benchmarks |

Rules:

- Configs carry a `compute_profile` field; every results JSON records which profile produced it.
- Local CPU runs use reduced, fixed subsets (see pinned evaluation protocol in Phase 3) so they finish in hours, not days.
- Never extrapolate benchmark numbers from one profile to the other.

## 3. Base Model

Start with a pretrained Wav2Vec2-Conformer CTC checkpoint, such as:

`facebook/wav2vec2-conformer-rope-large-960h-ft`

Verify checkpoint compatibility with the installed Transformers version.

Model-card reference (greedy CTC decoding): **WER 1.96 (test-clean) / 3.98 (test-other)**. Use as the sanity target for the Phase 3 FP32 baseline.

Training-scale phases (QAT, distillation) assume the `cloud-gpu` profile (see §2a); local CPU runs use reduced fixed subsets only.

Do not assume every Linear layer should be quantized.

## 4. Repository Structure

```text
one-bit-asr/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── baseline.yaml
│   ├── binary.yaml
│   ├── qat.yaml
│   └── distillation.yaml
├── src/
│   └── onebit_asr/
│       ├── data/
│       │   ├── dataset.py
│       │   ├── collator.py
│       │   └── text.py
│       ├── features/
│       │   └── audio.py
│       ├── models/
│       │   ├── baseline.py
│       │   ├── conformer_utils.py
│       │   └── replace_layers.py
│       ├── quantization/
│       │   ├── bitlinear.py
│       │   ├── binarization.py
│       │   ├── scaling.py
│       │   └── ste.py
│       ├── training/
│       │   ├── train.py
│       │   ├── qat.py
│       │   └── distillation.py
│       ├── decoding/
│       │   └── decode.py
│       └── evaluation/
│           ├── metrics.py
│           ├── model_size.py
│           ├── latency.py
│           └── benchmark.py
├── scripts/
│   ├── inspect_model.py
│   ├── infer.py
│   ├── train_baseline.py
│   ├── quantize.py
│   ├── train_qat.py
│   ├── train_distillation.py
│   └── benchmark.py
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   ├── test_bitlinear.py
│   ├── test_binarization.py
│   ├── test_ste.py
│   ├── test_layer_replacement.py
│   ├── test_ctc_shapes.py
│   └── test_model_io.py
├── results/
└── checkpoints/
```

## 5. Phase 0 — Environment

1. Create Python environment.
2. Install dependencies.
3. Pin tested versions.
4. Detect CPU/GPU.
5. Set reproducible seeds where practical.
6. Add an environment diagnostic script.

On Windows, decode audio via `soundfile` through Hugging Face `datasets`; do not rely on `torchaudio.load` (torchcodec dependency risk).

Note (verified 2026-09-12): current `datasets` audio decoding *requires* `torchcodec`, which on Windows additionally needs FFmpeg **shared** DLLs plus a `.pth`-registered DLL directory (Python 3.14 no longer searches the app dir). `scripts/setup_windows_ffmpeg.ps1` automates this; `torchcodec` is pinned in `requirements.txt`. On Linux/cloud-GPU, install FFmpeg shared libraries via the system package manager.

Report:
- Python version
- PyTorch version
- CUDA availability
- GPU name
- CUDA version
- Transformers version
- Torchaudio version

### Acceptance
A clean environment installs successfully and imports all required dependencies.

## 6. Phase 1 — Model Inspection

Create `scripts/inspect_model.py`.

Load the pretrained model and print:
- architecture
- total parameters
- trainable parameters
- every `nn.Linear`
- input/output dimensions
- parameter count per Linear layer
- parameter count by category

Identify:
- attention projections
- FFN projections
- convolution layers
- feature extractor
- feature projection
- CTC head
- normalization layers

### Acceptance
The agent produces a clear map of candidate Linear layers without modifying the model.

## 7. Phase 2 — Baseline Inference

Create:

`scripts/infer.py`

Accept an audio path and output its transcription.

Pipeline:

```text
Audio → Processor → Wav2Vec2-Conformer → Logits → Decode → Text
```

### Acceptance
A real speech recording can be transcribed successfully.

## 8. Phase 3 — FP32 Baseline

Evaluate the original model on a reproducible evaluation subset. Start small for development, then scale.

Pinned protocol:

- Decoding: **greedy CTC (argmax)** for every experiment arm; no language model (LM beam search is optional future work).
- Text normalization before WER/CER: lowercase + strip punctuation (the tokenizer emits uppercase).
- Dev evaluation: fixed 256-utterance subset of LibriSpeech `clean` `validation`, fixed seed, recorded in the results JSON.
- Final evaluation: full LibriSpeech `test-clean` on the `cloud-gpu` profile; on `local`, a recorded fixed subset is acceptable if full evaluation is CPU-time-prohibitive.

Measure:
- WER
- CER
- parameter count
- checkpoint size
- theoretical FP32 weight size
- peak memory
- inference latency
- real-time factor (RTF)
- throughput

Save machine-readable results, e.g.:

`results/baseline.json`

Never insert illustrative benchmark values.

### Acceptance
The FP32 baseline is reproducible and becomes the reference for every later experiment.

## 9. Phase 4 — Toy BitLinear

Before modifying ASR, implement a tiny standalone BitLinear.

Concept:

```text
Master W
  ↓
Binary quantization
  ↓
Wb = sign(W)
  ↓
Scaling
  ↓
αWb
  ↓
Linear operation
```

Keep:
- master weights
- binary weights
- scale

separate.

Do not overwrite master weights during training.

## 10. Phase 5 — Binary Quantization

Implement configurable:

`Wb = sign(W)`

with:

`W ≈ αWb`

Isolate scaling in its own module.

Document:
- what is quantized
- scaling granularity
- how scale is calculated/learned
- gradient behavior

Start with a simple, understandable strategy.

Default strategy (configurable): per-tensor scale α = mean(|W|) (XNOR-Net-style); α is recomputed each forward from the FP master weights (not a learned parameter).

## 11. Phase 6 — Straight-Through Estimator

Implement and document the STE:

```text
Forward:
W → sign(W)

Backward:
approximate gradient through quantization
```

Default STE: straight pass-through gradient to the master weights (gradient clipping variants configurable later).

Tests must prove gradients reach master weights and parameters update normally.

## 12. Phase 7 — BitLinear Tests

Test:
1. Binary forward weights
2. Correct scaling
3. Output shape matching `nn.Linear`
4. Gradient existence
5. Optimizer updates
6. Save/load
7. Reasonable toy numerical behavior

Do not integrate until these pass.

## 13. Phase 8 — Layer Replacement

Create:

`src/onebit_asr/models/replace_layers.py`

Implement safe configurable replacement:

`nn.Linear → BitLinear`

Allow selection by module name/category.

Example:

```yaml
layers:
  attention_q: true
  attention_k: true
  attention_v: true
  attention_output: true
  ffn: true
  ctc_head: false
```

Do not hard-code architecture assumptions.

## 14. Phase 9 — First 1-Bit Experiment

Start conservatively:

```text
Attention projections → BitLinear
FFN projections       → BitLinear
CTC head              → FP
Feature extractor     → FP
Normalization         → FP
Convolution           → FP
```

This is an experimental configuration, not a permanent rule.

## 15. Phase 10 — Post-Training Quantization

Take the FP32 model and quantize selected weights without retraining.

Purpose:

> Measure raw accuracy degradation caused by binarization.

Expectation: 1-bit PTQ on a fine-tuned model will likely degrade WER severely, possibly near-total collapse. Record this as a result, not a bug. The purpose of QAT is to quantify how much accuracy is recoverable.

Compare:
- FP32 WER
- 1-bit PTQ WER
- WER delta
- model size

## 16. Phase 11 — Quantization-Aware Training

Train with:

```text
FP master weights
       ↓
binary quantization
       ↓
1-bit forward pass
       ↓
CTC loss
       ↓
STE backward pass
       ↓
update FP master weights
```

Make quantization configuration explicit.

Recipe requirements:

- Freeze the CNN feature extractor during QAT (standard wav2vec2 fine-tuning practice; it also stays FP).
- BitLinear must quantize inside `forward()` so the stock Hugging Face `Trainer` can run QAT unmodified (plan rule 5).
- Fine-tune from the already-fine-tuned FP32 checkpoint with a small learning rate; record LR, subset, and epochs in results.

## 17. Phase 12 — QAT Evaluation

Compare:

```text
FP32 baseline
1-bit PTQ
1-bit QAT
```

Measure:
- WER
- CER
- checkpoint size
- packed binary weight size
- memory
- latency
- RTF
- throughput

Distinguish **storage compression** from **actual hardware acceleration**.

## 18. Phase 13 — Knowledge Distillation

Teacher:

`FP32 Wav2Vec2-Conformer`

Student:

`1-bit Wav2Vec2-Conformer`

Concept:

```text
                 Audio
                   │
          ┌────────┴────────┐
          ↓                 ↓
     FP32 Teacher       1-bit Student
          │                 │
          ↓                 ↓
      predictions       predictions
          │                 │
          └────────┬────────┘
                   ↓
            Distillation loss
```

Student training should combine:
- ground-truth CTC objective
- teacher guidance

Use configurable loss weights.

Loss: `L = λ_ctc · CTC(student, y) + λ_kd · T² · KL(student_logits/T ‖ teacher_logits/T)`, frame-level, with temperature T and λ weights in config.

Efficiency requirement: precompute frozen-teacher logits once per training subset and reuse them, so distillation costs barely more than QAT alone.

## 19. Phase 14 — Controlled Quantization Experiments

At minimum:

```text
A. FP32 baseline
B. 1-bit post-training quantization
C. 1-bit QAT
D. 1-bit QAT + distillation
```

Optional:
- 2-bit QAT
- INT8 baseline
- different BitLinear layer subsets
- different scaling strategies
- low-bit activations

## 20. Phase 15 — Layer Sensitivity

Determine which Conformer components tolerate binarization.

Test configurations such as:
- attention only
- FFN only
- attention + FFN
- attention + FFN + output

Measure:
- WER
- model size
- latency

Goal:

> Find the best efficiency/accuracy trade-off.

## 21. Phase 16 — Binary Model Storage

Do not report theoretical 1-bit size as actual size.

Implement binary packing and measure:

```text
FP32 checkpoint
Naive binary representation
Packed binary representation
Final checkpoint including scales/non-binary parameters
```

Implementation detail: `torch.save` of bool/uint8 tensors stores ≥1 byte per weight — that is not 1-bit storage. Real packing must use bit-level packing (e.g., `numpy.packbits`). Reported final size = packed binary weights + FP32 scales + all non-quantized FP parameters + container overhead.

## 22. Phase 17 — Inference Benchmarking

On fixed hardware measure:

### Latency
- average
- p50
- p95

### Memory
- peak RAM
- peak GPU VRAM

### Throughput
- samples/sec

### Real-Time Factor

`RTF = processing time / audio duration`

RTF < 1 means faster than real-time.

Do not assume binary weights automatically make inference faster.

## 23. Phase 18 — Actual Binary Compute

Only after correctness and benchmarking are established, investigate:
- binary weight packing
- bitwise operations
- CPU optimization
- custom kernels

A binary model can be smaller without being faster if the runtime expands weights back to floating point.

## 24. Phase 19 — Optional Streaming ASR

After offline ASR works, optionally investigate:

```text
Microphone
 ↓
Audio chunks
 ↓
Feature extraction
 ↓
Streaming inference
 ↓
Partial transcription
```

This must not delay the core 1-bit work.

## 25. Final Comparison

Produce:

| Model | Weight Precision | WER | CER | Model Size | Memory | Latency | RTF |
|---|---|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 32-bit | measured | measured | measured | measured | measured | measured |
| 1-bit PTQ | 1-bit | measured | measured | measured | measured | measured | measured |
| 1-bit QAT | 1-bit | measured | measured | measured | measured | measured | measured |
| 1-bit QAT + KD | 1-bit | measured | measured | measured | measured | measured | measured |

Use actual experimental values only.

## 26. README Requirements

Document:
1. Project purpose
2. Why 1-bit ASR
3. Wav2Vec2-Conformer architecture
4. BitLinear
5. Binary quantization
6. STE
7. QAT
8. Knowledge distillation
9. Dataset
10. Installation
11. Training
12. Evaluation
13. Benchmark results
14. Limitations
15. Future work

Clearly label which parts are reused and which are our implementation.

## 27. Agent Rules

The coding agent MUST:

1. Inspect the model before modifying it.
2. Keep the FP32 baseline working.
3. Make one architectural change at a time.
4. Run tests after major changes.
5. Reuse established libraries.
6. Avoid premature CUDA/kernel optimization.
7. Measure real performance rather than assuming theoretical speedups.
8. Keep quantization configurable.
9. Preserve reproducibility.
10. Document important design decisions.
11. Never fabricate benchmark results.
12. Clearly separate baseline code from our 1-bit modifications.

## 28. Definition of Done

- [ ] Wav2Vec2-Conformer baseline runs.
- [ ] Baseline WER is measured.
- [ ] Relevant Linear layers are identified.
- [ ] BitLinear is implemented and tested.
- [ ] Binary quantization is implemented.
- [ ] Scaling is implemented.
- [ ] STE is implemented and tested.
- [ ] Selected Conformer layers can be replaced.
- [ ] 1-bit PTQ runs.
- [ ] 1-bit QAT runs.
- [ ] Knowledge distillation runs.
- [ ] Binary model storage is measured correctly.
- [ ] FP32 vs 1-bit performance is benchmarked.
- [ ] Layer sensitivity experiments are completed.
- [ ] Results are reproducible.
- [ ] README documents architecture and experiments.
- [ ] No unsupported performance claims remain.

# 29. FIRST AGENT TASK — DO ONLY THIS

The first coding-agent task is intentionally small.

### Agent should ONLY:

1. Create the repository structure.
2. Set up dependencies.
3. Load the selected Wav2Vec2-Conformer checkpoint.
4. Create `scripts/inspect_model.py`.
5. Print the architecture.
6. List every `nn.Linear` with dimensions and parameter counts.
7. Categorize attention, FFN, CTC-head, and other Linear layers.
8. Create a minimal smoke-test inference script.
9. Run the smoke test.
10. Report what was found.

### Agent must NOT yet:

- implement BitLinear
- quantize anything
- modify the Conformer
- train the model
- implement custom CUDA kernels
- perform QAT
- perform distillation

The first task exists to establish a **known-good baseline and architectural map** before any modification occurs.
