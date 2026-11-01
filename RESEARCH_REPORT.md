# Low-Bit Wav2Vec2-Conformer ASR: Final Research Report

**Status:** final research freeze, 2026-11-01  
**Scope:** LibriSpeech clean, greedy CTC, no language model, no test-set tuning  
**Selected models:** FP32 reference, 2-bit FFN accuracy winner, Strategy A compression winner

## Executive summary

- FP32 reference: **1.93% full test-clean WER**, 2,373,507,712 B.
- 2-bit FFN: **2.58% full test-clean WER**, historical artifact 1,131,062,436 B;
  standalone v2 artifact 1,131,269,460 B.
- Strategy A (36 attention projections + 79 FFN layers at 2-bit): **2.74% full
  test-clean WER**, historical artifact 989,509,572 B; standalone v2 artifact
  989,716,588 B. It is the only selected model below 1 GB.
- Storage compression is real. **Inference acceleration is not claimed and was
  not observed**: packed weights are expanded to FP32 before `F.linear`.
- The v2 artifacts add required BatchNorm/positional buffers. Historical
  artifacts remain unchanged but are not standalone deploy artifacts.
- Short-utterance oversampling, Strategy C, per-channel scaling, and beam-search
  decoding were rejected. No full training was launched after the approved
  pilot.

## 1. Problem statement and motivation

The target is a compact CTC speech recognizer that preserves the accuracy of a
large pretrained Wav2Vec2-Conformer while reducing deployable weight storage.
The project asks three questions:

1. How far can linear-weight precision be reduced before clean ASR degrades?
2. Can mixed-precision layer selection recover accuracy at a sub-1 GB size?
3. Does packed low-bit storage also reduce inference time?

The answer is asymmetric: **2-bit weight storage is effective; packed inference
is slower** in this FP-activation implementation.

## 2. Model architecture

- Pretrained model: `facebook/wav2vec2-conformer-rope-large-960h-ft`.
- Architecture: Wav2Vec2-Conformer, 24 encoder layers, RoPE relative attention,
  convolution module, CTC head.
- Parameters: **593,376,928**.
- Tokenizer: 32-token CTC vocabulary, uppercase characters, `|` word delimiter,
  blank id 0.
- Training data: LibriSpeech `train.100`; evaluation: fixed dev-clean subsets and
  full test-clean (2,620 utterances), greedy CTC, normalized lowercase text.

## 3. Quantization methodology

### 1-bit path

```text
Wq = alpha * sign(W),   alpha = mean(|W|)
```

The sign is STE-backed. FP32 masters are retained; quantization is recomputed in
each forward.

### 2-bit path

A symmetric mid-rise four-level codebook:

```text
levels = {-1.5, -0.5, +0.5, +1.5}
code   = clamp(floor(W / alpha) + 2, 0, 3)
Wq     = (code - 1.5) * alpha
```

- One FP32 `alpha = mean(|W|)` per matrix in the selected models.
- Four 2-bit codes per byte; scales and non-quantized FP state are included in
  the measured artifact.
- Exact zero maps to +0.5 alpha; values beyond ±2 alpha saturate.
- No zero level; no learned scale.
- Optional `per_output_channel_mean_abs` exists but was rejected (Section 9).

### Packed artifact correction

The original packer saved parameters but not buffers. BatchNorm running
statistics and positional buffers were therefore absent; historical artifacts
reloaded only when the QAT checkpoint supplied those buffers. The corrected v2
packer includes `named_buffers()`. The historical artifacts and their recorded
sizes are preserved; v2 artifacts are the standalone deploy files.

## 4. BitLinear implementation

`BitLinear` subclasses `nn.Linear`, preserves parameter names/shapes, and stores
FP32 master weights. Its forward derives quantized weights, so the stock HF
Trainer works unchanged. `PackedBitLinear` stores only packed codes, FP32 scales,
biases, and non-quantized FP state for inference. Its forward unpacks codes and
multiplies by scales on every call. This is **storage** compression, not a
bitwise compute kernel.

## 5. Quantization-aware training

- Initialization: FP32 pretrained model or a selected QAT checkpoint.
- Data: 10,000 `train.100` utterances for full runs; 2,000 for pilots.
- Batch 4 × gradient accumulation 4; AdamW; lr 2e-5; linear schedule; seed 42.
- 1,875 optimizer steps for full runs; 500 for pilots.
- FP32 masters; STE gradients; feature extractor frozen.
- Dev-only best-checkpoint selection; test evaluated once after selection.
- No test-set tuning.

## 6. Mixed precision and layer sensitivity

The fixed FFN exclusion set contains **17 layers** selected from an FFN screen
(top relative-L2 and WER-delta layers). The 2-bit FFN model therefore has 79
2-bit FFN layers and 17 FP32 FFN layers, with all attention FP32.

Strategy A adds **36 attention projections at 2-bit** while keeping **60
attention projections FP32**, plus the same 79 2-bit FFN layers and 17 FP32 FFN
exclusions. The attention selection came from a checkpoint-aware 2-bit
sensitivity screen; it was not selected on test.

## 7. 1-bit versus 2-bit results

| Model | Precision | Full test WER | Artifact |
|---|---|---:|---:|
| FP32 reference | 32-bit | **1.93%** | 2,373,507,712 B |
| Mixed FFN 1-bit | 1-bit FFN + FP32 exclusions | 5.46% | 1,089,643,604 B |
| 2-bit FFN | 2-bit FFN + FP32 exclusions | **2.58%** | 1,131,062,436 B historical |
| Strategy A | 2-bit FFN + 36 attention 2-bit | **2.74%** | 989,509,572 B historical |

The 2-bit codebook recovers a large amount of the 1-bit accuracy loss. The
accuracy winner remains the 2-bit FFN model; Strategy A trades another
**0.15 pp** for **12.5% smaller** storage.

## 8. Final accuracy/compression and inference benchmark

All benchmark arms used the same A100-SXM4-80GB, the same 256 dev-clean
utterances (seed 42), greedy CTC, and separate processes per arm. The packed
arms used standalone v2 artifacts.

| Model | Dev WER | Dev CER | Full test WER | Artifact (v2) | Historical artifact | Model load | Peak host RAM | Peak GPU | Acoustic forward | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 2.0185% | 0.4934% | 1.93% | 2,373,507,712 B | 2,373,507,712 B | 2.31 s | 1,590 MB | 3.14 GB | 51.25 ms | 52.77 ms |
| 2-bit FFN | 2.4606% | 0.6770% | 2.58% | 1,131,269,460 B | 1,131,062,436 B | 7.87 s | 1,659 MB | 1.89 GB | 1,005.12 ms | 1,006.68 ms |
| Strategy A | 2.7297% | 0.7706% | 2.74% | 989,716,588 B | 989,509,572 B | 7.61 s | 1,643 MB | 1.75 GB | 1,126.61 ms | 1,128.18 ms |

- Full test WER is the historical QAT result; the benchmark is dev-only.
- FFN packed v2 dev WER differs from the QAT dev reference (2.4414%) by
  **+0.0192 pp**, one word; A matches exactly.
- Storage compression: Strategy A is **58.3% smaller** than FP32 and **12.5%
  smaller** than 2-bit FFN.
- Acoustic forward is **19.6×** slower for FFN and **22.0×** slower for A than
  FP32; end-to-end is **19.1×/21.4×** slower. The packed runtime is a storage
  format, not an accelerator.
- Peak host RAM is not reduced because the runtime expands packed weights and
  retains FP state/caches.

## 9. Failed experiments and statistical interpretation

| Experiment | Result | Decision |
|---|---|---|
| 1-bit attention A/B pilots | A 10.79%, B 17.92% dev; 0.992/0.797 GB | Not promoted |
| Per-output-channel A | 2.9115% dev-128; 2.8066% dev-256 vs A 2.8835%; Δ 0.0769 pp; bootstrap −0.113..+0.260 pp | Not promoted |
| Strategy C | 3.77% dev-128; 997,373,612 B | Rejected, dominated |
| Short-utterance oversampling | 2.7682% vs A 2.7297% dev-256; bootstrap −0.273..+0.191 pp | Rejected |
| Blank penalty 0.5 | 2.7105% vs 2.7297%; one word; p=0.46 | Not promoted |
| CTC beam 10/50 | 2.8066%; 0.23/1.33 s per utterance | Not promoted |

Interpretation: pilot deltas below ~0.1 pp on 128–256 utterances are not reliable
without a confidence interval. The per-channel and short-oversampling results
were positive/negative directions but statistically indistinguishable from zero.
The project therefore keeps the full-run selected models, not pilot checkpoints.

## 10. Final model selection

- **FP32 reference:** 1.93% full test-clean; 2.374 GB; SHA-256 `97bb9761…91f1d`.
- **Accuracy winner:** 2-bit FFN; 2.58% full test-clean; 1.131 GB.
- **Compression winner:** Strategy A; 2.74% full test-clean; 989.5 MB historical,
  989.7 MB standalone v2. Selected checkpoint step 1000; no further training.
- **Rejected:** all pilots listed in Section 9; Strategy C and the
  short-utterance checkpoint are not selected.

Selected checkpoint hashes:
- FFN: `87005f36bca859fe80bdb48f7a02a18841fed0a5e3495b2d877d2fd1a42befdb`
- A: `fc9fb3dd3521809fe2d90b7150167f6a403dd72f3f69429142bea80e74b64358`
- v2 artifacts: FFN `e8e138050f2252a3b9aa327aec87cbe3340f244509ab2b6953d491bd8881c856`;
  A `182c62d18ef00af897a21506848f1ec8246070596c8611988294bc7f80293ca9`.

## 11. Limitations

1. **Packed runtime is slower.** Codes are expanded to FP weights before
   `F.linear`; no inference acceleration is claimed.
2. **Historical artifacts are not standalone.** They omit BatchNorm/positional
   buffers and require the QAT checkpoint. v2 fixes this; historical sizes remain
   the recorded 989,509,572 / 1,131,062,436 B.
3. **Greedy decoding only.** Beam search and a blank penalty were tested and did
   not improve dev WER; no language model was tested.
4. **Clean LibriSpeech only.** No noisy, accented, multilingual, or domain-shift
   evaluation.
5. **Dev-set layer selection.** The fixed exclusion/attention choices are
   dev-derived; no test tuning was performed.
6. **Pilot statistical power is limited.** 128–256 utterances; single seed for
   most pilots.
7. **The 2-bit FFN packed v2 dev WER differs by one word** from its QAT
   reference; A matches exactly on dev.
8. **Peak host RAM is not reduced** by storage packing; model-load overhead and
   runtime expansion remain.
9. **No true bitwise kernel.** FP activations prevent XNOR/popcount replacement
   without a different model.

## 12. Future research directions

- Hardware-aware 2-bit/bitwise kernels with quantized activations; benchmark
  against the current expansion path.
- Word/character language-model rescoring on CTC N-best, tuned only on dev.
- Learned or per-channel scales with a statistically powered pilot.
- Better mixed-precision allocation using end-to-end sensitivity, especially for
  short/rare-word utterances.
- Generalization to noisy, accented, and domain-shifted speech.
- Optimized artifact loading with memory-mapped/streamed packed weights.
- Distillation or structured pruning to reduce non-quantized FP parameters,
  which dominate the sub-1 GB artifact.

## Reproducibility

```bash
# Full test suite
.venv/bin/python -m pytest -q

# Controlled deployment benchmark (one arm per process; same GPU)
python scripts/final_benchmark.py --arm fp32 --output results/final_benchmark/fp32.json
python scripts/final_benchmark.py --arm ffn2bit --output results/final_benchmark/ffn2bit.json
python scripts/final_benchmark.py --arm strategy_a --output results/final_benchmark/strategy_a.json
python scripts/aggregate_final_benchmark.py

# Read-only audit
python scripts/final_audit.py
```

Environment: Python 3.13, PyTorch 2.11.0+cu126, Transformers 5.x,
`ONEBIT_AUDIO_DECODER=soundfile`, NVIDIA A100-SXM4-80GB. The selected
checkpoints/artifacts are large and remain outside Git via `.gitignore`; hashes
and sizes are recorded in `results/final_audit.json` and the model card.

## Publication/demonstration declaration

The **research study is ready for publication** with the limitations above,
because the selected models, hashes, WERs, artifact distinction, and negative
results are documented and reproducible.

A **demonstration is ready for accuracy and storage compression**, using the v2
standalone artifacts. It is **not ready as a speed/edge deployment claim**:
packed inference is slower, and the artifacts require FP32 expansion. A real-time
or accelerated demonstration would require a true low-bit kernel and a separate
release.
