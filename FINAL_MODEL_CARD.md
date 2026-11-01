# Final Model Card

**Freeze date:** 2026-11-01  
**Scope:** LibriSpeech clean, greedy CTC, no language model, no test-set tuning  
**Reference:** `facebook/wav2vec2-conformer-rope-large-960h-ft`

## Selected models

| Role | Model | Full test-clean WER | Dev-256 WER | Historical artifact | Standalone v2 artifact |
|---|---|---:|---:|---:|---:|
| Reference | FP32 | **1.93%** | 2.0185% | 2,373,507,712 B | 2,373,507,712 B |
| Accuracy winner | 2-bit FFN | **2.58%** | 2.4414% QAT / 2.4606% packed | 1,131,062,436 B | **1,131,269,460 B** |
| Compression winner | Strategy A | **2.74%** | 2.7297% | 989,509,572 B | **989,716,588 B** |

The v2 artifacts are self-contained: they include packed 2-bit weights, FP32
scales, non-quantized FP parameters, and BatchNorm/positional buffers. The
historical artifacts remain in place and match their recorded sizes, but they
require the QAT checkpoint for those buffers and are not standalone deploy files.

## Checkpoints

### 2-bit FFN accuracy winner

- Path: `checkpoints/qat_ffn_mixed_2bit_full`
- Selected state: best dev step **1500** (restored best; 1,875 completed)
- SHA-256: `87005f36bca859fe80bdb48f7a02a18841fed0a5e3495b2d877d2fd1a42befdb`
- 79 FFN BitLinear layers, 17 fixed FP32 FFN exclusions, attention FP32
- 2-bit symmetric codebook, per-tensor mean-abs scales
- Historical packed: `checkpoints/qat_ffn_mixed_2bit_full_packed/model_packed.safetensors`
- Standalone v2: `checkpoints/qat_ffn_mixed_2bit_full_packed_v2/model_packed.safetensors`
- v2 SHA-256: `e8e138050f2252a3b9aa327aec87cbe3340f244509ab2b6953d491bd8881c856`
- Historical reload: passed against the QAT checkpoint; v2 standalone dev WER
  differs from QAT by one word (+0.0192 pp).

### Strategy A compression winner

- Path: `checkpoints/attn_2bit_on_2bitffn_A_full`
- Selected state: best dev step **1000** (restored best; 1,875 completed)
- SHA-256: `fc9fb3dd3521809fe2d90b7150167f6a403dd72f3f69429142bea80e74b64358`
- 36 attention projections 2-bit, 60 attention projections FP32
- 79 FFN layers 2-bit, 17 fixed FP32 FFN exclusions
- 115 BitLinear layers total; per-tensor mean-abs scales
- Historical packed: `checkpoints/attn_2bit_on_2bitffn_A_full_packed/model_packed.safetensors`
- Standalone v2: `checkpoints/attn_2bit_on_2bitffn_A_full_packed_v2/model_packed.safetensors`
- v2 SHA-256: `182c62d18ef00af897a21506848f1ec8246070596c8611988294bc7f80293ca9`
- Historical reload: passed against the QAT checkpoint; v2 standalone dev WER
  matches QAT exactly.

## Controlled dev-256 inference benchmark

Same A100-SXM4-80GB, same 256 dev utterances, seed 42, greedy CTC, one process
per arm. Full test WER is historical; the benchmark does not use test data.

| Model | Dev WER | Dev CER | Artifact | Load | Peak host | Peak GPU | Forward | End-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 2.0185% | 0.4934% | 2,373,507,712 B | 2.31 s | 1,590 MB | 3.14 GB | 51.25 ms | 52.77 ms |
| 2-bit FFN | 2.4606% | 0.6770% | 1,131,269,460 B | 7.87 s | 1,659 MB | 1.89 GB | 1,005.12 ms | 1,006.68 ms |
| Strategy A | 2.7297% | 0.7706% | 989,716,588 B | 7.61 s | 1,643 MB | 1.75 GB | 1,126.61 ms | 1,128.18 ms |

**No inference acceleration is claimed.** Packed weights are expanded to FP32
before `F.linear`; the measured packed runtime is substantially slower than
FP32. The benchmark is a storage-compression and latency characterization, not a
speedup claim.

## Deployment packages and streaming

Small deployable packages are in `deploy/models/<id>/`:

- `config.json`, tokenizer/processor files, manifest, metadata, `SHA256SUMS`.
- `model_packed.safetensors` is the Git LFS candidate; it is absent locally
  unless downloaded. The demo falls back to the verified local v2 artifact.

The live browser interface uses the actual `ModelManager` and packed runtime.
It is a chunked-overlap approximation, not stateful streaming. It expands codes
to FP32 before `F.linear`; no inference acceleration is claimed. CPU fallback is
functional but not real-time. See `docs/STREAMING.md`.

## Rejected experiments

- Strategy C (24 K + 10 output projections): 3.77% dev-128, 997,373,612 B;
  dominated by A.
- Per-output-channel scaling: 2.9115% dev-128; 2.8066% dev-256 vs A 2.8835%;
  Δ 0.0769 pp; bootstrap includes zero; not promoted.
- Short-utterance oversampling: 2.7682% vs A 2.7297% dev-256; rejected.
- Blank penalty 0.5: 2.7105%, one word, p=0.46; not promoted.
- CTC beam 10/50: 2.8066%, slower; not promoted.
- 1-bit attention A/B pilots: not promoted.

## Reproduction

```bash
python scripts/final_benchmark.py --arm fp32 --output results/final_benchmark/fp32.json
python scripts/final_benchmark.py --arm ffn2bit --output results/final_benchmark/ffn2bit.json
python scripts/final_benchmark.py --arm strategy_a --output results/final_benchmark/strategy_a.json
python scripts/aggregate_final_benchmark.py
python scripts/final_audit.py
.venv/bin/python -m pytest -q
```

Artifacts are large and are intentionally excluded from Git; hashes and sizes are
recorded here and in `results/final_audit.json`.
