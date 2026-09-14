# One-Bit Wav2Vec2-Conformer ASR

Research project: progressively replace FP linear layers of a pretrained
Wav2Vec2-Conformer CTC model (`facebook/wav2vec2-conformer-rope-large-960h-ft`)
with custom 1-bit **BitLinear** layers, recovering accuracy via QAT and
knowledge distillation.

**Core question:** can a Wav2Vec2-Conformer ASR model be made substantially
smaller using 1-bit weights while retaining competitive accuracy?

See `one_bit_asr_implementation_plan.md` for the full plan.

## Versioning

Releases follow `major.minor.patch` (semantic versioning) and are pushed as
annotated git tags `vX.Y.Z`:

- **major** — breaking changes (config format, results schema, CLI behavior)
- **minor** — new phases/features (BitLinear, QAT, distillation, benchmarks)
- **patch** — bug fixes, doc updates, pins

`pyproject.toml` carries the current version. Tag a release with:

```powershell
git tag -a vX.Y.Z -m "Release vX.Y.Z: <summary>"
git push origin main vX.Y.Z
```

## Status

- [x] Phase 0 — environment (`.venv`, pinned `requirements.txt`, `scripts/check_env.py` → PASS, `results/env.json`)
- [x] Phase 1 — model inspection (`scripts/inspect_model.py` → `results/model_inspection.json`, see findings below)
- [x] Phase 2 — inference smoke test (`scripts/infer.py --demo` → exact-match transcription on a LibriSpeech sample)
- [x] Phase 3 — FP32 baseline (`scripts/eval_baseline.py` → `results/baseline.json`, see below)
- [ ] Phase 4+ — see `one_bit_asr_implementation_plan.md` (everything beyond Phase 3 is scaffolded as placeholders)

## FP32 baseline (Phase 3, `local` CPU profile)

Fixed 256-utterance subset of LibriSpeech clean `validation` (seed 42, IDs in
`results/baseline.json`), greedy CTC, normalized text:

| Metric | Value |
|---|---|
| WER | 0.0202 (2.02%) |
| CER | 0.0049 (0.49%) |
| RTF | 0.284 |
| Latency p50 / p95 | 1.60s / 5.02s |
| Peak RSS | 3.5 GB |
| Checkpoint on disk | 4.75 GB (`model.safetensors` + duplicate `pytorch_model.bin`) |

WER matches the model-card reference (1.96 test-clean) — the baseline is sane
and is now the reference for all 1-bit experiments.

## Key findings (Phase 1)

Base model `facebook/wav2vec2-conformer-rope-large-960h-ft` (`Wav2Vec2ConformerForCTC`):

- 593,376,928 params, FP32 (~2.37 GB), 24 × `Wav2Vec2ConformerEncoderLayer`
- 194 `nn.Linear` layers holding 504.2M params (**84.97%** of the model)
- Per layer: `ffn1/2.intermediate_dense` (1024→4096), `ffn1/2.output_dense`
  (4096→1024), `self_attn.linear_q/k/v/out` (1024→1024) — note the HF naming
  (not fairseq-style `q_proj`/`linear1`); layer-replacement configs must match it
- Quantization candidate pool (attention + FFN, CTC head / feature projection excluded): **503.7M params (84.87%)**
- Reference WER from model card (greedy): 1.96 test-clean / 3.98 test-other

## Results

Machine-readable outputs live in `results/` (tracked in git):

- `env.json` — environment diagnostic report
- `model_inspection.json` — full Linear-layer map with categories

## Setup (Windows, CPU)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## First steps

```powershell
.venv\Scripts\python.exe scripts\check_env.py
.\scripts\setup_windows_ffmpeg.ps1   # one-time: FFmpeg shared DLLs for audio decoding
.venv\Scripts\python.exe scripts\inspect_model.py
.venv\Scripts\python.exe scripts\infer.py --demo
```

## Windows audio-decoding note

Hugging Face `datasets` audio decoding requires `torchcodec`, which needs
FFmpeg **shared** libraries (`avcodec-*.dll`, ...). A static `ffmpeg.exe` is
not enough, and Python 3.14 no longer searches the app directory for DLLs.
`scripts\setup_windows_ffmpeg.ps1` downloads a BtbN `win64-gpl-shared` build,
copies its DLLs into `.venv\Scripts`, and registers that directory via a
`.pth` file. Re-run it whenever `.venv` is recreated. (Linux/cloud-GPU
profiles install FFmpeg shared libraries via the system package manager
instead.)

## Compute profiles

- `local` (CPU): inspection, inference, dev subsets, BitLinear unit work, PTQ.
- `cloud-gpu`: QAT, distillation, full benchmarks. Same configs, same pinned
  dependency versions (see `configs/`).

## Layout

- `src/onebit_asr/` — package (data, features, models, quantization, training,
  decoding, evaluation)
- `configs/` — baseline / binary / QAT / distillation experiment configs
- `scripts/` — entry points (`inspect_model.py`, `infer.py`, ...)
- `tests/` — unit tests (BitLinear tests gate integration per the plan)
- `results/` — machine-readable experiment outputs (tracked in git)
- `checkpoints/` — model checkpoints (ignored by git)
