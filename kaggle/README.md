# Kaggle execution — attention-only 1-bit QAT

Local Windows + OpenCode → Kaggle CLI → Kaggle GPU. No OpenCode on Kaggle, no
SSH, no GPU rental. **GitHub is the source of truth**: the kernel clones the
repo at the exact commit SHA that `scripts/kaggle_submit.ps1` pins.

## What runs

`kaggle/kernel.py` (a Kaggle *script* kernel) does, in order:

1. Report Python / torch / CUDA / GPU name / VRAM.
2. Clone `Gunjan00001/ASR_1bit` at the pinned revision.
3. Install `kaggle/requirements-kaggle.txt`, which pins the matching CUDA 13
   stack (`torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`) because the Kaggle
   image shipped an older mismatched pair; stale `torchaudio` is removed. Then
   `pip install -e .`.
4. Assert `torch.cuda.is_available()`.
5. Verify audio decoding (torchcodec, else `ONEBIT_AUDIO_DECODER=soundfile`).
6. Verify the attention-only replacement is **exactly 96 BitLinear**.
7. Run the existing GPU QAT probe (`scripts/train_qat.py --probe`) with an
   effective T4 config (`fp16: true`, `gradient_checkpointing` per attempt).
8. **Require the VRAM headroom gate to pass.** If attempt 1 fails, retry once
   with `gradient_checkpointing: true`; if that fails too, abort.
9. Run the full attention-only QAT → `results/qat_attn_repro.json`,
   `checkpoints/qat_attn/` (the `*_repro` name protects the original v7 result
   `results/qat_attn.json`, which keeps its historical `save_load_verified: false`).
10. Write the real packed 1-bit deploy artifact →
    `checkpoints/qat_attn_packed/model_packed.safetensors` + `size_report.json`
    (also copied to `results/qat_attn_repro_size_report.json`).
11. Copy results / logs / checkpoints into `/kaggle/working`.

`configs/qat_attn.yaml` stays pristine (experiment definition only). The
hardware-specific settings live in a generated effective config,
`configs/kaggle_effective_attn.yaml` (created inside the Kaggle clone only).

## Local commands

```powershell
# Submit (requires a clean, pushed tree; pins the current HEAD SHA)
powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_submit.ps1
# optional 24 GB accelerator instead of T4:
powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_submit.ps1 -Accelerator NvidiaL4

# Status
powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_status.ps1

# Download outputs to kaggle\output\
powershell -ExecutionPolicy Bypass -File .\scripts\kaggle_output.ps1
```

Raw CLI equivalents:

```powershell
kaggle kernels push -p kaggle\build
kaggle kernels status gunjanpal/asr-1bit-qat-attn
kaggle kernels output gunjanpal/asr-1bit-qat-attn -p kaggle\output
```

## Files

| File | Role |
|---|---|
| `kernel.py` | Kaggle script kernel (streams logs, enforces the probe gate). |
| `kernel-metadata.template.json` | Metadata template (`__KERNEL_ID__`, `__MACHINE_SHAPE__`). |
| `requirements-kaggle.txt` | Deps minus the torch stack. |
| `constraints-kaggle.txt` | Fail-fast pins for torch/torchvision. |
| `verify_replacement.py` | Asserts exactly 96 BitLinear layers. |

`kaggle/build/` and `kaggle/output/` are generated locally and git-ignored.
