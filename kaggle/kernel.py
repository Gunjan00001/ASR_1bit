#!/usr/bin/env python3
"""Kaggle execution kernel - attention-only 1-bit QAT (Phase 11).

Workflow: local Windows + OpenCode -> Kaggle CLI -> this kernel (Kaggle GPU).

This file is a Kaggle *script* kernel. scripts/kaggle_submit.ps1 pushes it and
replaces the ``__REPO_REVISION__`` placeholder with the exact pushed GitHub
commit SHA, so every run is tied to a reproducible revision. GitHub remains the
source of truth: this kernel clones the repo at that SHA.

Flow:
  1. report environment (python / torch / CUDA / GPU / VRAM)
  2. clone the GitHub repo at the pinned revision
  3. install Kaggle deps WITHOUT touching the preinstalled CUDA torch stack
  4. assert torch.cuda.is_available()
  5. verify audio decoding (torchcodec, else soundfile fallback)
  6. verify attention-only replacement == 96 BitLinear layers
  7. run the existing GPU QAT probe (fp16 for T4)
  8. require the VRAM headroom gate to pass; else retry ONCE with
     gradient_checkpointing=true; else ABORT (never silently shrink the run)
  9. run the full attention-only QAT
 10. write the real packed 1-bit deploy artifact
 11. copy results / logs / checkpoints into /kaggle/working
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

REPO_URL = "https://github.com/Gunjan00001/ASR_1bit.git"
REPO_REVISION = "__REPO_REVISION__"

CONFIG_PRISTINE = "configs/qat_attn.yaml"
CONFIG_EFFECTIVE = "configs/kaggle_effective_attn.yaml"

PROBE_OUTPUT_1 = "results/qat_probe_attn.json"
PROBE_OUTPUT_2 = "results/qat_probe_attn_gc.json"
FULL_OUTPUT = "results/qat_attn.json"
CHECKPOINT_DIR = "checkpoints/qat_attn"
PACKED_DIR = "checkpoints/qat_attn_packed"
REPLACEMENT_REPORT = "replacement_check.json"
EXPECTED_REPLACEMENTS = 96

WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp")
REPO_DIR = TMP / "ASR_1bit"
LOG_DIR = WORK / "logs"
MASTER_LOG = LOG_DIR / "kaggle_qat_attn.log"

# --- Environment (must be set before importing torch / HF) -------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", str(TMP / "hf"))
os.environ.setdefault("HF_HUB_CACHE", str(TMP / "hf" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(TMP / "hf" / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(TMP / "hf" / "transformers"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Reduce CUDA allocator fragmentation on the 16 GB T4.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def log(message: str) -> None:
    print(message, flush=True)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(MASTER_LOG, "a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    except Exception:  # noqa: BLE001
        pass


def run(cmd, cwd: Path = REPO_DIR, check: bool = True, extra_env: dict | None = None) -> int:
    """Run a command, streaming combined output to console and the master log."""
    cmd = [str(c) for c in cmd]
    log(f"$ {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        try:
            with open(MASTER_LOG, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:  # noqa: BLE001
            pass
    proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def read_json(path: Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# --- 1. Environment ----------------------------------------------------------

def report_environment() -> dict:
    import platform

    import torch

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
        info["ram_available_gb"] = round(vm.available / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    for cg in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            if os.path.isfile(cg):
                raw = open(cg, encoding="utf-8").read().strip()
                if raw and raw != "max":
                    info.setdefault("cgroup_memory_limit_gb", {})[cg] = round(int(raw) / 1e9, 2)
        except Exception:  # noqa: BLE001
            pass
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": props.name,
            "gpu_total_memory_bytes": int(props.total_memory),
            "gpu_total_memory_gb": round(props.total_memory / 1e9, 2),
            "gpu_compute_capability": f"{props.major}.{props.minor}",
        })
    log("=== Environment ===")
    log(json.dumps(info, indent=2))
    return info


# --- 2. Clone ----------------------------------------------------------------

def clone_repo() -> None:
    log(f"=== Cloning {REPO_URL} @ {REPO_REVISION} ===")
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--quiet", REPO_URL, str(REPO_DIR)], cwd=TMP)
    run(["git", "checkout", "--quiet", REPO_REVISION], cwd=REPO_DIR)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), text=True,
    ).strip()
    if head != REPO_REVISION:
        raise RuntimeError(f"checkout mismatch: HEAD={head} expected={REPO_REVISION}")
    log(f"Checked out {head}")


# --- 3/4. Dependencies + CUDA assertion --------------------------------------

STACK_CHECK = (
    "import json, torch, torchvision;"
    "assert torch.cuda.is_available(), 'CUDA not available after install';"
    "from torchvision.io import decode_image;"
    "from transformers import Wav2Vec2ConformerForCTC;"
    "d=torch.cuda.get_device_properties(0);"
    "print(json.dumps({'torch':torch.__version__,'torchvision':torchvision.__version__,"
    "'cuda':torch.version.cuda,'device':d.name,'total_gb':d.total_memory/1e9,"
    "'capability':f'{d.major}.{d.minor}','count':torch.cuda.device_count()}))"
)


def install_dependencies() -> None:
    log("=== Installing Kaggle dependencies (matching CUDA 13 torch stack) ===")
    run([sys.executable, "-m", "pip", "install", "-r", "kaggle/requirements-kaggle.txt",
         "-c", "kaggle/constraints-kaggle.txt"], cwd=REPO_DIR)
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=REPO_DIR)
    log("=== Removing stale torchaudio (unused; built against the old torch) ===")
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], cwd=REPO_DIR,
        check=False)
    log("=== Asserting torch/torchvision/transformers stack is intact ===")
    code = run([sys.executable, "-c", STACK_CHECK], cwd=REPO_DIR, check=False)
    if code != 0:
        raise RuntimeError(
            "torch/torchvision/transformers stack broken after install; aborting"
        )


# --- 5. Audio decoder --------------------------------------------------------

DECODER_CHECK = "from torchcodec.decoders import AudioDecoder; print('torchcodec ok')"

DECODE_SAMPLE_CHECK = (
    "from onebit_asr.data.dataset import load_train_subset;"
    "s=load_train_subset(subset_size=1, seed=42, split='train.100');"
    "print('decoded', s[0]['id'], s[0]['audio'].shape)"
)


def configure_audio_decoder() -> None:
    log("=== Verifying audio decoder ===")
    if run([sys.executable, "-c", DECODER_CHECK], cwd=REPO_DIR, check=False) == 0:
        log("torchcodec import OK")
    else:
        log("torchcodec unavailable -> using soundfile decoder (ONEBIT_AUDIO_DECODER=soundfile)")
        os.environ["ONEBIT_AUDIO_DECODER"] = "soundfile"

    if run([sys.executable, "-c", DECODE_SAMPLE_CHECK], cwd=REPO_DIR, check=False) != 0:
        log("default decoding failed -> retrying with soundfile decoder")
        os.environ["ONEBIT_AUDIO_DECODER"] = "soundfile"
        if run([sys.executable, "-c", DECODE_SAMPLE_CHECK], cwd=REPO_DIR, check=False) != 0:
            raise RuntimeError("could not decode a LibriSpeech sample with any audio backend")


# --- 6/7. Replacement + effective config + probe -----------------------------

def write_effective_config(gradient_checkpointing: bool) -> None:
    """Write the T4-specific effective config that extends the pristine spec."""
    content = (
        "# GENERATED by kaggle/kernel.py - do not edit or commit.\n"
        f"extends: qat_attn.yaml\n"
        "qat:\n"
        "  training:\n"
        "    fp16: true          # T4 (Turing) has no bf16\n"
        "    bf16: false\n"
        f"    gradient_checkpointing: {str(bool(gradient_checkpointing)).lower()}\n"
        # HF Trainer's epoch checkpoints carry optimizer state (~5 GB each) and
        # would exhaust Kaggle's 20 GB /kaggle/working quota. We persist the
        # final model ourselves via save_qat_checkpoint, so disable them.
        "    save_strategy: \"no\"\n"
    )
    (REPO_DIR / CONFIG_EFFECTIVE).write_text(content, encoding="utf-8")
    log(f"Effective config ({CONFIG_EFFECTIVE}):")
    log(content)


def verify_replacement() -> dict:
    log("=== Verifying attention-only replacement ===")
    run([sys.executable, "kaggle/verify_replacement.py", "--config", CONFIG_EFFECTIVE,
         "--output", REPLACEMENT_REPORT], cwd=REPO_DIR)
    result = read_json(REPO_DIR / REPLACEMENT_REPORT)
    if result.get("n_bitlinear") != EXPECTED_REPLACEMENTS:
        raise RuntimeError(
            f"replacement count {result.get('n_bitlinear')} != {EXPECTED_REPLACEMENTS}"
        )
    return result


def run_probe(output_rel: str) -> dict:
    """Run the probe. A hard crash (e.g. CUDA OOM before any measurement) is a
    gate failure, not a fatal error, so the caller can retry with gradient
    checkpointing."""
    code = run([sys.executable, "scripts/train_qat.py", "--config", CONFIG_EFFECTIVE,
                "--probe", "--output", output_rel], cwd=REPO_DIR, check=False,
               extra_env={"PYTHONUNBUFFERED": "1"})
    if code != 0:
        log(f"probe process exited with code {code}; treating as gate failure")
        return {}
    return read_json(REPO_DIR / output_rel)


def gate(probe: dict) -> tuple[bool, str]:
    if not probe:
        return False, "probe produced no output"
    if not probe.get("is_real_gpu_probe"):
        return False, "not a real GPU probe (is_real_gpu_probe=false)"
    headroom = probe.get("vram_headroom", {})
    if not headroom.get("applicable"):
        return False, "VRAM headroom not applicable"
    return bool(headroom.get("passes")), json.dumps(headroom)


# --- 9/10. Full run + packing ------------------------------------------------

def log_disk(tag: str) -> None:
    try:
        usage = shutil.disk_usage(WORK)
        log(f"[disk:{tag}] /kaggle/working free {usage.free / 1e9:.2f} GB "
            f"of {usage.total / 1e9:.2f} GB")
    except Exception:  # noqa: BLE001
        pass


def cleanup_trainer_checkpoints() -> None:
    """Remove HF Trainer epoch checkpoints (optimizer state) to free disk.

    With save_strategy 'no' there should be none, but this is a safety net.
    """
    base = REPO_DIR / CHECKPOINT_DIR
    removed = 0
    for path in sorted(base.glob("checkpoint-*")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    if removed:
        log(f"removed {removed} Trainer checkpoint dir(s) from {CHECKPOINT_DIR}")
    log_disk("after-trainer-cleanup")


def run_full_training() -> None:
    log("=== Probe passed: running FULL attention-only QAT ===")
    log_disk("before-full")
    run([sys.executable, "scripts/train_qat.py", "--config", CONFIG_EFFECTIVE,
         "--output", FULL_OUTPUT, "--checkpoint-dir", CHECKPOINT_DIR], cwd=REPO_DIR)
    cleanup_trainer_checkpoints()
    log_disk("after-full")


def pack_model() -> None:
    log("=== Writing packed 1-bit deploy artifact ===")
    run([sys.executable, "scripts/pack_qat_model.py", "--config", CONFIG_EFFECTIVE,
         "--checkpoint", CHECKPOINT_DIR,
         "--output", f"{PACKED_DIR}/model_packed.safetensors",
         "--report", f"{PACKED_DIR}/size_report.json"], cwd=REPO_DIR)


# --- 11. Collect outputs -----------------------------------------------------

def collect_outputs() -> None:
    log("=== Collecting outputs into /kaggle/working ===")
    results_out = WORK / "results"
    results_out.mkdir(parents=True, exist_ok=True)
    for rel in ["results/qat_attn.json", PROBE_OUTPUT_1, PROBE_OUTPUT_2,
                REPLACEMENT_REPORT]:
        src = REPO_DIR / rel
        if src.is_file():
            shutil.copy2(src, results_out / src.name)
            log(f"copied {rel} -> {results_out / src.name}")
    for rel in [CHECKPOINT_DIR, PACKED_DIR]:
        src = REPO_DIR / rel
        if src.is_dir():
            dst = WORK / rel
            if dst.exists():
                shutil.rmtree(dst)
            # Never copy HF Trainer checkpoint-* dirs (optimizer state, ~5 GB).
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoint-*"))
            log(f"copied {rel} -> {dst}")

    results = read_json(results_out / "qat_attn.json")
    probe = read_json(results_out / "qat_probe_attn.json")
    replacement = read_json(results_out / "replacement_check.json")
    summary = {
        "revision": REPO_REVISION,
        "config_pristine": CONFIG_PRISTINE,
        "config_effective": CONFIG_EFFECTIVE,
        "replacement": replacement,
        "probe_gate": {
            "headroom": probe.get("vram_headroom"),
            "is_real_gpu_probe": probe.get("is_real_gpu_probe"),
        },
        "eval": results.get("eval"),
        "checkpoint": results.get("checkpoint", {}).get("path"),
        "packed_size_report": read_json(WORK / PACKED_DIR / "size_report.json"),
    }
    (WORK / "kaggle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("=== Summary ===")
    log(json.dumps(summary, indent=2))


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    try:
        report_environment()
        log_disk("start")
        clone_repo()
        install_dependencies()
        configure_audio_decoder()
        write_effective_config(gradient_checkpointing=False)
        verify_replacement()

        probe = run_probe(PROBE_OUTPUT_1)
        passed, reason = gate(probe)
        log(f"Probe attempt 1 gate: {'PASS' if passed else 'FAIL'} ({reason})")
        if not passed:
            log("=== Gate failed: retrying probe with gradient_checkpointing=true ===")
            write_effective_config(gradient_checkpointing=True)
            probe = run_probe(PROBE_OUTPUT_2)
            passed, reason = gate(probe)
            log(f"Probe attempt 2 (grad-checkpointing) gate: "
                f"{'PASS' if passed else 'FAIL'} ({reason})")
            if not passed:
                log("=== ABORT: VRAM headroom gate failed after retry; not reducing the run ===")
                collect_outputs()
                return 2

        run_full_training()
        try:
            pack_model()
        except Exception:  # noqa: BLE001
            log("WARNING: packed-artifact generation failed; training results preserved")
            log(traceback.format_exc())
        collect_outputs()
        log("=== DONE ===")
        return 0
    except Exception:  # noqa: BLE001
        log("=== FAILED ===")
        log(traceback.format_exc())
        try:
            collect_outputs()
        except Exception:  # noqa: BLE001
            log("(output collection also failed; partial artifacts remain in /kaggle/working)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
