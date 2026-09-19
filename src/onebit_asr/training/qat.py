"""Quantization-aware training (Phase 11).

QAT assembly only — it does **not** run the real cloud-GPU experiment by
itself. Everything here is exercised locally by the CPU smoke/dry run and by
unit tests.

Design (preserves the approved Phase 9/10 decisions):
- Student is the same Wav2Vec2-Conformer CTC architecture, initialized from the
  fine-tuned FP32 checkpoint.
- Selected attention + FFN ``nn.Linear`` layers are replaced by the existing
  :class:`BitLinear` (FP master weights, quantization inside ``forward``,
  STE gradients, per-tensor ``alpha = mean(|W|)``).
- Only the CNN feature extractor is frozen.
- Stock Hugging Face ``Trainer`` is used, unmodified.
- Quantization/expansion stays inside ``BitLinear``, so no Trainer changes are
  needed and the checkpoint holds ordinary FP32 master weights (NOT packed).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
from torch import nn

from onebit_asr.models.replace_layers import replace_linears
from onebit_asr.quantization.bitlinear import BitLinear

FEATURE_EXTRACTOR_PATH = "wav2vec2_conformer.feature_extractor"


# --- Model assembly ---------------------------------------------------------

def setup_qat_model(model, layers: dict, scale_mode: str = "per_tensor_mean_abs",
                    ste_clip: float | None = None):
    """Replace selected Linear layers with BitLinear, keeping FP32 masters.

    ``model`` must already be loaded from the fine-tuned FP32 checkpoint.
    Returns ``(model, replacement_report)``.
    """
    report = replace_linears(model, layers, scale_mode=scale_mode, ste_clip=ste_clip)
    return model, report


def freeze_feature_extractor(model: nn.Module) -> int:
    """Freeze the CNN feature extractor parameters. Returns count frozen."""
    feature_extractor = model.get_submodule(FEATURE_EXTRACTOR_PATH)
    frozen = 0
    for param in feature_extractor.parameters():
        param.requires_grad = False
        frozen += param.numel()
    return frozen


def freeze_report(model: nn.Module) -> dict:
    """Trainable/frozen parameter counts grouped by module role."""

    def group_of(name: str) -> str:
        if name.startswith(FEATURE_EXTRACTOR_PATH):
            return "feature_extractor_frozen"
        if name.startswith("wav2vec2_conformer.feature_projection"):
            return "feature_projection"
        if name.startswith(("lm_head", "wav2vec2_conformer.lm_head")):
            return "ctc_head"
        if name.startswith("wav2vec2_conformer.encoder"):
            module_name = name.rsplit(".", 1)[0]
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                module = None
            if isinstance(module, BitLinear) and name.endswith(".weight"):
                return "encoder_bitlinear"
            return "encoder_other"
        return "other"

    groups: dict = {}
    for name, param in model.named_parameters():
        bucket = groups.setdefault(group_of(name), {"tensors": 0, "params": 0,
                                                   "params_trainable": 0,
                                                   "params_frozen": 0})
        bucket["tensors"] += 1
        bucket["params"] += param.numel()
        if param.requires_grad:
            bucket["params_trainable"] += param.numel()
        else:
            bucket["params_frozen"] += param.numel()

    feature_extractor = model.get_submodule(FEATURE_EXTRACTOR_PATH)
    return {
        "feature_extractor_frozen": all(
            not p.requires_grad for p in feature_extractor.parameters()
        ),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_by_group": groups,
    }


def bitlinear_grad_stats(model: nn.Module) -> dict:
    """Count BitLinear layers with non-zero master-weight gradients.

    NOTE: gradients are zeroed by the optimizer after each training step, so
    this must be called immediately after a backward pass (it is 0 after
    ``trainer.train()`` returns). See ``probe_gradient_flow``.
    """
    total = 0
    with_grad = 0
    grad_norm = 0.0
    for module in model.modules():
        if isinstance(module, BitLinear):
            total += 1
            if module.weight.grad is not None:
                norm = float(module.weight.grad.detach().norm())
                if norm > 0:
                    with_grad += 1
                    grad_norm += norm
    return {"total_bitlinear": total, "with_nonzero_grad": with_grad,
            "total_grad_norm": grad_norm}


def probe_gradient_flow(model: nn.Module, batch: dict) -> dict:
    """Run one forward/backward on ``batch`` and report where gradients landed.

    This is the honest way to measure QAT gradient flow, because Trainer zeroes
    gradients after the final step. Returns BitLinear grad stats plus a check
    that the frozen feature extractor received no gradient.
    """
    # The batch comes from a CPU DataLoader, but the model may already be on a
    # CUDA device (stock Trainer places it there in __init__). Move tensors.
    device = next(model.parameters()).device
    batch = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    loss = out.loss
    loss.backward()
    stats = bitlinear_grad_stats(model)
    feature_extractor = model.get_submodule(FEATURE_EXTRACTOR_PATH)
    frozen_grad_zero = all(
        p.grad is None or torch.count_nonzero(p.grad) == 0
        for p in feature_extractor.parameters()
    )
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return {
        "probe_loss": float(loss.detach()),
        "loss_is_finite": bool(torch.isfinite(loss)),
        **stats,
        "frozen_feature_extractor_grad_is_zero": frozen_grad_zero,
    }


def master_weight_snapshot(model: nn.Module) -> dict:
    """Digest + stats of BitLinear FP master weights (to verify updates / FP-ness)."""
    digest = hashlib.sha256()
    abs_sum = 0.0
    numel = 0
    exactly_pm1 = 0
    with torch.no_grad():
        for name, module in sorted(model.named_modules()):
            if isinstance(module, BitLinear):
                weight = module.weight.detach().to(torch.float32).cpu().contiguous()
                digest.update(name.encode())
                digest.update(weight.numpy().tobytes())
                abs_sum += float(weight.abs().sum())
                numel += weight.numel()
                exactly_pm1 += int((weight.abs() == 1.0).sum())
    return {
        "digest": digest.hexdigest(),
        "abs_sum": abs_sum,
        "numel": numel,
        "fraction_exactly_pm1": (exactly_pm1 / numel) if numel else 0.0,
    }


def master_weight_delta(before: dict, after: dict) -> dict:
    """Compare two snapshots to prove the optimizer updated the masters."""
    return {
        "digest_changed": before["digest"] != after["digest"],
        "abs_sum_before": before["abs_sum"],
        "abs_sum_after": after["abs_sum"],
        "abs_sum_delta": after["abs_sum"] - before["abs_sum"],
        "bitlinear_numel": after["numel"],
        "fraction_exactly_pm1_after": after["fraction_exactly_pm1"],
        "remain_fp_not_binary": after["fraction_exactly_pm1"] < 0.5,
    }


# --- Trainer assembly -------------------------------------------------------

def make_loss_recorder():
    """Return a real ``TrainerCallback`` if transformers is available."""
    from transformers import TrainerCallback

    class _Recorder(TrainerCallback):
        def __init__(self) -> None:
            self.history: list[dict] = []

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D102
            if logs and "loss" in logs:
                self.history.append({"step": state.global_step, "loss": float(logs["loss"])})

    return _Recorder()


def resolve_warmup_steps(training: dict, train_num_samples: int | None, smoke: bool) -> int:
    """Convert the configurable ``warmup_ratio`` into ``warmup_steps``.

    transformers 5.x ``TrainingArguments`` accepts ``warmup_steps`` only, so the
    ratio is resolved against the actual number of optimizer steps here.
    """
    ratio = float(training.get("warmup_ratio", 0.0) or 0.0)
    if ratio <= 0:
        return 0
    eff_batch = max(1, int(training["batch_size"]) * int(training.get("gradient_accumulation_steps", 1)))
    if smoke:
        total_steps = int(training.get("max_steps", 8))
    else:
        if not train_num_samples:
            return 0
        epochs = int(math.ceil(float(training.get("epochs", 1))))
        total_steps = max(1, math.ceil(int(train_num_samples) / eff_batch) * epochs)
    return max(1, int(ratio * total_steps))


def build_training_arguments(training: dict, output_dir: str, smoke: bool = False,
                             train_num_samples: int | None = None):
    """Build stock ``TrainingArguments`` from the config training block.

    ``gradient_checkpointing`` is taken explicitly from the config (never
    hard-coded); when enabled, the Trainer handles enabling checkpointing.
    ``warmup_ratio`` is resolved to ``warmup_steps`` (transformers 5.x API).
    """
    from transformers import TrainingArguments

    warmup_steps = resolve_warmup_steps(training, train_num_samples, smoke)
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=int(training["batch_size"]),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training.get("epochs", 1)),
        warmup_steps=warmup_steps,
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        optim=training.get("optimizer", "adamw_torch"),
        lr_scheduler_type=training.get("lr_scheduler", "linear"),
        logging_steps=int(training.get("logging_steps", 10)),
        save_strategy=training.get("save_strategy", "no"),
        seed=int(training.get("seed", 42)),
        fp16=bool(training.get("fp16", False)),
        bf16=bool(training.get("bf16", False)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", False)),
        report_to=[],
        remove_unused_columns=True,
        dataloader_num_workers=0,
    )
    if smoke:
        kwargs["num_train_epochs"] = 1.0
        kwargs["max_steps"] = int(training.get("max_steps", 8))
        kwargs["save_strategy"] = "no"
        kwargs["logging_steps"] = 1
    return TrainingArguments(**kwargs)


def build_trainer(model, train_dataset, collator, training_args, loss_recorder=None):
    """Create a stock ``Trainer`` (unmodified)."""
    from transformers import Trainer

    callbacks = [loss_recorder] if loss_recorder is not None else None
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )


# --- Checkpointing (FP master weights; NOT 1-bit packed) --------------------

CHECKPOINT_FORMAT = "fp32_master_weights"


def save_qat_checkpoint(model: nn.Module, out_dir: str | Path) -> dict:
    """Save FP32 master weights + an explicit manifest.

    The saved file is a normal FP32 checkpoint. It is **not** bit-packed; the
    1-bit behavior only exists at forward time (see ``expected_reconstruction``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)

    weight_files = sorted(p for p in out_dir.glob("*.safetensors")) + \
        sorted(p for p in out_dir.glob("pytorch_model.bin"))
    manifest = {
        "format": CHECKPOINT_FORMAT,
        "is_1bit_packed": False,
        "description": (
            "FP32 master weights for every parameter (BitLinear masters, biases, "
            "LayerNorms, convolutions, feature projection, CTC head). Structurally "
            "identical to the original FP32 checkpoint; binary weights exist only "
            "at forward time."
        ),
        "expected_reconstruction": (
            "Wav2Vec2ConformerForCTC.from_pretrained(dir) then replace_linears(...) "
            "with the same layer selection; forward() binarizes from these masters."
        ),
        "bitlinear_layers": sum(1 for m in model.modules() if isinstance(m, BitLinear)),
        "total_params": sum(p.numel() for p in model.parameters()),
        "files": [p.name for p in weight_files],
        "sha256": {p.name: _sha256(p) for p in weight_files},
    }
    (out_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_qat_checkpoint(out_dir: str | Path, model_cls, layers: dict,
                        scale_mode: str = "per_tensor_mean_abs",
                        ste_clip: float | None = None):
    """Reload FP32 masters and reconstruct the binarized model."""
    model = model_cls.from_pretrained(out_dir)
    model, report = setup_qat_model(model, layers, scale_mode=scale_mode, ste_clip=ste_clip)
    return model, report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- GPU probe (Phase 11 §7b): measure before committing to a full run ------

def reset_gpu_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_memory_stats() -> dict:
    """Peak memory for this process (torch) plus whole-GPU usage (nvidia-smi).

    On a CPU host this returns ``cuda: False`` with null fields, so the probe
    still works locally for validation.
    """
    if not torch.cuda.is_available():
        return {
            "cuda": False,
            "device": "cpu",
            "device_name": None,
            "total_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "nvidia_smi_memory_used_mb": _nvidia_smi_used_mb(),
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "device": f"cuda:{torch.cuda.current_device()}",
        "device_name": props.name,
        "total_bytes": int(props.total_memory),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "nvidia_smi_memory_used_mb": _nvidia_smi_used_mb(),
    }


def _nvidia_smi_used_mb() -> int | None:
    """Whole-GPU used memory in MB (includes other processes); None if absent."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=15,
        )
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def extrapolate_training_time(
    seconds_per_optimizer_step: float,
    subset_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> dict:
    """Extrapolate full-run time from the probe's measured seconds/step."""
    if seconds_per_optimizer_step <= 0:
        raise ValueError("seconds_per_optimizer_step must be positive")
    micro_batches = math.ceil(int(subset_size) * int(epochs) / max(1, int(batch_size)))
    optimizer_steps = math.ceil(micro_batches / max(1, int(gradient_accumulation_steps)))
    total_seconds = seconds_per_optimizer_step * optimizer_steps
    return {
        "micro_batches_total": micro_batches,
        "optimizer_steps_total": optimizer_steps,
        "estimated_seconds": total_seconds,
        "estimated_minutes": total_seconds / 60,
        "estimated_hours": total_seconds / 3600,
    }


def vram_headroom(peak_bytes: int | None, total_bytes: int | None,
                  required_fraction: float = 0.15) -> dict:
    """Check peak VRAM leaves the required fraction of total free."""
    if peak_bytes is None or total_bytes is None or total_bytes <= 0:
        return {"applicable": False, "required_fraction": required_fraction,
                "passes": None, "free_bytes": None, "free_fraction": None}
    free = int(total_bytes) - int(peak_bytes)
    free_fraction = free / int(total_bytes)
    return {
        "applicable": True,
        "required_fraction": required_fraction,
        "passes": free_fraction >= required_fraction,
        "free_bytes": free,
        "free_fraction": free_fraction,
    }
