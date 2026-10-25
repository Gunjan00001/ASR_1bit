"""Knowledge distillation (Phase 13): FP32 teacher -> 1-bit student.

Loss (plan §18):

    L = lambda_ctc * CTC(student, y)
      + lambda_kd * T^2 * KL(student_logits/T || teacher_logits/T)

The teacher is frozen FP32 and its logits are precomputed **once per training
subset** (``precompute_teacher_logits``) so distillation costs barely more than
QAT alone at training time. Teacher logits are stored as FP16 on CPU and aligned
with the student by sample order; per-sample valid lengths are carried in a
``teacher_mask`` so padded frames never enter the KL term.

The student is the same :class:`BitLinear`-replaced Wav2Vec2-Conformer used by
QAT: FP master weights, ``alpha * sign(W)`` inside ``forward``, pass-through
STE. Nothing about the quantization method changes here.

Debugging notes
---------------
- **KL direction matters.** This is ``KL(student || teacher)``, not the reverse.
  ``F.kl_div(log_student, teacher_probs)`` computes the reverse and will change
  the gradient; the loss here uses the explicit formula.
- **The KL is summed over frames**, matching the summed CTC loss. A mean would
  make the KD term ~1000× smaller (frames ~1000) and λ_kd would be meaningless.
- **The teacher mask is not optional in spirit.** Padded teacher frames contain
  zeros; without the mask they would add a large spurious KL term.
- **Teacher logits are precomputed once** and cached. If you change the teacher,
  split, subset size, or seed, the cache key changes; pass
  ``--refresh-teacher-cache`` to force a recompute.
- **`remove_unused_columns=False` is required** for the Trainer: the batch has
  `teacher_logits` / `teacher_mask`, which are not in the model signature.
  `DistillationTrainer.compute_loss` pops them before calling the model.
- **Inspect `loss_components`.** If KD is ≫ CTC, the student is optimizing the
  teacher distribution; lower λ_kd. This is exactly how λ_kd=0.5 was diagnosed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from onebit_asr.data.collator import DataCollatorCTCWithPadding


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_mask: torch.Tensor | None,
    *,
    ctc_loss: torch.Tensor,
    temperature: float = 2.0,
    lambda_ctc: float = 0.5,
    lambda_kd: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """Combine the ground-truth CTC loss with frame-level teacher KL.

    Args:
        student_logits: ``(B, T_student, V)``.
        teacher_logits: ``(B, T_teacher, V)``; padded with zeros.
        teacher_mask: ``(B, T_teacher)`` with 1 for real frames, 0 for padding.
            ``None`` means every frame up to the common length is valid.
        ctc_loss: the student's CTC loss against the ground-truth labels.
        temperature: KD temperature ``T``.
        lambda_ctc / lambda_kd: loss weights.

    Returns:
        ``(total_loss, {"ctc": ..., "kd": ...})``.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if student_logits.size(-1) != teacher_logits.size(-1):
        raise ValueError(
            f"vocabulary mismatch: student {student_logits.size(-1)} vs "
            f"teacher {teacher_logits.size(-1)}"
        )

    # Same architecture and same (frozen) conv stack => equal lengths. Cropping
    # to the common minimum keeps the math valid even if that ever changes.
    frames = min(student_logits.size(1), teacher_logits.size(1))
    student = student_logits[:, :frames, :].float()
    teacher = teacher_logits[:, :frames, :].float()

    if teacher_mask is None:
        mask = torch.ones(student.shape[:2], dtype=student.dtype, device=student.device)
    else:
        mask = teacher_mask[:, :frames].to(device=student.device, dtype=student.dtype)

    temp = float(temperature)
    # Temperature softening: divide logits before softmax. T^2 restores the
    # gradient scale so λ_kd does not have to change with T.
    log_student = F.log_softmax(student / temp, dim=-1)
    log_teacher = F.log_softmax(teacher / temp, dim=-1)
    student_probs = log_student.exp()
    # KL(student || teacher) over the vocabulary, summed over valid frames.
    # A sum (not a mean) keeps the KD term on the same scale as the summed CTC
    # loss, so the configured lambda weights are directly comparable.
    # The mask is important: teacher padding is zeros, which would otherwise
    # contribute a large, meaningless KL term at the end of every sequence.
    kd_per_frame = (student_probs * (log_student - log_teacher)).sum(-1)
    kd = (kd_per_frame * mask).sum() * (temp * temp)

    total = float(lambda_ctc) * ctc_loss + float(lambda_kd) * kd
    return total, {
        "ctc": float(ctc_loss.detach()),
        "kd": float(kd.detach()),
        "kd_mean_per_frame": float(kd.detach()) / (temp * temp)
        / float(mask.sum().clamp_min(1.0)),
        "valid_frames": float(mask.sum()),
    }


@torch.inference_mode()
def precompute_teacher_logits(
    teacher: nn.Module,
    dataset,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float16,
    max_samples: int | None = None,
    progress_every: int = 500,
) -> list[torch.Tensor]:
    """Run the frozen teacher over ``dataset`` once and return per-sample logits.

    Each returned tensor is ``(T_i, V)`` on CPU in ``dtype``. Processing one
    utterance at a time avoids padding artifacts entirely; on an A100 this is a
    few minutes for 10k utterances and is paid once for all epochs.

    Why one at a time? Batching would require computing each sample's valid
    output length (the conv stack's length formula) and cropping; a mistake there
    silently corrupts the KD target. Correctness first; the cache makes the cost
    amortized.
    """
    teacher.eval()
    teacher.to(device)
    n = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    logits_list: list[torch.Tensor] = []
    for i in range(n):
        row = dataset[i]
        input_values = torch.as_tensor(
            np.asarray(row["input_values"], dtype=np.float32)
        ).unsqueeze(0).to(device)
        out = teacher(input_values=input_values).logits[0]
        logits_list.append(out.detach().to(dtype=dtype).cpu())
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  teacher logits {i + 1}/{n}", flush=True)
    return logits_list


def save_teacher_logits(logits_list: list[torch.Tensor], path: str | Path, meta: dict | None = None) -> dict:
    """Persist precomputed teacher logits (CPU, FP16) with provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logits": logits_list,
        "n_samples": len(logits_list),
        "dtype": str(logits_list[0].dtype) if logits_list else "none",
        "meta": meta or {},
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "n_samples": len(logits_list),
        "dtype": payload["dtype"],
        "bytes": path.stat().st_size if path.is_file() else None,
        "meta": payload["meta"],
    }


def load_teacher_logits(path: str | Path) -> tuple[list[torch.Tensor], dict]:
    """Load precomputed teacher logits and their metadata."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return payload["logits"], payload.get("meta", {})


class DistillationDataset(torch.utils.data.Dataset):
    """Pair a CTC feature dataset with precomputed per-sample teacher logits."""

    def __init__(self, base, teacher_logits: list[torch.Tensor]) -> None:
        if len(base) != len(teacher_logits):
            raise ValueError(
                f"dataset/teacher length mismatch: {len(base)} vs {len(teacher_logits)}"
            )
        self.base = base
        self.teacher_logits = teacher_logits

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        row = self.base[index]
        return {
            "input_values": np.asarray(row["input_values"], dtype=np.float32),
            "labels": list(row["labels"]),
            "teacher_logits": self.teacher_logits[index].numpy(),
        }


@dataclass
class DistillationCollator:
    """Pad inputs/labels (CTC) and teacher logits, returning a frame mask."""

    processor: object
    padding: bool | str = True
    label_pad_token_id: int = -100

    def __post_init__(self) -> None:
        self._ctc = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=self.padding,
            label_pad_token_id=self.label_pad_token_id,
        )

    def __call__(self, features: list[dict]) -> dict:
        # Pop the teacher data BEFORE the CTC collator: that collator calls the
        # processor/tokenizer and must not see extra columns. The dataset wrapper
        # creates a fresh dict per __getitem__, so popping is safe.
        teacher = [f.pop("teacher_logits") for f in features]
        batch = self._ctc(features)

        # Pad teacher logits to the longest teacher sequence in this batch. The
        # mask marks real frames; zero-padded frames are excluded from the KL.
        max_frames = max(np.asarray(t).shape[0] for t in teacher)
        vocab = np.asarray(teacher[0]).shape[-1]
        padded = torch.zeros(len(teacher), max_frames, vocab, dtype=torch.float32)
        mask = torch.zeros(len(teacher), max_frames, dtype=torch.float32)
        for i, t in enumerate(teacher):
            t = torch.as_tensor(np.asarray(t), dtype=torch.float32)
            n = t.shape[0]
            padded[i, :n] = t
            mask[i, :n] = 1.0
        batch["teacher_logits"] = padded
        batch["teacher_mask"] = mask
        return batch


def build_distillation_trainer(
    model,
    train_dataset,
    collator,
    training_args,
    *,
    temperature: float = 2.0,
    lambda_ctc: float = 0.5,
    lambda_kd: float = 0.5,
    loss_recorder=None,
):
    """Create a stock ``Trainer`` subclass with the combined CTC+KD loss."""
    from transformers import Trainer

    class DistillationTrainer(Trainer):
        """Trainer whose loss is ``lambda_ctc*CTC + lambda_kd*T^2*KL``."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.temperature = float(temperature)
            self.lambda_ctc = float(lambda_ctc)
            self.lambda_kd = float(lambda_kd)
            self.loss_components: list[dict] = []

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # These columns are not accepted by the HF model's forward. Pop them
            # before the call. `remove_unused_columns=False` is what lets them
            # reach this method at all.
            teacher_logits = inputs.pop("teacher_logits")
            teacher_mask = inputs.pop("teacher_mask")
            outputs = model(**inputs)
            ctc_loss = outputs.loss
            if ctc_loss is None:
                raise RuntimeError("distillation training requires labels for the CTC loss")
            total, parts = distillation_loss(
                outputs.logits,
                teacher_logits,
                teacher_mask,
                ctc_loss=ctc_loss,
                temperature=self.temperature,
                lambda_ctc=self.lambda_ctc,
                lambda_kd=self.lambda_kd,
            )
            # Record the components so a failed run can be diagnosed (e.g. KD
            # dominating CTC). Keep them small: one dict per micro-batch.
            parts["total"] = float(total.detach())
            self.loss_components.append(parts)
            return (total, outputs) if return_outputs else total

        def loss_summary(self) -> dict:
            if not self.loss_components:
                return {}
            keys = ("total", "ctc", "kd", "kd_mean_per_frame", "valid_frames")
            return {
                key: float(np.mean([c[key] for c in self.loss_components]))
                for key in keys
                if key in self.loss_components[0]
            } | {"n_micro_batches": len(self.loss_components)}

    callbacks = [loss_recorder] if loss_recorder is not None else None
    return DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )


def teacher_logits_cache_path(cache_dir: str | Path, *, teacher_id: str, split: str,
                              subset_size: int, seed: int) -> Path:
    """Deterministic cache path for precomputed teacher logits."""
    safe = teacher_id.replace("/", "__")
    return Path(cache_dir) / f"teacher_logits_{safe}_{split}_n{subset_size}_s{seed}.pt"
