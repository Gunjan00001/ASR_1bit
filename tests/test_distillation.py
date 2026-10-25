"""Distillation tests (Phase 13).

Tiny synthetic tensors only: loss math, mask handling, collator padding, and
dataset/teacher alignment. The 593M checkpoint is never loaded here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from onebit_asr.training.distillation import (
    DistillationCollator,
    DistillationDataset,
    distillation_loss,
    teacher_logits_cache_path,
)


def test_kd_is_zero_when_student_equals_teacher():
    logits = torch.randn(2, 5, 7)
    total, parts = distillation_loss(
        logits, logits.clone(), None, ctc_loss=torch.tensor(3.0),
        temperature=2.0, lambda_ctc=0.0, lambda_kd=1.0,
    )
    assert parts["kd"] == pytest.approx(0.0, abs=1e-5)
    assert total == pytest.approx(0.0, abs=1e-5)


def test_kd_is_positive_when_distributions_differ():
    student = torch.randn(2, 5, 7)
    teacher = torch.randn(2, 5, 7)
    _total, parts = distillation_loss(
        student, teacher, None, ctc_loss=torch.tensor(1.0),
        temperature=2.0, lambda_ctc=0.0, lambda_kd=1.0,
    )
    assert parts["kd"] > 0


def test_lambda_weights_are_applied_exactly():
    student = torch.randn(1, 3, 5)
    teacher = torch.randn(1, 3, 5)
    ctc = torch.tensor(2.0)
    total_ctc, _ = distillation_loss(
        student, teacher, None, ctc_loss=ctc, temperature=1.0,
        lambda_ctc=1.0, lambda_kd=0.0,
    )
    assert total_ctc.item() == pytest.approx(2.0)

    _total, parts = distillation_loss(
        student, teacher, None, ctc_loss=ctc, temperature=1.0,
        lambda_ctc=0.0, lambda_kd=0.5,
    )
    total_kd, _ = distillation_loss(
        student, teacher, None, ctc_loss=ctc, temperature=1.0,
        lambda_ctc=0.0, lambda_kd=1.0,
    )
    assert total_kd.item() == pytest.approx(parts["kd"], rel=1e-5)
    assert parts["kd"] == pytest.approx(total_kd.item())


def test_teacher_mask_excludes_padded_frames():
    student = torch.randn(1, 5, 4)
    teacher = torch.randn(1, 5, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
    _t1, masked = distillation_loss(
        student, teacher, mask, ctc_loss=torch.tensor(0.0), temperature=1.0,
        lambda_ctc=0.0, lambda_kd=1.0,
    )
    _t2, cropped = distillation_loss(
        student[:, :2], teacher[:, :2], None, ctc_loss=torch.tensor(0.0),
        temperature=1.0, lambda_ctc=0.0, lambda_kd=1.0,
    )
    assert masked["kd"] == pytest.approx(cropped["kd"], rel=1e-5)
    assert masked["valid_frames"] == 2.0


def test_temperature_zero_rejected():
    with pytest.raises(ValueError, match="temperature"):
        distillation_loss(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), None,
            ctc_loss=torch.tensor(1.0), temperature=0.0,
        )


def test_vocab_mismatch_rejected():
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        distillation_loss(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 4), None,
            ctc_loss=torch.tensor(1.0),
        )


# --- Collator / dataset -----------------------------------------------------

class _Tok:
    pad_token_id = 0

    def pad(self, features, padding=True, return_tensors="pt"):
        max_len = max(len(f["input_ids"]) for f in features)
        ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in features]
        mask = [[1] * len(f["input_ids"]) + [0] * (max_len - len(f["input_ids"]))
                for f in features]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class _Proc:
    tokenizer = _Tok()

    def pad(self, features, padding=True, return_tensors="pt"):
        max_len = max(len(f["input_values"]) for f in features)
        vals = [np.pad(np.asarray(f["input_values"], dtype=np.float32),
                       (0, max_len - len(f["input_values"]))) for f in features]
        return {"input_values": torch.tensor(np.stack(vals)),
                "attention_mask": torch.tensor(
                    [[1] * len(f["input_values"]) + [0] * (max_len - len(f["input_values"]))
                     for f in features])}


def test_collator_pads_teacher_logits_and_mask():
    features = [
        {"input_values": [0.1, 0.2, 0.3], "labels": [4, 5],
         "teacher_logits": np.zeros((2, 7), dtype=np.float16)},
        {"input_values": [0.4], "labels": [6],
         "teacher_logits": np.ones((4, 7), dtype=np.float16)},
    ]
    batch = DistillationCollator(processor=_Proc())(features)
    assert batch["input_values"].shape == (2, 3)
    assert batch["labels"].shape == (2, 2)
    assert batch["teacher_logits"].shape == (2, 4, 7)
    assert batch["teacher_logits"].dtype == torch.float32
    assert batch["teacher_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    # Teacher columns are removed from the feature dicts passed to the model.
    assert all("teacher_logits" not in f for f in features)


def test_distillation_dataset_requires_alignment():
    class _Base:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {"input_values": [0.0], "labels": [1]}

    with pytest.raises(ValueError, match="length mismatch"):
        DistillationDataset(_Base(), [torch.zeros(1, 3)])

    ds = DistillationDataset(_Base(), [torch.zeros(1, 3), torch.zeros(2, 3)])
    assert ds[1]["teacher_logits"].shape == (2, 3)
    assert ds[0]["labels"] == [1]


def test_teacher_cache_path_is_deterministic_and_distinct():
    a = teacher_logits_cache_path("cache", teacher_id="facebook/foo",
                                  split="train.100", subset_size=10000, seed=42)
    b = teacher_logits_cache_path("cache", teacher_id="facebook/foo",
                                  split="train.100", subset_size=10000, seed=42)
    c = teacher_logits_cache_path("cache", teacher_id="facebook/foo",
                                  split="train.100", subset_size=1000, seed=42)
    assert a == b
    assert a != c
    assert "n10000" in a.name
