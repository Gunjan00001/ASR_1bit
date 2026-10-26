"""Resume-path tests: latest-checkpoint discovery and a tiny Trainer restart.

The tiny Trainer test is deliberately synthetic (a one-parameter model, three
rows) and runs on CPU. It exercises the actual Hugging Face checkpoint/optimizer
resume path without touching the 593M ASR model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from onebit_asr.training.resume import find_latest_checkpoint, list_checkpoints


def _make_checkpoint(root: Path, step: int, *, with_state: bool = True) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True)
    if with_state:
        (path / "trainer_state.json").write_text('{"global_step": %d}' % step)
    return path


def test_find_latest_checkpoint_uses_numeric_step_and_skips_incomplete(tmp_path):
    _make_checkpoint(tmp_path, 500)
    _make_checkpoint(tmp_path, 1000)
    _make_checkpoint(tmp_path, 1500, with_state=False)
    (tmp_path / "checkpoint-bad").mkdir()
    (tmp_path / "model.safetensors").write_text("not a checkpoint dir")

    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-1000"
    assert [p.name for p in list_checkpoints(tmp_path)] == ["checkpoint-500", "checkpoint-1000"]


def test_find_latest_checkpoint_missing_root_returns_none(tmp_path):
    assert find_latest_checkpoint(tmp_path / "does-not-exist") is None


class _ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor([1.0, 2.0, 3.0])
        self.y = torch.tensor([2.0, 4.0, 6.0])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, y):
        return {"loss": ((self.w * x - y) ** 2).mean()}


def test_tiny_trainer_resume_continues_from_checkpoint(tmp_path):
    from transformers import Trainer, TrainingArguments

    dataset = _ToyDataset()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        max_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=1,
        logging_steps=1,
        learning_rate=0.01,
        report_to=[],
        disable_tqdm=True,
        seed=0,
    )
    trainer = Trainer(model=_ToyModel(), args=args, train_dataset=dataset)
    trainer.train()
    checkpoint = find_latest_checkpoint(tmp_path)
    assert checkpoint is not None and checkpoint.name == "checkpoint-1"
    assert trainer.state.global_step == 1

    resumed_args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        max_steps=2,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=1,
        logging_steps=1,
        learning_rate=0.01,
        report_to=[],
        disable_tqdm=True,
        seed=0,
    )
    resumed = Trainer(model=_ToyModel(), args=resumed_args, train_dataset=dataset)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2
    assert (tmp_path / "checkpoint-2").is_dir()
