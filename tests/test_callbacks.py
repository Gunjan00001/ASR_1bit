"""Dev-WER callback tests (validation-based checkpoint selection).

Uses a tiny nn.Linear model and a stub eval function; no checkpoint, dataset,
GPU, or Trainer is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from onebit_asr.training.callbacks import DevWERCallback


class _State:
    def __init__(self, step):
        self.global_step = step


def test_callback_selects_and_restores_best_state():
    model = nn.Linear(1, 1, bias=False)
    calls = {"n": 0}
    wer_sequence = [0.30, 0.20, 0.25]

    def eval_fn(_model):
        value = wer_sequence[calls["n"]]
        calls["n"] += 1
        return {"wer": value, "cer": value / 2}

    cb = DevWERCallback(eval_fn, eval_every_steps=1, verbose=False)
    for step, value in enumerate([1.0, 2.0, 3.0], start=1):
        with torch.no_grad():
            model.weight.fill_(value)
        cb.on_step_end(None, _State(step), None, model=model)

    assert [h["wer"] for h in cb.history] == [0.30, 0.20, 0.25]
    assert cb.best_wer == 0.20
    assert cb.best_step == 2
    assert cb.best_state is not None

    # Training continues and changes the weights again; on_train_end restores
    # the step-2 state.
    with torch.no_grad():
        model.weight.fill_(99.0)
    cb.on_train_end(None, _State(3), None, model=model)
    assert cb.restored_best is True
    assert torch.allclose(model.weight, torch.tensor([[2.0]]))

    summary = cb.summary()
    assert summary["best_dev_wer"] == 0.20
    assert summary["best_step"] == 2
    assert summary["final_dev_wer"] == 0.25
    assert summary["restored_best"] is True


def test_callback_skips_duplicate_step_and_non_cadence():
    model = nn.Linear(1, 1, bias=False)
    calls = {"n": 0}

    def eval_fn(_model):
        calls["n"] += 1
        return {"wer": 0.5, "cer": 0.2}

    cb = DevWERCallback(eval_fn, eval_every_steps=2, verbose=False)
    cb.on_step_end(None, _State(1), None, model=model)   # not cadence
    cb.on_step_end(None, _State(2), None, model=model)   # eval
    cb.on_step_end(None, _State(2), None, model=model)   # duplicate step
    cb.on_step_end(None, _State(4), None, model=model)   # eval
    assert calls["n"] == 2
    assert [h["step"] for h in cb.history] == [2, 4]


def test_callback_rejects_nonpositive_cadence():
    try:
        DevWERCallback(lambda _m: {"wer": 0.0, "cer": 0.0}, eval_every_steps=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
