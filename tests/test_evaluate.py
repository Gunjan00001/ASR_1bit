"""Evaluation runner tests (Phase 11 hardening): eval-mode guarantee.

The pinned protocol is inference. A training loop leaves the model in train
mode, so ``evaluate_model`` must force eval mode for the measurement and
restore the previous mode afterwards.
"""

import torch
from torch import nn


class _TinyLogitsModel(nn.Module):
    """Returns fixed logits and records its own training flag per call."""

    def __init__(self, n_classes=4, steps=3):
        super().__init__()
        self.linear = nn.Linear(2, n_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.training_flags = []
        self.steps = steps

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    def forward(self, input_values):
        self.training_flags.append(self.training)
        # Deterministic logits regardless of dropout, so WER is stable.
        batch = input_values.shape[0]
        logits = torch.zeros(batch, self.steps, self.linear.out_features)
        logits[:, :, 1] = 1.0
        return self._Out(logits)


class _Inputs:
    def __init__(self, input_values):
        self.input_values = input_values


class _StubProcessor:
    pad_token_id = 0

    def __call__(self, audio, sampling_rate=16_000, return_tensors="pt", padding=True):
        return _Inputs(torch.zeros(1, 8))

    def batch_decode(self, ids):
        return ["a"]


def test_evaluate_model_forces_eval_mode_and_restores():
    from onebit_asr.evaluation.evaluate import evaluate_model

    model = _TinyLogitsModel()
    model.train()  # simulate state right after training
    samples = [{"id": "x", "audio": [0.0] * 8, "text": "a"}]

    evaluate_model(model, _StubProcessor(), samples, verbose=False)

    assert model.training_flags, "model.forward was never called"
    assert all(flag is False for flag in model.training_flags), "eval mode not used"
    assert model.training is True, "previous train mode was not restored"
