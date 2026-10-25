"""Latency/benchmark wrapper tests (Phase 17). Tiny synthetic model only."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from onebit_asr.evaluation.benchmark import benchmark_and_evaluate, comparison_table
from onebit_asr.evaluation.latency import benchmark_model


class _TinyModel(nn.Module):
    def __init__(self, n_classes=4, steps=3):
        super().__init__()
        self.linear = nn.Linear(2, n_classes)
        self.steps = steps

    def forward(self, input_values):
        batch = input_values.shape[0]
        logits = torch.zeros(batch, self.steps, self.linear.out_features)
        logits[:, :, 1] = 1.0
        return SimpleNamespace(logits=logits)


class _Inputs:
    def __init__(self, input_values):
        self.input_values = input_values


class _StubProcessor:
    pad_token_id = 0

    def __call__(self, audio, sampling_rate=16_000, return_tensors="pt", padding=True):
        return _Inputs(torch.zeros(1, 8))

    def batch_decode(self, ids):
        return ["a"]


def _samples(n=3):
    return [{"id": str(i), "audio": [0.0] * 8, "text": "a"} for i in range(n)]


def test_benchmark_model_reports_latency_and_memory():
    model = _TinyModel()
    result = benchmark_model(model, _StubProcessor(), _samples(3),
                             warmup=1, verbose=False)
    assert result["n_utterances"] == 3
    assert result["warmup_utterances"] == 1
    assert result["rtf"] >= 0
    assert result["latency_s"]["p50"] >= 0
    assert result["throughput_utt_per_s"] > 0
    assert "peak_rss_mb" in result["memory"]
    assert result["memory"]["cuda"] is False
    assert result["device"] == "cpu"


def test_benchmark_and_evaluate_and_table():
    report = benchmark_and_evaluate(
        _TinyModel(), _StubProcessor(), _samples(2), label="tiny", warmup=1,
        verbose=False,
    )
    assert report["accuracy"]["wer"] == 0.0
    assert report["performance"]["n_utterances"] == 2
    table = comparison_table([report])
    assert "| tiny |" in table
    assert "WER" in table
