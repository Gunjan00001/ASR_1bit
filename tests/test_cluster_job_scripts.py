"""Static regression checks for the PBS full-run script.

The shell script embeds Python. A typo there aborts a 2-hour job before training
(observed: ``torch.device_count()`` does not exist). These tests catch that class
of error without needing a GPU.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL_PBS = ROOT / "cluster" / "qat_ffn_full.pbs"


def test_full_pbs_cuda_preflight_uses_cuda_device_api():
    text = FULL_PBS.read_text()
    assert "torch.cuda.device_count()" in text
    assert "torch.device_count()" not in text
    assert "assert torch.cuda.is_available()" in text


def test_full_pbs_uses_unique_full_outputs_and_full_protocol():
    text = FULL_PBS.read_text()
    assert "qat_ffn_control_full" in text
    assert "qat_ffn_mixed_full" in text
    assert "eval_utterances: 256" in text
    assert "save_strategy: \"steps\"" in text
    assert "save_steps: 250" in text
    assert "qat_ffn_mixed.yaml" in text


def test_attention_pilot_pbs_packs_and_verifies_artifact():
    text = (ROOT / "cluster" / "qat_attention_mixed_pilot.pbs").read_text()
    assert "--pilot" in text
    assert "POLICY_ID" in text
    assert "pack_qat_model.py" in text
    assert "verify_packed_artifact.py" in text
    assert "results/${POLICY_ID}_pilot.json" in text
    assert "results/${POLICY_ID}_pilot_size_report.json" in text
    assert "results/packed_verify_${POLICY_ID}_pilot.json" in text
    assert "attention_v1_cpu16_category.yaml" in text
    assert "attention_v1_cpu16_sensitive24.yaml" not in text  # selected at submit time
