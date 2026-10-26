"""Controlled 2-bit config differs from mixed FFN only in bit width."""

from __future__ import annotations

from pathlib import Path

from onebit_asr.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def test_2bit_config_only_changes_bit_width():
    base = load_config(ROOT / "configs/qat_ffn_mixed.yaml")
    cand = load_config(ROOT / "configs/qat_ffn_mixed_2bit.yaml")

    assert cand["quantization"]["bits"] == 2
    assert base["quantization"].get("bits", 1) == 1
    assert cand["layers"] == base["layers"]
    assert cand["quantization"]["exclude_layers"] == base["quantization"]["exclude_layers"]
    assert len(cand["quantization"]["exclude_layers"]) == 17
    for key in ("scale", "ste", "ste_clip"):
        assert cand["quantization"][key] == base["quantization"][key]
    assert cand["qat"]["training"] == base["qat"]["training"]
    assert cand["qat"]["pilot"] == base["qat"]["pilot"]
    assert cand["eval"] == base["eval"]
    assert cand["quantization"]["bits"] != base["quantization"].get("bits", 1)
