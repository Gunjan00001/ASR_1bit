"""Tests for the mixed-attention policy config generator."""

from __future__ import annotations

import json
from pathlib import Path

from onebit_asr.config import load_config

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "configs" / "qat_ffn_mixed.yaml"


def _screen() -> dict:
    cats = ["attention_q", "attention_k", "attention_v", "attention_output"]
    layers = []
    for layer in range(24):
        for cat in cats:
            layers.append({
                "name": f"wav2vec2_conformer.encoder.layers.{layer}.self_attn.{cat}",
                "category": cat,
                "layer": layer,
                "in_features": 1024,
                "out_features": 1024,
                "delta_wer_vs_fp32": {"attention_q": 0.03, "attention_k": 0.001,
                                     "attention_v": 0.02, "attention_output": 0.04}[cat],
                "relative_l2": {"attention_q": 0.10, "attention_k": 0.01,
                                "attention_v": 0.08, "attention_output": 0.12}[cat],
                "weight_reconstruction_error": {"attention_q": 0.8, "attention_k": 0.6,
                                                "attention_v": 0.75, "attention_output": 0.9}[cat],
            })
    groups = {}
    for cat in cats:
        rows = [r for r in layers if r["category"] == cat]
        groups[cat] = {
            "n_layers": 24,
            "mean_delta_wer": sum(r["delta_wer_vs_fp32"] for r in rows) / 24,
            "mean_relative_l2": sum(r["relative_l2"] for r in rows) / 24,
            "mean_weight_reconstruction_error": sum(
                r["weight_reconstruction_error"] for r in rows
            ) / 24,
        }
    return {"layers": layers, "group_summary": groups}


def test_category_policy_selects_least_sensitive_and_preserves_ffn_list(tmp_path):
    from scripts.make_attention_mixed_config import build_policy

    base = load_config(BASE)
    base_excluded = list(base["quantization"]["exclude_layers"])
    config, meta = build_policy(
        _screen(), base["layers"], base_excluded, base["quantization"],
        policy="category", metric="delta_wer",
    )
    assert meta["selected_category"] == "attention_k"
    assert config["layers"]["attention_k"] is True
    for cat in ("attention_q", "attention_v", "attention_output"):
        assert config["layers"][cat] is False
    assert config["layers"]["ffn_in"] is True
    assert config["layers"]["ffn_out"] is True
    assert config["quantization"]["exclude_layers"] == base_excluded
    assert meta["quantized_attention_layer_count"] == 24
    assert meta["estimated_size"]["estimated_artifact_gb"] < 1.09


def test_sensitive_policy_keeps_top_k_and_preserves_ffn_list():
    from scripts.make_attention_mixed_config import build_policy

    base = load_config(BASE)
    base_excluded = list(base["quantization"]["exclude_layers"])
    config, meta = build_policy(
        _screen(), base["layers"], base_excluded, base["quantization"],
        policy="sensitive", metric="relative_l2", keep_count=24,
    )
    assert all(config["layers"][c] is True for c in (
        "attention_q", "attention_k", "attention_v", "attention_output"
    ))
    assert len(meta["attention_kept_fp32"]) == 24
    assert all(name in config["quantization"]["exclude_layers"]
               for name in base_excluded)
    assert all(name in config["quantization"]["exclude_layers"]
               for name in meta["attention_kept_fp32"])
    assert meta["quantized_attention_layer_count"] == 72
    assert meta["estimated_size"]["estimated_artifact_gb"] < 0.9


def test_generated_config_loads_and_keeps_exclusions(tmp_path):
    from scripts.make_attention_mixed_config import build_policy

    base = load_config(BASE)
    base_excluded = list(base["quantization"]["exclude_layers"])
    config, _meta = build_policy(
        _screen(), base["layers"], base_excluded, base["quantization"],
        policy="sensitive", metric="relative_l2", keep_count=24,
    )
    config["extends"] = str(BASE)
    path = tmp_path / "policy.yaml"
    path.write_text(json.dumps(config))
    # JSON is valid YAML; load through the same extends-aware loader.
    loaded = load_config(path)
    assert loaded["quantization"]["exclude_layers"][:len(base_excluded)] == base_excluded
    assert loaded["layers"]["ffn_in"] is True
    assert loaded["layers"]["attention_q"] is True
