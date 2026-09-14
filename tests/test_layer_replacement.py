"""Layer replacement tests on a small synthetic model (Phase 8 / Phase 7).

The synthetic model mirrors the naming of the real HF Wav2Vec2-Conformer
checkpoint discovered in Phase 1, but with tiny dimensions. The 593M
checkpoint is never loaded here.
"""

import pytest
import torch
import torch.nn as nn

from onebit_asr.models.conformer_utils import (
    categorize,
    get_parent_module,
    select_by_category,
    selected_categories,
)
from onebit_asr.models.replace_layers import (
    replace_linears,
    report_replacement,
    summarize_by_category,
)
from onebit_asr.quantization.bitlinear import BitLinear

D = 8


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(D, D)
        self.linear_k = nn.Linear(D, D)
        self.linear_v = nn.Linear(D, D)
        self.linear_out = nn.Linear(D, D)


class _FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate_dense = nn.Linear(D, 2 * D)
        self.output_dense = nn.Linear(2 * D, D)


class _ConformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.ffn1 = _FFN()
        self.ffn2 = _FFN()
        self.layer_norm = nn.LayerNorm(D)
        self.depthwise_conv = nn.Conv1d(D, D, kernel_size=3, padding=1)


class _FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, D)
        self.layer_norm = nn.LayerNorm(D)


class _SyntheticConformer(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.feature_projection = _FeatureProjection()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_ConformerLayer() for _ in range(n_layers)])
        self.lm_head = nn.Linear(D, 5)  # CTC head

    def forward(self, x):
        x = self.feature_projection.projection(x)
        for layer in self.encoder.layers:
            x = x + layer.self_attn.linear_out(layer.self_attn.linear_q(x))
            x = layer.ffn1.output_dense(torch.relu(layer.ffn1.intermediate_dense(x)))
            x = layer.ffn2.output_dense(torch.relu(layer.ffn2.intermediate_dense(x)))
        return self.lm_head(x)


ALL_LAYERS = {
    "attention_q": True, "attention_k": True, "attention_v": True,
    "attention_output": True, "ffn_in": True, "ffn_out": True,
    "feature_projection": False, "ctc_head": False,
}


def test_taxonomy_matches_hf_naming():
    assert categorize("wav2vec2_conformer.encoder.layers.0.self_attn.linear_q") == "attention_q"
    assert categorize("wav2vec2_conformer.encoder.layers.0.self_attn.linear_k") == "attention_k"
    assert categorize("wav2vec2_conformer.encoder.layers.0.self_attn.linear_v") == "attention_v"
    assert categorize("wav2vec2_conformer.encoder.layers.0.self_attn.linear_out") == "attention_output"
    assert categorize("wav2vec2_conformer.encoder.layers.0.ffn1.intermediate_dense") == "ffn_in"
    assert categorize("wav2vec2_conformer.encoder.layers.0.ffn1.output_dense") == "ffn_out"
    assert categorize("wav2vec2_conformer.feature_projection.projection") == "feature_projection"
    assert categorize("lm_head") == "ctc_head"


def test_unknown_selection_key_raises():
    with pytest.raises(ValueError, match="Unknown layer-selection key"):
        selected_categories({"ffn": True})  # old-style key must not silently pass


def test_select_by_category_counts():
    model = _SyntheticConformer(n_layers=2)
    chosen = select_by_category(model, ALL_LAYERS)
    # 2 layers x (4 attn + 2 ffn1 + 2 ffn2) = 16
    assert len(chosen) == 16
    cats = {c for _n, _m, c in chosen}
    assert cats == {"attention_q", "attention_k", "attention_v",
                    "attention_output", "ffn_in", "ffn_out"}


def test_replacement_swaps_only_selected_categories():
    model = _SyntheticConformer(n_layers=2)
    report = replace_linears(model, ALL_LAYERS)

    assert report["n_replaced"] == 16
    # feature projection + lm_head = 2 retained
    assert report["n_retained"] == 2
    retained_cats = {e["category"] for e in report["retained"]}
    assert retained_cats == {"feature_projection", "ctc_head"}

    # Structural assertions on the mutated model.
    layer0 = model.encoder.layers[0]
    assert isinstance(layer0.self_attn.linear_q, BitLinear)
    assert isinstance(layer0.ffn1.intermediate_dense, BitLinear)
    assert isinstance(layer0.ffn2.output_dense, BitLinear)
    assert type(model.feature_projection.projection) is nn.Linear
    assert type(model.lm_head) is nn.Linear
    assert type(layer0.layer_norm) is nn.LayerNorm
    assert type(layer0.depthwise_conv) is nn.Conv1d


def test_replacement_preserves_weights_numerically():
    model = _SyntheticConformer(n_layers=1)
    before = model.encoder.layers[0].self_attn.linear_q.weight.detach().clone()
    before_bias = model.encoder.layers[0].self_attn.linear_q.bias.detach().clone()

    replace_linears(model, ALL_LAYERS)

    after = model.encoder.layers[0].self_attn.linear_q
    assert torch.allclose(after.weight.detach(), before)
    assert torch.allclose(after.bias.detach(), before_bias)


def test_forward_still_runs_after_replacement():
    model = _SyntheticConformer(n_layers=2)
    x = torch.randn(3, 4)
    with torch.no_grad():
        before = model(x)
    replace_linears(model, ALL_LAYERS)
    with torch.no_grad():
        after = model(x)
    assert after.shape == before.shape
    assert torch.isfinite(after).all()


def test_selective_replacement_attention_only():
    model = _SyntheticConformer(n_layers=2)
    report = replace_linears(model, {
        "attention_q": True, "attention_k": True, "attention_v": True,
        "attention_output": True, "ffn_in": False, "ffn_out": False,
        "feature_projection": False, "ctc_head": False,
    })
    assert report["n_replaced"] == 8
    assert isinstance(model.encoder.layers[1].self_attn.linear_v, BitLinear)
    assert type(model.encoder.layers[1].ffn1.intermediate_dense) is nn.Linear


def test_selective_replacement_ffn_only():
    model = _SyntheticConformer(n_layers=1)
    report = replace_linears(model, {
        "attention_q": False, "attention_k": False, "attention_v": False,
        "attention_output": False, "ffn_in": True, "ffn_out": True,
        "feature_projection": False, "ctc_head": False,
    })
    assert report["n_replaced"] == 4
    assert isinstance(model.encoder.layers[0].ffn1.intermediate_dense, BitLinear)
    assert isinstance(model.encoder.layers[0].ffn2.output_dense, BitLinear)
    assert type(model.encoder.layers[0].self_attn.linear_q) is nn.Linear


def test_replacement_is_idempotent():
    model = _SyntheticConformer(n_layers=1)
    replace_linears(model, ALL_LAYERS)
    report = replace_linears(model, ALL_LAYERS)
    assert report["n_replaced"] == 8
    assert all(e.get("note") == "already replaced" for e in report["replaced"])


def test_report_and_summary_render():
    model = _SyntheticConformer(n_layers=1)
    report = replace_linears(model, ALL_LAYERS)
    text = report_replacement(report)
    assert "replaced" in text and "retained" in text
    assert "attention_q" in text and "ctc_head" in text
    summary = summarize_by_category(report["replaced"])
    assert summary["attention_q"]["count"] == 1


def test_get_parent_module_returns_owner():
    model = _SyntheticConformer(n_layers=1)
    parent, attr = get_parent_module(model, "encoder.layers.0.self_attn.linear_q")
    assert attr == "linear_q"
    assert parent is model.encoder.layers[0].self_attn
