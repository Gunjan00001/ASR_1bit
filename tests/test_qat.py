"""QAT tests (Phase 11) — tiny synthetic models only.

The 593M checkpoint is never loaded here. These tests verify the required QAT
behaviors before any expensive training:

- feature extractor freezing (and that it truly receives no gradient)
- selected layers are BitLinear; non-selected layers stay FP
- BitLinear FP master weights receive gradients and are updated by the optimizer
- master weights stay FP (never overwritten by binary values)
- forward/backward produces finite CTC loss; a tiny overfit reduces loss
- checkpoint save/load round-trip (FP master weights; not packed)
- CTC collator padding behavior
- config validation (leakage guard)
- gradient_checkpointing is taken from config, never hard-coded
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from onebit_asr.data.collator import DataCollatorCTCWithPadding
from onebit_asr.data.dataset import load_train_subset
from onebit_asr.models.conformer_utils import select_by_category
from onebit_asr.training.qat import (
    CHECKPOINT_FORMAT,
    bitlinear_grad_stats,
    build_training_arguments,
    build_trainer,
    extrapolate_training_time,
    freeze_feature_extractor,
    freeze_report,
    gpu_memory_stats,
    load_qat_checkpoint,
    make_loss_recorder,
    master_weight_delta,
    master_weight_snapshot,
    probe_gradient_flow,
    save_qat_checkpoint,
    setup_qat_model,
    vram_headroom,
)
from onebit_asr.quantization.bitlinear import BitLinear

ATTN_FFN = {
    "attention_q": True, "attention_k": True, "attention_v": True,
    "attention_output": True, "ffn_in": True, "ffn_out": True,
    "feature_projection": False, "ctc_head": False,
}


# --- Tiny model fixture (real HF class, tiny dims) --------------------------

@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Wav2Vec2ConformerConfig, Wav2Vec2ConformerForCTC

    config = Wav2Vec2ConformerConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=[8, 8],
        conv_stride=[2, 2],
        conv_kernel=[3, 3],
        num_conv_pos_embedding_groups=2,
        feat_extract_norm="layer",
        pad_token_id=0,
        vocab_size=10,
        ctc_loss_reduction="mean",
    )
    torch.manual_seed(0)
    model = Wav2Vec2ConformerForCTC(config)
    model.eval()
    return model


def _fresh_qat_model(tiny_model):
    model = copy.deepcopy(tiny_model)
    model, _report = setup_qat_model(model, ATTN_FFN)
    freeze_feature_extractor(model)
    return model


def _batch(seq_len=400, n_labels=3, token_ids=(1, 2, 3)):
    input_values = torch.randn(1, seq_len)
    labels = torch.tensor([list(token_ids[:n_labels])], dtype=torch.long)
    return input_values, labels


# --- Tests ------------------------------------------------------------------

def test_freeze_marks_feature_extractor_frozen_and_rest_trainable(tiny_model):
    model = _fresh_qat_model(tiny_model)
    fe = model.get_submodule("wav2vec2_conformer.feature_extractor")
    assert all(not p.requires_grad for p in fe.parameters())
    # BitLinear masters + lm_head trainable
    assert model.lm_head.weight.requires_grad
    assert model.wav2vec2_conformer.feature_projection.projection.weight.requires_grad
    assert model.wav2vec2_conformer.encoder.layers[0].self_attn.linear_q.weight.requires_grad


def test_freeze_report_counts_by_group(tiny_model):
    model = _fresh_qat_model(tiny_model)
    report = freeze_report(model)
    assert report["feature_extractor_frozen"] is True
    assert report["frozen_params"] > 0
    assert report["trainable_params"] > 0
    assert report["frozen_params"] + report["trainable_params"] == sum(
        p.numel() for p in model.parameters()
    )
    groups = report["trainable_by_group"]
    assert groups["feature_extractor_frozen"]["params_frozen"] > 0
    assert groups["encoder_bitlinear"]["params_trainable"] > 0


def test_selected_layers_are_bitlinear_and_others_remain_fp(tiny_model):
    model = _fresh_qat_model(tiny_model)
    before = select_by_category(model, ATTN_FFN)
    assert len(before) == 2 * 8  # 2 layers x (4 attn + 2 + 2 ffn)
    assert all(isinstance(m, BitLinear) for _n, m, _c in before)
    assert type(model.lm_head) is nn.Linear
    assert type(model.wav2vec2_conformer.feature_projection.projection) is nn.Linear
    # Conformer layer norms are named ffn1_layer_norm / final_layer_norm etc.
    assert type(model.wav2vec2_conformer.encoder.layers[0].ffn1_layer_norm) is nn.LayerNorm


def test_bitlinear_master_weights_receive_gradients(tiny_model):
    model = _fresh_qat_model(tiny_model)
    model.train()
    input_values, labels = _batch()
    out = model(input_values=input_values, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    stats = bitlinear_grad_stats(model)
    assert stats["total_bitlinear"] == 16
    assert stats["with_nonzero_grad"] == 16
    assert stats["total_grad_norm"] > 0


def test_optimizer_updates_master_weights_and_they_stay_fp(tiny_model):
    model = _fresh_qat_model(tiny_model)
    layer = model.wav2vec2_conformer.encoder.layers[0].self_attn.linear_q
    before = layer.weight.detach().clone()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
    model.train()
    input_values, labels = _batch()
    out = model(input_values=input_values, labels=labels)
    out.loss.backward()
    opt.step()
    after = layer.weight.detach()
    assert not torch.allclose(before, after)
    # Master weights are genuinely FP, not ±1.
    assert not set(after.flatten().tolist()) <= {-1.0, 1.0}


def test_frozen_feature_extractor_gets_no_gradient(tiny_model):
    model = _fresh_qat_model(tiny_model)
    model.train()
    fe = model.get_submodule("wav2vec2_conformer.feature_extractor")
    input_values, labels = _batch()
    out = model(input_values=input_values, labels=labels)
    out.loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in fe.parameters())


def test_forward_backward_produces_finite_loss(tiny_model):
    model = _fresh_qat_model(tiny_model)
    model.train()
    input_values, labels = _batch()
    out = model(input_values=input_values, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_loss_decreases_on_tiny_overfit(tiny_model):
    model = _fresh_qat_model(tiny_model)
    model.train()
    input_values, labels = _batch()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    def loss_now():
        return model(input_values=input_values, labels=labels).loss

    with torch.no_grad():
        initial = float(loss_now())
    for _ in range(20):
        opt.zero_grad()
        loss = loss_now()
        loss.backward()
        opt.step()
    final = float(loss_now().detach())
    assert final < initial, f"loss did not decrease: {initial:.4f} -> {final:.4f}"


def test_checkpoint_save_load_roundtrip(tiny_model, tmp_path):
    model = _fresh_qat_model(tiny_model)
    model.eval()
    input_values, labels = _batch()
    with torch.no_grad():
        expected = model(input_values=input_values).logits

    manifest = save_qat_checkpoint(model, tmp_path)
    assert manifest["is_1bit_packed"] is False
    assert manifest["format"] == CHECKPOINT_FORMAT
    assert (tmp_path / "checkpoint_manifest.json").is_file()

    from transformers import Wav2Vec2ConformerForCTC

    reloaded, _report = load_qat_checkpoint(tmp_path, Wav2Vec2ConformerForCTC, ATTN_FFN)
    reloaded.eval()
    with torch.no_grad():
        got = reloaded(input_values=input_values).logits
    assert torch.allclose(expected, got, atol=1e-5)


def test_collator_pads_inputs_and_labels(tiny_model):
    class _Tok:
        pad_token_id = 0
        pad_token = "<pad>"

        def pad(self, features, padding=True, return_tensors="pt"):
            import numpy as np

            max_len = max(len(f["input_ids"]) for f in features)
            ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in features]
            mask = [[1] * len(f["input_ids"]) + [0] * (max_len - len(f["input_ids"]))
                    for f in features]
            torch_ = __import__("torch")
            return {"input_ids": torch_.tensor(ids), "attention_mask": torch_.tensor(mask)}

    class _Proc:
        tokenizer = _Tok()

        def pad(self, features, padding=True, return_tensors="pt"):
            import numpy as np
            import torch as _t

            max_len = max(len(f["input_values"]) for f in features)
            vals = [np.pad(f["input_values"], (0, max_len - len(f["input_values"])))
                    for f in features]
            return {"input_values": _t.tensor(np.stack(vals)),
                    "attention_mask": _t.tensor(
                        [[1] * len(f["input_values"]) + [0] * (max_len - len(f["input_values"]))
                         for f in features])}

    features = [
        {"input_values": [0.1, 0.2, 0.3], "labels": [4, 5]},
        {"input_values": [0.4], "labels": [6]},
    ]
    batch = DataCollatorCTCWithPadding(processor=_Proc())(features)
    assert batch["input_values"].shape == (2, 3)
    assert batch["labels"].shape == (2, 2)
    assert batch["labels"][1].tolist() == [6, -100]
    assert batch["attention_mask"].shape == (2, 3)


def test_leakage_guard_rejects_validation_split():
    with pytest.raises(ValueError, match="Refusing to train"):
        load_train_subset(subset_size=1, seed=0, split="validation")
    with pytest.raises(ValueError, match="Refusing to train"):
        load_train_subset(subset_size=1, seed=0, split="test")


def test_gradient_checkpointing_flag_is_respected(tmp_path):
    base = {
        "batch_size": 1, "gradient_accumulation_steps": 1, "learning_rate": 2e-5,
        "epochs": 1, "logging_steps": 1, "seed": 0,
    }
    off = build_training_arguments({**base, "gradient_checkpointing": False}, str(tmp_path))
    on = build_training_arguments({**base, "gradient_checkpointing": True}, str(tmp_path))
    default = build_training_arguments(base, str(tmp_path))
    assert off.gradient_checkpointing is False
    assert on.gradient_checkpointing is True
    assert default.gradient_checkpointing is False  # explicit default, not hard-coded True


def test_probe_gradient_flow_reports_bitlinear_grads(tiny_model):
    model = _fresh_qat_model(tiny_model)
    input_values, labels = _batch()
    probe = probe_gradient_flow(model, {"input_values": input_values, "labels": labels})
    assert probe["loss_is_finite"] is True
    assert probe["total_bitlinear"] == 16
    assert probe["with_nonzero_grad"] == 16
    assert probe["frozen_feature_extractor_grad_is_zero"] is True
    # Probe zeroes gradients afterwards (does not leave state behind).
    assert all(m.weight.grad is None for m in model.modules() if isinstance(m, BitLinear))


def test_master_weight_delta_detects_update_and_fpness(tiny_model):
    model = _fresh_qat_model(tiny_model)
    before = master_weight_snapshot(model)
    assert before["numel"] > 0
    assert before["fraction_exactly_pm1"] < 0.5

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.train()
    input_values, labels = _batch()
    out = model(input_values=input_values, labels=labels)
    out.loss.backward()
    opt.step()

    after = master_weight_snapshot(model)
    delta = master_weight_delta(before, after)
    assert delta["digest_changed"] is True
    assert delta["remain_fp_not_binary"] is True
    assert abs(delta["abs_sum_delta"]) > 0


def test_extrapolate_training_time_matches_manual_math():
    # 1000 samples x 3 epochs, batch 4, accum 4 -> 750 micro-batches, 188 steps
    out = extrapolate_training_time(2.0, subset_size=1000, batch_size=4,
                                    gradient_accumulation_steps=4, epochs=3)
    assert out["micro_batches_total"] == 750
    assert out["optimizer_steps_total"] == 188  # ceil(750 / 4)
    assert abs(out["estimated_seconds"] - 376.0) < 1e-9


def test_extrapolate_training_time_rejects_nonpositive_step_time():
    with pytest.raises(ValueError, match="must be positive"):
        extrapolate_training_time(0.0, subset_size=10, batch_size=1,
                                  gradient_accumulation_steps=1, epochs=1)


def test_vram_headroom_pass_and_fail():
    total = 24 * 1000**3
    ok = vram_headroom(int(total * 0.80), total)          # 20% free
    assert ok["applicable"] is True and ok["passes"] is True
    bad = vram_headroom(int(total * 0.95), total)         # 5% free
    assert bad["passes"] is False


def test_vram_headroom_not_applicable_on_cpu():
    out = vram_headroom(None, None)
    assert out["applicable"] is False and out["passes"] is None


def test_gpu_memory_stats_on_cpu_is_labelled():
    stats = gpu_memory_stats()
    if not torch.cuda.is_available():
        assert stats["cuda"] is False
        assert stats["device"] == "cpu"
        assert stats["peak_allocated_bytes"] is None


def test_build_trainer_returns_stock_trainer(tiny_model, tmp_path):
    from transformers import Trainer

    model = _fresh_qat_model(tiny_model)
    args = build_training_arguments(
        {"batch_size": 1, "learning_rate": 2e-5, "epochs": 1, "seed": 0,
         "gradient_checkpointing": False}, str(tmp_path))
    from datasets import Dataset

    ds = Dataset.from_dict({"input_values": [[0.0] * 400], "labels": [[1, 2]]})

    class _Proc:
        tokenizer = type("T", (), {"pad_token_id": 0})()

        def pad(self, features, padding=True, return_tensors="pt"):
            import numpy as np
            import torch as _t
            return {"input_values": _t.tensor(np.stack(
                [np.asarray(f["input_values"]) for f in features]))}

    trainer = build_trainer(model, ds, DataCollatorCTCWithPadding(processor=_Proc()), args,
                            make_loss_recorder())
    assert isinstance(trainer, Trainer)
    assert trainer.model is model
