import json
from pathlib import Path

from onebit_asr.config import load_config

ROOT = Path(__file__).resolve().parent.parent
PLAN = json.loads((ROOT / "results/short_oversample_plan.json").read_text())
CONFIG = load_config(ROOT / "configs/attn_2bit_A_short_oversample.yaml")


def test_plan_composition_and_weights():
    assert len(PLAN["samples"]) == 2000
    assert PLAN["n_short"] == 500
    assert PLAN["n_long"] == 1500
    assert PLAN["unique_short_rows"] == 365
    assert PLAN["short_repeat_count"] == 135
    assert sum(1 for sample in PLAN["samples"] if sample["short"]) == 500
    assert PLAN["target_short_fraction"] == 0.25
    assert PLAN["target_short_word_share"] < 0.07
    assert PLAN["effective_sampling_weights"]["short"] > 6.0
    assert PLAN["leakage_check"]["dev_256_overlap"] == []
    assert PLAN["leakage_check"]["split"] == "train.100"
    assert PLAN["leakage_check"]["test_used"] is False


def test_config_preserves_strategy_a_policy_and_init():
    assert CONFIG["qat"]["init_checkpoint"] == "checkpoints/attn_2bit_on_2bitffn_A_full"
    assert CONFIG["quantization"]["bits"] == 2
    assert CONFIG["quantization"]["scale"] == "per_tensor_mean_abs"
    assert len(CONFIG["quantization"]["exclude_layers"]) == 77
    assert sum("self_attn" in name for name in CONFIG["quantization"]["exclude_layers"]) == 60
    assert sum("ffn" in name for name in CONFIG["quantization"]["exclude_layers"]) == 17
    assert CONFIG["qat"]["pilot"]["subset_size"] == 2000
    assert CONFIG["qat"]["pilot"]["max_steps"] == 500
    assert CONFIG["qat"]["pilot"]["eval_utterances"] == 256
    assert CONFIG["qat"]["pilot"]["batch_size"] == 4
    assert CONFIG["qat"]["pilot"]["gradient_accumulation_steps"] == 4
    assert CONFIG["qat"]["training"]["learning_rate"] == 2e-05
    assert CONFIG["qat"]["training"]["seed"] == 42
