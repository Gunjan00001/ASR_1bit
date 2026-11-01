from copy import deepcopy
from pathlib import Path

from onebit_asr.config import load_config

ROOT = Path(__file__).resolve().parent.parent
PER_CHANNEL = "per_output_channel_mean_abs"


def test_per_channel_config_changes_only_scale():
    control = load_config(ROOT / "configs/attn_2bit_on_2bitffn_A.yaml")
    candidate = load_config(ROOT / "configs/attn_2bit_A_per_channel.yaml")

    assert control["quantization"]["scale"] == "per_tensor_mean_abs"
    assert candidate["quantization"]["scale"] == PER_CHANNEL

    control = deepcopy(control)
    candidate = deepcopy(candidate)
    control.pop("_config_path")
    candidate.pop("_config_path")
    control["quantization"].pop("scale")
    candidate["quantization"].pop("scale")
    assert candidate == control


def test_per_channel_config_keeps_A_policy():
    cfg = load_config(ROOT / "configs/attn_2bit_A_per_channel.yaml")
    assert cfg["quantization"]["bits"] == 2
    assert cfg["qat"]["init_checkpoint"] == "checkpoints/qat_ffn_mixed_2bit_full"
    assert len(cfg["quantization"]["exclude_layers"]) == 77
    assert sum(
        1 for name in cfg["quantization"]["exclude_layers"] if "self_attn" in name
    ) == 60
    assert cfg["layers"] == {
        "attention_q": True,
        "attention_k": True,
        "attention_v": True,
        "attention_output": True,
        "ffn_in": True,
        "ffn_out": True,
        "feature_projection": False,
        "ctc_head": False,
    }
