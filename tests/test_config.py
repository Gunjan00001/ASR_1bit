"""Config ``extends`` composition tests.

Regression guard for the provenance bug where ``_config_path`` recorded the
root of the ``extends`` chain (e.g. ``baseline.yaml``) instead of the config
that was actually requested (e.g. ``qat_attn.yaml``). Reproducibility reports
(training / packing size reports) rely on this field.
"""

from pathlib import Path

from onebit_asr.config import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_config_path_is_the_requested_leaf_not_the_extends_root():
    cfg = load_config(CONFIGS / "qat_attn.yaml")
    # qat_attn -> qat -> binary -> baseline; the leaf must win.
    assert Path(cfg["_config_path"]).name == "qat_attn.yaml"


def test_extends_composition_still_merges_parent_keys():
    cfg = load_config(CONFIGS / "qat_attn.yaml")
    # Leaf override.
    assert cfg["layers"]["attention_q"] is True
    assert cfg["layers"]["ffn_in"] is False
    # Inherited from qat.yaml / binary.yaml / baseline.yaml.
    assert cfg["qat"]["training"]["subset_size"] == 10000
    assert "quantization" in cfg
