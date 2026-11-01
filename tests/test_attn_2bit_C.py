import json
from pathlib import Path
from onebit_asr.config import load_config
ROOT=Path(__file__).resolve().parent.parent
def test_C_policy_counts_and_ffn_preservation():
    base=load_config(ROOT/"configs/qat_ffn_mixed_2bit.yaml")
    c=load_config(ROOT/"configs/attn_2bit_C.yaml")
    policy=json.loads((ROOT/"results/attn_2bit_C_policy.json").read_text())
    assert c["quantization"]["bits"]==2
    assert c["qat"]["init_checkpoint"]=="checkpoints/qat_ffn_mixed_2bit_full"
    assert len(c["quantization"]["exclude_layers"])==79
    assert c["quantization"]["exclude_layers"][:17]==base["quantization"]["exclude_layers"]
    assert sum(1 for n in c["quantization"]["exclude_layers"] if "self_attn" in n)==62
    assert c["layers"]["attention_k"] is True and c["layers"]["ffn_in"] is True
    assert policy["n_quantized_attention"]==34 and policy["n_kept_attention_fp32"]==62
    assert len(policy["quantized_output_layers"])==10
    assert not any(".attention_k" in n or "linear_k" in n for n in c["quantization"]["exclude_layers"])
    assert sum(1 for n in policy["kept_attention_layers"] if "linear_out" in n)==14
