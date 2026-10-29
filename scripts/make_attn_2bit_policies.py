"""Generate the A/B 2-bit attention policies from the measured screen.

A: quantize the 36 least-sensitive attention projections (estimated <1.0 GB).
B: quantize the 72 least-sensitive, keeping the 24 most sensitive FP32.
Both inherit the 79 2-bit FFN layers, the fixed 17 FFN exclusions, and initialize
from the selected 2-bit FFN full checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ATTN = ("attention_q", "attention_k", "attention_v", "attention_output")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="results/layer_sensitivity_attention_2bit_on_2bit_ffn.json")
    ap.add_argument("--base-config", default="configs/qat_ffn_mixed_2bit.yaml")
    ap.add_argument("--init-checkpoint", default="checkpoints/qat_ffn_mixed_2bit_full")
    ap.add_argument("--base-artifact-bytes", type=int, default=1_131_062_436)
    args = ap.parse_args()
    screen = json.loads((ROOT / args.screen).read_text())
    base = yaml.safe_load((ROOT / args.base_config).read_text())
    base_excl = []  # resolved below through the extends chain
    # Resolve the base config (17 FFN exclusions and FFN 2-bit layers) through
    # the normal loader so the generated config cannot silently lose them.
    from onebit_asr.config import load_config
    resolved = load_config(ROOT / args.base_config)
    base_excl = list(resolved["quantization"]["exclude_layers"])
    if len(base_excl) != 17:
        raise SystemExit(f"expected 17 base exclusions, got {len(base_excl)}")
    rows = screen["layers"]
    least = sorted(rows, key=lambda r: (r["delta_wer_vs_base"], r["relative_l2"],
                                        r["weight_reconstruction_error"]))
    most = list(reversed(least))
    per_saved = 4 * 1024 * 1024 - (1024 * 1024 // 4) - 4
    for name, n_quant, kept in (("A", 36, least[36:]), ("B", 72, most[:24])):
        quantized = [r["name"] for r in least[:n_quant]]
        kept_names = [r["name"] for r in rows if r["name"] not in quantized]
        if name == "B":
            kept_names = [r["name"] for r in most[:24]]
            quantized = [r["name"] for r in rows if r["name"] not in set(kept_names)]
        config = {
            "extends": "qat_ffn_mixed_2bit.yaml",
            "compute_profile": "cloud-gpu",
            "qat": {"init_checkpoint": args.init_checkpoint},
            "layers": {k: True for k in ATTN} | {
                "ffn_in": True, "ffn_out": True,
                "feature_projection": False, "ctc_head": False,
            },
            "quantization": {
                "bits": 2,
                "exclude_layers": base_excl + kept_names,
            },
        }
        out = ROOT / "configs" / f"attn_2bit_on_2bitffn_{name}.yaml"
        out.write_text(yaml.safe_dump(config, sort_keys=False))
        meta = {
            "policy": name, "screen": str(ROOT / args.screen),
            "base_checkpoint": args.init_checkpoint,
            "n_quantized_attention": len(quantized),
            "n_kept_attention_fp32": len(kept_names),
            "n_ffn_exclusions": len(base_excl),
            "quantized_attention": quantized,
            "kept_attention_fp32": kept_names,
            "estimated_artifact_bytes": args.base_artifact_bytes - per_saved * len(quantized),
            "estimated_artifact_gb": (args.base_artifact_bytes - per_saved * len(quantized)) / 1e9,
            "estimated_saved_bytes": per_saved * len(quantized),
            "preliminary": True,
            "note": "Selection from a 16-utterance PTQ screen; validate with QAT pilot.",
        }
        (ROOT / "results" / f"attn_2bit_policy_{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"{name}: quantize {len(quantized)}, keep {len(kept_names)} FP32, "
              f"est {meta['estimated_artifact_gb']:.4f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
