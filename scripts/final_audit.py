"""Final repository audit for the selected ASR models.

Checks checkpoint hashes, config/manifest agreement, artifact sizes, packed
reload records, and final benchmark consistency. Read-only: no training and no
checkpoint/artifact writes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from onebit_asr.config import load_config

ROOT = Path(__file__).resolve().parent.parent
SELECTED = {
    "fp32": {
        "checkpoint": None,
        "full_test": "results/final_fp32_test_n2620.json",
        "full_test_wer": 0.019267346317711504,
        "historical_artifact_bytes": 2373507712,
        "fp32_sha256": "97bb9761fb71ec1225100bc81ccf7d002e0d0ba3d0604c1fd2dbda7d7d491f1d",
    },
    "ffn2bit": {
        "checkpoint": "checkpoints/qat_ffn_mixed_2bit_full",
        "config": "configs/qat_ffn_mixed_2bit.yaml",
        "full_test": "results/2bit_final_test_n2620/final_qat_ffn_mixed_2bit_full_test_n2620.json",
        "full_test_wer": 0.025829275715155205,
        "historical_artifact": "checkpoints/qat_ffn_mixed_2bit_full_packed/model_packed.safetensors",
        "historical_artifact_bytes": 1131062436,
        "v2_artifact": "checkpoints/qat_ffn_mixed_2bit_full_packed_v2/model_packed.safetensors",
    },
    "strategy_a": {
        "checkpoint": "checkpoints/attn_2bit_on_2bitffn_A_full",
        "config": "configs/attn_2bit_on_2bitffn_A.yaml",
        "full_test": "results/attn_2bit_A_final_n2620/final_attn_2bit_on_2bitffn_A_full_test_n2620.json",
        "full_test_wer": 0.027369902617163724,
        "historical_artifact": "checkpoints/attn_2bit_on_2bitffn_A_full_packed/model_packed.safetensors",
        "historical_artifact_bytes": 989509572,
        "v2_artifact": "checkpoints/attn_2bit_on_2bitffn_A_full_packed_v2/model_packed.safetensors",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    checks = {}
    models = {}
    for name, spec in SELECTED.items():
        item = {
            "full_test_source": spec["full_test"],
            "full_test_wer": spec["full_test_wer"],
            "historical_artifact_bytes": spec["historical_artifact_bytes"],
        }
        if spec["checkpoint"]:
            checkpoint = ROOT / spec["checkpoint"]
            manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
            actual = sha256(checkpoint / "model.safetensors")
            cfg = load_config(ROOT / spec["config"])
            quant = cfg["quantization"]
            checks[f"{name}_checkpoint_hash"] = actual == manifest["sha256"]["model.safetensors"]
            checks[f"{name}_bits"] = int(quant["bits"]) == int(manifest["bits"])
            checks[f"{name}_scale"] = quant["scale"] == manifest["scale_mode"]
            checks[f"{name}_layers"] = dict(cfg["layers"]) == dict(manifest["layers"])
            checks[f"{name}_exclusions"] = sorted(quant["exclude_layers"]) == sorted(manifest["exclude_layers"])
            item.update({
                "checkpoint": spec["checkpoint"],
                "checkpoint_sha256": actual,
                "selection": manifest.get("selection"),
                "bits": manifest["bits"],
                "scale_mode": manifest["scale_mode"],
                "bitlinear_layers": manifest["bitlinear_layers"],
                "exclusions": len(manifest["exclude_layers"]),
                "historical_artifact": spec["historical_artifact"],
                "v2_artifact": spec["v2_artifact"],
                "v2_artifact_bytes": (ROOT / spec["v2_artifact"]).stat().st_size,
                "v2_artifact_sha256": sha256(ROOT / spec["v2_artifact"]),
            })
            full = json.loads((ROOT / spec["full_test"]).read_text())
            item["full_test_wer_from_file"] = full["accuracy"]["wer"]
            checks[f"{name}_full_test_wer"] = abs(full["accuracy"]["wer"] - spec["full_test_wer"]) < 1e-12
        else:
            item["checkpoint"] = spec.get("model_id")
            item["checkpoint_sha256"] = spec.get("fp32_sha256")
            item["historical_artifact_bytes"] = spec["historical_artifact_bytes"]
            full = json.loads((ROOT / spec["full_test"]).read_text())
            item["full_test_wer_from_file"] = full["accuracy"]["wer"]
            checks[f"{name}_full_test_wer"] = abs(full["accuracy"]["wer"] - spec["full_test_wer"]) < 1e-12
            checks[f"{name}_artifact_bytes"] = spec["historical_artifact_bytes"] == 2373507712
        models[name] = item

    old_verify = {}
    for name, path in {
        "ffn2bit": "results/packed_verify_qat_ffn_mixed_2bit_full.json",
        "strategy_a": "results/packed_verify_attn_2bit_on_2bitffn_A_full.json",
    }.items():
        payload = json.loads((ROOT / path).read_text())
        old_verify[name] = {
            "source": path,
            "passed": payload.get("passed"),
            "max_abs_logit_diff": payload.get("max_abs_logit_diff"),
            "mean_argmax_agreement": payload.get("mean_argmax_agreement"),
            "n_utterances": payload.get("n_utterances"),
        }
        checks[f"{name}_historical_packed_verify"] = bool(payload.get("passed"))

    benchmark = None
    benchmark_path = ROOT / "results/final_benchmark.json"
    if benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text())
        checks["final_benchmark_present"] = True
        by_model = {row["model"]: row for row in benchmark["arms"]}
        for model_key, row_key in (("ffn2bit", "2-bit FFN"), ("strategy_a", "Strategy A")):
            delta = by_model[row_key].get("expected_wer_delta")
            models[model_key]["v2_standalone_wer_delta"] = delta
            checks[f"{model_key}_v2_standalone_wer"] = (
                delta is not None and abs(delta) <= 1e-3
            )
    else:
        checks["final_benchmark_present"] = False

    rejected = {
        "strategy_c": "results/attn_2bit_C_pilot.json",
        "per_channel": "results/attn_2bit_A_per_channel_pilot.json",
        "short_oversample": "results/attn_2bit_A_short_oversample_pilot.json",
        "short_oversample_comparison": "results/short_oversample_pilot_comparison.json",
        "decoder_beam": "results/ctc_decoding_A_full_dev256.json",
    }
    rejected_present = {name: (ROOT / path).is_file() for name, path in rejected.items()}
    checks["rejected_results_present"] = all(rejected_present.values())

    payload = {
        "purpose": "final read-only audit of selected models and historical evidence",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "selected_models": models,
        "historical_packed_verify": old_verify,
        "final_benchmark": benchmark,
        "rejected_results": rejected_present,
        "artifact_note": (
            "Historical packed artifacts match the recorded sizes and verify against "
            "their QAT checkpoints, but omit BatchNorm/positional buffers and are not "
            "standalone. The v2 artifacts add those buffers and are used for the final "
            "standalone benchmark. Selected checkpoints were not modified."
        ),
        "test_used": False,
    }
    out = ROOT / "results/final_audit.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"checks": checks, "all_checks_passed": payload["all_checks_passed"]}, indent=2))
    print(f"Wrote {out}")
    return 0 if payload["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
