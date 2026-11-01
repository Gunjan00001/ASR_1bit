"""Verify the short-utterance pilot plan without training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from onebit_asr.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    config = load_config(ROOT / "configs/attn_2bit_A_short_oversample.yaml")
    plan = json.loads((ROOT / "results/short_oversample_plan.json").read_text())
    manifest = json.loads(
        (ROOT / "checkpoints/attn_2bit_on_2bitffn_A_full/checkpoint_manifest.json").read_text()
    )
    actual_sha = sha256(ROOT / "checkpoints/attn_2bit_on_2bitffn_A_full/model.safetensors")
    expected_sha = manifest["sha256"]["model.safetensors"]
    if actual_sha != expected_sha or plan["checkpoint_sha256"] != actual_sha:
        raise SystemExit("A checkpoint hash mismatch")

    exclude = list(config["quantization"]["exclude_layers"])
    attention_excluded = [x for x in exclude if "self_attn" in x]
    ffn_excluded = [x for x in exclude if "ffn" in x]
    checks = {
        "checkpoint_hash_matches_manifest": actual_sha == expected_sha,
        "plan_checkpoint_hash_matches": plan["checkpoint_sha256"] == actual_sha,
        "init_checkpoint_is_A_full": config["qat"]["init_checkpoint"]
        == "checkpoints/attn_2bit_on_2bitffn_A_full",
        "bits_2": config["quantization"]["bits"] == 2,
        "scale_per_tensor": config["quantization"]["scale"] == "per_tensor_mean_abs",
        "attention_excluded_60": len(attention_excluded) == 60,
        "attention_quantized_36": 96 - len(attention_excluded) == 36,
        "ffn_excluded_17": len(ffn_excluded) == 17,
        "ffn_quantized_79": 96 - len(ffn_excluded) == 79,
        "plan_rows_2000": len(plan["samples"]) == 2000,
        "plan_short_500": plan["n_short"] == 500,
        "plan_long_1500": plan["n_long"] == 1500,
        "plan_short_frequency_025": abs(plan["target_short_fraction"] - 0.25) < 1e-12,
        "no_dev_overlap": not plan["leakage_check"]["dev_256_overlap"],
        "test_not_used": plan["leakage_check"]["test_used"] is False,
        "train_split_only": plan["leakage_check"]["split"] == "train.100",
        "short_rows_are_repeated": plan["short_repeat_count"] == 135,
    }
    pilot = config["qat"]["pilot"]
    settings = {
        "subset_size": pilot["subset_size"],
        "max_steps": pilot["max_steps"],
        "eval_every_steps": pilot["eval_every_steps"],
        "eval_utterances": pilot["eval_utterances"],
        "batch_size": pilot["batch_size"],
        "gradient_accumulation_steps": pilot["gradient_accumulation_steps"],
        "learning_rate": config["qat"]["training"]["learning_rate"],
        "seed": config["qat"]["training"]["seed"],
        "optimizer": config["qat"]["training"]["optimizer"],
        "lr_scheduler": config["qat"]["training"]["lr_scheduler"],
    }
    checks["settings_fixed"] = (
        settings["subset_size"] == 2000
        and settings["max_steps"] == 500
        and settings["eval_utterances"] == 256
        and settings["batch_size"] == 4
        and settings["gradient_accumulation_steps"] == 4
        and settings["learning_rate"] == 2e-05
        and settings["seed"] == 42
    )
    result = {
        "purpose": "preflight for approved 500-step short-utterance pilot",
        "checkpoint": "checkpoints/attn_2bit_on_2bitffn_A_full",
        "checkpoint_sha256": actual_sha,
        "selection": manifest.get("selection"),
        "config": "configs/attn_2bit_A_short_oversample.yaml",
        "plan": "results/short_oversample_plan.json",
        "checks": checks,
        "settings": settings,
        "policy": {
            "attention_quantized": 36,
            "attention_fp32": len(attention_excluded),
            "ffn_quantized": 79,
            "ffn_fp32": len(ffn_excluded),
            "bits": config["quantization"]["bits"],
            "scale": config["quantization"]["scale"],
        },
        "sampling": {
            "short_definition": plan["short_definition"],
            "short_frequency": plan["target_short_fraction"],
            "short_word_share": plan["target_short_word_share"],
            "weights": plan["effective_sampling_weights"],
            "short_repeat_count": plan["short_repeat_count"],
        },
        "all_passed": all(checks.values()),
    }
    out = ROOT / "results/short_oversample_preflight.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
