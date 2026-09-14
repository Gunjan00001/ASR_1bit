"""Phase 9/10 — first 1-bit experiment: post-training quantization (PTQ).

Workflow (no retraining, FP32 reference untouched):

1. Load the original FP32 checkpoint.
2. Load ``configs/binary.yaml`` (attention + FFN -> BitLinear, heads/FP left FP).
3. Replace the configured ``nn.Linear`` layers with ``BitLinear``.
4. Do NOT retrain, do NOT modify the baseline checkpoint.
5. Evaluate with the *exact* Phase 3 protocol (same 256-utt subset, seed,
   greedy CTC, identical text normalization).
6. Record measured WER/CER, delta vs FP32, size accounting, provenance.

Writes:
    results/ptq_binary.json   (metrics + config + provenance + utterances)
    results/size_report.json  (FP32 vs naive/packed binary accounting)

The plan expects PTQ to degrade WER substantially. That is a recorded result,
not a failure, and no tuning-around happens here.

Usage:
    python scripts/quantize.py [--config configs/binary.yaml]
"""

import argparse
import datetime
import importlib.metadata
import json
import platform
from pathlib import Path

from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.evaluation.evaluate import evaluate_model, print_summary
from onebit_asr.evaluation.model_size import binary_model_size_report
from onebit_asr.models.replace_layers import replace_linears, report_replacement

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def pkg_version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="1-bit PTQ evaluation.")
    parser.add_argument("--config", default="configs/binary.yaml")
    parser.add_argument("--baseline", default=str(RESULTS / "baseline.json"))
    parser.add_argument("--output", default=str(RESULTS / "ptq_binary.json"))
    parser.add_argument("--size-output", default=str(RESULTS / "size_report.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = cfg["model"]
    eval_cfg = cfg["eval"]
    subset_size, seed = eval_cfg["subset_size"], eval_cfg["seed"]
    quant_cfg = cfg.get("quantization", {})
    scale_mode = quant_cfg.get("scale", "per_tensor_mean_abs")
    ste_clip = quant_cfg.get("ste_clip", None)

    print(f"Loading {model_id} (CPU) ...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
    model.eval()

    print("Applying PTQ layer replacement ...")
    replacement = replace_linears(
        model, cfg["layers"], scale_mode=scale_mode, ste_clip=ste_clip
    )
    print(report_replacement(replacement))

    print(f"Loading dev subset (n={subset_size}, seed={seed}) ...")
    samples = load_dev_subset(subset_size=subset_size, seed=seed)

    print("Evaluating binarized model (pinned protocol) ...")
    metrics = evaluate_model(model, processor, samples)

    size = binary_model_size_report(model, repo_id=model_id, scale_mode=scale_mode)

    baseline = json.loads(Path(args.baseline).read_text())
    baseline_wer, baseline_cer = baseline["wer"], baseline["cer"]

    result = {
        "phase": "Phase 9/10 1-bit post-training quantization (PTQ)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "compute_profile": cfg.get("compute_profile", "local"),
        "model_id": model_id,
        "method": "post-training quantization, no retraining",
        "dataset": {"id": eval_cfg.get("dataset", "openslr/librispeech_asr"),
                    "config": eval_cfg.get("config", "clean"),
                    "split": eval_cfg.get("split", "validation"),
                    "subset_size": subset_size, "seed": seed},
        "decoding": "greedy",
        "text_normalization": "lowercase + strip punctuation",
        "config_path": str(Path(args.config).resolve()),
        "config": {"layers": cfg["layers"], "quantization": quant_cfg},
        **metrics,
        "wer_delta_vs_fp32": metrics["wer"] - baseline_wer,
        "cer_delta_vs_fp32": metrics["cer"] - baseline_cer,
        "baseline": {
            "path": str(Path(args.baseline).resolve()),
            "phase": baseline.get("phase"),
            "wer": baseline_wer,
            "cer": baseline_cer,
            "model_id": baseline.get("model_id"),
        },
        "replacement": {
            "n_replaced": replacement["n_replaced"],
            "n_retained": replacement["n_retained"],
            "replaced_params": replacement["replaced_params"],
            "retained_params": replacement["retained_params"],
            "scale_mode": replacement["scale_mode"],
            "ste_clip": replacement["ste_clip"],
            "layer_cls": replacement["layer_cls"],
            "replaced_categories": sorted({e["category"] for e in replacement["replaced"]}),
            "retained_categories": sorted({e["category"] for e in replacement["retained"]}),
        },
        "size": size,
        "versions": {
            "torch": pkg_version("torch"),
            "transformers": pkg_version("transformers"),
            "datasets": pkg_version("datasets"),
            "jiwer": pkg_version("jiwer"),
            "numpy": pkg_version("numpy"),
            "safetensors": pkg_version("safetensors"),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    size_out = Path(args.size_output)
    size_out.write_text(json.dumps(size, indent=2))

    print_summary(result)
    print(f"WER delta vs FP32: {result['wer_delta_vs_fp32']:+.4f}")
    print(f"Saved metrics to {out}")
    print(f"Saved size report to {size_out}")


if __name__ == "__main__":
    main()
