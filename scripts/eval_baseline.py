"""Phase 3 — FP32 baseline evaluation (pinned protocol).

Uses the shared, model-agnostic runner in
``onebit_asr.evaluation.evaluate`` so that the *same* protocol is applied to
the FP32 baseline (here) and to the 1-bit PTQ model (``scripts/quantize.py``).

Writes machine-readable ``results/baseline.json`` — the immutable reference
for all later experiments. Never hand-edit this file.

Usage:
    python scripts/eval_baseline.py [--config configs/baseline.yaml]
"""

import argparse
import datetime
import importlib.metadata
import json
import platform
from pathlib import Path

import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.evaluation.evaluate import evaluate_model, print_summary
from onebit_asr.evaluation.model_size import checkpoint_size_report

RESULTS = Path(__file__).resolve().parent.parent / "results"


def pkg_version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="FP32 baseline evaluation.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default=str(RESULTS / "baseline.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = cfg["model"]
    eval_cfg = cfg["eval"]
    subset_size, seed = eval_cfg["subset_size"], eval_cfg["seed"]

    print(f"Loading {model_id} (CPU) ...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
    model.eval()

    print(f"Loading dev subset (n={subset_size}, seed={seed}) ...")
    samples = load_dev_subset(subset_size=subset_size, seed=seed)

    metrics = evaluate_model(model, processor, samples)
    size = checkpoint_size_report(model_id)

    result = {
        "phase": "Phase 3 FP32 baseline",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "compute_profile": cfg.get("compute_profile", "local"),
        "model_id": model_id,
        "dataset": {"id": eval_cfg.get("dataset", "openslr/librispeech_asr"),
                    "config": eval_cfg.get("config", "clean"),
                    "split": eval_cfg.get("split", "validation"),
                    "subset_size": subset_size, "seed": seed},
        "decoding": "greedy",
        "text_normalization": "lowercase + strip punctuation",
        **metrics,
        **size,
        "versions": {
            "torch": pkg_version("torch"),
            "transformers": pkg_version("transformers"),
            "datasets": pkg_version("datasets"),
            "jiwer": pkg_version("jiwer"),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print_summary(result)
    print(f"Checkpoint: {size['checkpoint_bytes_safetensors'] / 1e9:.2f} GB safetensors "
          f"({size['checkpoint_bytes_full_download'] / 1e9:.2f} GB full download)")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
