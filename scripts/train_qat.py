"""Phase 11 — quantization-aware training entry point.

Two modes:

- ``--smoke``: a tiny CPU dry run (small train.100 subset, few steps, small dev
  eval). Cheap sanity check of the whole QAT pipeline; writes a separate
  ``results/qat_smoke.json``.
- full mode: the real cloud-GPU experiment (config-driven subset/epochs). NOT
  launched by this task; requires GPU access and explicit approval.

Both modes preserve the pinned evaluation protocol and never touch
``results/baseline.json``.

Usage:
    python scripts/train_qat.py --config configs/qat.yaml --smoke
    python scripts/train_qat.py --config configs/qat.yaml            # cloud-gpu only
"""

import argparse
import copy
import datetime
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.collator import DataCollatorCTCWithPadding
from onebit_asr.data.dataset import load_dev_subset, load_train_subset, prepare_ctc_features
from onebit_asr.evaluation.evaluate import evaluate_model, print_summary
from onebit_asr.models.replace_layers import report_replacement
from onebit_asr.training.qat import (
    CHECKPOINT_FORMAT,
    build_training_arguments,
    build_trainer,
    freeze_feature_extractor,
    freeze_report,
    load_qat_checkpoint,
    make_loss_recorder,
    master_weight_delta,
    master_weight_snapshot,
    probe_gradient_flow,
    save_qat_checkpoint,
    setup_qat_model,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def pkg_version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except Exception:  # noqa: BLE001
        return "unknown"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def resolve_training(cfg: dict, smoke: bool) -> dict:
    """Merge the smoke overrides over the full training block."""
    training = copy.deepcopy(cfg["qat"]["training"])
    if smoke:
        training.update(cfg["qat"].get("smoke", {}))
    return training


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 11 QAT (smoke or full).")
    parser.add_argument("--config", default="configs/qat.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny CPU dry run (recommended before any GPU run)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = Path(args.output or (RESULTS / ("qat_smoke.json" if args.smoke else "qat.json")))
    checkpoint_dir = Path(args.checkpoint_dir or
                          (ROOT / "checkpoints" / ("qat_smoke" if args.smoke else "qat")))

    eval_cfg = cfg["eval"]
    quant_cfg = cfg.get("quantization", {})
    scale_mode = quant_cfg.get("scale", "per_tensor_mean_abs")
    ste_clip = quant_cfg.get("ste_clip", None)
    training = resolve_training(cfg, args.smoke)

    # Leakage guard: training must never use the evaluation split.
    if training["split"] == eval_cfg.get("split"):
        raise SystemExit(
            f"Refusing to train on the evaluation split {training['split']!r}"
        )

    model_id = cfg["qat"]["init_from"]
    print(f"Loading {model_id} (init_from) ...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
    if model.config.pad_token_id is None:
        raise SystemExit("model.config.pad_token_id is None; CTC blank unknown")
    model, replacement = setup_qat_model(model, cfg["layers"], scale_mode=scale_mode, ste_clip=ste_clip)
    print(report_replacement(replacement))

    if cfg["qat"].get("freeze_feature_extractor", True):
        frozen = freeze_feature_extractor(model)
        print(f"Froze CNN feature extractor: {frozen:,} params")
    freeze = freeze_report(model)
    print(f"Trainable: {freeze['trainable_params']:,} | Frozen: {freeze['frozen_params']:,}")

    print(f"Loading train subset ({training['split']}, n={training['subset_size']}, seed={training['seed']}) ...")
    train_samples = load_train_subset(
        subset_size=training["subset_size"], seed=training["seed"], split=training["split"]
    )
    features, provenance = prepare_ctc_features(train_samples, processor)
    train_dataset = Dataset.from_list(features)

    collator = DataCollatorCTCWithPadding(processor=processor)
    training_args = build_training_arguments(
        training, str(checkpoint_dir), smoke=args.smoke, train_num_samples=len(features)
    )
    recorder = make_loss_recorder()
    trainer = build_trainer(model, train_dataset, collator, training_args, recorder)

    # Gradient-flow probe (before training: Trainer zeroes grads after each step).
    probe_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=collator)
    probe_batch = next(iter(probe_loader))
    gradient_flow = probe_gradient_flow(model, probe_batch)
    print(f"Gradient probe: {gradient_flow['with_nonzero_grad']}/"
          f"{gradient_flow['total_bitlinear']} BitLinear masters got gradients; "
          f"frozen feature extractor grad zero={gradient_flow['frozen_feature_extractor_grad_is_zero']}")

    masters_before = master_weight_snapshot(model)

    print(f"Training ({'smoke' if args.smoke else 'full'}): "
          f"lr={training['learning_rate']} epochs={training.get('epochs')} "
          f"gradient_checkpointing={training_args.gradient_checkpointing} "
          f"device={training_args.device} ...")
    trainer.train()

    masters_after = master_weight_snapshot(model)
    master_update = master_weight_delta(masters_before, masters_after)
    print(f"Master weights updated: {master_update['digest_changed']} "
          f"(|W| sum delta {master_update['abs_sum_delta']:+.2f}, "
          f"binary fraction {master_update['fraction_exactly_pm1_after']:.2e})")

    # --- Evaluation under the pinned protocol ---
    n_eval = training.get("eval_utterances", eval_cfg["subset_size"]) if args.smoke else eval_cfg["subset_size"]
    samples = load_dev_subset(subset_size=eval_cfg["subset_size"], seed=eval_cfg["seed"])[:n_eval]
    print(f"Evaluating on {len(samples)} dev utterances (pinned protocol) ...")
    model.eval()  # pinned protocol is inference; ensure dropout is off
    metrics = evaluate_model(model, processor, samples)

    # --- Checkpoint (FP master weights; not packed) + save/load verification ---
    manifest = save_qat_checkpoint(model, checkpoint_dir)
    reloaded, _ = load_qat_checkpoint(
        checkpoint_dir, Wav2Vec2ConformerForCTC, cfg["layers"],
        scale_mode=scale_mode, ste_clip=ste_clip,
    )
    reloaded.eval()
    inp = processor(samples[0]["audio"], sampling_rate=16_000, return_tensors="pt").input_values
    # Compare reloaded model against the trained model on one utterance.
    model.eval()
    with torch.no_grad():
        logits_trained = model(inp).logits
        logits_reloaded = reloaded(inp).logits
    save_load_ok = bool(torch.allclose(logits_trained, logits_reloaded, atol=1e-5))

    def load_result(path: Path) -> dict:
        return json.loads(path.read_text()) if path.is_file() else {}

    baseline = load_result(RESULTS / "baseline.json")
    ptq = load_result(RESULTS / "ptq_binary.json")
    baseline_wer, ptq_wer = baseline.get("wer"), ptq.get("wer")

    result = {
        "phase": "Phase 11 QAT (smoke)" if args.smoke else "Phase 11 QAT (full)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "compute_profile": "local" if not torch.cuda.is_available() else "cloud-gpu",
        "config_profile": cfg.get("compute_profile"),
        "git_commit": git_commit(),
        "model_id": model_id,
        "init_from": model_id,
        "quantization": {"scale": scale_mode, "ste": quant_cfg.get("ste"), "ste_clip": ste_clip},
        "layers": cfg["layers"],
        "replacement": {
            "n_replaced": replacement["n_replaced"],
            "n_retained": replacement["n_retained"],
            "replaced_params": replacement["replaced_params"],
            "retained_params": replacement["retained_params"],
            "replaced_categories": sorted({e["category"] for e in replacement["replaced"]}),
            "retained_categories": sorted({e["category"] for e in replacement["retained"]}),
        },
        "freeze": freeze,
        "training": {
            "dataset": training["dataset"],
            "config": training.get("config"),
            "split": training["split"],
            "subset_size": training["subset_size"],
            "seed": training["seed"],
            "sample_ids": [p["id"] for p in provenance],
            "batch_size": training["batch_size"],
            "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "learning_rate": training["learning_rate"],
            "epochs": training.get("epochs"),
            "max_steps": training.get("max_steps"),
            "warmup_ratio": training.get("warmup_ratio"),
            "warmup_steps": training_args.warmup_steps,
            "max_grad_norm": training.get("max_grad_norm"),
            "weight_decay": training.get("weight_decay"),
            "optimizer": training.get("optimizer"),
            "lr_scheduler": training.get("lr_scheduler"),
            "device": str(training_args.device),
            "steps_completed": trainer.state.global_step,
            "final_train_loss": next(
                (e["loss"] for e in reversed(trainer.state.log_history) if "loss" in e), None
            ),
            "train_loss_history": recorder.history,
        },
        "gradient_flow": gradient_flow,
        "master_weights": master_update,
        "eval": {
            "protocol": "fixed dev-clean subset, seed, greedy CTC, normalized text",
            "subset_size": eval_cfg["subset_size"],
            "n_utterances": metrics["n_utterances"],
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "rtf": metrics["rtf"],
            "latency_s": metrics["latency_s"],
            "peak_rss_mb": metrics["peak_rss_mb"],
            "sample_ids": metrics["sample_ids"],
        },
        "comparison": {
            "vs_fp32_baseline": {
                "wer": baseline_wer,
                "delta_wer": (metrics["wer"] - baseline_wer) if baseline_wer is not None else None,
            },
            "vs_ptq": {
                "wer": ptq_wer,
                "delta_wer": (metrics["wer"] - ptq_wer) if ptq_wer is not None else None,
            },
        },
        "checkpoint": {
            "path": str(checkpoint_dir),
            "format": CHECKPOINT_FORMAT,
            "is_1bit_packed": False,
            "manifest": manifest,
            "save_load_verified": save_load_ok,
        },
        "baseline": {"path": str(RESULTS / "baseline.json"), "wer": baseline_wer,
                     "cer": baseline.get("cer")},
        "ptq": {"path": str(RESULTS / "ptq_binary.json"), "wer": ptq_wer, "cer": ptq.get("cer")},
        "versions": {
            "torch": pkg_version("torch"),
            "transformers": pkg_version("transformers"),
            "datasets": pkg_version("datasets"),
            "jiwer": pkg_version("jiwer"),
            "numpy": pkg_version("numpy"),
            "safetensors": pkg_version("safetensors"),
        },
        "config_path": cfg.get("_config_path"),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print_summary(metrics)
    print(f"trainable params: {freeze['trainable_params']:,} | frozen: {freeze['frozen_params']:,}")
    print(f"save/load verified: {save_load_ok}")
    print(f"WER (QAT {'smoke' if args.smoke else 'full'}): {metrics['wer']:.4f} "
          f"| vs FP32 baseline: {baseline_wer}")
    print(f"Saved to {output}")
    if not save_load_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
