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
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.collator import DataCollatorCTCWithPadding
from onebit_asr.data.dataset import build_ctc_train_dataset, load_dev_subset
from onebit_asr.evaluation.evaluate import evaluate_model, print_summary
from onebit_asr.models.replace_layers import report_replacement
from onebit_asr.training.qat import (
    CHECKPOINT_FORMAT,
    build_training_arguments,
    build_trainer,
    extrapolate_training_time,
    freeze_feature_extractor,
    freeze_report,
    gpu_memory_stats,
    load_qat_checkpoint,
    make_loss_recorder,
    master_weight_delta,
    master_weight_snapshot,
    probe_gradient_flow,
    reset_gpu_peak_memory,
    save_qat_checkpoint,
    setup_qat_model,
    vram_headroom,
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


def resolve_training(cfg: dict, smoke: bool, probe: bool = False,
                     probe_steps: int | None = None) -> dict:
    """Merge smoke/probe overrides over the full training block."""
    training = copy.deepcopy(cfg["qat"]["training"])
    if smoke:
        training.update(cfg["qat"].get("smoke", {}))
    if probe:
        steps = probe_steps if probe_steps is not None else \
            cfg["qat"].get("probe", {}).get("steps", 30)
        training["max_steps"] = int(steps)
        training["epochs"] = 1
    return training


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 11 QAT (smoke, probe, or full).")
    parser.add_argument("--config", default="configs/qat.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny CPU dry run (recommended before any GPU run)")
    parser.add_argument("--probe", action="store_true",
                        help="short timed/VRAM probe; gates the full run (Phase 11 §7b)")
    parser.add_argument("--probe-steps", type=int, default=None,
                        help="override the configured probe step count")
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    probe_output = RESULTS / "qat_probe.json"
    if args.output:
        output = Path(args.output)
    elif args.probe:
        output = probe_output
    elif args.smoke:
        output = RESULTS / "qat_smoke.json"
    else:
        output = RESULTS / "qat.json"
    checkpoint_dir = Path(args.checkpoint_dir or
                          (ROOT / "checkpoints" / ("qat_smoke" if args.smoke else "qat")))

    eval_cfg = cfg["eval"]
    quant_cfg = cfg.get("quantization", {})
    scale_mode = quant_cfg.get("scale", "per_tensor_mean_abs")
    ste_clip = quant_cfg.get("ste_clip", None)
    training = resolve_training(cfg, args.smoke, args.probe, args.probe_steps)

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
    # Streamed, bounded-memory construction (see build_ctc_train_dataset): a
    # 10k-utterance subset materialized as Python lists OOM-kills the Kaggle
    # worker. Same selection/order as before.
    train_dataset, provenance = build_ctc_train_dataset(
        split=training["split"], subset_size=training["subset_size"],
        seed=training["seed"], processor=processor,
    )

    collator = DataCollatorCTCWithPadding(processor=processor)
    training_args = build_training_arguments(
        training, str(checkpoint_dir), smoke=(args.smoke or args.probe),
        train_num_samples=len(train_dataset),
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

    mode = "probe" if args.probe else ("smoke" if args.smoke else "full")
    print(f"Training ({mode}): "
          f"lr={training['learning_rate']} epochs={training.get('epochs')} "
          f"max_steps={training.get('max_steps')} "
          f"gradient_checkpointing={training_args.gradient_checkpointing} "
          f"device={training_args.device} ...")

    # --- Probe mode: short timed/VRAM run, then stop (no eval, no checkpoint) ---
    if args.probe:
        reset_gpu_peak_memory()
        train_output = trainer.train()
        runtime = float(train_output.metrics.get("train_runtime", 0.0))
        steps_done = max(1, trainer.state.global_step)
        seconds_per_step = runtime / steps_done

        mem = gpu_memory_stats()
        full = cfg["qat"]["training"]
        extrapolation = extrapolate_training_time(
            seconds_per_step,
            subset_size=int(full["subset_size"]),
            batch_size=int(training["batch_size"]),
            gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
            epochs=int(full.get("epochs", 1)),
        )
        headroom = vram_headroom(mem["peak_reserved_bytes"], mem["total_bytes"])

        probe_result = {
            "phase": "Phase 11 QAT GPU probe",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform": platform.platform(),
            "compute_profile": "cloud-gpu" if mem["cuda"] else "local",
            "config_profile": cfg.get("compute_profile"),
            "git_commit": git_commit(),
            "mode": mode,
            "is_real_gpu_probe": bool(mem["cuda"]),
            "notice": (
                "Measured probe values for planning the full run. If is_real_gpu_probe "
                "is false this was a CPU validation of the probe code, NOT a GPU result."
            ),
            "model_id": cfg["qat"]["init_from"],
            "layers": cfg["layers"],
            "quantization": {"scale": scale_mode, "ste": quant_cfg.get("ste"), "ste_clip": ste_clip},
            "freeze": freeze,
            "replacement": {
                "n_replaced": replacement["n_replaced"],
                "n_retained": replacement["n_retained"],
            },
            "probe": {
                "steps": steps_done,
                "batch_size": training["batch_size"],
                "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "fp16": training_args.fp16,
                "bf16": training_args.bf16,
                "dtype": "fp32" if not (training_args.fp16 or training_args.bf16)
                else ("fp16" if training_args.fp16 else "bf16"),
                "runtime_seconds": runtime,
                "seconds_per_optimizer_step": seconds_per_step,
                "train_loss_history": recorder.history,
            },
            "memory": mem,
            "extrapolation_to_full_run": {
                "full_subset_size": full["subset_size"],
                "full_epochs": full.get("epochs"),
                **extrapolation,
            },
            "vram_headroom": headroom,
            "verdict": {
                "full_run_recommended": bool(headroom["passes"]) if headroom["applicable"] else None,
                "reason": (
                    "CPU validation run: no GPU memory to assess."
                    if not headroom["applicable"]
                    else (
                        f"peak reserved {mem['peak_reserved_bytes'] / 1e9:.2f} GB of "
                        f"{mem['total_bytes'] / 1e9:.2f} GB; free fraction "
                        f"{headroom['free_fraction']:.1%}"
                    )
                ),
            },
            "gradient_flow": gradient_flow,
            "versions": {
                "torch": pkg_version("torch"),
                "transformers": pkg_version("transformers"),
                "datasets": pkg_version("datasets"),
            },
            "config_path": cfg.get("_config_path"),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(probe_result, indent=2))

        print(f"Probe: {steps_done} steps in {runtime:.1f}s "
              f"({seconds_per_step:.3f}s/step) on {mem['device_name'] or 'cpu'}")
        if mem["cuda"]:
            print(f"Peak reserved: {mem['peak_reserved_bytes'] / 1e9:.2f} GB / "
                  f"{mem['total_bytes'] / 1e9:.2f} GB "
                  f"(free {headroom['free_fraction']:.1%}; "
                  f"headroom {'OK' if headroom['passes'] else 'INSUFFICIENT'})")
        print(f"Extrapolated full run: {extrapolation['optimizer_steps_total']} steps, "
              f"~{extrapolation['estimated_hours']:.2f} h")
        print(f"Saved probe to {output}")
        return 0

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
    device = next(model.parameters()).device
    reloaded = reloaded.to(device)
    reloaded.eval()
    inp = processor(samples[0]["audio"], sampling_rate=16_000,
                    return_tensors="pt").input_values.to(device)
    # Compare reloaded model against the trained model on one utterance.
    model.eval()
    with torch.no_grad():
        logits_trained = model(inp).logits
        logits_reloaded = reloaded(inp).logits
    # GPU fp32 kernels are not bit-identical across processes, so a tight
    # atol=1e-5 on raw logits gives false negatives. Compare the decoded argmax
    # (what actually matters) plus a tolerant logit closeness.
    same_argmax = bool(torch.equal(logits_trained.argmax(dim=-1),
                                   logits_reloaded.argmax(dim=-1)))
    close_logits = bool(torch.allclose(logits_trained, logits_reloaded,
                                       atol=1e-3, rtol=1e-3))
    save_load_ok = same_argmax and close_logits

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
