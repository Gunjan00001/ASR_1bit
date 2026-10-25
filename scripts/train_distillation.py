"""Phase 13 — knowledge distillation training entry point.

Student: the configured ``BitLinear``-replaced Wav2Vec2-Conformer (default:
attention + FFN, the QAT configuration that collapsed under PTQ).
Teacher: the frozen FP32 checkpoint; its logits are precomputed once for the
training subset and reused across all epochs.

Modes:
    python scripts/train_distillation.py --config configs/distill_attn_ffn.yaml --smoke
    python scripts/train_distillation.py --config configs/distill_attn_ffn.yaml

The smoke mode builds a 16-utterance dataset and runs 8 steps on CPU or GPU; it
is a pipeline check, not an accuracy experiment. Full mode writes
``results/<output>`` and an FP32-master checkpoint (not bit-packed; packing is
``scripts/pack_qat_model.py``).

Debugging notes
---------------
- **Teacher cache key:** the cache path is derived from teacher id, split,
  subset size, and seed. If you change any of them, a new cache is built. Pass
  ``--refresh-teacher-cache`` to force a recompute even if a cache exists.
- **Free the teacher before the student.** The teacher is deleted and the CUDA
  cache is emptied after precompute; otherwise the student + optimizer can OOM.
- **Move the student before the probe.** The probe derives the device from the
  model; a CPU model with CUDA batches is a device-mismatch bug.
- **Inspect `loss_components`.** If KD ≫ CTC, lower λ_kd. The λ_kd=0.5 run
  failed for exactly this reason; λ_kd=0.1 improved it.
- **Save/load is a diagnostic** (argmax agreement + loose logits), as in QAT.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import build_ctc_train_dataset, load_dev_subset
from onebit_asr.evaluation.evaluate import evaluate_model, print_summary
from onebit_asr.models.replace_layers import report_replacement
from onebit_asr.training.distillation import (
    DistillationCollator,
    DistillationDataset,
    build_distillation_trainer,
    distillation_loss,
    load_teacher_logits,
    precompute_teacher_logits,
    save_teacher_logits,
    teacher_logits_cache_path,
)
from onebit_asr.training.qat import (
    CHECKPOINT_FORMAT,
    bitlinear_grad_stats,
    build_training_arguments,
    freeze_feature_extractor,
    freeze_report,
    load_qat_checkpoint,
    make_loss_recorder,
    master_weight_delta,
    master_weight_snapshot,
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
    training = copy.deepcopy(cfg["qat"]["training"])
    if smoke:
        training.update(cfg["qat"].get("smoke", {}))
    return training


def _probe_distillation_step(model, loader, device, distill_cfg) -> dict:
    """One forward/backward with the combined loss; verifies gradient flow."""
    batch = next(iter(loader))
    # Derive the device from the model, not from the caller: the probe must run
    # wherever the model currently lives (the stock Trainer moves it later).
    model_device = next(model.parameters()).device
    batch = {
        key: (value.to(model_device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }
    teacher_logits = batch.pop("teacher_logits")
    teacher_mask = batch.pop("teacher_mask")

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss, parts = distillation_loss(
        outputs.logits, teacher_logits, teacher_mask,
        ctc_loss=outputs.loss,
        temperature=distill_cfg.get("temperature", 2.0),
        lambda_ctc=distill_cfg.get("lambda_ctc", 0.5),
        lambda_kd=distill_cfg.get("lambda_kd", 0.5),
    )
    loss.backward()
    grad_stats = bitlinear_grad_stats(model)
    feature_extractor = model.get_submodule("wav2vec2_conformer.feature_extractor")
    frozen_grad_zero = all(
        p.grad is None or torch.count_nonzero(p.grad) == 0
        for p in feature_extractor.parameters()
    )
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return {
        "probe_loss": float(loss.detach()),
        "probe_parts": parts,
        "loss_is_finite": bool(torch.isfinite(loss)),
        **grad_stats,
        "frozen_feature_extractor_grad_is_zero": frozen_grad_zero,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 13 distillation.")
    parser.add_argument("--config", default="configs/distill_attn_ffn.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny pipeline check (16 utterances, 8 steps)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--teacher-cache", default=None,
                        help="explicit teacher-logits cache path (default: derived)")
    parser.add_argument("--refresh-teacher-cache", action="store_true",
                        help="recompute teacher logits even if the cache exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    distill_cfg = cfg.get("distillation", {})
    if "teacher" not in distill_cfg:
        raise SystemExit(f"{args.config} has no distillation.teacher")

    if args.output:
        output = Path(args.output)
    elif args.smoke:
        output = RESULTS / "distillation_smoke.json"
    else:
        output = RESULTS / "distill_attn_ffn.json"
    checkpoint_dir = Path(
        args.checkpoint_dir or (ROOT / "checkpoints" / ("distill_smoke" if args.smoke else "distill_attn_ffn"))
    )

    eval_cfg = cfg["eval"]
    quant_cfg = cfg.get("quantization", {})
    scale_mode = quant_cfg.get("scale", "per_tensor_mean_abs")
    ste_clip = quant_cfg.get("ste_clip", None)
    exclude_layers = list(quant_cfg.get("exclude_layers") or [])
    training = resolve_training(cfg, args.smoke)

    if training["split"] == eval_cfg.get("split"):
        raise SystemExit(
            f"Refusing to train on the evaluation split {training['split']!r}"
        )

    teacher_id = distill_cfg["teacher"]
    student_id = cfg["qat"]["init_from"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"Teacher: {teacher_id} (frozen FP32)")
    print(f"Student: {student_id} -> BitLinear {sorted(k for k, v in cfg['layers'].items() if v)}")

    processor = Wav2Vec2Processor.from_pretrained(student_id)

    print(f"Loading train subset ({training['split']}, n={training['subset_size']}, seed={training['seed']}) ...")
    train_dataset, provenance = build_ctc_train_dataset(
        split=training["split"], subset_size=training["subset_size"],
        seed=training["seed"], processor=processor,
    )
    print(f"Train dataset: {len(train_dataset)} utterances")

    # --- Teacher logits: precompute once, reuse across epochs -----------------
    # The cache key includes teacher/split/subset/seed. Changing any of them
    # produces a different path, so a stale cache cannot be silently reused.
    cache_path = Path(args.teacher_cache) if args.teacher_cache else teacher_logits_cache_path(
        distill_cfg.get("teacher_logits_cache", ROOT / "checkpoints" / "teacher_logits"),
        teacher_id=teacher_id, split=training["split"],
        subset_size=training["subset_size"], seed=training["seed"],
    )
    cache_info: dict = {"path": str(cache_path), "reused": False}
    if cache_path.is_file() and not args.refresh_teacher_cache:
        print(f"Reusing teacher logits cache: {cache_path}")
        teacher_logits, cache_meta = load_teacher_logits(cache_path)
        cache_info["reused"] = True
        cache_info["meta"] = cache_meta
    else:
        print("Precomputing teacher logits (once per training subset) ...")
        teacher = Wav2Vec2ConformerForCTC.from_pretrained(teacher_id)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        teacher_logits = precompute_teacher_logits(
            teacher, train_dataset, device=device,
            max_samples=len(train_dataset),
            progress_every=500 if len(train_dataset) > 100 else 4,
        )
        del teacher
        # Free the teacher's CUDA allocations before loading the student and
        # its optimizer state; otherwise a 80 GB GPU can still OOM.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cache_info.update(save_teacher_logits(
            teacher_logits, cache_path,
            meta={
                "teacher_id": teacher_id,
                "split": training["split"],
                "subset_size": training["subset_size"],
                "seed": training["seed"],
                "source": "one forward pass per utterance, no padding",
            },
        ))
        cache_info["reused"] = False

    if len(teacher_logits) != len(train_dataset):
        raise SystemExit(
            f"teacher logits {len(teacher_logits)} != dataset {len(train_dataset)}"
        )
    vocab = int(teacher_logits[0].shape[-1])
    print(f"Teacher logits: {len(teacher_logits)} samples, vocab {vocab}")

    # --- Student ---------------------------------------------------------------
    print("Building 1-bit student ...")
    model = Wav2Vec2ConformerForCTC.from_pretrained(student_id)
    model, replacement = setup_qat_model(
        model, cfg["layers"], scale_mode=scale_mode, ste_clip=ste_clip,
        exclude_layers=exclude_layers,
    )
    print(report_replacement(replacement))
    if cfg["qat"].get("freeze_feature_extractor", True):
        frozen = freeze_feature_extractor(model)
        print(f"Froze CNN feature extractor: {frozen:,} params")
    freeze = freeze_report(model)
    print(f"Trainable: {freeze['trainable_params']:,} | Frozen: {freeze['frozen_params']:,}")

    # Put the student on the accelerator before the probe. The stock Trainer
    # would move it at train() time; doing it explicitly keeps the pre-training
    # gradient probe on the same device as training.
    model.to(device)

    distill_dataset = DistillationDataset(train_dataset, teacher_logits)
    collator = DistillationCollator(processor=processor)
    # `remove_unused_columns=False` is required: the batch carries
    # teacher_logits/teacher_mask, which are not fields in the model's forward
    # signature. Without it, the Trainer would drop them before compute_loss.
    training_args = build_training_arguments(
        training, str(checkpoint_dir), smoke=args.smoke,
        train_num_samples=len(distill_dataset), remove_unused_columns=False,
    )
    recorder = make_loss_recorder()
    trainer = build_distillation_trainer(
        model, distill_dataset, collator, training_args,
        temperature=float(distill_cfg.get("temperature", 2.0)),
        lambda_ctc=float(distill_cfg.get("lambda_ctc", 0.5)),
        lambda_kd=float(distill_cfg.get("lambda_kd", 0.5)),
        loss_recorder=recorder,
    )

    probe_loader = torch.utils.data.DataLoader(
        distill_dataset, batch_size=2, collate_fn=collator
    )
    # Probe before train(): the optimizer zeroes gradients after each step, so a
    # post-training probe would report zeros. This is a real forward/backward on
    # one batch with the combined CTC+KD loss.
    probe = _probe_distillation_step(model, probe_loader, device, distill_cfg)
    print(f"Distillation probe: loss={probe['probe_loss']:.2f} "
          f"(ctc={probe['probe_parts']['ctc']:.2f}, "
          f"kd={probe['probe_parts']['kd']:.2f}); "
          f"{probe['with_nonzero_grad']}/{probe['total_bitlinear']} BitLinear masters "
          f"got gradients")

    masters_before = master_weight_snapshot(model)

    print(f"Training (distillation, {'smoke' if args.smoke else 'full'}): "
          f"lr={training['learning_rate']} epochs={training.get('epochs')} "
          f"max_steps={training.get('max_steps')} "
          f"T={distill_cfg.get('temperature')} "
          f"lambda_ctc={distill_cfg.get('lambda_ctc')} "
          f"lambda_kd={distill_cfg.get('lambda_kd')} "
          f"device={training_args.device} ...")
    trainer.train()

    masters_after = master_weight_snapshot(model)
    master_update = master_weight_delta(masters_before, masters_after)
    print(f"Master weights updated: {master_update['digest_changed']} "
          f"(|W| sum delta {master_update['abs_sum_delta']:+.2f}, "
          f"binary fraction {master_update['fraction_exactly_pm1_after']:.2e})")

    n_eval = training.get("eval_utterances", eval_cfg["subset_size"]) if args.smoke else eval_cfg["subset_size"]
    samples = load_dev_subset(subset_size=eval_cfg["subset_size"], seed=eval_cfg["seed"])[:n_eval]
    print(f"Evaluating on {len(samples)} dev utterances (pinned protocol) ...")
    model.eval()
    metrics = evaluate_model(model, processor, samples)

    # FP32 masters, not a packed artifact. `pack_qat_model.py` builds the
    # deployable 1-bit file from this checkpoint. Record the layer selection and
    # scale mode so the loader can verify them.
    manifest = save_qat_checkpoint(
        model, checkpoint_dir,
        layers=cfg["layers"], scale_mode=scale_mode, ste_clip=ste_clip,
        config_path=cfg.get("_config_path"), init_from=student_id,
        exclude_layers=exclude_layers,
    )
    reloaded, _ = load_qat_checkpoint(
        checkpoint_dir, Wav2Vec2ConformerForCTC, cfg["layers"],
        scale_mode=scale_mode, ste_clip=ste_clip,
        exclude_layers=exclude_layers,
    )
    model_device = next(model.parameters()).device
    reloaded = reloaded.to(model_device)
    reloaded.eval()
    inp = processor(samples[0]["audio"], sampling_rate=16_000,
                    return_tensors="pt").input_values.to(model_device)
    with torch.no_grad():
        logits_trained = model(inp).logits
        logits_reloaded = reloaded(inp).logits
    logit_max_abs_diff = float((logits_trained - logits_reloaded).abs().max())
    argmax_agreement = float(
        (logits_trained.argmax(dim=-1) == logits_reloaded.argmax(dim=-1)).float().mean()
    )
    save_load_ok = bool(
        torch.allclose(logits_trained, logits_reloaded, atol=1e-1, rtol=1e-1)
        and argmax_agreement >= 0.99
    )

    def load_result(path: Path) -> dict:
        return json.loads(path.read_text()) if path.is_file() else {}

    baseline = load_result(RESULTS / "baseline.json")
    ptq = load_result(RESULTS / "ptq_binary.json")
    qat_attn_ffn = load_result(RESULTS / "qat_attn_ffn.json")
    qat_attn = load_result(RESULTS / "qat_attn_a100.json")
    baseline_wer = baseline.get("wer")
    ptq_wer = ptq.get("wer")
    qat_af_wer = qat_attn_ffn.get("eval", {}).get("wer")
    qat_a_wer = qat_attn.get("eval", {}).get("wer")

    result = {
        "phase": "Phase 13 knowledge distillation (smoke)" if args.smoke
        else "Phase 13 knowledge distillation (full)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "compute_profile": "cloud-gpu" if torch.cuda.is_available() else "local",
        "config_profile": cfg.get("compute_profile"),
        "git_commit": git_commit(),
        "config_path": cfg.get("_config_path"),
        "model_id": student_id,
        "init_from": student_id,
        "teacher": {
            "model_id": teacher_id,
            "frozen": True,
            "precision": "fp32",
            "logits_cache": cache_info,
            "vocab_size": vocab,
        },
        "student": {
            "layers": cfg["layers"],
            "quantization": {
                "scale": scale_mode,
                "ste": quant_cfg.get("ste"),
                "ste_clip": ste_clip,
                "exclude_layers": exclude_layers,
                "mixed_precision": bool(exclude_layers),
            },
            "replacement": {
                "n_replaced": replacement["n_replaced"],
                "n_retained": replacement["n_retained"],
                "n_excluded": replacement["n_excluded"],
                "replaced_params": replacement["replaced_params"],
                "retained_params": replacement["retained_params"],
                "excluded_params": replacement["excluded_params"],
                "exclude_layers": replacement["exclude_layers"],
                "replaced_categories": sorted({e["category"] for e in replacement["replaced"]}),
                "retained_categories": sorted({e["category"] for e in replacement["retained"]}),
            },
            "freeze": freeze,
        },
        "distillation": {
            "lambda_ctc": float(distill_cfg.get("lambda_ctc", 0.5)),
            "lambda_kd": float(distill_cfg.get("lambda_kd", 0.5)),
            "temperature": float(distill_cfg.get("temperature", 2.0)),
            "loss": "lambda_ctc*CTC + lambda_kd*T^2*KL(student/T || teacher/T), KL summed over valid frames",
            "teacher_logits_precomputed": True,
        },
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
        "loss_components": trainer.loss_summary(),
        "distillation_probe": probe,
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
            "vs_qat_attn_ffn": {
                "wer": qat_af_wer,
                "delta_wer": (metrics["wer"] - qat_af_wer) if qat_af_wer is not None else None,
            },
            "vs_qat_attn_a100": {
                "wer": qat_a_wer,
                "delta_wer": (metrics["wer"] - qat_a_wer) if qat_a_wer is not None else None,
            },
        },
        "checkpoint": {
            "path": str(checkpoint_dir),
            "format": CHECKPOINT_FORMAT,
            "is_1bit_packed": False,
            "manifest": manifest,
            "save_load_verified": save_load_ok,
            "save_load_argmax_agreement": argmax_agreement,
            "save_load_max_abs_logit_diff": logit_max_abs_diff,
        },
        "versions": {
            "torch": pkg_version("torch"),
            "transformers": pkg_version("transformers"),
            "datasets": pkg_version("datasets"),
            "jiwer": pkg_version("jiwer"),
            "numpy": pkg_version("numpy"),
            "safetensors": pkg_version("safetensors"),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print_summary(metrics)
    print(f"save/load verified: {save_load_ok}")
    print(f"WER (distillation): {metrics['wer']:.4f} | vs FP32: {baseline_wer} "
          f"| vs QAT attn+FFN: {qat_af_wer}")
    print(f"Saved to {output}")
    # As in QAT: exit 1 is a save/load diagnostic, not "the experiment failed".
    # The cluster runner still packs whenever the result JSON exists.
    return 0 if save_load_ok else 1


if __name__ == "__main__":
    sys.exit(main())
