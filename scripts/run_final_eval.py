"""Phase 12/14/15/17 — final evaluation, sensitivity sweep, and benchmark.

Loads several experiment arms, evaluates each under the *same* pinned protocol,
and benchmarks latency/memory/throughput on the same hardware. Results are
written per arm plus a combined JSON/Markdown table.

Arms
----
- ``fp32``: the untouched FP32 checkpoint (reference).
- ``ptq_attention`` / ``ptq_ffn`` / ``ptq_full`` / ``ptq_full_plus_fproj`` /
  ``ptq_all_linears``: post-training quantization sensitivity variants.
- ``qat_attention_a100`` / ``qat_ffn_a100`` / ``qat_attn_ffn``: QAT checkpoints.
- ``distill_attn_ffn`` / ``distill_attn_ffn_kd01``: distillation checkpoints.

The script skips an arm whose checkpoint is missing (recorded explicitly), so it
can be run before every training job has finished.

Debugging notes
---------------
- **A checkpoint is only complete if its weights file exists.** The Trainer
  creates `output_dir` when a job starts; `is_dir()` alone is not enough. A
  missing/incomplete arm is recorded in `skipped`, not treated as a crash.
- **One bad arm must not abort the table.** The loop catches per-arm exceptions
  and continues; check the `skipped` list in the combined JSON.
- **`--no-benchmark` is accuracy-only.** It is used for the full 2620-utterance
  test-clean so the run does not pay for a second inference pass. Performance
  columns are `n/a` in that table; warmed-up numbers come from the 256 subsets.
- **Do not compare host RSS across arms in one process.** It is a process
  high-water mark; GPU peak memory is reset per arm by the benchmark.

Usage:
    python scripts/run_final_eval.py --split dev --n 256
    python scripts/run_final_eval.py --split test --n 256 \
        --arms fp32,ptq_full,qat_attn_ffn,distill_attn_ffn
    python scripts/run_final_eval.py --split test --n 2620 \
        --arms fp32,qat_attn_ffn,distill_attn_ffn_kd01 --no-benchmark
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset, load_test_subset
from onebit_asr.evaluation.benchmark import benchmark_and_evaluate, comparison_table
from onebit_asr.evaluation.evaluate import evaluate_model, print_summary
from onebit_asr.evaluation.model_size import fp32_weight_bytes, quick_size_report
from onebit_asr.models.replace_layers import replace_linears
from onebit_asr.training.qat import load_qat_checkpoint

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BASE_LAYERS = {
    "attention_q": False,
    "attention_k": False,
    "attention_v": False,
    "attention_output": False,
    "ffn_in": False,
    "ffn_out": False,
    "feature_projection": False,
    "ctc_head": False,
}

ARM_DEFS = {
    "fp32": {"kind": "fp32"},
    "ptq_attention": {"kind": "ptq", "layers": {
        **BASE_LAYERS, "attention_q": True, "attention_k": True,
        "attention_v": True, "attention_output": True}},
    "ptq_ffn": {"kind": "ptq", "layers": {
        **BASE_LAYERS, "ffn_in": True, "ffn_out": True}},
    "ptq_full": {"kind": "ptq", "layers": {
        **BASE_LAYERS, "attention_q": True, "attention_k": True,
        "attention_v": True, "attention_output": True,
        "ffn_in": True, "ffn_out": True}},
    "ptq_full_plus_fproj": {"kind": "ptq", "layers": {
        **BASE_LAYERS, "attention_q": True, "attention_k": True,
        "attention_v": True, "attention_output": True,
        "ffn_in": True, "ffn_out": True, "feature_projection": True}},
    "ptq_all_linears": {"kind": "ptq", "layers": {
        **BASE_LAYERS, "attention_q": True, "attention_k": True,
        "attention_v": True, "attention_output": True,
        "ffn_in": True, "ffn_out": True,
        "feature_projection": True, "ctc_head": True}},
    "qat_attention_a100": {
        "kind": "qat",
        "config": "configs/qat_attn.yaml",
        "checkpoint": "checkpoints/qat_attn_a100",
    },
    "qat_ffn_a100": {
        "kind": "qat",
        "config": "configs/qat_ffn.yaml",
        "checkpoint": "checkpoints/qat_ffn_a100",
    },
    "qat_ffn_control_devsel": {
        "kind": "qat",
        "config": "configs/qat_ffn.yaml",
        "checkpoint": "checkpoints/qat_ffn_control_devsel",
    },
    "qat_ffn_per_channel": {
        "kind": "qat",
        "config": "configs/qat_ffn_per_channel.yaml",
        "checkpoint": "checkpoints/qat_ffn_per_channel",
    },
    "qat_ffn_control_full": {
        "kind": "qat",
        "config": "configs/cluster_effective_qat_ffn_control_full.yaml",
        "checkpoint": "checkpoints/qat_ffn_control_full",
    },
    "qat_ffn_mixed_full": {
        "kind": "qat",
        "config": "configs/cluster_effective_qat_ffn_mixed_full.yaml",
        "checkpoint": "checkpoints/qat_ffn_mixed_full",
    },
    "qat_ffn_mixed_2bit_full": {
        "kind": "qat",
        "config": "configs/cluster_effective_qat_ffn_mixed_2bit_full.yaml",
        "checkpoint": "checkpoints/qat_ffn_mixed_2bit_full",
    },
    "qat_attn_ffn": {
        "kind": "qat",
        "config": "configs/qat_attn_ffn.yaml",
        "checkpoint": "checkpoints/qat_attn_ffn",
    },
    "distill_attn_ffn": {
        "kind": "qat",
        "config": "configs/distill_attn_ffn.yaml",
        "checkpoint": "checkpoints/distill_attn_ffn",
    },
    "distill_attn_ffn_kd01": {
        "kind": "qat",
        "config": "configs/distill_attn_ffn_kd01.yaml",
        "checkpoint": "checkpoints/distill_attn_ffn_kd01",
    },
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def checkpoint_is_complete(path: Path) -> bool:
    """A QAT checkpoint is usable only if its weights file exists.

    The Trainer creates ``output_dir`` as soon as a job starts, so an
    ``is_dir()`` check is not enough: a concurrently training job would look
    like a valid checkpoint and ``from_pretrained`` would fail mid-evaluation.
    """
    return path.is_dir() and any(
        (path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    )


def build_arm(name: str, arm: dict, model_id: str, device: torch.device) -> tuple:
    """Load/reconstruct one arm. Returns ``(model, arm_info)``."""
    if arm["kind"] == "fp32":
        model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
        info = {"kind": "fp32", "layers": None, "checkpoint": None}

    elif arm["kind"] == "ptq":
        cfg = load_config(ROOT / "configs" / "binary.yaml")
        quant = cfg.get("quantization", {})
        model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
        report = replace_linears(
            model, arm["layers"],
            scale_mode=quant.get("scale", "per_tensor_mean_abs"),
            ste_clip=quant.get("ste_clip"),
        )
        info = {
            "kind": "ptq",
            "layers": arm["layers"],
            "scale_mode": quant.get("scale", "per_tensor_mean_abs"),
            "n_replaced": report["n_replaced"],
            "replaced_params": report["replaced_params"],
            "checkpoint": None,
        }

    elif arm["kind"] == "qat":
        cfg_path = ROOT / arm["config"]
        checkpoint = ROOT / arm["checkpoint"]
        if not checkpoint_is_complete(checkpoint):
            raise FileNotFoundError(checkpoint)
        cfg = load_config(cfg_path)
        quant = cfg.get("quantization", {})
        exclude_layers = list(quant.get("exclude_layers") or [])
        bits = int(quant.get("bits", 1))
        model, report = load_qat_checkpoint(
            checkpoint, Wav2Vec2ConformerForCTC, cfg["layers"],
            scale_mode=quant.get("scale", "per_tensor_mean_abs"),
            ste_clip=quant.get("ste_clip"),
            exclude_layers=exclude_layers, bits=bits,
        )
        info = {
            "kind": "qat",
            "layers": cfg["layers"],
            "scale_mode": quant.get("scale", "per_tensor_mean_abs"),
            "bits": bits,
            "exclude_layers": exclude_layers,
            "n_replaced": report["n_replaced"],
            "n_excluded": report["n_excluded"],
            "replaced_params": report["replaced_params"],
            "excluded_params": report["excluded_params"],
            "checkpoint": str(arm["checkpoint"]),
            "config": arm["config"],
        }
    else:
        raise ValueError(f"unknown arm kind {arm['kind']!r}")

    model.to(device)
    model.eval()
    return model, info


def main() -> int:
    parser = argparse.ArgumentParser(description="Final arm evaluation + benchmark.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", default=",".join(ARM_DEFS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-dir", default=str(RESULTS))
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="accuracy only (skip the second latency pass); useful "
                             "for the full 2620-utterance test-clean run")
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_DEFS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}. Choose from {sorted(ARM_DEFS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    baseline_cfg = load_config(ROOT / "configs" / "baseline.yaml")
    model_id = baseline_cfg["model"]
    processor = Wav2Vec2Processor.from_pretrained(model_id)

    print(f"Loading {args.split} subset (n={args.n}, seed={args.seed}) ...")
    if args.split == "dev":
        samples = load_dev_subset(subset_size=args.n, seed=args.seed)
    else:
        samples = load_test_subset(subset_size=args.n, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []

    for name in arms:
        arm = ARM_DEFS[name]
        print("=" * 72)
        print(f"ARM: {name}  {arm}")
        try:
            model, info = build_arm(name, arm, model_id, device)
        except FileNotFoundError as exc:
            print(f"SKIP: checkpoint missing or incomplete ({exc})")
            skipped.append({"arm": name, "reason": f"missing checkpoint: {exc}"})
            continue
        except Exception as exc:  # noqa: BLE001 - one bad arm must not abort the run
            print(f"SKIP: {type(exc).__name__}: {exc}")
            skipped.append({"arm": name, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        if args.no_benchmark:
            # Accuracy-only path: one inference pass, no latency benchmark. Used
            # for the full 2620-utterance test-clean. The performance dict is a
            # placeholder so the comparison table can render `n/a`.
            accuracy = evaluate_model(model, processor, samples,
                                      verbose=not args.no_progress)
            if not args.no_progress:
                print_summary(accuracy)
            report = {
                "label": name,
                "accuracy": accuracy,
                "performance": {
                    "rtf": None,
                    "latency_s": {"p50": None, "p95": None},
                    "throughput_utt_per_s": None,
                    "memory": {"peak_rss_mb": None, "cuda": False},
                    "warmed_up": False,
                    "note": "accuracy-only run (--no-benchmark)",
                },
            }
        else:
            report = benchmark_and_evaluate(
                model, processor, samples, label=name, warmup=args.warmup,
                verbose=not args.no_progress,
            )
        report["arm"] = name
        report["arm_info"] = info
        report["split"] = args.split
        report["subset_size"] = args.n
        report["seed"] = args.seed
        report["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["platform"] = platform.platform()
        report["git_commit"] = git_commit()

        total_params = sum(p.numel() for p in model.parameters())
        if info["kind"] == "fp32":
            size = {
                "total_params": total_params,
                "fp32_weight_bytes": fp32_weight_bytes(total_params),
            }
        else:
            size = {
                "total_params": total_params,
                **quick_size_report(model, scale_mode=info["scale_mode"],
                                    bits=info.get("bits")),
            }
        report["size"] = size

        out = output_dir / f"final_{name}_{args.split}_n{args.n}.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {out}")
        rows.append(report)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined = {
        "phase": "Phase 12/14/15/17 final comparison",
        "split": args.split,
        "subset_size": args.n,
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dataset": "openslr/librispeech_asr clean " + ("validation" if args.split == "dev" else "test"),
        "decoding": "greedy",
        "text_normalization": "lowercase + strip punctuation",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "arms": rows,
        "skipped": skipped,
        "table_markdown": comparison_table(rows),
    }
    combined_path = output_dir / f"final_comparison_{args.split}_n{args.n}.json"
    combined_path.write_text(json.dumps(combined, indent=2))
    md_path = output_dir / f"final_comparison_{args.split}_n{args.n}.md"
    md_path.write_text(combined["table_markdown"] + "\n")
    print("=" * 72)
    print(combined["table_markdown"])
    print(f"Wrote {combined_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
