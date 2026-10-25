"""Benchmark orchestration (Phase 17).

High-level helper that pairs the accuracy runner with the performance runner.
The model is loaded by the caller (see ``scripts/run_final_eval.py``), so this
module stays model-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

from onebit_asr.evaluation.evaluate import evaluate_model, print_summary
from onebit_asr.evaluation.latency import benchmark_model


def benchmark_and_evaluate(
    model,
    processor,
    samples: list[dict],
    *,
    label: str,
    warmup: int = 3,
    verbose: bool = True,
) -> dict:
    """Run the pinned accuracy protocol and the performance protocol."""
    accuracy = evaluate_model(model, processor, samples, verbose=verbose)
    performance = benchmark_model(
        model, processor, samples, warmup=warmup, verbose=verbose
    )
    if verbose:
        print(f"[{label}] accuracy:")
        print_summary(accuracy)
        print(f"[{label}] performance: RTF={performance['rtf']:.4f} "
              f"p50={performance['latency_s']['p50']:.3f}s "
              f"p95={performance['latency_s']['p95']:.3f}s "
              f"device={performance['device']}")
    return {
        "label": label,
        "accuracy": accuracy,
        "performance": performance,
    }


def write_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def comparison_table(rows: list[dict]) -> str:
    """Render a compact markdown comparison table from per-arm reports."""
    header = (
        "| Arm | WER | CER | RTF | p50 (s) | p95 (s) | utt/s | Peak RSS (MB) | Peak VRAM (GB) | Deploy size (GB) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    def _f(value, places=4):
        return f"{value:.{places}f}" if isinstance(value, (int, float)) else "n/a"

    for row in rows:
        acc = row["accuracy"]
        perf = row.get("performance", {})
        size = row.get("size", {})
        vram = perf.get("memory", {}).get("peak_reserved_bytes")
        vram_gb = f"{vram / 1e9:.2f}" if vram else "n/a"
        deploy_bytes = size.get("packed_artifact_components_bytes") or size.get("fp32_weight_bytes")
        deploy_gb = f"{deploy_bytes / 1e9:.3f}" if deploy_bytes else "n/a"
        latency = perf.get("latency_s", {})
        lines.append(
            f"| {row['label']} | {acc['wer']:.4f} | {acc['cer']:.4f} | "
            f"{_f(perf.get('rtf'))} | {_f(latency.get('p50'), 3)} | "
            f"{_f(latency.get('p95'), 3)} | {_f(perf.get('throughput_utt_per_s'), 3)} | "
            f"{_f(perf.get('memory', {}).get('peak_rss_mb'), 0)} | {vram_gb} | {deploy_gb} |\n"
        )
    return "".join(lines)
