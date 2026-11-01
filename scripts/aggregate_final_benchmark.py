"""Aggregate the three final benchmark arms into one comparison."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results/final_benchmark"
FILES = {
    "FP32 reference": DIR / "fp32.json",
    "2-bit FFN": DIR / "ffn2bit.json",
    "Strategy A": DIR / "strategy_a.json",
}
FULL_TEST = {
    "FP32 reference": 0.019267346317711504,
    "2-bit FFN": 0.025829275715155205,
    "Strategy A": 0.027369902617163724,
}
HISTORICAL_ARTIFACT = {
    "FP32 reference": 2373507712,
    "2-bit FFN": 1131062436,
    "Strategy A": 989509572,
}


def main() -> int:
    arms = {name: json.loads(path.read_text()) for name, path in FILES.items()}
    fp32 = arms["FP32 reference"]
    rows = []
    for name, result in arms.items():
        rows.append({
            "model": name,
            "dev_wer": result["wer"],
            "dev_cer": result["cer"],
            "expected_wer_delta": result.get("expected_wer_delta"),
            "full_test_wer": FULL_TEST[name],
            "artifact_bytes": result["artifact_bytes"],
            "historical_artifact_bytes": HISTORICAL_ARTIFACT[name],
            "storage_vs_fp32": result["artifact_bytes"] / fp32["artifact_bytes"],
            "model_load_s": result["model_load_s"],
            "peak_host_rss_mb": result["peak_host_rss_mb"],
            "peak_gpu_allocated_bytes": result["inference_peak_gpu_allocated_bytes"],
            "forward_mean_s": result["forward_latency_s"]["mean"],
            "forward_p95_s": result["forward_latency_s"]["p95"],
            "e2e_mean_s": result["e2e_latency_s"]["mean"],
            "e2e_p95_s": result["e2e_latency_s"]["p95"],
            "acoustic_rtf": result["acoustic_rtf"],
            "e2e_rtf": result["e2e_rtf"],
            "forward_slowdown_vs_fp32": (
                result["forward_latency_s"]["mean"] / fp32["forward_latency_s"]["mean"]
            ),
            "e2e_slowdown_vs_fp32": (
                result["e2e_latency_s"]["mean"] / fp32["e2e_latency_s"]["mean"]
            ),
            "load_info": result.get("load_info"),
            "transcripts": result.get("transcripts"),
        })
    payload = {
        "phase": "final controlled deployment benchmark (dev-256)",
        "protocol": {
            "split": "dev-clean",
            "n": 256,
            "seed": 42,
            "decoding": "greedy",
            "text_normalization": "lowercase + strip punctuation",
            "device": fp32.get("device_name"),
            "test_used": False,
        },
        "artifact_note": (
            "v2 artifacts add BatchNorm/positional buffers so they are standalone. "
            "Historical artifact bytes remain recorded separately. Packed runtime "
            "expands codes to FP weights before F.linear; this is storage "
            "compression, not inference acceleration."
        ),
        "arms": rows,
        "conclusion": {
            "accuracy_order": "FP32 > 2-bit FFN > Strategy A",
            "storage_order": "Strategy A < 2-bit FFN < FP32",
            "inference_acceleration_claim": False,
            "reason": (
                "PackedBitLinear unpacks 2-bit codes and expands FP weights on every "
                "forward; the measured forward/e2e latency is compared, not claimed "
                "as a speedup."
            ),
        },
    }
    out = ROOT / "results/final_benchmark.json"
    out.write_text(json.dumps(payload, indent=2))
    md = ["| Model | Dev WER | Dev CER | Full test WER | Artifact | Historical | Load s | Peak RAM | Forward mean | E2E mean |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        md.append(
            f"| {row['model']} | {row['dev_wer']*100:.4f}% | {row['dev_cer']*100:.4f}% | "
            f"{row['full_test_wer']*100:.2f}% | {row['artifact_bytes']:,} B | "
            f"{row['historical_artifact_bytes']:,} B | {row['model_load_s']:.2f} | "
            f"{row['peak_host_rss_mb']:.0f} MB | {row['forward_mean_s']*1000:.2f} ms | "
            f"{row['e2e_mean_s']*1000:.2f} ms |"
        )
    (ROOT / "results/final_benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
