"""Phase 3 — FP32 baseline evaluation (pinned protocol).

- Fixed 256-utterance LibriSpeech clean-validation subset (fixed seed)
- Greedy CTC decoding, batch size 1, CPU
- Lowercase + strip-punctuation normalization before WER/CER
- Records latency (mean/p50/p95), RTF, throughput, peak RSS, sizes
- Writes machine-readable results/baseline.json (the reference for all
  later experiments — never hand-edit).

Usage:
    python scripts/eval_baseline.py [--config configs/baseline.yaml]
"""

import argparse
import datetime
import json
import platform
import statistics
import time
from pathlib import Path

import psutil
import torch
import yaml
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.data.text import normalize_pair
from onebit_asr.evaluation.metrics import compute_cer, compute_wer
from onebit_asr.evaluation.model_size import (
    checkpoint_size_bytes,
    count_parameters,
    fp32_weight_bytes,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def pkg_version(module: str, dist: str | None = None) -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version(dist or module)
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="FP32 baseline evaluation.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default=str(RESULTS / "baseline.json"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_id = cfg["model"]
    eval_cfg = cfg["eval"]
    subset_size, seed = eval_cfg["subset_size"], eval_cfg["seed"]
    assert eval_cfg["decoding"] == "greedy", "Phase 3 pins greedy decoding"

    print(f"Loading {model_id} (CPU) ...")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
    model.eval()
    total_params, _ = count_parameters(model)
    ckpt_bytes, ckpt_files = checkpoint_size_bytes(model_id)

    print(f"Loading dev subset (n={subset_size}, seed={seed}) ...")
    samples = load_dev_subset(subset_size=subset_size, seed=seed)

    proc = psutil.Process()
    peak_rss = 0
    refs, hyps, details = [], [], []
    latencies, audio_secs = [], []
    total_proc = 0.0

    for i, s in enumerate(samples):
        inputs = processor(s["audio"], sampling_rate=16_000,
                           return_tensors="pt", padding=True)
        audio_sec = len(s["audio"]) / 16_000
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        hyp = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
        dt = time.perf_counter() - t0

        ref_n, hyp_n = normalize_pair(s["text"], hyp)
        refs.append(ref_n)
        hyps.append(hyp_n)
        latencies.append(dt)
        audio_secs.append(audio_sec)
        total_proc += dt
        peak_rss = max(peak_rss, proc.memory_info().rss)
        details.append({"id": s["id"], "reference": ref_n, "hypothesis": hyp_n,
                        "latency_s": dt, "audio_s": audio_sec})
        if (i + 1) % 32 == 0:
            print(f"  {i + 1}/{len(samples)} ...")

    total_audio = sum(audio_secs)
    result = {
        "phase": "Phase 3 FP32 baseline",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "compute_profile": cfg.get("compute_profile", "local"),
        "model_id": model_id,
        "dataset": {"id": "openslr/librispeech_asr", "config": "clean",
                    "split": "validation", "subset_size": subset_size, "seed": seed},
        "decoding": "greedy",
        "text_normalization": "lowercase + strip punctuation",
        "sample_ids": [d["id"] for d in details],
        "n_utterances": len(details),
        "wer": compute_wer(refs, hyps),
        "cer": compute_cer(refs, hyps),
        "total_audio_s": total_audio,
        "total_processing_s": total_proc,
        "rtf": total_proc / total_audio,
        "throughput_utt_per_s": len(details) / total_proc,
        "throughput_audio_s_per_s": total_audio / total_proc,
        "latency_s": {"mean": statistics.fmean(latencies),
                      "p50": percentile(latencies, 50),
                      "p95": percentile(latencies, 95),
                      "max": max(latencies)},
        "peak_rss_mb": peak_rss / 1e6,
        "total_params": total_params,
        "fp32_weight_bytes": fp32_weight_bytes(total_params),
        "checkpoint_bytes": ckpt_bytes,
        "checkpoint_files": [Path(f).name for f in ckpt_files],
        "versions": {
            "torch": pkg_version("torch"),
            "transformers": pkg_version("transformers"),
            "datasets": pkg_version("datasets"),
            "jiwer": pkg_version("jiwer"),
        },
        "utterances": details,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("=" * 64)
    print(f"WER: {result['wer']:.4f}  |  CER: {result['cer']:.4f}")
    print(f"RTF: {result['rtf']:.3f}  |  p50 latency: {result['latency_s']['p50']:.2f}s"
          f"  |  peak RSS: {result['peak_rss_mb']:.0f} MB")
    print(f"Params: {total_params:,}  |  checkpoint: {ckpt_bytes / 1e9:.2f} GB")
    print(f"Saved to {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
