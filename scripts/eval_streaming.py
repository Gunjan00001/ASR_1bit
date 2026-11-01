"""Evaluate the online chunked-streaming session on dev-clean.

No training. Every configuration uses the selected packed model, greedy CTC,
the same dev subset, and the same online session code as the browser backend.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import time
from pathlib import Path

import jiwer
import numpy as np
import torch

from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.serving.manager import ModelManager
from onebit_asr.data.text import normalize_text
from onebit_asr.serving.metrics import compute_reference_metrics
from onebit_asr.serving.streaming import StreamingSession

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [
    ("app-default-2.0s", 2.0, 0.5),
    ("1.0s-window", 1.0, 0.25),
    ("320ms-window", 0.32, 0.08),
    ("3.0s-window", 3.0, 0.75),
]


def run_config(manager, model_id, samples, chunk_ms, window_s, overlap_s):
    rows = []
    chunks = int(16000 * chunk_ms / 1000)
    for sample in samples:
        session = StreamingSession(
            manager,
            model_id,
            sample_rate=16000,
            chunk_ms=chunk_ms,
            window_s=window_s,
            overlap_s=overlap_s,
        )
        audio = np.asarray(sample["audio"], dtype=np.float32)
        for start in range(0, len(audio), chunks):
            session.append_pcm(audio[start:start + chunks].astype("<f4").tobytes())
        final = session.finalize(sample["text"])
        ref = sample["text"]
        rows.append({
            "id": sample["id"],
            "reference": ref,
            "final": final["refined_text"],
            "streaming": final["streaming_text"],
            "final_metrics": final["metrics"],
            "streaming_metrics": final["streaming_metrics"],
            "first_partial_latency_s": final["first_partial_latency_s"],
            "average_update_interval_s": final["average_update_interval_s"],
            "finalization_latency_s": final["finalization_latency_s"],
            "processing_s": final["processing_s"],
            "rtf": final["rtf"],
            "partial_updates": final["partial_updates"],
            "chunks_received": final["chunks_received"],
            "dropped_chunks": final["dropped_chunks"],
            "audio_duration_s": final["audio_duration_s"],
        })
    refs = [normalize_text(r["reference"]) for r in rows]
    finals = [normalize_text(r["final"]) for r in rows]
    streams = [normalize_text(r["streaming"]) for r in rows]
    return {
        "config": {"chunk_ms": chunk_ms, "window_s": window_s, "overlap_s": overlap_s},
        "n": len(rows),
        "final_wer": float(jiwer.wer(refs, finals)),
        "final_cer": float(jiwer.cer(refs, finals)),
        "streaming_wer": float(jiwer.wer(refs, streams)),
        "streaming_cer": float(jiwer.cer(refs, streams)),
        "first_partial_latency_s": statistics.fmean(
            r["first_partial_latency_s"] for r in rows if r["first_partial_latency_s"] is not None
        ),
        "average_update_interval_s": statistics.fmean(
            r["average_update_interval_s"] for r in rows if r["average_update_interval_s"] is not None
        ) if any(r["average_update_interval_s"] is not None for r in rows) else None,
        "finalization_latency_s": statistics.fmean(r["finalization_latency_s"] for r in rows),
        "rtf": statistics.fmean(r["rtf"] for r in rows),
        "partial_updates": sum(r["partial_updates"] for r in rows),
        "dropped_chunks": sum(r["dropped_chunks"] for r in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ffn2bit,strategy_a")
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/streaming_eval.json")
    parser.add_argument("--configs", default="app-default-2.0s,1.0s-window,320ms-window,3.0s-window")
    args = parser.parse_args()
    manager = ModelManager(device="auto")
    samples = load_dev_subset(subset_size=args.n, seed=args.seed)
    selected_configs = {name for name in args.configs.split(",") if name}
    result = {
        "phase": "online chunked-streaming evaluation (dev-clean)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "device": str(manager.device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "n": len(samples),
        "test_used": False,
        "configurations": {},
        "note": (
            "Online session code used by the browser backend. Final text is a "
            "full-utterance refinement; streaming text is the chunk-merged hypothesis. "
            "This is not true stateful encoder streaming."
        ),
    }
    for model_id in [x for x in args.models.split(",") if x]:
        load = manager.load(model_id)
        model_result = {"load": load, "configs": {}}
        for name, window, overlap in CONFIGS:
            if name not in selected_configs:
                continue
            print(model_id, name, flush=True)
            model_result["configs"][name] = run_config(
                manager, model_id, samples, 320, window, overlap
            )
        if torch.cuda.is_available():
            model_result["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        result["configurations"][model_id] = model_result
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({m: {n: {k: v[k] for k in ("final_wer", "final_cer", "streaming_wer", "streaming_cer", "first_partial_latency_s", "average_update_interval_s", "finalization_latency_s", "rtf", "partial_updates", "dropped_chunks")} for n, v in d["configs"].items()} for m, d in result["configurations"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
