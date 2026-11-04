"""Evaluate stateful bounded-context streaming against offline inference.

Dev-clean only. This script uses the same StatefulStreamingSession as the
browser backend, captures current-segment logits, and compares them with one
offline encoder pass. It reports the approximation rather than assuming exact
equivalence.
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
from onebit_asr.data.text import normalize_text
from onebit_asr.serving.manager import ModelManager
from onebit_asr.serving.metrics import compute_reference_metrics
from onebit_asr.serving.stateful_streaming import StatefulStreamingSession

ROOT = Path(__file__).resolve().parent.parent


def metrics(ref: str, hyp: str) -> dict | None:
    return compute_reference_metrics(ref, hyp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ffn2bit,strategy_a")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--current-s", type=float, default=1.0)
    parser.add_argument("--left-s", type=float, default=0.8)
    parser.add_argument("--right-s", type=float, default=0.32)
    parser.add_argument("--chunk-ms", type=int, default=320)
    parser.add_argument("--output", default="results/stateful_eval_cpu.json")
    args = parser.parse_args()
    manager = ModelManager(device="auto")
    samples = load_dev_subset(subset_size=args.n, seed=args.seed)
    result = {
        "phase": "stateful bounded-context streaming evaluation (dev-clean)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "device": str(manager.device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": args.seed,
        "n": len(samples),
        "test_used": False,
        "config": {
            "chunk_ms": args.chunk_ms,
            "current_s": args.current_s,
            "left_s": args.left_s,
            "right_s": args.right_s,
        },
        "models": {},
    }
    for model_id in [x for x in args.models.split(",") if x]:
        load = manager.load(model_id)
        rows = []
        for sample in samples:
            session = StatefulStreamingSession(
                manager, model_id, manager.tokenizer(),
                chunk_ms=args.chunk_ms,
                current_s=args.current_s,
                left_s=args.left_s,
                right_s=args.right_s,
                capture_logits=True,
            )
            events = []
            audio = np.asarray(sample["audio"], dtype=np.float32)
            chunk = int(16000 * args.chunk_ms / 1000)
            for seq, start in enumerate(range(0, len(audio), chunk)):
                events.extend(session.append_pcm(
                    audio[start:start + chunk].astype("<f4").tobytes(), sequence=seq
                ))
            final = session.finalize(sample["text"])
            offline_logits = manager.logits_array(audio, model_id)["logits"]
            stream_logits = (
                np.concatenate(session.logits_chunks, axis=0)
                if session.logits_chunks else np.zeros((0, offline_logits.shape[1]))
            )
            n = min(len(offline_logits), len(stream_logits))
            max_diff = float(np.max(np.abs(offline_logits[:n] - stream_logits[:n]))) if n else None
            mean_diff = float(np.mean(np.abs(offline_logits[:n] - stream_logits[:n]))) if n else None
            agreement = (
                float(np.mean(np.argmax(offline_logits[:n], axis=-1) ==
                              np.argmax(stream_logits[:n], axis=-1)))
                if n else None
            )
            partial_events = [e for e in events if e.get("type") == "partial"]
            last_partial = partial_events[-1] if partial_events else {}
            offline_text = manager.transcribe_array(audio, model_id)["transcript"]
            row = {
                "id": sample["id"],
                "reference": normalize_text(sample["text"]),
                "final": final["text"],
                "last_partial": last_partial.get("text", ""),
                "stable_prefix": last_partial.get("final_text", ""),
                "final_metrics": final["metrics"],
                "partial_metrics": metrics(sample["text"], last_partial.get("text", "")),
                "stable_metrics": metrics(sample["text"], last_partial.get("final_text", "")),
                "offline": offline_text,
                "offline_metrics": metrics(sample["text"], offline_text),
                "first_partial_latency_s": final["first_partial_latency_s"],
                "average_update_interval_s": final["average_update_interval_s"],
                "p95_update_interval_s": final["p95_update_interval_s"],
                "finalization_latency_s": final["finalization_latency_s"],
                "processing_s": final["processing_s"],
                "rtf": final["rtf"],
                "encoder_calls": final["encoder_calls"],
                "encoder_recompute_ratio": final["encoder_recompute_ratio"],
                "committed_words": final["committed_words"],
                "revisions": final["revisions"],
                "dropped_chunks": final["dropped_chunks"],
                "audio_duration_s": final["audio_duration_s"],
                "max_abs_logit_diff": max_diff,
                "mean_abs_logit_diff": mean_diff,
                "frame_argmax_agreement": agreement,
                "stream_frames": int(stream_logits.shape[0]),
                "offline_frames": int(offline_logits.shape[0]),
            }
            rows.append(row)
            print(model_id, sample["id"], "final", row["final"], "rtf", row["rtf"], flush=True)
        refs = [r["reference"] for r in rows]
        final = [normalize_text(r["final"]) for r in rows]
        partial = [normalize_text(r["last_partial"]) for r in rows]
        stable = [normalize_text(r["stable_prefix"]) for r in rows]
        offline = [normalize_text(r["offline"]) for r in rows]
        result["models"][model_id] = {
            "load": load,
            "rows": rows,
            "final_wer": float(jiwer.wer(refs, final)),
            "final_cer": float(jiwer.cer(refs, final)),
            "partial_wer": float(jiwer.wer(refs, partial)),
            "partial_cer": float(jiwer.cer(refs, partial)),
            "stable_prefix_wer": float(jiwer.wer(refs, stable)),
            "stable_prefix_cer": float(jiwer.cer(refs, stable)),
            "offline_wer": float(jiwer.wer(refs, offline)),
            "offline_cer": float(jiwer.cer(refs, offline)),
            "first_partial_latency_s": statistics.fmean(r["first_partial_latency_s"] for r in rows if r["first_partial_latency_s"] is not None),
            "average_update_interval_s": statistics.fmean(r["average_update_interval_s"] for r in rows if r["average_update_interval_s"] is not None) if any(r["average_update_interval_s"] is not None for r in rows) else None,
            "p95_update_interval_s": statistics.fmean(r["p95_update_interval_s"] for r in rows if r["p95_update_interval_s"] is not None) if any(r["p95_update_interval_s"] is not None for r in rows) else None,
            "finalization_latency_s": statistics.fmean(r["finalization_latency_s"] for r in rows),
            "rtf": statistics.fmean(r["rtf"] for r in rows),
            "encoder_recompute_ratio": statistics.fmean(r["encoder_recompute_ratio"] for r in rows),
            "max_abs_logit_diff": max(r["max_abs_logit_diff"] for r in rows if r["max_abs_logit_diff"] is not None) if any(r["max_abs_logit_diff"] is not None for r in rows) else None,
            "frame_argmax_agreement": statistics.fmean(r["frame_argmax_agreement"] for r in rows if r["frame_argmax_agreement"] is not None) if any(r["frame_argmax_agreement"] is not None for r in rows) else None,
            "dropped_chunks": sum(r["dropped_chunks"] for r in rows),
        }
        if torch.cuda.is_available():
            result["models"][model_id]["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({m: {k: v[k] for k in ("final_wer", "partial_wer", "stable_prefix_wer", "offline_wer", "first_partial_latency_s", "average_update_interval_s", "p95_update_interval_s", "finalization_latency_s", "rtf", "encoder_recompute_ratio", "frame_argmax_agreement")} for m, v in result["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY