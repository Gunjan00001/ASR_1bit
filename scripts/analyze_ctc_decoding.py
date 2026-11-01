"""Error analysis and decoder comparison for Strategy A on dev-clean.

This script is evaluation-only. It loads a QAT checkpoint, runs one forward pass
per utterance, and compares greedy, blank-penalized greedy, and prefix beam
search on exactly the same logits. It also records CTC blank statistics and
word/character error breakdowns. Test data is never used.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np
import psutil
import torch
import torchaudio
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.data.text import normalize_text
from onebit_asr.decoding.decode import greedy_decode, prefix_beam_search
from onebit_asr.training.qat import load_qat_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def decode_ids(processor, ids: list[int]) -> str:
    # The tokenizer's default group_tokens=True applies CTC collapsing again.
    # These IDs are already collapsed, so grouping would delete every doubled
    # letter (e.g. "tolled" -> "told").
    return normalize_text(
        processor.tokenizer.decode(ids, skip_special_tokens=True, group_tokens=False)
    )


def word_errors(refs: list[str], hyps: list[str]) -> tuple[int, int, int]:
    processed = jiwer.process_words(refs, hyps)
    return processed.substitutions, processed.deletions, processed.insertions


def confusion_counts(refs: list[str], hyps: list[str]) -> dict:
    substitutions, deletions, insertions = Counter(), Counter(), Counter()
    for ref, hyp in zip(refs, hyps):
        alignment = jiwer.process_words([ref], [hyp]).alignments[0]
        ref_tokens, hyp_tokens = ref.split(), hyp.split()
        for chunk in alignment:
            if chunk.type == "substitute":
                r = " ".join(ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx])
                h = " ".join(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx])
                substitutions[(r, h)] += 1
            elif chunk.type == "delete":
                for token in ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx]:
                    deletions[token] += 1
            elif chunk.type == "insert":
                for token in hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                    insertions[token] += 1
    return {
        "substitutions": [
            {"reference": r, "hypothesis": h, "count": n}
            for (r, h), n in substitutions.most_common(30)
        ],
        "deletions": [{"word": w, "count": n} for w, n in deletions.most_common(25)],
        "insertions": [{"word": w, "count": n} for w, n in insertions.most_common(25)],
    }


def quartile_bins(rows: list[dict], key: str) -> list[dict]:
    values = sorted(row[key] for row in rows)
    edges = [0.0] + [values[int((len(values) - 1) * q)] for q in (0.25, 0.5, 0.75)]
    edges.append(max(values) + 1e-9)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        group = [row for row in rows if lo <= row[key] < hi]
        if not group:
            continue
        words = sum(row["ref_words"] for row in group)
        errors = sum(row["S"] + row["D"] + row["I"] for row in group)
        out.append({
            "lo": lo,
            "hi": hi,
            "n": len(group),
            "words": words,
            "errors": errors,
            "wer": errors / words,
            "mean_utterance_wer": statistics.fmean(row["wer"] for row in group),
        })
    return out


def forced_alignment(log_probs: torch.Tensor, target_ids: list[int]) -> dict | None:
    if not target_ids or len(target_ids) > log_probs.shape[0]:
        return None
    try:
        targets = torch.tensor([target_ids], dtype=torch.int32)
        alignment, _scores = torchaudio.functional.forced_align(
            log_probs.unsqueeze(0), targets, blank=0
        )
        path = alignment[0]
        blank_fraction = float((path == 0).float().mean())
        durations = [int((path == i).sum()) for i in range(len(target_ids))]
        return {
            "blank_fraction": blank_fraction,
            "target_tokens": len(target_ids),
            "mean_target_duration": float(np.mean(durations)),
            "min_target_duration": int(min(durations)),
        }
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Dev-only CTC error and decoder audit.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam-widths", default="10,50")
    parser.add_argument("--blank-penalties", default="-1,0.5,1,2")
    parser.add_argument("--expected-greedy-wer", type=float, default=None,
                        help="abort if the greedy baseline does not reproduce this WER")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    quant = cfg.get("quantization", {})
    checkpoint = ROOT / args.checkpoint
    processor = Wav2Vec2Processor.from_pretrained(cfg["qat"]["init_from"])
    model, _report = load_qat_checkpoint(
        checkpoint,
        Wav2Vec2ConformerForCTC,
        cfg["layers"],
        scale_mode=quant.get("scale", "per_tensor_mean_abs"),
        ste_clip=quant.get("ste_clip"),
        exclude_layers=list(quant.get("exclude_layers") or []),
        bits=int(quant.get("bits", 1)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    samples = load_dev_subset(subset_size=args.n, seed=args.seed)
    beam_widths = [int(x) for x in args.beam_widths.split(",") if x.strip()]
    blank_penalties = [float(x) for x in args.blank_penalties.split(",") if x.strip()]

    decoder_names = ["greedy"] + [f"greedy_blank_{b:g}" for b in blank_penalties]
    decoder_names += [f"beam_{w}" for w in beam_widths]
    decoder_outputs = {name: [] for name in decoder_names}
    decoder_latencies = {name: [] for name in decoder_names}
    refs, rows = [], []
    proc = psutil.Process()
    peak_rss_mb = 0.0
    forward_s = 0.0

    for index, sample in enumerate(samples):
        inputs = processor(sample["audio"], sampling_rate=16_000,
                           return_tensors="pt", padding=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits[0].float().cpu()
        forward_s += time.perf_counter() - t0
        ref = normalize_text(sample["text"])
        refs.append(ref)

        def record(name: str, decode_fn) -> str:
            t0 = time.perf_counter()
            ids = decode_fn()
            text = decode_ids(processor, ids)
            t2 = time.perf_counter()
            decoder_latencies[name].append(t2 - t0)
            decoder_outputs[name].append(text)
            return text

        greedy_text = record("greedy", lambda: greedy_decode(logits))
        for beta in blank_penalties:
            record(
                f"greedy_blank_{beta:g}",
                lambda beta=beta: greedy_decode(logits, blank_penalty=beta),
            )
        for width in beam_widths:
            record(
                f"beam_{width}",
                lambda width=width: prefix_beam_search(
                    logits, beam_width=width, nbest=1
                )[0][1],
            )

        greedy_words = jiwer.process_words([ref], [greedy_text])
        greedy_chars = jiwer.process_characters([ref], [greedy_text])
        ref_words = len(ref.split())
        char_count = len(ref.replace(" ", ""))
        probs = torch.softmax(logits, dim=-1)
        argmax = logits.argmax(dim=-1)
        ref_tokens = processor.tokenizer(ref.upper(), add_special_tokens=False).input_ids
        align = forced_alignment(torch.log_softmax(logits, dim=-1), ref_tokens)
        rows.append({
            "id": sample["id"],
            "reference": ref,
            "hypothesis": greedy_text,
            "audio_s": len(sample["audio"]) / 16_000,
            "ref_words": ref_words,
            "ref_chars": char_count,
            "S": greedy_words.substitutions,
            "D": greedy_words.deletions,
            "I": greedy_words.insertions,
            "H": greedy_words.hits,
            "wer": greedy_words.wer,
            "cer": greedy_chars.cer,
            "blank_fraction": float((argmax == 0).float().mean()),
            "mean_max_probability": float(probs.max(dim=-1).values.mean()),
            "emitted_tokens": len(greedy_decode(logits)),
            "forced_alignment": align,
        })
        peak_rss_mb = max(peak_rss_mb, proc.memory_info().rss / 1e6)
        if not args.no_progress and (index + 1) % 32 == 0:
            print(f"  {index + 1}/{len(samples)}", flush=True)

    results = {}
    for name in decoder_names:
        hyps = decoder_outputs[name]
        w = jiwer.process_words(refs, hyps)
        c = jiwer.process_characters(refs, hyps)
        lat = decoder_latencies[name]
        results[name] = {
            "wer": float(w.wer),
            "cer": float(c.cer),
            "substitutions": int(w.substitutions),
            "deletions": int(w.deletions),
            "insertions": int(w.insertions),
            "hits": int(w.hits),
            "decode_latency_s": {
                "mean": statistics.fmean(lat),
                "p50": float(np.percentile(lat, 50)),
                "p95": float(np.percentile(lat, 95)),
                "total": float(sum(lat)),
            },
            "hypotheses": hyps,
        }

    greedy = results["greedy"]
    if args.expected_greedy_wer is not None:
        if abs(greedy["wer"] - args.expected_greedy_wer) > 1e-6:
            raise SystemExit(
                "greedy baseline did not reproduce the pinned WER: "
                f"got {greedy['wer']:.12f}, expected {args.expected_greedy_wer:.12f}"
            )
    total_errors = greedy["substitutions"] + greedy["deletions"] + greedy["insertions"]
    total_words = greedy["hits"] + greedy["substitutions"] + greedy["deletions"]
    sorted_errors = sorted((row["S"] + row["D"] + row["I"] for row in rows), reverse=True)
    error_concentration = {
        "utterances_with_errors": sum(1 for row in rows if row["S"] + row["D"] + row["I"] > 0),
        "perfect_utterances": sum(1 for row in rows if row["S"] + row["D"] + row["I"] == 0),
        "top10_error_share": sum(sorted_errors[:10]) / total_errors if total_errors else 0.0,
        "top20_error_share": sum(sorted_errors[:20]) / total_errors if total_errors else 0.0,
        "top50_error_share": sum(sorted_errors[:50]) / total_errors if total_errors else 0.0,
    }
    align_rows = [row for row in rows if row["forced_alignment"]]
    correlations = {}
    for key in ("audio_s", "ref_words", "blank_fraction", "mean_max_probability"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        wers = np.asarray([row["wer"] for row in rows], dtype=np.float64)
        correlations[key] = float(np.corrcoef(values, wers)[0, 1]) if values.std() > 0 else None
    forced_align_summary = None
    if align_rows:
        forced_align_summary = {
            "n": len(align_rows),
            "correlation_blank_fraction_with_wer": float(
                np.corrcoef(
                    [r["forced_alignment"]["blank_fraction"] for r in align_rows],
                    [r["wer"] for r in align_rows],
                )[0, 1]
            ),
            "mean_target_duration": statistics.fmean(
                r["forced_alignment"]["mean_target_duration"] for r in align_rows
            ),
            "min_target_duration": min(
                r["forced_alignment"]["min_target_duration"] for r in align_rows
            ),
        }

    payload = {
        "phase": "dev-only CTC error analysis and decoder audit",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": "dev-clean",
        "seed": args.seed,
        "n": len(samples),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "forward_seconds": forward_s,
        "forward_rtf": forward_s / sum(row["audio_s"] for row in rows),
        "peak_rss_mb": peak_rss_mb,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "decoders": {
            name: {k: v for k, v in result.items() if k != "hypotheses"}
            for name, result in results.items()
        },
        "per_utterance": rows,
        "error_breakdown": {
            "words": total_words,
            "word_errors": total_errors,
            "substitution_rate": greedy["substitutions"] / total_errors if total_errors else 0.0,
            "deletion_rate": greedy["deletions"] / total_errors if total_errors else 0.0,
            "insertion_rate": greedy["insertions"] / total_errors if total_errors else 0.0,
            "concentration": error_concentration,
        },
        "confusions": confusion_counts(refs, decoder_outputs["greedy"]),
        "by_duration": quartile_bins(rows, "audio_s"),
        "by_reference_length": quartile_bins(rows, "ref_words"),
        "correlations": correlations,
        "forced_alignment": forced_align_summary,
        "decoder_hypotheses": {name: result["hypotheses"] for name, result in results.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({name: {k: v for k, v in result.items() if k not in ("hypotheses", "decode_latency_s")}
                      for name, result in results.items()}, indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
