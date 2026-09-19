"""Model-agnostic evaluation runner (Phase 9 refactor).

Extracted from the Phase 3 baseline script so that *any* in-memory model
(FP32 baseline or a binarized/PTQ model) can be measured under the exact same
pinned protocol:

- greedy CTC decoding (argmax, no language model)
- lowercase + strip-punctuation normalization before WER/CER
- caller supplies the fixed dev subset (same 256 utterances, same seed)
- batch size 1, CPU, per-utterance latency + peak RSS

The evaluator never loads a model itself, so it works for loaded FP32 models,
in-memory BitLinear replacements, or (in future) QAT checkpoints.
"""

from __future__ import annotations

import statistics
import time

import psutil
import torch

from onebit_asr.data.text import normalize_pair
from onebit_asr.evaluation.metrics import compute_cer, compute_wer
from onebit_asr.evaluation.model_size import count_parameters, fp32_weight_bytes


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (no numpy dependency)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def evaluate_model(
    model,
    processor,
    samples: list[dict],
    *,
    decoding: str = "greedy",
    progress_every: int = 32,
    verbose: bool = True,
) -> dict:
    """Evaluate ``model`` on ``samples`` under the pinned protocol.

    Args:
        model: any callable model returning ``.logits`` (HF CTC models and
            binarized variants both qualify).
        processor: HF ``Wav2Vec2Processor`` used for feature extraction and
            decoding.
        samples: list of ``{"id", "audio", "text"}`` dicts (e.g. from
            :func:`onebit_asr.data.dataset.load_dev_subset`).
        decoding: only ``"greedy"`` is supported (the pinned protocol).
        progress_every: print a progress line every N utterances (0 disables).
        verbose: print progress / summary lines.

    Returns:
        Metrics dict (WER/CER, latency, RTF, throughput, peak RSS, params).
    """
    if decoding != "greedy":
        raise ValueError(f"pinned protocol requires greedy decoding, got {decoding!r}")

    # The pinned protocol is inference: guarantee eval mode (no dropout), even if
    # the caller just finished a training loop that left the model in train mode.
    was_training = model.training
    model.eval()

    # Place inputs on the model's device (GPU QAT training leaves the model on
    # CUDA; the pinned protocol is device-agnostic).
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")

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
            logits = model(inputs.input_values.to(device)).logits
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
        if verbose and progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{len(samples)} ...")

    total_audio = sum(audio_secs)
    total_params, _ = count_parameters(model)
    if was_training:
        model.train()
    return {
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
        "utterances": details,
    }


def print_summary(result: dict) -> None:
    """Print a compact, consistent summary block."""
    print("=" * 64)
    print(f"WER: {result['wer']:.4f}  |  CER: {result['cer']:.4f}")
    print(f"RTF: {result['rtf']:.3f}  |  p50 latency: {result['latency_s']['p50']:.2f}s"
          f"  |  peak RSS: {result['peak_rss_mb']:.0f} MB")
    print(f"Params: {result['total_params']:,}")
    print("=" * 64)
