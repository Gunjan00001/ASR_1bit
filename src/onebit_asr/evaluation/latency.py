"""Inference latency / memory / throughput measurement (Phase 17).

This module is deliberately separate from :mod:`onebit_asr.evaluation.evaluate`:
the evaluator owns the *accuracy* protocol (WER/CER), while this module owns the
*performance* protocol (warmup, synchronized timings, GPU peak memory). Keeping
them separate means adding a benchmark can never silently change the pinned WER
protocol.

Measurement notes
-----------------
- CUDA timings are synchronized around every forward pass.
- The first ``warmup`` utterances are run untimed to exclude CUDA context /
  kernel-autotuning effects; peak memory is reset after warmup.
- Latency is reported both end-to-end (feature extraction + forward + greedy
  decode) and model-only (forward + decode), so preprocessing costs are visible.
- RTF uses end-to-end processing time divided by audio duration.

Debugging notes
---------------
- **Sync is mandatory on CUDA.** Without `torch.cuda.synchronize`, the timer
  measures kernel-launch time instead of execution time.
- **Warmup is mandatory.** The first forward allocates workspaces and selects
  kernels; including it inflates the mean, especially on a shared GPU.
- **The peak memory reset happens after warmup**, so the reported peak is the
  steady-state inference peak, not the one-time load/context peak.
- **This function does not compute WER.** Use `evaluate_model` for accuracy;
  mixing the two would let a benchmark change the pinned protocol.
- **Host RSS is a high-water mark per process**, just like the evaluator. GPU
  peaks are per-process allocator peaks; other users' memory is not included.
"""

from __future__ import annotations

import statistics
import time

import psutil
import torch

from onebit_asr.evaluation.evaluate import percentile


def _sync(device: torch.device) -> None:
    # No-op on CPU; on CUDA this makes the perf_counter readings meaningful.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


@torch.no_grad()
def benchmark_model(
    model,
    processor,
    samples: list[dict],
    *,
    sample_rate: int = 16_000,
    warmup: int = 3,
    progress_every: int = 64,
    verbose: bool = True,
) -> dict:
    """Benchmark ``model`` on ``samples`` (no accuracy computation).

    Returns latency (mean/p50/p95/min/max), preprocessing vs model-only time,
    RTF, throughput, peak RSS, and CUDA peak memory when applicable.
    """
    if not samples:
        raise ValueError("benchmark_model needs at least one sample")

    was_training = model.training
    model.eval()
    device = _device_of(model)

    n_warmup = min(int(warmup), len(samples))
    for s in samples[:n_warmup]:
        # Untimed warmup: allocate CUDA workspaces, autotune kernels, fault in
        # any lazily loaded weights. Do not record these timings.
        inputs = processor(s["audio"], sampling_rate=sample_rate,
                           return_tensors="pt", padding=True)
        _sync(device)
        model(inputs.input_values.to(device))
        _sync(device)

    if device.type == "cuda":
        # Start the peak measurement after warmup so it reflects steady-state
        # inference, not model loading.
        torch.cuda.reset_peak_memory_stats(device)
    proc = psutil.Process()
    peak_rss = proc.memory_info().rss

    latencies, model_times, preprocess_times, audio_secs = [], [], [], []
    for i, s in enumerate(samples):
        audio_sec = len(s["audio"]) / sample_rate
        audio_secs.append(audio_sec)

        t0 = time.perf_counter()
        inputs = processor(s["audio"], sampling_rate=sample_rate,
                           return_tensors="pt", padding=True)
        t1 = time.perf_counter()

        _sync(device)
        t2 = time.perf_counter()
        logits = model(inputs.input_values.to(device)).logits
        processor.batch_decode(torch.argmax(logits, dim=-1))
        _sync(device)
        t3 = time.perf_counter()

        preprocess_times.append(t1 - t0)
        model_times.append(t3 - t2)
        latencies.append(t3 - t0)
        peak_rss = max(peak_rss, proc.memory_info().rss)
        if verbose and progress_every and (i + 1) % progress_every == 0:
            print(f"  benchmark {i + 1}/{len(samples)} ...", flush=True)

    total_audio = sum(audio_secs)
    total_time = sum(latencies)
    total_model = sum(model_times)

    memory: dict = {"peak_rss_mb": peak_rss / 1e6}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        memory.update({
            "cuda": True,
            "device_name": props.name,
            "total_vram_bytes": int(props.total_memory),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        })
    else:
        memory["cuda"] = False

    if was_training:
        model.train()

    return {
        "n_utterances": len(samples),
        "warmup_utterances": n_warmup,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "total_audio_s": total_audio,
        "total_processing_s": total_time,
        "total_model_s": total_model,
        "rtf": total_time / total_audio,
        "rtf_model_only": total_model / total_audio,
        "throughput_utt_per_s": len(samples) / total_time,
        "throughput_audio_s_per_s": total_audio / total_time,
        "latency_s": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "model_latency_s": {
            "mean": statistics.fmean(model_times),
            "p50": percentile(model_times, 50),
            "p95": percentile(model_times, 95),
        },
        "preprocess_latency_s": {
            "mean": statistics.fmean(preprocess_times),
            "p50": percentile(preprocess_times, 50),
        },
        "memory": memory,
    }
