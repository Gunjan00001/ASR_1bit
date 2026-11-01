"""Controlled deployment benchmark for FP32, 2-bit FFN, and Strategy A.

Runs one arm per process so peak host RSS is not contaminated by a previous
model. The packed arms rebuild from the packed artifact, not from FP32 masters:
the architecture is instantiated from config, packed layers are replaced from
the artifact, and all non-quantized tensors are loaded from the artifact.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import time
from pathlib import Path

import jiwer
import psutil
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.data.text import normalize_text
from onebit_asr.models.conformer_utils import get_parent_module
from onebit_asr.quantization.packed import PackedBitLinear

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "facebook/wav2vec2-conformer-rope-large-960h-ft"
DEV_N = 256
SEED = 42

ARMS = {
    "fp32": {
        "kind": "fp32",
        "expected_wer": 0.020184544405997693,
    },
    "ffn2bit": {
        "kind": "packed",
        "config": "configs/cluster_effective_qat_ffn_mixed_2bit_full.yaml",
        "checkpoint": "checkpoints/qat_ffn_mixed_2bit_full",
        "artifact": "checkpoints/qat_ffn_mixed_2bit_full_packed_v2/model_packed.safetensors",
        "expected_wer": 0.024413687043444827,
    },
    "strategy_a": {
        "kind": "packed",
        "config": "configs/cluster_effective_attn_2bit_on_2bitffn_A_full.yaml",
        "checkpoint": "checkpoints/attn_2bit_on_2bitffn_A_full",
        "artifact": "checkpoints/attn_2bit_on_2bitffn_A_full_packed_v2/model_packed.safetensors",
        "expected_wer": 0.027297193387158785,
    },
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_packed(checkpoint: Path, artifact_path: Path) -> tuple[torch.nn.Module, dict]:
    manifest_path = Path(str(artifact_path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    artifact = load_file(str(artifact_path), device="cpu")
    config = AutoConfig.from_pretrained(checkpoint)
    model = Wav2Vec2ConformerForCTC(config)
    replaced = 0
    for packed_key in [key for key in artifact if key.endswith("_packed")]:
        base = packed_key[: -len("_packed")]
        module_name = base.rsplit(".", 1)[0]
        layer = PackedBitLinear.from_artifact(
            artifact[packed_key],
            tuple(manifest["weight_shapes"][base]),
            artifact[f"{base}_scale"],
            artifact.get(f"{module_name}.bias"),
            bits=int(manifest["bits"]),
        )
        parent, attr = get_parent_module(model, module_name)
        setattr(parent, attr, layer)
        replaced += 1
    if replaced != int(manifest["n_bitlinear_layers"]):
        raise ValueError(f"replaced {replaced} layers, manifest says {manifest['n_bitlinear_layers']}")
    state = {key: artifact[key] for key in manifest["non_quantized_keys"]}
    model_keys = set(model.state_dict())
    missing_artifact = [key for key in state if key not in model_keys]
    if missing_artifact:
        raise ValueError(f"artifact keys not present in model: {missing_artifact[:5]}")
    load_result = model.load_state_dict(state, strict=False)
    if load_result.unexpected_keys:
        raise ValueError(f"unexpected artifact keys: {load_result.unexpected_keys[:5]}")
    return model, {
        "artifact_manifest": str(manifest_path),
        "n_packed_layers": replaced,
        "bits": int(manifest["bits"]),
        "scale_mode": manifest.get("scale_mode"),
        "manifest": manifest,
    }


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main() -> int:
    parser = argparse.ArgumentParser(description="Final controlled ARM benchmark.")
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--n", type=int, default=DEV_N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--expected-wer", type=float, default=None)
    parser.add_argument("--expected-wer-tolerance", type=float, default=5e-6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arm = ARMS[args.arm]
    expected = args.expected_wer if args.expected_wer is not None else arm.get("expected_wer")
    proc = psutil.Process()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    samples = load_dev_subset(subset_size=args.n, seed=args.seed)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    if arm["kind"] == "fp32":
        model = Wav2Vec2ConformerForCTC.from_pretrained(MODEL_ID)
        load_info = {"kind": "fp32", "checkpoint": None, "artifact": None}
    else:
        checkpoint = ROOT / arm["checkpoint"]
        artifact = ROOT / arm["artifact"]
        model, load_info = load_packed(checkpoint, artifact)
        load_info.update({
            "kind": "packed",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
            "artifact": str(artifact),
            "artifact_sha256": sha256(artifact),
            "artifact_bytes": artifact.stat().st_size,
        })
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    load_peak_alloc = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    load_peak_reserved = int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else None
    load_rss = proc.memory_info().rss / 1e6
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    refs, hyps, transcripts = [], [], []
    forward_times, e2e_times = [], []
    total_audio = 0.0
    timed = 0
    peak_rss = load_rss
    for sample in samples[: min(args.warmup, len(samples))]:
        warm_inputs = processor(sample["audio"], sampling_rate=16_000,
                                 return_tensors="pt", padding=True)
        with torch.no_grad():
            model(warm_inputs.input_values.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
    for sample in samples:
        e2e0 = time.perf_counter()
        inputs = processor(sample["audio"], sampling_rate=16_000,
                           return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        f0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_values).logits
        if device.type == "cuda":
            torch.cuda.synchronize()
        f1 = time.perf_counter()
        hyp = processor.batch_decode(logits.argmax(dim=-1))[0]
        e2e1 = time.perf_counter()
        ref = normalize_text(sample["text"])
        hyp = normalize_text(hyp)
        refs.append(ref)
        hyps.append(hyp)
        forward_times.append(f1 - f0)
        e2e_times.append(e2e1 - e2e0)
        total_audio += len(sample["audio"]) / 16_000
        if len(transcripts) < 3:
            transcripts.append({"id": sample["id"], "reference": ref, "hypothesis": hyp})
        timed += 1
        peak_rss = max(peak_rss, proc.memory_info().rss / 1e6)

    wer = float(jiwer.wer(refs, hyps))
    cer = float(jiwer.cer(refs, hyps))
    wer_delta = None if expected is None else wer - expected
    if expected is not None and abs(wer_delta) > args.expected_wer_tolerance:
        raise SystemExit(
            f"{args.arm}: WER {wer:.12f} differs from stored {expected:.12f} "
            f"by {wer_delta:+.12f}, beyond tolerance "
            f"{args.expected_wer_tolerance:.12f}"
        )
    result = {
        "phase": "final controlled deployment benchmark",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "arm": args.arm,
        "kind": arm["kind"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_utterances": timed,
        "warmup_utterances": min(args.warmup, len(samples)),
        "seed": args.seed,
        "decoding": "greedy",
        "text_normalization": "lowercase + strip punctuation",
        "model_load_s": load_s,
        "load_peak_gpu_allocated_bytes": load_peak_alloc,
        "load_peak_gpu_reserved_bytes": load_peak_reserved,
        "loaded_host_rss_mb": load_rss,
        "peak_host_rss_mb": peak_rss,
        "inference_peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "inference_peak_gpu_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else None
        ),
        "wer": wer,
        "cer": cer,
        "expected_wer": expected,
        "expected_wer_delta": wer_delta,
        "total_audio_s": total_audio,
        "forward_latency_s": {
            "mean": statistics.fmean(forward_times),
            "p50": percentile(forward_times, 50),
            "p95": percentile(forward_times, 95),
            "max": max(forward_times),
        },
        "e2e_latency_s": {
            "mean": statistics.fmean(e2e_times),
            "p50": percentile(e2e_times, 50),
            "p95": percentile(e2e_times, 95),
            "max": max(e2e_times),
        },
        "acoustic_rtf": statistics.fmean(forward_times) / (total_audio / timed),
        "e2e_rtf": statistics.fmean(e2e_times) / (total_audio / timed),
        "artifact_bytes": (
            int(model.num_parameters() * 4) if arm["kind"] == "fp32"
            else load_info["artifact_bytes"]
        ),
        "load_info": {k: v for k, v in load_info.items() if k != "manifest"},
        "transcripts": transcripts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in (
        "arm", "wer", "cer", "artifact_bytes", "model_load_s",
        "load_peak_gpu_allocated_bytes", "inference_peak_gpu_allocated_bytes",
        "peak_host_rss_mb", "forward_latency_s", "e2e_latency_s",
        "acoustic_rtf", "e2e_rtf")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
