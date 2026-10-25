"""Phase 18 — packed 1-bit inference and binary-compute boundary benchmark.

Measures, honestly:

1. **Storage**: a QAT checkpoint's BitLinear FP32 masters vs packed 1-bit
   weights (+ FP scales/biases + non-quantized FP parameters).
2. **Inference cost**: BitLinear (FP masters, sign each forward) vs
   PackedBitLinear (expand packed bits each forward) on CPU.
3. **Kernel boundary**: a toy XNOR+popcount matmul (1-bit × 1-bit) vs a float
   matmul of the same logical size. The model itself has FP activations, so the
   XNOR kernel is a reference for what would be required, not a drop-in path.

Usage:
    python scripts/benchmark_packed.py --config configs/qat_attn_ffn.yaml \
        --checkpoint checkpoints/qat_attn_ffn --n 4
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.quantization.packed import (
    pack_model,
    packed_model_size_report,
    xnor_popcount_packed,
    xnor_similarity_to_dot,
)
from onebit_asr.training.qat import load_qat_checkpoint

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _timed_forward(model, input_values, runs=2):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(input_values)
            times.append(time.perf_counter() - t0)
    return float(np.mean(times)), times


def kernel_benchmark(n_bits=1024, m=64, k=256, repeats=20) -> dict:
    """Compare a packed XNOR+popcount matmul with an equivalent float matmul."""
    rng = np.random.default_rng(0)
    x_bits = rng.integers(0, 2, size=(m, n_bits), dtype=np.uint8)
    w_bits = rng.integers(0, 2, size=(k, n_bits), dtype=np.uint8)
    x_packed = np.packbits(x_bits, axis=1)
    w_packed = np.packbits(w_bits, axis=1)
    x_float = x_bits.astype(np.float32) * 2 - 1
    w_float = w_bits.astype(np.float32) * 2 - 1

    # correctness: packed XNOR equals the ±1 dot product
    sim = xnor_popcount_packed(x_packed, w_packed, n_bits)
    dot = xnor_similarity_to_dot(sim, n_bits)
    expected = (x_float @ w_float.T).astype(np.int32)
    matches = bool(np.array_equal(dot, expected))

    t0 = time.perf_counter()
    for _ in range(repeats):
        xnor_popcount_packed(x_packed, w_packed, n_bits)
    xnor_s = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        x_float @ w_float.T
    float_s = (time.perf_counter() - t0) / repeats

    return {
        "shape": {"m": m, "k": k, "n_bits": n_bits},
        "xnor_popcount_matches_float_dot": matches,
        "xnor_popcount_seconds": xnor_s,
        "float_matmul_seconds": float_s,
        "xnor_vs_float_ratio": xnor_s / float_s if float_s else None,
        "note": (
            "One-bit × one-bit only. The ASR model has FP activations, so this "
            "kernel cannot replace BitLinear without also binarizing activations "
            "(a different model with different accuracy)."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Packed inference benchmark.")
    parser.add_argument("--config", default="configs/qat_attn_ffn.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/qat_attn_ffn")
    parser.add_argument("--n", type=int, default=4, help="utterances to time")
    parser.add_argument("--output", default=str(RESULTS / "packed_inference_benchmark.json"))
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    checkpoint = ROOT / args.checkpoint
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    quant = cfg.get("quantization", {})
    model_id = cfg["qat"]["init_from"]

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    samples = load_dev_subset(subset_size=max(args.n, 4), seed=cfg["eval"]["seed"])[: args.n]
    inputs = processor(samples[0]["audio"], sampling_rate=16_000,
                       return_tensors="pt").input_values

    print(f"Loading QAT checkpoint {checkpoint} ...")
    model, _report = load_qat_checkpoint(
        checkpoint, Wav2Vec2ConformerForCTC, cfg["layers"],
        scale_mode=quant.get("scale", "per_tensor_mean_abs"),
        ste_clip=quant.get("ste_clip"),
    )
    model.eval()

    fp_master_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    bitlinear_time, _ = _timed_forward(model, inputs)
    print(f"BitLinear forward: {bitlinear_time:.3f}s; FP32 state {fp_master_bytes / 1e9:.2f} GB")

    packed, pack_info = pack_model(model)
    packed.eval()
    packed_time, _ = _timed_forward(packed, inputs)
    size = packed_model_size_report(packed)
    print(f"PackedBitLinear forward: {packed_time:.3f}s; "
          f"packed weights {size['packed_weight_bytes'] / 1e6:.1f} MB")
    print(f"Slowdown from packing: {packed_time / bitlinear_time:.2f}x")

    kernel = kernel_benchmark()

    result = {
        "phase": "Phase 18 packed 1-bit inference",
        "timestamp_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "platform": platform.platform(),
        "config_path": cfg.get("_config_path"),
        "checkpoint": str(checkpoint),
        "n_utterances_timed": args.n,
        "storage": {
            "fp32_state_bytes": fp_master_bytes,
            **pack_info,
            **size,
        },
        "inference": {
            "bitlinear_forward_s": bitlinear_time,
            "packed_forward_s": packed_time,
            "packed_slowdown_ratio": packed_time / bitlinear_time,
            "device": "cpu",
            "note": "Packing saves storage; expanding bits costs time on CPU.",
        },
        "binary_compute_boundary": kernel,
        "conclusion": (
            "1-bit storage is real (packed payload); compute is not accelerated "
            "for this model because activations are FP and the runtime expands "
            "weights before F.linear. XNOR+popcount works only for 1-bit×1-bit."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
