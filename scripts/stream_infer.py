"""Phase 19 — chunked streaming inference demo.

Runs the same greedy CTC model over overlapping chunks, prints partial
transcripts, and compares the merged chunked output with a full-utterance
decode (WER/CER under the pinned normalization).

Examples:
    python scripts/stream_infer.py --sample-index 0 --chunk 2.0 --overlap 0.5
    python scripts/stream_infer.py --audio path.wav --checkpoint checkpoints/qat_attn_ffn \
        --config configs/qat_attn_ffn.yaml
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.config import load_config
from onebit_asr.data.dataset import load_dev_subset
from onebit_asr.data.text import normalize_pair
from onebit_asr.evaluation.metrics import compute_cer, compute_wer
from onebit_asr.streaming import StreamingTranscriber, greedy_full_utterance
from onebit_asr.training.qat import load_qat_checkpoint

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunked streaming demo.")
    parser.add_argument("--config", default="configs/qat_attn_ffn.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="QAT checkpoint dir; omit to use the FP32 model")
    parser.add_argument("--audio", default=None, help="16 kHz audio file")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="dev-subset sample to use when --audio is absent")
    parser.add_argument("--chunk", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--output", default=str(RESULTS / "streaming_demo.json"))
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    model_id = cfg["qat"]["init_from"]
    processor = Wav2Vec2Processor.from_pretrained(model_id)

    reference = ""
    sample_id = None
    if args.audio:
        audio, sr = sf.read(args.audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            raise SystemExit(f"expected 16 kHz audio, got {sr}")
    else:
        sample = load_dev_subset(subset_size=max(args.sample_index + 1, 8),
                                 seed=cfg["eval"]["seed"])[args.sample_index]
        audio = np.asarray(sample["audio"], dtype=np.float32)
        reference = sample["text"]
        sample_id = sample["id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        checkpoint = ROOT / args.checkpoint
        if not checkpoint.is_dir():
            raise SystemExit(f"checkpoint not found: {checkpoint}")
        quant = cfg.get("quantization", {})
        model, _ = load_qat_checkpoint(
            checkpoint, Wav2Vec2ConformerForCTC, cfg["layers"],
            scale_mode=quant.get("scale", "per_tensor_mean_abs"),
            ste_clip=quant.get("ste_clip"),
        )
        model_label = str(args.checkpoint)
    else:
        model = Wav2Vec2ConformerForCTC.from_pretrained(model_id)
        model_label = f"{model_id} (FP32)"
    model.to(device)

    print(f"Model: {model_label}")
    print(f"Audio: {len(audio) / 16_000:.2f}s"
          + (f"  sample_id={sample_id}" if sample_id else ""))

    streaming = StreamingTranscriber(
        model, processor, chunk_s=args.chunk, overlap_s=args.overlap, device=device
    )
    t0 = time.perf_counter()
    chunked = streaming.transcribe(audio)
    wall = time.perf_counter() - t0

    print("\nPartial transcripts:")
    for p in chunked["partials"]:
        print(f"  [{p['start_s']:6.2f}-{p['end_s']:6.2f}s] {p['merged_text']}")

    t0 = time.perf_counter()
    full_text = greedy_full_utterance(model, processor, audio, device=device)
    full_time = time.perf_counter() - t0

    result = {
        "phase": "Phase 19 chunked streaming demo",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "model": model_label,
        "sample_id": sample_id,
        "audio_seconds": len(audio) / 16_000,
        "chunk_s": args.chunk,
        "overlap_s": args.overlap,
        **chunked,
        "wall_clock_s": wall,
        "full_utterance": {
            "text": full_text,
            "processing_s": full_time,
            "rtf": full_time / (len(audio) / 16_000),
        },
    }

    if reference:
        ref_n, chunk_n = normalize_pair(reference, chunked["text"])
        _ref_n2, full_n = normalize_pair(reference, full_text)
        result["reference"] = ref_n
        result["chunked_wer"] = compute_wer([ref_n], [chunk_n])
        result["chunked_cer"] = compute_cer([ref_n], [chunk_n])
        result["full_utterance_wer"] = compute_wer([ref_n], [full_n])
        result["full_utterance_cer"] = compute_cer([ref_n], [full_n])
        print(f"\nreference : {reference}")
        print(f"streamed  : {chunked['text']}")
        print(f"full      : {full_text}")
        print(f"WER: chunked {result['chunked_wer']:.4f} | full {result['full_utterance_wer']:.4f}")
    else:
        print(f"\nstreamed: {chunked['text']}\nfull: {full_text}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
