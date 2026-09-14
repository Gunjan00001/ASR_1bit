"""Phase 2 — minimal inference smoke test (FP32 baseline, greedy CTC).

Usage:
    python scripts/infer.py path/to/audio_16khz.wav   # transcribe a file
    python scripts/infer.py --demo                    # transcribe one real
                                                      # LibriSpeech sample
Pipeline: Audio -> Processor -> Wav2Vec2-Conformer -> Logits -> Decode -> Text
"""

import argparse

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

MODEL_ID = "facebook/wav2vec2-conformer-rope-large-960h-ft"
SAMPLE_RATE = 16_000


def transcribe(model, processor, audio: np.ndarray) -> str:
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


def load_demo():
    from datasets import load_dataset

    ds = load_dataset("openslr/librispeech_asr", "clean",
                      split="validation", streaming=True)
    sample = next(iter(ds))
    audio = np.asarray(sample["audio"]["array"], dtype=np.float32)
    sr = int(sample["audio"]["sampling_rate"])
    if sr != SAMPLE_RATE:
        raise SystemExit(f"Demo sample is {sr} Hz, expected {SAMPLE_RATE} Hz.")
    return audio, sample.get("text", "")


def main():
    parser = argparse.ArgumentParser(description="Transcribe 16 kHz audio.")
    parser.add_argument("audio", nargs="?", help="Path to a 16 kHz WAV file.")
    parser.add_argument("--demo", action="store_true",
                        help="Transcribe one streaming LibriSpeech sample.")
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    if not args.audio and not args.demo:
        parser.error("provide an audio path or --demo")

    print(f"Loading {args.model} (CPU) ...")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ConformerForCTC.from_pretrained(args.model)
    model.eval()

    if args.demo:
        audio, reference = load_demo()
        print(f"Demo sample: {len(audio) / SAMPLE_RATE:.1f}s of audio")
    else:
        audio, sr = sf.read(args.audio, dtype="float32")
        if sr != SAMPLE_RATE:
            raise SystemExit(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz. Resample first.")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        reference = ""

    text = transcribe(model, processor, audio)
    print(f"\nTranscription: {text}")
    if reference:
        print(f"Reference    : {reference}")
    if not text.strip():
        raise SystemExit("SMOKE TEST FAILED: empty transcription")


if __name__ == "__main__":
    main()
