"""LibriSpeech subset loading and CTC feature preparation.

- ``load_dev_subset``: the pinned Phase 3 development-evaluation subset
  (fixed 256-utterance clean validation, fixed seed). Never used for training.
- ``load_train_subset``: training data from the clean **train** splits only
  (default ``train.100``). Kept strictly separate from the validation split to
  avoid evaluation leakage.
- ``prepare_ctc_features``: turn raw samples into ``{input_values, labels}``
  for CTC training.

Sample IDs are recorded in results JSON so any run can be reproduced exactly.
"""

from __future__ import annotations

import io
import os

import numpy as np
from datasets import Audio, Dataset, load_dataset

DATASET_ID = "openslr/librispeech_asr"
CONFIG = "clean"
DEV_SPLIT = "validation"
TRAIN_SPLIT = "train.100"
EXPECTED_SR = 16_000


def _audio_decoder() -> str:
    """Selected audio decoder backend (default: HF ``datasets`` / torchcodec).

    Set ``ONEBIT_AUDIO_DECODER=soundfile`` to bypass torchcodec/FFmpeg and
    decode the raw FLAC/audio bytes with ``soundfile`` instead. Used on Kaggle
    when torchcodec is unavailable; the default path is unchanged.
    """
    return os.environ.get("ONEBIT_AUDIO_DECODER", "").strip().lower()


def _decode_audio_soundfile(entry: dict) -> tuple[np.ndarray, int]:
    """Decode an ``Audio(decode=False)`` entry (bytes or path) via soundfile."""
    import soundfile as sf

    raw = entry.get("bytes")
    if raw is not None:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    else:
        audio, sr = sf.read(entry["path"], dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _load_subset(split: str, subset_size: int, seed: int) -> list[dict]:
    ds = load_dataset(DATASET_ID, CONFIG, split=split)
    use_soundfile = _audio_decoder() == "soundfile"
    if use_soundfile:
        ds = ds.cast_column("audio", Audio(decode=False))
    if len(ds) < subset_size:
        raise ValueError(
            f"Split {split!r} has {len(ds)} utterances, need {subset_size}"
        )
    sub = ds.shuffle(seed=seed).select(range(subset_size))
    samples = []
    for row in sub:
        if use_soundfile:
            audio, sr = _decode_audio_soundfile(row["audio"])
        else:
            audio = np.asarray(row["audio"]["array"], dtype=np.float32)
            sr = int(row["audio"]["sampling_rate"])
        if sr != EXPECTED_SR:
            raise ValueError(f"Sample {row.get('id')}: {sr} Hz, expected {EXPECTED_SR} Hz")
        samples.append({"id": str(row.get("id", "")), "audio": audio, "text": row["text"]})
    return samples


def load_dev_subset(subset_size: int = 256, seed: int = 42) -> list[dict]:
    """Pinned development-evaluation subset (clean validation)."""
    return _load_subset(DEV_SPLIT, subset_size, seed)


def load_train_subset(subset_size: int = 1000, seed: int = 42, split: str = TRAIN_SPLIT) -> list[dict]:
    """Training subset from the clean train splits (default ``train.100``).

    Raises if asked for the validation split, so a QAT run cannot silently
    train on the evaluation data.
    """
    if split == DEV_SPLIT or split.startswith("validation") or split.startswith("test"):
        raise ValueError(
            f"Refusing to train on split {split!r}: it overlaps the pinned "
            f"evaluation data ({DEV_SPLIT!r}). Use a train split."
        )
    return _load_subset(split, subset_size, seed)


def build_ctc_train_dataset(split: str, subset_size: int, seed: int, processor):
    """Build a CTC training ``Dataset`` with bounded memory.

    Equivalent selection to ``load_train_subset`` + ``prepare_ctc_features`` +
    ``Dataset.from_list``, but the features are *streamed* straight into the
    Arrow-backed ``Dataset`` so peak RAM is bounded by the writer buffer, not by
    ``subset_size``. This matters for large subsets (e.g. 10,000 utterances)
    where materializing the decoded audio twice triggers an OOM kill on Kaggle.

    Returns ``(dataset, provenance)`` where ``provenance`` is the list of
    ``{"id", "text"}`` for the selected utterances (sample IDs are recorded in
    results JSON). The leakage guard is identical to ``load_train_subset``.
    """
    if split == DEV_SPLIT or split.startswith("validation") or split.startswith("test"):
        raise ValueError(
            f"Refusing to train on split {split!r}: it overlaps the pinned "
            f"evaluation data ({DEV_SPLIT!r}). Use a train split."
        )

    ds = load_dataset(DATASET_ID, CONFIG, split=split)
    use_soundfile = _audio_decoder() == "soundfile"
    if use_soundfile:
        ds = ds.cast_column("audio", Audio(decode=False))
    if len(ds) < subset_size:
        raise ValueError(f"Split {split!r} has {len(ds)} utterances, need {subset_size}")

    sub = ds.shuffle(seed=seed).select(range(subset_size))

    # Provenance without touching the audio column (cheap; no decoding).
    provenance = [
        {"id": str(row["id"]), "text": row["text"]}
        for row in sub.select_columns(["id", "text"])
    ]

    def _generator():
        for row in sub:
            if use_soundfile:
                audio, sr = _decode_audio_soundfile(row["audio"])
            else:
                audio = np.asarray(row["audio"]["array"], dtype=np.float32)
                sr = int(row["audio"]["sampling_rate"])
            if sr != EXPECTED_SR:
                raise ValueError(
                    f"Sample {row.get('id')}: {sr} Hz, expected {EXPECTED_SR} Hz"
                )
            labels = processor.tokenizer(row["text"]).input_ids
            yield {"input_values": audio, "labels": list(labels)}

    dataset = Dataset.from_generator(_generator)
    return dataset, provenance


def prepare_ctc_features(samples: list[dict], processor) -> tuple["list[dict]", "list[dict]"]:
    """Convert raw samples into CTC training features.

    Returns ``(features, provenance)`` where ``features`` is a list of
    ``{"input_values": np.ndarray, "labels": list[int]}`` suitable for the
    trainer, and ``provenance`` is a list of ``{"id", "text", "n_frames",
    "n_labels"}`` kept out of the training set for reporting.
    """
    features, provenance = [], []
    for s in samples:
        input_values = processor(s["audio"], sampling_rate=EXPECTED_SR).input_values
        if isinstance(input_values, list):
            input_values = np.asarray(input_values, dtype=np.float32)
        if input_values.ndim > 1:
            input_values = input_values[0]
        labels = processor.tokenizer(s["text"]).input_ids
        features.append({"input_values": input_values.astype(np.float32), "labels": labels})
        provenance.append(
            {"id": s["id"], "text": s["text"],
             "n_frames": int(input_values.shape[-1]), "n_labels": len(labels)}
        )
    return features, provenance
