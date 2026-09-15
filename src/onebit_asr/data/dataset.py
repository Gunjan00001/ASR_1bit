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

import numpy as np
from datasets import load_dataset

DATASET_ID = "openslr/librispeech_asr"
CONFIG = "clean"
DEV_SPLIT = "validation"
TRAIN_SPLIT = "train.100"
EXPECTED_SR = 16_000


def _load_subset(split: str, subset_size: int, seed: int) -> list[dict]:
    ds = load_dataset(DATASET_ID, CONFIG, split=split)
    if len(ds) < subset_size:
        raise ValueError(
            f"Split {split!r} has {len(ds)} utterances, need {subset_size}"
        )
    sub = ds.shuffle(seed=seed).select(range(subset_size))
    samples = []
    for row in sub:
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
