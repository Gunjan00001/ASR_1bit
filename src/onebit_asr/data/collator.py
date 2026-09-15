"""CTC data collator with dynamic padding (Phase 11).

Pads ``input_values`` to the longest item in the batch (via the processor) and
pads ``labels`` with ``-100`` so they are ignored by the CTC loss. This is the
standard Hugging Face wav2vec2-CTC collation pattern; the CTC blank token is
the model's ``pad_token_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DataCollatorCTCWithPadding:
    """Collate ``{input_values, labels}`` dicts into padded tensors."""

    processor: object
    padding: bool | str = True
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict]) -> dict:
        input_features = [
            {"input_values": np.asarray(f["input_values"], dtype=np.float32)}
            for f in features
        ]
        label_features = [{"input_ids": list(f["labels"])} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), self.label_pad_token_id
        )
        batch["labels"] = labels
        if "attention_mask" not in batch:
            # Wav2Vec2FeatureExtractor.pad normally supplies attention_mask.
            # Fallback: the feature extractor pads input_values with 0.0.
            batch["attention_mask"] = collate_attention_mask(batch["input_values"], 0.0)
        return batch

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DataCollatorCTCWithPadding(padding={self.padding!r}, "
            f"label_pad_token_id={self.label_pad_token_id})"
        )


def collate_attention_mask(input_values: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
    """Build an attention mask from padded input values (1 = real, 0 = pad)."""
    return input_values.ne(pad_value).long()
