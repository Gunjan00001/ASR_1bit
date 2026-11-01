"""Reference-transcript metrics using the research normalization pipeline."""

from __future__ import annotations

import jiwer

from onebit_asr.data.text import normalize_pair


def compute_reference_metrics(reference: str, hypothesis: str) -> dict | None:
    ref, hyp = normalize_pair(reference or "", hypothesis or "")
    if not ref.strip():
        return None
    words = jiwer.process_words([ref], [hyp])
    chars = jiwer.process_characters([ref], [hyp])
    return {
        "reference": ref,
        "hypothesis": hyp,
        "wer": float(words.wer),
        "cer": float(chars.cer),
        "substitutions": int(words.substitutions),
        "deletions": int(words.deletions),
        "insertions": int(words.insertions),
        "hits": int(words.hits),
    }
