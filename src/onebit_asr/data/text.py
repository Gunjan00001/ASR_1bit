"""Text normalization for WER/CER (plan §Phase 3 pinned protocol).

The CTC tokenizer emits UPPERCASE text; normalize both reference and
hypothesis identically: lowercase, strip punctuation, collapse whitespace.
"""

import re
import unicodedata

_WS = re.compile(r"\s+")
# Keep letters, numbers, and spaces; drop everything else (punctuation, symbols).
_KEEP = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _KEEP.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def normalize_pair(reference: str, hypothesis: str) -> tuple[str, str]:
    return normalize_text(reference), normalize_text(hypothesis)
