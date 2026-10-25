"""Chunked (pseudo-streaming) CTC inference (Phase 19).

Honest scope
------------
Wav2Vec2-Conformer is a non-streaming encoder: its self-attention sees the whole
utterance. This module implements a **chunked** baseline — overlapping windows,
per-window greedy CTC, and a word-level merge — which gives partial transcripts
and a real-time interface without claiming true streaming state reuse.

Consequences, measured rather than hidden:
- each chunk re-runs the full encoder (compute overhead scales with overlap);
- chunk boundaries can truncate words, so accuracy is expected to be worse than
  full-utterance decoding;
- the interface is still useful for latency and partial-output experiments.

A true streaming implementation would need cached attention/convolution state
and a streaming-capable encoder; that is future work (plan §19 / §18).
"""

from __future__ import annotations

import time

import numpy as np
import torch


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int = 16_000,
    chunk_s: float = 2.0,
    overlap_s: float = 0.5,
    min_tail_s: float = 0.1,
) -> list[dict]:
    """Split ``audio`` into overlapping windows that cover the whole signal."""
    if chunk_s <= 0:
        raise ValueError("chunk_s must be positive")
    if overlap_s < 0 or overlap_s >= chunk_s:
        raise ValueError("overlap_s must satisfy 0 <= overlap_s < chunk_s")

    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    size = int(round(chunk_s * sample_rate))
    step = int(round((chunk_s - overlap_s) * sample_rate))
    if size <= 0 or step <= 0:
        raise ValueError("chunk/overlap produce a non-positive window or step")

    if n <= size:
        return [{
            "start_sample": 0,
            "end_sample": n,
            "start_s": 0.0,
            "end_s": n / sample_rate,
            "audio": audio,
        }]

    starts = list(range(0, n - size + 1, step))
    tail_start = n - size
    if starts[-1] < tail_start:
        # The regular stride stopped before the end. Append one final full window
        # so the tail is covered. If the tail is already inside the previous
        # window's overlap (less than `min_tail_s`), just shift that window
        # instead of adding a nearly-duplicate one.
        if tail_start - starts[-1] < min_tail_s * sample_rate:
            starts[-1] = tail_start
        else:
            starts.append(tail_start)

    chunks = []
    for start in starts:
        end = min(start + size, n)
        chunks.append({
            "start_sample": int(start),
            "end_sample": int(end),
            "start_s": start / sample_rate,
            "end_s": end / sample_rate,
            "audio": audio[start:end],
        })
    return chunks


def _tokens(text: str) -> list[str]:
    return [t for t in text.strip().split() if t]


def merge_hypotheses(previous: str, new: str, max_overlap_words: int = 30) -> str:
    """Merge two overlapping chunk hypotheses by matching a suffix/prefix.

    If no overlap is found the texts are concatenated; the overlap window is the
    caller's control over how often that happens. This is a heuristic: it does
    not align words acoustically, so it can duplicate text when the overlap does
    not cover a full word (observed in the demo: ``... HIGH CALL INSIDE THE HIGH
    CALLOR ...``).
    """
    prev_tokens = _tokens(previous)
    new_tokens = _tokens(new)
    if not prev_tokens:
        return " ".join(new_tokens)
    if not new_tokens:
        return " ".join(prev_tokens)

    # Try the longest suffix of `previous` that equals a prefix of `new`. The
    # comparison is case-insensitive because the CTC tokenizer is uppercase but
    # the merge should not depend on it.
    max_overlap = min(len(prev_tokens), len(new_tokens), int(max_overlap_words))
    for k in range(max_overlap, 0, -1):
        if [t.lower() for t in prev_tokens[-k:]] == [t.lower() for t in new_tokens[:k]]:
            return " ".join(prev_tokens + new_tokens[k:])
    return " ".join(prev_tokens + new_tokens)


class StreamingTranscriber:
    """Run a CTC model over overlapping chunks and merge the text."""

    def __init__(
        self,
        model,
        processor,
        *,
        chunk_s: float = 2.0,
        overlap_s: float = 0.5,
        sample_rate: int = 16_000,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.chunk_s = float(chunk_s)
        self.overlap_s = float(overlap_s)
        self.sample_rate = int(sample_rate)
        if device is None:
            try:
                device = next(model.parameters()).device
            except (AttributeError, StopIteration):
                device = torch.device("cpu")
        self.device = torch.device(device)

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, on_partial=None) -> dict:
        """Return the final merged text plus per-chunk partials and timings."""
        self.model.eval()
        chunks = chunk_audio(
            audio, sample_rate=self.sample_rate,
            chunk_s=self.chunk_s, overlap_s=self.overlap_s,
        )
        hypothesis = ""
        partials = []
        total_audio = 0.0
        total_time = 0.0
        for i, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            inputs = self.processor(
                chunk["audio"], sampling_rate=self.sample_rate,
                return_tensors="pt", padding=True,
            )
            # The full encoder sees only this window. There is no cached
            # attention/convolution state, which is exactly why this is
            # "chunked" and not a streaming encoder.
            logits = self.model(inputs.input_values.to(self.device)).logits
            chunk_text = self.processor.batch_decode(
                torch.argmax(logits, dim=-1)
            )[0].strip()
            dt = time.perf_counter() - t0
            duration = len(chunk["audio"]) / self.sample_rate
            total_audio += duration
            total_time += dt
            # Merge the new chunk into the running hypothesis. The overlap is
            # the only mechanism preventing duplicated words; it is heuristic.
            hypothesis = merge_hypotheses(hypothesis, chunk_text)
            partial = {
                "chunk": i,
                "start_s": chunk["start_s"],
                "end_s": chunk["end_s"],
                "chunk_text": chunk_text,
                "merged_text": hypothesis,
                "chunk_processing_s": dt,
                "chunk_rtf": dt / duration if duration else None,
            }
            partials.append(partial)
            if on_partial is not None:
                on_partial(partial)

        return {
            "text": hypothesis,
            "n_chunks": len(chunks),
            "chunk_s": self.chunk_s,
            "overlap_s": self.overlap_s,
            "total_audio_s": total_audio,
            "total_processing_s": total_time,
            "rtf": total_time / total_audio if total_audio else None,
            "partials": partials,
            "scope": (
                "chunked pseudo-streaming: full encoder per overlapping window; "
                "not a true streaming encoder"
            ),
        }


def greedy_full_utterance(model, processor, audio: np.ndarray,
                          sample_rate: int = 16_000,
                          device: torch.device | str | None = None) -> str:
    """Reference non-streaming greedy decode of the whole utterance."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    with torch.no_grad():
        inputs = processor(audio, sampling_rate=sample_rate,
                           return_tensors="pt", padding=True)
        logits = model(inputs.input_values.to(device)).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()
