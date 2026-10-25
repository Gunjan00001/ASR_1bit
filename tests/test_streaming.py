"""Streaming chunking + merge tests (Phase 19). No model is loaded."""

from __future__ import annotations

import numpy as np
import pytest

from onebit_asr.streaming import chunk_audio, merge_hypotheses


def test_chunks_cover_signal_and_overlap():
    audio = np.arange(16_000 * 5, dtype=np.float32)
    chunks = chunk_audio(audio, chunk_s=2.0, overlap_s=0.5)
    assert chunks[0]["start_sample"] == 0
    assert chunks[-1]["end_sample"] == len(audio)
    for a, b in zip(chunks, chunks[1:]):
        assert b["start_sample"] < a["end_sample"], "consecutive chunks must overlap"
    assert all(c["end_sample"] <= len(audio) for c in chunks)
    # Every sample is covered.
    covered = np.zeros(len(audio), dtype=bool)
    for c in chunks:
        covered[c["start_sample"]:c["end_sample"]] = True
    assert covered.all()


def test_short_audio_is_one_chunk():
    audio = np.zeros(16_000, dtype=np.float32)
    chunks = chunk_audio(audio, chunk_s=2.0, overlap_s=0.5)
    assert len(chunks) == 1
    assert chunks[0]["start_sample"] == 0
    assert chunks[0]["end_sample"] == len(audio)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError, match="overlap_s"):
        chunk_audio(np.zeros(16_000, dtype=np.float32), chunk_s=1.0, overlap_s=1.0)
    with pytest.raises(ValueError, match="chunk_s"):
        chunk_audio(np.zeros(16_000, dtype=np.float32), chunk_s=0.0)


def test_merge_exact_word_overlap():
    assert merge_hypotheses("HELLO WORLD", "WORLD AGAIN") == "HELLO WORLD AGAIN"
    assert merge_hypotheses("HELLO WORLD", "WORLD") == "HELLO WORLD"


def test_merge_is_case_insensitive_but_preserves_first_text():
    assert merge_hypotheses("hello world", "World again") == "hello world again"


def test_merge_without_overlap_concatenates():
    assert merge_hypotheses("HELLO", "WORLD") == "HELLO WORLD"


def test_merge_empty_sides():
    assert merge_hypotheses("", "A B") == "A B"
    assert merge_hypotheses("A B", "") == "A B"
    assert merge_hypotheses("", "") == ""


def test_merge_respects_max_overlap():
    # A suffix larger than max_overlap_words is not detected, so the texts are
    # concatenated (documented heuristic behaviour).
    assert merge_hypotheses("A B C", "A B C", max_overlap_words=1) == "A B C A B C"
