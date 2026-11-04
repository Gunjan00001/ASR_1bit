"""Incremental greedy CTC decoder with state across chunk boundaries.

This decoder consumes CTC logits for consecutive, non-overlapping output frames.
It preserves the previous frame token, blank state, the current partial word, and
all committed words. Words are committed at the tokenizer's ``|`` delimiter; the
final word is flushed at end-of-stream.
"""

from __future__ import annotations

import numpy as np


class IncrementalCTCDecoder:
    def __init__(self, tokenizer, blank_id: int = 0, word_delimiter: str = "|") -> None:
        self.tokenizer = tokenizer
        self.blank_id = int(blank_id)
        self.vocab = tokenizer.get_vocab()
        self.id_to_token = {int(v): k for k, v in self.vocab.items()}
        self.word_delimiter = word_delimiter
        self.previous_frame_token: int | None = None
        self.committed_words: list[str] = []
        self.current_chars: list[str] = []
        self.tokens_emitted = 0
        self.frames_consumed = 0
        self.revisions = 0
        self.commit_times: list[float] = []

    def _token_text(self, token_id: int) -> str | None:
        if token_id == self.blank_id:
            return None
        token = self.id_to_token.get(int(token_id))
        if token is None or token in {"<s>", "</s>", "<unk>", "<pad>"}:
            return None
        if token == self.word_delimiter:
            return self.word_delimiter
        return token

    def _commit_current_word(self) -> str:
        word = "".join(self.current_chars).strip()
        self.current_chars = []
        if not word:
            return ""
        self.committed_words.append(word)
        return word

    def feed_logits(self, logits: np.ndarray | "torch.Tensor") -> dict:
        """Consume new frame logits and return the updated transcript state."""
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        logits = np.asarray(logits)
        if logits.ndim != 2:
            raise ValueError(f"logits must be [frames, vocab], got {logits.shape}")
        ids = np.argmax(logits, axis=-1)
        new_segment = ""
        for token_id in ids:
            self.frames_consumed += 1
            token_id = int(token_id)
            if self.previous_frame_token == token_id:
                # Same emitted frame label: CTC collapse; no new symbol.
                continue
            self.previous_frame_token = token_id
            token = self._token_text(token_id)
            if token is None:
                continue
            if token == self.word_delimiter:
                segment = self._commit_current_word()
                if segment:
                    new_segment = segment
                continue
            self.current_chars.append(token)
            self.tokens_emitted += 1
            self.revisions += 1
        return {
            "new_committed": new_segment,
            **self.snapshot(),
        }

    def snapshot(self) -> dict:
        final_text = " ".join(self.committed_words)
        partial_text = "".join(self.current_chars)
        return {
            "final_text": final_text,
            "partial_text": partial_text,
            "text": (final_text + (" " if final_text and partial_text else "") + partial_text).strip(),
            "committed_words": len(self.committed_words),
            "provisional_chars": len(self.current_chars),
            "tokens_emitted": self.tokens_emitted,
            "frames_consumed": self.frames_consumed,
            "revisions": self.revisions,
        }

    def finalize(self) -> dict:
        segment = self._commit_current_word()
        return {
            "new_committed": segment,
            **self.snapshot(),
        }

    def reset(self) -> None:
        self.previous_frame_token = None
        self.committed_words = []
        self.current_chars = []
        self.tokens_emitted = 0
        self.frames_consumed = 0
        self.revisions = 0
        self.commit_times = []
