"""Stateful bounded-context streaming for the non-causal Conformer.

The encoder is bidirectional; exact single-pass KV-cache equivalence is not
valid without changing the trained attention semantics. This path processes
each non-overlapping current segment once with persistent left/right audio
context and persistent CTC state. The encoder is not re-run over the full
history. It is a bounded-context approximation, not exact offline equivalence.
"""

from __future__ import annotations

import time

import numpy as np

from onebit_asr.data.text import normalize_text
from onebit_asr.serving.audio import (
    SAMPLE_RATE,
    chunk_rms,
    prepare_pcm_f32,
)
from onebit_asr.serving.incremental_ctc import IncrementalCTCDecoder
from onebit_asr.serving.metrics import compute_reference_metrics


class StreamingProtocolError(ValueError):
    pass


class StatefulStreamingSession:
    def __init__(
        self,
        manager,
        model_id: str,
        tokenizer,
        *,
        sample_rate: int = SAMPLE_RATE,
        chunk_ms: int = 320,
        current_s: float = 1.0,
        left_s: float = 0.8,
        right_s: float = 0.32,
        eou_silence_s: float = 0.8,
        speech_rms: float = 0.008,
        expect_sequence: bool = True,
        capture_logits: bool = False,
    ) -> None:
        self.manager = manager
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.sample_rate = int(sample_rate)
        self.chunk_ms = int(chunk_ms)
        self.current_s = float(current_s)
        self.left_s = float(left_s)
        self.right_s = float(right_s)
        if self.current_s <= 0 or self.left_s < 0 or self.right_s < 0:
            raise ValueError("current_s must be positive; contexts must be non-negative")
        self.current_samples = int(round(self.current_s * SAMPLE_RATE))
        self.left_samples = int(round(self.left_s * SAMPLE_RATE))
        self.right_samples = int(round(self.right_s * SAMPLE_RATE))
        self.decoder = IncrementalCTCDecoder(tokenizer)
        self.audio = np.empty(0, dtype=np.float32)
        self.buffer_start_sample = 0
        self.next_segment_start = 0
        self.chunks_received = 0
        self.expected_sequence = 0
        self.sequence_errors = 0
        self.first_audio_time: float | None = None
        self.first_partial_latency: float | None = None
        self.update_times: list[float] = []
        self.commit_events: list[dict] = []
        self.processing_s = 0.0
        self.encoder_audio_s = 0.0
        self.decoded_audio_s = 0.0
        self.encoder_calls = 0
        self.speech_seen = False
        self.silence_s = 0.0
        self.eou_sent = False
        self.eou_silence_s = float(eou_silence_s)
        self.speech_rms = float(speech_rms)
        self.expect_sequence = bool(expect_sequence)
        self.capture_logits = bool(capture_logits)
        self.logits_chunks: list[np.ndarray] = []
        self.logit_frame_ranges: list[tuple[int, int]] = []

    @property
    def audio_duration_s(self) -> float:
        return (self.buffer_start_sample + len(self.audio)) / SAMPLE_RATE

    def append_pcm(self, payload: bytes, sequence: int | None = None) -> list[dict]:
        if self.expect_sequence:
            if sequence is None:
                raise StreamingProtocolError("Sequence number is required.")
            if sequence != self.expected_sequence:
                self.sequence_errors += 1
                raise StreamingProtocolError(
                    f"Expected audio chunk {self.expected_sequence}, got {sequence}."
                )
            self.expected_sequence += 1
        chunk = prepare_pcm_f32(payload, self.sample_rate)
        if chunk.size == 0:
            return []
        if self.first_audio_time is None:
            self.first_audio_time = time.perf_counter()
        self.audio = np.concatenate([self.audio, chunk])
        self.chunks_received += 1
        rms = chunk_rms(chunk)
        if rms >= self.speech_rms:
            self.speech_seen = True
            self.silence_s = 0.0
        else:
            self.silence_s += len(chunk) / SAMPLE_RATE
        events = self._process_available()
        if (
            self.speech_seen
            and self.silence_s >= self.eou_silence_s
            and not self.eou_sent
        ):
            self.eou_sent = True
            events.append({
                "type": "eou",
                "silence_s": self.silence_s,
                "text": self.decoder.snapshot()["text"],
                "note": "Energy-based end-of-utterance hint.",
            })
        return events

    def _window_abs_end(self) -> int:
        return self.buffer_start_sample + len(self.audio)

    def _decode_segment(self, seg_start: int, seg_end: int, window_end: int) -> dict | None:
        window_start = max(0, seg_start - self.left_samples)
        if window_end <= window_start:
            return None
        window = self.audio[
            window_start - self.buffer_start_sample:
            window_end - self.buffer_start_sample
        ]
        if len(window) < 320:
            return None
        result = self.manager.logits_array(window, self.model_id, sample_rate=SAMPLE_RATE)
        self.processing_s += result["latency_s"]
        self.encoder_calls += 1
        self.encoder_audio_s += len(window) / SAMPLE_RATE
        self.decoded_audio_s += (seg_end - seg_start) / SAMPLE_RATE
        offset = (0 if seg_start == window_start
                  else self.manager.feat_frames(seg_start - window_start))
        end = self.manager.feat_frames(seg_end - window_start)
        offset = max(0, min(offset, result["logits"].shape[0]))
        end = max(offset, min(end, result["logits"].shape[0]))
        current_logits = result["logits"][offset:end]
        if current_logits.shape[0] == 0:
            return None
        if self.capture_logits:
            self.logits_chunks.append(np.array(current_logits, copy=True))
            self.logit_frame_ranges.append((offset, end))
        state = self.decoder.feed_logits(current_logits)
        state["text"] = normalize_text(state["text"])
        state["final_text"] = normalize_text(state["final_text"])
        state["partial_text"] = normalize_text(state["partial_text"])
        state["new_committed"] = normalize_text(state["new_committed"])
        now = time.perf_counter()
        self.update_times.append(now)
        if self.first_partial_latency is None and state["text"].strip():
            self.first_partial_latency = now - (self.first_audio_time or now)
        if state["new_committed"]:
            self.commit_events.append({
                "word": state["new_committed"],
                "audio_s": seg_end / SAMPLE_RATE,
                "wall_s": now - (self.first_audio_time or now),
            })
        return {
            "state": state,
            "latency_s": result["latency_s"],
            "window_s": len(window) / SAMPLE_RATE,
            "seg_start_s": seg_start / SAMPLE_RATE,
            "seg_end_s": seg_end / SAMPLE_RATE,
        }

    def _process_available(self) -> list[dict]:
        events = []
        while True:
            seg_start = self.next_segment_start
            seg_end = seg_start + self.current_samples
            window_start = max(0, seg_start - self.left_samples)
            window_end = seg_end + self.right_samples
            if window_end > self._window_abs_end():
                break
            decoded = self._decode_segment(seg_start, seg_end, window_end)
            if decoded is None:
                break
            state = decoded["state"]
            if state["new_committed"]:
                events.append({
                    "type": "final_segment",
                    "text": state["new_committed"],
                    "final_text": state["final_text"],
                    "seg_start_s": decoded["seg_start_s"],
                    "seg_end_s": decoded["seg_end_s"],
                })
            events.append({
                "type": "partial",
                "text": state["text"],
                "final_text": state["final_text"],
                "partial_text": state["partial_text"],
                "chunk": len(self.update_times),
                "seg_start_s": decoded["seg_start_s"],
                "seg_end_s": decoded["seg_end_s"],
                "encoder_window_s": decoded["window_s"],
                "latency_s": decoded["latency_s"],
                "rtf": (self.processing_s / decoded["seg_end_s"]) if decoded["seg_end_s"] else None,
                "audio_duration_s": self.audio_duration_s,
                "mode": "stateful-bounded-context",
                "committed_words": state["committed_words"],
                "provisional_chars": state["provisional_chars"],
                "revisions": state["revisions"],
            })
            self.next_segment_start = seg_end
            keep_from = max(0, self.next_segment_start - self.left_samples)
            if keep_from > self.buffer_start_sample:
                drop = keep_from - self.buffer_start_sample
                self.audio = self.audio[drop:]
                self.buffer_start_sample = keep_from
        return events

    def finalize(self, reference: str = "") -> dict:
        t0 = time.perf_counter()
        events = self._process_available()
        audio_end = self._window_abs_end()
        while self.next_segment_start < audio_end:
            seg_start = self.next_segment_start
            seg_end = min(seg_start + self.current_samples, audio_end)
            if seg_end - seg_start < int(0.2 * SAMPLE_RATE):
                break
            decoded = self._decode_segment(seg_start, seg_end, audio_end)
            if decoded is None:
                break
            state = decoded["state"]
            if state["new_committed"]:
                events.append({
                    "type": "final_segment",
                    "text": state["new_committed"],
                    "final_text": state["final_text"],
                    "seg_start_s": decoded["seg_start_s"],
                    "seg_end_s": decoded["seg_end_s"],
                })
            self.next_segment_start = seg_end
        decoder_final = self.decoder.finalize()
        final_text = normalize_text(decoder_final["text"])
        intervals = [b - a for a, b in zip(self.update_times, self.update_times[1:])]
        p95_update = float(np.percentile(intervals, 95)) if intervals else None
        stabilization = [e["wall_s"] for e in self.commit_events]
        finalization_s = time.perf_counter() - t0
        return {
            "type": "final",
            "text": final_text,
            "final_text": final_text,
            "streaming_text": final_text,
            "refined_text": final_text,
            "metrics": compute_reference_metrics(reference, final_text),
            "streaming_metrics": compute_reference_metrics(reference, final_text),
            "mode": "stateful-bounded-context",
            "exact_offline_equivalence": False,
            "audio_duration_s": self.audio_duration_s,
            "chunks_received": self.chunks_received,
            "sequence_errors": self.sequence_errors,
            "current_s": self.current_s,
            "left_context_s": self.left_s,
            "right_context_s": self.right_s,
            "chunk_ms": self.chunk_ms,
            "partial_updates": len(self.update_times),
            "first_partial_latency_s": self.first_partial_latency,
            "average_update_interval_s": (
                sum(intervals) / len(intervals) if intervals else None
            ),
            "p95_update_interval_s": p95_update,
            "word_stabilization_latency_s": (
                sum(stabilization) / len(stabilization) if stabilization else None
            ),
            "finalization_latency_s": finalization_s,
            "processing_s": self.processing_s,
            "rtf": self.processing_s / self.audio_duration_s if self.audio_duration_s else None,
            "encoder_calls": self.encoder_calls,
            "encoder_audio_s": self.encoder_audio_s,
            "decoded_audio_s": self.decoded_audio_s,
            "encoder_recompute_ratio": (
                self.encoder_audio_s / self.decoded_audio_s if self.decoded_audio_s else None
            ),
            "committed_words": decoder_final["committed_words"],
            "provisional_chars": decoder_final["provisional_chars"],
            "revisions": decoder_final["revisions"],
            "commit_events": self.commit_events,
            "dropped_chunks": 0,
            "events_before_final": events,
            "limitation": (
                "Bounded-context stateful approximation: bidirectional Conformer "
                "attention is evaluated over left/current/right windows; no exact "
                "KV-cache equivalence is claimed. The final transcript is the "
                "incremental decoder output, not a full-utterance refinement."
            ),
        }

    def reset(self) -> None:
        self.__init__(
            self.manager,
            self.model_id,
            self.tokenizer,
            sample_rate=self.sample_rate,
            chunk_ms=self.chunk_ms,
            current_s=self.current_s,
            left_s=self.left_s,
            right_s=self.right_s,
            eou_silence_s=self.eou_silence_s,
            speech_rms=self.speech_rms,
            expect_sequence=self.expect_sequence,
            capture_logits=self.capture_logits,
        )
