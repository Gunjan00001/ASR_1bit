"""Online chunked-streaming session with stable-prefix commits.

This is intentionally named an approximation: the Wav2Vec2-Conformer encoder
has no cached attention/convolution state here. Each window runs the full
encoder. The final event also exposes a full-utterance refinement, which is
labeled separately from the streamed hypothesis.
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
from onebit_asr.serving.metrics import compute_reference_metrics
from onebit_asr.streaming import merge_hypotheses


def common_word_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for left, right in zip(a, b):
        if left.lower() != right.lower():
            break
        out.append(left)
    return out


class StablePrefix:
    """Commit only words stable across consecutive partial hypotheses."""

    def __init__(self, holdback_words: int = 2) -> None:
        self.holdback_words = max(0, int(holdback_words))
        self.committed: list[str] = []
        self.previous: list[str] | None = None
        self.last_segment: list[str] = []

    def update(self, text: str) -> tuple[str, str, list[str]]:
        words = text.split()
        if self.previous is None:
            common = []
        else:
            common = common_word_prefix(self.previous, words)
        target = max(0, len(common) - self.holdback_words)
        segment = []
        if target > len(self.committed):
            segment = common[len(self.committed):target]
            self.committed = common[:target]
        self.previous = words
        self.last_segment = segment
        partial = words[len(self.committed):]
        return " ".join(self.committed), " ".join(partial), segment

    def finalize(self, text: str) -> tuple[str, str]:
        words = text.split()
        self.committed = words
        self.previous = words
        self.last_segment = []
        return " ".join(words), ""


class StreamingSession:
    def __init__(
        self,
        manager,
        model_id: str,
        *,
        sample_rate: int = SAMPLE_RATE,
        chunk_ms: int = 320,
        window_s: float = 2.0,
        overlap_s: float = 0.5,
        eou_silence_s: float = 0.8,
        speech_rms: float = 0.008,
    ) -> None:
        self.manager = manager
        self.model_id = model_id
        self.sample_rate = int(sample_rate)
        self.chunk_ms = int(chunk_ms)
        self.window_s = float(window_s)
        self.overlap_s = float(overlap_s)
        if not (0 <= overlap_s < window_s):
            raise ValueError("overlap_s must satisfy 0 <= overlap_s < window_s")
        self.window_samples = int(round(window_s * SAMPLE_RATE))
        self.step_samples = int(round((window_s - overlap_s) * SAMPLE_RATE))
        if self.window_samples <= 0 or self.step_samples <= 0:
            raise ValueError("window/overlap produce non-positive sample counts")
        self.audio = np.empty(0, dtype=np.float32)
        self.next_start = 0
        self.chunks_received = 0
        self.merged_text = ""
        self.stable = StablePrefix()
        self.partial_count = 0
        self.update_times: list[float] = []
        self.processing_s = 0.0
        self.first_audio_time: float | None = None
        self.first_partial_latency: float | None = None
        self.speech_seen = False
        self.silence_s = 0.0
        self.eou_sent = False
        self.eou_silence_s = float(eou_silence_s)
        self.speech_rms = float(speech_rms)
        self.started_at = time.perf_counter()

    @property
    def audio_duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    def append_pcm(self, payload: bytes) -> list[dict]:
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
                "text": self.merged_text,
                "note": "Energy-based end-of-utterance hint; the client may stop recording.",
            })
        return events

    def _decode_window(self, window: np.ndarray, start_s: float) -> dict:
        result = self.manager.transcribe_array(
            window, self.model_id, sample_rate=SAMPLE_RATE
        )
        self.processing_s += result["latency_s"]
        return result

    def _process_available(self) -> list[dict]:
        events = []
        while self.next_start + self.window_samples <= len(self.audio):
            start = self.next_start
            end = start + self.window_samples
            window = self.audio[start:end]
            result = self._decode_window(window, start / SAMPLE_RATE)
            chunk_text = normalize_text(result["transcript"])
            if chunk_text:
                self.merged_text = merge_hypotheses(self.merged_text, chunk_text)
            committed, partial, segment = self.stable.update(self.merged_text)
            now = time.perf_counter()
            self.partial_count += 1
            self.update_times.append(now)
            if self.first_partial_latency is None and self.merged_text.strip():
                self.first_partial_latency = now - (self.first_audio_time or now)
            if segment:
                events.append({
                    "type": "final_segment",
                    "text": " ".join(segment),
                    "final_text": committed,
                    "window_start_s": start / SAMPLE_RATE,
                    "window_end_s": end / SAMPLE_RATE,
                })
            events.append({
                "type": "partial",
                "text": self.merged_text,
                "final_text": committed,
                "partial_text": partial,
                "chunk": self.partial_count,
                "window_start_s": start / SAMPLE_RATE,
                "window_end_s": end / SAMPLE_RATE,
                "latency_s": result["latency_s"],
                "rtf": result["rtf"],
                "audio_duration_s": self.audio_duration_s,
                "mode": "chunked-overlap",
            })
            self.next_start += self.step_samples
        return events

    def finalize(self, reference: str = "") -> dict:
        t0 = time.perf_counter()
        events = self._process_available()
        while self.next_start < len(self.audio):
            start = self.next_start
            end = min(start + self.window_samples, len(self.audio))
            if end - start < int(0.2 * SAMPLE_RATE):
                break
            result = self._decode_window(self.audio[start:end], start / SAMPLE_RATE)
            chunk_text = normalize_text(result["transcript"])
            if chunk_text:
                self.merged_text = merge_hypotheses(self.merged_text, chunk_text)
            self.next_start += self.step_samples
        streaming_text = self.merged_text.strip()
        refined_text = ""
        if len(self.audio) >= int(0.2 * SAMPLE_RATE):
            refined = self.manager.transcribe_array(
                self.audio, self.model_id, sample_rate=SAMPLE_RATE
            )
            refined_text = refined["transcript"]
            self.processing_s += refined["latency_s"]
        final_text = refined_text or streaming_text
        self.stable.finalize(final_text)
        finalization_s = time.perf_counter() - t0
        intervals = [
            b - a for a, b in zip(self.update_times, self.update_times[1:])
        ]
        return {
            "type": "final",
            "text": final_text,
            "final_text": final_text,
            "streaming_text": streaming_text,
            "refined_text": refined_text,
            "metrics": compute_reference_metrics(reference, final_text),
            "streaming_metrics": compute_reference_metrics(reference, streaming_text),
            "mode": "chunked-overlap + full-utterance refinement",
            "audio_duration_s": self.audio_duration_s,
            "chunks_received": self.chunks_received,
            "window_s": self.window_s,
            "overlap_s": self.overlap_s,
            "chunk_ms": self.chunk_ms,
            "partial_updates": self.partial_count,
            "first_partial_latency_s": self.first_partial_latency,
            "average_update_interval_s": (
                sum(intervals) / len(intervals) if intervals else None
            ),
            "finalization_latency_s": finalization_s,
            "processing_s": self.processing_s,
            "rtf": self.processing_s / self.audio_duration_s if self.audio_duration_s else None,
            "dropped_chunks": 0,
            "eou_silence_s": self.eou_silence_s,
            "events_before_final": events,
            "limitation": (
                "Not true stateful streaming: every window runs the full encoder. "
                "The final text is a full-utterance refinement; streaming_text is "
                "the chunk-merged hypothesis."
            ),
        }

    def reset(self) -> None:
        self.__init__(
            self.manager,
            self.model_id,
            sample_rate=self.sample_rate,
            chunk_ms=self.chunk_ms,
            window_s=self.window_s,
            overlap_s=self.overlap_s,
            eou_silence_s=self.eou_silence_s,
            speech_rms=self.speech_rms,
        )
