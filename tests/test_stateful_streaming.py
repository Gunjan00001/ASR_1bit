import numpy as np
import pytest

from onebit_asr.serving.stateful_streaming import (
    StatefulStreamingSession,
    StreamingProtocolError,
)


class Tok:
    def get_vocab(self):
        return {"<pad>": 0, "A": 1, "|": 4}


class FakeManager:
    def __init__(self):
        self.calls = 0

    def feat_frames(self, samples):
        return max(1, int(round(samples / 320)))

    def logits_array(self, audio, model_id, sample_rate=16000):
        self.calls += 1
        frames = max(1, len(audio) // 320)
        logits = np.full((frames, 5), -10.0, dtype=np.float32)
        logits[0, 1] = 10.0
        logits[-1, 4] = 10.0
        return {"logits": logits, "latency_s": 0.01,
                "audio_duration_s": len(audio) / 16000, "device": "cpu"}


def pcm(seconds=0.2):
    return np.zeros(int(16000 * seconds), dtype="<f4").tobytes()


def test_stateful_session_tracks_state_and_encoder_calls():
    session = StatefulStreamingSession(
        FakeManager(), "fake", Tok(), current_s=0.2, left_s=0.0, right_s=0.0
    )
    session.append_pcm(pcm(), sequence=0)
    session.append_pcm(pcm(), sequence=1)
    final = session.finalize("a")
    assert final["type"] == "final"
    assert final["text"] == "a a"
    assert final["encoder_calls"] >= 2
    assert final["encoder_recompute_ratio"] == 1.0
    assert final["sequence_errors"] == 0


def test_stateful_sequence_gap_is_rejected_and_reported():
    session = StatefulStreamingSession(
        FakeManager(), "fake", Tok(), current_s=0.2, left_s=0.0, right_s=0.0
    )
    with pytest.raises(StreamingProtocolError, match="Expected audio chunk 0"):
        session.append_pcm(pcm(), sequence=1)
    assert session.sequence_errors == 1


def test_stateful_reset_and_independent_sessions():
    manager = FakeManager()
    a = StatefulStreamingSession(manager, "a", Tok(), current_s=0.2, left_s=0, right_s=0)
    b = StatefulStreamingSession(manager, "b", Tok(), current_s=0.2, left_s=0, right_s=0)
    a.append_pcm(pcm(), sequence=0)
    a.finalize("a")
    assert a.decoder.snapshot()["text"]
    assert b.decoder.snapshot()["text"] == ""
    a.reset()
    assert a.decoder.snapshot()["text"] == ""
    assert a.expected_sequence == 0
