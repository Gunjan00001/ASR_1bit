import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from onebit_asr.serving.api import create_app
from onebit_asr.serving.audio import (
    AudioValidationError,
    prepare_audio_file,
    prepare_pcm_f32,
)
from onebit_asr.serving.metrics import compute_reference_metrics
from onebit_asr.serving.streaming import StablePrefix


def _write_audio(path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr, subtype="PCM_16")


def test_prepare_wav_and_flac(tmp_path):
    for suffix in (".wav", ".flac"):
        path = tmp_path / f"tone{suffix}"
        _write_audio(path)
        prepared = prepare_audio_file(path)
        assert prepared["sample_rate"] == 16000
        assert abs(prepared["duration_s"] - 1.0) < 0.01
        assert prepared["channels"] == 1


def test_prepare_resamples_and_mono(tmp_path):
    path = tmp_path / "tone48.wav"
    t = np.arange(48000, dtype=np.float32) / 48000
    stereo = np.stack([0.1 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    sf.write(path, stereo, 48000, subtype="PCM_16")
    prepared = prepare_audio_file(path)
    assert prepared["original_sample_rate"] == 48000
    assert prepared["sample_rate"] == 16000
    assert prepared["channels"] == 2
    assert abs(prepared["duration_s"] - 1.0) < 0.02


def test_invalid_audio_rejected(tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    for path in (empty, tmp_path / "missing.wav"):
        try:
            prepare_audio_file(path)
        except AudioValidationError:
            pass
        else:
            raise AssertionError("invalid audio should be rejected")


def test_short_audio_rejected(tmp_path):
    path = tmp_path / "short.wav"
    _write_audio(path, seconds=0.05)
    try:
        prepare_audio_file(path)
    except AudioValidationError as exc:
        assert "too short" in str(exc)
    else:
        raise AssertionError("short audio should be rejected")


def test_pcm_roundtrip_and_resample():
    pcm = (np.sin(np.linspace(0, 2 * np.pi, 3200)) * 1000).astype("<f4").tobytes()
    out = prepare_pcm_f32(pcm, 16000)
    assert len(out) == 3200
    resampled = prepare_pcm_f32(pcm, 32000)
    assert len(resampled) == 1600


def test_reference_metrics_and_normalization():
    result = compute_reference_metrics("Hello, world!", "hello word")
    assert result is not None
    assert result["wer"] == 0.5
    assert result["substitutions"] == 1
    assert result["deletions"] == 0
    assert result["insertions"] == 0


def test_stable_prefix_commits_only_stable_words():
    tracker = StablePrefix(holdback_words=1)
    assert tracker.update("hello world") == ("", "hello world", [])
    committed, partial, segment = tracker.update("hello world again")
    assert committed == "hello"
    assert partial == "world again"
    assert segment == ["hello"]
    committed, partial, segment = tracker.update("hello world again")
    assert committed == "hello world"
    assert partial == "again"
    assert segment == ["world"]


class FakeManager:
    def status(self):
        return {"loaded_model_id": "fake", "device": "cpu",
                "cuda_available": False, "models": []}

    def available_models(self):
        return [{"id": "fake", "display_name": "Fake", "kind": "packed",
                 "artifact_available": True, "artifact_bytes": 10,
                 "quantization": None, "limitations": []}]

    def public_metadata(self, model_id):
        return self.available_models()[0]

    def set_device(self, device):
        return "cpu"

    def load(self, model_id):
        return {"model_id": model_id, "cached": False, "load_time_s": 0.1,
                "device": "cpu", "n_packed_layers": 0, "bits": 2}

    def transcribe_array(self, audio, model_id, sample_rate=16000):
        duration = len(audio) / 16000
        return {"model_id": model_id, "transcript": "hello world",
                "audio_duration_s": duration, "latency_s": 0.01,
                "rtf": 0.01 / duration if duration else None,
                "device": "cpu", "artifact_bytes": 10, "quantization": None}


def test_api_and_websocket_stream():
    app = create_app(FakeManager())
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/models").status_code == 200
        metrics = client.post("/api/metrics", json={
            "reference": "hello world", "hypothesis": "hello word"
        })
        assert metrics.status_code == 200
        assert metrics.json()["wer"] == 0.5
        with client.websocket_connect(
            "/ws/stream?model_id=fake&window_s=0.2&overlap_s=0.05"
        ) as ws:
            assert ws.receive_json()["type"] == "status"
            assert ws.receive_json()["type"] == "ready"
            pcm = np.zeros(int(16000 * 0.32), dtype="<f4").tobytes()
            ws.send_bytes(pcm)
            partial_seen = False
            for _ in range(8):
                event = ws.receive_json()
                if event["type"] == "partial":
                    partial_seen = True
                    break
            assert partial_seen
            ws.send_json({"type": "stop", "reference": "hello world"})
            final_seen = False
            for _ in range(8):
                event = ws.receive_json()
                if event["type"] == "final":
                    final_seen = True
                    assert event["text"] == "hello world"
                    assert event["metrics"]["wer"] == 0.0
                    break
            assert final_seen
