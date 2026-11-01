"""Audio decoding, validation, resampling, and level helpers for the demo."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 16_000
MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_DURATION_S = 120.0
MIN_DURATION_S = 0.20
SUPPORTED_SUFFIXES = {".wav", ".flac", ".ogg", ".oga", ".mp3"}


class AudioValidationError(ValueError):
    """Raised for unsupported, corrupt, oversized, or invalid audio."""


def _decode(path_or_bytes: str | Path | bytes) -> tuple[np.ndarray, int]:
    try:
        if isinstance(path_or_bytes, (str, Path)):
            path = Path(path_or_bytes)
            if path.stat().st_size > MAX_FILE_BYTES:
                raise AudioValidationError(
                    f"Audio file exceeds {MAX_FILE_BYTES / 1e6:.0f} MB."
                )
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise AudioValidationError(
                    f"Unsupported extension {suffix or '(none)'}. "
                    "Use WAV, FLAC, OGG, or MP3. M4A requires an ffmpeg backend."
                )
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
        else:
            if len(path_or_bytes) > MAX_FILE_BYTES:
                raise AudioValidationError(
                    f"Audio upload exceeds {MAX_FILE_BYTES / 1e6:.0f} MB."
                )
            audio, sr = sf.read(io.BytesIO(path_or_bytes), dtype="float32",
                                always_2d=True)
    except AudioValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface codec details to the UI
        raise AudioValidationError(f"Could not decode audio: {exc}") from exc
    if audio.size == 0:
        raise AudioValidationError("The audio file is empty.")
    return np.asarray(audio, dtype=np.float32), int(sr)


def resample_mono(audio: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise AudioValidationError("Audio must be mono or a channel-first 2-D array.")
    if not np.isfinite(audio).all():
        raise AudioValidationError("Audio contains NaN or infinite samples.")
    if sr != target_sr:
        waveform = torch.from_numpy(np.array(audio, copy=True)).unsqueeze(0)
        audio = torchaudio.functional.resample(waveform, sr, target_sr).squeeze(0).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def prepare_audio_file(path_or_bytes: str | Path | bytes) -> dict:
    raw, sr = _decode(path_or_bytes)
    channels = int(raw.shape[1])
    duration_original = raw.shape[0] / sr
    if duration_original > MAX_DURATION_S:
        raise AudioValidationError(
            f"Audio is {duration_original:.1f}s; limit is {MAX_DURATION_S:.0f}s."
        )
    audio = resample_mono(raw, sr)
    if len(audio) < int(MIN_DURATION_S * SAMPLE_RATE):
        raise AudioValidationError(
            f"Audio is too short ({len(audio) / SAMPLE_RATE:.2f}s); "
            f"minimum is {MIN_DURATION_S:.1f}s."
        )
    return {
        "audio": audio,
        "sample_rate": SAMPLE_RATE,
        "original_sample_rate": sr,
        "channels": channels,
        "duration_s": len(audio) / SAMPLE_RATE,
        "rms": float(np.sqrt(np.mean(np.square(audio)))),
    }


def prepare_pcm_f32(payload: bytes, sample_rate: int) -> np.ndarray:
    """Decode little-endian float32 PCM from the browser and resample to 16 kHz."""
    if not payload or len(payload) % 4:
        raise AudioValidationError("PCM chunk must be non-empty float32 bytes.")
    raw = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(raw).all():
        raise AudioValidationError("PCM chunk contains NaN or infinity.")
    if sample_rate != SAMPLE_RATE:
        raw = resample_mono(raw, sample_rate)
    return np.ascontiguousarray(raw, dtype=np.float32)


def chunk_rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def pcm16_to_float32(payload: bytes) -> np.ndarray:
    if not payload or len(payload) % 2:
        raise AudioValidationError("PCM16 chunk must be non-empty even-length bytes.")
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
