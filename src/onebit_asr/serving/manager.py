"""Lazy, single-model serving manager for the streaming demo."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
from pathlib import Path

import torch
from transformers import Wav2Vec2ConformerForCTC, Wav2Vec2Processor

from onebit_asr.data.text import normalize_text
from onebit_asr.serving.audio import SAMPLE_RATE, resample_mono
from onebit_asr.serving.packed_runtime import load_packed_model

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPLOY = ROOT / "deploy/models"


class ServingError(RuntimeError):
    """User-facing backend error."""


class ModelManager:
    """Load one model at a time; reuse it across requests; serialize inference."""

    def __init__(self, deploy_root: str | Path | None = None,
                 device: str = "auto") -> None:
        import os

        self.deploy_root = Path(deploy_root or os.environ.get("ONEBIT_DEPLOY_DIR", DEFAULT_DEPLOY))
        self._lock = threading.RLock()
        self._model = None
        self._processor = None
        self._loaded_id: str | None = None
        self._load_info: dict = {}
        self._load_time_s = 0.0
        self.device = self._resolve_device(device)
        self._metadata = self._read_metadata()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        device = (device or "auto").lower()
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise ServingError("CUDA was requested but is not available.")
            return torch.device("cuda")
        if device == "cpu":
            return torch.device("cpu")
        raise ServingError(f"Unknown device {device!r}; use auto, cpu, or cuda.")

    def _read_metadata(self) -> dict[str, dict]:
        models = {}
        if not self.deploy_root.is_dir():
            return models
        for path in sorted(self.deploy_root.iterdir()):
            meta_path = path / "metadata.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            if meta.get("kind") not in {"fp32", "packed"}:
                continue
            models[meta["id"]] = {**meta, "package_dir": str(path)}
        return models

    def available_models(self) -> list[dict]:
        order = ["ffn2bit", "strategy_a", "fp32"]
        ids = [x for x in order if x in self._metadata]
        ids += [x for x in self._metadata if x not in ids]
        return [self.public_metadata(x) for x in ids]

    def public_metadata(self, model_id: str) -> dict:
        if model_id not in self._metadata:
            raise ServingError(f"Unknown model {model_id!r}.")
        meta = self._metadata[model_id]
        artifact = self._resolve_artifact_path(meta)
        return {
            "id": meta["id"],
            "display_name": meta.get("display_name", meta["id"]),
            "description": meta.get("description", ""),
            "kind": meta["kind"],
            "artifact_bytes": meta.get("artifact_bytes"),
            "historical_artifact_bytes": meta.get("historical_artifact_bytes"),
            "artifact_available": True if meta["kind"] == "fp32" else artifact is not None,
            "quantization": meta.get("quantization"),
            "full_test_wer": meta.get("full_test_wer"),
            "dev_wer": meta.get("dev_wer"),
            "dev_wer_note": meta.get("dev_wer_note"),
            "limitations": meta.get("limitations", []),
            "runtime": meta.get(
                "runtime",
                "FP32 expansion before F.linear" if meta["kind"] == "packed" else "FP32",
            ),
            "checkpoint_sha256": meta.get("checkpoint_sha256"),
            "artifact_sha256": meta.get("artifact_sha256"),
        }

    def _resolve_artifact_path(self, meta: dict) -> Path | None:
        if meta["kind"] == "fp32":
            return None
        package = Path(meta["package_dir"]) / "model_packed.safetensors"
        if package.is_file():
            return package
        fallback = meta.get("local_fallback_artifact")
        if fallback:
            path = ROOT / fallback
            if path.is_file():
                return path
        return None

    def _resolve_model_dir(self, meta: dict) -> Path:
        package = Path(meta["package_dir"])
        if (package / "config.json").is_file():
            return package
        checkpoint = meta.get("checkpoint")
        if checkpoint:
            return ROOT / checkpoint
        raise ServingError(f"No config for model {meta['id']!r}.")

    def set_device(self, device: str) -> torch.device:
        with self._lock:
            resolved = self._resolve_device(device)
            if resolved != self.device:
                self.unload()
                self.device = resolved
            return self.device

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            self._loaded_id = None
            self._load_info = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, model_id: str) -> dict:
        with self._lock:
            if model_id not in self._metadata:
                raise ServingError(f"Unknown model {model_id!r}.")
            if self._loaded_id == model_id and self._model is not None:
                return {
                    "model_id": model_id,
                    "cached": True,
                    "load_time_s": 0.0,
                    **self._load_info,
                    "device": str(self.device),
                }
            self.unload()
            meta = self._metadata[model_id]
            model_dir = self._resolve_model_dir(meta)
            artifact_path = self._resolve_artifact_path(meta)
            t0 = time.perf_counter()
            try:
                if meta["kind"] == "fp32":
                    model_id_hf = meta.get("model_id")
                    if not model_id_hf:
                        raise ServingError("FP32 model id missing.")
                    processor = Wav2Vec2Processor.from_pretrained(model_dir)
                    model = Wav2Vec2ConformerForCTC.from_pretrained(model_id_hf)
                    info = {
                        "artifact": None,
                        "artifact_bytes": meta.get("artifact_bytes"),
                        "kind": "fp32",
                    }
                else:
                    if artifact_path is None:
                        raise ServingError(
                            f"Packed artifact for {model_id} is not present. "
                            "Download the Git LFS file or use the local v2 artifact."
                        )
                    processor = Wav2Vec2Processor.from_pretrained(model_dir)
                    model, info = load_packed_model(model_dir, artifact_path, self.device)
                    info["kind"] = "packed"
                model.to(self.device).eval()
                self._model = model
                self._processor = processor
                self._loaded_id = model_id
                self._load_info = info
                self._load_time_s = time.perf_counter() - t0
                return {
                    "model_id": model_id,
                    "cached": False,
                    "load_time_s": self._load_time_s,
                    "device": str(self.device),
                    **info,
                }
            except torch.cuda.OutOfMemoryError as exc:
                self.unload()
                raise ServingError(
                    "GPU memory exhausted while loading the model. "
                    "Try CPU or close other GPU jobs."
                ) from exc
            except ServingError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.unload()
                raise ServingError(f"Model load failed: {exc}") from exc

    def tokenizer(self):
        with self._lock:
            if self._processor is None:
                raise ServingError("Load a model before requesting its tokenizer.")
            return self._processor.tokenizer

    def logits_array(self, audio, model_id: str,
                     sample_rate: int = SAMPLE_RATE) -> dict:
        """Run the encoder and return frame logits without decoding.

        This is the primitive used by the stateful bounded-context session. It
        keeps the loaded model and processor across calls.
        """
        with self._lock:
            if self._loaded_id != model_id or self._model is None:
                self.load(model_id)
            if sample_rate != SAMPLE_RATE:
                audio = resample_mono(audio, sample_rate)
            audio = np.asarray(audio, dtype="float32")
            inputs = self._processor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    logits = self._model(inputs.input_values.to(self.device)).logits
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as exc:
                self.unload()
                raise ServingError(
                    "GPU memory exhausted during streaming inference. Try CPU."
                ) from exc
            elapsed = time.perf_counter() - t0
            return {
                "model_id": model_id,
                "logits": logits[0].detach().float().cpu().numpy(),
                "latency_s": elapsed,
                "audio_duration_s": len(audio) / SAMPLE_RATE,
                "device": str(self.device),
            }

    def feat_frames(self, samples: int) -> int:
        """Number of encoder frames for a raw-sample count (current model)."""
        with self._lock:
            if self._model is None:
                raise ServingError("Load a model before querying feature frames.")
            return int(
                self._model._get_feat_extract_output_lengths(
                    torch.tensor([int(samples)])
                ).item()
            )

    def transcribe_array(self, audio, model_id: str,
                         sample_rate: int = SAMPLE_RATE) -> dict:
        """Transcribe an in-memory 16 kHz mono waveform."""
        with self._lock:
            if self._loaded_id != model_id or self._model is None:
                self.load(model_id)
            if sample_rate != SAMPLE_RATE:
                audio = resample_mono(audio, sample_rate)
            audio = audio.astype("float32", copy=False)
            inputs = self._processor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    logits = self._model(inputs.input_values.to(self.device)).logits
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                hyp = self._processor.batch_decode(logits.argmax(dim=-1))[0]
            except torch.cuda.OutOfMemoryError as exc:
                self.unload()
                raise ServingError(
                    "GPU memory exhausted during inference. Try CPU."
                ) from exc
            elapsed = time.perf_counter() - t0
            audio_s = len(audio) / SAMPLE_RATE
            return {
                "model_id": model_id,
                "transcript": normalize_text(hyp),
                "audio_duration_s": audio_s,
                "latency_s": elapsed,
                "rtf": elapsed / audio_s if audio_s else None,
                "device": str(self.device),
                "artifact_bytes": self._load_info.get("artifact_bytes"),
                "quantization": self._metadata[model_id].get("quantization"),
            }

    def status(self) -> dict:
        with self._lock:
            return {
                "loaded_model_id": self._loaded_id,
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "models": self.available_models(),
            }
