"""FastAPI application for the live chunked-streaming ASR demo."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from onebit_asr.serving.audio import prepare_audio_file
from onebit_asr.serving.manager import ModelManager, ServingError
from onebit_asr.serving.metrics import compute_reference_metrics
from onebit_asr.serving.streaming import StreamingSession
from onebit_asr.serving.stateful_streaming import StatefulStreamingSession, StreamingProtocolError

ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "frontend"


def create_app(manager: ModelManager | None = None) -> FastAPI:
    app = FastAPI(title="One-Bit ASR Streaming Demo", version="1.0.0")
    app.state.manager = manager or ModelManager()
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "status": app.state.manager.status()}

    @app.get("/api/models")
    def models():
        return {"models": app.state.manager.available_models(),
                "status": app.state.manager.status()}

    @app.get("/api/status")
    def status():
        return app.state.manager.status()

    @app.post("/api/metrics")
    def metrics_endpoint(request: dict = Body(...)):
        return compute_reference_metrics(
            str(request.get("reference", "")),
            str(request.get("hypothesis", "")),
        )

    @app.post("/api/transcribe")
    async def transcribe_file(
        audio: UploadFile = File(...),
        model_id: str = Form("ffn2bit"),
        reference: str = Form(""),
    ):
        try:
            payload = await audio.read()
            prepared = await asyncio.to_thread(prepare_audio_file, payload)
            result = await asyncio.to_thread(
                app.state.manager.transcribe_array,
                prepared["audio"],
                model_id,
                prepared["sample_rate"],
            )
            result["audio_metadata"] = {
                "duration_s": prepared["duration_s"],
                "original_sample_rate": prepared["original_sample_rate"],
                "channels": prepared["channels"],
            }
            result["metrics"] = compute_reference_metrics(
                reference, result["transcript"]
            )
            return JSONResponse(result)
        except ServingError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"Transcription failed: {exc}"},
                                status_code=400)

    @app.websocket("/ws/stream")
    async def stream(websocket: WebSocket):
        await websocket.accept()
        query = websocket.query_params
        model_id = query.get("model_id", "ffn2bit")
        device = query.get("device", "auto")
        chunk_ms = int(query.get("chunk_ms", "320"))
        window_s = float(query.get("window_s", "2.0"))
        overlap_s = float(query.get("overlap_s", "0.5"))
        current_s = float(query.get("current_s", "1.0"))
        left_s = float(query.get("left_s", "0.8"))
        right_s = float(query.get("right_s", "0.32"))
        sample_rate = int(query.get("sample_rate", "16000"))
        streaming_version = query.get("streaming_version", "stateful")
        seq_header = query.get("seq_header", "1") == "1"
        reference = ""
        try:
            await websocket.send_json({
                "type": "status",
                "status": "loading_model",
                "model_id": model_id,
            })
            app.state.manager.set_device(device)
            load_info = await asyncio.to_thread(app.state.manager.load, model_id)
            model_meta = app.state.manager.public_metadata(model_id)
            if streaming_version == "stateful":
                session = StatefulStreamingSession(
                    app.state.manager,
                    model_id,
                    app.state.manager.tokenizer(),
                    sample_rate=sample_rate,
                    chunk_ms=chunk_ms,
                    current_s=current_s,
                    left_s=left_s,
                    right_s=right_s,
                    expect_sequence=seq_header,
                )
                mode = "stateful-bounded-context"
                note = (
                    "Stateful bounded-context approximation: persistent CTC state, "
                    "non-overlapping current segments, full Conformer over left/current/right "
                    "context. Exact KV-cache equivalence is not claimed."
                )
                config = {
                    "chunk_ms": chunk_ms, "current_s": current_s,
                    "left_s": left_s, "right_s": right_s,
                    "sample_rate": sample_rate, "mode": mode,
                }
            else:
                session = StreamingSession(
                    app.state.manager,
                    model_id,
                    sample_rate=sample_rate,
                    chunk_ms=chunk_ms,
                    window_s=window_s,
                    overlap_s=overlap_s,
                )
                mode = "chunked-overlap"
                note = (
                    "Legacy chunked approximation: full encoder per overlapping window. "
                    "Final text is a full-utterance refinement."
                )
                config = {
                    "chunk_ms": chunk_ms, "window_s": window_s,
                    "overlap_s": overlap_s, "sample_rate": sample_rate, "mode": mode,
                }
            await websocket.send_json({
                "type": "ready",
                "model": model_meta,
                "load": load_info,
                "config": config,
                "note": note,
            })
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    payload_bytes = message["bytes"]
                    sequence = None
                    if streaming_version == "stateful" and seq_header:
                        if len(payload_bytes) < 4:
                            raise StreamingProtocolError(
                                "Audio frame is missing its 4-byte sequence header."
                            )
                        sequence = int.from_bytes(payload_bytes[:4], "little")
                        payload_bytes = payload_bytes[4:]
                    events = await asyncio.to_thread(
                        session.append_pcm, payload_bytes, sequence
                    ) if streaming_version == "stateful" else await asyncio.to_thread(
                        session.append_pcm, payload_bytes
                    )
                    for event in events:
                        await websocket.send_json(event)
                elif "text" in message and message["text"] is not None:
                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "error",
                                                   "message": "Invalid JSON control message."})
                        continue
                    kind = payload.get("type")
                    if kind == "start":
                        reference = payload.get("reference", reference)
                    elif kind == "reference":
                        reference = payload.get("text", "")
                    elif kind == "reset":
                        session.reset()
                        reference = ""
                        await websocket.send_json({"type": "reset"})
                    elif kind == "stop":
                        reference = payload.get("reference", reference)
                        final = await asyncio.to_thread(session.finalize, reference)
                        await websocket.send_json(final)
                        break
                    elif kind == "ping":
                        await websocket.send_json({"type": "pong"})
            await websocket.close()
        except WebSocketDisconnect:
            return
        except (ServingError, StreamingProtocolError) as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
                await websocket.close()
            except Exception:  # noqa: BLE001
                return
        except Exception as exc:  # noqa: BLE001
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Streaming session failed: {exc}",
                })
                await websocket.close()
            except Exception:  # noqa: BLE001
                return

    return app


app = create_app()
