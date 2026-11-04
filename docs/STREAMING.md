# Live Streaming ASR Demo

## What this is

A browser-based research interface for the selected Wav2Vec2-Conformer CTC
models. Microphone audio is captured with an **AudioWorklet**, sent as
sequence-numbered float32 PCM over a **WebSocket**, decoded by a persistent
incremental CTC decoder, and streamed back as provisional and committed words.

The encoder is **not** a stateful streaming encoder. Wav2Vec2-Conformer uses
bidirectional self-attention; exact KV-cache streaming would change its trained
semantics. The default path is the best justified bounded-context approximation:

- a **current segment** is emitted once;
- the segment is encoded with persistent **left** and **right** audio context;
- the encoder is not re-run over the full history;
- the CTC decoder and audio/context state persist across chunks;
- the final transcript is the incremental decoder output, not a separate full
  offline pass.

The browser labels both the mode and its limitation.

## Architecture

```text
Browser microphone
  → AudioWorklet (320 ms float32 PCM; 4-byte sequence header)
  → WebSocket /ws/stream?streaming_version=stateful
  → ModelManager (lazy model; serialized inference)
  → StatefulStreamingSession
       current segment: 1.0 s
       left context:    0.8 s
       right context:   0.32 s
       encoder call per non-overlapping current segment
       monotonic feature-frame offset selection
       IncrementalCTCDecoder (previous frame token, blank/repeat state,
         current word, committed words, flush at end)
  → partial / final_segment / eou / final JSON events
  → UI: committed words dark; provisional chars amber
```

The model receives a bounded left/current/right window. It does not receive the
full history, but it also does not have exact stateful attention. The measured
encoder recomputation ratio is **1.85×** the current audio for the default
settings, versus repeated full-history recomputation.

### Included context

| Boundary | Required context | State |
|---|---|---|
| Feature extractor | short convolution context | bounded raw window |
| Conformer attention | bidirectional over the window | left/current/right |
| Conformer convolution | local temporal context | same window |
| CTC decoder | previous frame token | persistent |
| Word commitment | tokenizer `|` delimiter | persistent |
| Final flush | last partial word | at stop |

An **exact** KV-cache implementation would require a causal/chunkwise attention
mask or a retrained streaming-compatible encoder. Neither is claimed here.

## Models

The backend uses the selected standalone v2 artifacts:

- 2-bit FFN: 1,131,269,460 B, 79 FFN layers 2-bit, 17 FP32 FFN exclusions.
- Strategy A: 989,716,588 B, 36 attention + 79 FFN 2-bit; 60 attention + 17
  FFN FP32.
- FP32 reference: external `facebook/wav2vec2-conformer-rope-large-960h-ft`.

Packed weights are expanded to FP32 before `F.linear`; this is storage
compression, not an inference accelerator.

## Launch

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
pip install -r requirements-demo.txt
# Install PyTorch/torchaudio for the target CUDA or CPU platform if needed.
python scripts/run_streaming_app.py --host 127.0.0.1 --port 7860 --device auto
```

Open `http://127.0.0.1:7860`, choose a model, click **Start recording**, allow
microphone permission, and speak. Chrome/Edge/Firefox with AudioWorklet and
microphone support are required.

The optional legacy `streaming_version=chunked` path remains available for
comparison; the UI uses `stateful`.

### Local model resolution

The demo looks for `deploy/models/<id>/model_packed.safetensors`; if absent it
uses the verified local v2 artifact under `checkpoints/`. For a clean clone,
place the LFS files in `deploy/models/` (see `deploy/models/README.md`). It never
silently substitutes a different model.

## Controls and privacy

- Start/stop microphone, reset, copy, export TXT, export metrics JSON.
- Session timer, live RMS meter, audio duration.
- Committed vs provisional words; first-partial/current update latency; RTF;
  encoder recomputation ratio.
- Optional reference transcript → WER/CER and S/D/I via the research
  normalization (`lowercase`, punctuation stripped, whitespace collapsed).
- Optional local feedback preview; no feedback is transmitted or persisted.
- Audio and transcripts are held in memory only for the active session.

## API

- `GET /api/models`
- `GET /api/health`, `GET /api/status`
- `POST /api/transcribe` (optional file check; WAV/FLAC/OGG/MP3; M4A requires
  ffmpeg)
- `POST /api/metrics`
- `WS /ws/stream?model_id=...&device=...&chunk_ms=320&current_s=1.0&left_s=0.8&right_s=0.32&streaming_version=stateful&seq_header=1`

Binary frames are `uint32 little-endian sequence` + float32 PCM. Missing,
duplicate, or out-of-order chunks are rejected with an explicit error. Final
events are flushed by the incremental decoder.

## Measured stateful evaluation

Four dev-clean utterances on CPU, 320 ms chunks, current 1.0 s / left 0.8 s /
right 0.32 s (`results/stateful_eval_cpu.json`; the GPU job was blocked by the
account's four-job limit):

| Model | Final WER | Partial WER | Stable-prefix WER | Offline WER | First partial | Avg / p95 update | Finalization | RTF | Encoder ratio | Frame agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2-bit FFN | 3.57% | 28.57% | 32.14% | 3.57% | 2.65 s | 2.27 / 2.73 s | 2.58 s | 2.81 | 1.85× | 95.5% |
| Strategy A | 3.57% | 28.57% | 32.14% | 3.57% | 2.70 s | 2.23 / 2.45 s | 2.73 s | 2.76 | 1.85× | 96.1% |

- Final transcript WER equals the offline reference on this four-utterance
  sample; it is **not** a statistically powered result.
- Partial/stable WER is still poor and is reported separately. The UI does not
  label the final text as a streaming partial.
- Max absolute logit differences are large (up to 10.73/8.68) because the
  bounded-context encoder is not offline-equivalent; frame argmax agreement is
  95.5–96.1%. Exact stateful equivalence is not claimed.
- CPU RTF is >1; this is not a real-time claim. The GPU streaming benchmark was
  not executed because the account's four-job GPU limit was full. The offline
  A100 benchmark RTF (0.137 FFN / 0.153 A) is not a streaming measurement.

The real WebSocket/AudioWorklet path was exercised with Chrome's fake microphone
(not an accuracy result) and with a direct dev-utterance stream; the direct
2.595 s stream returned the exact transcript for both models.

## Legacy comparison

The old overlapping-window session remains in `src/onebit_asr/serving/streaming.py`
and is selectable with `streaming_version=chunked`. It re-runs the encoder on
each overlapping window and performs a full-utterance refinement on stop. It is
not the default and is not described as stateful.

## Troubleshooting

- **Microphone permission denied:** allow microphone access; `https` or
  `localhost` is required by browser media APIs.
- **No AudioWorklet:** use a current Chrome, Edge, or Firefox.
- **Model not available:** fetch the LFS file or restore the local v2 artifact;
  the backend will not substitute another model.
- **Very slow live updates:** CPU fallback; packed weights expand to FP32. Use a
  CUDA device for a live demo.
- **M4A file rejected:** no ffmpeg backend; convert to WAV/FLAC/OGG/MP3.
- **Final text differs from provisional:** expected; the incremental decoder
  commits only delimited words and flushes the final word at stop.
- **LFS download fails:** the selected pair is ~2.12 GB, above GitHub's free 1 GB
  allowance; use a paid data pack or a documented release fallback.
