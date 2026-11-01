# Live Streaming ASR Demo

## What this is

A browser-based research interface for the selected Wav2Vec2-Conformer CTC
models. Microphone audio is captured with an **AudioWorklet**, sent as raw
float32 PCM over a **WebSocket**, decoded by the Python backend, and streamed
back as provisional and finalized words.

This is a **chunked-overlap approximation**, not a stateful streaming encoder.
The Wav2Vec2-Conformer self-attention sees a full window; the implementation
does not cache attention or convolution state. The final event additionally
runs a full-utterance refinement and labels it as such.

## Architecture

```text
Browser microphone
  → AudioWorklet (320 ms chunks, RMS level)
  → WebSocket /ws/stream
  → ModelManager (lazy, single model, serialized inference)
  → StreamingSession
       overlapping 2.0 s windows / 0.5 s overlap
       greedy CTC per window
       word-overlap merge
       stable-prefix commits (hold back 2 words)
  → partial / final_segment / final JSON events
  → UI: provisional words in amber, finalized words in white
```

On stop, the final `refined_text` is decoded from the complete utterance. The
`streaming_text` field is the chunk-merged hypothesis; the UI exposes the
limitation and both values are returned by the API. This avoids presenting the
offline refinement as a pure streaming result.

## Models

The backend uses the selected standalone v2 artifacts:

- 2-bit FFN: 1,131,269,460 B, 79 FFN layers 2-bit, 17 FP32 FFN exclusions.
- Strategy A: 989,716,588 B, 36 attention + 79 FFN 2-bit; 60 attention + 17
  FFN FP32.
- FP32 reference: external `facebook/wav2vec2-conformer-rope-large-960h-ft`.

The runtime expands packed codes to FP32 before `F.linear`; it is a storage
format, not an inference accelerator.

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

### Local model resolution

The demo looks for `deploy/models/<id>/model_packed.safetensors`; if absent it
uses the verified local v2 artifact under `checkpoints/`. For a clean clone,
place the LFS files in `deploy/models/` (see `deploy/models/README.md`). It never
silently substitutes a different model.

### GPU note

Packed models are slow on CPU: the measured 2.6 s dev utterance took 6.6 s
(FFN) / 8.4 s (A) of processing after load, with first partial latency
2.5 / 2.9 s. CPU is a functional fallback, not real-time. On the A100
benchmark, the same packed models had offline RTF 0.137 (FFN) / 0.153 (A);
streaming uses the same full-encoder window cost. Use `--device cuda` on a
machine with CUDA for a live demonstration.

## Controls and privacy

- Start/stop microphone, reset, copy, export TXT.
- Session timer, live RMS meter, audio duration.
- Provisional vs finalized words.
- Optional reference transcript → WER/CER and S/D/I via the research
  normalization (`lowercase`, punctuation stripped, whitespace collapsed).
- Optional local feedback preview; no feedback is transmitted or persisted.
- Audio and transcripts are held in memory only for the active session. No
  external service or persistence is used.

## API

- `GET /api/models`
- `GET /api/health`, `GET /api/status`
- `POST /api/transcribe` (optional file check; WAV/FLAC/OGG/MP3; M4A requires
  an ffmpeg backend)
- `POST /api/metrics` (reference/hypothesis)
- `WS /ws/stream?model_id=...&device=...&chunk_ms=320&window_s=2.0&overlap_s=0.5`

WebSocket events: `status`, `ready`, `partial`, `final_segment`, `eou`,
`final`, `error`.

## Streaming evaluation

The online session was evaluated on a 4-utterance dev-clean fallback on CPU
(`results/streaming_eval_cpu.json`), using 320 ms PCM chunks. These are
preliminary CPU measurements, not GPU claims and not test results.

| Model / window | Final WER | Streaming WER | First partial | Avg update | Finalization | RTF | Updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| FFN · 2.0 s / 0.5 s | 3.57% | 25.00% | 2.08 s | 2.43 s | 5.22 s | 2.77 | 4 |
| A · 2.0 s / 0.5 s | 3.57% | 21.43% | 2.67 s | 2.31 s | 5.18 s | 2.92 | 4 |
| FFN · 320 ms window | 3.57% | 117.86% | 4.91 s | 2.04 s | 7.81 s | 11.11 | 43 |
| A · 320 ms window | 3.57% | 150.00% | 4.05 s | 2.19 s | 3.03 s | 9.43 | 43 |

`Final WER` is the full-utterance refinement and is identical across configs;
it is **not** a pure streaming metric. The 2.0 s overlap window is the app
default. The 320 ms window is rejected: its chunk-merged hypothesis is far worse
and its RTF is 9–11× on CPU. The 4-utterance sample is too small for a
statistical claim.

The GPU streaming evaluation was submitted but blocked by the account's
four-job GPU limit; no GPU streaming WER is claimed. The offline A100 benchmark
measured RTF 0.137 (FFN) / 0.153 (A) and GPU memory 1.89/1.75 GB, but that is
offline, not a streaming WER measurement.

## Streaming evaluation

The online session is evaluated separately from the offline research pipeline.
The evaluation script uses dev-clean only and the exact backend session:

```bash
python scripts/eval_streaming.py \
  --models ffn2bit,strategy_a --n 32 \
  --output results/streaming_eval.json
```

The reported metrics include final/streaming WER and CER, first-partial latency,
average update interval, finalization latency, RTF, partial updates, dropped
chunks, and audio duration. The final text is a full-utterance refinement; it is
not claimed as an equivalent pure-streaming result.

## Troubleshooting

- **Microphone permission denied:** allow microphone access for the page; `https`
  or `localhost` is required by browser media APIs.
- **No AudioWorklet:** use a current Chrome, Edge, or Firefox. The app reports
  unsupported browsers explicitly.
- **Model not available:** the packed weight may be absent after a fresh clone.
  Fetch the LFS file or restore the verified local v2 artifact; the backend will
  not substitute another model.
- **Very slow live updates:** the demo is running on CPU. Packed weights are
  expanded to FP32; use a CUDA device for a live demo.
- **M4A file rejected:** the installed backend has no ffmpeg decoder. Convert to
  WAV/FLAC/OGG/MP3 first.
- **Final text differs from provisional text:** expected by design. The final
  event is a full-utterance refinement; `streaming_text` is the chunk-merged
  hypothesis.
- **LFS download fails:** confirm `git lfs install` and check the account's LFS
  data pack; the selected pair is ~2.12 GB, above GitHub's free 1 GB allowance.
