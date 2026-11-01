const $ = (id) => document.getElementById(id);
const state = {
  models: [],
  model: null,
  ws: null,
  recording: false,
  starting: false,
  ready: false,
  socketReady: false,
  audioContext: null,
  mediaStream: null,
  source: null,
  worklet: null,
  silentGain: null,
  timer: null,
  startedAt: 0,
  audioSeconds: 0,
  levels: Array(96).fill(0.012),
  finalText: "",
  partialText: "",
  lastFinalEvent: null,
  finalizing: false,
  finalWaitTimer: null,
};

const els = {
  modelSelect: $("modelSelect"),
  deviceSelect: $("deviceSelect"),
  recordButton: $("recordButton"),
  recordLabel: $("recordLabel"),
  stopButton: $("stopButton"),
  resetButton: $("resetButton"),
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  timer: $("timer"),
  levelCanvas: $("levelCanvas"),
  levelText: $("levelText"),
  audioClock: $("audioClock"),
  finalText: $("finalText"),
  partialText: $("partialText"),
  latencyValue: $("latencyValue"),
  rtfValue: $("rtfValue"),
  durationValue: $("durationValue"),
  updatesValue: $("updatesValue"),
  firstPartialValue: $("firstPartialValue"),
  finalizationValue: $("finalizationValue"),
  modeNote: $("modeNote"),
  referenceInput: $("referenceInput"),
  metricsButton: $("metricsButton"),
  metricsPanel: $("metricsPanel"),
  werValue: $("werValue"),
  cerValue: $("cerValue"),
  subValue: $("subValue"),
  delValue: $("delValue"),
  insValue: $("insValue"),
  copyButton: $("copyButton"),
  downloadButton: $("downloadButton"),
  themeToggle: $("themeToggle"),
  backendInfo: $("backendInfo"),
  modelName: $("modelName"),
  modelDescription: $("modelDescription"),
  modelTestWer: $("modelTestWer"),
  modelDevWer: $("modelDevWer"),
  modelArtifact: $("modelArtifact"),
  modelQuant: $("modelQuant"),
  modelRuntime: $("modelRuntime"),
  modelLimitations: $("modelLimitations"),
  chunkInput: $("chunkInput"),
  windowInput: $("windowInput"),
  overlapInput: $("overlapInput"),
  cfgChunk: $("cfgChunk"),
  cfgWindow: $("cfgWindow"),
  cfgOverlap: $("cfgOverlap"),
  autoStopCheck: $("autoStopCheck"),
  fileInput: $("fileInput"),
  fileButton: $("fileButton"),
  fileResult: $("fileResult"),
  accuracySelect: $("accuracySelect"),
  correctedInput: $("correctedInput"),
  notesInput: $("notesInput"),
  noiseCheck: $("noiseCheck"),
  accentCheck: $("accentCheck"),
  buildFeedback: $("buildFeedback"),
  feedbackPreview: $("feedbackPreview"),
};

function formatBytes(bytes) {
  if (!bytes) return "—";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(3)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${bytes} B`;
}

function formatPercent(value) {
  return value == null ? "—" : `${(value * 100).toFixed(2)}%`;
}

function setStatus(text, kind = "") {
  els.statusText.textContent = text;
  els.statusDot.className = `status-dot ${kind}`;
}

function setError(message) {
  setStatus(message, "error");
  console.error(message);
}

function updateModelInfo() {
  const m = state.model;
  if (!m) return;
  els.modelName.textContent = m.display_name;
  els.modelDescription.textContent = m.description;
  els.modelTestWer.textContent = formatPercent(m.full_test_wer);
  els.modelDevWer.textContent = m.dev_wer == null
    ? "—" : `${formatPercent(m.dev_wer)}${m.dev_wer_note ? " (packed)" : ""}`;
  els.modelArtifact.textContent = formatBytes(m.artifact_bytes);
  const q = m.quantization;
  els.modelQuant.textContent = q
    ? `${q.bits}-bit · ${q.bitlinear_layers} layers · ${q.scale}`
    : "FP32 reference";
  els.modelRuntime.textContent = m.runtime;
  els.modelLimitations.replaceChildren(
    ...(m.limitations || []).map((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      return li;
    })
  );
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    state.models = payload.models || [];
    els.modelSelect.replaceChildren(
      ...state.models.map((m) => {
        const option = document.createElement("option");
        option.value = m.id;
        option.textContent = m.display_name;
        return option;
      })
    );
    if (state.models.length) {
      state.model = state.models.find((m) => m.id === "ffn2bit") || state.models[0];
      els.modelSelect.value = state.model.id;
      updateModelInfo();
    }
    const status = payload.status;
    els.backendInfo.textContent = `Backend: ${status.device} · CUDA ${status.cuda_available ? "available" : "unavailable"}`;
    setStatus(status.device === "cuda" ? "Ready · GPU backend" : "Ready · CPU backend", "live");
  } catch (error) {
    setError(`Backend unavailable: ${error.message}`);
  }
}

function selectedModel() {
  return state.models.find((m) => m.id === els.modelSelect.value) || state.model;
}

function updateSettingsLabels() {
  els.cfgChunk.textContent = `${els.chunkInput.value} ms`;
  els.cfgWindow.textContent = `${Number(els.windowInput.value).toFixed(1)} s`;
  els.cfgOverlap.textContent = `${Number(els.overlapInput.value).toFixed(1)} s`;
}

function updateAudioClock() {
  els.audioClock.textContent = `${state.audioSeconds.toFixed(1)} s captured`;
  els.durationValue.textContent = `${state.audioSeconds.toFixed(1)} s`;
}

function startTimer() {
  state.startedAt = performance.now();
  clearInterval(state.timer);
  state.timer = setInterval(() => {
    const elapsed = (performance.now() - state.startedAt) / 1000;
    const mins = Math.floor(elapsed / 60);
    const secs = Math.floor(elapsed % 60);
    const tenth = Math.floor((elapsed * 10) % 10);
    els.timer.textContent = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${tenth}`;
  }, 100);
}

function stopTimer() {
  clearInterval(state.timer);
  state.timer = null;
}

function drawLevel() {
  const canvas = els.levelCanvas;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== Math.floor(rect.width * dpr) || canvas.height !== Math.floor(rect.height * dpr)) {
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  }
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  const bars = 96;
  const gap = Math.max(1, Math.floor(width / bars * 0.28));
  const barWidth = Math.max(1, width / bars - gap);
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const values = state.levels.slice(-bars);
  const base = height * 0.82;
  values.forEach((value, index) => {
    const amplitude = Math.min(1, value * 7.5);
    const barHeight = Math.max(2, amplitude * height * 0.72);
    const x = index * (barWidth + gap);
    ctx.globalAlpha = 0.35 + 0.65 * (index / bars);
    ctx.fillRect(x, base - barHeight, barWidth, barHeight);
  });
  ctx.globalAlpha = 1;
  requestAnimationFrame(drawLevel);
}

function setRecordingUI(recording) {
  state.recording = recording;
  els.recordButton.classList.toggle("recording", recording);
  els.recordLabel.textContent = recording ? "Recording…" : "Start recording";
  els.stopButton.disabled = !(recording || state.starting);
  els.recordButton.disabled = recording || state.starting;
  els.modelSelect.disabled = recording;
  els.deviceSelect.disabled = recording;
}

function resetTranscript() {
  state.finalText = "";
  state.partialText = "";
  state.lastFinalEvent = null;
  els.finalText.innerHTML = '<span class="placeholder">Finalized words will appear here.</span>';
  els.partialText.textContent = "";
  els.latencyValue.textContent = "—";
  els.rtfValue.textContent = "—";
  els.updatesValue.textContent = "0";
  els.firstPartialValue.textContent = "—";
  els.finalizationValue.textContent = "—";
  els.durationValue.textContent = "0.0 s";
  els.metricsPanel.hidden = true;
  state.audioSeconds = 0;
  state.levels = [];
  els.timer.textContent = "00:00.0";
  updateAudioClock();
}

function setTranscript(finalText, partialText) {
  state.finalText = finalText || "";
  state.partialText = partialText || "";
  if (state.finalText) {
    els.finalText.textContent = state.finalText;
  } else {
    els.finalText.innerHTML = '<span class="placeholder">Finalized words will appear here.</span>';
  }
  els.partialText.textContent = state.partialText;
}

function showMetrics(metrics) {
  if (!metrics) {
    els.metricsPanel.hidden = true;
    return;
  }
  els.metricsPanel.hidden = false;
  els.werValue.textContent = formatPercent(metrics.wer);
  els.cerValue.textContent = formatPercent(metrics.cer);
  els.subValue.textContent = metrics.substitutions;
  els.delValue.textContent = metrics.deletions;
  els.insValue.textContent = metrics.insertions;
}

async function fetchReferenceMetrics() {
  const reference = els.referenceInput.value.trim();
  const hypothesis = state.lastFinalEvent?.text || state.finalText || "";
  if (!reference || !hypothesis) {
    showMetrics(null);
    setStatus("Enter a reference and finish a transcript first.", "error");
    return;
  }
  try {
    const response = await fetch("/api/metrics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reference, hypothesis }),
    });
    const metrics = await response.json();
    if (!response.ok) throw new Error(metrics.error || `HTTP ${response.status}`);
    showMetrics(metrics);
  } catch (error) {
    showMetrics(null);
    setStatus(`Metrics failed: ${error.message}`, "error");
  }
}

function handleEvent(message) {
  switch (message.type) {
    case "status":
      setStatus(message.status === "loading_model" ? "Loading model on backend…" : message.status, "busy");
      break;
    case "ready":
      state.socketReady = true;
      state.finalizing = false;
      state.model = message.model;
      updateModelInfo();
      els.modeNote.textContent = `Mode: ${message.note}`;
      setStatus(`Recording ready · ${message.load.device} · ${message.load.cached ? "cached" : `loaded ${message.load.load_time_s.toFixed(1)}s`}`, "live");
      connectMicrophone();
      break;
    case "partial":
      if (state.finalizing) break;
      setTranscript(message.final_text || "", message.partial_text || "");
      els.latencyValue.textContent = `${(message.latency_s * 1000).toFixed(0)} ms`;
      els.rtfValue.textContent = message.rtf == null ? "—" : message.rtf.toFixed(3);
      els.durationValue.textContent = `${message.audio_duration_s.toFixed(1)} s`;
      els.updatesValue.textContent = message.chunk;
      if (message.final_text) {
        els.finalText.textContent = message.final_text;
      }
      break;
    case "final_segment":
      if (state.finalizing) break;
      if (message.final_text) {
        setTranscript(message.final_text, state.partialText);
      }
      break;
    case "eou":
      setStatus("End-of-utterance silence detected.", "busy");
      if (els.autoStopCheck.checked && state.recording) {
        stopRecording();
      }
      break;
    case "final":
      state.finalizing = false;
      state.lastFinalEvent = message;
      setTranscript(message.final_text || message.text || "", "");
      if (message.streaming_text && message.streaming_text !== message.refined_text) {
        els.modeNote.textContent = `Mode: ${message.mode}. Streaming hypothesis: ${message.streaming_text}`;
      } else {
        els.modeNote.textContent = `Mode: ${message.mode}`;
      }
      els.firstPartialValue.textContent = message.first_partial_latency_s == null
        ? "—" : `${message.first_partial_latency_s.toFixed(2)} s`;
      els.finalizationValue.textContent = `${message.finalization_latency_s.toFixed(2)} s`;
      els.rtfValue.textContent = message.rtf == null ? "—" : message.rtf.toFixed(3);
      els.durationValue.textContent = `${message.audio_duration_s.toFixed(1)} s`;
      if (message.metrics) showMetrics(message.metrics);
      setStatus("Finalized. You can copy or download the transcript.", "live");
      clearTimeout(state.finalWaitTimer);
      closeSocket();
      break;
    case "error":
      state.starting = false;
      state.finalizing = false;
      setError(message.message || "Backend error.");
      stopCapture();
      closeSocket();
      setRecordingUI(false);
      break;
    default:
      break;
  }
}

async function startRecording() {
  if (state.recording) return;
  resetTranscript();
  const model = selectedModel();
  if (!model) {
    setError("Select a model first.");
    return;
  }
  state.starting = true;
  setRecordingUI(false);
  setStatus("Requesting microphone…", "busy");
  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    state.audioContext = new AudioCtx({ sampleRate: 16000 });
    if (!state.audioContext.audioWorklet) {
      throw new Error("This browser does not support AudioWorklet. Use a current Chromium/Firefox browser.");
    }
    await state.audioContext.audioWorklet.addModule("/static/audio-worklet.js");
    state.source = state.audioContext.createMediaStreamSource(state.mediaStream);
    state.worklet = new AudioWorkletNode(state.audioContext, "mic-capture", {
      processorOptions: { chunkMs: Number(els.chunkInput.value) || 320 },
    });
    state.silentGain = state.audioContext.createGain();
    state.silentGain.gain.value = 0;
    state.worklet.connect(state.silentGain);
    state.silentGain.connect(state.audioContext.destination);
    state.starting = false;
    state.worklet.port.onmessage = (event) => {
      const { pcm, rms } = event.data;
      state.levels.push(rms);
      if (state.levels.length > 140) state.levels.shift();
      state.audioSeconds += pcm.byteLength / 4 / state.audioContext.sampleRate;
      updateAudioClock();
      els.levelText.textContent = rms > 0.008 ? "SPEECH / SIGNAL" : "MIC SIGNAL";
      if (state.socketReady && state.ws?.readyState === WebSocket.OPEN) {
        state.ws.send(pcm);
      }
    };
    connectSocket();
  } catch (error) {
    state.starting = false;
    setError(`Microphone error: ${error.message}`);
    stopCapture();
    setRecordingUI(false);
  }
}

function connectSocket() {
  const model = selectedModel();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const params = new URLSearchParams({
    model_id: model.id,
    device: els.deviceSelect.value,
    chunk_ms: els.chunkInput.value,
    window_s: els.windowInput.value,
    overlap_s: els.overlapInput.value,
    sample_rate: String(state.audioContext?.sampleRate || 16000),
  });
  state.ws = new WebSocket(`${protocol}://${location.host}/ws/stream?${params}`);
  state.ws.binaryType = "arraybuffer";
  state.ws.onopen = () => {
    setStatus("Model loading on backend…", "busy");
    state.ws.send(JSON.stringify({
      type: "start",
      reference: els.referenceInput.value,
    }));
  };
  state.ws.onmessage = (event) => {
    try {
      handleEvent(JSON.parse(event.data));
    } catch (error) {
      setError(`Invalid backend event: ${error.message}`);
    }
  };
  state.ws.onerror = () => {
    setError("WebSocket connection failed. Is the backend running?");
    stopCapture();
  };
  state.ws.onclose = () => {
    state.socketReady = false;
    if (state.recording) {
      setStatus("Connection closed.", "error");
      stopCapture();
    }
  };
}

function connectMicrophone() {
  if (!state.source || !state.worklet) return;
  state.source.connect(state.worklet);
  state.audioContext.resume();
  setRecordingUI(true);
  startTimer();
  setStatus("Listening · speak naturally", "live");
  els.levelText.textContent = "LISTENING";
}

function stopRecording() {
  if (!state.recording && !state.ws && !state.starting) return;
  state.starting = false;
  state.finalizing = true;
  setStatus("Finalizing · running full-utterance refinement…", "busy");
  stopTimer();
  stopCapture();
  setRecordingUI(false);
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({
      type: "stop",
      reference: els.referenceInput.value,
    }));
    clearTimeout(state.finalWaitTimer);
    state.finalWaitTimer = setTimeout(() => {
      if (state.ws?.readyState === WebSocket.OPEN) {
        setError("Finalization timed out.");
        closeSocket();
      }
    }, 90000);
  } else {
    setError("No active stream to finalize.");
  }
}

function stopCapture() {
  stopTimer();
  if (state.source) {
    try { state.source.disconnect(); } catch (_) {}
    state.source = null;
  }
  if (state.worklet) {
    try { state.worklet.disconnect(); } catch (_) {}
    state.worklet = null;
  }
  if (state.silentGain) {
    try { state.silentGain.disconnect(); } catch (_) {}
    state.silentGain = null;
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
  }
  if (state.audioContext) {
    state.audioContext.close().catch(() => {});
    state.audioContext = null;
  }
}

function closeSocket() {
  if (state.ws) {
    try { state.ws.close(); } catch (_) {}
    state.ws = null;
  }
  state.socketReady = false;
}

function resetAll() {
  if (state.recording || state.ws) {
    stopRecording();
  }
  stopCapture();
  closeSocket();
  resetTranscript();
  state.finalizing = false;
  setStatus("Ready to record.", "live");
  els.modeNote.textContent = "Mode: chunked streaming approximation";
  setRecordingUI(false);
}

async function copyTranscript() {
  const text = state.lastFinalEvent?.text || state.finalText || state.partialText;
  if (!text) {
    setStatus("No transcript to copy yet.", "error");
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Transcript copied.", "live");
  } catch (error) {
    setError(`Copy failed: ${error.message}`);
  }
}

function downloadTranscript() {
  const text = state.lastFinalEvent?.text || state.finalText || state.partialText;
  if (!text) {
    setStatus("No transcript to download yet.", "error");
    return;
  }
  const blob = new Blob([text + "\n"], { type: "text/plain" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `onebit-asr-${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  setStatus("Transcript downloaded.", "live");
}

async function transcribeFile() {
  const file = els.fileInput.files?.[0];
  if (!file) {
    setStatus("Choose an audio file first.", "error");
    return;
  }
  els.fileButton.disabled = true;
  els.fileResult.hidden = false;
  els.fileResult.textContent = "Transcribing file…";
  try {
    const form = new FormData();
    form.append("audio", file);
    form.append("model_id", selectedModel().id);
    form.append("reference", els.referenceInput.value);
    const response = await fetch("/api/transcribe", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    els.fileResult.replaceChildren();
    const transcript = document.createElement("div");
    transcript.innerHTML = `<strong>${payload.transcript || "(empty)"}</strong>`;
    const meta = document.createElement("div");
    meta.className = "muted";
    meta.textContent = `${selectedModel().display_name} · ${payload.latency_s.toFixed(2)} s · RTF ${payload.rtf?.toFixed(3) ?? "—"} · ${payload.audio_metadata.duration_s.toFixed(1)} s audio`;
    els.fileResult.append(transcript, meta);
    if (payload.metrics) {
      const metrics = document.createElement("div");
      metrics.className = "muted";
      metrics.textContent = `WER ${formatPercent(payload.metrics.wer)} · CER ${formatPercent(payload.metrics.cer)} · S/D/I ${payload.metrics.substitutions}/${payload.metrics.deletions}/${payload.metrics.insertions}`;
      els.fileResult.append(metrics);
    }
  } catch (error) {
    els.fileResult.textContent = `File transcription failed: ${error.message}`;
    setError(`File transcription failed: ${error.message}`);
  } finally {
    els.fileButton.disabled = false;
  }
}

function buildFeedback() {
  const payload = {
    model: selectedModel()?.id,
    transcript: state.lastFinalEvent?.text || state.finalText || "",
    reference: els.referenceInput.value,
    accuracy: els.accuracySelect.value,
    corrected_transcript: els.correctedInput.value,
    notes: els.notesInput.value,
    flags: {
      background_noise: els.noiseCheck.checked,
      accent_pronunciation: els.accentCheck.checked,
    },
    timestamp: new Date().toISOString(),
    persisted: false,
  };
  els.feedbackPreview.hidden = false;
  els.feedbackPreview.textContent = JSON.stringify(payload, null, 2);
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("onebit-theme", theme);
}

function init() {
  const savedTheme = localStorage.getItem("onebit-theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  loadModels();
  updateSettingsLabels();
  drawLevel();
  els.modelSelect.addEventListener("change", () => {
    state.model = selectedModel();
    updateModelInfo();
    setStatus(`Selected ${state.model.display_name}.`, "live");
  });
  els.recordButton.addEventListener("click", startRecording);
  els.stopButton.addEventListener("click", stopRecording);
  els.resetButton.addEventListener("click", resetAll);
  els.copyButton.addEventListener("click", copyTranscript);
  els.downloadButton.addEventListener("click", downloadTranscript);
  els.metricsButton.addEventListener("click", fetchReferenceMetrics);
  els.fileButton.addEventListener("click", transcribeFile);
  els.buildFeedback.addEventListener("click", buildFeedback);
  els.themeToggle.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
  [els.chunkInput, els.windowInput, els.overlapInput].forEach((input) => {
    input.addEventListener("input", updateSettingsLabels);
  });
  window.addEventListener("beforeunload", () => {
    stopCapture();
    closeSocket();
  });
  updateAudioClock();
}

init();
