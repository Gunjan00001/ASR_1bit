class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunkMs = options?.processorOptions?.chunkMs ?? 320;
    this.chunkSize = Math.max(256, Math.round(sampleRate * chunkMs / 1000));
    this.buffer = new Float32Array(this.chunkSize);
    this.offset = 0;
    this.sumSquares = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channel = input[0];
    for (let i = 0; i < channel.length; i += 1) {
      const sample = channel[i];
      this.buffer[this.offset++] = sample;
      this.sumSquares += sample * sample;
      if (this.offset >= this.chunkSize) {
        const rms = Math.sqrt(this.sumSquares / this.chunkSize);
        const out = this.buffer;
        this.port.postMessage({ pcm: out.buffer, rms }, [out.buffer]);
        this.buffer = new Float32Array(this.chunkSize);
        this.offset = 0;
        this.sumSquares = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);
