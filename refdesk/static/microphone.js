// Capture mono PCM before browser compression so technical words reach STT intact.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Int16Array(1024);
    this.used = 0;
    this.recording = true;
    this.port.onmessage = () => {
      this.recording = false;
      this.sendChunk();
      this.port.postMessage(null);
    };
  }

  sendChunk() {
    if (!this.used) return;
    const chunk = this.chunk.slice(0, this.used);
    this.port.postMessage(chunk, [chunk.buffer]);
    this.used = 0;
  }

  process(inputs) {
    if (!this.recording) return false;
    for (const sample of inputs[0][0] || []) {
      this.chunk[this.used++] = 32767 * Math.max(-1, Math.min(1, sample));
      if (this.used === this.chunk.length) this.sendChunk();
    }
    return true;
  }
}

registerProcessor("refdesk-pcm-capture", PcmCapture);
