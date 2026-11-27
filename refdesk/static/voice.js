// Microphone and PCM playback adapters share the normal ask callback.
const SAMPLE_RATE = 16000;

async function mediaRequest(path, body, type, signal) {
  const response = await fetch(`/api/${path}`, {
    method: "POST",
    headers: { "Content-Type": type },
    body,
    signal,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.error || "Voice connection failed.");
  }
  return response;
}

function waveFile(parts) {
  const length = parts.reduce((total, part) => total + part.length * 2, 0);
  const header = new DataView(new ArrayBuffer(44));
  for (const [offset, marker] of [[0, "RIFF"], [8, "WAVEfmt "], [36, "data"]]) {
    for (let i = 0; i < marker.length; i++) header.setUint8(offset + i, marker.charCodeAt(i));
  }
  header.setUint32(4, length + 36, true);
  header.setUint32(16, 16, true);
  header.setUint16(20, 1, true);
  header.setUint16(22, 1, true);
  header.setUint32(24, SAMPLE_RATE, true);
  header.setUint32(28, SAMPLE_RATE * 2, true);
  header.setUint16(32, 2, true);
  header.setUint16(34, 16, true);
  header.setUint32(40, length, true);
  return new Blob([header, ...parts], { type: "audio/wav" });
}

class Microphone {
  async start(session) {
    const audio = session.playback.audio;
    await audio.audioWorklet.addModule("/microphone.js");
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    if (session.controller.signal.aborted) {
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    session.stream = stream;
    const source = audio.createMediaStreamSource(stream);
    const recorder = new AudioWorkletNode(audio, "refdesk-pcm-capture", { outputChannelCount: [1] });
    source.connect(recorder);
    recorder.connect(audio.destination);
    session.recorder = recorder;
    session.recording = new Promise((resolve, reject) => {
      const parts = [];
      recorder.port.onmessage = ({ data }) => {
        if (data === null) return resolve(waveFile(parts));
        if (data.some((sample) => Math.abs(sample) > 420)) session.lastSpeech = performance.now();
        parts.push(data);
      };
      session.controller.signal.addEventListener("abort", () => reject(new Error("Voice stopped.")), { once: true });
      recorder.addEventListener("processorerror", () => reject(new Error("Microphone capture failed.")), { once: true });
    });
    session.recording.catch(() => {});
  }

  async transcribe(session) {
    session.recorder.port.postMessage("finish");
    const recording = await session.recording;
    for (const track of session.stream.getTracks()) track.stop();
    const response = await mediaRequest("transcribe", recording, recording.type, session.controller.signal);
    return response.json();
  }
}

class PcmPlayback {
  constructor() {
    this.audio = new AudioContext({ latencyHint: "interactive", sampleRate: SAMPLE_RATE });
    this.monitor = this.audio.createAnalyser();
    this.monitor.connect(this.audio.destination);
  }

  async speak(result, signal, firstAudible) {
    const speech = result.text
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/\[\d+\]|https?:\/\/\S+|[`#*]/g, "")
      .trim()
      .slice(0, 1400);
    const response = await mediaRequest("speech", JSON.stringify({ text: speech, trace: result.id }),
      "application/json", signal);
    const synthesisMs = Number(response.headers.get("Server-Timing")?.match(/dur=([\d.]+)/)?.[1]);
    const reader = response.body.getReader();
    const wave = new Float32Array(this.monitor.fftSize);
    let carry = null;
    let nextTime = 0;
    let observed = false;
    const probe = setInterval(() => {
      this.monitor.getFloatTimeDomainData(wave);
      if (!observed && wave.some((sample) => Math.abs(sample) >= 0.005)) {
        observed = true;
        firstAudible(synthesisMs, (this.audio.outputLatency || 0) * 1000);
      }
    }, 10);
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        let bytes = value;
        if (carry !== null) {
          bytes = new Uint8Array(value.length + 1);
          bytes[0] = carry;
          bytes.set(value, 1);
        }
        carry = bytes.length % 2 ? bytes.at(-1) : null;
        const length = Math.floor(bytes.length / 2);
        if (!length) continue;
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const buffer = this.audio.createBuffer(1, length, SAMPLE_RATE);
        const samples = buffer.getChannelData(0);
        for (let i = 0; i < length; i++) samples[i] = view.getInt16(2 * i, true) / 32768;
        const source = this.audio.createBufferSource();
        source.buffer = buffer;
        source.connect(this.monitor);
        nextTime = Math.max(nextTime, this.audio.currentTime + 0.04);
        source.start(nextTime);
        nextTime += buffer.duration;
        this.lastSource = source;
      }
      if (carry !== null || !this.lastSource) throw new Error("The speech stream ended before audio arrived.");
      await new Promise((resolve) => {
        if (signal.aborted || this.audio.currentTime >= nextTime) return resolve();
        this.lastSource.onended = resolve;
        signal.addEventListener("abort", resolve, { once: true });
      });
    } finally {
      clearInterval(probe);
      reader.releaseLock();
    }
  }
}

export async function connectVoice({ button, cancel, status, question, ask, ready, onActive }) {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode || !window.AudioContext) return;
  const health = await fetch("/api/health").then((response) => response.json()).catch(() => ({}));
  if (!health.voice) return;
  const microphone = new Microphone();
  let current = null;
  button.hidden = false;
  status.hidden = false;

  function phase(session, name, text) {
    session.phase = name;
    status.textContent = text;
    const actions = { recording: "Finish recording", review: "Ask aloud" };
    button.textContent = actions[name] || "Start voice";
    button.disabled = !actions[name];
    cancel.hidden = false;
  }

  function clean(session) {
    clearTimeout(session.timeout);
    if (session.recorder) session.recorder.disconnect();
    for (const track of session.stream?.getTracks() || []) track.stop();
    if (session.playback.audio.state !== "closed") session.playback.audio.close().catch(() => {});
    if (current === session) {
      current = null;
      onActive(false);
      cancel.hidden = true;
      button.disabled = false;
      button.textContent = "Start voice";
    }
  }

  async function finish(session) {
    clearTimeout(session.timeout);
    const end = performance.now();
    phase(session, "transcribing", "Transcribing recording…");
    const signal = session.controller.signal;
    try {
      const transcript = await microphone.transcribe(session);
      if (signal.aborted) return;
      if (!transcript.text) throw new Error("No speech was recognized. Record again or type a question.");
      question.value = transcript.text;
      phase(session, "review", "Review the transcript, then choose Ask aloud.");
      question.focus();
      const confirmed = await new Promise((resolve) => {
        session.confirm = resolve;
        signal.addEventListener("abort", () => resolve(null), { once: true });
      });
      if (!confirmed || signal.aborted) return;
      const submitted = performance.now();
      phase(session, "searching", "Searching repository sources…");
      const result = await ask(confirmed, AbortSignal.any([signal, AbortSignal.timeout(95000)]));
      if (signal.aborted) return;
      if (!result) throw new Error("The voice question was interrupted. Review the transcript and use Ask to retry.");
      phase(session, "speaking", "Reading the answer aloud…");
      await session.playback.speak(result, signal, (synthesisMs, outputLatency) => {
        const metrics = {
          id: result.id,
          transcript: transcript.text,
          transcription_ms: transcript.transcription_ms,
          retrieval_ms: result.metrics.retrieval_ms,
          generation_ms: result.metrics.generation_ms,
          synthesis_first_chunk_ms: synthesisMs,
          end_to_audio_onset_ms: performance.now() - Math.min(session.lastSpeech ?? end, end) + outputLatency,
          finish_to_audio_onset_ms: performance.now() - end + outputLatency,
          submit_to_audio_onset_ms: performance.now() - submitted + outputLatency,
        };
        console.info("Refdesk voice timing", JSON.stringify({ ...metrics, transcript: undefined }));
        document.dispatchEvent(new CustomEvent("refdesk-voice-timing", { detail: metrics }));
      });
      if (!signal.aborted) status.textContent = "Voice turn complete.";
    } catch (error) {
      if (!signal.aborted) status.textContent = error.message === "Failed to fetch"
        ? "Voice connection lost. Record again or continue in text." : error.message;
    } finally {
      clean(session);
    }
  }

  button.addEventListener("click", async () => {
    if (current?.phase === "recording") return finish(current);
    if (current?.phase === "review") {
      if (question.value.trim()) current.confirm(question.value.trim());
      else status.textContent = "Enter a question before asking.";
      return;
    }
    if (current || !ready()) {
      if (!current) status.textContent = "Index a repository and wait for the current request before starting voice.";
      return;
    }
    const session = {
      controller: new AbortController(),
      playback: new PcmPlayback(),
    };
    current = session;
    onActive(true);
    phase(session, "opening", "Allow microphone access to record a question.");
    try {
      await session.playback.audio.resume();
      await microphone.start(session);
      if (session.controller.signal.aborted) return;
      phase(session, "recording", "Recording. Choose Finish recording when you’re done.");
      session.timeout = setTimeout(() => finish(session), 15000);
    } catch {
      if (!session.controller.signal.aborted) status.textContent = "Microphone unavailable. Check permissions or type your question.";
      clean(session);
    }
  });
  question.addEventListener("keydown", (event) => {
    if (current?.phase === "review" && event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      button.click();
    }
  });
  cancel.addEventListener("click", () => {
    if (!current) return;
    current.controller.abort();
    clean(current);
    status.textContent = "Voice stopped. You can continue in text.";
  });
}
