"""Optional local STT and streamed PCM speech adapters."""

import io
import logging
import os
import shutil
import subprocess
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, Timer
from time import perf_counter

LOG = logging.getLogger("refdesk")
SAMPLE_RATE = 16000
MAX_RECORDING_BYTES = 1024 * 1024


class PocketSphinx:
    def __init__(self, decoder):
        self.decoder = decoder

    def transcribe(self, samples):
        if len(samples) < SAMPLE_RATE // 2:
            return ""
        decoder = self.decoder(samprate=SAMPLE_RATE, loglevel="ERROR")
        decoder.start_utt()
        decoder.process_raw(samples, full_utt=True)
        decoder.end_utt()
        hypothesis = decoder.hyp()
        return hypothesis.hypstr.strip() if hypothesis else ""


class Flite:
    def __init__(self, ffmpeg):
        self.ffmpeg = ffmpeg

    def stream(self, text):
        with TemporaryDirectory(prefix="refdesk-speech-") as folder:
            Path(folder, "speech.txt").write_text(text, encoding="utf-8")
            command = [
                self.ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "flite=textfile=speech.txt:voice=slt",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "s16le",
                "pipe:1",
            ]
            with subprocess.Popen(
                command, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            ) as process:
                timer = Timer(15, process.kill)
                timer.start()
                try:
                    while chunk := process.stdout.read1(4096):
                        yield chunk
                    if process.wait():
                        raise RuntimeError("Speech synthesis failed.")
                finally:
                    timer.cancel()
                    if process.poll() is None:
                        process.kill()


class Voice:
    def __init__(self, recognizer, synthesizer):
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self.slots = BoundedSemaphore(2)

    def transcribe(self, data, content_type):
        if content_type != "audio/wav":
            raise ValueError("Voice needs mono WAV audio.")
        try:
            with wave.open(io.BytesIO(data)) as recording:
                frames = recording.getnframes()
                if (
                    recording.getnchannels() != 1
                    or recording.getsampwidth() != 2
                    or recording.getframerate() != SAMPLE_RATE
                    or frames > SAMPLE_RATE * 20
                ):
                    raise ValueError("Voice needs up to 20 seconds of 16 kHz mono PCM.")
                samples = recording.readframes(frames)
                if len(samples) != frames * 2:
                    raise ValueError(
                        "The WAV recording ended before all samples arrived."
                    )
        except (wave.Error, EOFError) as exc:
            raise ValueError("The WAV recording could not be read.") from exc
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("Voice is busy. Try again or type your question.")
        start = perf_counter()
        try:
            text = self.recognizer.transcribe(samples)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "Transcription failed. Record again or type your question."
            ) from exc
        finally:
            self.slots.release()
        elapsed = (perf_counter() - start) * 1000
        LOG.info(
            "transcription provider=pocketsphinx audio_bytes=%d ms=%.2f",
            len(data),
            elapsed,
        )
        return {"text": text, "transcription_ms": elapsed}


def configured_voice():
    if os.getenv("REFDESK_VOICE") != "1":
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        LOG.warning("Voice unavailable. Install FFmpeg with the flite filter.")
        return None
    try:
        from pocketsphinx import Decoder

        probe = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            timeout=3,
            check=True,
        )
        if b" flite " not in probe.stdout:
            LOG.warning("Voice unavailable. FFmpeg needs the flite filter.")
            return None
    except (ImportError, OSError, subprocess.SubprocessError):
        LOG.warning("Voice unavailable. Check PocketSphinx and FFmpeg setup.")
        return None
    return Voice(PocketSphinx(Decoder), Flite(ffmpeg))
