import io
import unittest
import wave
from unittest.mock import Mock

from refdesk.speech import SAMPLE_RATE, Voice


def waveform(rate=SAMPLE_RATE):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(rate)
        recording.writeframes(b"\x00\x00" * SAMPLE_RATE)
    return stream.getvalue()


class SpeechTest(unittest.TestCase):
    def test_adapter_receives_pcm_and_reports_elapsed_time(self):
        recognizer = Mock()
        recognizer.transcribe.return_value = "where is the guide"
        result = Voice(recognizer, Mock()).transcribe(waveform(), "audio/wav")
        recognizer.transcribe.assert_called_once_with(b"\x00\x00" * SAMPLE_RATE)
        self.assertEqual(result["text"], "where is the guide")
        self.assertGreaterEqual(result["transcription_ms"], 0)

    def test_invalid_and_truncated_media_fail_before_recognition(self):
        recognizer = Mock()
        voice = Voice(recognizer, Mock())
        for blob, format in [
            (b"not audio", "audio/wav"),
            (waveform(44100), "audio/wav"),
            (waveform()[:-8], "audio/wav"),
            (waveform(), "audio/mp4"),
        ]:
            with self.subTest(format=format), self.assertRaises(ValueError):
                voice.transcribe(blob, format)
        recognizer.transcribe.assert_not_called()
