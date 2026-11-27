"""Run the local Refdesk HTTP application."""

import argparse
import json
import logging
import mimetypes
from collections import OrderedDict
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import urlsplit

from .chat import answer
from .github import SourceError, ingest, repository
from .providers import configured_provider
from .retrieval import Index
from .speech import MAX_RECORDING_BYTES, configured_voice

STATIC = Path(__file__).parent / "static"
LOG = logging.getLogger("refdesk")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def begin(self, content_type, status=200, length=None, synthesis_ms=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if length is None:
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Server-Timing", f"synthesis;dur={synthesis_ms:.2f}")
        else:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()

    def send(self, body, content_type, status=200):
        self.begin(content_type, status, len(body))
        self.wfile.write(body)

    def json(self, value, status=200):
        self.send(json.dumps(value).encode(), "application/json; charset=utf-8", status)

    def allowed(self):
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        try:
            if urlsplit(f"//{host}").hostname in {"localhost", "127.0.0.1", "::1"} and (
                not origin or origin in {f"http://{host}", f"https://{host}"}
            ):
                return True
        except ValueError:
            pass
        self.close_connection = True
        self.json(
            {"error": "Open Refdesk from localhost or an SSH localhost tunnel."}, 403
        )
        return False

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/health":
            return self.json({"status": "ok", "voice": self.server.voice is not None})
        if path == "/":
            path = "/index.html"
        if path not in {
            "/index.html",
            "/app.js",
            "/style.css",
            "/voice.js",
            "/microphone.js",
        }:
            return self.json({"error": "Page not found."}, 404)
        file = STATIC / path[1:]
        if not file.is_file():
            return self.json({"error": "Page not found."}, 404)
        self.send(
            file.read_bytes(),
            mimetypes.guess_type(file)[0] or "application/octet-stream",
        )

    def do_POST(self):
        if not self.allowed():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            path = urlsplit(self.path).path
            if path in {"/api/transcribe", "/api/speech"} and self.server.voice is None:
                self.close_connection = True
                return self.json({"error": "Voice is disabled on this server."}, 404)
            if path == "/api/transcribe":
                if not 0 < length <= MAX_RECORDING_BYTES:
                    raise ValueError("Record a clip of at most 20 seconds and 1 MiB.")
                return self.json(
                    self.server.voice.transcribe(
                        self.rfile.read(length), self.headers.get_content_type()
                    )
                )
            if (
                self.headers.get_content_type() != "application/json"
                or not 0 < length <= 32768
            ):
                raise ValueError("Send a JSON request of at most 32 KB.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise TypeError("Send a JSON object.")
            if path == "/api/speech":
                text = data.get("text")
                if not isinstance(text, str) or not 0 < len(text.strip()) <= 1400:
                    raise ValueError("Speech input needs 1–1400 characters.")
                return self.stream_speech(text.strip(), data.get("trace"))
            if path == "/api/flag":
                trace = data.get("id")
                if (
                    not isinstance(trace, str)
                    or len(trace) != 32
                    or any(c not in "0123456789abcdef" for c in trace)
                ):
                    raise ValueError("Invalid answer ID.")
                LOG.info("feedback trace=%s unhelpful=true", trace)
                return self.json({"flagged": True})
            if path not in {"/api/repository", "/api/chat"}:
                return self.json({"error": "Endpoint not found."}, 404)
            url = data.get("url")
            if not isinstance(url, str):
                raise TypeError("Enter the GitHub repository URL.")
            repo = repository(url)
            if path == "/api/repository":
                index = self.server.index(repo, refresh=data.get("refresh") is True)
                snapshot = index.snapshot
                return self.json(
                    {
                        "repo": repo,
                        "branch": snapshot.branch,
                        "revision": snapshot.revision,
                        "files": snapshot.files,
                        "passages": len(snapshot.evidence),
                        "warnings": snapshot.warnings,
                        "provider": self.server.provider.name,
                    }
                )
            question, previous = data.get("question"), data.get("previous", [])
            if not isinstance(question, str) or not 0 < len(question.strip()) <= 1600:
                raise ValueError("Enter a question of 1–1600 characters.")
            if (
                not isinstance(previous, list)
                or len(previous) > 4
                or any(
                    not isinstance(turn, str) or len(turn) > 1600 for turn in previous
                )
            ):
                raise ValueError("Send up to four previous questions.")
            index = self.server.index(repo)
            return self.json(
                answer(index, question.strip(), previous, self.server.provider)
            )
        except (UnicodeError, json.JSONDecodeError):
            self.close_connection = True
            self.json({"error": "Invalid JSON request."}, 400)
        except (TypeError, ValueError) as exc:
            self.close_connection = True
            self.json({"error": str(exc)}, 400)
        except (SourceError, RuntimeError) as exc:
            LOG.warning("request_error path=%s error=%s", self.path, exc)
            self.json({"error": str(exc)}, 502)
        except (BrokenPipeError, ConnectionResetError):
            LOG.info("client_disconnected path=%s", self.path)
        except Exception:
            LOG.exception("unexpected_request_error")
            self.json({"error": "An unexpected error occurred. Try again."}, 500)

    def stream_speech(self, text, trace):
        voice = self.server.voice
        if not voice.slots.acquire(blocking=False):
            raise RuntimeError("Voice is busy. Try again or read the answer.")
        start = perf_counter()
        sent_headers = False
        if (
            not isinstance(trace, str)
            or len(trace) != 32
            or any(c not in "0123456789abcdef" for c in trace)
        ):
            trace = None
        try:
            with closing(voice.synthesizer.stream(text)) as chunks:
                first = next(chunks, b"")
                if not first:
                    raise RuntimeError("Speech synthesis returned no audio.")
                first_ms = (perf_counter() - start) * 1000
                self.begin("application/octet-stream", synthesis_ms=first_ms)
                sent_headers = True
                self.pcm_chunk(first)
                for chunk in chunks:
                    self.pcm_chunk(chunk)
                self.wfile.write(b"0\r\n\r\n")
                LOG.info(
                    "synthesis trace=%s provider=flite first_ms=%.2f total_ms=%.2f",
                    trace,
                    first_ms,
                    (perf_counter() - start) * 1000,
                )
        except (BrokenPipeError, ConnectionResetError):
            LOG.info("speech_interrupted trace=%s", trace)
            self.close_connection = True
        except (OSError, RuntimeError) as exc:
            LOG.warning("synthesis_error trace=%s error=%s", trace, type(exc).__name__)
            if sent_headers:
                self.close_connection = True
            else:
                raise RuntimeError(
                    "Speech synthesis is unavailable. Read the answer in chat."
                ) from exc
        finally:
            voice.slots.release()

    def pcm_chunk(self, chunk):
        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")


class Server(ThreadingHTTPServer):
    def __init__(self, address, provider):
        super().__init__(address, Handler)
        self.indexes = OrderedDict()
        self.index_lock = Lock()
        self.provider = provider
        self.voice = configured_voice()

    def index(self, repo, refresh=False):
        # shortcut: serialize cache misses, until simultaneous ingestion is a measured need
        with self.index_lock:
            if refresh or repo not in self.indexes:
                snapshot = ingest(repo)
                start = perf_counter()
                self.indexes[repo] = Index(snapshot)
                LOG.info(
                    "index repo=%s revision=%s build_ms=%.2f",
                    repo,
                    snapshot.revision,
                    (perf_counter() - start) * 1000,
                )
                if len(self.indexes) > 6:
                    self.indexes.popitem(last=False)
            self.indexes.move_to_end(repo)
            return self.indexes[repo]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with Server(("127.0.0.1", args.port), configured_provider()) as server:
        LOG.info("Refdesk listening at http://localhost:%d", server.server_port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
