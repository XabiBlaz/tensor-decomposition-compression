"""Run the local Refdesk HTTP application."""

import argparse
import json
import logging
import mimetypes
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from .chat import answer
from .github import SourceError, ingest, repository
from .providers import configured_provider
from .retrieval import Index

STATIC = Path(__file__).parent / "static"
LOG = logging.getLogger("refdesk")


class Handler(BaseHTTPRequestHandler):
    def send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
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
        self.json(
            {"error": "Open Refdesk from localhost or an SSH localhost tunnel."}, 403
        )
        return False

    def do_GET(self):
        if not self.allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/health":
            return self.json({"status": "ok"})
        if path == "/":
            path = "/index.html"
        if path not in {
            "/index.html",
            "/app.js",
            "/style.css",
            "/voice.js",
            "/mark.svg",
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
            if (
                self.headers.get_content_type() != "application/json"
                or not 0 < length <= 32768
            ):
                raise ValueError("Send a JSON request of at most 32 KB.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise TypeError("Send a JSON object.")
            path = urlsplit(self.path).path
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
            self.json({"error": "Invalid JSON request."}, 400)
        except (TypeError, ValueError) as exc:
            self.json({"error": str(exc)}, 400)
        except (SourceError, RuntimeError) as exc:
            LOG.warning("request_error path=%s error=%s", self.path, exc)
            self.json({"error": str(exc)}, 502)
        except (BrokenPipeError, ConnectionResetError):
            LOG.info("client_disconnected path=%s", self.path)
        except Exception:
            LOG.exception("unexpected_request_error")
            self.json({"error": "An unexpected error occurred. Try again."}, 500)


class Server(ThreadingHTTPServer):
    def __init__(self, address, provider):
        super().__init__(address, Handler)
        self.indexes = OrderedDict()
        self.index_lock = Lock()
        self.provider = provider

    def index(self, repo, refresh=False):
        # shortcut: serialize cache misses, until simultaneous ingestion is a measured need
        with self.index_lock:
            if refresh or repo not in self.indexes:
                self.indexes[repo] = Index(ingest(repo))
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
