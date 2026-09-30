"""Small local HTTP service; compression dependencies load only in job processes."""

import argparse
import json
import mimetypes
import os
import shutil
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .jobs import JobManager, contained_path, read_json


STATIC = Path(__file__).parent / "static"
PRESETS = Path(__file__).parent / "presets.json"


def handler_for(manager):
    class Handler(BaseHTTPRequestHandler):
        def _headers(self, status, content_type, length, *, attachment=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()

        def _json(self, value, status=200):
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def _local_request(self):
            # Prevent drive-by browser requests / DNS rebinding against a local job runner.
            host = self.headers.get("Host", "")
            if urlsplit("//" + host).hostname not in {"localhost", "127.0.0.1", "::1"}:
                self._json({"error": "Open the UI through localhost or an SSH localhost tunnel."}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin and (urlsplit(origin).netloc != host or urlsplit(origin).scheme not in {"http", "https"}):
                self._json({"error": "Cross-origin requests are not allowed."}, 403)
                return False
            return True

        def _file(self, path, *, attachment=False):
            if not path.is_file():
                raise FileNotFoundError("Artifact not found.")
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self._headers(200, content_type, path.stat().st_size, attachment=path.name if attachment else None)
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile)

        def do_GET(self):
            if not self._local_request():
                return
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/api/health":
                    self._json({"status": "ok"})
                elif path == "/api/presets":
                    self._json({"presets": read_json(PRESETS)})
                elif path == "/api/jobs":
                    self._json({"jobs": manager.list()})
                elif path.startswith("/api/jobs/"):
                    parts = path.split("/")
                    identifier = parts[3]
                    if len(parts) == 4:
                        self._json(manager.get(identifier))
                    elif len(parts) > 5 and parts[4] == "artifacts":
                        directory = manager._directory(identifier)
                        artifact = contained_path(directory, "/".join(parts[5:]))
                        self._file(artifact, attachment=True)
                    else:
                        raise FileNotFoundError("Unknown endpoint.")
                else:
                    name = "index.html" if path == "/" else path.removeprefix("/static/").lstrip("/")
                    if name not in {"index.html", "app.js", "styles.css"}:
                        raise FileNotFoundError("Unknown page.")
                    self._file(STATIC / name)
            except (ValueError, FileNotFoundError) as error:
                self._json({"error": str(error)}, 404)

        def do_POST(self):
            if not self._local_request():
                return
            if self.path == "/api/uploads":
                try:
                    if self.headers.get_content_type() != "application/octet-stream":
                        raise ValueError("Send the .pt or .pth file as application/octet-stream.")
                    filename = self.headers.get("X-Filename", "")
                    suffix = Path(filename).suffix.lower()
                    if suffix not in {".pt", ".pth"}:
                        raise ValueError("Upload a .pt or .pth weights file.")
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > 2 * 1024 ** 3:
                        raise ValueError("Upload size must be between 1 byte and 2 GiB.")
                    identifier = uuid.uuid4().hex
                    destination = manager.upload_root / f"{identifier}{suffix}"
                    remaining = length
                    try:
                        with destination.open("xb") as output:
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise ValueError("The upload ended before all bytes arrived.")
                                output.write(chunk)
                                remaining -= len(chunk)
                    except Exception:
                        destination.unlink(missing_ok=True)
                        raise
                    self._json({"id": identifier, "path": str(destination), "bytes": length}, 201)
                except (ValueError, TypeError, OSError) as error:
                    self._json({"error": str(error)}, 400)
                return
            if self.path != "/api/jobs":
                self._json({"error": "Unknown endpoint."}, 404)
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Send application/json.")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 262144:
                    raise ValueError("The request must contain at most 256 KiB of JSON.")
                payload = json.loads(self.rfile.read(length), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number.")))
                self._json(manager.submit(payload), 202)
            except (ValueError, TypeError, UnicodeError) as error:
                self._json({"error": str(error)}, 400)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local TN Compression web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--runs-dir", default=os.environ.get("TN_RUNS_DIR", "runs/ui"))
    parser.add_argument("--data-dir", default=os.environ.get("TN_DATA_DIR", "data"))
    parser.add_argument("--uploads-dir", default=os.environ.get("TN_UPLOADS_DIR", "uploads"))
    parser.add_argument("--stage-timeout", type=int, default=21600, help="Maximum seconds per subprocess stage.")
    args = parser.parse_args(argv)
    manager = JobManager(args.runs_dir, args.data_dir, upload_root=args.uploads_dir,
                         timeout=args.stage_timeout)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(manager))
    print(f"TN Compression: http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
