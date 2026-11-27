"""Read-only GitHub access for public, commit-pinned snapshots."""

import io
import json
import logging
import os
import re
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .sources import Evidence, priority, split_file

MAX_ARCHIVE = 24 * 1024 * 1024
MAX_FILE = 192 * 1024
MAX_FILES = 100
MAX_PASSAGES = 3500
LOG = logging.getLogger("refdesk")


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Snapshot:
    repo: str
    branch: str
    revision: str
    evidence: tuple
    files: int
    warnings: tuple


def repository(url):
    match = re.fullmatch(
        r"https://github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/([A-Za-z0-9_.-]{1,100})/?",
        url.strip(),
    )
    if not match:
        raise ValueError(
            "Enter a public GitHub URL like https://github.com/owner/repository."
        )
    owner, name = match.groups()
    name = name.removesuffix(".git")
    if name in {"", ".", ".."}:
        raise ValueError("Enter a valid GitHub repository name.")
    return f"{owner}/{name}".lower()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download(url, limit):
    req = Request(
        url,
        headers={"User-Agent": "refdesk/1.0", "Accept": "application/vnd.github+json"},
    )
    with build_opener(NoRedirect()).open(req, timeout=15) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise SourceError("GitHub returned a response above the size limit.")
    return body


def default_head(repo):
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "credential.helper=",
                "-c",
                "http.followRedirects=false",
                "ls-remote",
                "--symref",
                f"https://github.com/{repo}.git",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            timeout=18,
            check=True,
            env={
                **os.environ,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SourceError(
            "GitHub could not resolve the public repository. Check the URL and network."
        ) from exc
    branch = re.search(r"^ref: refs/heads/(.+)\tHEAD$", result.stdout, re.MULTILINE)
    revision = re.search(r"^([a-f0-9]{40})\tHEAD$", result.stdout, re.MULTILINE)
    if not branch or not revision:
        raise SourceError("The repository has no readable default-branch commit.")
    return branch[1], revision[1]


def file_evidence(archive_bytes, repo, revision):
    evidence = []
    total = 0
    limited = False
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            candidates = []
            for member in archive.infolist():
                _, _, path = member.filename.partition("/")
                if (
                    not path
                    or member.is_dir()
                    or (member.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    continue
                rank = priority(path)
                if rank is None:
                    continue
                if member.file_size > MAX_FILE:
                    limited = True
                    continue
                candidates.append((rank, path, member))
            candidates.sort(key=lambda row: (row[0], row[1].count("/"), row[1]))
            limited |= len(candidates) > MAX_FILES
            for _, path, member in candidates[:MAX_FILES]:
                data = archive.read(member)
                if b"\x00" in data:
                    continue
                try:
                    content = data.decode("utf-8-sig")
                except UnicodeError:
                    continue
                passages = list(split_file(repo, revision, path, content))
                if passages:
                    remaining = MAX_PASSAGES - len(evidence)
                    evidence.extend(passages[:remaining])
                    total += 1
                    if len(evidence) == MAX_PASSAGES:
                        limited = True
                        break
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        raise SourceError("GitHub returned an unreadable source archive.") from exc
    return evidence, total, limited


def activity(repo, revision, endpoint):
    query = (
        "per_page=8"
        if endpoint == "releases"
        else "state=closed&sort=updated&per_page=12"
    )
    try:
        rows = json.loads(
            download(
                f"https://api.github.com/repos/{repo}/{endpoint}?{query}", 1024 * 1024
            )
        )
    except (HTTPError, URLError, OSError, ValueError, SourceError) as exc:
        LOG.info(
            "github_api repo=%s endpoint=%s error=%s",
            repo,
            endpoint,
            type(exc).__name__,
        )
        return [], f"{endpoint.capitalize()} unavailable from GitHub."
    result = []
    for row in rows:
        body = (row.get("body") or "").strip()
        if (
            not body
            or row.get("pull_request")
            or row.get("draft")
            or row.get("prerelease")
        ):
            continue
        if endpoint == "issues":
            number = row["number"]
            path, kind, ref = f"#{number}", "issue", revision
            url = f"https://github.com/{repo}/issues/{number}"
            title = row["title"]
        else:
            ref = row["tag_name"]
            path, kind = ref, "release"
            from urllib.parse import quote

            url = f"https://github.com/{repo}/releases/tag/{quote(ref, safe='')}"
            title = row.get("name") or ref
        date = row.get("published_at") or row.get("updated_at") or ""
        for start in range(0, min(len(body), 6400), 1600):
            result.append(
                Evidence(
                    kind,
                    path,
                    ref,
                    url,
                    f"{title} ({date[:10]})",
                    body[start : start + 1600],
                    True,
                )
            )
    return result, None


def ingest(repo):
    start = perf_counter()
    branch, revision = default_head(repo)
    try:
        archive = download(
            f"https://codeload.github.com/{repo}/zip/{revision}", MAX_ARCHIVE
        )
    except (URLError, OSError) as exc:
        raise SourceError("The public source archive could not be downloaded.") from exc
    evidence, files, limited = file_evidence(archive, repo, revision)
    warnings = ["Source selection reached a size or count limit."] if limited else []
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [
            pool.submit(activity, repo, revision, name)
            for name in ("releases", "issues")
        ]
        for call in calls:
            items, warning = call.result()
            evidence.extend(items)
            if warning:
                warnings.append(warning)
    if not evidence:
        raise SourceError("No supported documentation or examples were found.")
    LOG.info(
        "ingest repo=%s revision=%s files=%d passages=%d ms=%.2f warnings=%s",
        repo,
        revision,
        files,
        len(evidence),
        (perf_counter() - start) * 1000,
        warnings,
    )
    return Snapshot(repo, branch, revision, tuple(evidence), files, tuple(warnings))
