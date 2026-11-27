"""Select and split line-addressable support material."""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import quote


@dataclass(frozen=True)
class Evidence:
    kind: str
    path: str
    revision: str
    url: str
    heading: str
    text: str
    historical: bool = False


DOCS = {".md", ".mdx", ".rst", ".adoc", ".txt", ".markdown", ".org"}
EXAMPLES = {
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".rs",
    ".js",
    ".ts",
    ".go",
    ".rb",
    ".c",
    ".cpp",
    ".java",
}
SUPPORT = {
    "readme",
    "install",
    "installation",
    "contributing",
    "changelog",
    "changes",
    "news",
    "faq",
    "troubleshooting",
    "getting-started",
    "usage",
    "guide",
    "quickstart",
}


def priority(name):
    path = PurePosixPath(name.lower())
    if any(
        p.startswith(".")
        or p in {"vendor", "node_modules", "_build", "fixtures", "assets"}
        for p in path.parts
    ):
        return None
    stem = path.stem.split(".")[0]
    is_doc = path.suffix in DOCS or not path.suffix and stem in SUPPORT
    if stem == "readme" and is_doc:
        return 0 if len(path.parts) == 1 else 2
    if is_doc:
        return 1 if stem in SUPPORT else 3
    if path.suffix in EXAMPLES and any(
        p in {"examples", "samples", "demos", "tutorials"} for p in path.parts[:-1]
    ):
        return 4
    return None


def split_file(repo, revision, path, content):
    stem = PurePosixPath(path).stem.lower()
    historical = stem in {
        "changelog",
        "changes",
        "news",
        "history",
        "releasenotes",
    } or bool(
        {"archive", "archived", "legacy", "historical", "old"}
        & set(path.lower().split("/")[:-1])
    )
    heading = path
    block = []
    fence = None
    first = 1

    def passage(end):
        text = "\n".join(block).strip()
        if text:
            url = f"https://github.com/{repo}/blob/{revision}/{quote(path)}#L{first}-L{end}"
            return Evidence("file", path, revision, url, heading, text, historical)
        return None

    lines = content.splitlines()
    for index, line in enumerate(lines, 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        header = None if fence else re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if (
            not fence
            and not marker
            and index < len(lines)
            and line.strip()
            and re.fullmatch(r"[=\-~^`]{3,}", lines[index].strip())
        ):
            header = re.match(r"^(.+)$", line)
        leading_into_code = (
            block
            and block[-1].rstrip().endswith(":")
            and index < len(lines)
            and re.match(r"^\s*(`{3,}|~{3,})", lines[index])
        )
        if (
            header
            or (
                not fence
                and len(block) >= 10
                and not line.strip()
                and not leading_into_code
            )
            or len(block) >= 32
        ):
            item = passage(index - 1)
            if item:
                yield item
            block = []
            first = index
        if header:
            heading = header[1]
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        if not block and not line.strip():
            first = index + 1
        else:
            block.append(line[:1200])
    item = passage(len(lines))
    if item:
        yield item
