"""Small per-snapshot lexical index for commands, paths, and prose."""

import math
import re
from collections import Counter

STOP = set(
    """a about again all am an and any are as at be been but by can could did do does
for from get give had has have how i if in inside into is it me my of on or our please also
each make
repo repository should so some tell than that the their them then there these they
this those to use used using want was were what when where which who why will with would
you your current currently now latest still old older outdated previous originally
version versions earlier historical today""".split()
)
ALIASES = {
    "installation": "install",
    "installing": "install",
    "installed": "install",
    "setup": "install",
    "configuration": "config",
    "configure": "config",
    "benchmarking": "benchmark",
    "benchmarks": "benchmark",
    "testing": "test",
    "tests": "test",
    "contributing": "contribute",
    "contribution": "contribute",
    "searches": "search",
    "searching": "search",
    "ignores": "ignore",
    "ignored": "ignore",
    "supports": "support",
    "supported": "support",
    "dependencies": "dependency",
    "interface": "ui",
    "bundles": "bundle",
    "compressed": "compress",
    "compression": "compress",
    "binaries": "binary",
    "flags": "flag",
    "enables": "enable",
    "enabling": "enable",
    "adds": "add",
    "applying": "apply",
    "around": "surround",
    "caches": "cache",
    "directories": "directory",
    "interprets": "interpret",
    "interpreting": "interpret",
    "keeps": "keep",
    "lines": "line",
    "literally": "literal",
    "matches": "match",
    "patterns": "pattern",
    "reading": "read",
    "reads": "read",
    "reconstructs": "reload",
    "respecting": "respect",
    "respects": "respect",
    "surrounding": "surround",
    "tested": "test",
    "verified": "test",
    "verifies": "test",
    "verify": "test",
    "web": "ui",
}
FOLLOW_UP = re.compile(
    r"\b(it|its|that|those|there|this|they|same|also|above)\b|^\s*(and|what about)\b",
    re.IGNORECASE,
)
HISTORY = re.compile(
    r"\b(outdated|old(?:er)?|previous|historical|formerly|used to|changelog|release notes)\b",
    re.IGNORECASE,
)
PROCEDURAL = {
    "start",
    "run",
    "runs",
    "command",
    "flag",
    "enable",
    "release",
    "add",
    "rules",
    "interpret",
    "apply",
    "print",
    "see",
}
VERSION = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")

# shortcut: a bounded English vocabulary, until broader semantic recall is measured
PHRASES = (
    (re.compile(r"\bsaved (bundle|checkpoint)\b"), r"\1"),
    (re.compile(r"\bround-to-nearest\b"), "rounding"),
    (
        re.compile(r"\b(?:writ(?:es?|ing) changes back to|modif(?:y|ies|ying))\b"),
        "modify",
    ),
    (re.compile(r"\b(\d+)[- ]bit\b"), r"\1bit bit"),
    (
        re.compile(
            r"\b(?:pickled model object|serialized (?:python )?(?:model object|nn\.module))\b"
        ),
        "serialized model",
    ),
    (
        re.compile(
            r"\b(?:preview which (?:files|paths) (?:will|would) be (?:scanned|searched) without (?:matching|searching) anything|"
            r"(?:print|list|show) the files that (?:[\w-]+ )?would search,? but (?:don.t|do not) actually search them)\b"
        ),
        "preview searchable file search",
    ),
    (
        re.compile(r"\b(?:read-only|writ(?:e|ing) to)\b"),
        "write access",
    ),
    (
        re.compile(r"\b(?:docker|cpu|gpu) image\b"),
        "container image",
    ),
    (
        re.compile(
            r"\b(?:in (?:my|the|a|your) browser|https?://(?:localhost|127\.0\.0\.1):\d+)\b"
        ),
        "browser",
    ),
    (
        re.compile(
            r"\b(?:bring (?:my|your) own|without downloading (?:a dataset|one)|avoid(?:s|ing)? the dataset download)\b"
        ),
        "local",
    ),
    (
        re.compile(
            r"\b(?:order (?:the )?output alphabetically by (?:file ?name|path)|sort(?:ing)? (?:its |the )?output by (?:file ?name|path))\b"
        ),
        "sort output path",
    ),
)
IDENTIFIER = re.compile(
    r"(?<![\w-])--[a-zA-Z][\w-]*|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"
    r"|(?<![\w./-])(?:[\w.-]+/)*[\w.-]+\.(?:toml|yaml|yml|json|md|rst|py|sh|ini|cfg|txt|onnx|pkl)\b"
)


def terms(text):
    text = re.sub(r"\s+", " ", text.lower().replace("*", " ").replace("`", " "))
    for expression, canonical in PHRASES:
        text = expression.sub(canonical, text)
    words = re.findall(r"[a-z0-9]+", text)
    return [
        ALIASES.get(word, word) for word in words if word not in STOP and len(word) > 1
    ]


class Index:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.rows = []
        frequency = Counter()
        for source in snapshot.evidence:
            body = Counter(terms(source.text))
            title = set(terms(f"{source.heading} {source.path}"))
            vocabulary = set(body) | title
            self.rows.append(
                (source, body, title, vocabulary, max(sum(body.values()), 1))
            )
            frequency.update(vocabulary)
        self.weights = {
            term: math.log1p((len(self.rows) - n + 0.5) / (n + 0.5))
            for term, n in frequency.items()
        }

    def search(self, question, previous=(), limit=6):
        all_terms = set(terms(question))
        query = all_terms - PROCEDURAL or all_terms
        soft_terms = all_terms - query
        context = (
            set(terms(previous[-1])) - query
            if previous and (FOLLOW_UP.search(question) or len(query) < 2)
            else set()
        )
        if not query:
            query, context = context, set()
        if not query:
            return []
        historical = HISTORY.search(question)
        versions = VERSION.findall(question)
        identifiers = {
            re.compile(rf"(?<![\w./-]){re.escape(value)}(?![\w./-])")
            for value in IDENTIFIER.findall(question)
        }
        scored = []
        for source, counts, title, vocabulary, length in self.rows:
            hits = query & vocabulary
            if len(hits) < (1 if len(query) <= 2 else math.ceil(len(query) * 0.50)):
                continue
            searchable = (
                f"{source.path} {source.heading} {source.revision} {source.text}"
            )
            if identifiers and not all(item.search(searchable) for item in identifiers):
                continue
            # Exact version strings cannot be established by separate number matches.
            if versions and not all(
                re.search(
                    rf"(?<![\d.])v?{re.escape(v.lstrip('v'))}(?![\d.])",
                    searchable,
                )
                for v in versions
            ):
                continue
            normalizer = 1.2 * (0.55 + 0.45 * length / 100)
            score = sum(
                self.weights[t]
                * (0.6 * (t in title) + 2.2 * counts[t] / (counts[t] + normalizer))
                for t in sorted(hits)
            )
            score *= (len(hits) / len(query)) ** 1.5
            score += sum(
                self.weights[t] * (0.45 + 0.9 * (t in title))
                for t in sorted(soft_terms & vocabulary)
            )
            if "enable" in soft_terms and re.search(
                r"(?m)^\s*(?:`{3,}|~{3,})", source.text
            ):
                score += 1.8
            if "flag" in soft_terms:
                score += sum(
                    4.5
                    for term in sorted(hits)
                    if re.search(rf"--{term}\b", source.text, re.IGNORECASE)
                )
            if "command" in soft_terms:
                score += sum(
                    5
                    for term in sorted(hits)
                    if re.search(
                        rf"(?m)^[ \t]*[\w-]+ {term}(?![\w-])",
                        source.text,
                        re.IGNORECASE,
                    )
                )
            if "checksum" in hits and re.search(
                r"(?i)sha[- ]?\d+|blake\d|md5", source.text
            ):
                score += 3
            score += 0.42 * sum(self.weights[t] for t in sorted(context & vocabulary))
            if source.historical:
                score *= 1.12 if historical or versions else 0.40
            scored.append((score, source, len(hits) == len(query)))
        scored.sort(key=lambda row: row[0], reverse=True)
        return scored[:limit]
