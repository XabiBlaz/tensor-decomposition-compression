import io
import json
import unittest
import zipfile
from unittest.mock import patch

from refdesk.chat import answer
from refdesk.github import Snapshot, activity, file_evidence, repository
from refdesk.retrieval import Index
from refdesk.sources import split_file

REV = "1" * 40
REPO = "example/project"


def index_for(*files):
    evidence = tuple(
        item for path, text in files for item in split_file(REPO, REV, path, text)
    )
    return Index(Snapshot(REPO, "main", REV, evidence, len(files), ()))


class TextTest(unittest.TestCase):
    def test_public_url_allowlist(self):
        self.assertEqual(
            repository(" https://github.com/Owner/Repo.git/ "), "owner/repo"
        )
        for url in (
            "http://github.com/a/b",
            "https://github.com.evil.test/a/b",
            "https://github.com@evil.test/a/b",
            "https://github.com/a/b/issues",
            "https://github.com/a/..",
            "https://github.com/a/b?token=abc",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                repository(url)

    def test_archive_paths_and_pinned_lines_without_extraction(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("p/doc space/SETUP.rst", "Build\n=====\n\nRun make.\n")
            archive.writestr("p/.hidden/README.md", "ignore")
            archive.writestr("p/examples/build.sh", "#!/bin/sh\nmake\n")
            archive.writestr("p/docs/binary.md", b"invalid\x00body")
        evidence, files, limited = file_evidence(buffer.getvalue(), REPO, REV)
        self.assertEqual(files, 2)
        self.assertFalse(limited)
        self.assertTrue(evidence[0].url.endswith("/doc%20space/SETUP.rst#L1-L4"))
        self.assertEqual(evidence[0].heading, "Build")
        self.assertEqual(evidence[0].revision, REV)

    def test_activity_citations_and_pull_request_exclusion(self):
        with patch(
            "refdesk.github.download",
            return_value=json.dumps(
                [
                    {
                        "number": 5,
                        "title": "Fixed",
                        "body": "An issue",
                        "updated_at": "2026-01-01",
                    },
                    {
                        "number": 6,
                        "title": "Pull request",
                        "body": "Code",
                        "pull_request": {"url": ""},
                    },
                ]
            ).encode(),
        ):
            items, warning = activity(REPO, REV, "issues")
        self.assertIsNone(warning)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].historical)
        self.assertTrue(items[0].url.endswith("/issues/5"))
        with patch(
            "refdesk.github.download",
            return_value=json.dumps(
                [
                    {"tag_name": "v1.0", "body": "A stable release"},
                    {
                        "tag_name": "v1.1-rc",
                        "body": "A preview release",
                        "prerelease": True,
                    },
                ]
            ).encode(),
        ):
            items, warning = activity(REPO, REV, "releases")
        self.assertEqual(
            [(item.kind, item.revision) for item in items], [("release", "v1.0")]
        )

    def test_follow_up_and_abstention(self):
        index = index_for(
            (
                "docs/desktop.md",
                "# Desktop viewer\nUse Viewer for rendering.\n# Export\nThe desktop viewer exports SVG.",
            ),
            ("docs/server.md", "# Server\nExport routes export routes as JSON."),
        )
        result = answer(index, "Can it export?", ["How do I use the desktop viewer?"])
        self.assertEqual(result["citations"][0]["path"], "docs/desktop.md")
        outcome = answer(index, "How do I configure the Oracle Kubernetes exporter?")
        self.assertEqual(outcome["status"], "insufficient")

    def test_paraphrase_preserves_documented_option(self):
        index = index_for(
            (
                "README.md",
                "# File selection\n`--list`: Print the files that finder would search, but don't actually search them.",
            ),
        )
        result = answer(
            index,
            "How do I preview which paths will be scanned without matching anything?",
        )
        self.assertEqual(result["status"], "supported")
        self.assertIn("--list", result["citations"][0]["text"])

    def test_similar_option_names_and_bit_widths_are_not_equivalent(self):
        index = index_for(
            (
                "README.md",
                "# Options\nSet cache mode with --mode-cache.\nThe 4-bit rounding reference is experimental.",
            )
        )
        for question in (
            "How do I set cache mode with --cache-mode?",
            "Is the 9-bit rounding reference experimental?",
        ):
            with self.subTest(question=question):
                self.assertEqual(answer(index, question)["status"], "insufficient")

    def test_source_keeps_a_code_example_with_its_lead_in(self):
        text = (
            "# Setup\n"
            + "Preparation\n" * 9
            + "Run the local server:\n\n```sh\nserve --local\n```\n"
        )
        evidence = list(split_file(REPO, REV, "README.md", text))
        self.assertEqual(len(evidence), 1)
        self.assertIn("Run the local server:\n\n```sh\nserve --local", evidence[0].text)
        self.assertTrue(evidence[0].url.endswith("#L1-L15"))

    def test_outdated_source_for_current_question(self):
        index = index_for(
            ("README.md", "# Server port\nThe current server port is 9000."),
            ("CHANGELOG.md", "1.2.0\n=====\nThe server port was 8000."),
        )
        current = answer(index, "What is the server port?")
        old = answer(index, "What was the server port in 1.2.0?")
        comparison = answer(index, "Is the previous server port still used?")
        self.assertIn("9000", current["text"])
        self.assertEqual(old["status"], "historical")
        self.assertIn("8000", old["text"])
        self.assertEqual(comparison["status"], "historical")
        self.assertEqual(len(comparison["citations"]), 2)

    def test_model_must_cite_known_evidence(self):
        class InvalidProvider:
            name = "invalid"

            def answer(self, *_):
                return {
                    "status": "supported",
                    "text": "The port is 9000 [8]",
                    "ids": [8],
                    "cost_usd": 0,
                }

        index = index_for(("README.md", "# Port\nThe port is 9000."))
        result = answer(index, "What port?", provider=InvalidProvider())
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
