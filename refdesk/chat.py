"""Grounding and conversation policy shared by every input mode."""

import logging
import re
from dataclasses import asdict
from time import perf_counter
from uuid import uuid4

from .providers import ABSTAIN, Extractive

LOG = logging.getLogger("refdesk")


def answer(index, question, previous=(), provider=None):
    provider = provider or Extractive()
    start = perf_counter()
    trace = uuid4().hex
    candidates = index.search(question, previous)
    retrieval_ms = (perf_counter() - start) * 1000
    evidence = [source for _, source, _ in candidates]
    LOG.info(
        "retrieve trace=%s repo=%s revision=%s ms=%.2f evidence=%s",
        trace,
        index.snapshot.repo,
        index.snapshot.revision,
        retrieval_ms,
        [item.url for item in evidence],
    )

    generated = perf_counter()
    selected = evidence
    comparison = re.search(
        r"\b(still|outdated|changed|superseded|conflict)\b", question, re.IGNORECASE
    )
    if isinstance(provider, Extractive) and candidates:
        # shortcut: exact lexical coverage for quotes, until evaluated synthesis is configured
        selected = [
            source for score, source, exact in candidates if exact and score >= 0.5
        ]
    if not selected:
        result = {
            "status": "insufficient",
            "text": ABSTAIN,
            "ids": [],
            "cost_usd": 0,
            "tokens": {},
        }
    elif comparison and isinstance(provider, Extractive):
        current = next((item for item in selected if not item.historical), None)
        old = next((item for item in selected if item.historical), None)
        if current and old:
            selected = [current, old]
            result = {
                "status": "historical",
                "ids": [1, 2],
                "cost_usd": 0,
                "text": (
                    "The current snapshot and a historical source both mention this. "
                    "Use the current instructions for the indexed revision and verify your version "
                    "before applying the historical note.\n\n"
                    f"Current snapshot [1]\n{current.text}\n\nHistorical source [2]\n{old.text}"
                ),
            }
        else:
            result = provider.answer(question, previous, selected)
    else:
        try:
            result = provider.answer(question, previous, selected)
        except Exception as exc:
            LOG.warning(
                "provider_error trace=%s provider=%s error=%s",
                trace,
                provider.name,
                type(exc).__name__,
            )
            raise RuntimeError(
                "The answer provider is unavailable. Your question can be retried."
            ) from exc

    raw_ids = result.get("ids")
    text = result.get("text")
    status = result.get("status")
    markers = set(re.findall(r"\[(\d+)\]", text)) if isinstance(text, str) else set()
    if (
        not isinstance(raw_ids, list)
        or not isinstance(text, str)
        or not text.strip()
        or status not in {"supported", "historical", "insufficient"}
        or any(type(i) is not int or not 1 <= i <= len(selected) for i in raw_ids)
        or (
            status != "insufficient"
            and (
                not raw_ids
                or (
                    not isinstance(provider, Extractive)
                    and {str(i) for i in raw_ids} != markers
                )
            )
        )
    ):
        LOG.warning(
            "invalid_provider_output trace=%s provider=%s", trace, provider.name
        )
        status, text, raw_ids = "insufficient", ABSTAIN, []
    if status == "insufficient":
        text, raw_ids = ABSTAIN, []
    elif status == "supported" and all(selected[i - 1].historical for i in raw_ids):
        status = "historical"
        text = (
            f"Historical evidence for a different version or point in time.\n\n{text}"
        )
    citations = [{"id": i, **asdict(selected[i - 1])} for i in dict.fromkeys(raw_ids)]
    generation_ms = (perf_counter() - generated) * 1000
    total_ms = (perf_counter() - start) * 1000
    LOG.info(
        "answer trace=%s provider=%s called=%s status=%s citations=%s generation_ms=%.2f total_ms=%.2f tokens=%s cost_usd=%s",
        trace,
        provider.name,
        bool(selected),
        status,
        [item["url"] for item in citations],
        generation_ms,
        total_ms,
        result.get("tokens"),
        result.get("cost_usd"),
    )
    return {
        "id": trace,
        "status": status,
        "text": text,
        "citations": citations,
        "revision": index.snapshot.revision,
        "provider": provider.name,
        "metrics": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "cost_usd": result.get("cost_usd"),
        },
    }
