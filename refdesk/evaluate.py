"""Run a synthetic, source-oracle evaluation against real pinned repositories."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

from .chat import answer
from .github import ingest
from .providers import configured_provider
from .retrieval import Index


def percentile(values, q):
    return round(sorted(values)[math.ceil(len(values) * q) - 1], 3)


def relevant(source, case):
    return source["path"] in case["paths"] and all(
        fact.lower() in source["text"].lower() for fact in case["facts"]
    )


def evaluate(suite, repeats, provider):
    results = []
    for group in suite:
        start = perf_counter()
        snapshot = ingest(group["repo"].lower())
        index = Index(snapshot)
        ingest_ms = (perf_counter() - start) * 1000
        if snapshot.revision != group["revision"]:
            raise RuntimeError(
                f"{snapshot.repo} moved to {snapshot.revision}. Review the source oracles before changing the pin."
            )
        timings = []
        cases = []
        for case in group["cases"]:
            previous = case.get("previous", [])
            retrieved = index.search(case["question"], previous, limit=6)
            for _ in range(repeats):
                result = answer(index, case["question"], previous, provider)
                timings.append(result["metrics"])
            expected = case.get(
                "status", "supported" if "facts" in case else "insufficient"
            )
            rank = None
            support = None
            complete = result["status"] == expected
            if "facts" in case:
                rank = next(
                    (
                        i
                        for i, (_, source, _) in enumerate(retrieved, 1)
                        if relevant(vars(source), case)
                    ),
                    None,
                )
                complete &= any(
                    relevant(source, case) for source in result["citations"]
                ) and all(
                    fact.lower() in result["text"].lower() for fact in case["facts"]
                )
            if result["citations"]:
                support = sum(
                    "facts" in case and relevant(s, case) for s in result["citations"]
                ) / len(result["citations"])
            cases.append(
                {
                    "id": case["id"],
                    "type": case["type"],
                    "expected": expected,
                    "actual": result["status"],
                    "relevant_rank": rank,
                    "citation_support": support,
                    "completed": complete,
                    "citations": [s["url"] for s in result["citations"]],
                }
            )

        ranks = [
            case["relevant_rank"]
            for case in cases
            if case["expected"] != "insufficient"
        ]
        citations = [
            case["citation_support"]
            for case in cases
            if case["citation_support"] is not None
        ]
        abstentions = [case for case in cases if case["expected"] == "insufficient"]
        results.append(
            {
                "repository": snapshot.repo,
                "revision": snapshot.revision,
                "files": snapshot.files,
                "passages": len(snapshot.evidence),
                "warnings": snapshot.warnings,
                "ingest_ms": round(ingest_ms, 2),
                "retrieval_hit_at_6": sum(r is not None for r in ranks) / len(ranks),
                "retrieval_mrr_at_6": sum(1 / r for r in ranks if r is not None)
                / len(ranks),
                "citation_support": sum(citations) / len(citations)
                if citations
                else None,
                "abstained": sum(case["completed"] for case in abstentions),
                "should_abstain": len(abstentions),
                "completed": sum(case["completed"] for case in cases),
                "cost_usd": sum(t["cost_usd"] for t in timings)
                if all(t["cost_usd"] is not None for t in timings)
                else None,
                "latency_ms": {
                    key: {
                        "p50": percentile([t[key] for t in timings], 0.50),
                        "p95": percentile([t[key] for t in timings], 0.95),
                    }
                    for key in ("retrieval_ms", "generation_ms", "total_ms")
                },
                "cases": cases,
            }
        )
    return {
        "provider": provider.name,
        "synthetic": True,
        "repeats_per_case": repeats,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path(__file__).parent / "evaluation" / "cases.json",
    )
    args = parser.parse_args()
    if not 1 <= args.repeats <= 25:
        parser.error("--repeats must be between 1 and 25")
    suite = json.loads(args.suite.read_text())
    report = evaluate(suite, args.repeats, configured_provider())
    formatted = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(formatted)
    print(formatted, end="")


if __name__ == "__main__":
    main()
