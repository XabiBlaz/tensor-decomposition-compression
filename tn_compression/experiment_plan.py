"""Typed experiment inputs and deterministic selection from analyzer evidence."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .analyzer.schema import AnalysisReport, CandidateResult
from .analyzer.service import greedy_candidate_order

Configuration = tuple[str, ...]
STRATEGIES = ("evidence", "current", "random")


@dataclass(frozen=True)
class ExperimentRequest:
    device: str
    runtime: str
    max_trials_per_strategy: int
    seed: int
    max_quality_loss: float
    max_serialized_bytes: int
    max_memory_bytes: int
    max_latency_ms: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported experiment request schema.")
        if self.runtime != "pytorch":
            raise ValueError("Experiments require the PyTorch bundle runtime.")
        if not isinstance(self.device, str) or not re.fullmatch(
            r"cpu|cuda(?::(?:0|[1-9]\d*))?", self.device
        ):
            raise ValueError("Choose an explicit device: cpu, cuda or cuda:N.")
        for name in (
            "max_trials_per_strategy",
            "max_serialized_bytes",
            "max_memory_bytes",
            "seed",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid experiment {name}.")
        for name in ("max_quality_loss", "max_latency_ms"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid experiment {name}.")
        if self.max_latency_ms == 0 or self.seed >= 2**64:
            raise ValueError(
                "Latency must be positive and seed must fit in 64 unsigned bits."
            )


@dataclass(frozen=True)
class ExperimentPlan:
    request: ExperimentRequest
    evidence: tuple[Configuration, ...]
    current: Configuration
    random: tuple[Configuration, ...]
    schedule: tuple[str, ...]
    schema_version: int = field(default=1, init=False)


def compatible(left: str, right: str) -> bool:
    return (
        left != right
        and not left.startswith(right + ".")
        and not right.startswith(left + ".")
    )


def prefixes(order: Sequence[CandidateResult]) -> list[Configuration]:
    paths: list[str] = []
    selected: list[str] = []
    result = []
    for row in order:
        if all(compatible(row.layer_path, path) for path in paths):
            paths.append(row.layer_path)
            selected.append(row.candidate_id)
            result.append(tuple(sorted(selected)))
    return result


def validate_configuration(
    configuration: Configuration, pool: Mapping[str, CandidateResult]
) -> None:
    if not configuration or configuration != tuple(sorted(set(configuration))):
        raise ValueError(
            "Trial configuration must have distinct, sorted candidate IDs."
        )
    if any(candidate_id not in pool for candidate_id in configuration):
        raise ValueError(
            "Trial contains a candidate outside the compatible evidence pool."
        )
    paths = [pool[key].layer_path for key in configuration]
    if any(
        not compatible(left, right)
        for idx, left in enumerate(paths)
        for right in paths[idx + 1 :]
    ):
        raise ValueError("Trial contains overlapping layer paths.")


def make_plan(
    request: ExperimentRequest, report: AnalysisReport, original_serialized_bytes: int
) -> tuple[ExperimentPlan, dict[str, CandidateResult]]:
    if report.level != "validated" or report.requested_backend != request.runtime:
        raise ValueError("Plan requires validated evidence for the requested runtime.")
    order = [
        row
        for row in greedy_candidate_order(report.candidates)
        if row.structurally_eligible
        and not row.protected
        and row.checkpoint_reconstruction.get("supported")
        and row.backend_support.get("supported")
        and row.model_fingerprint == report.model_fingerprint
    ]
    pool = {row.candidate_id: row for row in order}
    if len(order) != len(pool) or request.max_trials_per_strategy > len(pool):
        raise ValueError(
            f"Trial budget requires at least {request.max_trials_per_strategy} distinct "
            f"compatible candidates. Found {len(pool)}."
        )

    # Search three greedy chains and all singleton configurations.
    singles = [(key,) for key in sorted(pool)]
    by_quality = sorted(order, key=lambda row: (damage(row), row.candidate_id))
    by_size = sorted(order, key=lambda row: (-bytes_saved(row), row.candidate_id))
    catalog = set(singles + prefixes(order) + prefixes(by_quality) + prefixes(by_size))
    estimates = {
        keys: (
            sum(bytes_saved(pool[key]) for key in keys),
            sum(damage(pool[key]) for key in keys),
        )
        for keys in catalog
    }
    minimum = max(0, original_serialized_bytes - request.max_serialized_bytes)
    maximum = max(saving for saving, _ in estimates.values())
    count = request.max_trials_per_strategy
    chosen = []
    for index in range(count):
        target = minimum + max(0, maximum - minimum) * index // max(1, count - 1)

        def priority(keys: Configuration, target: int = target) -> tuple:
            saving, loss = estimates[keys]
            return (
                saving < minimum,
                loss > request.max_quality_loss,
                abs(saving - target),
                loss,
                keys,
            )

        best = min(catalog, key=priority)
        catalog.remove(best)
        chosen.append(best)

    rng = random.Random(request.seed)
    random_trials: dict[Configuration, None] = {}
    for _ in range(count):
        shuffled = rng.sample(order, len(order))
        random_trials[rng.choice(prefixes(shuffled))] = None
    rng.shuffle(singles)
    random_trials.update(dict.fromkeys(singles))
    schedule = tuple(
        strategy
        for _ in range(count)
        for strategy in rng.sample(STRATEGIES, len(STRATEGIES))
    )
    plan = ExperimentPlan(
        request, tuple(chosen), tuple(pool), tuple(random_trials)[:count], schedule
    )
    for keys in (*plan.evidence, *plan.random):
        validate_configuration(keys, pool)
    return plan, pool


def bytes_saved(row: CandidateResult) -> int:
    assert row.estimated_bytes_saved is not None
    return row.estimated_bytes_saved


def damage(row: CandidateResult) -> float:
    assert row.quality_loss is not None
    return row.quality_loss


def current_trial(
    plan: ExperimentPlan,
    pool: Mapping[str, CandidateResult],
    history: Sequence[Mapping[str, Any]],
) -> Configuration:
    """Cumulative damage-per-byte selection with rollback after failed quality."""
    attempted = {tuple(result["candidate_ids"]) for result in history}
    accepted: Configuration = ()
    for result in history:
        proposal = tuple(result["candidate_ids"])
        if (
            result.get("quality_loss") is not None
            and result["quality_loss"] <= plan.request.max_quality_loss
            and set(accepted) < set(proposal)
        ):
            accepted = proposal
            if result["serialized_bytes"] <= plan.request.max_serialized_bytes:
                accepted = ()
    for key in plan.current:
        if all(
            compatible(pool[key].layer_path, pool[other].layer_path)
            for other in accepted
        ):
            proposal = tuple(sorted((*accepted, key)))
            if proposal not in attempted:
                return proposal
    # The first greedy walk can exhaust nonoverlapping edits before the budget.
    return next((key,) for key in plan.current if (key,) not in attempted)


def frontier(rows: Sequence[Mapping[str, Any]], direction: str) -> list[str]:
    """Return IDs of successful measurements undominated on all four axes."""
    points = {
        row["trial_id"]: (
            -row["quality"] if direction == "higher" else row["quality"],
            row["serialized_bytes"],
            row["memory_bytes"],
            row["latency_ms"],
        )
        for row in rows
        if row["status"] == "measured"
    }
    return [
        key
        for key, point in points.items()
        if not any(
            other != point and all(a <= b for a, b in zip(other, point))
            for other in points.values()
        )
    ]
