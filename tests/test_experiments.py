import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from tn_compression import experiments
from tn_compression.analyzer.schema import AnalysisReport, CandidateResult, capability
from tn_compression.experiment_plan import (
    STRATEGIES,
    ExperimentRequest,
    current_trial,
    frontier,
    make_plan,
    validate_configuration,
)


@pytest.fixture
def request_spec():
    return ExperimentRequest("cpu", "pytorch", 2, 31, 0.1, 100, 200, 10)


@pytest.mark.parametrize(
    "change",
    [
        {"runtime": "onnx"},
        {"device": "auto"},
        {"max_trials_per_strategy": 0},
        {"max_trials_per_strategy": True},
        {"max_quality_loss": float("nan")},
        {"max_serialized_bytes": -1},
        {"max_memory_bytes": 0},
        {"max_latency_ms": 0},
        {"schema_version": 2},
    ],
)
def test_request_constraints(request_spec, change):
    with pytest.raises(ValueError):
        replace(request_spec, **change)


def candidate(path, rank, loss, *, supported=True):
    return CandidateResult(
        model_fingerprint="model",
        layer_path=path,
        module_type="Linear",
        method="svd",
        configuration={"rank": rank},
        structurally_eligible=True,
        original_parameters=25,
        candidate_parameters=rank * 2,
        original_artifact_bytes=100,
        estimated_artifact_bytes=rank * 8,
        quality_loss=loss,
        status="validated",
        checkpoint_reconstruction=capability(supported, "fixture"),
        backend_support=capability(True, "fixture"),
    )


def test_plan_is_seeded_and_checks_budget_and_compatibility(request_spec):
    rows = [
        candidate("head", 4, 0.02),
        candidate("head", 2, 0.07),
        candidate("stem", 4, 0.04),
        candidate("stem.child", 2, 0.08),
        candidate("excluded", 2, 0.001, supported=False),
    ]
    report = AnalysisReport("validated", "model", "classification", "pytorch", rows, {})
    plan, pool = make_plan(request_spec, report, 220)
    assert make_plan(request_spec, report, 220)[0] == plan
    assert json.loads(json.dumps(asdict(plan)))["request"]["seed"] == 31
    assert all(plan.schedule.count(name) == 2 for name in STRATEGIES)
    assert plan.evidence[0] == tuple(
        sorted((rows[0].candidate_id, rows[2].candidate_id))
    )
    assert rows[-1].candidate_id not in pool
    for keys in [*plan.evidence, *plan.random]:
        validate_configuration(keys, pool)
    for items in [(0, 1), (2, 3), (4,)]:
        with pytest.raises(ValueError):
            validate_configuration(
                tuple(sorted(rows[index].candidate_id for index in items)), pool
            )
    with pytest.raises(ValueError, match="compatible candidates"):
        make_plan(replace(request_spec, max_trials_per_strategy=5), report, 220)

    first = current_trial(plan, pool, [])
    assert first == (rows[0].candidate_id,)
    rejected = {"candidate_ids": first, "quality_loss": 0.2, "serialized_bytes": 152}
    assert current_trial(plan, pool, [rejected]) == (rows[2].candidate_id,)
    accepted = {**rejected, "quality_loss": 0.02}
    assert current_trial(plan, pool, [accepted]) == plan.evidence[0]


def test_measured_constraints_and_frontier(request_spec):
    row = {
        "trial_id": "a",
        "status": "measured",
        "quality": 0.8,
        "serialized_bytes": 100,
        "memory_bytes": 200,
        "latency_ms": 10,
    }
    tradeoff = {**row, "trial_id": "b", "quality": 0.9, "latency_ms": 11}
    dominated = {**row, "trial_id": "c", "quality": 0.7, "memory_bytes": 201}
    failed = {**row, "trial_id": "d", "status": "failed", "quality": 1.0}
    assert frontier([row, tradeoff, dominated, failed], "higher") == ["a", "b"]
    assert frontier([row, {**dominated, "memory_bytes": 200}], "lower") == ["c"]
    assert experiments.decision(row, 0.1, request_spec) == []
    assert experiments.decision(
        {**dominated, "serialized_bytes": 101, "latency_ms": 11}, 0.11, request_spec
    ) == [
        "quality",
        "serialized_bytes",
        "memory_bytes",
        "latency_ms",
    ]
    with pytest.raises(ValueError, match="iterations"):
        experiments.benchmark_settings(
            {"iterations": 0}, request_spec, "classification"
        )


def test_records_failures_and_resumes_without_spending_attempts_twice(
    tmp_path, monkeypatch, request_spec
):
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    model._tn_model_spec = {"source": "torchvision"}
    calibration = [(torch.eye(8), torch.arange(8))]
    validation = [(torch.eye(8) * 0.5, torch.arange(8))]
    config = {
        "task": "classification",
        "num_classes": 8,
        "analysis": {
            "candidate_grid": {
                "linear": {"methods": ["svd"], "ranks": [1, 2]},
                "quantization": {"enabled": False},
            },
        },
    }
    invoked = []

    def save_bundle(_model, path, *, metadata):
        path.mkdir()
        (path / "weights.pt").write_bytes(bytes(220))
        (path / "manifest.json").touch()

    def save_trial(_original, path, *_args):
        trial_id = path.parent.name
        invoked.append(trial_id)
        if len(invoked) == 2:
            raise KeyboardInterrupt
        if len(invoked) == 3:
            raise RuntimeError("simulated materialization failure")

    def measure(
        path, _adapter, _validation, _settings, result, _lock_fd, baseline=None
    ):
        result.update(
            status="measured",
            quality=0.7,
            evaluation={"loss": 0.7},
            serialized_bytes=90,
            memory_bytes=180,
            latency_ms=9,
        )
        if baseline is not None:
            result["quality_loss"] = 0.0

    monkeypatch.setattr(experiments, "save_bundle", save_bundle)
    monkeypatch.setattr(experiments, "_save_trial", save_trial)
    monkeypatch.setattr(experiments, "_measure", measure)

    def run(*, plan_only=False, validation_ids=("validation:0",)):
        return experiments.run_experiment(
            model,
            config,
            request_spec,
            tmp_path,
            calibration=calibration,
            calibration_ids=["calibration:0"],
            validation=validation,
            validation_ids=validation_ids,
            benchmark={},
            plan_only=plan_only,
        )

    assert run(plan_only=True)["status"] == "planned"
    assert not invoked
    with pytest.raises(KeyboardInterrupt):
        run()
    with pytest.raises(ValueError, match="changed configuration"):
        run(validation_ids=["validation:changed"])

    progress = run()
    assert progress["status"] == "complete"
    assert len(invoked) == len(set(invoked)) == 6
    assert all(row["attempted"] == 2 for row in progress["strategies"].values())
    interrupted = progress["trials"][1]
    assert interrupted["trial_id"] == invoked[1]
    assert "Interrupted" in interrupted["error"]
    assert "materialization failure" in progress["trials"][2]["error"]
    assert {row["trial_id"] for row in progress["trials"] if row["accepted"]} == set(
        progress["feasible_frontier"]
    )
    assert run() == progress
    assert len(invoked) == 6
    for row in progress["trials"]:
        record = Path(tmp_path, "trials", row["trial_id"], "result.json")
        assert json.loads(record.read_text()) == row


def test_benchmark_seed_is_independent_of_bundle_reconstruction(monkeypatch):
    pytest.importorskip("psutil")
    from tn_compression import checkpoints
    from tn_compression.benchmark import _worker

    inputs = []

    class Model(torch.nn.Conv2d):
        def forward(self, image):
            inputs.append(image.clone())
            return super().forward(image)

    def restore(path, *, device):
        torch.rand(int(path))
        return Model(3, 3, 1), {}

    monkeypatch.setattr(checkpoints, "load_bundle", restore)
    options = {
        "task": "classification",
        "device": "cpu",
        "input_shape": [1, 3, 4, 4],
        "warmup": 0,
        "iterations": 1,
        "threads": 1,
        "seed": 47,
    }
    assert _worker("1", options)["status"] == "ok"
    assert _worker("99", options)["status"] == "ok"
    torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)
