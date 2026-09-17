import json

import pytest

from tn_compression.analyzer.schema import AnalysisReport, CandidateResult, capability


def candidate(configuration=None):
    return CandidateResult(
        model_fingerprint="model-sha",
        layer_path="encoder.0",
        module_type="Linear",
        method="svd",
        configuration=configuration or {"rank": 4},
        structurally_eligible=True,
        original_parameters=72,
        candidate_parameters=64,
        original_artifact_bytes=288,
        estimated_artifact_bytes=256,
        checkpoint_reconstruction=capability(True, "sequential_linear_recipe", verified=True),
        backend_support=capability(True, "ordinary_pytorch_modules", verified=True),
    )


def test_candidate_identifier_is_stable_and_configuration_specific():
    first = candidate()
    reordered = candidate({"rank": 4})
    changed = candidate({"rank": 3})
    assert first.candidate_id == reordered.candidate_id
    assert first.candidate_id != changed.candidate_id
    assert first.estimated_bytes_saved == 32


def test_analysis_schema_round_trips_through_json():
    report = AnalysisReport(
        level="structural",
        model_fingerprint="model-sha",
        task="classification",
        requested_backend="pytorch",
        candidates=[candidate()],
        capabilities={"model_loading": capability(True, "supplied_module", verified=True)},
    )
    restored = AnalysisReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored == report


def test_candidate_identifier_tracks_resolved_configuration():
    value = candidate({"width": 4, "selection": "activation"})
    proposal_id = value.candidate_id
    value.configuration["retained_indices"] = [0, 2, 4, 5]
    resolved_id = value.refresh_candidate_id()
    assert resolved_id != proposal_id
    assert value.refresh_candidate_id() == resolved_id
    serialized = value.to_dict()
    assert CandidateResult.from_dict(serialized).candidate_id == resolved_id
    serialized["configuration"]["retained_indices"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="does not match"):
        CandidateResult.from_dict(serialized)
