import copy

import torch

from tn_compression.analyzer.context import analysis_context


def test_analysis_context_is_stable_and_binds_content_identifiers_and_preprocessing():
    config = {
        "seed": 7,
        "task": "classification",
        "data": {"kind": "synthetic", "normalization": "unit"},
        "analysis": {"candidate_grid": {"linear": {"ranks": [1, 2]}}},
    }
    batches = [(torch.arange(8).reshape(2, 4), torch.tensor([0, 1]))]
    first = analysis_context("model", config, calibration_batches=batches,
                             calibration_ids=["sample-0", "sample-1"])
    repeated = analysis_context("model", copy.deepcopy(config), calibration_batches=batches,
                                calibration_ids=["sample-0", "sample-1"])
    assert repeated == first

    changed_content = [(batches[0][0].clone(), batches[0][1])]
    changed_content[0][0][0, 0] += 1
    assert analysis_context(
        "model", config, calibration_batches=changed_content,
        calibration_ids=["sample-0", "sample-1"])["fingerprint"] != first["fingerprint"]

    changed_config = copy.deepcopy(config)
    changed_config["data"]["normalization"] = "imagenet"
    assert analysis_context(
        "model", changed_config, calibration_batches=batches,
        calibration_ids=["sample-0", "sample-1"])["fingerprint"] != first["fingerprint"]

    assert analysis_context(
        "model", config, calibration_batches=batches,
        calibration_ids=["different"])["fingerprint"] != first["fingerprint"]
