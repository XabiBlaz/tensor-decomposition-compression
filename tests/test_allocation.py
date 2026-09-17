from types import SimpleNamespace

import pytest
import torch

from tn_compression.allocation import allocate_ranks, candidate_frontier


def test_frontier_discards_non_saving_and_dominated_candidates():
    rows = [{"path": "a", "rank": rank, "bytes_saved": saved, "delta_nll": damage}
            for rank, saved, damage in [(4, 0, 0), (3, 10, 2), (2, 20, 1), (1, 30, 3)]]
    assert [row["rank"] for row in candidate_frontier(rows)] == [2, 1]


class TwoProjectionLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4, bias=False)
        self.b = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.a.weight.copy_(torch.eye(4))
            self.b.weight.copy_(torch.eye(4))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        inputs = torch.nn.functional.one_hot(input_ids, 4).float()
        return SimpleNamespace(logits=self.b(self.a(inputs)))


def test_allocation_enforces_quality_and_reports_infeasible_target():
    batches = [{"input_ids": torch.tensor([[3, 3, 3, 3]]), "attention_mask": torch.ones(1, 4)}]
    model = TwoProjectionLM()
    originals = (model.a, model.b)
    result = allocate_ranks(model, batches, batches, {"a": [1], "b": [1]}, target_bytes=64, max_delta_nll=0)
    assert result["status"] == "infeasible"
    assert (model.a, model.b) == originals
    assert result["model_tensor_bytes"] == 128
    result = allocate_ranks(model, batches, batches, {"a": [1], "b": [1]}, target_bytes=64, max_delta_nll=2)
    assert result["status"] == "feasible" and len(result["rounds"]) == 2
    assert result["final"]["nll"] - result["baseline"]["nll"] <= 2
    # Round two was measured on the compressed first layer, not a sum of two
    # independently measured damages from the original model.
    assert result["rounds"][1]["search"]["baseline"]["nll"] == result["rounds"][0]["observed"]["nll"]
