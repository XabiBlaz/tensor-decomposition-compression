"""Serializable evidence records shared by the API, CLI and future UI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_candidate_id(model_fingerprint: str, layer_path: str, method: str,
                        configuration: Mapping[str, Any]) -> str:
    """Identify a proposed transformation independently of measured evidence."""
    payload = {
        "model_fingerprint": model_fingerprint,
        "layer_path": layer_path,
        "method": method,
        "configuration": dict(configuration),
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:20]


def capability(supported: bool, reason: str, *, verified: bool = False) -> Dict[str, Any]:
    return {"supported": bool(supported), "verified": bool(verified), "reason": str(reason)}


@dataclass
class CandidateResult:
    model_fingerprint: str
    layer_path: str
    module_type: str
    method: str
    configuration: Dict[str, Any]
    structurally_eligible: bool
    original_parameters: int
    candidate_parameters: Optional[int]
    estimated_artifact_bytes: Optional[int]
    candidate_id: str = ""
    rejection_reason: Optional[str] = None
    protected: bool = False
    original_artifact_bytes: Optional[int] = None
    measured_artifact_bytes: Optional[int] = None
    normalized_local_error: Optional[float] = None
    local_error: Optional[Dict[str, Any]] = None
    full_model_metrics: Optional[Dict[str, Any]] = None
    full_model_metric_delta: Optional[Dict[str, Any]] = None
    quality_loss: Optional[float] = None
    calibration_evidence: Optional[Dict[str, Any]] = None
    checkpoint_reconstruction: Dict[str, Any] = field(
        default_factory=lambda: capability(False, "not_assessed"))
    backend_support: Dict[str, Any] = field(
        default_factory=lambda: capability(False, "not_assessed"))
    status: str = "discovered"
    decision: str = "pending"
    decision_reason: Optional[str] = None

    def __post_init__(self):
        if not self.candidate_id:
            self.candidate_id = stable_candidate_id(
                self.model_fingerprint, self.layer_path, self.method, self.configuration)

    @property
    def estimated_bytes_saved(self) -> Optional[int]:
        if self.original_artifact_bytes is None or self.estimated_artifact_bytes is None:
            return None
        return self.original_artifact_bytes - self.estimated_artifact_bytes

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["estimated_bytes_saved"] = self.estimated_bytes_saved
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateResult":
        fields = dict(value)
        fields.pop("estimated_bytes_saved", None)
        return cls(**fields)


@dataclass
class AnalysisReport:
    level: str
    model_fingerprint: str
    task: str
    requested_backend: str
    candidates: List[CandidateResult]
    capabilities: Dict[str, Any]
    baseline_metrics: Optional[Dict[str, Any]] = None
    cumulative_metrics: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    compression_plan: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisReport":
        fields = dict(value)
        if fields.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported analyzer schema version.")
        fields["candidates"] = [CandidateResult.from_dict(item) for item in fields["candidates"]]
        return cls(**fields)
