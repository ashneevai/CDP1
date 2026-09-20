from pathlib import Path

import pytest

from packages.criticality import CriticalityLevel
from packages.domain.common import BoundingBox
from packages.evidence import EvidenceClass, EvidencePolicy, EvidencePolicyUnavailableError
from packages.evidence_decision import (
    DecisionContext,
    EvidenceDecisionService,
    FieldDisposition,
)
from packages.ocr.contracts import OCRCandidate
from packages.route_registry import (
    RouteLifecycle,
    RouteNotApprovedError,
    RouteRegistry,
    RouteRegistryUnavailableError,
)

BOX = BoundingBox(x0=0, y0=0, x1=10, y1=5, image_width=10, image_height=5)


def candidate(engine: str) -> OCRCandidate:
    return OCRCandidate(
        value="JANE DOE", raw_value="JANE DOE", engine=engine,
        model_name=engine, model_version="1", preprocessing_variant="test",
        raw_confidence=.99, calibrated_confidence=None, bounding_box=BOX,
        latency_ms=1,
    )


def decision_context() -> DecisionContext:
    return DecisionContext(
        field_name="patient_name", document_family="CMS1500",
        criticality=CriticalityLevel.C2, blocks_stp=True,
        candidates=[candidate("tesseract"), candidate("paddleocr")],
        deterministic_evidence={"HARD_VALIDATION_PASSED"},
        hard_validation_passed=True, registration_confidence=.95,
    )


def test_registry_enforces_explicit_route_lifecycle():
    registry = RouteRegistry.load()

    assert len(registry.routes) == 10
    assert {route.status for route in registry.routes} == {
        RouteLifecycle.PRODUCTION_APPROVED,
        RouteLifecycle.EVALUATION_ONLY,
    }
    assert {route.field for route in registry.routes_for_mode("runtime")} == {
        "federal_tax_no",
        "insured_id_number",
        "insured_dob",
        "insured_name",
        "patient_dob",
        "patient_name",
        "provider_npi",
        "total_charge",
    }
    assert len(registry.routes_for_mode("evaluation")) == 10
    assert registry.routes_for_mode("shadow") == ()


def test_registry_rejects_evaluation_route_in_runtime():
    registry = RouteRegistry.load()
    route = registry.find_any("type_of_bill", "UB04")
    assert route is not None
    assert route.status.value == "EVALUATION_ONLY"

    with pytest.raises(RouteNotApprovedError, match="not allowed in runtime"):
        registry.require(route.route_id, mode="runtime")


def test_missing_registry_fails_closed(tmp_path: Path):
    with pytest.raises(RouteRegistryUnavailableError, match="unavailable"):
        RouteRegistry.load(tmp_path / "missing.yaml")


def test_missing_evidence_policy_fails_closed(tmp_path: Path):
    with pytest.raises(EvidencePolicyUnavailableError, match="unavailable"):
        EvidencePolicy.load(tmp_path / "missing-policy.yaml")


def test_evaluation_only_confirmation_cannot_influence_runtime_decision():
    context = DecisionContext(
        field_name="type_of_bill",
        document_family="UB04",
        criticality=CriticalityLevel.C2,
        blocks_stp=True,
        candidates=[candidate("tesseract"), candidate("paddleocr")],
        deterministic_evidence={"HARD_VALIDATION_PASSED"},
        hard_validation_passed=True,
        registration_confidence=0.95,
    )
    runtime = EvidenceDecisionService(route_mode="runtime").decide(context)
    evaluation = EvidenceDecisionService(route_mode="evaluation").decide(context)

    assert runtime.disposition is not FieldDisposition.AUTO_ACCEPTED
    assert any(code.startswith("ROUTE_STATUS_REJECTED:") for code in runtime.reason_codes)
    assert runtime.evidence_bundle is not None
    assert runtime.evidence_bundle.route_status == "EVALUATION_ONLY"
    assert runtime.evidence_bundle.route_mode == "runtime"
    assert runtime.evidence_bundle.rejected_route_ids == [runtime.evidence_bundle.route_id]
    assert EvidenceClass.E2 not in runtime.evidence_bundle.available_classes

    assert evaluation.disposition is not FieldDisposition.AUTO_ACCEPTED
    assert evaluation.evidence_bundle is not None
    assert evaluation.evidence_bundle.route_status == "EVALUATION_ONLY"
    assert evaluation.evidence_bundle.route_mode == "evaluation"
    assert evaluation.evidence_bundle.rejected_route_ids == []
    agreement = next(
        item for item in evaluation.evidence_bundle.items
        if item.evidence_class is EvidenceClass.E2
    )
    assert agreement.evidence_type == "OCR_AGREEMENT_UNKNOWN_DEPENDENCY"
    assert not agreement.independent


def test_candidate_engine_must_be_authorized_by_production_route():
    context = DecisionContext(
        field_name="provider_npi", document_family="CMS1500",
        criticality=CriticalityLevel.C2, blocks_stp=True,
        candidates=[candidate("paddleocr"), candidate("gemini")],
        deterministic_evidence={"CHECKSUM_VALID"}, hard_validation_passed=True,
    )
    decision = EvidenceDecisionService(route_mode="runtime").decide(context)
    assert decision.evidence_bundle is not None
    assert len(decision.evidence_bundle.candidate_ids) == 1
    assert any(code.startswith("CANDIDATE_ENGINE_NOT_AUTHORIZED:") for code in decision.reason_codes)
