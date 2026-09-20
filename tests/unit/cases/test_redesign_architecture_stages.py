"""Regression and unit tests for redesign architecture stages A–H."""

from __future__ import annotations

from PIL import Image

from packages.calibration import (
    CalibrationDataset,
    CalibrationExample,
    select_threshold_for_precision,
)
from packages.candidate_evidence import CandidateEvidenceRecord, CandidateEvidenceStore
from packages.claim_decision import HitlRoute, route_claim_hitl
from packages.extraction_recovery.field_cascade import semantic_accept
from packages.field_authority import accept_field, independent_evidence_count
from packages.financial_reconciliation import FinancialDisposition, reconcile_claim_total
from packages.geometry_authority import (
    charge_region_verdict,
    is_pos_like_currency,
    reject_pos_as_charge,
)
from packages.image_evidence import InkDisposition, analyze_roi, requires_field_hitl
from packages.ocr_portfolio import monetary_crop_variants
from packages.package_intelligence import (
    PageClass,
    build_claim_package,
    classify_page_signals,
)


def test_blank_confirmed_optional_skips_hitl():
    img = Image.new("RGB", (80, 24), color=(255, 255, 255))
    evidence = analyze_roi(img, field_optional=True, ocr_empty=True)
    assert evidence.disposition == InkDisposition.BLANK_CONFIRMED
    assert requires_field_hitl(evidence, field_optional=True) is False


def test_form_ruling_dashes_are_blank_not_unreadable():
    """Box-28 dashed guidelines must not look like recoverable amount ink."""
    img = Image.new("L", (120, 28), color=255)
    pixels = img.load()
    # Horizontal dash train (form underline).
    for x0 in range(8, 110, 12):
        for x in range(x0, min(x0 + 7, 115)):
            for y in range(12, 15):
                pixels[x, y] = 0
    evidence = analyze_roi(img.convert("RGB"), ocr_empty=True)
    assert evidence.disposition == InkDisposition.BLANK_CONFIRMED
    assert evidence.features.get("ruling_only") == 1.0
    assert requires_field_hitl(evidence, field_optional=False) is False


def test_ink_present_unreadable_requires_hitl():
    # High-contrast ink strokes without OCR.
    img = Image.new("L", (80, 24), color=255)
    pixels = img.load()
    for x in range(10, 70):
        for y in range(8, 16):
            pixels[x, y] = 0
    evidence = analyze_roi(img.convert("RGB"), ocr_empty=True, geometry_valid=True)
    assert evidence.disposition in {
        InkDisposition.INK_PRESENT_UNREADABLE,
        InkDisposition.OCR_EMPTY_UNCLASSIFIED,
    }
    if evidence.disposition == InkDisposition.INK_PRESENT_UNREADABLE:
        assert requires_field_hitl(evidence, field_optional=False) is True


def test_package_intelligence_detects_separator_and_cms():
    sep = classify_page_signals(
        page_index=0,
        ocr_text="SourceHOV Document Separator",
        barcode_text="SEP-001",
    )
    assert sep.page_class == PageClass.SEPARATOR
    assert sep.is_separator is True
    cms = classify_page_signals(page_index=1, form_family="CMS1500", confidence=0.9)
    assert cms.page_class == PageClass.CMS1500
    assert cms.allows_cms_geometry is True
    package = build_claim_package(
        package_id="pkg-1",
        claim_id="M048DJJF.013",
        pages=[sep, cms],
    )
    assert package.complete is True
    assert any(i.value == "SEPARATOR_DETECTED" for i in package.issues)


def test_pos_11_rejected_outside_box_24f():
    # Reference-space bbox squarely in Box 24B (POS column).
    verdict = charge_region_verdict((420, 1500, 470, 1540), already_reference=True)
    assert verdict.authorised is False
    assert verdict.reason == "CHARGE_GEOMETRY_POS_BLEED"
    reject, reason = reject_pos_as_charge("11.00", (420.0, 1500.0, 470.0, 1540.0))
    assert reject is True
    assert "POS" in reason or "GEOMETRY" in reason


def test_charge_inside_box_24f_authorised():
    verdict = charge_region_verdict((1050, 1500, 1180, 1540), already_reference=True)
    assert verdict.authorised is True
    assert verdict.region == "BOX_24F"


def test_m048djjf_013_pos_11_cannot_be_total_or_line_charge():
    """Regression: POS 11 must not AUTO as service-line or total charge."""
    assert is_pos_like_currency("11.00")
    ok, reason = semantic_accept("total_charge", "11.00")
    assert ok is False
    assert reason == "CURRENCY_POS_LIKE_REQUIRES_GEOMETRY"

    # Single POS-like line without box-28 → POS_BLEED_REJECTED (not STP).
    result = reconcile_claim_total(
        box28_value=None,
        service_lines=[
            {
                "charges": "11.00",
                "semantic_region": "BOX_24B",
                "bbox": [420, 1500, 470, 1540],
                "row_id": "1",
            }
        ],
        charge_column_verified=True,
        all_service_rows_detected=True,
        independent_evidence_paths=2,
    )
    assert result.disposition == FinancialDisposition.POS_BLEED_REJECTED
    assert result.accepted_total is None

    route = route_claim_hitl(
        registration_ok=True,
        package_complete=True,
        unresolved_critical_fields=["total_charge"],
        financial_disposition=result.disposition.value,
    )
    assert route["route"] == HitlRoute.FINANCIAL_RECONCILIATION_HITL.value


def test_m048djjf_006_no_auto_without_corroboration():
    """297.00 / incomplete corroboration must not AUTO."""
    result = reconcile_claim_total(
        box28_value=None,
        service_lines=[
            {"charges": "270.00", "row_id": "1", "semantic_region": "BOX_24F",
             "bbox": [1050, 1500, 1180, 1540]},
            {"charges": "27.00", "row_id": "2", "semantic_region": "BOX_24F",
             "bbox": [1050, 1550, 1180, 1590]},
        ],
        charge_column_verified=True,
        all_service_rows_detected=True,
        independent_evidence_paths=1,  # insufficient for derived line-sum authority
    )
    assert result.accepted_total is None
    assert result.disposition in {
        FinancialDisposition.LINE_SUM_UNCORROBORATED,
        FinancialDisposition.INCOMPLETE_SERVICE_LINES,
        FinancialDisposition.CHARGE_COLUMN_UNVERIFIED,
    }


def test_direct_total_corroborated_within_cent():
    result = reconcile_claim_total(
        box28_value="281.00",
        service_lines=[
            {"charges": "270.00", "row_id": "1", "bbox": [1050, 1500, 1180, 1540]},
            {"charges": "11.00", "row_id": "2", "bbox": [1050, 1550, 1180, 1590]},
        ],
        charge_column_verified=True,
        all_service_rows_detected=True,
        independent_evidence_paths=1,
    )
    # 11.00 in 24F is a real charge here; box-28 matches line sum.
    assert result.disposition == FinancialDisposition.DIRECT_TOTAL_CORROBORATED
    assert result.accepted_total == "281.00"


def test_field_authority_blocks_llm_only_critical():
    decision = accept_field(
        field_name="total_charge",
        valid_geometry=True,
        valid_semantics=True,
        valid_format=True,
        calibrated_confidence=0.99,
        field_threshold=0.95,
        independent_evidence=1,
        required_evidence=1,
        unresolved_conflict=False,
        llm_only=True,
        critical=True,
    )
    assert decision.accepted is False
    assert decision.reason == "LLM_VLM_NOT_AUTHORITATIVE"


def test_independent_evidence_counts_groups_not_duplicate_reads():
    cands = [
        {"engine": "paddleocr", "value": "100.00"},
        {"engine": "paddleocr", "value": "100.00"},  # same family
        {"engine": "rapidocr", "value": "100.00"},
    ]
    assert independent_evidence_count(cands) == 2


def test_candidate_evidence_store_persists_lineage():
    store = CandidateEvidenceStore()
    store.add(
        CandidateEvidenceRecord(
            claim_id="M048DJJF.013",
            package_id="pkg",
            document_id="doc",
            page_number=1,
            field_name="charges",
            crop_bbox=(420, 1500, 470, 1540),
            authorised_semantic_region="BOX_24B",
            preprocessing_variant="original",
            engine="rapidocr",
            model_version="v1",
            raw_text="11",
            normalized_value="11.00",
            geometry_valid=False,
            rejection_reason="CHARGE_GEOMETRY_POS_BLEED",
        )
    )
    assert len(store) == 1
    assert store.for_field("charges")[0].geometry_valid is False


def test_monetary_variants_include_governed_set():
    img = Image.new("RGB", (40, 16), color=(240, 240, 240))
    variants = monetary_crop_variants(img)
    ids = {v.variant_id for v in variants}
    assert {"original", "scale_2x", "scale_4x", "inverted", "grayscale", "morph_close"}.issubset(ids)


def test_calibration_rejects_frozen_split_and_selects_precision_threshold():
    ds = CalibrationDataset()
    ds.add(
        CalibrationExample(
            field_name="total_charge",
            field_type="CURRENCY",
            document_class="CMS1500",
            ocr_confidence=0.9,
            engine_agreement=1.0,
            geometry_score=0.95,
            ink_quality=0.8,
            preprocessing_type="original",
            validation_ok=1.0,
            reconciliation_strength=1.0,
            source_independence=1.0,
            conflict=0.0,
            label_accepted_correct=1,
            split="development",
        )
    )
    try:
        ds.add(
            CalibrationExample(
                field_name="total_charge",
                field_type="CURRENCY",
                document_class="CMS1500",
                ocr_confidence=0.9,
                engine_agreement=1.0,
                geometry_score=0.95,
                ink_quality=0.8,
                preprocessing_type="original",
                validation_ok=1.0,
                reconciliation_strength=1.0,
                source_independence=1.0,
                conflict=0.0,
                label_accepted_correct=1,
                split="frozen_validation",
            )
        )
        assert False, "expected ValueError"
    except ValueError:
        pass
    thr = select_threshold_for_precision([0.99, 0.9, 0.5], [1, 1, 0], target_precision=0.995)
    assert thr is not None
