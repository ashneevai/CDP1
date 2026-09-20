"""STP/HITL tuning: CMS-1500 span cleanup and weak-E4 opt-out."""

from __future__ import annotations

from pathlib import Path

from packages.criticality import CriticalityLevel
from packages.evidence.models import EvidenceBundle, EvidenceClass, EvidenceItem
from packages.evidence.policy import EvidencePolicy
from packages.extraction_recovery.span_selection import select_field_span


def test_cms1500_name_span_keeps_surname_particle():
    """Multi-token surnames are printed left of the comma token."""
    selected = select_field_span(
        "2. PATIENT'S NAME (Last Name, First Name, Middle Initial)\nSAN NICOLAS, WILLIAM",
        "PERSON_NAME",
        "patient_name",
    )
    assert selected.selected_text == "SAN NICOLAS, WILLIAM"
    plain = select_field_span("MORALES, KENITHA", "PERSON_NAME", "patient_name")
    assert plain.selected_text == "MORALES, KENITHA"
    particles = select_field_span("DE LA CRUZ, MARIA", "PERSON_NAME", "insured_name")
    assert particles.selected_text == "DE LA CRUZ, MARIA"


def test_cms1500_name_span_strips_box_header_and_keeps_last_first():
    selected = select_field_span(
        "2. PATIENT'S NAME (Last Name, First Name, Middle lnitial)\nCAMARATO, JOSHUA",
        "PERSON_NAME",
        "patient_name",
    )
    assert selected.selected_text == "CAMARATO, JOSHUA"


def test_insured_name_strips_ocr_garble_1nsured_and_furst():
    selected = select_field_span(
        "4. 1NSURED'S NAME (Last Name. Furst Name, Middle Initial)\nSLOGER,\nCHLOE",
        "PERSON_NAME",
        "insured_name",
    )
    assert selected.selected_text == "SLOGER, CHLOE"


def test_patient_name_strips_glued_header_namelastname():
    selected = select_field_span(
        "2.PATIENT'SNAMELastName,FirstName,Middle Initia\nHLMESMALIK",
        "PERSON_NAME",
        "patient_name",
    )
    assert "PATIENT" not in selected.selected_text.upper()
    assert "HLMESMALIK" in selected.selected_text.replace(" ", "").upper()


def test_header_only_insured_name_not_name_shaped():
    from packages.extraction_recovery.field_cascade import semantic_accept

    selected = select_field_span(
        "4. INSURED'S NAME (Last Name,First Name,Middie Initial)",
        "PERSON_NAME",
        "insured_name",
    )
    ok, reason = semantic_accept("insured_name", selected.selected_text)
    assert not ok
    assert reason == "NAME_LABEL_CONTAMINATED"


def test_cms1500_member_id_span_extracts_trailing_identifier():
    selected = select_field_span(
        "1a. INSURED'S ID, NUMBER\n(For Program in Item 1)\n993751319",
        "ALPHANUMERIC_ID",
        "insured_id_number",
    )
    assert selected.selected_text == "993751319"


def test_cms1500_dob_token_assembly_and_rejects_garbage():
    ok = select_field_span(
        "3. PATIENT'S BIRTH DATE\nMM\nDD\nYY\n1\n4\n20\nM",
        "DATE",
        "patient_dob",
    )
    assert ok.selected_text == "01/04/2020"
    bad = select_field_span(
        "3. PATIENT'S BIATH DATE\n68\n11990\nM",
        "DATE",
        "patient_dob",
    )
    assert "19/90" not in bad.selected_text


def test_dob_midstream_letter_treated_as_separator_not_digit():
    """Blind HITL: OCR emits '01i081996' where i is a damaged slash, not a 1."""
    from packages.extraction_recovery.field_cascade import semantic_accept

    for raw in ("01i081996", "01i08 1996", "01l081996"):
        selected = select_field_span(raw, "DATE", "patient_dob")
        assert selected.selected_text == "01/08/1996", raw
        ok, reason = semantic_accept("patient_dob", selected.selected_text)
        assert ok and reason == "DATE_SHAPED", raw


def test_currency_npi_bleed_yields_empty_for_hitl():
    selected = select_field_span("L\nNPI", "CURRENCY", "total_charge")
    assert selected.selected_text == ""
    assert "NPI_LABEL_BLEED" in selected.reason_codes


def test_patient_name_policy_rejects_weak_e4_without_independent_confirmation():
    """Weak E4 supports patient_name but cannot replace independent E2/E5/E6."""
    policy = EvidencePolicy.load(Path("config/evidence_policies.yaml"))
    bundle = EvidenceBundle(
        field_name="patient_name",
        evidence_items=(
            EvidenceItem(
                evidence_class=EvidenceClass.E1,
                evidence_type="OCR_EXTRACTION",
                evidence_family="ocr",
                source="rapidocr",
                value="CAMARATO, JOSHUA",
            ),
            EvidenceItem(
                evidence_class=EvidenceClass.E3,
                evidence_type="TEMPLATE_REGISTRATION_CONFIRMED",
                evidence_family="registration",
                source="geometry",
                value="ok",
                metadata={"field_specific": True},
            ),
            EvidenceItem(
                evidence_class=EvidenceClass.E4,
                evidence_type="FORMAT_VALID",
                evidence_family="deterministic",
                source="validation",
                value="CAMARATO, JOSHUA",
                metadata={"strength": "WEAK"},
            ),
        ),
    )
    ok, available, missing, reasons = policy.evaluate(
        "patient_name", CriticalityLevel.C2, bundle, document_family="CMS1500"
    )
    assert not ok
    # Weak format validation is eligible support for this field, but cannot
    # authorize the value without independent E2, reference E5, or identity E6.
    assert "E4" in available
    assert missing == ("E2",)
    assert any("E2" in reason for reason in reasons)


def test_roi_insets_shrink_dob_and_charge_windows():
    from packages.extraction_recovery.roi_insets import inset_bbox
    dob = inset_bbox((667, 402, 886, 459), "patient_dob")
    assert dob[1] > 402  # top inset removes header band
    assert dob[2] < 886  # right inset clears sex checkbox
    charge = inset_bbox((1280, 1750, 1470, 1811), "total_charge")
    assert charge[0] > 1280  # left inset clears NPI legend bleed


def test_currency_rejects_npi_adjacent_one_dollar_artifact():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("1\nNPI", "CURRENCY", "total_charge")
    assert selected.selected_text == ""
    assert "NPI_LABEL_BLEED" in selected.reason_codes


def test_claim_total_e6_from_service_lines():
    from packages.claim_evidence.builder import ClaimEvidenceBuilder
    result = ClaimEvidenceBuilder.load().build(
        claim_id="c1",
        document_family="CMS1500",
        claim_values={"total_charge": "30.00"},
        service_lines=[{"charges": "20.00"}, {"charges": "10.00"}],
    )
    assert "CLAIM_TOTAL_CONFIRMED" in {i.evidence_type for i in result.evidence_items}


def test_dob_assembles_when_year_token_is_middle():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("112\n11970\n05", "DATE", "patient_dob")
    assert selected.selected_text == "12/05/1970"


def test_dob_assembles_day_edge_yyyy_month_token_order():
    """Track-B residual: digit-band OCR '116 11946 07' → 07/16/1946 (ink only)."""
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("116 11946 07", "DATE", "patient_dob")
    assert selected.selected_text == "07/16/1946"


def test_currency_rejects_non_digit_glyph_crop():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("一", "CURRENCY", "total_charge")
    assert selected.selected_text == ""


def test_dob_assembles_dd_yyyy_mm_token_order():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("29\n1983\n10", "DATE", "patient_dob")
    assert selected.selected_text == "10/29/1983"


def test_currency_repairs_p_separator_to_cents():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("2084P080", "CURRENCY", "total_charge")
    assert selected.selected_text == "2084.80"


def test_currency_rejects_leading_minus_total_as_form_artifact():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("-2084P080", "CURRENCY", "total_charge")
    assert selected.selected_text == ""
    assert "CURRENCY_LEADING_MINUS" in selected.reason_codes


def test_currency_accepts_whole_dollar_service_line_charges():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("225", "CURRENCY", "charges")
    assert selected.selected_text == "225.00"


def test_name_span_strips_header_before_label_phrases():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span(
        "2.PATIENTS NAME(Lasi Name,First Name,MiddleInitial)\nDOLIET\nMARGARET M",
        "PERSON_NAME",
        "patient_name",
    )
    assert selected.selected_text == "DOLIET MARGARET M"


def test_member_id_repairs_ocr_zero_and_equals():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span(
        "1a.INSURED'S LD. NUMBER\n(For Program in ltem 1)\n0SC74765420",
        "ALPHANUMERIC_ID",
        "insured_id_number",
    )
    assert selected.selected_text == "OSC74765420"
    selected2 = select_field_span(
        "1a. INSURED'S I.D. NUM8ER\n(For Program in ltem 1)\nNALC\nP32=84957",
        "ALPHANUMERIC_ID",
        "insured_id_number",
    )
    assert selected2.selected_text == "P32-84957"


def test_dob_assembles_when_year_has_trailing_period():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("29\n1983\n10.", "DATE", "patient_dob")
    assert selected.selected_text == "10/29/1983"

def test_dob_assembles_cjk_confusable_digit_stream():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("04 1 了 .9 9 1 1", "DATE", "patient_dob")
    assert selected.selected_text == "04/17/1991"


def test_insured_name_prefers_ink_below_header_boilerplate():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span(
        "4. INSURED'S NAME (Last NaTe, First NaTe, Midale Inilial)\n2\nMCQUEEN, VASHONDA",
        "PERSON_NAME",
        "insured_name",
    )
    assert selected.selected_text == "MCQUEEN, VASHONDA"


def test_insured_name_route_authority_present():
    from pathlib import Path

    from packages.route_registry.registry import RouteRegistry
    registry = RouteRegistry.load(Path("config/ocr_field_routes.yaml"))
    route = registry.find("insured_name", "CMS1500", mode="runtime")
    assert route is not None
    assert route.status.value == "PRODUCTION_APPROVED"
    assert route.primary_engine == "rapidocr"
    assert route.confirmation_engine == "paddleocr"
    assert route.route_id == "ANY.insured_name.rapidocr.paddleocr.v1"

def test_dob_assembles_trailing_letter_bleed_and_three_digit_year():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("12 26l 108", "DATE", "patient_dob")
    assert selected.selected_text == "12/26/2008"


def test_dob_rejects_impossible_calendar_day():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("11 31 93", "DATE", "patient_dob")
    assert selected.selected_text != "11/31/1993"
    assert "11/31" not in selected.selected_text


def test_currency_repairs_i_slash_zero_zero_handwritten_charge():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("I/00", "CURRENCY", "charges")
    assert selected.selected_text == "100.00"
    assert "CURRENCY_CONFUSABLE_REPAIRED" in selected.reason_codes


def test_currency_repairs_l00_confusable_charge():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("L00", "CURRENCY", "charges")
    assert selected.selected_text == "100.00"


def test_dob_repairs_three_digit_year_missing_century_one():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("03 19 983", "DATE", "patient_dob")
    assert selected.selected_text == "03/19/1983"


def test_dob_merges_split_day_around_century_repaired_year():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("03 1 983 1 9 7 1", "DATE", "patient_dob")
    assert selected.selected_text == "03/19/1983"


def test_dob_does_not_merge_day_when_year_already_four_digits():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("12 2 1983 6", "DATE", "patient_dob")
    assert selected.selected_text == "12/02/1983"


def test_dob_repairs_trailing_edge_on_ten_xx_year():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("12 26l 1083", "DATE", "patient_dob")
    assert selected.selected_text == "12/26/2008"


def test_dob_locates_century_clipped_year_after_day_fragments():
    from packages.extraction_recovery.span_selection import select_field_span
    selected = select_field_span("03 9 1 983 1 1", "DATE", "patient_dob")
    assert selected.selected_text == "03/19/1983"


def test_dob_u7_confusable_and_glued_year_assembles():
    """Handwritten 0→U and day||year glue (11990) must assemble observed DOB."""
    from packages.extraction_recovery.span_selection import select_field_span

    span = select_field_span("U7 11 11990 M", "DATE", "patient_dob")
    assert span.selected_text == "07/11/1990"


def test_dob_split_day_fragments_before_full_year():
    """Single-digit day fragments merge only when uniquely calendar-valid."""
    from packages.extraction_recovery.span_selection import select_field_span

    span = select_field_span("6 4 1 1974", "DATE", "patient_dob")
    assert span.selected_text == "06/14/1974"
    ambiguous = select_field_span("12 2 1 1983", "DATE", "patient_dob")
    assert ambiguous.selected_text in {"12/02/1983", "12/2/1983", ""}


def test_dob_trailing_edge_one_peel_on_month_day():
    """3-digit MM/DD ending in edge-1 peel before century-clipped year."""
    from packages.extraction_recovery.span_selection import select_field_span

    assert select_field_span("051 291 196", "DATE", "patient_dob").selected_text == "05/29/1996"
    assert select_field_span("051 031 1982", "DATE", "patient_dob").selected_text == "05/03/1982"
    assert select_field_span("021 241 196", "DATE", "patient_dob").selected_text == "02/24/1996"


def test_dob_header_safe_compact_and_single_l_confusable():
    """MM/DD headers must not D→0-poison compact; standalone L→1."""
    from packages.extraction_recovery.span_selection import select_field_span

    span = select_field_span("MM DD 0 9 2 9 L 9 6", "DATE", "patient_dob")
    assert span.selected_text == "09/29/1996"
    assert select_field_span("09 29 196", "DATE", "patient_dob").selected_text == "09/29/1996"


def test_cross_variant_span_fusion_recovers_dob():
    from packages.extraction_recovery.field_cascade import FieldCascade

    calls = []

    def recognize_fn(field_name, bbox, field_type, engines):
        calls.append(bbox)
        if len(calls) == 1:
            return (
                [{"value": "6 4 1 g 7", "raw_value": "6\n4\n1\ng\n7"}],
                [{"engine": engines[0], "reason": "OBSERVED"}],
                "POLICY_SATISFIED",
            )
        if len(calls) == 2:
            return (
                [{"value": "ign i 1974", "raw_value": "ign\ni\n1974"}],
                [{"engine": engines[0], "reason": "OBSERVED"}],
                "POLICY_SATISFIED",
            )
        return (
            [{"value": "", "raw_value": ""}],
            [{"engine": engines[0], "reason": "EMPTY"}],
            "EMPTY",
        )

    result = FieldCascade().recognize(
        field_name="patient_dob",
        primary_bbox=(672, 424, 871, 457),
        cell={"x0": 667, "y0": 402, "x1": 886, "y1": 459},
        image_size=(1700, 2200),
        recognize_fn=recognize_fn,
    )
    assert result.accepted is True
    assert result.candidates
    assert result.candidates[0]["value"] == "06/14/1974"
    assert any(s.variant_id == "cross_variant_span" for s in result.cascade_trace)


def test_dob_yy_pivot_aligns_with_reconciler_not_future():
    """Independent-300 v12.1 learning: YY=30 under <=36→20xx minted FUTURE DOBs."""
    from packages.extraction_recovery.span_selection import select_field_span

    assert select_field_span("9 1 30", "DATE", "patient_dob").selected_text == "09/01/1930"
    assert select_field_span("09 01 34", "DATE", "patient_dob").selected_text == "09/01/1934"
    assert select_field_span("12 21 30", "DATE", "patient_dob").selected_text == "12/21/1930"
    # YY<30 still maps to 20xx when not future.
    assert select_field_span("09 01 02", "DATE", "patient_dob").selected_text == "09/01/2002"


def test_dob_colon_comma_period_confusables_shape_as_separators():
    """HJE5.016-class DI ink: ``7:30.77`` / ``7:30,77`` → calendar DOB."""
    from packages.extraction_recovery.span_selection import (
        _assemble_dob_from_tokens,
        _normalize_dob_punct_separators,
        select_field_span,
    )

    assert _normalize_dob_punct_separators("7:30.77") == "7/30/77"
    assert _normalize_dob_punct_separators("7:30,77") == "7/30/77"
    assert _assemble_dob_from_tokens("7:30.77") == "07/30/1977"
    assert _assemble_dob_from_tokens("7:30,77") == "07/30/1977"
    assert select_field_span("7:30.77", "DATE", "patient_dob").selected_text == (
        "07/30/1977"
    )
    assert select_field_span("7:30,77", "DATE", "patient_dob").selected_text == (
        "07/30/1977"
    )


def test_person_name_period_is_not_last_first_separator():
    """OCR mid-name periods must not invent Last, First (DATST, EY)."""
    from packages.extraction_recovery.span_selection import select_field_span

    span = select_field_span("TOHNCON\nDAtSt.EY", "PERSON_NAME", "patient_name")
    assert span.selected_text != "DATST, EY"
    assert "TOHNCON" in (span.selected_text or "").upper()
