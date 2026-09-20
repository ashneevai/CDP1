"""Sample-derived charge stem rules. No claim-id branches, no golden values."""

from packages.claim_evidence.line_sum_authority import (
    line_has_dual_engine_agreement,
    line_has_gpt4o_local_consensus,
    line_sum_auto_eligible,
)
from packages.ocr_portfolio.monetary_recognizer import (
    apply_charge_line_resolution,
    is_ruling_tick_charge,
    resolve_service_charge,
    ruling_geometry_supports_charge,
)
from scripts.ocr_from_geometry import dob_boxed_cells_complete


def _cand(engine, value, raw=None, variant="CURRENCY_DECIMAL_V2"):
    return {
        "engine": engine,
        "value": value,
        "raw_value": raw if raw is not None else value,
        "preprocessing_variant": variant,
    }


def test_units_bleed_with_ruling_clip_selects_vision_stem():
    """Full window 2001, dollars-ruling 20, vision 200 → 200. Not 2001."""
    cands = [
        _cand("paddleocr", "2001.00", "2001"),
        _cand("paddleocr", "20.00", "20", "CURRENCY_DECIMAL_V2|dollars_ruling"),
        _cand("rapidocr", "20.00", "20.", "CURRENCY_DECIMAL_V2|dollars_ruling"),
        _cand("azure_gpt4o_crop", "200.00", "200.00", "gpt4o_crop_residual"),
    ]
    value, tag = resolve_service_charge("2001.00", cands)
    assert value == "200.00"
    assert tag == "RULING_CLIPPED_ZERO_STEM"
    line = {"charges": value, "candidates": cands}
    assert ruling_geometry_supports_charge(cands, "200.00")
    assert line_has_gpt4o_local_consensus(line)
    ok, reason = line_sum_auto_eligible([line])
    assert not ok and reason == "GPT4O_LOCAL_NEEDS_BOX28"
    ok, reason = line_sum_auto_eligible([line], corroborating_values=["200.00"])
    assert ok and reason == "BOX28_OR_DI_CORROBORATED"


def test_local_stem_beats_one_digit_bleed_sibling():
    """Paddle 270 + rapid 2703 + vision 270 → 270, including trailing digit 3."""
    cands = [
        _cand("paddleocr", "270.00", "270"),
        _cand("rapidocr", "2703.00", "2703"),
        _cand("paddleocr", "27.00", "27", "CURRENCY_DECIMAL_V2|dollars_ruling"),
        _cand("azure_gpt4o_crop", "270.00", "270.00", "gpt4o_crop_residual"),
    ]
    value, tag = resolve_service_charge("2703.00", cands)
    assert value == "270.00"
    assert tag == "LOCAL_STEM_OVER_BLEED"
    assert line_has_gpt4o_local_consensus({"charges": value, "candidates": cands})


def test_clipped_zero_without_bleed_sibling_still_supports_consensus():
    """Ruling crop 21 and vision 210.00 — the dashed line ate the trailing 0."""
    cands = [
        _cand("paddleocr", "21.00", "1\n21n"),
        _cand("rapidocr", "21.00", "21:", "CURRENCY_DECIMAL_V2|dollars_ruling"),
        _cand("azure_gpt4o_crop", "210.00", "210.00", "gpt4o_crop_residual"),
    ]
    value, tag = resolve_service_charge("210.00", cands)
    assert value == "210.00"
    assert tag == "RULING_CLIPPED_ZERO_STEM"
    assert line_has_gpt4o_local_consensus({"charges": "210.00", "candidates": cands})


def test_digit_drop_without_ruling_variant_does_not_auto():
    """13 vs 131 stays fail-closed. Geometry, not a shorter-wins threshold."""
    cands = [
        _cand("paddleocr", "13.00", "13"),
        _cand("azure_gpt4o_crop", "131.00", "131.00", "gpt4o_crop_residual"),
    ]
    value, tag = resolve_service_charge("131.00", cands)
    assert value == "131.00"
    assert tag == ""
    assert not ruling_geometry_supports_charge(cands, "13.00")
    assert not line_has_gpt4o_local_consensus(
        {"charges": "131.00", "candidates": cands}
    )
    assert not line_has_gpt4o_local_consensus(
        {
            "charges": "6430.00",
            "candidates": [
                _cand("paddleocr", "643.00"),
                _cand("azure_gpt4o_crop", "6430.00", "6430.00", "gpt4o_crop_residual"),
            ],
        }
    )


def test_ruling_tick_rows_are_dropped_and_cents_glue_blocks_auto():
    tick = {
        "charges": "111.00",
        "candidates": [
            _cand("paddleocr", "", "1\n1\n1"),
            _cand("rapidocr", "111.00", "111"),
            _cand("azure_gpt4o_crop", "4972.00", "4972", "gpt4o_crop_residual"),
        ],
    }
    kept_line = {
        "charges": "49.00",
        "candidates": [
            _cand("rapidocr", "49.00", "49"),
            _cand("azure_gpt4o_crop", "49.00", "49.00", "gpt4o_crop_residual"),
        ],
    }
    assert is_ruling_tick_charge(tick["candidates"])
    resolved = apply_charge_line_resolution([kept_line, tick])
    assert len(resolved) == 1
    assert resolved[0]["charges"] == "49.00"
    assert resolved[0].get("cents_unresolved") is True
    assert not line_has_gpt4o_local_consensus(resolved[0])
    ok, _reason = line_sum_auto_eligible(resolved)
    assert not ok


def test_blank_grid_ticks_do_not_inflate_a_corroborated_line():
    real = {
        "charges": "115.00",
        "candidates": [
            _cand("paddleocr", "115.00", "115"),
            _cand("rapidocr", "115.00", "115"),
            _cand("azure_gpt4o_crop", "115.00", "115.00", "gpt4o_crop_residual"),
        ],
    }
    blank = {
        "charges": "111.00",
        "candidates": [
            _cand("paddleocr", "", "1\n1\n1"),
            _cand("rapidocr", "", "一\n1"),
            _cand("azure_gpt4o_crop", "111.00", "111.00", "gpt4o_crop_residual"),
        ],
    }
    resolved = apply_charge_line_resolution([real, blank])
    assert len(resolved) == 1
    assert resolved[0]["charges"] == "115.00"
    ok, reason = line_sum_auto_eligible(resolved)
    assert not ok and reason == "GPT4O_LOCAL_NEEDS_BOX28"
    ok, reason = line_sum_auto_eligible(resolved, corroborating_values=["115.00"])
    assert ok and reason == "BOX28_OR_DI_CORROBORATED"


def test_ruled_cents_reconstructed_from_split_raw():
    """Raw ``34`` / ``25`` with vision ``34.25`` is dollars|cents, not 34.00."""
    cands = [
        _cand("paddleocr", "25.00", "34\n25"),
        _cand("rapidocr", "34.00", "25\n34"),
        _cand("paddleocr", "34.00", "34", "CURRENCY_DECIMAL_V2|dollars_ruling"),
        _cand("azure_gpt4o_crop", "34.25", "34.25", "gpt4o_crop_residual"),
    ]
    value, tag = resolve_service_charge("34.00", cands)
    assert value == "34.25"
    assert tag == "RULED_CENTS_RECONSTRUCTED"
    line = {"charges": value, "candidates": cands}
    assert line_has_gpt4o_local_consensus(line)
    ok, reason = line_sum_auto_eligible([line])
    assert not ok and reason == "GPT4O_LOCAL_NEEDS_BOX28"
    ok, reason = line_sum_auto_eligible([line], corroborating_values=["34.25"])
    assert ok and reason == "BOX28_OR_DI_CORROBORATED"


def test_dual_engine_must_match_selected_charge():
    line = {
        "charges": "200.00",
        "candidates": [
            _cand("paddleocr", "20.00", "20C"),
            _cand("rapidocr", "20.00", "20C"),
        ],
    }
    assert not line_has_dual_engine_agreement(line)


def test_boxed_dob_rejects_single_digit_day():
    assert not dob_boxed_cells_complete("07", "1", "46")
    assert not dob_boxed_cells_complete("7", "16", "46")
    assert dob_boxed_cells_complete("07", "16", "46")
    assert dob_boxed_cells_complete("07", "16", "1946")


def test_non_ones_digit_keeps_row():
    """A ``4`` beside ``111`` is not a pure ruling tick (do not drop the row)."""
    cands = [
        _cand("rapidocr", "111.00", "111"),
        _cand("tesseract_digits", "", "4"),
    ]
    assert not is_ruling_tick_charge(cands, "111.00")


def test_quarantine_keys_match_claim_id_spellings():
    from scripts.score_hackathon_gt_accuracy import _quarantined

    table = {("M048DJJM.013", "total_charge"), ("Group A/M048DJJM.013", "total_charge")}
    assert _quarantined("Group A__M048DJJM.013", "total_charge", table)
    assert not _quarantined("Group A__M048DJJM.013", "patient_name", table)
