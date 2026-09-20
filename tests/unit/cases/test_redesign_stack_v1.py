"""Tests for redesign-stack-v1 architecture."""

from __future__ import annotations

from PIL import Image

from packages.architecture import (
    assert_ladder_order,
    charge_residual_ladder,
    estimate_acceptance_risk,
    gpt_may_be_sole_monetary_authority,
    independence_group_for_engine,
    load_redesign_stack,
    next_step_after,
)
from packages.ocr.independence import independence_group
from workers.complex_tables import complex_tables_enabled, recognize_complex_table
from workers.monkeyocr import monkeyocr_enabled, recognize_table
from workers.openocr_svtr import openocr_svtr_enabled, recognize_openocr_svtr


def test_redesign_stack_loads_capability_table():
    load_redesign_stack.cache_clear()
    stack = load_redesign_stack()
    assert stack.version == "redesign-stack-v1"
    assert stack.status == "ACTIVE"
    assert "registration" in stack.capabilities
    assert "llm_vlm" in stack.capabilities
    assert "hitl_ui" in stack.capabilities
    assert "validation" in stack.capabilities
    assert "calibration" in stack.capabilities
    assert stack.capabilities["llm_vlm"].raw.get("monetary_authority") is False
    assert gpt_may_be_sole_monetary_authority(stack) is False
    assert "gpt4o_empty_finance_sweep" in stack.charge_residual_ladder
    assert "field_scoped_hitl" in stack.charge_residual_ladder


def test_independence_groups_include_openocr_and_vl():
    assert independence_group("openocr_svtr") == "PADDLE_FAMILY"
    assert independence_group("monkeyocr") == "VL_TABLE_FAMILY"
    assert independence_group_for_engine("azure_gpt4o_crop") == "GPT4O_FAMILY"


def test_acceptance_risk_blocks_gpt_only_charge():
    est = estimate_acceptance_risk(
        field_name="total_charge",
        calibrated_confidence=0.99,
        engine_count=1,
        gpt4o_only=True,
    )
    assert est.review_recommended is True
    assert est.risk >= 0.9
    assert est.method == "gpt_not_monetary_authority"


def test_charge_ladder_order_and_next_step():
    ladder = charge_residual_ladder()
    assert ladder[0] == "local_paddle_rapid"
    assert ladder[-1] == "field_scoped_hitl"
    assert assert_ladder_order(list(ladder)) is True
    assert assert_ladder_order(["gpt4o_empty_finance_sweep", "local_paddle_rapid"]) is False
    nxt = next_step_after(["local_paddle_rapid", "tesseract_digits_fill"])
    assert nxt is not None
    assert nxt.id == "gpt4o_empty_finance_sweep"
    assert "openocr_svtr_optional" not in ladder


def test_openocr_svtr_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CDP_OPENOCR_SVTR", raising=False)
    assert openocr_svtr_enabled() is False
    img = Image.new("RGB", (64, 32), color=(255, 255, 255))
    result = recognize_openocr_svtr(img)
    assert result.attempted is False
    assert result.reason == "OPENOCR_SVTR_DISABLED"


def test_monkeyocr_stub_unavailable_when_enabled(monkeypatch):
    monkeypatch.setenv("CDP_MONKEYOCR", "1")
    assert monkeyocr_enabled() is True
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    result = recognize_table(img)
    assert result.reason == "MONKEYOCR_UNAVAILABLE"
    assert result.review_only is True


def test_complex_tables_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CDP_PADDLEOCR_VL_TABLE", raising=False)
    monkeypatch.delenv("CDP_MONKEYOCR", raising=False)
    assert complex_tables_enabled() is False
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    result = recognize_complex_table(img)
    assert result.attempted is False
    assert result.reason == "COMPLEX_TABLES_DISABLED"


def test_complex_tables_monkey_path_review_only(monkeypatch):
    monkeypatch.delenv("CDP_PADDLEOCR_VL_TABLE", raising=False)
    monkeypatch.setenv("CDP_MONKEYOCR", "1")
    img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    result = recognize_complex_table(img)
    assert result.review_only is True
    assert "monkeyocr" in result.engines_tried
    assert result.reason == "COMPLEX_TABLES_UNAVAILABLE"
