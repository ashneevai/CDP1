"""When box-28 OCR is empty, invalid, or strongly contradicts observed lines,
prefer LINE_TOTALS_RECONCILED from service-line ink (never invent amounts).

AUTO on line-sum is fail-closed. Eligible path:

  - service-line sum corroborated by independently extracted box-28 / DI
    (``amounts_corroborate``, which may use digit-drop twins — box-28/DI only).

Dual-engine or gpt-4o+local agreement on service-line crops alone is **not**
enough for AUTO: those share the Box 24F evidence path and are not a second
printed occurrence of the claim total. GPT-4o never independently promotes a
critical charge field.

Shell / form-noise locals are not corroboration. Shared crop, parent evidence,
or independence_group lineage means candidates are not independent → HITL.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_CHARGE_FIELDS = ("total_charge", "total_charges", "charges", "charge_amount")
_INDEPENDENT_ENGINES = frozenset({"paddleocr", "rapidocr", "azure_document_intelligence_read"})
_LOCAL_ENGINES = frozenset({"paddleocr", "rapidocr"})
_GPT4O_FAMILY = "azure_gpt4o_crop"
# CMS-1500 box-28 digit-soup / form-ruling OCR (208408, 420840) is not a total.
_MAX_PLAUSIBLE_CLAIM_TOTAL = Decimal("99999.99")


def parse_currency(value: object) -> Decimal | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    # OCR often emits space/colon as the dollars|cents separator, or a
    # thousands space (``4 972``). Repair before stripping non-digits.
    text = raw.lstrip("$").replace(",", "")
    text = re.sub(r"[Oo]", "0", text)  # confusable O→0 in amounts
    text = re.sub(r"[gG]", "0", text)  # confusable g→0 (``60g.00``)
    m = re.fullmatch(r"(\d{1,6})[\s:.](\d{2})", text)
    if m:
        text = f"{m.group(1)}.{m.group(2)}"
    else:
        m = re.fullmatch(r"(\d{1,3})\s(\d{3})", text)
        if m:
            text = f"{m.group(1)}{m.group(2)}.00"
        else:
            m = re.fullmatch(r"(\d{1,3})\s(\d{3})[\s:.](\d{2})", text)
            if m:
                text = f"{m.group(1)}{m.group(2)}.{m.group(3)}"
    try:
        return Decimal(re.sub(r"[^0-9.-]", "", text))
    except (InvalidOperation, ValueError):
        return None


def format_currency(amount: Decimal) -> str:
    return format(amount.quantize(Decimal("0.01")), "f")


def observed_line_charges(service_lines: list[dict] | None) -> list[Decimal]:
    charges: list[Decimal] = []
    for line in service_lines or []:
        for key in _CHARGE_FIELDS:
            parsed = parse_currency(line.get(key))
            if parsed is not None and parsed >= 0:
                charges.append(parsed)
                break
    return charges


def is_suspicious_tiny_total(amount: Decimal) -> bool:
    """Match field-cascade CURRENCY_SUSPICIOUS_TINY (single digit before cents)."""
    text = format_currency(amount)
    return bool(re.fullmatch(r"[0-9]\.\d{2}", text))


def _currency_digit_string(amount: object) -> str:
    text = str(amount or "").strip().lstrip("$").replace(",", "")
    if not text:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    return re.sub(r"\D", "", text)


def is_implausible_charge_total(amount: object) -> bool:
    """True for form-ruling digit soup / impossible CMS-1500 claim totals."""
    parsed = parse_currency(amount)
    if parsed is None:
        return False
    if parsed > _MAX_PLAUSIBLE_CLAIM_TOTAL:
        return True
    # Six+ digit dollars without cents separators are almost always ROI bleed.
    digits = _currency_digit_string(amount)
    return len(digits) >= 6 and parsed >= Decimal(100000)


def is_implausible_corroborator(value: object, line_total: object) -> bool:
    """Ignore box-28 junk that would force BOX28_OR_DI_CONFLICT vs real line-sum."""
    if is_implausible_charge_total(value):
        return True
    amount = parse_currency(value)
    total = parse_currency(line_total)
    if amount is None or total is None or total <= 0:
        return False
    # >3× or <1/3 the observed line-sum and off by >$50 → form noise / truncated cell.
    ratio_hi = total * Decimal(3)
    ratio_lo = total / Decimal(3)
    if amount > ratio_hi and abs(amount - total) > Decimal(50):
        return True
    return bool(amount < ratio_lo and abs(amount - total) > Decimal(50))


def should_defer_box28_to_line_sum(
    box28_value: object,
    service_lines: list[dict] | None,
    *,
    relative_contradiction: Decimal = Decimal("0.50"),
    min_lines: int = 2,
) -> bool:
    """Return True when box-28 should be cleared so LINE_TOTALS_RECONCILED owns E6."""
    charges = observed_line_charges(service_lines)
    if not charges:
        return False
    box = parse_currency(box28_value)
    if box is None:
        return True
    if is_suspicious_tiny_total(box) or is_implausible_charge_total(box):
        return True
    observed = sum(charges, Decimal(0))
    if observed <= 0:
        return False
    difference = abs(box - observed)
    tolerance = max(Decimal("1.00"), observed * relative_contradiction)
    # Multi-line: defer on strong contradiction (existing rule).
    if len(charges) >= min_lines:
        return difference > tolerance
    # Single observed line: defer only when box-28 is wildly off the line
    # amount (Track-B residual: box-28 ``22.00`` vs line ``305.00``). Plausible
    # near-miss box-28 (e.g. 400 vs 305) stays with OCR — do not invent.
    return difference > tolerance


def line_sum_total(service_lines: list[dict] | None) -> str | None:
    charges = observed_line_charges(service_lines)
    if not charges:
        return None
    return format_currency(sum(charges, Decimal(0)))


def is_currency_digit_drop_twin(left: object, right: object) -> bool:
    """True when one reading is a truncated digit-prefix of the other (1–2 digits)."""
    a, b = _currency_digit_string(left), _currency_digit_string(right)
    if not a or not b or a == b:
        return False
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    if not longer.startswith(shorter):
        return False
    if not (1 <= (len(longer) - len(shorter)) <= 2):
        return False
    extra = longer[len(shorter) :]
    return not (extra and set(extra) <= {"0"})


def amounts_within_tolerance(
    left: object,
    right: object,
    *,
    absolute: Decimal = Decimal("1.00"),
    relative: Decimal = Decimal("0.05"),
) -> bool:
    a, b = parse_currency(left), parse_currency(right)
    if a is None or b is None:
        return False
    diff = abs(a - b)
    target = max(abs(a), abs(b), Decimal(1))
    return diff <= max(absolute, target * relative)


def _decimal_place_conflict(left: object, right: object) -> bool:
    """True when one amount is the other shifted by exactly two decimal places.

    ``49.72`` and ``4972.00`` are the same digits with a different cents column.
    That is not a dropped leading digit and must not corroborate.
    """
    a, b = parse_currency(left), parse_currency(right)
    if a is None or b is None or a == b:
        return False
    return a * 100 == b or b * 100 == a


def is_decimal_place_shift(left: object, right: object) -> bool:
    """True when one amount is the other shifted by exactly two decimal places.

    ``49.72`` and ``4972.00`` are the same digits with a different cents column.
    That is not a dropped leading digit and must not corroborate.
    """
    return _decimal_place_conflict(left, right)


def amounts_corroborate(left: object, right: object) -> bool:
    """Box-28 / DI path only: equal, within tolerance, or digit-drop twin.

    Do NOT use for dual-engine or gpt-4o+local AUTO promotion — those paths
    require exact / $1 via ``amounts_within_tolerance(..., relative=0)``.
    Digit-drop (e.g. ``13``↔``131``) is intentional here so independently
    extracted box-28 / Azure DI can still corroborate a line-sum.
    A two-place decimal shift (``49.72`` vs ``4972.00``) is a conflict.
    """
    if _decimal_place_conflict(left, right):
        return False
    if parse_currency(left) is None or parse_currency(right) is None:
        return False
    if amounts_within_tolerance(left, right):
        return True
    return is_currency_digit_drop_twin(left, right)


def _engine_family(engine: object) -> str:
    name = str(engine or "").strip().casefold()
    # gpt-4o is a residual reader, not an independent OCR engine for LINE_TOTALS
    # dual-engine AUTO (paddle+gpt4o agreeing on the same wrong amount → FA).
    if "gpt4o" in name or "gpt-4o" in name:
        return _GPT4O_FAMILY
    if "azure" in name or "document_intelligence" in name:
        return "azure_document_intelligence_read"
    if "rapid" in name:
        return "rapidocr"
    if "paddle" in name:
        return "paddleocr"
    if "tesseract" in name:
        return "tesseract"
    return name


def candidate_independence_key(cand: object) -> tuple:
    """Explicit evidence lineage for independence checks.

    Captures producing_engine, source crop, preprocessing path, candidate value,
    confidence, parent_evidence_id, and independence_group when present.
    """
    if not isinstance(cand, dict):
        return ()
    engine = cand.get("producing_engine") or cand.get("engine") or ""
    crop = cand.get("source_crop_id") or cand.get("source_crop") or ""
    prep = cand.get("preprocessing_path") or cand.get("preprocessing_variant") or ""
    value = cand.get("value") if cand.get("value") is not None else cand.get("candidate_value")
    conf = (
        cand.get("confidence")
        if cand.get("confidence") is not None
        else cand.get("calibrated_confidence")
        if cand.get("calibrated_confidence") is not None
        else cand.get("raw_confidence")
    )
    parent = cand.get("parent_evidence_id") or ""
    group = cand.get("independence_group") or ""
    return (
        str(engine).strip(),
        str(crop).strip(),
        str(prep).strip(),
        str(value if value is not None else "").strip(),
        str(conf if conf is not None else "").strip(),
        str(parent).strip(),
        str(group).strip(),
    )


def candidates_are_independent(a: object, b: object) -> bool:
    """False when candidates share crop, parent, or independence_group lineage.

    Same source crop, same upstream OCR output lineage (parent_evidence_id),
    or the same non-empty independence_group must not count as independent
    corroboration for critical charge AUTO.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    parent_a = str(a.get("parent_evidence_id") or "").strip()
    parent_b = str(b.get("parent_evidence_id") or "").strip()
    if parent_a and parent_b and parent_a == parent_b:
        return False
    # Local derived from the gpt-4o evidence id itself.
    gpt_id = str(b.get("evidence_id") or "").strip()
    if parent_a and gpt_id and parent_a == gpt_id:
        return False
    local_id = str(a.get("evidence_id") or "").strip()
    if parent_b and local_id and parent_b == local_id:
        return False
    group_a = str(a.get("independence_group") or "").strip()
    group_b = str(b.get("independence_group") or "").strip()
    if group_a and group_b and group_a == group_b:
        return False
    crop_a = str(a.get("source_crop_id") or a.get("source_crop") or "").strip()
    crop_b = str(b.get("source_crop_id") or b.get("source_crop") or "").strip()
    return not (crop_a and crop_b and crop_a == crop_b)


def _candidate_amount(cand: dict) -> Decimal | None:
    text = cand.get("value")
    if text is None or not str(text).strip():
        return None
    parsed = parse_currency(text)
    if parsed is None or is_implausible_charge_total(parsed):
        return None
    return parsed


def _candidate_amounts_by_family(
    candidates: list[dict] | None,
    *,
    families: frozenset[str] | None = None,
) -> dict[str, Decimal]:
    """Map engine family → first currency-shaped amount.

    Uses shaped ``value`` only — never raw OCR soup (``9AA`` → 9) which poisoned
    gpt-4o+local consensus when an earlier empty paddle shell preceded a good read.
    """
    allowed = families
    by_engine: dict[str, Decimal] = {}
    for cand in candidates or []:
        if not isinstance(cand, dict):
            continue
        family = _engine_family(cand.get("engine") or cand.get("producing_engine"))
        if allowed is not None and family not in allowed:
            continue
        if family in by_engine:
            continue
        parsed = _candidate_amount(cand)
        if parsed is None:
            continue
        by_engine[family] = parsed
    return by_engine


def _shaped_candidate_amounts(candidates: list[dict] | None) -> dict[str, Decimal]:
    """Map independent engine family → first currency-shaped amount."""
    return _candidate_amounts_by_family(candidates, families=_INDEPENDENT_ENGINES)


def _first_candidate_for_family(
    candidates: list[dict] | None,
    family: str,
) -> dict | None:
    for cand in candidates or []:
        if not isinstance(cand, dict):
            continue
        if _engine_family(cand.get("engine") or cand.get("producing_engine")) != family:
            continue
        if _candidate_amount(cand) is None:
            continue
        return cand
    return None


def _selected_provenance_engine(line: dict) -> str:
    """Best-effort producing engine for the selected line charge.

    Explicit provenance wins. Fallback prefers a non-gpt-4o candidate that
    matches the selected amount so candidate order cannot falsely attribute
    an independently produced local value to gpt-4o.
    """
    for key in ("producing_engine", "engine", "selected_engine", "provenance_engine"):
        raw = line.get(key)
        if raw:
            return str(raw)
    prov = line.get("provenance")
    if isinstance(prov, dict):
        for key in ("producing_engine", "engine", "source_engine"):
            raw = prov.get(key)
            if raw:
                return str(raw)
    target = parse_currency(line.get("charges") or line.get("charge_amount"))
    if target is None:
        return ""
    target_txt = format_currency(target)
    gpt_match = ""
    for cand in line.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        parsed = _candidate_amount(cand)
        if parsed is None:
            continue
        if not _exact_or_dollar_agree(target_txt, format_currency(parsed)):
            continue
        engine = str(cand.get("producing_engine") or cand.get("engine") or "")
        if _engine_family(engine) != _GPT4O_FAMILY:
            return engine
        if not gpt_match:
            gpt_match = engine
    return gpt_match


def _exact_or_dollar_agree(left: object, right: object) -> bool:
    """Strict AUTO agreement: exact or within $1 — never digit-drop twins."""
    return amounts_within_tolerance(
        left,
        right,
        absolute=Decimal("1.00"),
        relative=Decimal(0),
    )


def line_has_dual_engine_agreement(line: dict) -> bool:
    """True when ≥2 independent engines agree on the same currency amount.

    Exact / $1 tolerance only — digit-drop twins are NOT dual-engine agreement
    for LINE_TOTALS AUTO (13↔131 class false accepts on hard-15).
    """
    amounts = _shaped_candidate_amounts(line.get("candidates") if isinstance(line, dict) else None)
    if len(amounts) < 2:
        return False
    values = list(amounts.values())
    primary = values[0]
    if not all(
        _exact_or_dollar_agree(format_currency(primary), format_currency(other))
        for other in values[1:]
    ):
        return False
    # Engines must agree with the selected charge, not only with each other
    # (20.00/20.00 must not AUTO a selected 200.00).
    target = parse_currency(line.get("charges") or line.get("charge_amount"))
    if target is None:
        return True
    return _exact_or_dollar_agree(format_currency(target), format_currency(primary))


def line_has_gpt4o_local_consensus(line: dict) -> bool:
    """gpt-4o + ≥1 independent usable local agree with the selected charge.

    Agreement uses exact / $1 only (``relative=0``) — digit-drop twins do NOT
    promote via this path. Blank shells and form-noise locals are not usable
    corroboration; missing usable locals → False (no gpt-4o-only fallback).

    Candidates that share ``parent_evidence_id``, non-empty ``independence_group``,
    or ``source_crop_id`` with the gpt-4o reading are not independent. If the
    selected value's provenance is gpt-4o family and no independent local
    agrees, fail closed.
    """
    if not isinstance(line, dict):
        return False
    if line.get("cents_unresolved"):
        return False
    target = parse_currency(line.get("charges") or line.get("charge_amount"))
    if target is None or is_implausible_charge_total(target):
        return False
    target_txt = format_currency(target)
    candidates = [c for c in (line.get("candidates") or []) if isinstance(c, dict)]

    gpt_cand = _first_candidate_for_family(candidates, _GPT4O_FAMILY)
    if gpt_cand is None:
        return False
    gpt_amt = _candidate_amount(gpt_cand)
    if gpt_amt is None:
        return False
    gpt_txt = format_currency(gpt_amt)
    if not _exact_or_dollar_agree(target_txt, gpt_txt):
        return False

    # Dollars-ruling geometry: a clipped zero or units-bleed sibling confirms
    # the vision stem. Same-crop engines are not a second evidence family;
    # the ruling split is the separate geometric check. Digit-drop twins
    # without a dollars_ruling candidate stay fail-closed (13 vs 131).
    try:
        from packages.ocr_portfolio.monetary_recognizer import (
            ruling_geometry_supports_charge,
        )

        if ruling_geometry_supports_charge(candidates, target_txt):
            return True
    except Exception:  # noqa: BLE001, S110 -- optional geometry must fail closed
        pass

    # gpt-4o-attributed selection is fine when a usable independent local
    # confirms the same amount (local confirmation path).

    usable_by_family: dict[str, list[str]] = {}

    def _pos_like(value: object) -> bool:
        try:
            from packages.geometry_authority import is_pos_like_currency

            return bool(is_pos_like_currency(value))
        except Exception:  # noqa: BLE001
            return False

    for cand in candidates:
        fam = _engine_family(cand.get("engine") or cand.get("producing_engine"))
        if fam not in _LOCAL_ENGINES:
            continue
        local_amt = _candidate_amount(cand)
        if local_amt is None:
            continue
        local_txt = format_currency(local_amt)
        # POS codes (11) and implausible tails are not votes against a vision read.
        # A POS-shaped dollar stem that is the same amount within $1
        # (34.00 beside ruled 34.25) is not a place-of-service code.
        if (
            _pos_like(local_txt)
            and not _pos_like(gpt_txt)
            and not _exact_or_dollar_agree(local_txt, gpt_txt)
            and not _exact_or_dollar_agree(local_txt, target_txt)
        ):
            continue
        if is_implausible_corroborator(local_txt, gpt_txt):
            continue
        if is_suspicious_tiny_total(local_amt) and not is_suspicious_tiny_total(gpt_amt):
            continue
        if not candidates_are_independent(cand, gpt_cand):
            continue
        usable_by_family.setdefault(fam, []).append(local_txt)

    if not usable_by_family:
        return False

    for texts in usable_by_family.values():
        if not any(
            _exact_or_dollar_agree(target_txt, text) and _exact_or_dollar_agree(gpt_txt, text)
            for text in texts
        ):
            return False
    return True


# Back-compat alias used in earlier docs/tests.
line_has_gpt4o_triple_consensus = line_has_gpt4o_local_consensus


def dual_engine_line_fraction(service_lines: list[dict] | None) -> tuple[int, int]:
    """Return (agreed_lines, observed_charge_lines)."""
    agreed = 0
    observed = 0
    for line in service_lines or []:
        if not isinstance(line, dict):
            continue
        has_charge = any(parse_currency(line.get(k)) is not None for k in _CHARGE_FIELDS)
        if not has_charge:
            continue
        observed += 1
        if line_has_dual_engine_agreement(line):
            agreed += 1
    return agreed, observed


def gpt4o_local_line_fraction(service_lines: list[dict] | None) -> tuple[int, int]:
    """Return (gpt4o+local consensus lines, observed charge lines)."""
    agreed = 0
    observed = 0
    for line in service_lines or []:
        if not isinstance(line, dict):
            continue
        has_charge = any(parse_currency(line.get(k)) is not None for k in _CHARGE_FIELDS)
        if not has_charge:
            continue
        observed += 1
        if line_has_gpt4o_local_consensus(line):
            agreed += 1
    return agreed, observed


def line_sum_auto_eligible(
    service_lines: list[dict] | None,
    *,
    box28_value: object = None,
    corroborating_values: list[object] | None = None,
) -> tuple[bool, str]:
    """Gate LINE_TOTALS E6 AUTO (fail-closed).

    Eligible when (charge tech stack):
      - every charge line has independent gpt-4o + usable local consensus
        (exact / $1; selected value not gpt-4o-only), or
      - plausible currency-shaped box-28 / DI corroborates the line sum, or
      - multi-line (≥2) and every line has exact dual-engine agreement.

    gpt-4o+local is checked *before* box-28 conflict so weak digits-first box-28
    (222 vs line 200) cannot veto a strong line consensus. Junk box-28 soup is
    ignored. Single-line paddle+rapid alone remains insufficient (hard-15 FA).
    GPT-4o never independently promotes critical charges.
    """
    total = line_sum_total(service_lines)
    if total is None:
        return False, "NO_LINE_CHARGES"

    # POS bleed: a lone POS-like amount (11.00) must never AUTO as claim total.
    try:
        from packages.geometry_authority import is_pos_like_currency

        if is_pos_like_currency(total):
            box = parse_currency(box28_value)
            extra = [
                parse_currency(v)
                for v in (corroborating_values or [])
                if parse_currency(v) is not None
            ]
            if box is None and not extra:
                return False, "POS_LIKE_LINE_SUM_REJECTED"
            # Even with corroborators, POS-like totals need non-POS corroboration.
            if box is not None and is_pos_like_currency(format_currency(box)):
                return False, "POS_LIKE_LINE_SUM_REJECTED"
    except Exception:  # noqa: BLE001, S110 -- optional geometry must fail closed
        pass

    lines = [ln for ln in (service_lines or []) if isinstance(ln, dict)]
    charge_lines = [
        ln
        for ln in lines
        if any(parse_currency(ln.get(k)) is not None for k in _CHARGE_FIELDS)
    ]

    corroborators = []
    for value in [box28_value, *(corroborating_values or [])]:
        parsed = parse_currency(value)
        if parsed is None or is_suspicious_tiny_total(parsed):
            continue
        if is_implausible_corroborator(value, total):
            continue
        corroborators.append(value)
    if corroborators:
        if any(amounts_corroborate(total, value) for value in corroborators):
            return True, "BOX28_OR_DI_CORROBORATED"
        # Plausible currency-shaped box-28 / DI disagrees → HITL, not false STP.
        return False, "BOX28_OR_DI_CONFLICT"

    # No independent Box 28 / DI total: do not AUTO from line consensus alone.
    # Printed CMS-1500 authority is Box 24F Σ ↔ Box 28; dual-engine / gpt+local
    # on the same service-line crops is not a second printed occurrence.
    gpt_agreed, gpt_observed = gpt4o_local_line_fraction(charge_lines)
    if gpt_observed >= 1 and gpt_agreed >= gpt_observed:
        return False, "GPT4O_LOCAL_NEEDS_BOX28"

    agreed, observed = dual_engine_line_fraction(service_lines)
    if observed == 0:
        return False, "NO_LINE_CHARGES"
    if observed == 1:
        if agreed >= 1:
            return False, "SINGLE_LINE_DUAL_ENGINE_NEEDS_BOX28"
        return False, "SINGLE_LINE_REQUIRES_DI"
    if agreed >= observed >= 2:
        return False, "DUAL_ENGINE_NEEDS_BOX28"
    return False, "MULTI_LINE_UNCORROBORATED"
