"""Precision-safe charge total selection (no new OCR, no threshold fitting).

Closes recurring false-accept classes:
  C1 — units-bleed cents (.07/.10/.22/.32/.43) when a .00 sibling exists
  C2 — dollars-ruling trailing digit (70.00 → 701.00) when a full-window stem exists
  C3 — place-shift / digit-soup totals (already rejected upstream)

Never invents amounts. Prefer fail-closed (keep primary) over unsafe collapse
like ``251.00 → 25.00``.
"""

from __future__ import annotations

import re

from packages.claim_evidence.line_sum_authority import (
    format_currency,
    is_decimal_place_shift,
    is_implausible_charge_total,
    line_sum_total,
    parse_currency,
)

# Cents that almost always come from Box 24G units / dashed-ruling bleed into
# the cents column on typed CMS-1500 charges (whole-dollar claims dominate).
_BLEED_CENTS = frozenset({"01", "07", "10", "22", "32", "40", "43"})
_FULL_TAGS = frozenset({"full", "line_full", "geometry", "line_geometry", "primary"})
_RULING_TAGS = frozenset({"ruling", "line_ruling"})


def _digits(text: object) -> str:
    return re.sub(r"\D", "", str(text or ""))


def _dollars_part(amount: object) -> str:
    parsed = parse_currency(amount)
    if parsed is None:
        return ""
    text = format_currency(parsed)
    return text.split(".", 1)[0]


def _cents_part(amount: object) -> str:
    parsed = parse_currency(amount)
    if parsed is None:
        return ""
    text = format_currency(parsed)
    if "." not in text:
        return ""
    return text.split(".", 1)[1]


def is_units_bleed_cents(amount: object) -> bool:
    """True when cents look like units/ruling bleed rather than typed cents."""
    cents = _cents_part(amount)
    if not cents or cents == "00":
        return False
    return cents in _BLEED_CENTS


def is_ruling_tail_extension(shorter: object, longer: object) -> bool:
    """True when ``longer`` is ``shorter`` plus one ruling-tail digit (1/4/5).

    Both amounts must be whole dollars (``.00``). The shorter stem must be at
    least two digits so ``5.00``/``51.00`` style noise does not collapse.
    """
    sa = parse_currency(shorter)
    sb = parse_currency(longer)
    if sa is None or sb is None:
        return False
    ta, tb = format_currency(sa), format_currency(sb)
    if not (ta.endswith(".00") and tb.endswith(".00")):
        return False
    a, b = _dollars_part(ta), _dollars_part(tb)
    if not a or not b or a == b or len(a) < 2 or len(b) < 2:
        return False
    if b.startswith(a) and len(b) == len(a) + 1 and b[-1] in {"1", "4", "5"}:
        return True
    return bool(a.startswith(b) and len(a) == len(b) + 1 and a[-1] in {"1", "4", "5"})


def _variant(cand: dict) -> str:
    return str(cand.get("preprocessing_variant") or "").casefold()


def _is_ruling_variant(cand: dict) -> bool:
    return "dollars_ruling" in _variant(cand)


def _is_geometry_variant(cand: dict) -> bool:
    return "geometry_cents" in _variant(cand)


def _is_derived_variant(cand: dict) -> bool:
    v = _variant(cand)
    return "derived_from_observed_line" in v or "phase2-line-sum" in v


def collect_charge_candidates(
    *,
    primary: object = None,
    field_payload: dict | None = None,
    service_lines: list[dict] | None = None,
) -> list[tuple[str, str]]:
    """Return ``(amount, source_tag)`` pairs from Box 28 + line OCR shells."""
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(raw: object, tag: str) -> None:
        parsed = parse_currency(raw)
        if parsed is None:
            return
        text = format_currency(parsed)
        if is_implausible_charge_total(text):
            return
        key = (text, tag)
        if key in seen:
            return
        seen.add(key)
        found.append((text, tag))

    if primary not in (None, ""):
        _add(primary, "primary")

    payloads: list[dict] = []
    if isinstance(field_payload, dict):
        payloads.append(field_payload)
        ocr = field_payload.get("ocr")
        if isinstance(ocr, dict):
            payloads.append(ocr)

    for payload in payloads:
        for cand in payload.get("candidates") or []:
            if not isinstance(cand, dict) or _is_derived_variant(cand):
                continue
            tag = (
                "ruling"
                if _is_ruling_variant(cand)
                else ("geometry" if _is_geometry_variant(cand) else "full")
            )
            _add(cand.get("value") or cand.get("raw_value"), tag)
        for row in (
            [payload.get("ranked_candidate")]
            if payload.get("ranked_candidate")
            else []
        ) + list(payload.get("alternatives") or []):
            if not row:
                continue
            ocr = row.get("ocr_candidate") or {}
            if _is_derived_variant(ocr):
                continue
            tag = (
                "ruling"
                if _is_ruling_variant(ocr)
                else ("geometry" if _is_geometry_variant(ocr) else "full")
            )
            _add(ocr.get("value") or ocr.get("raw_value"), tag)

    for line in service_lines or []:
        if not isinstance(line, dict):
            continue
        for key in ("charges", "charge_amount"):
            if line.get(key) not in (None, ""):
                _add(line.get(key), "line")
                break
        for cand in line.get("candidates") or []:
            if not isinstance(cand, dict) or _is_derived_variant(cand):
                continue
            tag = (
                "line_ruling"
                if _is_ruling_variant(cand)
                else (
                    "line_geometry" if _is_geometry_variant(cand) else "line_full"
                )
            )
            _add(cand.get("value") or cand.get("raw_value"), tag)

    # Multi-line Σ is the whole-dollar sibling when Box 28 only has bleed cents
    # (DJJM.028: primary 400.40, lines 200+200 → 400.00).
    summed = line_sum_total(service_lines)
    if summed is not None:
        _add(summed, "line_sum")

    return found


def prefer_safe_charge_amount(left: object, right: object) -> str | None:
    """Prefer bleed-safe amount between two readings.

    Untagged ruling-tail / digit-drop twins (``70`` vs ``701``, ``25`` vs
    ``251``) abstain — those need source tags in ``resolve_safe_charge_total``
    so we never collapse a real stem like ``251 → 25``.
    """
    a = parse_currency(left)
    b = parse_currency(right)
    if a is None and b is None:
        return None
    if a is None:
        return format_currency(b) if b is not None else None
    if b is None:
        return format_currency(a)
    ta, tb = format_currency(a), format_currency(b)
    if ta == tb:
        return ta

    # C2 without tags is ambiguous with digit-drop twins — abstain.
    if is_ruling_tail_extension(ta, tb):
        return None

    # C1: same dollar stem — prefer .00 over bleed cents.
    if _dollars_part(ta) == _dollars_part(tb):
        if ta.endswith(".00") and is_units_bleed_cents(tb):
            return ta
        if tb.endswith(".00") and is_units_bleed_cents(ta):
            return tb
        if ta.endswith(".00") and not tb.endswith(".00"):
            return ta
        if tb.endswith(".00") and not ta.endswith(".00"):
            return tb

    if is_decimal_place_shift(ta, tb):
        return None
    return None


def resolve_safe_charge_total(
    *,
    primary: object = None,
    field_payload: dict | None = None,
    service_lines: list[dict] | None = None,
) -> tuple[str | None, str]:
    """Select a precision-safe total or abstain.

    Returns ``(amount, reason)``. Only applies explicit C1/C2 repairs; never
    walks digit-drop chains that could collapse ``251 → 25``.
    """
    candidates = collect_charge_candidates(
        primary=primary,
        field_payload=field_payload,
        service_lines=service_lines,
    )
    if not candidates:
        return None, "NO_CANDIDATES"

    primary_txt = None
    if parse_currency(primary) is not None:
        primary_txt = format_currency(parse_currency(primary))

    by_tag: dict[str, list[str]] = {}
    for amount, tag in candidates:
        by_tag.setdefault(tag, []).append(amount)

    full_amounts = [
        amt
        for tag, amounts in by_tag.items()
        if tag in _FULL_TAGS
        for amt in amounts
        if amt.endswith(".00")
    ]
    ruling_amounts = [
        amt
        for tag, amounts in by_tag.items()
        if tag in _RULING_TAGS
        for amt in amounts
    ]
    all_amounts = [amt for amt, _tag in candidates]

    # C2: full-window stem beats dollars-ruling that added trailing 1/4/5.
    # Only shrink when the *longer* form is ruling-tagged (or primary matches
    # that ruling). Never treat a shorter ruling crop as license to collapse
    # a longer primary (251 → 25).
    for full in full_amounts:
        fd = _dollars_part(full)
        if len(fd) < 2:
            continue
        for ruling in ruling_amounts:
            rd = _dollars_part(ruling)
            if not (
                rd.startswith(fd)
                and len(rd) == len(fd) + 1
                and rd[-1] in {"1", "4", "5"}
            ):
                continue
            if primary_txt in {None, ruling, full} or (
                primary_txt is not None and _dollars_part(primary_txt) == rd
            ):
                return full, "RULING_TAIL_TO_FULL_STEM"
        if primary_txt is not None:
            pd = _dollars_part(primary_txt)
            if (
                pd.startswith(fd)
                and len(pd) == len(fd) + 1
                and pd[-1] in {"1", "4", "5"}
                and any(_dollars_part(r) == pd for r in ruling_amounts)
            ):
                return full, "RULING_TAIL_TO_FULL_STEM"

    # C1: bleed cents → same-dollar .00 sibling (OCR candidate or line-sum Σ).
    if primary_txt and is_units_bleed_cents(primary_txt):
        dollars = _dollars_part(primary_txt)
        sibling = f"{dollars}.00"
        if sibling in all_amounts:
            if sibling in by_tag.get("line_sum", []):
                return sibling, "BLEED_CENTS_TO_LINE_SUM"
            return sibling, "BLEED_CENTS_TO_WHOLE_DOLLAR"

    # Bleed line/geometry amount with .00 sibling when primary already .00.
    for amount in all_amounts:
        if not is_units_bleed_cents(amount):
            continue
        sibling = f"{_dollars_part(amount)}.00"
        if sibling in all_amounts and primary_txt in {amount, sibling, None}:
            if primary_txt == sibling:
                return sibling, "PRIMARY_UNCHANGED"
            if primary_txt == amount:
                return sibling, "BLEED_CENTS_TO_WHOLE_DOLLAR"

    if primary_txt:
        return primary_txt, "PRIMARY_UNCHANGED"
    # No primary: prefer a whole-dollar full read when unambiguous.
    if len(set(full_amounts)) == 1:
        return full_amounts[0], "SAFE_CHARGE_FROM_CANDIDATES"
    return None, "NO_SAFE_PRIMARY"
