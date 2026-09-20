"""Box 28 ↔ Box 24F line-sum corroboration for total_charge AUTO.

Fail-closed authority. AUTO only when both printed occurrences independently
pass parser integrity, agree on amount, and use non-overlapping semantic ROIs.

Never concatenates overlapping OCR windows, engine alternatives, neighbouring
cells, or units columns. Digit identity still comes from existing readers;
this module only adjudicates integrity and independence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from packages.claim_evidence.line_sum_authority import (
    format_currency,
    is_decimal_place_shift,
    is_implausible_charge_total,
    line_sum_total,
    parse_currency,
)
from packages.geometry_authority.cms1500_regions import (
    CMS1500_BOX28,
    CMS1500_CHARGE_CENTS_X,
    CMS1500_LINE_COLUMNS,
)

_CHARGE_KEYS = ("charges", "charge_amount", "total_charge", "total_charges")


@dataclass(frozen=True)
class ParserIntegrityResult:
    passed: bool
    amount: str | None
    raw_digit_sequence: str
    glyph_count: int
    mapped_digit_count: int
    reasons: tuple[str, ...]
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "amount": self.amount,
            "raw_digit_sequence": self.raw_digit_sequence,
            "glyph_count": self.glyph_count,
            "mapped_digit_count": self.mapped_digit_count,
            "reasons": list(self.reasons),
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class Box28Evidence:
    raw_tokens: tuple[str, ...]
    canonical_glyph_polygons: tuple[tuple[tuple[float, float], ...], ...]
    canonical_glyph_centres: tuple[tuple[float, float], ...]
    dollar_glyphs: tuple[str, ...]
    cents_glyphs: tuple[str, ...]
    unit_zone_glyphs: tuple[str, ...]
    normalized_amount: str | None
    region: tuple[float, float, float, float] | None
    integrity: ParserIntegrityResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_tokens": list(self.raw_tokens),
            "canonical_glyph_polygons": [
                [list(pt) for pt in poly] for poly in self.canonical_glyph_polygons
            ],
            "canonical_glyph_centres": [list(c) for c in self.canonical_glyph_centres],
            "dollar_glyphs": list(self.dollar_glyphs),
            "cents_glyphs": list(self.cents_glyphs),
            "unit_zone_glyphs": list(self.unit_zone_glyphs),
            "normalized_amount": self.normalized_amount,
            "region": list(self.region) if self.region else None,
            "integrity": self.integrity.to_dict(),
        }


@dataclass(frozen=True)
class Box24FRowEvidence:
    line_number: int
    raw_tokens: tuple[str, ...]
    amount: str | None
    canonical_glyph_polygons: tuple[tuple[tuple[float, float], ...], ...]
    canonical_glyph_centres: tuple[tuple[float, float], ...]
    region: tuple[float, float, float, float] | None
    integrity: ParserIntegrityResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_number": self.line_number,
            "raw_tokens": list(self.raw_tokens),
            "amount": self.amount,
            "canonical_glyph_polygons": [
                [list(pt) for pt in poly] for poly in self.canonical_glyph_polygons
            ],
            "canonical_glyph_centres": [list(c) for c in self.canonical_glyph_centres],
            "region": list(self.region) if self.region else None,
            "integrity": self.integrity.to_dict(),
        }


@dataclass(frozen=True)
class PredicateTrace:
    name: str
    value: bool | str | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "detail": self.detail}


@dataclass
class Box28LineSumDecision:
    disposition: str  # AUTO_ACCEPTED | HUMAN_REVIEW_REQUIRED
    authority_reason: str
    amount: str | None
    box28: Box28Evidence | None = None
    box24f_rows: tuple[Box24FRowEvidence, ...] = ()
    line_sum_amount: str | None = None
    line_sum_integrity: ParserIntegrityResult | None = None
    regions_independent: bool | None = None
    predicates: list[PredicateTrace] = field(default_factory=list)
    failed_predicate: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "authority_reason": self.authority_reason,
            "amount": self.amount,
            "box28": self.box28.to_dict() if self.box28 else None,
            "box24f_rows": [row.to_dict() for row in self.box24f_rows],
            "line_sum_amount": self.line_sum_amount,
            "line_sum_integrity": (
                self.line_sum_integrity.to_dict() if self.line_sum_integrity else None
            ),
            "regions_independent": self.regions_independent,
            "predicates": [p.to_dict() for p in self.predicates],
            "failed_predicate": self.failed_predicate,
        }


def _digits_only(text: object) -> str:
    return re.sub(r"\D", "", str(text or ""))


def _region_tuple(raw: object) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        try:
            return (
                float(raw["x0"]),
                float(raw["y0"]),
                float(raw["x1"]),
                float(raw["y1"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        except (TypeError, ValueError):
            return None
    return None


def _region_is_cents_clipped(
    region: tuple[float, float, float, float] | None,
) -> bool:
    """True when the crop ends before the printed cents column."""
    if region is None:
        return False
    return float(region[2]) < CMS1500_CHARGE_CENTS_X + 12


def _overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _area(r: tuple[float, float, float, float]) -> float:
    return max(0.0, (r[2] - r[0]) * (r[3] - r[1]))


def _default_box24f_region(line_number: int) -> tuple[float, float, float, float]:
    """Semantic Box 24F ROI for a service-line row (reference CMS-1500)."""
    ch0, ch1 = CMS1500_LINE_COLUMNS["charges"]
    # Service-line strip starts ~y=1458 with ~55px row pitch on the template.
    y0 = 1458.0 + max(0, line_number - 1) * 55.0
    y1 = y0 + 55.0
    return (float(ch0), y0, float(ch1), y1)


def evaluate_parser_integrity(
    *,
    amount: object,
    raw_digit_sequence: object = "",
    dollar_glyphs: list[str] | tuple[str, ...] | None = None,
    cents_glyphs: list[str] | tuple[str, ...] | None = None,
    unit_zone_glyphs: list[str] | tuple[str, ...] | None = None,
    canonical_glyph_centres: list | tuple | None = None,
    require_cents: bool = True,
    allow_leading_contamination_drop: bool = False,
) -> ParserIntegrityResult:
    """One-to-one digit↔glyph integrity. Reject soup, shifts, and unmapped digits."""
    reasons: list[str] = []
    parsed = parse_currency(amount)
    if parsed is None:
        return ParserIntegrityResult(
            False,
            None,
            _digits_only(raw_digit_sequence or amount),
            0,
            0,
            ("AMOUNT_UNPARSED",),
            "AMOUNT_UNPARSED",
        )
    if is_implausible_charge_total(amount):
        return ParserIntegrityResult(
            False,
            format_currency(parsed),
            _digits_only(raw_digit_sequence or amount),
            0,
            0,
            ("IMPLAUSIBLE_TOTAL",),
            "IMPLAUSIBLE_TOTAL",
        )
    normalized = format_currency(parsed)
    amount_digits = _digits_only(normalized)
    raw_digits = _digits_only(raw_digit_sequence) or amount_digits
    dollars = tuple(str(g) for g in (dollar_glyphs or ()) if str(g).isdigit())
    cents = tuple(str(g) for g in (cents_glyphs or ()) if str(g).isdigit())
    units = tuple(str(g) for g in (unit_zone_glyphs or ()) if str(g).isdigit())
    mapped = dollars + cents
    glyph_count = len(mapped)
    centres = list(canonical_glyph_centres or [])

    # Broken glyph splits (units bleed, 3+ cents, or glyphs that disagree with
    # the normalised amount) fall through to the token path.
    glyph_split_usable = bool(mapped) and len(cents) == 2 and not units
    if units:
        reasons.append("UNITS_ZONE_PRESENT")
    if mapped and glyph_split_usable:
        mapped_digits = "".join(mapped)
        if (
            mapped_digits != amount_digits
            and mapped_digits != raw_digits
            and "".join(dollars) + "".join(cents) != amount_digits
        ):
            reasons.append("GLYPH_AMOUNT_MISMATCH_FALLBACK_TOKEN")
            glyph_split_usable = False
    if mapped and not glyph_split_usable:
        reasons.append("GLYPH_SPLIT_UNUSABLE")
        mapped = ()
        glyph_count = 0
        dollars = ()
        cents = ()

    if require_cents and mapped and len(cents) != 2:
        return ParserIntegrityResult(
            False,
            normalized,
            raw_digits,
            glyph_count,
            len(mapped),
            tuple(reasons) + ("CENTS_GLYPH_COUNT",),
            "CENTS_GLYPH_COUNT",
        )
    if mapped:
        mapped_digits = "".join(mapped)
        if (
            mapped_digits != amount_digits
            and mapped_digits != raw_digits
            and "".join(dollars) + "".join(cents) != amount_digits
        ):
            return ParserIntegrityResult(
                False,
                normalized,
                raw_digits,
                glyph_count,
                len(mapped),
                tuple(reasons) + ("GLYPH_AMOUNT_MISMATCH",),
                "GLYPH_AMOUNT_MISMATCH",
            )
        if len(mapped) != len(amount_digits):
            return ParserIntegrityResult(
                False,
                normalized,
                raw_digits,
                glyph_count,
                len(mapped),
                tuple(reasons) + ("DIGIT_GLYPH_COUNT_MISMATCH",),
                "DIGIT_GLYPH_COUNT_MISMATCH",
            )
        if centres and len(centres) < len(mapped):
            return ParserIntegrityResult(
                False,
                normalized,
                raw_digits,
                glyph_count,
                len(mapped),
                tuple(reasons) + ("UNMAPPED_GLYPH_PROVENANCE",),
                "UNMAPPED_GLYPH_PROVENANCE",
            )
        # Duplicate centre positions imply duplicated OCR windows / glyphs.
        if centres:
            rounded = {(round(float(c[0]), 1), round(float(c[1]), 1)) for c in centres}
            if len(rounded) < len(mapped):
                return ParserIntegrityResult(
                    False,
                    normalized,
                    raw_digits,
                    glyph_count,
                    len(mapped),
                    tuple(reasons) + ("DUPLICATED_GLYPH_PROVENANCE",),
                    "DUPLICATED_GLYPH_PROVENANCE",
                )
        reasons.append("GLYPH_ONE_TO_ONE")
    else:
        # Token-only path (visible decimal / unusable glyph split).
        if raw_digits != amount_digits:
            if (
                len(raw_digits) == len(amount_digits) + 1
                and "1" in raw_digits
                and raw_digits.replace("1", "", 1) == amount_digits
            ):
                reasons.append("RULING_TICK_DROPPED")
            elif (
                allow_leading_contamination_drop
                and len(raw_digits) == len(amount_digits) + 1
                and raw_digits[1:] == amount_digits
            ):
                # Single leading form-noise digit (box label / ruling bleed).
                reasons.append("LEADING_CONTAMINATION_DROPPED")
            else:
                return ParserIntegrityResult(
                    False,
                    normalized,
                    raw_digits,
                    0,
                    0,
                    tuple(reasons) + ("RAW_DIGIT_COUNT_MISMATCH",),
                    "RAW_DIGIT_COUNT_MISMATCH",
                )
        if len(amount_digits) >= 6:
            return ParserIntegrityResult(
                False,
                normalized,
                raw_digits,
                0,
                0,
                tuple(reasons) + ("DIGIT_SOUP",),
                "DIGIT_SOUP",
            )
        reasons.append("TOKEN_DIGIT_COUNT_OK")

    reasons.append("PARSER_INTEGRITY_PASS")
    return ParserIntegrityResult(
        True,
        normalized,
        raw_digits,
        glyph_count,
        len(mapped) if mapped else len(amount_digits),
        tuple(reasons),
        None,
    )


def _geometry_observation(payload: dict | None) -> dict:
    if not payload:
        return {}
    for attempt in payload.get("attempts") or []:
        reason = str(attempt.get("reason") or "")
        if "GEOMETRY_CENTS" in reason and "UNDERREAD" not in reason:
            if "CENTS_CLIPPED" in reason or "BOX24F_CENTS_CLIPPED" in reason:
                continue
            obs = attempt.get("observation")
            if isinstance(obs, dict):
                # Reject adopted=False / empty under-reads even if reason missed.
                if obs.get("adopted") is False:
                    continue
                return obs
    for cand in payload.get("candidates") or []:
        if str(cand.get("preprocessing_variant") or "") == "GEOMETRY_CENTS":
            bbox = cand.get("bounding_box")
            region = _region_tuple(bbox)
            if _region_is_cents_clipped(region):
                continue
            prov = cand.get("provenance")
            if isinstance(prov, dict) and prov:
                return prov
            return {
                "raw_digit_sequence": cand.get("raw_value") or "",
                "canonical_monetary_value": cand.get("value"),
                "text": cand.get("raw_value") or "",
            }
    return {}


def _candidate_amounts(payload: dict | None) -> list[str]:
    """Currency-shaped amounts from OCR candidates (no window concatenation)."""
    found: list[str] = []
    seen: set[str] = set()
    for cand in (payload or {}).get("candidates") or []:
        variant = str(cand.get("preprocessing_variant") or "").casefold()
        if "derived_from_observed_line" in variant or "phase2-line-sum" in variant:
            continue
        bbox = cand.get("bounding_box")
        # Skip cents-clipped GEOMETRY shells — those are not authoritative.
        if (
            str(cand.get("preprocessing_variant") or "") == "GEOMETRY_CENTS"
            and _region_is_cents_clipped(_region_tuple(bbox))
        ):
            continue
        for key in ("value", "raw_value"):
            parsed = parse_currency(cand.get(key))
            if parsed is None:
                continue
            text = format_currency(parsed)
            if text in seen or is_implausible_charge_total(text):
                continue
            seen.add(text)
            found.append(text)
            break
    return found


def _box28_alternate_amounts(
    *,
    primary: object,
    raw_tokens: tuple[str, ...],
) -> list[str]:
    """Alternate Box 28 parses from the same raw token set (no new OCR)."""
    alts: list[str] = []
    seen: set[str] = set()

    def _add(raw: object) -> None:
        parsed = parse_currency(raw)
        if parsed is None:
            return
        text = format_currency(parsed)
        if text in seen or is_implausible_charge_total(text):
            return
        seen.add(text)
        alts.append(text)

    _add(primary)
    for token in raw_tokens:
        digits = _digits_only(token)
        if not digits:
            continue
        _add(token)
        # Implied-decimal forms of the raw digit run.
        if len(digits) >= 3:
            _add(f"{digits[:-2]}.{digits[-2:]}")
        # Single leading contamination digit (form label / ruling bleed).
        if len(digits) >= 4:
            rest = digits[1:]
            _add(f"{rest[:-2]}.{rest[-2:]}")
    return alts


def build_box28_evidence(
    *,
    amount: object,
    field_payload: dict | None = None,
    region: object = None,
    observation: dict | None = None,
    allow_leading_contamination_drop: bool = False,
) -> Box28Evidence:
    obs = observation or _geometry_observation(field_payload)
    # Under-read / non-adopted geometry must not override a better claim amount.
    if obs.get("adopted") is False:
        obs = {}
    raw_tokens: list[str] = []
    if obs.get("text"):
        raw_tokens.append(str(obs["text"]))
    if obs.get("raw_digit_sequence"):
        raw_tokens.append(str(obs["raw_digit_sequence"]))
    for cand in (field_payload or {}).get("candidates") or []:
        raw = str(cand.get("raw_value") or cand.get("value") or "").strip()
        if raw:
            raw_tokens.append(raw)
    # Under-read geometry still contributes raw digit evidence for alternate parses.
    for attempt in (field_payload or {}).get("ocr", {}).get("attempts") or []:
        reason = str(attempt.get("reason") or "")
        if "GEOMETRY_CENTS" not in reason:
            continue
        attempt_obs = attempt.get("observation")
        if not isinstance(attempt_obs, dict):
            continue
        if attempt_obs.get("raw_digit_sequence"):
            raw_tokens.append(str(attempt_obs["raw_digit_sequence"]))
        if attempt_obs.get("text"):
            raw_tokens.append(str(attempt_obs["text"]))
    # field_payload may be the extraction field_results row (ocr nested).
    ocr_block = (field_payload or {}).get("ocr")
    if isinstance(ocr_block, dict):
        for attempt in ocr_block.get("attempts") or []:
            reason = str(attempt.get("reason") or "")
            if "GEOMETRY_CENTS" not in reason:
                continue
            attempt_obs = attempt.get("observation")
            if not isinstance(attempt_obs, dict):
                continue
            if attempt_obs.get("raw_digit_sequence"):
                raw_tokens.append(str(attempt_obs["raw_digit_sequence"]))
            if attempt_obs.get("text"):
                raw_tokens.append(str(attempt_obs["text"]))
        for cand in ocr_block.get("candidates") or []:
            raw = str(cand.get("raw_value") or cand.get("value") or "").strip()
            if raw:
                raw_tokens.append(raw)
    if amount not in (None, ""):
        raw_tokens.append(str(amount))
    polygons = tuple(
        tuple(tuple(float(x) for x in pt) for pt in poly)
        for poly in (obs.get("page_glyph_polygons") or [])
        if isinstance(poly, (list, tuple))
    )
    centres = tuple(
        (float(c[0]), float(c[1]))
        for c in (obs.get("canonical_glyph_centres") or [])
        if isinstance(c, (list, tuple)) and len(c) >= 2
    )
    dollars = tuple(str(g) for g in (obs.get("dollar_glyphs") or []))
    cents = tuple(str(g) for g in (obs.get("cents_glyphs") or []))
    units = tuple(str(g) for g in (obs.get("unit_zone_glyphs") or []))
    # Prefer explicit amount when geometry glyphs are unusable / under-read.
    geo_amount = obs.get("canonical_monetary_value")
    if geo_amount and dollars and len([c for c in cents if str(c).isdigit()]) == 2 and not units:
        normalized: object = geo_amount
    else:
        normalized = amount or geo_amount
    raw_for_integrity = obs.get("raw_digit_sequence") or ""
    if not raw_for_integrity:
        for token in raw_tokens:
            if _digits_only(token):
                raw_for_integrity = token
                break
    integrity = evaluate_parser_integrity(
        amount=normalized,
        raw_digit_sequence=raw_for_integrity,
        dollar_glyphs=dollars,
        cents_glyphs=cents,
        unit_zone_glyphs=units,
        canonical_glyph_centres=centres,
        allow_leading_contamination_drop=allow_leading_contamination_drop,
    )
    resolved_region = (
        _region_tuple(region)
        or _region_tuple((field_payload or {}).get("canonical_region"))
        or _region_tuple((field_payload or {}).get("ocr_region"))
        or CMS1500_BOX28
    )
    return Box28Evidence(
        raw_tokens=tuple(dict.fromkeys(raw_tokens)),
        canonical_glyph_polygons=polygons,
        canonical_glyph_centres=centres,
        dollar_glyphs=dollars,
        cents_glyphs=cents,
        unit_zone_glyphs=units,
        normalized_amount=integrity.amount,
        region=resolved_region,
        integrity=integrity,
    )


def build_box24f_rows(service_lines: list[dict] | None) -> tuple[Box24FRowEvidence, ...]:
    rows: list[Box24FRowEvidence] = []
    for index, line in enumerate(service_lines or []):
        if not isinstance(line, dict):
            continue
        selected = None
        for key in _CHARGE_KEYS:
            if line.get(key) not in (None, ""):
                selected = line.get(key)
                break
        region = _region_tuple(line.get("canonical_region") or line.get("ocr_region"))
        clipped = _region_is_cents_clipped(region)
        obs = {} if clipped else _geometry_observation(line)
        # Prefer non-clipped geometry, else dual-engine / candidate consensus,
        # else the selected shell — never a cents-clipped geometry amount.
        amount: object = None
        if obs.get("canonical_monetary_value") and obs.get("adopted") is not False:
            amount = obs.get("canonical_monetary_value")
        if amount is None:
            cand_amounts = _candidate_amounts(line)
            if clipped and selected is not None:
                # Drop the clipped selected shell when candidates agree elsewhere.
                selected_fmt = (
                    format_currency(parse_currency(selected))
                    if parse_currency(selected) is not None
                    else None
                )
                others = [a for a in cand_amounts if a != selected_fmt]
                if others:
                    # Prefer the mode among non-clipped candidate amounts.
                    amount = max(set(others), key=others.count)
                elif selected_fmt and not clipped:
                    amount = selected_fmt
                elif others:
                    amount = others[0]
                elif cand_amounts:
                    amount = cand_amounts[0]
                else:
                    amount = selected
            elif cand_amounts:
                if selected is not None and parse_currency(selected) is not None:
                    selected_fmt = format_currency(parse_currency(selected))
                    if selected_fmt in cand_amounts:
                        amount = selected_fmt
                    else:
                        amount = max(set(cand_amounts), key=cand_amounts.count)
                else:
                    amount = max(set(cand_amounts), key=cand_amounts.count)
            else:
                amount = selected
        if amount is None:
            continue
        polygons = tuple(
            tuple(tuple(float(x) for x in pt) for pt in poly)
            for poly in (obs.get("page_glyph_polygons") or [])
            if isinstance(poly, (list, tuple))
        )
        centres = tuple(
            (float(c[0]), float(c[1]))
            for c in (obs.get("canonical_glyph_centres") or [])
            if isinstance(c, (list, tuple)) and len(c) >= 2
        )
        dollars = tuple(str(g) for g in (obs.get("dollar_glyphs") or []))
        cents = tuple(str(g) for g in (obs.get("cents_glyphs") or []))
        units = tuple(str(g) for g in (obs.get("unit_zone_glyphs") or []))
        # When using a non-geometry candidate amount, do not attach mismatched glyphs.
        geo_amt = obs.get("canonical_monetary_value")
        if (
            geo_amt
            and parse_currency(geo_amt) is not None
            and parse_currency(amount) is not None
            and parse_currency(geo_amt) != parse_currency(amount)
        ):
            polygons = ()
            centres = ()
            dollars = ()
            cents = ()
            units = ()
        integrity = evaluate_parser_integrity(
            amount=amount,
            raw_digit_sequence=(
                (obs.get("raw_digit_sequence") if obs else None)
                or (
                    _digits_only(amount)
                    if (
                        clipped
                        or (
                            parse_currency(selected) is not None
                            and parse_currency(amount) is not None
                            and parse_currency(selected) != parse_currency(amount)
                        )
                    )
                    else None
                )
                or line.get("raw_charges")
                or _digits_only(amount)
            ),
            dollar_glyphs=dollars,
            cents_glyphs=cents,
            unit_zone_glyphs=units,
            canonical_glyph_centres=centres,
        )
        raw_tokens = []
        if line.get("raw_charges"):
            raw_tokens.append(str(line.get("raw_charges")))
        if obs.get("raw_digit_sequence"):
            raw_tokens.append(str(obs["raw_digit_sequence"]))
        raw_tokens.extend(_candidate_amounts(line))
        rows.append(
            Box24FRowEvidence(
                line_number=int(line.get("line_number") or index + 1),
                raw_tokens=tuple(dict.fromkeys(raw_tokens)),
                amount=integrity.amount,
                canonical_glyph_polygons=polygons,
                canonical_glyph_centres=centres,
                region=region or _default_box24f_region(int(line.get("line_number") or index + 1)),
                integrity=integrity,
            )
        )
    return tuple(rows)


def regions_are_independent(
    box28_region: tuple[float, float, float, float] | None,
    line_regions: list[tuple[float, float, float, float] | None],
    *,
    max_overlap_fraction: float = 0.05,
) -> bool:
    """Box 28 and Box 24F must not share the same crop / duplicated ROI."""
    if box28_region is None:
        return False
    if not line_regions or any(region is None for region in line_regions):
        return False
    box_area = _area(box28_region)
    if box_area <= 0:
        return False
    for region in line_regions:
        assert region is not None
        overlap = _overlap_area(box28_region, region)
        if overlap <= 0:
            # Vertically separated is the CMS-1500 layout (24F above 28).
            continue
        smaller = min(box_area, _area(region))
        if smaller > 0 and (overlap / smaller) > max_overlap_fraction:
            return False
        # Same horizontal band → likely duplicated crop / shared window.
        if abs((box28_region[1] + box28_region[3]) / 2.0 - (region[1] + region[3]) / 2.0) < 40:
            return False
    return True


def _line_sum_from_rows(
    rows: tuple[Box24FRowEvidence, ...],
) -> tuple[str | None, Decimal | None]:
    if not rows:
        return None, None
    total = Decimal(0)
    for row in rows:
        parsed = parse_currency(row.amount)
        if parsed is None:
            return None, None
        total += parsed
    return format_currency(total), total


def evaluate_box28_line_sum_authority(
    *,
    box28_amount: object,
    service_lines: list[dict] | None,
    box28_field_payload: dict | None = None,
    box28_region: object = None,
    box28_observation: dict | None = None,
) -> Box28LineSumDecision:
    """Adjudicate total_charge AUTO from independent Box 28 ↔ line-sum evidence."""
    predicates: list[PredicateTrace] = []
    rows = build_box24f_rows(service_lines)
    line_sum_amount, line_total = _line_sum_from_rows(rows)
    # Fall back to observed line-sum helper when rows could not be built.
    if line_sum_amount is None:
        fallback = line_sum_total(service_lines)
        if fallback is not None:
            line_sum_amount = str(fallback)
            line_total = parse_currency(fallback)

    if not rows:
        line_integrity = ParserIntegrityResult(
            False, None, "", 0, 0, ("NO_LINE_CHARGES",), "NO_LINE_CHARGES"
        )
    elif not all(row.integrity.passed for row in rows):
        failed = next(row for row in rows if not row.integrity.passed)
        line_integrity = ParserIntegrityResult(
            False,
            line_sum_amount,
            "".join(row.integrity.raw_digit_sequence for row in rows),
            sum(row.integrity.glyph_count for row in rows),
            sum(row.integrity.mapped_digit_count for row in rows),
            ("ROW_INTEGRITY_FAILED", failed.integrity.rejection_reason or ""),
            "LINE_SUM_INTEGRITY_FAILED",
        )
    elif line_sum_amount is None or line_total is None:
        line_integrity = ParserIntegrityResult(
            False, None, "", 0, 0, ("LINE_SUM_UNPARSED",), "LINE_SUM_UNPARSED"
        )
    else:
        row_sum = sum(
            (parse_currency(row.amount) or Decimal(0) for row in rows), Decimal(0)
        )
        if abs(row_sum - line_total) > Decimal("0.01"):
            line_integrity = ParserIntegrityResult(
                False,
                line_sum_amount,
                "".join(row.integrity.raw_digit_sequence for row in rows),
                sum(row.integrity.glyph_count for row in rows),
                sum(row.integrity.mapped_digit_count for row in rows),
                ("ROW_SUM_MISMATCH",),
                "LINE_SUM_INTEGRITY_FAILED",
            )
        else:
            line_integrity = ParserIntegrityResult(
                True,
                line_sum_amount,
                "".join(row.integrity.raw_digit_sequence for row in rows),
                sum(row.integrity.glyph_count for row in rows),
                sum(row.integrity.mapped_digit_count for row in rows),
                ("LINE_SUM_INTEGRITY_PASS",),
                None,
            )

    box28 = build_box28_evidence(
        amount=box28_amount,
        field_payload=box28_field_payload,
        region=box28_region,
        observation=box28_observation,
    )

    # When primary Box 28 fails to match an integrity-passing line sum, try
    # alternate parses of the *same* Box 28 raw tokens (leading contamination /
    # implied decimal) — never borrow line glyphs into Box 28.
    if (
        line_integrity.passed
        and line_sum_amount
        and (
            not box28.integrity.passed
            or box28.normalized_amount is None
            or parse_currency(box28.normalized_amount) != parse_currency(line_sum_amount)
            or is_decimal_place_shift(box28.normalized_amount, line_sum_amount)
        )
    ):
        for alt in _box28_alternate_amounts(
            primary=box28_amount, raw_tokens=box28.raw_tokens
        ):
            if parse_currency(alt) != parse_currency(line_sum_amount):
                continue
            if is_decimal_place_shift(alt, line_sum_amount):
                continue
            alt_digits = _digits_only(alt)
            # Prefer the raw token whose digit run is a 0–1 leading-digit
            # extension of the alternate amount (same Box 28 OCR string).
            raw_for_alt = alt_digits
            for token in box28.raw_tokens:
                digits = _digits_only(token)
                if not digits:
                    continue
                if digits == alt_digits or (
                    len(digits) == len(alt_digits) + 1 and digits[1:] == alt_digits
                ):
                    raw_for_alt = digits
                    break
            alt_integrity = evaluate_parser_integrity(
                amount=alt,
                raw_digit_sequence=raw_for_alt,
                allow_leading_contamination_drop=True,
            )
            if not alt_integrity.passed:
                continue
            box28 = Box28Evidence(
                raw_tokens=box28.raw_tokens,
                canonical_glyph_polygons=(),
                canonical_glyph_centres=(),
                dollar_glyphs=(),
                cents_glyphs=(),
                unit_zone_glyphs=(),
                normalized_amount=alt_integrity.amount,
                region=box28.region,
                integrity=alt_integrity,
            )
            break

    independent = regions_are_independent(
        box28.region, [row.region for row in rows]
    )

    predicates.append(
        PredicateTrace(
            "box28.parser_integrity",
            box28.integrity.passed,
            box28.integrity.rejection_reason or ",".join(box28.integrity.reasons),
        )
    )
    predicates.append(
        PredicateTrace(
            "line_sum.parser_integrity",
            line_integrity.passed,
            line_integrity.rejection_reason or ",".join(line_integrity.reasons),
        )
    )
    amounts_equal = bool(
        box28.integrity.passed
        and line_integrity.passed
        and box28.normalized_amount
        and line_sum_amount
        and parse_currency(box28.normalized_amount) == parse_currency(line_sum_amount)
        and not is_decimal_place_shift(box28.normalized_amount, line_sum_amount)
    )
    predicates.append(
        PredicateTrace(
            "box28.amount == line_sum.amount",
            amounts_equal,
            f"box28={box28.normalized_amount} line_sum={line_sum_amount}",
        )
    )
    predicates.append(
        PredicateTrace(
            "evidence_regions_are_independent",
            independent,
            f"box28={box28.region} lines={[row.region for row in rows]}",
        )
    )

    # Persist predicate evaluation for Field Value Authority / Financial Reconciliation.
    for pred in predicates:
        print(
            f"[BOX28_LINE_SUM] predicate {pred.name}={pred.value} ({pred.detail})",
            flush=True,
        )

    if not box28.integrity.passed:
        return Box28LineSumDecision(
            disposition="HUMAN_REVIEW_REQUIRED",
            authority_reason="BOX28_INTEGRITY_FAILED",
            amount=None,
            box28=box28,
            box24f_rows=rows,
            line_sum_amount=line_sum_amount,
            line_sum_integrity=line_integrity,
            regions_independent=independent,
            predicates=predicates,
            failed_predicate="box28.parser_integrity",
        )
    if not line_integrity.passed:
        return Box28LineSumDecision(
            disposition="HUMAN_REVIEW_REQUIRED",
            authority_reason="LINE_SUM_INTEGRITY_FAILED",
            amount=None,
            box28=box28,
            box24f_rows=rows,
            line_sum_amount=line_sum_amount,
            line_sum_integrity=line_integrity,
            regions_independent=independent,
            predicates=predicates,
            failed_predicate="line_sum.parser_integrity",
        )
    if not amounts_equal:
        return Box28LineSumDecision(
            disposition="HUMAN_REVIEW_REQUIRED",
            authority_reason="AMOUNT_MISMATCH",
            amount=None,
            box28=box28,
            box24f_rows=rows,
            line_sum_amount=line_sum_amount,
            line_sum_integrity=line_integrity,
            regions_independent=independent,
            predicates=predicates,
            failed_predicate="box28.amount == line_sum.amount",
        )
    if not independent:
        return Box28LineSumDecision(
            disposition="HUMAN_REVIEW_REQUIRED",
            authority_reason="MISSING_INDEPENDENT_EVIDENCE",
            amount=None,
            box28=box28,
            box24f_rows=rows,
            line_sum_amount=line_sum_amount,
            line_sum_integrity=line_integrity,
            regions_independent=independent,
            predicates=predicates,
            failed_predicate="evidence_regions_are_independent",
        )

    predicates.append(
        PredicateTrace("authority_rule", True, "BOX28_LINE_SUM_CORROBORATED")
    )
    print(
        "[BOX28_LINE_SUM] predicate authority_rule=True (BOX28_LINE_SUM_CORROBORATED)",
        flush=True,
    )
    return Box28LineSumDecision(
        disposition="AUTO_ACCEPTED",
        authority_reason="BOX28_LINE_SUM_CORROBORATED",
        amount=box28.normalized_amount,
        box28=box28,
        box24f_rows=rows,
        line_sum_amount=line_sum_amount,
        line_sum_integrity=line_integrity,
        regions_independent=True,
        predicates=predicates,
        failed_predicate=None,
    )
