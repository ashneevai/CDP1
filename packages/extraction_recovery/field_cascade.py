"""Field-semantic cascade OCR for CMS-1500 operational STP.

Replaces bolted per-field crop retries with one ordered plan:

  crop variants → route engines → span selection → semantic accept

Principles
----------
1. Acceptance is field-shaped (date / currency / name / id), not "any text".
2. Engine order comes from governed ``ocr_field_routes.yaml`` (primary then
   confirmation), not a global RapidOCR-first default.
3. Crop variants are typed (digit-band DOB, NPI-cleared charge, etc.) and
   exhausted only until a semantic accept fires.
4. Empty / contaminated financial crops stay empty — cascade never invents
   amounts. Claim-total E6 remains crop-total ∩ Σ line charges.
5. Strategy id, crop ladders, and post-miss stages come from
   ``config/field_cascade_strategy.yaml`` (field-cascade-v12).
6. Dual-engine confirmation (primary + confirmation OBSERVED) before
   short-circuit; among engine candidates prefer multi-engine agreement
   after span-select, else first field-shaped value in route order.
7. Name/ID value-band crops run before full-cell primary; label-contaminated
   spans are never NAME_SHAPED / ID_SHAPED.
8. Tool stack: paddle+rapid primary, tesseract fill, gpt-4o residual for
   names/charges; Azure DI charge and PP-OCRv5 Server off by default.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .span_selection import select_field_span, span_datatype_for_field
from .strategy import crop_ladder_for, load_cascade_strategy

_ROUTE_PATH = Path(__file__).resolve().parents[2] / "config" / "ocr_field_routes.yaml"

# Map governed route names onto packages.ocr_router.ENGINE_ORDER ids.
_ENGINE_ALIASES = {
    "paddle": "paddleocr",
    "paddleocr": "paddleocr",
    "rapid": "rapidocr",
    "rapidocr": "rapidocr",
    "tesseract": "tesseract",
    "trocr": "trocr",
}

_DATE_SHAPE = re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$")
_CURRENCY_SHAPE = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+\.\d{2}$")
_NAME_TOKEN = re.compile(r"[A-Z][A-Z'\-]{1,30}")
_ID_SHAPE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{4,20}$")


@dataclass(frozen=True)
class CropVariant:
    variant_id: str
    bbox: tuple[int, int, int, int]
    reason: str


@dataclass(frozen=True)
class CascadeStepResult:
    variant_id: str
    bbox: tuple[int, int, int, int]
    engines: tuple[str, ...]
    selected_value: str
    raw_value: str
    accepted: bool
    accept_reason: str
    candidates: tuple[dict, ...]
    attempts: tuple[dict, ...]
    router_reason: str


@dataclass
class CascadeResult:
    field_name: str
    bbox: tuple[int, int, int, int]
    candidates: list[dict]
    attempts: list[dict]
    router_reason: str
    status: str
    cascade_trace: list[CascadeStepResult] = field(default_factory=list)
    accepted: bool = False
    accept_reason: str = "EXHAUSTED"
    strategy_id: str = "field-cascade-v12"


RecognizeFn = Callable[
    [str, tuple[int, int, int, int], str, tuple[str, ...]],
    tuple[list[dict], list[dict], str],
]


def _clamp(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return (x0, y0, x1, y1)


def load_route_engines(field_name: str, route_path: Path | None = None) -> tuple[str, ...]:
    """Return (primary, confirmation, …) engines for a governed field route.

    Speed knobs (env):
      CDP_OCR_PRIMARY_OVERRIDE=paddleocr|rapidocr — reorder primary for STP eval
      CDP_OCR_SELECTIVE_CONFIRM=1 — callers should stop after shaped primary
    """
    import os

    path = route_path or _ROUTE_PATH
    engines: list[str] = []
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        route = (payload.get("ocr_routes") or {}).get(field_name) or {}
        for key in ("primary_engine", "confirmation_engine"):
            raw = route.get(key)
            if not raw:
                continue
            engine = _ENGINE_ALIASES.get(str(raw).casefold())
            if engine and engine not in engines:
                engines.append(engine)
    for engine in ("paddleocr", "rapidocr", "tesseract"):
        if engine not in engines:
            engines.append(engine)
    override = (os.environ.get("CDP_OCR_PRIMARY_OVERRIDE") or "").strip().casefold()
    override = _ENGINE_ALIASES.get(override, override)
    if override and override in engines:
        engines = [override] + [engine for engine in engines if engine != override]
    return tuple(engines)


def field_requires_independent_confirmation(field_name: str) -> bool:
    """Patient name cannot STP on one engine.

    The governed policy accepts independent OCR plus a strong check, or
    multi-attribute identity. Selective confirm skips the second engine once
    the primary read is field-shaped, so E2 never forms.
    """
    return (field_name or "").casefold() == "patient_name"


def semantic_accept(
    field_name: str,
    value: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[bool, str]:
    """Return whether a span-selected value is field-shaped enough to stop."""
    text = (value or "").strip()
    if not text:
        return False, "EMPTY"
    name = (field_name or "").casefold()
    datatype = span_datatype_for_field(field_name, "")

    if name in {"patient_dob", "date_of_birth", "insured_dob"} or datatype == "DATE":
        if _DATE_SHAPE.fullmatch(text):
            return True, "DATE_SHAPED"
        return False, "NOT_DATE_SHAPED"

    if name in {"total_charge", "total_charges", "charges", "charge_amount"} or datatype == "CURRENCY":
        if name in {"total_charge", "total_charges"} and text.startswith("-"):
            return False, "CURRENCY_LEADING_MINUS"
        cleaned = text.lstrip("$").replace(",", "")
        cleaned = re.sub(r"[Oo]", "0", cleaned)
        cleaned = re.sub(r"[gG]", "0", cleaned)
        spaced = re.fullmatch(r"(\d{1,6})[\s:.](\d{2})", cleaned)
        if spaced:
            cleaned = f"{spaced.group(1)}.{spaced.group(2)}"
        else:
            thousands = re.fullmatch(r"(\d{1,3})\s(\d{3})", cleaned)
            if thousands:
                cleaned = f"{thousands.group(1)}{thousands.group(2)}.00"
        if _CURRENCY_SHAPE.fullmatch(text.lstrip("$")) or _CURRENCY_SHAPE.fullmatch(cleaned):
            if name in {"total_charge", "total_charges"} and re.fullmatch(r"[0-9]\.\d{2}", cleaned):
                return False, "CURRENCY_SUSPICIOUS_TINY"
            # Form-ruling digit soup (208408.00 / 420840.00) is not a claim total.
            try:
                amount = float(cleaned)
            except ValueError:
                amount = None
            if (
                name in {"total_charge", "total_charges"}
                and amount is not None
                and amount > 99999.99
            ):
                return False, "CURRENCY_IMPLAUSIBLE_TOTAL"
            # Claim-total POS bleed: 11.00 etc. cannot semantic-accept as box-28
            # without geometry proof that the crop is inside Box 28.
            try:
                from packages.geometry_authority import (
                    box28_contains,
                    is_pos_like_currency,
                )

                if name in {"total_charge", "total_charges"} and is_pos_like_currency(
                    cleaned
                ):
                    if bbox is not None and box28_contains(
                        bbox, image_size=image_size
                    ):
                        return True, "CURRENCY_SHAPED_BOX28"
                    return False, "CURRENCY_POS_LIKE_REQUIRES_GEOMETRY"
            except Exception:  # noqa: BLE001, S110 -- optional geometry must not break cascade
                pass
            return True, "CURRENCY_SHAPED"
        return False, "NOT_CURRENCY_SHAPED"

    if name in {"patient_name", "insured_name"} or datatype == "PERSON_NAME":
        upper = text.upper()
        # Form-label residue is not a person name even if token count looks ok.
        if re.search(
            r"(?:[I1L]NSUR[EFO0][DO0]'?S?|[PF]AT[I1L]?E?NT'?S?|PATENTS)\s*NAME",
            upper,
        ):
            return False, "NAME_LABEL_CONTAMINATED"
        if re.search(
            r"(?:LAST|FIRST|FURST|FST|MIDDLE)\s*NAME|MIDDLE\s*INITIAL",
            upper,
        ):
            return False, "NAME_LABEL_CONTAMINATED"
        tokens = _NAME_TOKEN.findall(upper)
        stop = {
            "PATIENT", "FATIENT", "INSURED", "1NSURED", "NAME", "LAST", "FIRST",
            "FURST", "FST", "MIDDLE", "INITIAL", "MIDDIE", "MIDALE",
        }
        tokens = [
            t for t in tokens
            if t not in stop
            and not t.endswith("NAME")
            and not re.fullmatch(r"[I1L]NSUR[EFO0][DO0]'?S?", t)
            and not re.fullmatch(r"[PF]AT[I1L]?E?NT'?S?", t)
            and "NAME" not in t  # glued PATIENTSNAMELASTNAME residue
        ]
        if len(tokens) >= 2 or (len(tokens) == 1 and len(tokens[0]) >= 3):
            return True, "NAME_SHAPED"
        return False, "NOT_NAME_SHAPED"

    if name in {"insured_id_number", "member_id"} or datatype == "ALPHANUMERIC_ID":
        compact = text.upper().replace(" ", "")
        # Reject header-only crops that still contain ID NUMBER / PROGRAM boilerplate.
        if re.search(r"INSUR|NUMBER|PROGRAM|ITEM", compact) and not _ID_SHAPE.fullmatch(
            re.sub(r"[^A-Z0-9]", "", compact)
        ):
            return False, "ID_LABEL_CONTAMINATED"
        if _ID_SHAPE.fullmatch(compact):
            return True, "ID_SHAPED"
        return False, "NOT_ID_SHAPED"

    return True, "NON_EMPTY"


def _normalize_for_agreement(field_name: str, value: str) -> str:
    """Normalize span-selected text so primary/confirmation can agree."""
    text = (value or "").strip()
    if not text:
        return ""
    name = (field_name or "").casefold()
    datatype = span_datatype_for_field(field_name, "")
    if name in {"patient_dob", "date_of_birth"} or datatype == "DATE":
        digits = re.sub(r"\D", "", text)
        if len(digits) == 8:
            return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
        return text.replace("-", "/")
    if name in {"total_charge", "total_charges", "charges", "charge_amount"} or datatype == "CURRENCY":
        return text.lstrip("$").replace(",", "")
    if name in {"insured_id_number", "member_id"} or datatype == "ALPHANUMERIC_ID":
        compact = text.upper().replace(" ", "").replace("-", "")
        if compact.isdigit():
            return compact.lstrip("0") or "0"
        return compact
    if name in {"patient_name", "insured_name"} or datatype == "PERSON_NAME":
        compact = re.sub(r"\s+", " ", text.upper()).strip()
        compact = re.sub(r"[^A-Z0-9 ]", "", compact)
        return re.sub(r"(?<=[A-Z])1(?=[A-Z]|$)", "I", compact.replace(" ", ""))
    return text


def pick_engine_candidates(
    field_name: str,
    candidates: list[dict],
    field_type: str = "",
    *,
    bbox: tuple[float, float, float, float] | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[str, str, str, list[dict]]:
    """Choose a cascade value from dual-engine candidates.

    Past learning: confirmation OCR must run, but the first engine's raw text
    is often header bleed / fragments while the second is field-shaped. Prefer:

    1. Multi-engine agreement after span-select (accuracy + E2 signal)
    2. First field-shaped value in candidate/engine order (primary wins ties)
    3. First non-empty span (keep best for later fusion / HITL)

    Returns (selected, raw, accept_reason, ordered_candidates).
    """
    datatype = span_datatype_for_field(field_name, field_type)
    shaped: list[tuple[str, str, dict, str]] = []
    nonempty: list[tuple[str, str, dict]] = []
    by_norm: dict[str, list[tuple[str, str, dict, str]]] = {}

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        raw = (cand.get("raw_value") or cand.get("value") or "") or ""
        seed = (cand.get("value") or raw or "").strip()
        if not seed and not raw.strip():
            continue
        span = select_field_span(seed or raw, datatype, field_name)
        selected = (span.selected_text or "").strip()
        if not selected:
            # Fall back to raw span when value was already empty.
            span = select_field_span(raw, datatype, field_name)
            selected = (span.selected_text or "").strip()
        if not selected:
            continue
        ok, reason = semantic_accept(
            field_name, selected, bbox=bbox, image_size=image_size
        )
        nonempty.append((selected, raw, cand))
        if ok:
            shaped.append((selected, raw, cand, reason))
            key = _normalize_for_agreement(field_name, selected)
            if key:
                by_norm.setdefault(key, []).append((selected, raw, cand, reason))

    # Prefer agreement across ≥2 distinct engines (paddle + rapid confirmation).
    for group in by_norm.values():
        engines = {
            str(item[2].get("engine") or "").casefold()
            for item in group
            if (item[2].get("engine") or "").strip()
        }
        if len(engines) >= 2:
            selected, raw, cand, reason = group[0]
            seen_ids = {id(item[2]) for item in group}
            ordered = [item[2] for item in group] + [
                c for c in candidates if id(c) not in seen_ids
            ]
            return selected, raw, f"MULTI_ENGINE_AGREEMENT:{reason}", ordered

    if shaped:
        selected, raw, cand, reason = shaped[0]
        ordered = [cand] + [c for c in candidates if c is not cand]
        return selected, raw, reason, ordered

    if nonempty:
        selected, raw, cand = nonempty[0]
        ordered = [cand] + [c for c in candidates if c is not cand]
        return selected, raw, "NON_EMPTY_NO_SHAPE", ordered

    return "", "", "EMPTY", list(candidates)


def _order_by_strategy_ladder(field_name: str, variants: list[CropVariant]) -> list[CropVariant]:
    """Order / filter crop variants using the declared strategy ladder."""
    ladder = crop_ladder_for(field_name)
    if not ladder:
        return variants
    by_id = {variant.variant_id: variant for variant in variants}
    ordered: list[CropVariant] = []
    for variant_id in ladder:
        variant = by_id.pop(variant_id, None)
        if variant is not None:
            ordered.append(variant)
    # Keep any code-defined variants not yet listed in YAML (forward-compatible).
    ordered.extend(by_id.values())
    return ordered


def crop_variants(
    field_name: str,
    primary: tuple[int, int, int, int],
    cell: Mapping[str, int],
    image_size: tuple[int, int],
) -> list[CropVariant]:
    """Typed crop ladder for critical CMS-1500 fields."""
    width, height = image_size
    cell_box = (int(cell["x0"]), int(cell["y0"]), int(cell["x1"]), int(cell["y1"]))
    cell_w = cell_box[2] - cell_box[0]
    cell_h = cell_box[3] - cell_box[1]
    variants: list[CropVariant] = [
        CropVariant("primary", _clamp(primary, width, height), "TEMPLATE_OR_INSET_PRIMARY"),
    ]
    name = (field_name or "").casefold()

    if name == "patient_dob":
        # Keep the year column: prior -8% right trim clipped typed YY on IJN2.022.
        lower = (
            max(primary[0], cell_box[0] + 2),
            max(primary[1], cell_box[3] - max(22, int(0.38 * cell_h))),
            min(max(primary[2], cell_box[2] - 2), cell_box[2] - 1),
            min(primary[3], cell_box[3] - 1),
        )
        variants.append(CropVariant("dob_digit_band", _clamp(lower, width, height), "DOB_DIGIT_BAND"))
        loose = (
            max(primary[0], cell_box[0] + 2),
            max(primary[1], cell_box[1] + int(0.22 * cell_h)),
            min(max(primary[2], cell_box[2] - 2), cell_box[2] - 1),
            min(primary[3], cell_box[3] - 2),
        )
        variants.append(CropVariant("dob_loose", _clamp(loose, width, height), "DOB_LOOSE_TOP"))
        year_wide = (
            max(primary[0], cell_box[0] + int(0.45 * cell_w)),
            max(primary[1], cell_box[3] - max(22, int(0.40 * cell_h))),
            min(cell_box[2] - 1, width),
            min(primary[3], cell_box[3] - 1),
        )
        variants.append(CropVariant("dob_year_wide", _clamp(year_wide, width, height), "DOB_YEAR_WIDE"))

    elif name in {"total_charge", "total_charges"}:
        # Value-only primary already excludes the "28. TOTAL CHARGE" caption when
        # ocr_from_geometry forces box28_value_only_bbox. Add a right-biased
        # tight crop (avoids left $ / caption bleed) and a slightly taller band.
        tight = (
            max(primary[0], primary[0] + max(20, int(0.18 * (primary[2] - primary[0])))),
            max(primary[1], primary[1] + 2),
            primary[2],
            primary[3],
        )
        variants.append(
            CropVariant("box28_tight", _clamp(tight, width, height), "BOX28_TIGHT_VALUE")
        )
        cleared = (
            max(primary[0], cell_box[0] + int(0.28 * cell_w)),
            max(primary[1], cell_box[1] + 2),
            min(primary[2], cell_box[2] - 2),
            min(primary[3], cell_box[3] - 2),
        )
        variants.append(CropVariant("charge_npi_cleared", _clamp(cleared, width, height), "CHARGE_NPI_CLEARED"))
        taller = (
            primary[0],
            max(0, primary[1] - 5),
            primary[2],
            min(height, primary[3] + 5),
        )
        variants.append(CropVariant("charge_taller", _clamp(taller, width, height), "CHARGE_TALLER"))

    elif name in {"patient_name", "insured_name"}:
        value_band = (
            max(primary[0], cell_box[0] + 2),
            max(primary[1], cell_box[1] + int(0.35 * cell_h)),
            min(primary[2], cell_box[2] - 2),
            min(primary[3], cell_box[3] - 1),
        )
        variants.append(CropVariant("name_value_band", _clamp(value_band, width, height), "NAME_VALUE_BAND"))

    elif name in {"insured_id_number", "member_id"}:
        value_band = (
            max(primary[0], cell_box[0] + 2),
            max(primary[1], cell_box[1] + int(0.40 * cell_h)),
            min(primary[2], cell_box[2] - 2),
            min(primary[3], cell_box[3] - 1),
        )
        variants.append(CropVariant("id_value_band", _clamp(value_band, width, height), "ID_VALUE_BAND"))

    seen: set[tuple[int, int, int, int]] = set()
    unique: list[CropVariant] = []
    for variant in variants:
        if variant.bbox in seen:
            continue
        seen.add(variant.bbox)
        unique.append(variant)
    return _order_by_strategy_ladder(field_name, unique)


def charge_column_windows(primary_x0: int, primary_x1: int) -> list[tuple[str, int, int]]:
    """Alternate service-line charge x-windows (pointer-bleed avoidance)."""
    windows = [
        ("charges_primary", primary_x0, primary_x1),
        # Right-shifted bands: live CMS-1500 amounts often sit past diagnosis pointer.
        ("charges_mid", max(primary_x0 + 40, 1000), min(max(primary_x1 + 40, 1145), 1210)),
        ("charges_right", 1050, 1165),
        # Cents column sits further right on some CMS renderers. Still left of units.
        ("charges_cents", 1050, 1180),
        ("charges_far_right", 1100, 1220),
    ]
    seen: set[tuple[int, int]] = set()
    out: list[tuple[str, int, int]] = []
    for variant_id, x0, x1 in windows:
        key = (x0, x1)
        if key in seen or x1 - x0 < 8:
            continue
        seen.add(key)
        out.append((variant_id, x0, x1))
    return out


def charge_windows_for_mode(
    primary_x0: int, primary_x1: int, *, fast: bool
) -> list[tuple[int, int]]:
    """In STP fast mode keep primary + mid + right — primary alone clips ink."""
    named = charge_column_windows(primary_x0, primary_x1)
    if not fast:
        return [(x0, x1) for _, x0, x1 in named]
    keep: list[tuple[int, int]] = []
    for variant_id, x0, x1 in named:
        if variant_id in {
            "charges_primary",
            "charges_mid",
            "charges_right",
            "charges_cents",
        }:
            keep.append((x0, x1))
        if len(keep) >= 4:
            break
    return keep or ([(named[0][1], named[0][2])] if named else [])


class FieldCascade:
    """Execute the crop × engine cascade for one field."""

    def __init__(
        self,
        *,
        route_path: Path | None = None,
        strategy_id: str | None = None,
    ) -> None:
        self._route_path = route_path
        self.strategy_id = strategy_id or load_cascade_strategy().strategy_id

    def recognize(
        self,
        *,
        field_name: str,
        primary_bbox: tuple[int, int, int, int],
        cell: Mapping[str, int],
        image_size: tuple[int, int],
        recognize_fn: RecognizeFn,
        field_type: str = "",
    ) -> CascadeResult:
        engines = load_route_engines(field_name, self._route_path)
        variants = crop_variants(field_name, primary_bbox, cell, image_size)
        trace: list[CascadeStepResult] = []
        best: CascadeStepResult | None = None

        for variant in variants:
            candidates, attempts, reason = recognize_fn(
                field_name,
                variant.bbox,
                field_type,
                engines,
            )
            selected, raw, pick_reason, ordered = pick_engine_candidates(
                field_name,
                list(candidates),
                field_type,
                bbox=variant.bbox,
                image_size=image_size,
            )
            candidates = ordered
            if selected and candidates:
                lead = dict(candidates[0])
                lead["value"] = selected
                if raw:
                    lead["raw_value"] = raw
                if pick_reason.startswith("MULTI_ENGINE_AGREEMENT:"):
                    lead["reason_code"] = pick_reason
                candidates[0] = lead
            ok = pick_reason not in {"EMPTY", "NON_EMPTY_NO_SHAPE"} and bool(selected)
            if ok:
                accept_reason = pick_reason
            else:
                accept_reason = pick_reason if pick_reason else "EMPTY"
                if selected:
                    ok, accept_reason = semantic_accept(
                        field_name,
                        selected,
                        bbox=variant.bbox,
                        image_size=image_size,
                    )

            # Tesseract fill: dual-engine confirmation short-circuits before the
            # fill engine whenever paddle+rapid return any usable text — even
            # header bleed. When the crop is still not field-shaped, run
            # tesseract alone and re-pick (observed ink only).
            already_tess = any(
                str((a.get("engine") if isinstance(a, dict) else "") or "").casefold()
                == "tesseract"
                for a in attempts
            )
            if (
                not ok
                and not already_tess
                and "tesseract" in {e.casefold() for e in engines}
            ):
                fill_cands, fill_attempts, fill_reason = recognize_fn(
                    field_name,
                    variant.bbox,
                    field_type,
                    ("tesseract",),
                )
                merged = list(candidates) + list(fill_cands)
                attempts = list(attempts) + list(fill_attempts)
                selected, raw, pick_reason, ordered = pick_engine_candidates(
                    field_name,
                    merged,
                    field_type,
                    bbox=variant.bbox,
                    image_size=image_size,
                )
                candidates = ordered
                if selected and candidates:
                    lead = dict(candidates[0])
                    lead["value"] = selected
                    if raw:
                        lead["raw_value"] = raw
                    lead["reason_code"] = f"TESSERACT_FILL:{pick_reason}"
                    candidates[0] = lead
                ok = pick_reason not in {"EMPTY", "NON_EMPTY_NO_SHAPE"} and bool(
                    selected
                )
                if ok:
                    accept_reason = f"TESSERACT_FILL:{pick_reason}"
                    reason = fill_reason or reason
                else:
                    accept_reason = pick_reason if pick_reason else accept_reason
                    if selected:
                        ok, base_reason = semantic_accept(
                            field_name,
                            selected,
                            bbox=variant.bbox,
                            image_size=image_size,
                        )
                        if ok:
                            accept_reason = f"TESSERACT_FILL:{base_reason}"

            step = CascadeStepResult(
                variant_id=variant.variant_id,
                bbox=variant.bbox,
                engines=engines,
                selected_value=selected,
                raw_value=raw,
                accepted=ok,
                accept_reason=accept_reason,
                candidates=tuple(candidates),
                attempts=tuple(attempts),
                router_reason=reason,
            )
            trace.append(step)
            if best is None or (selected and not (best.selected_value or "").strip()):
                best = step
            if ok:
                return CascadeResult(
                    field_name=field_name,
                    bbox=variant.bbox,
                    candidates=list(candidates),
                    attempts=list(attempts),
                    router_reason=reason,
                    status="OBSERVED",
                    cascade_trace=trace,
                    accepted=True,
                    accept_reason=accept_reason,
                    strategy_id=self.strategy_id,
                )

        # Cross-variant fusion (DOB/DATE only): digit-band may hold the year
        # while primary holds MM/DD. Span over observed ink only — never invent
        # glyphs. Gated so free-text fields do not mint incomplete shells.
        datatype = span_datatype_for_field(field_name, field_type)
        if field_name.casefold() in {
            "patient_dob",
            "date_of_birth",
            "insured_dob",
        } or datatype == "DATE":
            fused_bits: list[str] = []
            donor: dict | None = None
            for step in trace:
                if step.selected_value:
                    fused_bits.append(step.selected_value)
                if step.raw_value:
                    fused_bits.append(step.raw_value)
                for cand in step.candidates:
                    if donor is None and isinstance(cand, dict) and "raw_confidence" in cand:
                        donor = dict(cand)
                    val = (cand.get("value") or cand.get("raw_value") or "").strip()
                    if val:
                        fused_bits.append(val)
            fused = " ".join(fused_bits).strip()
            if fused:
                span = select_field_span(fused, datatype, field_name)
                fused_selected = span.selected_text or ""
                ok, accept_reason = semantic_accept(
                    field_name,
                    fused_selected,
                    bbox=primary_bbox,
                    image_size=image_size,
                )
                if ok and fused_selected:
                    shell = dict(donor) if donor else {}
                    shell.update(
                        {
                            "value": fused_selected,
                            "raw_value": fused,
                            "engine": shell.get("engine") or "rapidocr",
                            "model_name": shell.get("model_name") or "unknown",
                            "model_version": shell.get("model_version") or "unknown",
                            "preprocessing_variant": shell.get("preprocessing_variant")
                            or "recorded_canonical_region",
                            "raw_confidence": float(shell.get("raw_confidence") or 0.8),
                            "calibrated_confidence": shell.get("calibrated_confidence"),
                            "latency_ms": float(shell.get("latency_ms") or 0.0),
                            "validation_results": [],
                            "evidence_reference": None,
                            "estimated_cost_usd": 0.0,
                            "actual_cost_usd": None,
                            "preprocessing_version": shell.get("preprocessing_version") or "none",
                            "registration_confidence": shell.get("registration_confidence"),
                            "image_quality_score": shell.get("image_quality_score"),
                            "reason_code": "CROSS_VARIANT_SPAN_FUSION",
                            "span_selection": {
                                "selected_text": fused_selected,
                                "rule_id": span.rule_id,
                                "confidence": span.confidence,
                                "reason_codes": list(span.reason_codes),
                            },
                        }
                    )
                    if not isinstance(shell.get("provenance"), dict):
                        shell["provenance"] = None
                    if "bounding_box" not in shell:
                        box = best.bbox if best is not None else primary_bbox
                        x0, y0, x1, y1 = box
                        shell["bounding_box"] = {
                            "x0": float(x0),
                            "y0": float(y0),
                            "x1": float(x1),
                            "y1": float(y1),
                            "image_width": float(image_size[0]),
                            "image_height": float(image_size[1]),
                        }
                    fusion_step = CascadeStepResult(
                        variant_id="cross_variant_span",
                        bbox=(best.bbox if best is not None else primary_bbox),
                        engines=engines,
                        selected_value=fused_selected,
                        raw_value=fused,
                        accepted=True,
                        accept_reason=f"CROSS_VARIANT_SPAN_FUSION:{accept_reason}",
                        candidates=(shell,),
                        attempts=(),
                        router_reason="CROSS_VARIANT_SPAN_FUSION",
                    )
                    trace.append(fusion_step)
                    return CascadeResult(
                        field_name=field_name,
                        bbox=fusion_step.bbox,
                        candidates=list(fusion_step.candidates),
                        attempts=[],
                        router_reason="CROSS_VARIANT_SPAN_FUSION",
                        status="OBSERVED",
                        cascade_trace=trace,
                        accepted=True,
                        accept_reason=fusion_step.accept_reason,
                        strategy_id=self.strategy_id,
                    )

        if best is not None and best.candidates:
            return CascadeResult(
                field_name=field_name,
                bbox=best.bbox,
                candidates=list(best.candidates),
                attempts=list(best.attempts),
                router_reason=best.router_reason,
                status="OBSERVED" if best.selected_value else "NO_OBSERVATION",
                cascade_trace=trace,
                accepted=False,
                accept_reason=best.accept_reason,
                strategy_id=self.strategy_id,
            )
        return CascadeResult(
            field_name=field_name,
            bbox=primary_bbox,
            candidates=[],
            attempts=[],
            router_reason="EXHAUSTED",
            status="NO_OBSERVATION",
            cascade_trace=trace,
            accepted=False,
            accept_reason="EXHAUSTED",
            strategy_id=self.strategy_id,
        )
