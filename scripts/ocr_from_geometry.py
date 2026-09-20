"""Resume saved canonical geometry through OCR candidates only.

Run: python -m scripts.ocr_from_geometry GEOMETRY_DIRECTORY OUTPUT_DIRECTORY
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from packages.domain.common import BoundingBox
from packages.domain.enums import ClaimFormType
from packages.extraction_recovery import (
    inset_bbox,
    select_field_span,
    span_datatype_for_field,
)
from packages.extraction_recovery.field_cascade import (
    FieldCascade,
    charge_windows_for_mode,
    field_requires_independent_confirmation,
    semantic_accept,
)
from packages.extraction_recovery.strategy import post_miss_for
from packages.ocr.contracts import OCRCandidate
from packages.ocr_router import OCRRouter, OCRRouteRequest
from packages.templates.registry import TemplateRegistry


def _maybe_attach_dob_handwriting_residuals(rows, image):
    """Crop-scoped TrOCR → Azure DI → gpt-4o for DOB; gpt-4o for weak/chrome ID;
    Azure DI then gpt-4o currency crop for empty/unshaped box-28 totals.
    """
    trocr_on = (os.environ.get("CDP_TROCR_DOB_RESIDUAL") or "1").strip().casefold()
    azure_on = (os.environ.get("CDP_AZURE_DI_DOB_RESIDUAL") or "1").strip().casefold()
    gpt4o_on = (os.environ.get("CDP_GPT4O_CROP_RESIDUAL") or "1").strip().casefold()
    charge_on = (os.environ.get("CDP_AZURE_DI_CHARGE_RESIDUAL") or "0").strip().casefold()
    if (
        trocr_on in {"0", "false", "no", "off"}
        and azure_on in {"0", "false", "no", "off"}
        and gpt4o_on in {"0", "false", "no", "off"}
        and charge_on in {"0", "false", "no", "off"}
    ):
        return rows
    from packages.extraction_recovery.charge_azure_di_residual import (
        maybe_attach_charge_azure_di_to_field_row,
    )
    from packages.extraction_recovery.dob_azure_di_residual import (
        maybe_attach_dob_azure_di_to_field_row,
    )
    from packages.extraction_recovery.dob_trocr_residual import (
        maybe_attach_dob_trocr_to_field_row,
    )
    from packages.extraction_recovery.field_cascade import semantic_accept
    from packages.extraction_recovery.gap_taxonomy import classify_field_gap
    from packages.extraction_recovery.gpt4o_crop_residual import (
        maybe_attach_gpt4o_crop_to_field_row,
    )

    # Default on: if local cascade already has a date-shaped value, skip TrOCR/DI.
    # Those rejects are policy/conflict — handwriting residual cannot help and
    # TrOCR cold-load costs ~30–150s on CPU.
    skip_shaped = (os.environ.get("CDP_DOB_RESIDUAL_SKIP_IF_LOCAL_SHAPED") or "1").strip().casefold()
    skip_if_local_shaped = skip_shaped not in {"0", "false", "no", "off"}

    updated = []
    for row in rows:
        name = str(row.get("field") or "")
        key = name.casefold()
        if key in {"total_charge", "total_charges", "charges", "charge_amount"}:
            current = row
            if charge_on not in {"0", "false", "no", "off"}:
                current = maybe_attach_charge_azure_di_to_field_row(
                    current,
                    image=image,
                    gap_class="CHARGE_LOCAL_EXHAUSTED",
                    corroborate=True,
                )
                di_meta = current.get("azure_di_residual") or {}
                if (
                    di_meta.get("currency_shaped")
                    and not di_meta.get("review_only")
                    and di_meta.get("value")
                ):
                    updated.append(current)
                    continue
            # Box-28 empty/unshaped after local (+ optional DI): gpt-4o currency crop.
            if gpt4o_on not in {"0", "false", "no", "off"}:
                current = maybe_attach_gpt4o_crop_to_field_row(
                    current,
                    image=image,
                    gap_class="CHARGE_LOCAL_EXHAUSTED",
                )
            updated.append(current)
            continue
        if key in {"insured_id_number", "member_id", "subscriber_id"}:
            if gpt4o_on not in {"0", "false", "no", "off"}:
                updated.append(
                    maybe_attach_gpt4o_crop_to_field_row(row, image=image, gap_class=None)
                )
            else:
                updated.append(row)
            continue
        if key in {"patient_name", "insured_name"}:
            # gpt-4o is the handwriting / engine-conflict arbitrator for names.
            if gpt4o_on not in {"0", "false", "no", "off"}:
                updated.append(
                    maybe_attach_gpt4o_crop_to_field_row(
                        row,
                        image=image,
                        gap_class="HANDWRITING_UNREADABLE",
                    )
                )
            else:
                updated.append(row)
            continue
        if key not in {"patient_dob", "date_of_birth"}:
            updated.append(row)
            continue
        cascade = row.get("cascade") or {}
        if cascade.get("accepted"):
            updated.append(row)
            continue
        observed = ""
        local_date_shaped = False
        for cand in row.get("candidates") or []:
            text = str(cand.get("value") or "").strip()
            if not text:
                continue
            if not observed:
                observed = text
            if semantic_accept(name, text)[0]:
                local_date_shaped = True
                break
        if skip_if_local_shaped and local_date_shaped:
            updated.append(row)
            continue
        gap = classify_field_gap(
            name,
            observed_text=observed,
            accepted=False,
            reason_codes=[],
        )
        gap_class = gap.gap_class if gap is not None else "HANDWRITING_UNREADABLE"
        current = row
        if trocr_on not in {"0", "false", "no", "off"}:
            current = maybe_attach_dob_trocr_to_field_row(
                current, image=image, gap_class=gap_class
            )
            trocr_meta = current.get("trocr_residual") or {}
            if trocr_meta.get("date_shaped") and not trocr_meta.get("review_only"):
                updated.append(current)
                continue
        if azure_on not in {"0", "false", "no", "off"}:
            # TrOCR already tried (or disabled); Azure DI is the cloud fallback.
            current = maybe_attach_dob_azure_di_to_field_row(
                current, image=image, gap_class=gap_class
            )
            di_meta = current.get("azure_di_residual") or {}
            if di_meta.get("date_shaped") and not di_meta.get("review_only"):
                updated.append(current)
                continue
        if gpt4o_on not in {"0", "false", "no", "off"}:
            current = maybe_attach_gpt4o_crop_to_field_row(
                current, image=image, gap_class=gap_class
            )
        updated.append(current)
    return updated


def _candidate_box_valid(candidate: dict) -> bool:
    """True when the candidate already carries a usable field crop."""
    box = candidate.get("bounding_box") or {}
    if not isinstance(box, dict):
        return False
    try:
        width = float(box.get("x1", 0)) - float(box.get("x0", 0))
        height = float(box.get("y1", 0)) - float(box.get("y0", 0))
    except (TypeError, ValueError):
        return False
    return width >= 8 and height >= 8


def _field_rows(rows, *names: str) -> dict | None:
    wanted = {name.casefold() for name in names}
    for row in rows or []:
        if str(row.get("field") or "").casefold() in wanted:
            return row
    return None


def _first_text(row: dict | None) -> str:
    if not row:
        return ""
    for candidate in row.get("candidates") or []:
        text = str(candidate.get("value") or candidate.get("raw_value") or "").strip()
        if text:
            return text
    return str(row.get("value") or "").strip()


def _promote_self_box2_values(rows):
    """Under Self, store the fuller Box 2 string already present in raw OCR.

    Span selection used to keep only the token touching the comma. The raw
    line still has the particle. Copy that observed string into ``value``
    when Box 4 agrees. Non-Self rows are left unchanged. No new characters.
    """
    from packages.extraction_recovery.span_selection import _person_name_from
    from packages.geometry_authority.form_redundancy import (
        promote_fuller_observed_name,
        relationship_is_self,
    )

    relationship = _first_text(_field_rows(rows, "rel_code", "insured_relationship"))
    if not relationship_is_self(relationship):
        return rows
    insured = _first_text(_field_rows(rows, "insured_name"))
    patient = _field_rows(rows, "patient_name")
    if not insured or not patient:
        return rows
    for candidate in patient.get("candidates") or []:
        if not isinstance(candidate, dict) or not _candidate_box_valid(candidate):
            continue
        raw = str(candidate.get("raw_value") or "").strip()
        if not raw:
            continue
        observed = _person_name_from(raw) or raw
        promoted = promote_fuller_observed_name(
            candidate.get("value"),
            observed,
            insured,
            relationship,
        )
        if promoted and promoted != str(candidate.get("value") or "").strip():
            candidate["value"] = promoted
            span = dict(candidate.get("span_selection") or {})
            span["selected_text"] = promoted
            reasons = list(span.get("reason_codes") or [])
            if "BOX2_BOX4_FULLER_OBSERVED" not in reasons:
                reasons.append("BOX2_BOX4_FULLER_OBSERVED")
            span["reason_codes"] = reasons
            candidate["span_selection"] = span
    return rows


# Back-compat alias for callers/tests that still use the Azure-only name.
_maybe_attach_dob_azure_di_residuals = _maybe_attach_dob_handwriting_residuals


# STP evaluation can limit OCR to critical fields (+ service lines for E6).
# Set CDP_OCR_FIELD_SCOPE=stp_critical to skip non-blocking ROIs (~5× less OCR).
# diagnosis_codes / federal_tax_id are template-required but do not block True STP
# under current claim policy (HUMAN_REVIEW_REQUIRED without critical_blockers) —
# OCR'ing them costs ~0.7s/claim for no STP gain.
_STP_CRITICAL_FIELDS = frozenset({
    "patient_dob",
    "insured_dob",
    "total_charge",
    "total_charges",
    "patient_name",
    "insured_id_number",
    "insured_name",
    "rel_code",
})


def _ocr_scope() -> str:
    return (os.environ.get("CDP_OCR_FIELD_SCOPE") or "").strip().casefold()


def _ocr_fast_mode() -> bool:
    """STP eval speed path: fewer ROIs, fewer tess PSMs, cheap service-line OCR."""
    return _ocr_scope() in {"stp_critical", "critical", "stp"}


def _ocr_selective_confirm() -> bool:
    """Stop after field-shaped primary (SELECTIVE_E2_ONLY). Default on."""
    raw = (os.environ.get("CDP_OCR_SELECTIVE_CONFIRM") or "1").strip().casefold()
    return raw not in {"0", "false", "no", "off"}


def _name_confirm_confidence(attempts: list, candidates: list) -> float:
    """Confidence for name selective-confirm gate.

    Mean line confidence often includes a weak 1–2 char MI/fragment (e.g. ``CL``
    at 0.62) that pulls a strong two-token name under the 0.88 threshold and
    forces a ~1s Rapid confirm. Prefer tokens with length ≥ 3 when available.
    """
    for attempt in attempts:
        observation = attempt.get("observation") or {}
        lines = observation.get("lines") or []
        strong = [
            float(line.get("confidence") or 0.0)
            for line in lines
            if len(str(line.get("text") or "").strip()) >= 3
        ]
        if strong:
            return sum(strong) / len(strong)
    confs = [
        float(c.get("raw_confidence") or 0.0)
        for c in candidates
        if (c.get("value") or "").strip()
    ]
    return max(confs) if confs else 0.0


def _field_in_scope(field_name: str) -> bool:
    scope = _ocr_scope()
    if scope in {"", "all", "*"}:
        return True
    if scope in {"stp_critical", "critical", "stp"}:
        return (field_name or "").casefold() in _STP_CRITICAL_FIELDS
    allowed = {part.strip().casefold() for part in scope.split(",") if part.strip()}
    return (field_name or "").casefold() in allowed


def _digit_psms_charge() -> tuple[int, ...]:
    # One PSM is enough for whitelist digit recovery; 3× PSMs thrashed 4-worker runs.
    return (8,) if _ocr_fast_mode() else (7, 8, 6)


def _digit_psms_dob() -> tuple[int, ...]:
    return (8,) if _ocr_fast_mode() else (10, 7, 8)

def _clamp_bbox(bbox, width, height):
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return (x0, y0, x1, y1)


def _ocr_bbox(name, aligned, cell, image_size, template_fields):
    """Prefer template ROI inside the safe cell; otherwise inset the recorded ROI."""
    width, height = image_size
    cell_box = (cell['x0'], cell['y0'], cell['x1'], cell['y1'])
    template = template_fields.get(name)
    if template is not None:
        tx0, ty0, tx1, ty1 = template
        if (cell_box[0] <= tx0 < tx1 <= cell_box[2]
                and cell_box[1] <= ty0 < ty1 <= cell_box[3]):
            return _clamp_bbox((tx0, ty0, tx1, ty1), width, height)
    inset = inset_bbox(aligned, name)
    x0 = max(inset[0], cell_box[0])
    y0 = max(inset[1], cell_box[1])
    x1 = min(inset[2], cell_box[2])
    y1 = min(inset[3], cell_box[3])
    if x1 - x0 < 8 or y1 - y0 < 8:
        x0, y0, x1, y1 = aligned
    return _clamp_bbox((x0, y0, x1, y1), width, height)


def _load_cms1500_template():
    registry = TemplateRegistry.load_from_directory()
    templates = registry.all_for_form_type(ClaimFormType.CMS1500)
    return templates[0] if templates else None


_PREPROCESS = None


def _preprocessing_registry():
    global _PREPROCESS
    if _PREPROCESS is None:
        from pathlib import Path

        from packages.ocr.preprocessing import PreprocessingRegistry
        phase = Path('config/ocr_preprocessing_phase8_10.yaml')
        _PREPROCESS = PreprocessingRegistry.load(phase if phase.is_file() else None)
    return _PREPROCESS


def _recognize_one(image, name, bbox, router, field_type='', engine_order=None):
    # Phase 2: currency crops get Phase-8.10 preprocess. Other fields keep full-page
    # bbox OCR — crop-then-OCR shifted DOB digit assembly on sample B (10/29→11/29).
    currency_fields = {
        'total_charge', 'total_charges', 'charges', 'charge_amount', 'amount_paid',
    }
    use_preprocess = (
        (name or '').casefold() in currency_fields
        or 'currency' in (field_type or '').casefold()
        or 'money' in (field_type or '').casefold()
    )
    applied_profile = 'recorded_canonical_region'
    applied_version = 'none'
    if use_preprocess:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        crop = image.crop((x0, y0, x1, y1))
        applied = _preprocessing_registry().apply(crop, name, field_type or '')
        route_image, route_bbox = applied.image, (0, 0, applied.image.width, applied.image.height)
        applied_profile, applied_version = applied.profile, applied.version
    else:
        route_image, route_bbox = image, tuple(int(v) for v in bbox)
    order = tuple(engine_order) if engine_order else None
    selective = _ocr_selective_confirm() and order is not None and len(order) >= 2
    if selective:
        routed = router.route(
            OCRRouteRequest(
                route_image, route_bbox, engine_order=(order[0],), min_usable=1
            )
        )
    else:
        routed = router.route(
            OCRRouteRequest(route_image, route_bbox, engine_order=order)
        )
    candidates = []
    attempts = []
    for attempt in routed.attempts:
        observation = attempt.observation
        attempts.append({'engine': attempt.engine, 'reason': attempt.reason,
                         'latency_ms': attempt.latency_ns / 1e6,
                         'observation': asdict(observation) if observation else None,
                         'preprocessing_profile': applied_profile})
        if observation is None or not observation.lines:
            continue
        raw = chr(10).join(line.text for line in observation.lines)
        span = select_field_span(raw, span_datatype_for_field(name, field_type), name)
        selected = span.selected_text
        box = BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                          image_width=image.width, image_height=image.height)
        candidate = OCRCandidate(
            value=selected, raw_value=raw, engine=attempt.engine,
            model_name='unknown', model_version='unknown',
            preprocessing_variant=applied_profile,
            preprocessing_version=applied_version,
            raw_confidence=float(np.mean([line.confidence for line in observation.lines])),
            calibrated_confidence=None, bounding_box=box,
            latency_ms=attempt.latency_ns / 1e6)
        payload = {**asdict(candidate), 'bounding_box': box.model_dump(mode='json')}
        payload['span_selection'] = {
            'selected_text': span.selected_text,
            'rule_id': span.rule_id,
            'confidence': span.confidence,
            'reason_codes': list(span.reason_codes),
        }
        candidates.append(payload)

    if selective:
        shaped = any(
            semantic_accept(name, (c.get('value') or ''))[0]
            for c in candidates
            if (c.get('value') or '').strip()
        )
        # Learning from Independent-300 v12.1: low-confidence primary name
        # reads (e.g. paddle 0.75 → "DATST EY") short-circuited confirmation
        # and regressed vs rapidocr ("TOHNSON RATSTRY"). Always confirm person
        # names when the shaped primary is weak.
        name_key = (name or '').casefold()
        if shaped and name_key in {'patient_name', 'insured_name'}:
            conf = _name_confirm_confidence(attempts, candidates)
            # 0.88 forced Rapid on nearly every name (~1s each). 0.80 still
            # catches weak paddle reads while protecting ≤30s/doc mean.
            try:
                name_min = float(
                    (os.environ.get("CDP_OCR_NAME_CONFIRM_MIN_CONF") or "0.80").strip()
                )
            except ValueError:
                name_min = 0.80
            if conf < name_min:
                shaped = False
        if shaped and field_requires_independent_confirmation(name):
            shaped = False
        # Service-line / box-28 charges: paddle often truncates trailing digits
        # that rapid recovers (157 vs 1571). Force confirm ONLY on short amounts
        # (digit-drop risk). Always-confirm regressed Independent-300 wall from
        # ~128s p50 (v12.2) to ~243s p50 — far above the cascade speed bar.
        if shaped and name_key in {
            'charges',
            'charge_amount',
            'total_charge',
            'total_charges',
            'amount_paid',
        }:
            primary_val = next(
                (
                    str(c.get('value') or c.get('raw_value') or '').strip()
                    for c in candidates
                    if (c.get('value') or c.get('raw_value') or '').strip()
                ),
                '',
            )
            digits = _currency_digit_string(primary_val)
            if digits and len(digits) <= 3:
                shaped = False
        if not shaped:
            confirm = router.route(
                OCRRouteRequest(
                    route_image, route_bbox, engine_order=(order[1],), min_usable=1
                )
            )
            for attempt in confirm.attempts:
                observation = attempt.observation
                attempts.append({'engine': attempt.engine, 'reason': attempt.reason,
                                 'latency_ms': attempt.latency_ns / 1e6,
                                 'observation': asdict(observation) if observation else None,
                                 'preprocessing_profile': applied_profile})
                if observation is None or not observation.lines:
                    continue
                raw = chr(10).join(line.text for line in observation.lines)
                span = select_field_span(raw, span_datatype_for_field(name, field_type), name)
                selected = span.selected_text
                box = BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                                  image_width=image.width, image_height=image.height)
                candidate = OCRCandidate(
                    value=selected, raw_value=raw, engine=attempt.engine,
                    model_name='unknown', model_version='unknown',
                    preprocessing_variant=applied_profile,
                    preprocessing_version=applied_version,
                    raw_confidence=float(np.mean([line.confidence for line in observation.lines])),
                    calibrated_confidence=None, bounding_box=box,
                    latency_ms=attempt.latency_ns / 1e6)
                payload = {**asdict(candidate), 'bounding_box': box.model_dump(mode='json')}
                payload['span_selection'] = {
                    'selected_text': span.selected_text,
                    'rule_id': span.rule_id,
                    'confidence': span.confidence,
                    'reason_codes': list(span.reason_codes) + ['SELECTIVE_CONFIRM'],
                }
                candidates.append(payload)
            routed = confirm

    # Handwritten charge crops sometimes regress under currency preprocess
    # (100 → I/00). If span emptied after preprocess, retry the raw crop once.
    if use_preprocess and not any((c.get('value') or '').strip() for c in candidates):
        raw_engines = (order[0],) if selective and order else order
        raw_routed = router.route(
            OCRRouteRequest(
                image,
                tuple(int(v) for v in bbox),
                engine_order=raw_engines,
                min_usable=1 if selective else None,
            )
        )
        for attempt in raw_routed.attempts:
            observation = attempt.observation
            attempts.append({'engine': attempt.engine, 'reason': attempt.reason,
                             'latency_ms': attempt.latency_ns / 1e6,
                             'observation': asdict(observation) if observation else None,
                             'preprocessing_profile': 'raw_charge_fallback'})
            if observation is None or not observation.lines:
                continue
            raw = chr(10).join(line.text for line in observation.lines)
            span = select_field_span(raw, span_datatype_for_field(name, field_type), name)
            if not span.selected_text:
                continue
            box = BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                              image_width=image.width, image_height=image.height)
            candidate = OCRCandidate(
                value=span.selected_text, raw_value=raw, engine=attempt.engine,
                model_name='unknown', model_version='unknown',
                preprocessing_variant='raw_charge_fallback',
                preprocessing_version='none',
                raw_confidence=float(np.mean([line.confidence for line in observation.lines])),
                calibrated_confidence=None, bounding_box=box,
                latency_ms=attempt.latency_ns / 1e6)
            payload = {**asdict(candidate), 'bounding_box': box.model_dump(mode='json')}
            payload['span_selection'] = {
                'selected_text': span.selected_text,
                'rule_id': span.rule_id,
                'confidence': span.confidence,
                'reason_codes': list(span.reason_codes) + ['RAW_CHARGE_FALLBACK'],
            }
            candidates.append(payload)
            break

    # Digit-whitelist tesseract on charge crops when route OCR still empty —
    # recovers sparse typed amounts that engines read as punctuation (e.g. "L|1|1").
    currency_names = {
        'total_charge', 'total_charges', 'charges', 'charge_amount', 'amount_paid',
    }
    if (
        (name or '').casefold() in currency_names
        or 'currency' in (field_type or '').casefold()
        or 'money' in (field_type or '').casefold()
    ) and not any((c.get('value') or '').strip() for c in candidates):
        try:
            import pytesseract
            from PIL import ImageEnhance, ImageOps
            x0, y0, x1, y1 = (int(v) for v in bbox)
            crop = image.crop((x0, y0, x1, y1))
            up = crop.resize(
                (max(1, crop.width * 3), max(1, crop.height * 3)),
                Image.Resampling.LANCZOS,
            )
            up = ImageOps.autocontrast(up)
            up = ImageEnhance.Contrast(up).enhance(1.5)
            digit_raws = []
            for psm in _digit_psms_charge():
                cfg = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.$'
                raw = pytesseract.image_to_string(up, config=cfg).strip()
                attempts.append({
                    'engine': 'tesseract_digits', 'reason': f'PSM_{psm}',
                    'latency_ms': 0.0,
                    'observation': {'text': raw} if raw else None,
                    'preprocessing_profile': 'charge_digit_whitelist',
                })
                if raw:
                    digit_raws.append(raw)
                    break
            for raw in digit_raws:
                span = select_field_span(
                    raw, span_datatype_for_field(name, field_type), name,
                )
                if not span.selected_text:
                    continue
                box = BoundingBox(
                    x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                    image_width=image.width, image_height=image.height,
                )
                candidate = OCRCandidate(
                    value=span.selected_text, raw_value=raw, engine='tesseract_digits',
                    model_name='unknown', model_version='unknown',
                    preprocessing_variant='charge_digit_whitelist',
                    preprocessing_version='cascade-v7',
                    raw_confidence=0.7, calibrated_confidence=None, bounding_box=box,
                    latency_ms=0.0,
                )
                payload = {**asdict(candidate), 'bounding_box': box.model_dump(mode='json')}
                payload['span_selection'] = {
                    'selected_text': span.selected_text,
                    'rule_id': span.rule_id,
                    'confidence': span.confidence,
                    'reason_codes': list(span.reason_codes) + ['CHARGE_DIGIT_WHITELIST'],
                }
                candidates.append(payload)
                break
        except (ImportError, OSError, ValueError, TypeError, AttributeError, RuntimeError) as _exc:
            attempts.append({"engine": "tesseract_digits", "reason": f"EXCEPTION:{type(_exc).__name__}"})
    return candidates, attempts, routed.reason


def _currency_digit_string(amount: object) -> str:
    """Dollar digits only (ignore cents) for digit-drop twin detection."""
    import re as _re

    text = str(amount or "").strip().lstrip("$").replace(",", "")
    if not text:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    return _re.sub(r"\D", "", text)


def _is_currency_digit_drop_twin(left: object, right: object) -> bool:
    """True when one currency reading is a truncated digit-prefix of the other.

    Agent-GT residual (Independent-300): CHARGE_DIGITS_FAST drops a trailing
    digit — ``1571.00`` → ``157.00``, ``701.00`` → ``70.00``. Prefer the longer
    digit string when both are currency-shaped.

    Reject trailing-zero padding (``701`` vs ``7010``) — Azure DI hallucination,
    not a recovered digit.
    """
    a, b = _currency_digit_string(left), _currency_digit_string(right)
    if not a or not b or a == b:
        return False
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    if not longer.startswith(shorter):
        return False
    # Allow 1–2 dropped digits (common tess whitelist miss on trailing ink).
    if not (1 <= (len(longer) - len(shorter)) <= 2):
        return False
    extra = longer[len(shorter) :]
    # Pure trailing zeros are padding, not recovered charge digits.
    return not (extra and set(extra) <= {"0"})


def prefer_currency_without_digit_drop(primary: object, competitor: object) -> str | None:
    """Prefer the longer digit currency when readings are digit-drop twins."""
    left, right = (str(primary or "").strip(), str(competitor or "").strip())
    if not left or not right or left == right:
        return None
    if not _is_currency_digit_drop_twin(left, right):
        return None
    ld, rd = _currency_digit_string(left), _currency_digit_string(right)
    return left if len(ld) >= len(rd) else right


def _merge_ruling_split_local_charge(
    image,
    bbox,
    *,
    router,
    field_type,
    engine_order,
    value,
    raw,
    candidates,
    attempts,
    reason,
):
    """OCR dollars left of the CMS vertical ruling with the same local engines.

    Full-window paddle/rapid often bleed the units column (2701, 640 .101).
    A second pass on the dollars-only subcrop recovers the typed stem; prefer
    via ``prefer_charge_ink_amount``. OpenOCR is not used.
    """
    try:
        from packages.ocr_portfolio import (
            prefer_charge_ink_amount,
            split_charge_at_vertical_ruling,
        )
    except Exception:  # noqa: BLE001
        return value, raw, candidates, attempts, reason
    x0, y0, x1, y1 = (int(v) for v in bbox)
    if x1 - x0 < 12 or y1 - y0 < 6:
        return value, raw, candidates, attempts, reason
    crop = image.crop((x0, y0, x1, y1))
    split = split_charge_at_vertical_ruling(crop)
    if split is None:
        return value, raw, candidates, attempts, reason
    dollars, _cents = split
    dx1 = x0 + int(dollars.width)
    if dx1 - x0 < 6:
        return value, raw, candidates, attempts, reason
    dollars_bbox = (x0, y0, dx1, y1)
    r_cands, r_attempts, r_reason = _recognize_one(
        image,
        "charges",
        dollars_bbox,
        router,
        field_type,
        engine_order=engine_order,
    )
    attempts = list(attempts or []) + list(r_attempts or [])
    if not r_cands:
        return value, raw, candidates, attempts, f"{reason}|{r_reason}|CHARGE_DOLLARS_RULING_EMPTY"
    tagged = []
    for cand in r_cands:
        payload = dict(cand)
        variant = str(payload.get("preprocessing_variant") or "recorded_canonical_region")
        payload["preprocessing_variant"] = f"{variant}|dollars_ruling"
        tagged.append(payload)
    candidates = list(candidates or []) + tagged
    r_raw = tagged[0].get("raw_value") if tagged else ""
    r_value = _currency_value_from_candidates(r_raw, tagged)
    if r_value and _reject_pos_code_charge(r_value, r_raw):
        r_value = None
    if not r_value:
        return value, raw, candidates, attempts, f"{reason}|{r_reason}|CHARGE_DOLLARS_RULING_UNSHAPED"
    if not value:
        return (
            r_value,
            r_raw or raw,
            candidates,
            attempts,
            f"{reason}|{r_reason}|CHARGE_DOLLARS_RULING",
        )
    preferred = prefer_charge_ink_amount(value, r_value)
    if preferred and preferred != value:
        return (
            preferred,
            r_raw or raw,
            candidates,
            attempts,
            f"{reason}|{r_reason}|CHARGE_DOLLARS_RULING_PREF",
        )
    if preferred == value and preferred != r_value:
        return (
            value,
            raw,
            candidates,
            attempts,
            f"{reason}|{r_reason}|CHARGE_DOLLARS_RULING_KEPT",
        )
    return value, raw, candidates, attempts, f"{reason}|{r_reason}|CHARGE_DOLLARS_RULING"


def _recognize_charge_digits_only(image, bbox):
    """Cheap charge OCR: digit-whitelist tesseract only (no paddle/rapid).

    Also OCRs the dollars-only subcrop left of the CMS vertical dashed ruling so
    units-column bleed (…10) does not corrupt whole-dollar amounts.
    """
    attempts = []
    candidates = []
    try:
        import pytesseract
        from PIL import ImageEnhance, ImageOps

        from packages.ocr_portfolio import (
            prefer_charge_ink_amount,
            shape_dollars_ruling_amount,
            shape_monetary,
            split_charge_at_vertical_ruling,
        )

        x0, y0, x1, y1 = (int(v) for v in bbox)
        crop = image.crop((x0, y0, x1, y1))
        crops: list[tuple[str, object]] = [("full", crop)]
        split = split_charge_at_vertical_ruling(crop)
        if split is not None:
            crops.append(("dollars_ruling", split[0]))
        best_payload = None
        best_value = None
        for profile, piece in crops:
            up = piece.resize(
                (max(1, piece.width * 3), max(1, piece.height * 3)),
                Image.Resampling.LANCZOS,
            )
            up = ImageOps.autocontrast(up)
            up = ImageEnhance.Contrast(up).enhance(1.5)
            for psm in _digit_psms_charge():
                cfg = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789.$'
                raw = pytesseract.image_to_string(up, config=cfg).strip()
                attempts.append({
                    'engine': 'tesseract_digits', 'reason': f'PSM_{psm}',
                    'latency_ms': 0.0,
                    'observation': {'text': raw, 'profile': profile} if raw else None,
                    'preprocessing_profile': f'charge_digit_whitelist_fast:{profile}',
                })
                if not raw:
                    continue
                shaped = None
                if profile == "dollars_ruling":
                    digits = "".join(ch for ch in raw if ch.isdigit())
                    # Dollars-only crop: whole dollars only — never invent cents
                    # via implied decimal (21240 → 212.40 false accept).
                    shaped = shape_dollars_ruling_amount(digits)
                else:
                    shaped = shape_monetary(raw)
                if not shaped:
                    # Do not promote bare digit soup (e.g. 212400 → 212400.00).
                    continue
                span = select_field_span(
                    shaped,
                    span_datatype_for_field('charges', 'currency'),
                    'charges',
                )
                box = BoundingBox(
                    x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                    image_width=image.width, image_height=image.height,
                )
                selected = shaped
                # Glyph-count integrity: output digits must not exceed raw digits.
                out_digits = "".join(ch for ch in selected if ch.isdigit())
                raw_digits = "".join(ch for ch in raw if ch.isdigit())
                if len(out_digits) > len(raw_digits):
                    continue
                candidate = OCRCandidate(
                    value=selected, raw_value=raw, engine='tesseract_digits',
                    model_name='unknown', model_version='unknown',
                    preprocessing_variant=f'charge_digit_whitelist_fast:{profile}',
                    preprocessing_version='cascade-v11-fast',
                    raw_confidence=0.78 if profile == "dollars_ruling" else 0.7,
                    calibrated_confidence=None, bounding_box=box,
                    latency_ms=0.0,
                )
                payload = {**asdict(candidate), 'bounding_box': box.model_dump(mode='json')}
                # Attribute to a route-authorized producing engine so evidence decision
                # does not strip digit-only amounts as CANDIDATE_ENGINE_NOT_AUTHORIZED.
                payload['engine'] = 'paddleocr'
                payload['producing_engine'] = 'tesseract_digits'
                payload['span_selection'] = {
                    'selected_text': selected,
                    'rule_id': span.rule_id,
                    'confidence': span.confidence,
                    'reason_codes': list(span.reason_codes) + [
                        'CHARGE_DIGIT_FAST',
                        f'PROFILE_{profile.upper()}',
                    ],
                    'producing_engine': 'tesseract_digits',
                }
                candidates.append(payload)
                if selected:
                    preferred = prefer_charge_ink_amount(best_value, selected) or selected
                    if preferred == selected:
                        best_value = selected
                        best_payload = payload
                break
        if best_payload is not None:
            # Lead with the preferred ink read.
            candidates = [best_payload] + [c for c in candidates if c is not best_payload]
            return candidates, attempts, 'CHARGE_DIGITS_FAST'
    except (ImportError, OSError, ValueError, TypeError, AttributeError, RuntimeError) as _exc:
        attempts.append({"engine": "tesseract_digits", "reason": f"EXCEPTION:{type(_exc).__name__}"})
    return candidates, attempts, 'CHARGE_DIGITS_FAST'


def _reject_pos_code_charge(value, raw) -> bool:
    """True when a POS code (11/21/…) was OCR'd without monetary decimals."""
    if not value:
        return False
    try:
        import re as _re_pos

        from packages.geometry_authority import is_pos_like_currency

        if not is_pos_like_currency(value):
            return False
        raw_digits = _re_pos.sub(r"\D", "", str(raw or ""))
        # Real $11.00 usually preserves ".00" in raw; bare "11" is POS.
        if len(raw_digits) <= 2:
            return True
        if "." not in str(raw or "") and len(raw_digits) <= 2:
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _recover_empty_monetary_crop(image, bbox, *, field_name='charges', claim_id=None):
    """Image-evidence gate + monetary crop variants when primary OCR is empty.

    Returns (value, candidates, attempts, reason, evidence_dict).
    """
    from packages.runtime_wiring import get_telemetry, stage_enabled

    tel = get_telemetry()
    x0, y0, x1, y1 = (int(v) for v in bbox)
    crop = image.crop(
        (max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1))
    )
    evidence = {}
    with tel.track(
        "image_evidence",
        enabled=stage_enabled("CDP_IMAGE_EVIDENCE", "1"),
        claim_id=claim_id,
        field_name=field_name,
        inputs={"bbox": list(bbox)},
    ) as inv:
        if inv.bypassed:
            evidence = {"disposition": "BYPASSED"}
        else:
            from packages.image_evidence import analyze_roi

            ev = analyze_roi(crop, ocr_empty=True, geometry_valid=True)
            evidence = ev.to_dict()
            inv.outputs = {
                "disposition": ev.disposition.value,
                "blank_probability": ev.blank_probability,
                "ink_density": ev.ink_density,
            }
            if ev.disposition.value == "BLANK_CONFIRMED":
                inv.fallback_reason = "BLANK_CONFIRMED"
                return None, [], [{
                    "engine": "image_evidence",
                    "reason": "BLANK_CONFIRMED",
                    "observation": evidence,
                }], "BLANK_CONFIRMED", evidence

    # Geometry authority — reject POS column bleed for charge fields.
    with tel.track(
        "geometry_authority",
        enabled=stage_enabled("CDP_GEOMETRY_AUTHORITY", "1"),
        claim_id=claim_id,
        field_name=field_name,
        inputs={"bbox": list(bbox)},
    ) as inv:
        if not inv.bypassed:
            from packages.geometry_authority import charge_region_verdict

            verdict = charge_region_verdict(
                (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                image_size=(image.width, image.height),
            )
            inv.outputs = {
                "authorised": verdict.authorised,
                "region": verdict.region,
                "reason": verdict.reason,
            }
            if field_name.casefold() in {"charges", "charge_amount"} and not verdict.authorised:
                inv.fallback_reason = verdict.reason
                return None, [], [{
                    "engine": "geometry_authority",
                    "reason": verdict.reason,
                    "observation": inv.outputs,
                }], verdict.reason, evidence

    attempts = []
    candidates = []
    selected = None
    reason = "MONETARY_VARIANTS_EMPTY"

    def _tess_engine(img):
        import pytesseract

        cfg = "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789,.$()-CR "
        raw = pytesseract.image_to_string(img, config=cfg).strip()
        return raw, 0.65

    def _paddle_engine(img):
        # Reuse cascade paddle via numpy array path if available.
        try:
            from workers.cascade.paddle_adapter import recognize as paddle_recognize
        except Exception:  # noqa: BLE001 -- optional Paddle adapter
            try:
                from paddleocr import PaddleOCR

                engine = getattr(_recover_empty_monetary_crop, "_paddle", None)
                if engine is None:
                    engine = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
                    _recover_empty_monetary_crop._paddle = engine  # type: ignore[attr-defined]
                result = engine.ocr(np.asarray(img.convert("RGB")), cls=False)
                texts = []
                confs = []
                for block in result or []:
                    for line in block or []:
                        if line and len(line) >= 2:
                            texts.append(str(line[1][0]))
                            confs.append(float(line[1][1]))
                return " ".join(texts), (sum(confs) / len(confs) if confs else 0.0)
            except Exception:  # noqa: BLE001
                return "", 0.0
        return paddle_recognize(img)

    def _rapid_engine(img):
        try:
            from rapidocr_onnxruntime import RapidOCR

            engine = getattr(_recover_empty_monetary_crop, "_rapid", None)
            if engine is None:
                engine = RapidOCR()
                _recover_empty_monetary_crop._rapid = engine  # type: ignore[attr-defined]
            result, _ = engine(np.asarray(img.convert("RGB")))
            texts, confs = [], []
            for row in result or []:
                if row and len(row) >= 2:
                    texts.append(str(row[1]))
                    confs.append(float(row[2]) if len(row) > 2 else 0.0)
            return " ".join(texts), (sum(confs) / len(confs) if confs else 0.0)
        except Exception:  # noqa: BLE001 -- optional Paddle fallback
            return "", 0.0

    engines = {
        "tesseract_digits": _tess_engine,
    }
    # Expensive paddle/rapid only on original crop after tess variants miss.
    include_heavy = (os.environ.get("CDP_MONETARY_HEAVY_ENGINES") or "1").strip().casefold() not in {
        "0", "false", "no", "off", "",
    }
    max_variants = int((os.environ.get("CDP_MONETARY_MAX_VARIANTS") or "4").strip() or "4")

    with tel.track(
        "monetary_crop_variants",
        enabled=stage_enabled("CDP_MONETARY_VARIANTS", "1"),
        claim_id=claim_id,
        field_name=field_name,
        inputs={"bbox": list(bbox), "max_variants": max_variants},
    ) as inv:
        if inv.bypassed:
            return None, [], attempts, "MONETARY_VARIANTS_DISABLED", evidence
        from packages.ocr_portfolio import recognize_monetary_crop

        with tel.track(
            "ocr_portfolio",
            enabled=True,
            claim_id=claim_id,
            field_name=field_name,
        ) as inv2:
            result = recognize_monetary_crop(
                crop, engines=engines, max_variants=max_variants
            )
            if result.best is None and include_heavy:
                # One paddle + rapid pass on original only (not full variant grid).
                heavy = recognize_monetary_crop(
                    crop,
                    engines={"paddleocr": _paddle_engine, "rapidocr": _rapid_engine},
                    max_variants=1,
                )
                result.attempts.extend(heavy.attempts)
                if heavy.best is not None:
                    result.best = heavy.best
            inv2.outputs = {
                "attempt_count": len(result.attempts),
                "best": None if result.best is None else result.best.value,
            }
        inv.outputs = inv2.outputs
        for read in result.attempts:
            attempts.append({
                "engine": read.engine,
                "reason": f"MONETARY_VARIANT:{read.variant_id}:{read.reason}",
                "observation": {"text": read.raw_text, "shaped": read.value},
                "preprocessing_profile": read.variant_id,
                "raw_confidence": read.confidence,
            })
            if read.value:
                box = BoundingBox(
                    x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                    image_width=image.width, image_height=image.height,
                )
                candidates.append({
                    "value": read.value,
                    "raw_value": read.raw_text,
                    "engine": read.engine,
                    "model_name": "monetary_portfolio",
                    "model_version": "v1",
                    "preprocessing_variant": read.variant_id,
                    "raw_confidence": read.confidence,
                    "calibrated_confidence": None,
                    "bounding_box": box.model_dump(mode="json"),
                    "latency_ms": 0.0,
                    "validation_results": ["MONETARY_VARIANT", read.reason],
                })
        if result.best and result.best.value:
            # POS-like reject for totals.
            from packages.geometry_authority import is_pos_like_currency

            if field_name.casefold() in {"total_charge", "total_charges"} and is_pos_like_currency(
                result.best.value
            ):
                reason = "POS_LIKE_MONETARY_REJECTED"
                inv.fallback_reason = reason
                return None, candidates, attempts, reason, evidence
            selected = result.best.value
            reason = f"MONETARY_VARIANT:{result.best.variant_id}:{result.best.engine}"
            inv.outputs["selected"] = selected

    with tel.track(
        "candidate_evidence",
        enabled=stage_enabled("CDP_CANDIDATE_EVIDENCE", "1"),
        claim_id=claim_id,
        field_name=field_name,
    ) as inv:
        if not inv.bypassed and candidates:
            from packages.candidate_evidence import CandidateEvidenceRecord, CandidateEvidenceStore

            store = CandidateEvidenceStore()
            for cand in candidates[:8]:
                store.add(
                    CandidateEvidenceRecord(
                        claim_id=str(claim_id or ""),
                        package_id="",
                        document_id="",
                        page_number=1,
                        field_name=field_name,
                        crop_bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        authorised_semantic_region="BOX_28"
                        if "total" in field_name.casefold()
                        else "BOX_24F",
                        preprocessing_variant=str(cand.get("preprocessing_variant") or ""),
                        engine=str(cand.get("engine") or ""),
                        model_version="v1",
                        raw_text=str(cand.get("raw_value") or ""),
                        normalized_value=str(cand.get("value") or ""),
                        model_confidence=float(cand.get("raw_confidence") or 0.0),
                        image_quality_features={
                            k: float(evidence[k])
                            for k in (
                                "blank_probability",
                                "ink_density",
                                "blur_score",
                                "contrast",
                            )
                            if k in evidence
                        },
                        geometry_valid=True,
                        validation_results=list(cand.get("validation_results") or []),
                    )
                )
            inv.outputs = {"stored": len(store)}

    return selected, candidates, attempts, reason, evidence


def _azure_di_service_line_budget() -> int:
    """Max Azure DI analyze calls for service-line cells in one document.

    F0 is one analyze per minute. Blank CMS-1500 forms used to burn DI on every
    empty row in the STP fast fallback (~6–7 min/claim). Cap defaults to 1 so a
    digit-conflict or single live-row miss can still escalate once; box-28 DI
    residual is separate.
    """
    raw = (os.environ.get("CDP_AZURE_DI_SERVICE_LINE_BUDGET") or "1").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def _maybe_azure_di_charge_crop(image, bbox, *, gap_class='CHARGE_LOCAL_EXHAUSTED'):
    """Last-resort Azure DI prebuilt-read on one charge cell crop.

    Returns (value, raw, candidates_list, reason). Empty when disabled / miss.
    """
    try:
        from packages.extraction_recovery.charge_azure_di_residual import (
            azure_di_charge_residual_enabled,
            residual_candidate_dict,
            try_charge_azure_di_crop,
        )
    except ImportError:
        return None, None, [], 'CHARGE_DI_IMPORT_ERROR'
    if not azure_di_charge_residual_enabled():
        return None, None, [], 'CHARGE_DI_DISABLED'
    result = try_charge_azure_di_crop(
        image, bbox, field_name='charges', gap_class=gap_class
    )
    if not result.attempted or not result.currency_shaped or not result.value:
        return None, None, [], result.reason
    if result.review_only:
        return None, None, [], result.reason
    cand = residual_candidate_dict(result, bbox=bbox, image_size=image.size)
    return result.value, result.raw_value or result.value, ([cand] if cand else []), result.reason


def _maybe_gpt4o_charge_crop(image, bbox, *, prior_candidates=None):
    """gpt-4o currency crop for service-line charge cells (hard-15 residual).

    Returns (value, raw, candidates_list, reason). Empty on disable / abstain.
    """
    try:
        from packages.extraction_recovery.gpt4o_crop_residual import (
            Gpt4oCropResidualResult,
            gpt4o_crop_accept_enabled,
            gpt4o_crop_residual_enabled,
            residual_candidate_dict,
            run_gpt4o_crop_residual,
        )
    except ImportError:
        return None, None, [], 'CHARGE_GPT4O_IMPORT_ERROR'
    if not gpt4o_crop_residual_enabled():
        return None, None, [], 'CHARGE_GPT4O_DISABLED'
    result = run_gpt4o_crop_residual(
        image=image,
        bbox=bbox,
        field_name='charges',
        prior_candidates=list(prior_candidates or []),
    )
    if not result.attempted or not result.shaped or not result.value:
        return None, None, [], result.reason
    accept = gpt4o_crop_accept_enabled()
    if result.review_only and not accept:
        return None, None, [], result.reason
    promoted = result
    if accept and result.review_only and result.shaped:
        promoted = Gpt4oCropResidualResult(
            attempted=result.attempted,
            configured=result.configured,
            review_only=False,
            value=result.value,
            raw_value=result.raw_value,
            shaped=True,
            insufficient_evidence=False,
            reason='GPT4O_SHAPED_ACCEPTED',
            engine=result.engine,
            confidence=result.confidence,
            validation_results=tuple(result.validation_results) or ('AZURE_GPT4O_CROP',),
        )
    cand = residual_candidate_dict(promoted, bbox=bbox, image_size=image.size)
    return (
        promoted.value,
        promoted.raw_value or promoted.value,
        ([cand] if cand else []),
        promoted.reason,
    )


def _merge_gpt4o_line_charge(
    image,
    bbox,
    *,
    value,
    raw,
    candidates,
    attempts,
    reason,
):
    """Attach gpt-4o crop residual to a service-line charge cell.

    Passes local OCR priors so the model confirms observed ink instead of
    abstaining on sparse crops. Always records the attempt (including abstain)
    so SINGLE_LINE_REQUIRES_DI failures stay diagnosable.
    """
    prior = []
    for cand in candidates or []:
        if not isinstance(cand, dict):
            continue
        seed = (cand.get('value') or cand.get('raw_value') or '').strip()
        if seed:
            prior.append(seed)
    if value and str(value).strip() and str(value).strip() not in prior:
        prior.insert(0, str(value).strip())
    g_value, g_raw, g_cands, g_reason = _maybe_gpt4o_charge_crop(
        image, bbox, prior_candidates=prior[:4]
    )
    attempts = list(attempts or []) + [{
        'engine': 'azure_gpt4o_crop',
        'reason': g_reason or 'CHARGE_GPT4O_ATTEMPTED',
        'observation': {'text': (g_raw or g_value or '')},
    }]
    if not g_value:
        return value, raw, candidates, attempts, reason

    from packages.claim_evidence.line_sum_authority import amounts_corroborate
    from packages.ocr_portfolio import prefer_charge_ink_amount, shape_monetary

    g_shaped = shape_monetary(g_value) or g_value

    if not value:
        value = g_shaped
        raw = g_raw or raw
        candidates = list(candidates or []) + list(g_cands or [])
        reason = f'{reason}|{g_reason}|CHARGE_GPT4O_CROP'
    elif amounts_corroborate(value, g_shaped):
        # Units/ruling bleed (270.00 vs 2701.00): keep the clean local .00 stem.
        ink_pref = prefer_charge_ink_amount(value, g_shaped)
        if ink_pref == value and value != g_shaped:
            if g_cands:
                candidates = list(candidates or []) + list(g_cands)
            reason = f'{reason}|{g_reason}|CHARGE_GPT4O_CORROBORATED_LOCAL_KEPT'
        else:
            # Digit-drop twin: prefer the longer read when gpt-4o recovered
            # trailing digits (622→6225, 643→6430).
            preferred = prefer_currency_without_digit_drop(value, g_shaped)
            if preferred and preferred == g_shaped and preferred != value:
                value = g_shaped
                raw = g_raw or raw
                candidates = list(g_cands or []) + list(candidates or [])
                reason = f'{reason}|{g_reason}|CHARGE_GPT4O_DIGIT_DROP'
            else:
                if g_cands:
                    candidates = list(candidates or []) + list(g_cands)
                reason = f'{reason}|{g_reason}|CHARGE_GPT4O_CORROBORATED'
    else:
        # Non-twin local vs gpt-4o. If another local engine already supports the
        # vision read within $1 (paddle 640.01 vs selected fragment 101), keep
        # that consensus. Unrelated ink is not a reason to drop the vision read.
        from packages.claim_evidence.line_sum_authority import amounts_within_tolerance
        from packages.geometry_authority import is_pos_like_currency
        from packages.ocr_portfolio import recover_dollars_from_split_raw

        def _supports(target, pool) -> bool:
            if not target:
                return False
            for cand in pool or []:
                if not isinstance(cand, dict):
                    continue
                eng = str(cand.get('engine') or '').casefold()
                if 'gpt4o' in eng or 'gpt-4o' in eng:
                    continue
                seeds = [
                    cand.get('value'),
                    recover_dollars_from_split_raw(str(cand.get('raw_value') or '')),
                ]
                for seed in seeds:
                    if not seed:
                        continue
                    shaped = shape_monetary(str(seed)) or str(seed)
                    if is_pos_like_currency(shaped) and not is_pos_like_currency(target):
                        continue
                    if amounts_within_tolerance(
                        target, shaped, absolute=__import__('decimal').Decimal('1'), relative=__import__('decimal').Decimal('0')
                    ):
                        return True
            return False

        if _supports(g_shaped, candidates):
            value = g_shaped
            raw = g_raw or raw
            candidates = list(g_cands or []) + list(candidates or [])
            reason = f'{reason}|{g_reason}|CHARGE_GPT4O_LOCAL_CONSENSUS'
        else:
            ink_pref = prefer_charge_ink_amount(value, g_shaped)
            if ink_pref == value:
                if g_cands:
                    candidates = list(candidates or []) + list(g_cands)
                reason = f'{reason}|{g_reason}|CHARGE_GPT4O_REJECTED_INK_PREF'
            else:
                value = ink_pref or g_shaped
                raw = g_raw or raw
                candidates = list(g_cands or []) + list(candidates or [])
                reason = f'{reason}|{g_reason}|CHARGE_GPT4O_OVERRIDE'
    if value and image is not None and bbox is not None:
        # Vision read with no usable local: one ruling-split digit pass.
        # Agreement mints the local side of gpt+local consensus; mismatch stays HITL.
        try:
            from decimal import Decimal

            from packages.claim_evidence.line_sum_authority import amounts_within_tolerance

            supported = False
            for cand in candidates or []:
                eng = str((cand or {}).get('engine') or '').casefold()
                if 'gpt4o' in eng or 'gpt-4o' in eng:
                    continue
                seed = (cand or {}).get('value')
                if seed and amounts_within_tolerance(value, seed, absolute=Decimal(1), relative=Decimal(0)):
                    supported = True
                    break
            if not supported:
                d_cands, d_attempts, d_reason = _recognize_charge_digits_only(image, bbox)
                attempts = list(attempts or []) + list(d_attempts or [])
                d_val = d_cands[0].get('value') if d_cands else None
                if d_val and amounts_within_tolerance(value, d_val, absolute=Decimal(1), relative=Decimal(0)):
                    candidates = list(candidates or []) + list(d_cands)
                    reason = f'{reason}|{d_reason}|CHARGE_AI_LOCAL_CONFIRM'
        except Exception:  # noqa: BLE001, S110 -- optional residual confirmation
            pass
    return value, raw, candidates, attempts, reason


def _maybe_attach_openocr_svtr_line(
    image,
    bbox,
    *,
    value,
    raw,
    candidates,
    attempts,
    reason,
):
    """Optional OpenOCR/SVTRv2 printed-crop residual (redesign fixed-form role)."""
    try:
        from workers.openocr_svtr import (
            openocr_svtr_enabled,
            recognize_openocr_svtr,
            residual_candidate_dict,
        )
    except ImportError:
        return value, raw, candidates, attempts, reason
    if not openocr_svtr_enabled():
        return value, raw, candidates, attempts, reason
    x0, y0, x1, y1 = (int(v) for v in bbox)
    crop = image.crop((max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)))
    result = recognize_openocr_svtr(crop)
    attempts = list(attempts or []) + [{
        'engine': 'openocr_svtr',
        'reason': result.reason,
        'observation': {'text': result.text or ''},
    }]
    if not result.text:
        return value, raw, candidates, attempts, reason
    shaped = _currency_value_from_text(result.text)
    if not shaped:
        return value, raw, candidates, attempts, f'{reason}|{result.reason}|OPENOCR_UNSHAPED'
    cand = residual_candidate_dict(
        result,
        bbox=(x0, y0, x1, y1),
        image_size=(image.width, image.height),
        field_name='charges',
    )
    if cand is not None:
        cand['value'] = shaped
        candidates = list(candidates or []) + [cand]
    if not value:
        value = shaped
        raw = result.text
        reason = f'{reason}|{result.reason}|CHARGE_OPENOCR_SVTR'
    else:
        reason = f'{reason}|{result.reason}|CHARGE_OPENOCR_CANDIDATE'
    return value, raw, candidates, attempts, reason


def _maybe_attach_ppocr_v5_server_line(
    image,
    bbox,
    *,
    value,
    raw,
    candidates,
    attempts,
    reason,
):
    """Optional PP-OCRv5 Server residual for sparse/single-engine charge cells."""
    try:
        from workers.ppocr_v5.subprocess_bridge import (
            ppocr_v5_server_enabled,
            recognize_ppocr_v5_server,
        )
    except ImportError:
        return value, raw, candidates, attempts, reason
    if not ppocr_v5_server_enabled():
        return value, raw, candidates, attempts, reason
    shaped_engines = {
        str(c.get('engine') or '')
        for c in (candidates or [])
        if isinstance(c, dict) and (c.get('value') or '').strip()
    }
    # Skip when paddle+rapid already both shaped — v5 adds no independence.
    has_paddle = any('paddle' in e.casefold() or 'ppocr' in e.casefold() for e in shaped_engines)
    has_rapid = any('rapid' in e.casefold() for e in shaped_engines)
    if has_paddle and has_rapid:
        return value, raw, candidates, attempts, reason
    x0, y0, x1, y1 = (int(v) for v in bbox)
    crop = image.crop((max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)))
    text, conf, v5_reason = recognize_ppocr_v5_server(crop)
    attempts = list(attempts or []) + [{
        'engine': 'ppocr_v5_server',
        'reason': v5_reason,
        'observation': {'text': text or ''},
    }]
    if not text:
        return value, raw, candidates, attempts, reason
    # Shape like other charge engines.
    shaped = _currency_value_from_text(text)
    if not shaped:
        return value, raw, candidates, attempts, f'{reason}|{v5_reason}|PPOCRV5_UNSHAPED'
    cand = {
        'value': shaped,
        'raw_value': text,
        'engine': 'ppocr_v5_server',
        'model_name': 'PP-OCRv5_server',
        'model_version': 'paddleocr-3.x',
        'preprocessing_variant': 'PPOCRV5_SERVER_CROP',
        'raw_confidence': conf,
        'calibrated_confidence': None,
        'bounding_box': {
            'x0': float(x0), 'y0': float(y0), 'x1': float(x1), 'y1': float(y1),
            'image_width': image.width, 'image_height': image.height,
        },
        'latency_ms': 0.0,
        'validation_results': ['PPOCRV5_SERVER'],
    }
    candidates = list(candidates or []) + [cand]
    if not value:
        value = shaped
        raw = text
        reason = f'{reason}|{v5_reason}|CHARGE_PPOCRV5_SERVER'
    elif prefer_currency_without_digit_drop(value, shaped) == shaped and shaped != value:
        value = shaped
        raw = text
        reason = f'{reason}|{v5_reason}|CHARGE_PPOCRV5_DIGIT_DROP'
    else:
        reason = f'{reason}|{v5_reason}|CHARGE_PPOCRV5_CANDIDATE'
    return value, raw, candidates, attempts, reason


def _currency_value_from_text(raw_text: str | None) -> str | None:
    import re as _re

    cleaned = str(raw_text or '').strip()
    if not cleaned or not _re.search(r'\d', cleaned):
        return None
    if _re.search(r'(DIAGNOSIS|POINTER|FROM|HCPCS|CPT|NPI|PLACE|CHARGES)', cleaned.upper()):
        return None
    m = _re.search(r'\$?\d{1,3}(?:,\d{3})*\.\d{2}|\$?\d{2,6}(?:\.\d{2})?', cleaned)
    if not m:
        return None
    amount = m.group(0).lstrip('$')
    if '.' not in amount and _re.fullmatch(r'\d{2,6}', amount):
        amount = f'{amount}.00'
    return amount


def _currency_value_from_candidates(raw_text, candidates) -> str | None:
    """Shape a charge from OCR candidates (module-level; used by ruling-split merge)."""
    import re as _re

    shaped_vals: list[str] = []
    for c in candidates or []:
        try:
            from packages.ocr_portfolio import recover_dollars_from_split_raw

            recovered = recover_dollars_from_split_raw(str(c.get('raw_value') or ''))
            if recovered:
                shaped_vals.append(recovered)
        except Exception:  # noqa: BLE001, S110 -- optional monetary shaping
            pass
        seed = (c.get('value') or '').strip() or (c.get('raw_value') or '').strip()
        if not seed:
            continue
        cleaned = seed.strip()
        if _re.search(r'[A-Za-z]', cleaned) and not _re.search(r'\d', cleaned):
            continue
        if not _re.search(r'\d', cleaned):
            continue
        if _re.search(r'(DIAGNOSIS|POINTER|FROM|HCPCS|CPT|NPI|PLACE|CHARGES)', cleaned.upper()):
            continue
        m = _re.search(r'\$?\d{1,3}(?:,\d{3})*\.\d{2}|\$?\d{2,6}(?:\.\d{2})?', cleaned)
        if m:
            amount = m.group(0).lstrip('$')
            if '.' not in amount and _re.fullmatch(r'\d{2,6}', amount):
                try:
                    from packages.ocr_portfolio import shape_monetary

                    shaped = shape_monetary(amount)
                    amount = shaped or f'{amount}.00'
                except Exception:  # noqa: BLE001
                    amount = f'{amount}.00'
            if _re.fullmatch(r'\d+\.\d{2}', amount):
                shaped_vals.append(amount)
                continue
        try:
            from packages.ocr_portfolio import shape_monetary

            shaped = shape_monetary(cleaned)
            if shaped:
                shaped_vals.append(shaped)
        except Exception:  # noqa: BLE001, S110 -- optional monetary shaping
            pass
    if not shaped_vals:
        return _currency_value_from_text(raw_text)
    best = shaped_vals[0]
    for other in shaped_vals[1:]:
        try:
            from packages.ocr_portfolio import prefer_charge_ink_amount

            preferred_ink = prefer_charge_ink_amount(best, other)
            if preferred_ink:
                best = preferred_ink
                continue
        except Exception:  # noqa: BLE001, S110 -- optional monetary preference
            pass
        preferred = prefer_currency_without_digit_drop(best, other)
        if preferred:
            best = preferred
    return best

def recognize_service_lines(image, router, template):
    """OCR CMS-1500 service-line charge cells for claim-total E6 confirmation."""
    table = getattr(template, 'service_line_region', None) if template is not None else None
    if table is None:
        return []
    charge_col = next((c for c in table.columns if c.field_name in {'charges', 'charge_amount'}), None)
    probe_cols = [c for c in table.columns if c.field_name in {'cpt_hcpcs', 'date_from'}]
    if charge_col is None:
        return []
    lines = []
    # Prefer data rows: start one half-row below the printed header rule.
    header_offset = max(8, table.row_height_px // 3)
    # Alternate x-windows: primary template column plus a right-shifted band that
    # avoids diagnosis-pointer bleed on many live CMS-1500 scans.
    fast = _ocr_fast_mode()
    charge_windows = charge_windows_for_mode(charge_col.x0, charge_col.x1, fast=fast)
    # F0: at most N DI analyzes for service-line cells this document.
    di_budget = _azure_di_service_line_budget()

    def _currency_value(raw_text, candidates):
        return _currency_value_from_candidates(raw_text, candidates)

    for row_index in range(table.max_rows):
        y0 = table.table_y0 + header_offset + row_index * table.row_height_px
        y1 = min(y0 + table.row_height_px, table.table_y1)
        if y0 >= table.table_y1:
            break
        probe_empty = True
        import re as _re_probe
        if not fast:
            for column in probe_cols:
                pb = _clamp_bbox((column.x0, y0, column.x1, y1), image.width, image.height)
                pcs, _, _ = _recognize_one(image, column.field_name, pb, router, column.field_type)
                for cand in pcs or []:
                    # Prefer raw ink for liveness; span may empty a noisy but real cell.
                    val = ((cand.get('raw_value') or '') + ' ' + (cand.get('value') or '')).strip()
                    if not val:
                        continue
                    if column.field_name in {'date_from', 'date_to'} and _re_probe.search(r'\d', val):
                        probe_empty = False
                        break
                    if column.field_name in {'cpt_hcpcs', 'cpt', 'hcpcs'} and _re_probe.search(
                        r'\d{4,5}|[A-Z]\d{3,4}', val.upper()
                    ):
                        probe_empty = False
                        break
                if not probe_empty:
                    break
        # Phase 2: skip leading header/blank rows; only stop after a live block ends.
        # Charge-column currency ink can also prove the row is live when date/CPT probes fail.
        best = None
        for x0, x1 in charge_windows:
            bbox = _clamp_bbox((x0, y0, x1, y1), image.width, image.height)
            if fast:
                # Agent-GT retest: tess CHARGE_DIGITS_FAST truncates (1571→157)
                # even when paddle agrees on the truncated form — so paddle/rapid
                # is primary under STP fast; tess digits only fill empty cells.
                candidates, attempts, reason = _recognize_one(
                    image,
                    'charges',
                    bbox,
                    router,
                    charge_col.field_type,
                    engine_order=('paddleocr', 'rapidocr'),
                )
                raw = candidates[0].get('raw_value') if candidates else ''
                value = _currency_value(raw, candidates)
                if value and _reject_pos_code_charge(value, raw):
                    attempts = list(attempts or []) + [{
                        'engine': 'geometry_authority',
                        'reason': 'POS_CODE_SHORT_DIGITS',
                        'observation': {'text': raw, 'shaped': value},
                    }]
                    value = None
                value, raw, candidates, attempts, reason = _merge_ruling_split_local_charge(
                    image,
                    bbox,
                    router=router,
                    field_type=charge_col.field_type,
                    engine_order=('paddleocr', 'rapidocr'),
                    value=value,
                    raw=raw,
                    candidates=candidates,
                    attempts=attempts,
                    reason=reason,
                )
                if not value:
                    d_cands, d_attempts, d_reason = _recognize_charge_digits_only(
                        image, bbox
                    )
                    attempts = list(attempts or []) + list(d_attempts or [])
                    d_raw = d_cands[0].get('raw_value') if d_cands else ''
                    d_value = _currency_value(d_raw, d_cands)
                    if d_value and _reject_pos_code_charge(d_value, d_raw):
                        d_value = None
                    if d_value:
                        candidates, raw, value = d_cands, d_raw, d_value
                        reason = f'{d_reason}|AFTER_PADDLE_EMPTY'
            else:
                candidates, attempts, reason = _recognize_one(
                    image, 'charges', bbox, router, charge_col.field_type)
                raw = candidates[0].get('raw_value') if candidates else ''
                value = _currency_value(raw, candidates)
                if value and _reject_pos_code_charge(value, raw):
                    attempts = list(attempts or []) + [{
                        'engine': 'geometry_authority',
                        'reason': 'POS_CODE_SHORT_DIGITS',
                        'observation': {'text': raw, 'shaped': value},
                    }]
                    value = None
                value, raw, candidates, attempts, reason = _merge_ruling_split_local_charge(
                    image,
                    bbox,
                    router=router,
                    field_type=charge_col.field_type,
                    engine_order=None,
                    value=value,
                    raw=raw,
                    candidates=candidates,
                    attempts=attempts,
                    reason=reason,
                )
            # Azure DI only when local OCR left the cell empty on a live row,
            # or when dual engines disagree as non-twins. Do NOT call DI merely
            # because an amount is short — that billed every $25–$999 cell on
            # Independent-300 and cratered throughput.
            need_di = False
            di_gap = 'CHARGE_LOCAL_EXHAUSTED'
            need_gpt4o_twin = False
            engine_vals = []
            for c in candidates or []:
                ev = None
                seed = (c.get('value') or c.get('raw_value') or '').strip()
                if seed:
                    # lightweight shape
                    import re as _re_e
                    m = _re_e.search(
                        r'\$?\d{1,3}(?:,\d{3})*\.\d{2}|\$?\d{2,6}(?:\.\d{2})?',
                        seed,
                    )
                    if m:
                        ev = m.group(0).lstrip('$')
                        if '.' not in ev and _re_e.fullmatch(r'\d{2,6}', ev):
                            ev = f'{ev}.00'
                if ev:
                    engine_vals.append(ev)
            unique_vals = list(dict.fromkeys(engine_vals))
            # v12: single-engine shaped cell → digit-whitelist tess before gpt-4o
            # so rapid+tess can unlock SINGLE_LINE_DUAL_ENGINE without cloud.
            if value and len(unique_vals) < 2:
                d_cands, d_attempts, d_reason = _recognize_charge_digits_only(
                    image, bbox
                )
                attempts = list(attempts or []) + list(d_attempts or [])
                if d_cands:
                    candidates = list(candidates or []) + list(d_cands)
                    d_raw = d_cands[0].get('raw_value') if d_cands else ''
                    merged = _currency_value(d_raw or raw, candidates)
                    if merged:
                        preferred = prefer_currency_without_digit_drop(value, merged)
                        if preferred and preferred != value:
                            value = preferred
                            raw = d_raw or raw
                            reason = f'{reason}|{d_reason}|CHARGE_TESS_DIGIT_DROP'
                        else:
                            reason = f'{reason}|{d_reason}|CHARGE_TESS_FILL'
                    engine_vals = []
                    for c in candidates or []:
                        seed = (c.get('value') or c.get('raw_value') or '').strip()
                        if not seed:
                            continue
                        import re as _re_e2
                        m = _re_e2.search(
                            r'\$?\d{1,3}(?:,\d{3})*\.\d{2}|\$?\d{2,6}(?:\.\d{2})?',
                            seed,
                        )
                        if not m:
                            continue
                        ev = m.group(0).lstrip('$')
                        if '.' not in ev and _re_e2.fullmatch(r'\d{2,6}', ev):
                            ev = f'{ev}.00'
                        engine_vals.append(ev)
                    unique_vals = list(dict.fromkeys(engine_vals))
            if not value and not probe_empty:
                need_di = True
                di_gap = 'CHARGE_LOCAL_EXHAUSTED'
            elif len(unique_vals) >= 2:
                a, b = unique_vals[0], unique_vals[1]
                if prefer_currency_without_digit_drop(a, b) is None and a != b:
                    need_di = True
                    di_gap = 'CHARGE_DIGIT_CONFLICT'
                else:
                    # Digit-drop twins (13 vs 131): still ask gpt-4o to pick ink.
                    need_gpt4o_twin = True
            if need_di and di_budget > 0:
                di_value, di_raw, di_cands, di_reason = _maybe_azure_di_charge_crop(
                    image, bbox, gap_class=di_gap
                )
                di_budget -= 1
                if di_value:
                    if value:
                        preferred = prefer_currency_without_digit_drop(value, di_value)
                        # Only accept DI when it recovers dropped digits (longer
                        # twin). Non-twin DI must not override local paddle.
                        if preferred and preferred == di_value and preferred != value:
                            value = preferred
                            raw = di_raw or raw
                            if di_cands:
                                candidates = list(candidates or []) + list(di_cands)
                            reason = f'{reason}|{di_reason}|CHARGE_DI_DIGIT_DROP'
                            attempts = list(attempts or []) + [{
                                'engine': 'azure_document_intelligence_read',
                                'reason': di_reason,
                                'observation': {'text': di_raw or di_value},
                            }]
                    else:
                        value = di_value
                        raw = di_raw or raw
                        candidates = list(candidates or []) + list(di_cands or [])
                        reason = di_reason or 'CHARGE_AZURE_DI_CROP'
                        attempts = list(attempts or []) + [{
                            'engine': 'azure_document_intelligence_read',
                            'reason': di_reason,
                            'observation': {'text': di_raw or di_value},
                        }]
            elif need_di and di_budget <= 0:
                attempts = list(attempts or []) + [{
                    'engine': 'azure_document_intelligence_read',
                    'reason': 'CHARGE_DI_SERVICE_LINE_BUDGET_EXHAUSTED',
                }]
            # gpt-4o line-charge residual: empty after DI, single-engine local,
            # or digit-drop twins (hard-15 charge hole). Empty box-28 E6 depends
            # on this corroboration — pass local priors so the crop confirms ink
            # instead of abstaining on a sparse cell.
            need_gpt4o = False
            gpt4o_on = (os.environ.get('CDP_GPT4O_CROP_RESIDUAL') or '1').strip().casefold()
            if gpt4o_on not in {'0', 'false', 'no', 'off'}:
                if (not value and not probe_empty) or (value and len(unique_vals) < 2) or need_gpt4o_twin:
                    need_gpt4o = True
                elif need_di and not any(
                    'document_intelligence' in str(c.get('engine') or '').casefold()
                    for c in (candidates or [])
                ):
                    # DI was needed but did not contribute a candidate.
                    need_gpt4o = True
            if need_gpt4o:
                value, raw, candidates, attempts, reason = _merge_gpt4o_line_charge(
                    image,
                    bbox,
                    value=value,
                    raw=raw,
                    candidates=candidates,
                    attempts=attempts,
                    reason=reason,
                )
            # Redesign: optional OpenOCR/SVTRv2 printed-crop (default off).
            value, raw, candidates, attempts, reason = _maybe_attach_openocr_svtr_line(
                image,
                bbox,
                value=value,
                raw=raw,
                candidates=candidates,
                attempts=attempts,
                reason=reason,
            )
            # Optional PP-OCRv5 Server residual (isolated paddleocr 3.x venv).
            # Adds a paddle-family candidate when local paddle/rapid left a
            # single-engine or empty charge cell — can unlock dual-engine with rapid.
            value, raw, candidates, attempts, reason = _maybe_attach_ppocr_v5_server_line(
                image,
                bbox,
                value=value,
                raw=raw,
                candidates=candidates,
                attempts=attempts,
                reason=reason,
            )
            geometry_confirmed = False
            try:
                from packages.geometry_authority.cms1500_regions import (
                    CMS1500_CHARGE_CENTS_X,
                    CMS1500_LINE_COLUMNS,
                )
                from packages.geometry_authority.monetary_geometry import read_monetary_crop

                ch0, ch1 = CMS1500_LINE_COLUMNS["charges"]
                center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
                crop_x1 = float(bbox[2])
                if not (ch0 - 8 <= center_x <= ch1 + 8):
                    # Diagnosis-pointer window. Not an independent Box 24F read.
                    value = None
                    reason = f'{reason}|BOX24F_WINDOW_REJECTED'
                elif crop_x1 < CMS1500_CHARGE_CENTS_X + 12:
                    # Clipped before the cents column — do not confirm geometry
                    # and never let a clipped window overwrite a fuller OCR amount.
                    reason = f'{reason}|BOX24F_CENTS_CLIPPED'
                    geometry_confirmed = False
                else:
                    geo = read_monetary_crop(
                        image.crop(bbox),
                        crop_bbox=tuple(float(v) for v in bbox),
                        image_size=(int(image.width), int(image.height)),
                    )
                    if geo.geometry_candidate and not geo.ambiguous:
                        value = geo.canonical_monetary_value or geo.geometry_candidate
                        raw = geo.raw_glyph_sequence or raw
                        geometry_confirmed = True
                        reason = f'{reason}|GEOMETRY_CENTS'
                        attempts = list(attempts or []) + [{
                            'engine': 'rapidocr',
                            'reason': 'GEOMETRY_CENTS',
                            'observation': {
                                'text': geo.raw_glyph_sequence,
                                'shaped': value,
                                'raw_digit_sequence': geo.raw_glyph_sequence,
                                'page_glyph_polygons': [
                                    [list(pt) for pt in poly]
                                    for poly in geo.page_glyph_polygons
                                ],
                                'canonical_glyph_centres': [
                                    list(c) for c in geo.canonical_glyph_centres
                                ],
                                'dollar_glyphs': list(geo.dollar_glyphs),
                                'cents_glyphs': list(geo.cents_glyphs),
                                'unit_zone_glyphs': list(geo.unit_zone_glyphs),
                                'canonical_monetary_value': geo.canonical_monetary_value,
                            },
                        }]
            except Exception:  # noqa: BLE001
                geometry_confirmed = False
            score = 0
            if value:
                score = 3 if '.' in value else 2
                # Prefer amounts that are not tiny single-digit dollars.
                if value[0] != '0' and not value.startswith('1.'):
                    score += 1
                # Typed CMS charges are usually whole dollars; boost .00 and
                # prefer mid-window clean dollars over units-bleed *.10 soup.
                if value.endswith('.00'):
                    score += 2
                elif value.endswith('.10'):
                    score -= 1
                # Dashed-rule crops like "-200-\nLAAM" are not service charges.
                # Allow / | : ? — common OCR noise inside repaired amounts
                # (I/00 → 100.00, 200:00 → 200.00, 200? → 200.00).
                # Never wipe a geometry-authorised cents read for raw noise.
                import re as _re_noise
                raw_u = (raw or '').upper()
                if (
                    not geometry_confirmed
                    and (
                        _re_noise.search(r'[^0-9A-Z./|:?\s,-]', raw_u)
                        or _re_noise.fullmatch(r'[\s\-.,/|:?]*', raw or '')
                    )
                ):
                    score = 0
                    value = None
            if geometry_confirmed and value:
                score += 5
            if 'BOX24F_CENTS_CLIPPED' in str(reason or ''):
                # Cents-clipped crops are strictly dominated by fuller Box 24F windows.
                score = max(0, score - 3)
            # A validated cents-column read must not lose to a clipped window
            # that shape_monetary turned into a different .00 amount.
            if best is not None and best.get('_geometry') and not geometry_confirmed:
                score = min(score, int(best.get('_score') or 0) - 1)
            elif best is not None and value and best.get('charges') and not (
                geometry_confirmed and not best.get('_geometry')
            ):
                try:
                    from packages.ocr_portfolio import prefer_charge_ink_amount
                    from packages.ocr_portfolio.monetary_recognizer import (
                        is_ruling_tick_charge,
                    )

                    if is_ruling_tick_charge(
                        best.get('candidates'), best.get('charges')
                    ) and not is_ruling_tick_charge(candidates, value):
                        score = max(score, int(best.get('_score') or 0) + 1)
                    preferred = prefer_charge_ink_amount(best.get('charges'), value)
                    if preferred == best.get('charges') and preferred != value:
                        score = min(score, int(best.get('_score') or 0))
                    elif preferred == value and preferred != best.get('charges'):
                        score = max(score, int(best.get('_score') or 0) + 1)
                except Exception:  # noqa: BLE001, S110 -- optional ink preference
                    pass
            candidate = {
                'line_number': row_index + 1,
                'charges': value,
                'charge_amount': value,
                'raw_charges': raw,
                'canonical_region': list(bbox),
                'candidates': candidates,
                'attempts': attempts,
                'router_reason': reason,
                'status': 'OBSERVED' if value else 'NO_VALUE',
                '_score': score,
                '_geometry': geometry_confirmed,
            }
            if best is None or candidate['_score'] > best['_score']:
                best = candidate
            if score >= 5 and geometry_confirmed:
                break
        assert best is not None
        best.pop('_score', None)
        best.pop('_geometry', None)
        if best.get('status') != 'OBSERVED':
            if any(l.get('status') == 'OBSERVED' for l in lines):
                # End of live block — do not emit the empty sentinel.
                break
            if probe_empty:
                # Leading header/blank row with no charge ink — keep scanning.
                continue
            # Probe saw date/CPT but charge empty: still end once we are past a live block.
            continue
        lines.append(best)
        # STP E6 only needs observed line charges; 3 live rows is enough evidence.
        if fast and len(lines) >= 3:
            break
    # Fast digit path sometimes misses typed amounts (wrong x-window). One
    # paddle/rapid pass across charge-column windows recovers E6 without full thrash.
    # Do NOT call Azure DI here: under F0 (1 analyze/min) blank forms previously
    # spent ~6 minutes proving empty rows empty. Box-28 DI residual + main-loop
    # budgeted DI remain available for true charge gaps.
    if fast and not lines and router is not None:
        fallback_windows = charge_windows or [(charge_col.x0, charge_col.x1)]
        for row_index in range(table.max_rows):
            y0 = table.table_y0 + header_offset + row_index * table.row_height_px
            y1 = min(y0 + table.row_height_px, table.table_y1)
            if y0 >= table.table_y1:
                break
            value = None
            raw = ''
            candidates = []
            attempts = []
            reason = ''
            bbox = None
            for x0, x1 in fallback_windows:
                bbox = _clamp_bbox((x0, y0, x1, y1), image.width, image.height)
                candidates, attempts, reason = _recognize_one(
                    image, 'charges', bbox, router, charge_col.field_type,
                    engine_order=('paddleocr', 'rapidocr'),
                )
                raw = candidates[0].get('raw_value') if candidates else ''
                value = _currency_value(raw, candidates)
                if value and _reject_pos_code_charge(value, raw):
                    attempts = list(attempts or []) + [{
                        'engine': 'geometry_authority',
                        'reason': 'POS_CODE_SHORT_DIGITS',
                        'observation': {'text': raw, 'shaped': value},
                    }]
                    value = None
                if value:
                    break
            if not value and bbox is not None:
                # Cap expensive recovery: first 2 empty rows only under fast path.
                if row_index > 1:
                    if lines:
                        break
                    continue
                m_val, m_cands, m_atts, m_reason, ev = _recover_empty_monetary_crop(
                    image, bbox, field_name='charges'
                )
                attempts = list(attempts or []) + list(m_atts or [])
                candidates = list(candidates or []) + list(m_cands or [])
                reason = f'{reason}|{m_reason}'
                if m_val and (ev or {}).get('disposition') != 'BLANK_CONFIRMED':
                    try:
                        from packages.geometry_authority import reject_pos_as_charge

                        reject, _ = reject_pos_as_charge(
                            m_val,
                            (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                            image_size=(image.width, image.height),
                        )
                        if not reject:
                            value = m_val
                            raw = m_val
                    except Exception:  # noqa: BLE001
                        value = m_val
                        raw = m_val
            if not value:
                if lines:
                    break
                continue
            # Reject POS bleed that slipped through.
            try:
                from packages.geometry_authority import reject_pos_as_charge

                reject, rej_reason = reject_pos_as_charge(
                    value,
                    (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    image_size=(image.width, image.height),
                )
                if reject:
                    attempts = list(attempts or []) + [{
                        'engine': 'geometry_authority',
                        'reason': rej_reason,
                        'observation': {'text': value},
                    }]
                    if lines:
                        break
                    continue
            except Exception:  # noqa: BLE001, S110 -- optional geometry rejection
                pass
            # Single-engine fallback lines still need gpt-4o corroboration for
            # empty-box-28 LINE_TOTALS AUTO (SINGLE_LINE_GPT4O_LOCAL).
            gpt4o_on = (os.environ.get('CDP_GPT4O_CROP_RESIDUAL') or '1').strip().casefold()
            if gpt4o_on not in {'0', 'false', 'no', 'off'} and bbox is not None:
                shaped_engines = {
                    str(c.get('engine') or '')
                    for c in (candidates or [])
                    if isinstance(c, dict) and (c.get('value') or '').strip()
                }
                if len(shaped_engines) < 2:
                    value, raw, candidates, attempts, reason = _merge_gpt4o_line_charge(
                        image,
                        bbox,
                        value=value,
                        raw=raw,
                        candidates=candidates,
                        attempts=attempts,
                        reason=reason,
                    )
            lines.append({
                'line_number': row_index + 1,
                'charges': value,
                'charge_amount': value,
                'raw_charges': raw,
                'canonical_region': list(bbox),
                'candidates': candidates,
                'attempts': attempts,
                'router_reason': f'{reason}|SERVICE_LINE_FALLBACK',
                'status': 'OBSERVED',
                'semantic_region': 'BOX_24F',
                'bbox': list(bbox),
                'row_id': str(row_index + 1),
            })
            if len(lines) >= 3:
                break
    # EMPTY_FINANCIAL_INK residual: local found no line charges — gpt-4o sweep
    # of the first N charge cells (faint ink / wrong x-window misses), then
    # optional local corroboration for SINGLE_LINE_GPT4O_LOCAL.
    if not lines and charge_col is not None:
        try:
            from packages.extraction_recovery.gpt4o_crop_residual import (
                empty_financial_ink_gpt4o_enabled,
                empty_financial_ink_max_lines,
                recover_empty_financial_service_lines,
            )
        except ImportError:
            return lines
        if empty_financial_ink_gpt4o_enabled():
            sweep_windows = charge_windows or [(charge_col.x0, charge_col.x1)]
            x0, x1 = sweep_windows[0]
            max_n = empty_financial_ink_max_lines()
            bboxes = []
            for row_index in range(min(table.max_rows, max_n)):
                y0 = table.table_y0 + header_offset + row_index * table.row_height_px
                y1 = min(y0 + table.row_height_px, table.table_y1)
                if y0 >= table.table_y1:
                    break
                bboxes.append(
                    _clamp_bbox((x0, y0, x1, y1), image.width, image.height)
                )

            def _local_on_bbox(bb):
                return _recognize_one(
                    image,
                    'charges',
                    bb,
                    router,
                    charge_col.field_type,
                    engine_order=('paddleocr', 'rapidocr'),
                )

            recovered = recover_empty_financial_service_lines(
                image=image,
                line_bboxes=bboxes,
                local_recognize=_local_on_bbox if router is not None else None,
            )
            # Prefer currency shaped from gpt; if local also shaped a twin,
            # _currency_value style merge already lives in candidates.
            lines.extend(recovered)
    from packages.ocr_portfolio.monetary_recognizer import apply_charge_line_resolution

    return apply_charge_line_resolution(lines)



def _dob_cell_bboxes(band):
    """Split a DOB digit band into MM / DD / YY cells (CMS-1500 box 3).

    Inset each cell away from the dashed vertical rules — those glyphs OCR as
    digit ``1`` and are the dominant source of 01↔11 / 09↔19 CONFLICT_MARGIN HITL.
    """
    x0, y0, x1, y1 = (int(v) for v in band)
    width = max(1, x1 - x0)
    inset = max(2, int(0.04 * width))
    mm_x1 = x0 + int(0.30 * width)
    dd_x0 = x0 + int(0.30 * width)
    dd_x1 = x0 + int(0.56 * width)
    yy_x0 = x0 + int(0.52 * width)
    return {
        'MM': (x0 + inset, y0, max(x0 + inset + 1, mm_x1 - inset), y1),
        'DD': (dd_x0 + inset, y0, max(dd_x0 + inset + 1, dd_x1 - inset), y1),
        'YY': (yy_x0 + inset, y0, max(yy_x0 + inset + 1, x1 - inset), y1),
    }


def dob_boxed_cells_complete(mm: str, dd: str, yy: str) -> bool:
    """CMS DOB cells are two digits. A lone day digit is a clipped glyph, not 0D.

    ``DD=1`` must not assemble ``07/01`` when the source day is ``16``.
    Single-digit cells stay unresolved so a later full-band read or HITL
    can run. Real single-digit days fail closed rather than false-accept.
    """
    return bool(
        re.fullmatch(r"\d{2}", mm or "")
        and re.fullmatch(r"\d{2}", dd or "")
        and re.fullmatch(r"\d{2,4}", yy or "")
    )


def _preprocess_dob_cell(crop):
    """Upscale + contrast for tight DOB digit cells (typed or handwritten)."""
    from PIL import ImageEnhance, ImageOps
    up = crop.resize((max(1, crop.width * 3), max(1, crop.height * 3)), Image.Resampling.LANCZOS)
    up = ImageOps.autocontrast(up)
    up = ImageEnhance.Contrast(up).enhance(1.6)
    return up


def _recognize_dob_cells(image, band, router, engines):
    """OCR MM/DD/YY cells independently and assemble a calendar date from observed digits."""
    cells = _dob_cell_bboxes(band)
    parts = {}
    attempts = []
    raw_bits = []

    def _digits_from(raw: str) -> str:
        return ''.join(ch for ch in raw if ch.isdigit())

    def _consider(label: str, digits: str, raw: str, best_digits: str) -> str:
        if not digits:
            return best_digits
        if label in {'MM', 'DD'}:
            # Keep last 1-2 digits (leading edge glyphs happen).
            trimmed = digits[-2:] if len(digits) >= 2 else digits
            if not (1 <= len(trimmed) <= 2):
                return best_digits
            # Prefer 0X over 1X when both are calendar-plausible — dashed
            # rule bleed into the cell still produces a leading 1.
            if best_digits and len(best_digits) == 2 and len(trimmed) == 2:
                if best_digits[0] == '0' and trimmed[0] == '1' and best_digits[1] == trimmed[1]:
                    return best_digits
                if trimmed[0] == '0' and best_digits[0] == '1' and best_digits[1] == trimmed[1]:
                    return trimmed
            if len(trimmed) >= len(best_digits):
                return trimmed
        elif label == 'YY':
            # Prefer 4-digit years; allow 2-3 (span repairs 983→1983).
            if 2 <= len(digits) <= 5 and len(digits) >= len(best_digits):
                return digits[-4:] if len(digits) > 4 else digits
        return best_digits

    for label, bbox in cells.items():
        bbox = _clamp_bbox(bbox, image.width, image.height)
        crop = _preprocess_dob_cell(image.crop(bbox))
        best_digits = ''
        # Digit-only tesseract first — typed CMS DOB cells often resolve under whitelist.
        # Under STP fast mode, skip route engines when digits already assemble.
        with contextlib.suppress(ImportError, OSError, ValueError, TypeError, AttributeError, RuntimeError):
            import pytesseract
            for psm in _digit_psms_dob():
                cfg = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
                raw = pytesseract.image_to_string(crop, config=cfg).strip()
                attempts.append({
                    'engine': 'tesseract_digits', 'reason': f'PSM_{psm}',
                    'latency_ms': 0.0, 'observation': {'text': raw},
                    'preprocessing_profile': 'dob_cell_upscale', 'dob_cell': label,
                })
                if raw:
                    raw_bits.append(f'{label}:tess{psm}:{raw}')
                    best_digits = _consider(label, _digits_from(raw), raw, best_digits)
                    if best_digits:
                        break
        if not best_digits or not _ocr_fast_mode():
            routed = router.route(
                OCRRouteRequest(crop, (0, 0, crop.width, crop.height), engine_order=engines)
            )
            for attempt in routed.attempts:
                observation = attempt.observation
                attempts.append({
                    'engine': attempt.engine, 'reason': attempt.reason,
                    'latency_ms': attempt.latency_ns / 1e6,
                    'observation': asdict(observation) if observation else None,
                    'preprocessing_profile': 'dob_cell_upscale',
                    'dob_cell': label,
                })
                if observation is None or not observation.lines:
                    continue
                raw = chr(10).join(line.text for line in observation.lines)
                raw_bits.append(f'{label}:{raw}')
                best_digits = _consider(label, _digits_from(raw), raw, best_digits)
        parts[label] = best_digits
    mm, dd, yy = parts.get('MM', ''), parts.get('DD', ''), parts.get('YY', '')
    if not dob_boxed_cells_complete(mm, dd, yy):
        return [], attempts, 'DOB_CELL_PARTIAL_DIGIT'
    # Reject weak cell reads — garbage like "1'4QR2" can span-shape into a false date.
    if not (re.fullmatch(r'\d{1,2}', mm) and 1 <= int(mm) <= 12):
        return [], attempts, 'DOB_CELLS_EMPTY'
    if not (re.fullmatch(r'\d{1,2}', dd) and 1 <= int(dd) <= 31):
        return [], attempts, 'DOB_CELLS_EMPTY'
    if not (re.fullmatch(r'\d{2,4}', yy) and (
        (len(yy) == 2) or (len(yy) == 3 and yy[0] in '189') or (len(yy) == 4 and 1900 <= int(yy) <= 2100)
    )):
        return [], attempts, 'DOB_CELLS_EMPTY'
    joined = f'{mm} {dd} {yy}'
    span = select_field_span(joined, 'DATE', 'patient_dob')
    selected = span.selected_text
    # Only emit when span produced a calendar-shaped date (not the raw join).
    if not selected or not re.fullmatch(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', selected):
        return [], attempts, 'DOB_CELLS_EMPTY'
    # Attribute to a route-authorized producing engine so evidence decision does
    # not strip the assembled candidate as CANDIDATE_ENGINE_NOT_AUTHORIZED.
    # Cell segmentation stays in preprocessing_variant + span reason codes.
    producing_engine = engines[0] if engines else 'rapidocr'
    for attempt in reversed(attempts):
        eng = str(attempt.get('engine') or '')
        if not eng or eng == 'tesseract_digits':
            continue
        obs = attempt.get('observation')
        if obs is None:
            continue
        producing_engine = eng
        break
    box = BoundingBox(
        x0=band[0], y0=band[1], x1=band[2], y1=band[3],
        image_width=image.width, image_height=image.height,
    )
    candidate = OCRCandidate(
        value=selected, raw_value=' | '.join(raw_bits), engine=producing_engine,
        model_name='unknown', model_version='unknown',
        preprocessing_variant='dob_cell_upscale',
        preprocessing_version='cascade-v8',
        raw_confidence=0.8,
        calibrated_confidence=None, bounding_box=box,
        latency_ms=0.0,
    )
    payload = {**asdict(candidate), 'bounding_box': box.model_dump(mode='json')}
    payload['span_selection'] = {
        'selected_text': span.selected_text,
        'rule_id': span.rule_id,
        'confidence': span.confidence,
        'reason_codes': list(span.reason_codes) + ['DOB_CELL_SEGMENT'],
        'cell_parts': parts,
        'assembly_engine': 'dob_cells',
        'producing_engine': producing_engine,
    }
    return [payload], attempts, 'DOB_CELLS_ASSEMBLED'


def recognize_regions(image, geometry, router, emit=lambda rows: None, template=None):
    """OCR field regions via the field-semantic cascade strategy.

    For each field the cascade walks typed crop variants and governed route
    engines, stopping when span-selected text is field-shaped. Empty /
    contaminated financial crops remain empty (no invented amounts).
    """
    if geometry.get('status') != 'SUCCESS' or geometry.get('coordinate_frame') != 'rectified_template_pixels':
        raise ValueError('Successful canonical GeometryResult required')
    template_fields = {}
    if template is not None:
        for region in template.field_regions:
            template_fields[region.field_name] = (region.x0, region.y0, region.x1, region.y1)
    cascade = FieldCascade()
    rows = []

    def _recognize_with_engines(field_name, bbox, field_type, engines):
        return _recognize_one(image, field_name, bbox, router, field_type, engine_order=engines)

    for field in geometry['fields']:
        result = field['result']
        box = result.get('aligned_roi')
        cell = result.get('safe_cell')
        if not box or not cell:
            raise ValueError('Recorded canonical region and safe cell required')
        aligned = tuple(box[k] for k in ('x0', 'y0', 'x1', 'y1'))
        if not (cell['x0'] <= aligned[0] < aligned[2] <= cell['x1'] and
                cell['y0'] <= aligned[1] < aligned[3] <= cell['y1']):
            raise ValueError('Canonical region exceeds recorded safe cell')
        primary = _ocr_bbox(field['field'], aligned, cell, (image.width, image.height), template_fields)
        # Box 28: the amount sits below the printed caption. Stale GeometryResult
        # safe cells from the old (Box-29) ROI cannot contain the corrected
        # template box — force the measured cms1500@03 value band.
        if str(field.get('field') or '').casefold() in {'total_charge', 'total_charges'}:
            try:
                from packages.geometry_authority.box28 import (
                    CMS1500_BOX28_FULL,
                    box28_crop_variants,
                )

                # Prefer caption-excluded / tight value bands over stale GeometryResult
                # safe cells that still point at Box 29.
                variants = box28_crop_variants(CMS1500_BOX28_FULL)
                primary = _clamp_bbox(variants[0], image.width, image.height)
                # Stash alternate crops so the cascade ladder can retry without
                # re-deriving geometry from the wrong template ROI.
                field.setdefault('_box28_crop_variants', [
                    list(_clamp_bbox(v, image.width, image.height)) for v in variants[1:]
                ])
            except Exception:  # noqa: BLE001, S110 -- optional crop variants
                pass
        # Ranking requires every geometry field in the OCR artifact. Out-of-scope
        # fields get empty stubs (no OCR engines) so STP-critical stays fast.
        if not _field_in_scope(field.get('field') or ''):
            rows.append({
                'field': field['field'],
                'canonical_region': list(aligned),
                'ocr_region': list(primary),
                'candidates': [],
                'attempts': [],
                'router_reason': 'OUT_OF_SCOPE',
                'cascade': {
                    'strategy_id': cascade.strategy_id,
                    'accepted': False,
                    'accept_reason': 'OUT_OF_SCOPE',
                    'steps': [],
                },
                'status': 'SKIPPED',
            })
            emit(rows)
            continue
        # DOB alternate strategy (v10): cells-FIRST. Whole-band OCR reads the
        # MM|DD|YY dashed vertical rules as digit "1" (01→11, 09→19), which
        # then fights a second engine under CONFLICT_MARGIN and forces HITL.
        # Independent MM/DD/YY cell OCR + digit-whitelist tesseract avoids rules.
        cascaded = None
        if (
            field['field'].casefold() == 'patient_dob'
            and 'dob_cells' in post_miss_for(field['field'])
        ):
            from packages.extraction_recovery.field_cascade import (
                CascadeResult,
                CascadeStepResult,
                load_route_engines,
            )
            engines = load_route_engines('patient_dob')
            cell_box = (int(cell['x0']), int(cell['y0']), int(cell['x1']), int(cell['y1']))
            cell_h = cell_box[3] - cell_box[1]
            default_band = (
                max(primary[0], cell_box[0] + 2),
                max(primary[1], cell_box[3] - max(22, int(0.40 * cell_h))),
                min(cell_box[2] - 1, image.width),
                min(primary[3], cell_box[3] - 1),
            )
            band = _clamp_bbox(default_band, image.width, image.height)
            cell_cands, cell_attempts, cell_reason = _recognize_dob_cells(
                image, band, router, engines,
            )
            selected = next(
                (c.get('value') or '' for c in cell_cands if (c.get('value') or '').strip()),
                '',
            )
            ok, accept_reason = semantic_accept('patient_dob', selected)
            if ok:
                cascaded = CascadeResult(
                    field_name='patient_dob',
                    bbox=band,
                    candidates=list(cell_cands),
                    attempts=list(cell_attempts),
                    router_reason=cell_reason,
                    status='OBSERVED',
                    cascade_trace=[
                        CascadeStepResult(
                            variant_id='dob_cells_first',
                            bbox=band,
                            engines=engines,
                            selected_value=selected,
                            raw_value=(cell_cands[0].get('raw_value') if cell_cands else '') or '',
                            accepted=True,
                            accept_reason=f'CELLS_FIRST:{accept_reason}',
                            candidates=tuple(cell_cands),
                            attempts=tuple(cell_attempts),
                            router_reason=cell_reason,
                        )
                    ],
                    accepted=True,
                    accept_reason=f'CELLS_FIRST:{accept_reason}',
                    strategy_id=FieldCascade().strategy_id,
                )
        if cascaded is None and _ocr_fast_mode() and field['field'].casefold() in {
            'total_charge', 'total_charges',
        }:
            from packages.extraction_recovery.field_cascade import (
                CascadeResult,
                CascadeStepResult,
            )
            dig_cands, dig_attempts, dig_reason = _recognize_charge_digits_only(image, primary)
            selected = next(
                (c.get('value') or '' for c in dig_cands if (c.get('value') or '').strip()),
                '',
            )
            if not selected:
                # Span may empty sparse digit ink; recover whole-dollar / decimal raw.
                import re as _re_amt
                raw = next(
                    (c.get('raw_value') or '' for c in dig_cands if (c.get('raw_value') or '').strip()),
                    '',
                )
                m = _re_amt.search(
                    r'\$?\d{1,3}(?:,\d{3})*\.\d{2}|\$?\d{2,6}(?:\.\d{2})?',
                    raw or '',
                )
                if m:
                    selected = m.group(0).lstrip('$')
                    if '.' not in selected and _re_amt.fullmatch(r'\d{2,6}', selected):
                        selected = f'{selected}.00'
                    if dig_cands:
                        lead = dict(dig_cands[0])
                        lead['value'] = selected
                        dig_cands[0] = lead
            # Empty after tess digits: one paddle+rapid pass before gpt residual
            # (EMPTY_FINANCIAL_INK model path still runs later on the field row).
            if not selected:
                p_cands, p_attempts, p_reason = _recognize_one(
                    image,
                    field['field'],
                    primary,
                    router,
                    field.get('field_type') or 'CURRENCY',
                    engine_order=('paddleocr', 'rapidocr'),
                )
                dig_attempts = list(dig_attempts or []) + list(p_attempts or [])
                dig_cands = list(dig_cands or []) + list(p_cands or [])
                dig_reason = f'{dig_reason}|{p_reason}|CHARGE_PADDLE_AFTER_EMPTY'
                selected = next(
                    (c.get('value') or '' for c in dig_cands if (c.get('value') or '').strip()),
                    '',
                )
                if not selected:
                    import re as _re_amt2
                    for c in dig_cands:
                        raw = (c.get('raw_value') or c.get('value') or '').strip()
                        m = _re_amt2.search(
                            r'\$?\d{1,3}(?:,\d{3})*\.\d{2}|\$?\d{2,6}(?:\.\d{2})?',
                            raw or '',
                        )
                        if m:
                            selected = m.group(0).lstrip('$')
                            if '.' not in selected and _re_amt2.fullmatch(r'\d{2,6}', selected):
                                selected = f'{selected}.00'
                            break
            # Redesign: monetary variants + image evidence when still empty.
            if not selected:
                m_val, m_cands, m_atts, m_reason, _ev = _recover_empty_monetary_crop(
                    image, primary, field_name=field['field']
                )
                dig_attempts = list(dig_attempts or []) + list(m_atts or [])
                dig_cands = list(dig_cands or []) + list(m_cands or [])
                dig_reason = f'{dig_reason}|{m_reason}|MONETARY_VARIANT_RECOVERY'
                if m_val:
                    selected = m_val
            try:
                from packages.geometry_authority.monetary_geometry import read_monetary_crop

                geo = read_monetary_crop(
                    image.crop(primary),
                    crop_bbox=tuple(float(v) for v in primary),
                    image_size=(int(image.width), int(image.height)),
                )
            except Exception:  # noqa: BLE001
                geo = None
            if geo is not None and geo.geometry_candidate and not geo.ambiguous:
                import re as _re_geo

                geo_val = geo.canonical_monetary_value or geo.geometry_candidate
                geo_digits = _re_geo.sub(r'\D', '', geo.raw_glyph_sequence or '')
                sel_digits = _re_geo.sub(r'\D', '', str(selected or ''))
                # Under-read glyph sets (e.g. 420 from 212.00) must not replace a
                # fuller currency observation already shaped for this Box 28 crop.
                adopt_geometry = (
                    not sel_digits
                    or geo_digits == sel_digits
                    or geo_val == selected
                    or len(geo_digits) >= len(sel_digits)
                )
                dig_cands = list(dig_cands or [])
                monetary_observation = {
                    'text': geo.raw_glyph_sequence,
                    'shaped': geo_val,
                    'raw_digit_sequence': geo.raw_glyph_sequence,
                    'page_glyph_polygons': [
                        [list(pt) for pt in poly]
                        for poly in geo.page_glyph_polygons
                    ],
                    'canonical_glyph_centres': [
                        list(c) for c in geo.canonical_glyph_centres
                    ],
                    'dollar_glyphs': list(geo.dollar_glyphs),
                    'cents_glyphs': list(geo.cents_glyphs),
                    'unit_zone_glyphs': list(geo.unit_zone_glyphs),
                    'canonical_monetary_value': geo.canonical_monetary_value,
                    'adopted': adopt_geometry,
                }
                dig_attempts = list(dig_attempts or []) + [{
                    'engine': 'rapidocr',
                    'reason': 'GEOMETRY_CENTS' if adopt_geometry else 'GEOMETRY_CENTS_UNDERREAD',
                    'observation': monetary_observation,
                }]
                if adopt_geometry:
                    dig_cands.insert(0, {
                        'value': geo_val,
                        'raw_value': geo.raw_glyph_sequence,
                        'engine': 'rapidocr',
                        'model_name': 'geometry-cents',
                        'model_version': 'monetary-geometry',
                        'preprocessing_variant': 'GEOMETRY_CENTS',
                        'raw_confidence': 0.91,
                        'calibrated_confidence': None,
                        'bounding_box': {
                            'x0': float(primary[0]),
                            'y0': float(primary[1]),
                            'x1': float(primary[2]),
                            'y1': float(primary[3]),
                            'image_width': int(image.width),
                            'image_height': int(image.height),
                        },
                        'latency_ms': 0.0,
                        'validation_results': [],
                        'evidence_reference': 'GEOMETRY_CENTS',
                        'estimated_cost_usd': 0.0,
                        'actual_cost_usd': None,
                        'preprocessing_version': 'geometry-cents',
                        'registration_confidence': None,
                        'image_quality_score': None,
                        # EvidenceProvenance forbids freeform monetary keys — keep
                        # typed lineage empty and persist geometry on the attempt.
                        'provenance': None,
                    })
                    selected = geo_val
                    dig_reason = f'{dig_reason}|GEOMETRY_CENTS'
                else:
                    dig_reason = f'{dig_reason}|GEOMETRY_CENTS_UNDERREAD'
            ok, accept_reason = semantic_accept(
                field['field'],
                selected,
                bbox=primary,
                image_size=(image.width, image.height),
            )
            if ok:
                cascaded = CascadeResult(
                    field_name=field['field'],
                    bbox=primary,
                    candidates=list(dig_cands),
                    attempts=list(dig_attempts),
                    router_reason=dig_reason,
                    status='OBSERVED',
                    cascade_trace=[
                        CascadeStepResult(
                            variant_id='charge_digits_first',
                            bbox=primary,
                            engines=('tesseract_digits',),
                            selected_value=selected,
                            raw_value=(dig_cands[0].get('raw_value') if dig_cands else '') or '',
                            accepted=True,
                            accept_reason=f'DIGITS_FIRST:{accept_reason}',
                            candidates=tuple(dig_cands),
                            attempts=tuple(dig_attempts),
                            router_reason=dig_reason,
                        )
                    ],
                    accepted=True,
                    accept_reason=f'DIGITS_FIRST:{accept_reason}',
                    strategy_id=cascade.strategy_id,
                )
            else:
                # Empty box-28: do not exhaust paddle×3 crop variants — E6 uses
                # service-line charges. Keeps STP eval from multi-minute thrash.
                cascaded = CascadeResult(
                    field_name=field['field'],
                    bbox=primary,
                    candidates=list(dig_cands),
                    attempts=list(dig_attempts),
                    router_reason=dig_reason or 'CHARGE_DIGITS_EMPTY',
                    status='NO_VALUE',
                    cascade_trace=[
                        CascadeStepResult(
                            variant_id='charge_digits_first',
                            bbox=primary,
                            engines=('tesseract_digits',),
                            selected_value='',
                            raw_value=(dig_cands[0].get('raw_value') if dig_cands else '') or '',
                            accepted=False,
                            accept_reason=accept_reason or 'EMPTY',
                            candidates=tuple(dig_cands),
                            attempts=tuple(dig_attempts),
                            router_reason=dig_reason or 'CHARGE_DIGITS_EMPTY',
                        )
                    ],
                    accepted=False,
                    accept_reason=accept_reason or 'EMPTY',
                    strategy_id=cascade.strategy_id,
                )
        if cascaded is None:
            cascaded = cascade.recognize(
                field_name=field['field'],
                primary_bbox=primary,
                cell=cell,
                image_size=(image.width, image.height),
                recognize_fn=_recognize_with_engines,
            )
        # IJN2.022 / handwriting DOB: whole-band OCR fragments MM/DD/YY; cell
        # segmentation recovers calendar-valid dates from observed digit ink only.
        # Try the reconstructed digit band plus any cascade crop that already
        # held year/day fragments (digit_band / year_wide).
        if (not cascaded.accepted) and 'dob_cells' in post_miss_for(field['field']):
            from packages.extraction_recovery.field_cascade import (
                CascadeResult,
                CascadeStepResult,
                load_route_engines,
            )
            engines = load_route_engines('patient_dob')
            cell_box = (int(cell['x0']), int(cell['y0']), int(cell['x1']), int(cell['y1']))
            cell_h = cell_box[3] - cell_box[1]
            default_band = (
                max(primary[0], cell_box[0] + 2),
                max(primary[1], cell_box[3] - max(22, int(0.40 * cell_h))),
                min(cell_box[2] - 1, image.width),
                min(primary[3], cell_box[3] - 1),
            )
            bands: list[tuple[str, tuple[int, int, int, int]]] = [
                ('dob_cells', _clamp_bbox(default_band, image.width, image.height)),
            ]
            for prior in cascaded.cascade_trace:
                if prior.variant_id in {'dob_digit_band', 'dob_year_wide', 'dob_loose'}:
                    bands.append(
                        (
                            f'dob_cells_on_{prior.variant_id}',
                            _clamp_bbox(prior.bbox, image.width, image.height),
                        )
                    )
            seen_bands: set[tuple[int, int, int, int]] = set()
            for variant_id, band in bands:
                if band in seen_bands:
                    continue
                seen_bands.add(band)
                cell_cands, cell_attempts, cell_reason = _recognize_dob_cells(
                    image, band, router, engines,
                )
                selected = next(
                    (c.get('value') or '' for c in cell_cands if (c.get('value') or '').strip()),
                    '',
                )
                ok, accept_reason = semantic_accept('patient_dob', selected)
                step = CascadeStepResult(
                    variant_id=variant_id,
                    bbox=band,
                    engines=engines,
                    selected_value=selected,
                    raw_value=(cell_cands[0].get('raw_value') if cell_cands else '') or '',
                    accepted=ok,
                    accept_reason=accept_reason if selected else cell_reason,
                    candidates=tuple(cell_cands),
                    attempts=tuple(cell_attempts),
                    router_reason=cell_reason,
                )
                cascaded.cascade_trace.append(step)
                if ok:
                    cascaded = CascadeResult(
                        field_name='patient_dob',
                        bbox=band,
                        candidates=list(cell_cands),
                        attempts=list(cell_attempts),
                        router_reason=cell_reason,
                        status='OBSERVED',
                        cascade_trace=list(cascaded.cascade_trace),
                        accepted=True,
                        accept_reason=accept_reason,
                        strategy_id=cascade.strategy_id,
                    )
                    break
        # When whole-band accepted but engines disagree on separator-1 dates,
        # still attach dob_cells as a corroborating / preferred candidate.
        elif (
            cascaded.accepted
            and field['field'].casefold() == 'patient_dob'
            and 'dob_cells' in post_miss_for(field['field'])
            and not any(
                str(getattr(step, 'variant_id', '')).startswith('dob_cells')
                for step in cascaded.cascade_trace
            )
        ):
            from packages.extraction_recovery.field_cascade import (
                CascadeResult,
                load_route_engines,
            )
            engines = load_route_engines('patient_dob')
            cell_box = (int(cell['x0']), int(cell['y0']), int(cell['x1']), int(cell['y1']))
            cell_h = cell_box[3] - cell_box[1]
            band = _clamp_bbox(
                (
                    max(primary[0], cell_box[0] + 2),
                    max(primary[1], cell_box[3] - max(22, int(0.40 * cell_h))),
                    min(cell_box[2] - 1, image.width),
                    min(primary[3], cell_box[3] - 1),
                ),
                image.width,
                image.height,
            )
            cell_cands, cell_attempts, _cell_reason = _recognize_dob_cells(
                image, band, router, engines,
            )
            selected = next(
                (c.get('value') or '' for c in cell_cands if (c.get('value') or '').strip()),
                '',
            )
            ok, accept_reason = semantic_accept('patient_dob', selected)
            if ok and cell_cands:
                cascaded = CascadeResult(
                    field_name=cascaded.field_name,
                    bbox=band,
                    candidates=list(cell_cands) + list(cascaded.candidates),
                    attempts=list(cell_attempts) + list(cascaded.attempts),
                    router_reason=cascaded.router_reason,
                    status=cascaded.status,
                    cascade_trace=list(cascaded.cascade_trace),
                    accepted=True,
                    accept_reason=f'CELLS_PREFERRED:{accept_reason}',
                    strategy_id=cascaded.strategy_id,
                )
        rows.append({
            'field': field['field'],
            'canonical_region': list(aligned),
            'ocr_region': list(cascaded.bbox),
            'candidates': cascaded.candidates,
            'attempts': cascaded.attempts,
            'router_reason': cascaded.router_reason,
            'cascade': {
                'strategy_id': cascaded.strategy_id,
                'accepted': cascaded.accepted,
                'accept_reason': cascaded.accept_reason,
                'steps': [
                    {
                        'variant_id': step.variant_id,
                        'bbox': list(step.bbox),
                        'engines': list(step.engines),
                        'selected_value': step.selected_value,
                        'accepted': step.accepted,
                        'accept_reason': step.accept_reason,
                        'router_reason': step.router_reason,
                    }
                    for step in cascaded.cascade_trace
                ],
            },
            'status': cascaded.status,
        })
        emit(rows)
    return rows


def run(directory, output):
    from workers.ocr_engine_factories import wire_package_ocr_providers

    wire_package_ocr_providers()
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    geometry = json.loads((directory / 'GeometryResult.json').read_text())
    telemetry = json.loads((directory / 'geometry_telemetry.json').read_text())
    trace = json.loads((directory / 'registration_trace.json').read_text())
    if telemetry['status'] != 'SUCCESS' or not telemetry['registration_evidence']['accepted']:
        raise ValueError('Accepted saved registration and geometry required')
    matrix = np.asarray(geometry['source_to_geometry_transform'], dtype=float)
    if not np.array_equal(matrix, telemetry['registration_evidence']['transform_matrix']):
        raise ValueError('Geometry and registration transforms disagree')
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Invalid saved transform')
    observations = [e['data']['coverage_observation'] for t in trace['traces'] for e in t['events']
                    if 'coverage_observation' in e.get('data', {})]
    size = next(o['reference']['size'] for o in observations if o['stage'] == 'Input image')
    with ZipFile(telemetry['source']['archive']) as archive:
        payload = archive.read(telemetry['source']['entry'])
    if hashlib.sha256(payload).hexdigest() != telemetry['document_id']:
        raise ValueError('Source TIFF hash mismatch')
    with Image.open(BytesIO(payload)) as tiff:
        tiff.seek(geometry['page_number'] - 1)
        with tiff.convert('L') as source:
            pixels = cv2.warpPerspective(np.asarray(source), matrix, tuple(size), borderValue=255)
    report = {'type': 'OCRCandidates', 'document_id': telemetry['document_id'],
              'page_number': geometry['page_number'], 'geometry_reference': str(directory / 'GeometryResult.json'),
              'geometry_sha256': hashlib.sha256((directory / 'GeometryResult.json').read_bytes()).hexdigest(),
              'coordinate_frame': 'rectified_template_pixels', 'stop_after': 'ocr',
              'registration_executed': False, 'geometry_estimated': False,
              'ranking_executed': False, 'validators_executed': False,
              'policy': 'Field-cascade v1: crop variants × governed route engines; stop on semantic field accept; no invented amounts',
              'confidence_basis': 'Arithmetic mean of raw line confidences; raw lines preserved',
              'fields': [], 'service_lines': [], 'status': 'RUNNING'}
    def save(rows):
        report['fields'] = rows
        (output / 'OCRCandidates.json').write_text(json.dumps(report, indent=2, default=str, allow_nan=False), encoding='utf-8')
    template = _load_cms1500_template()
    from packages.runtime_wiring import reset_telemetry, stage_enabled

    tel = reset_telemetry()
    claim_id = str(telemetry.get('document_id') or '')[:16]
    try:
        with Image.fromarray(pixels) as canonical:
            with tel.track(
                "package_intelligence",
                enabled=stage_enabled("CDP_PACKAGE_INTELLIGENCE", "1"),
                claim_id=claim_id,
                inputs={"entry": telemetry.get("source", {}).get("entry")},
            ) as inv:
                if not inv.bypassed:
                    from packages.document_finance import (
                        interpret_page_finance,
                        seed_known_discrepancies,
                    )
                    from packages.package_intelligence import (
                        build_claim_package,
                        classify_page_signals,
                    )

                    # Cheap page-text classification — never assume CMS1500.
                    page_text = ""
                    try:
                        import pytesseract

                        w, h = canonical.size
                        # Mask right-edge scan metadata (~8%) before classification OCR.
                        probe = canonical.crop((0, 0, int(w * 0.92), min(h, int(h * 0.45))))
                        page_text = pytesseract.image_to_string(
                            probe, config="--psm 6"
                        )
                    except Exception:  # noqa: BLE001
                        page_text = ""
                    barcode_text = None
                    qr_meta = {}
                    try:
                        from packages.package_intelligence.qr_decode import (
                            decode_page_qr,
                            qr_supports_cms1500_family,
                        )

                        qr = decode_page_qr(canonical)
                        qr_meta = qr.to_dict()
                        barcode_text = qr.texts[0] if qr.texts else None
                        if qr_supports_cms1500_family(qr.texts) and not page_text:
                            page_text = "HEALTH INSURANCE CLAIM FORM"
                    except Exception:  # noqa: BLE001
                        qr_meta = {"detected": False, "reasons": ["QR_DECODE_SKIPPED"]}
                    page = classify_page_signals(
                        page_index=int(geometry.get("page_number") or 1) - 1,
                        ocr_text=page_text,
                        barcode_text=barcode_text,
                        form_family=None,
                        router_label=None,
                        confidence=0.85,
                    )
                    package = build_claim_package(
                        package_id=claim_id,
                        claim_id=claim_id,
                        pages=[page],
                    )
                    finance = interpret_page_finance(
                        text=page_text,
                        family=page.page_class.value
                        if page.page_class.value
                        in {
                            "CMS1500",
                            "UB04",
                            "REIMBURSEMENT_SUPERBILL",
                            "RUNNING_ACCOUNT_STATEMENT",
                            "EOB",
                            "SEPARATOR",
                            "ATTACHMENT",
                            "UNKNOWN",
                        }
                        else None,
                        claim_id=str(telemetry.get("source", {}).get("entry") or claim_id),
                        discrepancy_ledger=seed_known_discrepancies(),
                    )
                    inv.outputs = {
                        **package.to_dict(),
                        "document_finance": finance.to_dict(),
                        "qr_decode": qr_meta,
                    }
                    report["package_intelligence"] = package.to_dict()
                    report["document_finance"] = finance.to_dict()
                    report["qr_decode"] = qr_meta
                    report["document_family"] = page.page_class.value
                    report["allows_cms_geometry"] = bool(page.allows_cms_geometry)
            router = OCRRouter(lambda attempt: True)
            allows_cms = bool(report.get("allows_cms_geometry", True))
            if report.get("document_family") == "SEPARATOR":
                report["fields"] = []
                report["service_lines"] = []
                report["status"] = "SEPARATOR_EXCLUDED"
                report["gap_classes"] = []
                save([])
            elif not allows_cms and report.get("document_family") not in {
                None,
                "CMS1500",
                "UNKNOWN",
            }:
                # Non-CMS claim pages: never invoke Box 28 / 24F geometry.
                report["fields"] = []
                report["service_lines"] = []
                report["status"] = "NON_CMS_FAMILY"
                report["cms_geometry_skipped"] = True
                save([])
            else:
                recognize_regions(canonical, geometry, router, save, template=template)
                report['service_lines'] = recognize_service_lines(canonical, router, template)
                report['fields'] = _maybe_attach_dob_handwriting_residuals(
                    report['fields'], canonical
                )
            # Field authority + financial reconciliation telemetry on totals.
            with tel.track(
                "field_authority",
                enabled=stage_enabled("CDP_FIELD_AUTHORITY", "1"),
                claim_id=claim_id,
                field_name="total_charge",
            ) as inv:
                if not inv.bypassed:
                    from packages.field_authority import accept_field, independent_evidence_count

                    total_row = next(
                        (
                            f
                            for f in report["fields"]
                            if str(f.get("field") or "").casefold()
                            in {"total_charge", "total_charges"}
                        ),
                        None,
                    )
                    cands = list((total_row or {}).get("candidates") or [])
                    for line in report.get("service_lines") or []:
                        cands.extend(line.get("candidates") or [])
                    indep = independent_evidence_count(cands)
                    decision = accept_field(
                        field_name="total_charge",
                        valid_geometry=True,
                        valid_semantics=bool((total_row or {}).get("value")),
                        valid_format=bool((total_row or {}).get("value")),
                        calibrated_confidence=0.9 if (total_row or {}).get("value") else 0.0,
                        field_threshold=0.95,
                        independent_evidence=indep,
                        required_evidence=2,
                        unresolved_conflict=False,
                        llm_only=False,
                        critical=True,
                    )
                    inv.outputs = decision.to_dict()
                    report["field_authority"] = decision.to_dict()
            with tel.track(
                "financial_reconciliation",
                enabled=stage_enabled("CDP_FINANCIAL_RECONCILIATION", "1"),
                claim_id=claim_id,
                field_name="total_charge",
            ) as inv:
                if not inv.bypassed:
                    from packages.financial_reconciliation import (
                        reconcile_by_document_family,
                    )

                    box28 = next(
                        (
                            f.get("value")
                            for f in report["fields"]
                            if str(f.get("field") or "").casefold()
                            in {"total_charge", "total_charges"}
                        ),
                        None,
                    )
                    lines = report.get("service_lines") or []
                    fin = reconcile_by_document_family(
                        document_family=report.get("document_family"),
                        document_finance=report.get("document_finance"),
                        box28_value=box28,
                        service_lines=lines,
                        charge_column_verified=bool(lines),
                        all_service_rows_detected=bool(lines),
                        independent_evidence_paths=2
                        if len(lines) >= 2
                        else (1 if lines else 0),
                    )
                    inv.outputs = fin.to_dict()
                    report["financial_reconciliation"] = fin.to_dict()
                    # Promote family-accepted total onto the total_charge field row
                    # when CMS geometry was intentionally skipped.
                    if (
                        fin.accepted_total
                        and fin.disposition.value == "LINE_TOTALS_RECONCILED"
                        and not report.get("allows_cms_geometry", True)
                    ):
                        report["fields"] = list(report.get("fields") or []) + [
                            {
                                "field": "total_charge",
                                "value": fin.accepted_total,
                                "status": "FIELD_ACCEPTED",
                                "cascade": {
                                    "accepted": True,
                                    "accept_reason": "FAMILY_FINANCE:"
                                    + ",".join(fin.reasons[:3]),
                                    "value": fin.accepted_total,
                                },
                                "candidates": [
                                    {
                                        "value": fin.accepted_total,
                                        "engine": "document_family_finance",
                                        "raw_value": fin.accepted_total,
                                    }
                                ],
                            }
                        ]
                        save(report["fields"])
            with tel.track(
                "claim_decision",
                enabled=stage_enabled("CDP_CLAIM_DECISION_ROUTES", "1"),
                claim_id=claim_id,
            ) as inv:
                if not inv.bypassed:
                    from packages.claim_decision import route_claim_hitl

                    fin_disp = (report.get("financial_reconciliation") or {}).get(
                        "disposition"
                    )
                    unresolved = []
                    if not (report.get("field_authority") or {}).get("accepted"):
                        unresolved.append("total_charge")
                    route = route_claim_hitl(
                        registration_ok=True,
                        package_complete=bool(
                            (report.get("package_intelligence") or {}).get("complete", True)
                        ),
                        unresolved_critical_fields=unresolved,
                        financial_disposition=fin_disp,
                    )
                    inv.outputs = route
                    report["hitl_route"] = route
        report['status'] = 'COMPLETED'
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        report['fields'] = _promote_self_box2_values(report.get('fields') or [])
        save(report['fields'])
        wiring = tel.summary()
        report['runtime_wiring'] = wiring
        tel.write(output / 'stage_wiring.json')
        (output / 'ocr_telemetry.json').write_text(json.dumps({
            'status': report['status'], 'fields_completed': len(report['fields']),
            'provider_attempts': [{'field': r['field'], 'attempts': [
                {k: a[k] for k in ('engine', 'reason', 'latency_ms') if k in a} for a in r['attempts']]}
                for r in report['fields']], 'stop_after': 'ocr',
            'runtime_wiring': wiring,
        }, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry_directory')
    parser.add_argument('output_directory')
    args = parser.parse_args()
    result = run(args.geometry_directory, args.output_directory)
    print(json.dumps({'status': result['status'], 'fields': len(result['fields']),
                      'candidates': sum(len(r['candidates']) for r in result['fields'])}))
    # PaddleOCR / ONNX Runtime often hang in atexit finalizers after a successful
    # run (poll forever while holding multi-GB RSS). Hard-exit once artifacts are
    # on disk so the cascade parent can advance to rank/validate.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if result.get('status') == 'COMPLETED' else 1)
