"""CMS-1500 repeated-field reconciliation (Box 2↔4 names, Box 3↔11a DOB).

These are independent printed occurrences on the same form. Agreement under a
valid Self relationship is corroborating evidence — not a second OCR engine on
the same crop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SELF_CODES = frozenset({"SELF", "18", "01", "1"})


def normalize_person_name(value: object) -> str:
    """Case/punct/space normalisation only — never invent characters."""
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = text.replace(",", " ")
    text = re.sub(r"[^A-Z\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def promote_fuller_observed_name(
    value: object,
    observed: object,
    other_box_value: object,
    relationship: object,
) -> str:
    """Return ``observed`` when it is a fuller Self twin of ``value``.

    Copies an already-printed string. Does not invent characters. Non-Self
    relationships keep ``value`` even when the two boxes match.
    """
    current = str(value or "").strip()
    fuller = str(observed or "").strip()
    if not current:
        return fuller
    if not fuller or not relationship_is_self(relationship):
        return current
    if not names_agree(fuller, other_box_value):
        return current
    current_tokens = normalize_person_name(current).split()
    fuller_tokens = normalize_person_name(fuller).split()
    if len(fuller_tokens) <= len(current_tokens):
        return current
    if not set(current_tokens).issubset(set(fuller_tokens)):
        return current
    return fuller


def prefer_fuller_self_name(
    observed_values: list[object],
    other_box_value: object,
) -> str | None:
    """Pick an already-observed name that agrees with the other box.

    Does not invent characters. A shorter OCR twin is ignored when a longer
    observed candidate already matches the independent box.
    """
    target = normalize_person_name(other_box_value)
    if not target:
        return None
    for raw in observed_values:
        text = str(raw or "").strip()
        if text and names_agree(text, target):
            return text
    return None


def names_agree(left: object, right: object) -> bool:
    a, b = normalize_person_name(left), normalize_person_name(right)
    if not a or not b:
        return False
    if a == b:
        return True
    # Token-set equality (LAST FIRST vs FIRST LAST).
    ta, tb = set(a.split()), set(b.split())
    return bool(ta) and ta == tb


def normalize_dob(value: object) -> str | None:
    """Return YYYY-MM-DD when the string is a calendar-shaped date."""
    raw = str(value or "").strip()
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 8:
        # Prefer MMDDYYYY (CMS checkbox style) then YYYYMMDD.
        for year, month, day in (
            (digits[4:8], digits[0:2], digits[2:4]),
            (digits[0:4], digits[4:6], digits[6:8]),
        ):
            try:
                y, m, d = int(year), int(month), int(day)
                if 1 <= m <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100:
                    return f"{y:04d}-{m:02d}-{d:02d}"
            except ValueError:
                continue
    # ISO / dashed
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", raw)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", raw)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def relationship_is_self(value: object) -> bool:
    key = str(value or "").strip().upper()
    return key in _SELF_CODES


@dataclass(frozen=True)
class NameRedundancyResult:
    agreed: bool
    patient_norm: str
    insured_norm: str
    reason: str


def reconcile_box2_box4_names(
    patient_name: object,
    insured_name: object,
    *,
    relationship: object = None,
) -> NameRedundancyResult:
    patient = normalize_person_name(patient_name)
    insured = normalize_person_name(insured_name)
    if not patient or not insured:
        return NameRedundancyResult(False, patient, insured, "NAME_MISSING")
    if not names_agree(patient, insured):
        return NameRedundancyResult(False, patient, insured, "NAME_MISMATCH")
    if (
        relationship is not None
        and str(relationship).strip()
        and not relationship_is_self(relationship)
    ):
        return NameRedundancyResult(False, patient, insured, "RELATIONSHIP_NOT_SELF")
    return NameRedundancyResult(True, patient, insured, "BOX2_BOX4_SELF_AGREE")


@dataclass(frozen=True)
class DobRedundancyResult:
    agreed: bool
    patient_iso: str | None
    insured_iso: str | None
    reason: str


def reconcile_box3_box11a_dob(
    patient_dob: object,
    insured_dob: object,
    *,
    relationship: object = None,
) -> DobRedundancyResult:
    patient = normalize_dob(patient_dob)
    insured = normalize_dob(insured_dob)
    if patient is None or insured is None:
        return DobRedundancyResult(False, patient, insured, "DOB_UNSHAPED")
    if patient != insured:
        return DobRedundancyResult(False, patient, insured, "DOB_MISMATCH")
    if (
        relationship is not None
        and str(relationship).strip()
        and not relationship_is_self(relationship)
    ):
        return DobRedundancyResult(False, patient, insured, "RELATIONSHIP_NOT_SELF")
    return DobRedundancyResult(True, patient, insured, "BOX3_BOX11A_SELF_AGREE")
