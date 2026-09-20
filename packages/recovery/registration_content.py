"""Post-registration content checks against CMS-1500 landmark labels.

Rejects warps that place insurance-type text into patient identity boxes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

_INSURANCE_TOKENS = ("MEDICARE", "MEDICAID", "TRICARE", "CHAMPVA", "FECA")
_NAME_LABEL_TOKENS = ("PATIENT", "NAME")
_DOB_LABEL_TOKENS = ("BIRTH", "DATE", "MM", "DD", "YY")

_region_ocr: Callable[[Image.Image, tuple[int, int, int, int]], str] | None = None
_confirm_region_ocr: Callable[[Image.Image, tuple[int, int, int, int]], str] | None = None


@dataclass(frozen=True)
class ContentValidationResult:
    accepted: bool
    reason: str
    patient_name_text: str = ""
    patient_dob_text: str = ""


def configure_registration_region_ocr(
    factory: Callable[[Image.Image, tuple[int, int, int, int]], str],
    confirm_factory: Callable[[Image.Image, tuple[int, int, int, int]], str] | None = None,
) -> None:
    """Composition root injects regional OCR used for content landmark checks.

    ``confirm_factory`` is a second reader. A single noisy pass must not veto a
    warp that already passed geometric acceptance.
    """
    global _region_ocr, _confirm_region_ocr
    _region_ocr = factory
    _confirm_region_ocr = confirm_factory


def identity_roi_reads_insurance_row(name_text: str, dob_text: str) -> bool:
    """True when identity crops read the insurance-type row instead of a name."""
    name_u = (name_text or "").upper()
    dob_u = (dob_text or "").upper()
    insurance_hits = sum(
        1 for token in _INSURANCE_TOKENS if token in name_u or token in dob_u
    )
    name_label_hits = sum(1 for token in _NAME_LABEL_TOKENS if token in name_u)
    if insurance_hits >= 2 and name_label_hits == 0:
        return True
    return bool(re.search(r"\bMEDICARE\b", name_u) and re.search(r"\bCHAMPVA\b", dob_u))


def _ocr_region(image: Image.Image, box: tuple[int, int, int, int]) -> str:
    if _region_ocr is None:
        raise RuntimeError(
            "Registration region OCR not configured; "
            "call configure_registration_region_ocr from composition root"
        )
    return _region_ocr(image, box)


def validate_cms1500_registration_content(
    warped: Image.Image,
    *,
    patient_name_box: tuple[int, int, int, int],
    patient_dob_box: tuple[int, int, int, int],
) -> ContentValidationResult:
    """Return accepted=False when identity ROIs still read insurance-type labels.

    A rejecting primary read is kept only when a second reader agrees. One
    engine hallucinating MEDICARE/MEDICAID on a geometrically accepted warp
    must not fail the claim closed.
    """
    name_text = _ocr_region(warped, patient_name_box)
    dob_text = _ocr_region(warped, patient_dob_box)
    if not identity_roi_reads_insurance_row(name_text, dob_text):
        return ContentValidationResult(
            True, "CONTENT_LANDMARKS_PLAUSIBLE", name_text, dob_text
        )
    if _confirm_region_ocr is not None:
        confirm_name = _confirm_region_ocr(warped, patient_name_box)
        confirm_dob = _confirm_region_ocr(warped, patient_dob_box)
        if not identity_roi_reads_insurance_row(confirm_name, confirm_dob):
            return ContentValidationResult(
                True,
                "CONTENT_VETO_UNCORROBORATED",
                name_text,
                dob_text,
            )
        name_text = confirm_name
        dob_text = confirm_dob
    import logging

    logging.getLogger("registration_content").warning(
        "identity ROI insurance-row veto name=%r dob=%r",
        name_text,
        dob_text,
    )
    return ContentValidationResult(
        False,
        "IDENTITY_ROI_READS_INSURANCE_TYPE_ROW",
        name_text,
        dob_text,
    )
