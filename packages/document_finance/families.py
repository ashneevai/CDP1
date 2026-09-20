"""Document-family classification for financial interpretation.

CMS Box 28 / Box 24F authority is valid only for CMS1500. Every other family
must use its own parser. Classification evidence is persisted for audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DocumentFamily(StrEnum):
    CMS1500 = "CMS1500"
    UB04 = "UB04"
    REIMBURSEMENT_SUPERBILL = "REIMBURSEMENT_SUPERBILL"
    RUNNING_ACCOUNT_STATEMENT = "RUNNING_ACCOUNT_STATEMENT"
    EOB = "EOB"
    SEPARATOR = "SEPARATOR"
    ATTACHMENT = "ATTACHMENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FamilyClassification:
    family: DocumentFamily
    confidence: float
    evidence: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family.value,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "anchors": list(self.anchors),
        }

    @property
    def allows_cms_geometry(self) -> bool:
        return self.family is DocumentFamily.CMS1500

    @property
    def is_claim_financial_page(self) -> bool:
        return self.family in {
            DocumentFamily.CMS1500,
            DocumentFamily.UB04,
            DocumentFamily.REIMBURSEMENT_SUPERBILL,
            DocumentFamily.RUNNING_ACCOUNT_STATEMENT,
            DocumentFamily.EOB,
        }


_SEPARATOR = (
    "DOCUMENT SEPARATOR",
    "SOURCEHOV",
    "SOURCE HOV",
    "BATCH SEPARATOR",
    "THIS PAGE INTENTIONALLY",
)
_CMS = (
    "HEALTH INSURANCE CLAIM FORM",
    "NATIONAL UNIFORM CLAIM COMMITTEE",
    "NUCC",
    "CMS-1500",
    "CMS1500",
)
_UB = ("UB-04", "UB04", "UNIFORM BILL", "NUBC")
_SUPERBILL = (
    "STATEMENT FOR INSURANCE REIMBURSEMENT",
    "INSURANCE REIMBURSEMENT",
    "REIMBURSEMENT STATEMENT",
)
_RUNNING = (
    "CHARGE(PAYMENT)",
    "CHARGE (PAYMENT)",
    "BALANCE FORWARD",
    "DATE OF SERVICE",
    "PROCEDURE & DIAGNOSIS",
    "RUNNING BALANCE",
)
_EOB = (
    "EXPLANATION OF BENEFITS",
    "EXPLANATION OF BENEFIT",
    "THIS IS NOT A BILL",
)


def classify_document_family(
    text: str | None,
    *,
    barcode_text: str | None = None,
    layout_hints: dict[str, Any] | None = None,
) -> FamilyClassification:
    """Classify a page from OCR/layout text. Never invents CMS geometry."""
    blob = f"{text or ''}\n{barcode_text or ''}".upper()
    evidence: list[str] = []
    anchors: list[str] = []

    def _hit(needles: tuple[str, ...], label: str) -> bool:
        for needle in needles:
            if needle in blob:
                evidence.append(f"{label}:{needle}")
                anchors.append(needle)
                return True
        return False

    if _hit(_SEPARATOR, "SEPARATOR") or (
        barcode_text and "SEP" in barcode_text.upper()
    ):
        if barcode_text and "SEP" in barcode_text.upper():
            evidence.append("SEPARATOR:BARCODE")
        return FamilyClassification(
            DocumentFamily.SEPARATOR, 0.95, tuple(evidence), tuple(anchors)
        )

    if _hit(_SUPERBILL, "SUPERBILL"):
        return FamilyClassification(
            DocumentFamily.REIMBURSEMENT_SUPERBILL,
            0.92,
            tuple(evidence),
            tuple(anchors),
        )

    running_hits = sum(1 for n in _RUNNING if n in blob)
    if running_hits >= 2 or ("BALANCE FORWARD" in blob and "CHARGE" in blob):
        for n in _RUNNING:
            if n in blob:
                evidence.append(f"RUNNING:{n}")
                anchors.append(n)
        return FamilyClassification(
            DocumentFamily.RUNNING_ACCOUNT_STATEMENT,
            0.9 if running_hits >= 2 else 0.75,
            tuple(dict.fromkeys(evidence)),
            tuple(dict.fromkeys(anchors)),
        )

    if _hit(_EOB, "EOB"):
        return FamilyClassification(
            DocumentFamily.EOB, 0.88, tuple(evidence), tuple(anchors)
        )

    if _hit(_CMS, "CMS"):
        return FamilyClassification(
            DocumentFamily.CMS1500, 0.93, tuple(evidence), tuple(anchors)
        )

    if _hit(_UB, "UB"):
        return FamilyClassification(
            DocumentFamily.UB04, 0.9, tuple(evidence), tuple(anchors)
        )

    hints = layout_hints or {}
    if hints.get("has_cms_grid"):
        evidence.append("LAYOUT:CMS_GRID")
        return FamilyClassification(
            DocumentFamily.CMS1500, 0.7, tuple(evidence), ("LAYOUT_CMS_GRID",)
        )
    if hints.get("has_ledger_columns"):
        evidence.append("LAYOUT:LEDGER_COLUMNS")
        return FamilyClassification(
            DocumentFamily.RUNNING_ACCOUNT_STATEMENT,
            0.65,
            tuple(evidence),
            ("LAYOUT_LEDGER",),
        )

    if re.search(r"\bATTACHMENT\b|\bSUPPORTING\b|\bCOVER\s*SHEET\b", blob):
        evidence.append("ATTACHMENT:KEYWORD")
        return FamilyClassification(
            DocumentFamily.ATTACHMENT, 0.55, tuple(evidence), ("ATTACHMENT",)
        )

    return FamilyClassification(
        DocumentFamily.UNKNOWN, 0.2, ("UNKNOWN_PAGE",), ()
    )


def cms_geometry_forbidden(family: DocumentFamily | str) -> bool:
    fam = DocumentFamily(str(family))
    return fam is not DocumentFamily.CMS1500
