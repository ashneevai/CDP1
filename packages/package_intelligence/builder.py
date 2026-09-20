"""Claim package intelligence: page identity before field extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from packages.document_finance.families import (
    DocumentFamily,
    classify_document_family,
)


class PageClass(StrEnum):
    CMS1500 = "CMS1500"
    UB04 = "UB04"
    REIMBURSEMENT_SUPERBILL = "REIMBURSEMENT_SUPERBILL"
    RUNNING_ACCOUNT_STATEMENT = "RUNNING_ACCOUNT_STATEMENT"
    STATEMENT = "STATEMENT"  # legacy alias → prefer RUNNING / SUPERBILL
    SUPERBILL = "SUPERBILL"  # legacy alias
    EOB = "EOB"
    ATTACHMENT = "ATTACHMENT"
    SEPARATOR = "SEPARATOR"
    NON_CLAIM = "NON_CLAIM"
    UNKNOWN = "UNKNOWN"


class PackageIssue(StrEnum):
    MISSING_PAGE = "MISSING_PAGE"
    DUPLICATE_PAGE = "DUPLICATE_PAGE"
    CONTINUATION_PAGE = "CONTINUATION_PAGE"
    SEPARATOR_DETECTED = "SEPARATOR_DETECTED"
    INCOMPLETE_PACKAGE = "INCOMPLETE_PACKAGE"
    WRONG_FAMILY_CMS_GEOMETRY = "WRONG_FAMILY_CMS_GEOMETRY"


@dataclass(frozen=True)
class PageIdentity:
    page_index: int
    page_class: PageClass
    confidence: float
    is_separator: bool = False
    is_continuation: bool = False
    barcode_text: str | None = None
    reasons: tuple[str, ...] = ()
    allows_cms_geometry: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_class": self.page_class.value,
            "confidence": self.confidence,
            "is_separator": self.is_separator,
            "is_continuation": self.is_continuation,
            "barcode_text": self.barcode_text,
            "reasons": list(self.reasons),
            "allows_cms_geometry": self.allows_cms_geometry,
        }


@dataclass
class ClaimPackage:
    package_id: str
    claim_id: str
    pages: list[PageIdentity] = field(default_factory=list)
    issues: list[PackageIssue] = field(default_factory=list)
    complete: bool = False
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "claim_id": self.claim_id,
            "pages": [p.to_dict() for p in self.pages],
            "issues": [i.value for i in self.issues],
            "complete": self.complete,
            "confidence": self.confidence,
        }

    @property
    def claim_pages(self) -> list[PageIdentity]:
        return [
            p
            for p in self.pages
            if not p.is_separator
            and p.page_class
            not in {PageClass.SEPARATOR, PageClass.NON_CLAIM}
        ]


_FAMILY_TO_PAGE = {
    DocumentFamily.CMS1500: PageClass.CMS1500,
    DocumentFamily.UB04: PageClass.UB04,
    DocumentFamily.REIMBURSEMENT_SUPERBILL: PageClass.REIMBURSEMENT_SUPERBILL,
    DocumentFamily.RUNNING_ACCOUNT_STATEMENT: PageClass.RUNNING_ACCOUNT_STATEMENT,
    DocumentFamily.EOB: PageClass.EOB,
    DocumentFamily.SEPARATOR: PageClass.SEPARATOR,
    DocumentFamily.ATTACHMENT: PageClass.ATTACHMENT,
    DocumentFamily.UNKNOWN: PageClass.UNKNOWN,
}


def classify_page_signals(
    *,
    page_index: int,
    form_family: str | None = None,
    ocr_text: str | None = None,
    barcode_text: str | None = None,
    router_label: str | None = None,
    confidence: float = 0.5,
) -> PageIdentity:
    """Classify a page. Prefer OCR/layout text over a forced CMS form_family."""
    classification = classify_document_family(
        ocr_text,
        barcode_text=barcode_text,
    )
    # Explicit router / form_family only when text did not already decide.
    if classification.family is DocumentFamily.UNKNOWN:
        forced = (form_family or router_label or "").strip()
        if forced:
            classification = classify_document_family(
                forced,
                barcode_text=barcode_text,
            )
            if classification.family is DocumentFamily.UNKNOWN:
                # legacy map
                upper = forced.upper().replace("-", "").replace(" ", "")
                legacy = {
                    "CMS1500": DocumentFamily.CMS1500,
                    "UB04": DocumentFamily.UB04,
                    "STATEMENT": DocumentFamily.RUNNING_ACCOUNT_STATEMENT,
                    "SUPERBILL": DocumentFamily.REIMBURSEMENT_SUPERBILL,
                    "REIMBURSEMENTSUPERBILL": DocumentFamily.REIMBURSEMENT_SUPERBILL,
                    "RUNNINGACCOUNTSTATEMENT": DocumentFamily.RUNNING_ACCOUNT_STATEMENT,
                    "EOB": DocumentFamily.EOB,
                    "ATTACHMENT": DocumentFamily.ATTACHMENT,
                    "SEPARATOR": DocumentFamily.SEPARATOR,
                }
                fam = legacy.get(upper)
                if fam is not None:
                    from packages.document_finance.families import FamilyClassification

                    classification = FamilyClassification(
                        fam, max(confidence, 0.7), (f"ROUTER:{forced}",), ()
                    )

    page_class = _FAMILY_TO_PAGE.get(classification.family, PageClass.UNKNOWN)
    text = f"{ocr_text or ''} {barcode_text or ''}".upper()
    continuation = "CONTINUED" in text or "PAGE 2" in text or "CONT." in text
    reasons = list(classification.evidence) or list(classification.anchors) or ["UNKNOWN_PAGE"]
    if continuation:
        reasons.append("CONTINUATION_HINT")

    return PageIdentity(
        page_index=page_index,
        page_class=page_class,
        confidence=max(confidence, classification.confidence)
        if page_class != PageClass.UNKNOWN
        else min(confidence, classification.confidence, 0.4),
        is_separator=page_class is PageClass.SEPARATOR,
        is_continuation=continuation,
        barcode_text=barcode_text,
        reasons=tuple(dict.fromkeys(reasons)),
        allows_cms_geometry=page_class is PageClass.CMS1500,
    )


def build_claim_package(
    *,
    package_id: str,
    claim_id: str,
    pages: list[PageIdentity],
    expected_page_count: int | None = None,
) -> ClaimPackage:
    issues: list[PackageIssue] = []
    for page in pages:
        if page.is_separator:
            issues.append(PackageIssue.SEPARATOR_DETECTED)
        if page.is_continuation:
            issues.append(PackageIssue.CONTINUATION_PAGE)
        if (not page.allows_cms_geometry) and page.page_class not in {
            PageClass.SEPARATOR,
            PageClass.UNKNOWN,
            PageClass.ATTACHMENT,
            PageClass.NON_CLAIM,
        }:
            # Non-CMS claim pages must not be fed CMS ROIs.
            pass

    primary = [
        p
        for p in pages
        if p.page_class
        in {
            PageClass.CMS1500,
            PageClass.UB04,
            PageClass.REIMBURSEMENT_SUPERBILL,
            PageClass.RUNNING_ACCOUNT_STATEMENT,
            PageClass.STATEMENT,
            PageClass.SUPERBILL,
            PageClass.EOB,
        }
    ]
    if (
        len(primary) > 1
        and not any(p.is_continuation for p in primary[1:])
        and len({p.page_index for p in primary}) < len(primary)
    ):
        issues.append(PackageIssue.DUPLICATE_PAGE)

    if expected_page_count is not None and len(pages) < expected_page_count:
        issues.append(PackageIssue.MISSING_PAGE)
        issues.append(PackageIssue.INCOMPLETE_PACKAGE)

    claim_pages = [
        p
        for p in pages
        if p.page_class
        in {
            PageClass.CMS1500,
            PageClass.UB04,
            PageClass.REIMBURSEMENT_SUPERBILL,
            PageClass.RUNNING_ACCOUNT_STATEMENT,
            PageClass.STATEMENT,
            PageClass.SUPERBILL,
            PageClass.EOB,
            PageClass.ATTACHMENT,
        }
        or p.is_continuation
    ]
    complete = bool(claim_pages) and PackageIssue.MISSING_PAGE not in issues
    if not claim_pages:
        issues.append(PackageIssue.INCOMPLETE_PACKAGE)
        complete = False

    conf = sum(p.confidence for p in pages) / len(pages) if pages else 0.0
    return ClaimPackage(
        package_id=package_id,
        claim_id=claim_id,
        pages=list(pages),
        issues=list(dict.fromkeys(issues)),
        complete=complete,
        confidence=float(conf),
    )
