"""Document-family financial pipeline: classify → parse → dedupe → reconcile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .discrepancy import DiscrepancyKind, DiscrepancyLedger, DiscrepancyRecord
from .families import DocumentFamily, FamilyClassification, classify_document_family
from .reimbursement_superbill import ReimbursementPageResult, parse_reimbursement_superbill
from .running_statement import LedgerReconcileResult, parse_running_account_statement
from .semantics import PackageFinancialSemantics, build_package_financials
from .transactions import FinancialTransaction


@dataclass
class DocumentFinanceResult:
    classification: FamilyClassification
    cms_geometry_used: bool = False
    reimbursement: ReimbursementPageResult | None = None
    ledger: LedgerReconcileResult | None = None
    package_financials: PackageFinancialSemantics | None = None
    transactions: list[FinancialTransaction] = field(default_factory=list)
    missing_source_fields: list[str] = field(default_factory=list)
    gap_class: str | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.to_dict(),
            "cms_geometry_used": self.cms_geometry_used,
            "reimbursement": self.reimbursement.to_dict() if self.reimbursement else None,
            "ledger": self.ledger.to_dict() if self.ledger else None,
            "package_financials": (
                self.package_financials.to_dict() if self.package_financials else None
            ),
            "transactions": [t.to_dict() for t in self.transactions],
            "missing_source_fields": list(self.missing_source_fields),
            "gap_class": self.gap_class,
            "reasons": list(self.reasons),
            "accepted_total_charges": (
                self.package_financials.total_charges
                if self.package_financials
                else None
            ),
        }


def interpret_page_finance(
    *,
    text: str | None = None,
    barcode_text: str | None = None,
    family: DocumentFamily | FamilyClassification | None = None,
    structured_lines: list[dict[str, Any]] | None = None,
    ledger_rows: list[dict[str, Any]] | None = None,
    printed_ending_balance: object = None,
    page_index: int = 0,
    claim_id: str = "",
    discrepancy_ledger: DiscrepancyLedger | None = None,
) -> DocumentFinanceResult:
    """Route financial interpretation by document family.

    Never invokes CMS Box 28/24F for non-CMS1500 families.
    """
    if isinstance(family, FamilyClassification):
        classification = family
    elif isinstance(family, DocumentFamily):
        classification = FamilyClassification(family, 1.0, ("EXPLICIT_FAMILY",), ())
    elif isinstance(family, str) and family.strip():
        aliases = {
            "CMS1500": DocumentFamily.CMS1500,
            "CMS-1500": DocumentFamily.CMS1500,
            "CMS_1500": DocumentFamily.CMS1500,
            "UB04": DocumentFamily.UB04,
            "UB-04": DocumentFamily.UB04,
            "UB_04": DocumentFamily.UB04,
            "SUPERBILL": DocumentFamily.REIMBURSEMENT_SUPERBILL,
            "REIMBURSEMENT_SUPERBILL": DocumentFamily.REIMBURSEMENT_SUPERBILL,
            "STATEMENT": DocumentFamily.RUNNING_ACCOUNT_STATEMENT,
            "RUNNING_ACCOUNT_STATEMENT": DocumentFamily.RUNNING_ACCOUNT_STATEMENT,
            "EOB": DocumentFamily.EOB,
            "SEPARATOR": DocumentFamily.SEPARATOR,
            "ATTACHMENT": DocumentFamily.ATTACHMENT,
            "UNKNOWN": DocumentFamily.UNKNOWN,
        }
        fam = aliases.get(family.strip().upper().replace(" ", "_"))
        if fam is None:
            classification = classify_document_family(text, barcode_text=barcode_text)
        else:
            classification = FamilyClassification(
                fam, 0.95, (f"EXPLICIT_FAMILY:{family}",), ()
            )
    else:
        classification = classify_document_family(text, barcode_text=barcode_text)

    result = DocumentFinanceResult(classification=classification)
    fam = classification.family

    if fam is DocumentFamily.SEPARATOR:
        result.reasons.append("SEPARATOR_EXCLUDED_FROM_CLAIM_EXTRACTION")
        result.gap_class = None
        return result

    if fam is DocumentFamily.ATTACHMENT:
        result.reasons.append("ATTACHMENT_CONTEXT_ONLY")
        return result

    if fam is DocumentFamily.CMS1500:
        result.cms_geometry_used = True
        result.reasons.append("CMS1500_USES_BOX28_AND_24F")
        return result

    if fam is DocumentFamily.UB04:
        result.reasons.append("UB04_TOTAL_CHARGE_FORM_LOCATOR_REQUIRED")
        result.gap_class = "SOURCE_FIELD_ABSENT" if not structured_lines else None
        return result

    if fam is DocumentFamily.REIMBURSEMENT_SUPERBILL:
        parsed = parse_reimbursement_superbill(
            text or "", structured_lines=structured_lines
        )
        result.reimbursement = parsed
        result.transactions = parsed.as_transactions(page_index=page_index)
        result.missing_source_fields = list(parsed.source_field_absent)
        result.package_financials = build_package_financials(
            transactions=result.transactions,
            printed_page_totals=[parsed.printed_total] if parsed.printed_total else [],
            derived_page_totals=[parsed.derived_total] if parsed.derived_total else [],
            amount_paid=parsed.amount_paid,
        )
        if parsed.totals_agree:
            result.reasons.append("SUPERBILL_PRINTED_TOTAL_ACCEPTED")
        else:
            result.reasons.append("SUPERBILL_NEEDS_REVIEW")
        for field_name in result.missing_source_fields:
            if discrepancy_ledger is not None:
                discrepancy_ledger.add(
                    DiscrepancyRecord(
                        kind=DiscrepancyKind.SOURCE_FIELD_ABSENT,
                        claim_id=claim_id,
                        field_name=field_name,
                        detail="Field not present on reimbursement statement page.",
                    )
                )
            result.gap_class = "SOURCE_FIELD_ABSENT"
        return result

    if fam is DocumentFamily.RUNNING_ACCOUNT_STATEMENT:
        ledger = parse_running_account_statement(
            ledger_rows or [],
            page_index=page_index,
            printed_ending_balance=printed_ending_balance,
        )
        result.ledger = ledger
        result.transactions = list(ledger.transactions)
        result.package_financials = build_package_financials(
            transactions=result.transactions,
            opening_balance=ledger.opening_balance,
            ending_balance=ledger.printed_ending_balance or ledger.ending_balance,
            adjustments=ledger.adjustments,
        )
        if ledger.balanced:
            result.reasons.append("LEDGER_BALANCED_SERVICE_CHARGES_ARE_TOTAL")
        else:
            result.reasons.append("LEDGER_UNBALANCED")
        result.reasons.append("BOTTOM_TOTAL_IS_ENDING_BALANCE_NOT_CHARGE")
        return result

    if fam is DocumentFamily.EOB:
        result.reasons.append("EOB_CONTRACT_FIELDS_REQUIRED")
        return result

    result.reasons.append("UNKNOWN_FAMILY_NO_CMS_GEOMETRY")
    result.gap_class = "WRONG_DOCUMENT_FAMILY"
    if discrepancy_ledger is not None and claim_id:
        discrepancy_ledger.add(
            DiscrepancyRecord(
                kind=DiscrepancyKind.WRONG_DOCUMENT_FAMILY,
                claim_id=claim_id,
                detail="Page family unknown; CMS geometry forbidden.",
            )
        )
    return result
