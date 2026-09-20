"""Deterministic financial reconciliation dispositions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from packages.claim_evidence.line_sum_authority import (
    format_currency,
    is_suspicious_tiny_total,
    line_sum_auto_eligible,
    line_sum_total,
    observed_line_charges,
    parse_currency,
)
from packages.geometry_authority import is_pos_like_currency, reject_pos_as_charge


class FinancialDisposition(StrEnum):
    DIRECT_TOTAL_CORROBORATED = "DIRECT_TOTAL_CORROBORATED"
    LINE_TOTALS_RECONCILED = "LINE_TOTALS_RECONCILED"
    CROSS_PAGE_TOTAL_CORROBORATED = "CROSS_PAGE_TOTAL_CORROBORATED"
    LINE_SUM_UNCORROBORATED = "LINE_SUM_UNCORROBORATED"
    TOTAL_CONFLICT = "TOTAL_CONFLICT"
    INCOMPLETE_SERVICE_LINES = "INCOMPLETE_SERVICE_LINES"
    CHARGE_COLUMN_UNVERIFIED = "CHARGE_COLUMN_UNVERIFIED"
    POS_BLEED_REJECTED = "POS_BLEED_REJECTED"
    EMPTY_FINANCIAL_INK = "EMPTY_FINANCIAL_INK"
    SEPARATOR_EXCLUDED = "SEPARATOR_EXCLUDED"
    FAMILY_CONTEXT_ONLY = "FAMILY_CONTEXT_ONLY"
    WRONG_FAMILY_NO_CMS_GEOMETRY = "WRONG_FAMILY_NO_CMS_GEOMETRY"


@dataclass(frozen=True)
class FinancialReconcileResult:
    disposition: FinancialDisposition
    accepted_total: str | None
    line_sum: str | None
    reasons: tuple[str, ...]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "accepted_total": self.accepted_total,
            "line_sum": self.line_sum,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


def _unique_complete_line_charges(service_lines: list[dict] | None) -> list[Decimal]:
    """Sum unique complete service-line charges; exclude POS-like bleed."""
    seen: set[str] = set()
    charges: list[Decimal] = []
    for line in service_lines or []:
        if not isinstance(line, dict):
            continue
        # Exclude payment/adjustment/balance-shaped keys if present as only signal.
        if line.get("column_role") in {"payment", "adjustment", "balance", "place_of_service"}:
            continue
        raw = None
        for key in ("charges", "charge_amount", "total_charges", "total_charge"):
            if line.get(key) is not None:
                raw = line.get(key)
                break
        parsed = parse_currency(raw)
        if parsed is None:
            continue
        # Geometry / POS bleed gate.
        bbox = line.get("charge_bbox") or line.get("bbox")
        region = line.get("semantic_region") or line.get("authorised_semantic_region")
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            reject, _reason = reject_pos_as_charge(
                raw,
                (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                geometry_region=str(region) if region else None,
            )
            if reject:
                continue
        elif is_pos_like_currency(raw) and str(region or "").upper() in {
            "BOX_24B",
            "PLACE_OF_SERVICE",
            "POS",
        }:
            continue
        key = format_currency(parsed)
        # Distinct service rows may share the same amount (260+260). Prefer
        # explicit row identity; never collapse solely by currency string.
        row_id = str(
            line.get("row_id")
            or line.get("line_index")
            or line.get("line_number")
            or f"anon:{len(seen)}:{key}"
        )
        if row_id in seen:
            continue
        seen.add(row_id)
        charges.append(parsed)
    return charges


def reconcile_claim_total(
    *,
    box28_value: object = None,
    service_lines: list[dict] | None = None,
    charge_column_verified: bool = False,
    all_service_rows_detected: bool = False,
    continuation_pages_included: bool = True,
    material_pages_accounted: bool = True,
    independent_evidence_paths: int = 0,
    cross_page_total: object = None,
) -> FinancialReconcileResult:
    """Deterministic monetary reconciliation with explicit dispositions."""
    details: dict[str, Any] = {}
    raw_lines = [ln for ln in (service_lines or []) if isinstance(ln, dict)]
    raw_pos_like = []
    for line in raw_lines:
        raw = None
        for key in ("charges", "charge_amount", "total_charges", "total_charge"):
            if line.get(key) is not None:
                raw = line.get(key)
                break
        if raw is None:
            continue
        region = str(line.get("semantic_region") or line.get("authorised_semantic_region") or "")
        bbox = line.get("charge_bbox") or line.get("bbox")
        reject = False
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            reject, _ = reject_pos_as_charge(
                raw,
                (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                geometry_region=region or None,
            )
        elif is_pos_like_currency(raw) and region.upper() in {
            "BOX_24B",
            "PLACE_OF_SERVICE",
            "POS",
        }:
            reject = True
        if reject or (
            is_pos_like_currency(raw)
            and region.upper() in {"BOX_24B", "PLACE_OF_SERVICE", "POS"}
        ):
            raw_pos_like.append(raw)

    unique_charges = _unique_complete_line_charges(service_lines)
    details["unique_line_count"] = len(unique_charges)
    details["raw_observed_line_count"] = len(observed_line_charges(service_lines))
    details["pos_bleed_rejected_lines"] = len(raw_pos_like)

    # Hard reject when the only monetary signal is POS bleed (e.g. 11.00 from 24B).
    if (
        not unique_charges
        and raw_pos_like
        and parse_currency(box28_value) is None
    ):
        return FinancialReconcileResult(
            disposition=FinancialDisposition.POS_BLEED_REJECTED,
            accepted_total=None,
            line_sum=None,
            reasons=("POS_LIKE_ONLY_SIGNAL", "M048DJJF_013_GUARD"),
            details=details,
        )

    if (
        len(unique_charges) == 1
        and is_pos_like_currency(format_currency(unique_charges[0]))
        and parse_currency(box28_value) is None
        and independent_evidence_paths < 2
    ):
        return FinancialReconcileResult(
            disposition=FinancialDisposition.POS_BLEED_REJECTED,
            accepted_total=None,
            line_sum=format_currency(unique_charges[0]),
            reasons=("POS_LIKE_SINGLE_LINE_WITHOUT_BOX28", "M048DJJF_013_GUARD"),
            details=details,
        )

    if not charge_column_verified and unique_charges:
        # Still allow corroborated paths below, but flag when unverified alone.
        details["charge_column_verified"] = False

    if not all_service_rows_detected and unique_charges:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.INCOMPLETE_SERVICE_LINES,
            accepted_total=None,
            line_sum=format_currency(sum(unique_charges, Decimal(0))) if unique_charges else None,
            reasons=("SERVICE_ROWS_INCOMPLETE",),
            details=details,
        )

    if unique_charges and not charge_column_verified:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.CHARGE_COLUMN_UNVERIFIED,
            accepted_total=None,
            line_sum=format_currency(sum(unique_charges, Decimal(0))),
            reasons=("CHARGE_COLUMN_UNVERIFIED",),
            details=details,
        )

    line_sum = (
        format_currency(sum(unique_charges, Decimal(0))) if unique_charges else line_sum_total(service_lines)
    )
    box = parse_currency(box28_value)
    cross = parse_currency(cross_page_total)

    if box is not None and is_suspicious_tiny_total(box):
        box = None
        details["box28_suppressed"] = "CURRENCY_SUSPICIOUS_TINY"

    if not unique_charges and box is None:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.EMPTY_FINANCIAL_INK,
            accepted_total=None,
            line_sum=None,
            reasons=("NO_LINE_CHARGES", "NO_BOX28"),
            details=details,
        )

    if not material_pages_accounted or not continuation_pages_included:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.INCOMPLETE_SERVICE_LINES,
            accepted_total=None,
            line_sum=line_sum,
            reasons=("MATERIAL_OR_CONTINUATION_PAGES_MISSING",),
            details=details,
        )

    # Direct total corroborated by line sum within $0.01.
    if box is not None and line_sum is not None:
        ls = parse_currency(line_sum)
        if (
            ls is not None
            and abs(box - ls) <= Decimal("0.01")
            and (independent_evidence_paths >= 1 or charge_column_verified)
        ):
            return FinancialReconcileResult(
                disposition=FinancialDisposition.DIRECT_TOTAL_CORROBORATED,
                accepted_total=format_currency(box),
                line_sum=line_sum,
                reasons=("BOX28_MATCHES_LINE_SUM",),
                details=details,
            )
        if ls is not None and abs(box - ls) > Decimal("0.01"):
            return FinancialReconcileResult(
                disposition=FinancialDisposition.TOTAL_CONFLICT,
                accepted_total=None,
                line_sum=line_sum,
                reasons=("BOX28_LINE_SUM_MISMATCH",),
                details={**details, "box28": format_currency(box), "delta": float(abs(box - ls))},
            )

    if cross is not None and line_sum is not None:
        ls = parse_currency(line_sum)
        if ls is not None and abs(cross - ls) <= Decimal("0.01"):
            return FinancialReconcileResult(
                disposition=FinancialDisposition.CROSS_PAGE_TOTAL_CORROBORATED,
                accepted_total=format_currency(cross),
                line_sum=line_sum,
                reasons=("CROSS_PAGE_MATCHES_LINE_SUM",),
                details=details,
            )

    eligible, gate_reason = line_sum_auto_eligible(
        service_lines,
        box28_value=box28_value,
    )
    # line_sum_auto_eligible already encodes fail-closed dual-engine / gpt+local /
    # box-28 corroboration — including SINGLE_LINE_* paths. Do not re-impose a
    # raw line-count path gate that would mark eligible single-line claims as
    # uncorroborated while the cascade AUTOs them.
    if eligible and line_sum is not None:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.LINE_TOTALS_RECONCILED,
            accepted_total=line_sum,
            line_sum=line_sum,
            reasons=(gate_reason or "LINE_SUM_AUTO_ELIGIBLE", "INDEPENDENT_PATHS_OK"),
            details={**details, "independent_evidence_paths": independent_evidence_paths},
        )

    if line_sum is not None:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.LINE_SUM_UNCORROBORATED,
            accepted_total=None,
            line_sum=line_sum,
            reasons=(gate_reason or "LINE_SUM_NEEDS_CORROBORATION",),
            details=details,
        )

    return FinancialReconcileResult(
        disposition=FinancialDisposition.EMPTY_FINANCIAL_INK,
        accepted_total=None,
        line_sum=None,
        reasons=("NO_RECOVERABLE_TOTAL",),
        details=details,
    )


def reconcile_by_document_family(
    *,
    document_family: str | None,
    document_finance: dict[str, Any] | None = None,
    box28_value: object = None,
    service_lines: list[dict] | None = None,
    charge_column_verified: bool = False,
    all_service_rows_detected: bool = False,
    independent_evidence_paths: int = 0,
) -> FinancialReconcileResult:
    """Dispatch financial authority by document family.

    CMS1500 keeps Box 28 / 24F reconcile. Other families never invent a CMS total.
    """
    family = (document_family or "").upper().replace("-", "").replace(" ", "_")
    finance = document_finance or {}
    details: dict[str, Any] = {"document_family": document_family or "UNKNOWN"}

    if family in {"SEPARATOR"}:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.SEPARATOR_EXCLUDED,
            accepted_total=None,
            line_sum=None,
            reasons=("SEPARATOR_NOT_A_CLAIM_PAGE",),
            details=details,
        )

    if family in {"ATTACHMENT"}:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.FAMILY_CONTEXT_ONLY,
            accepted_total=None,
            line_sum=None,
            reasons=("ATTACHMENT_CONTEXT_ONLY",),
            details=details,
        )

    if family in {"REIMBURSEMENT_SUPERBILL", "SUPERBILL"}:
        pkg = finance.get("package_financials") or {}
        reb = finance.get("reimbursement") or {}
        total = pkg.get("total_charges") or reb.get("page_total_charges")
        derived = reb.get("derived_total")
        if reb.get("totals_agree") and total:
            return FinancialReconcileResult(
                disposition=FinancialDisposition.LINE_TOTALS_RECONCILED,
                accepted_total=str(total),
                line_sum=str(derived or total),
                reasons=("SUPERBILL_PRINTED_TOTAL_EQUALS_LINE_SUM", "FAMILY_PARSER"),
                details={**details, "amount_paid": reb.get("amount_paid")},
            )
        if derived:
            return FinancialReconcileResult(
                disposition=FinancialDisposition.LINE_SUM_UNCORROBORATED,
                accepted_total=None,
                line_sum=str(derived),
                reasons=("SUPERBILL_PRINTED_TOTAL_NOT_CORROBORATED",),
                details=details,
            )
        return FinancialReconcileResult(
            disposition=FinancialDisposition.EMPTY_FINANCIAL_INK,
            accepted_total=None,
            line_sum=None,
            reasons=("SUPERBILL_NO_SERVICE_LINES",),
            details=details,
        )

    if family in {"RUNNING_ACCOUNT_STATEMENT", "STATEMENT"}:
        pkg = finance.get("package_financials") or {}
        ledger = finance.get("ledger") or {}
        total = pkg.get("total_charges") or ledger.get("total_charges_from_ledger")
        if ledger.get("balanced") and total:
            return FinancialReconcileResult(
                disposition=FinancialDisposition.LINE_TOTALS_RECONCILED,
                accepted_total=str(total),
                line_sum=str(total),
                reasons=(
                    "LEDGER_EQUATION_BALANCED",
                    "ENDING_BALANCE_NOT_TOTAL_CHARGE",
                    "FAMILY_PARSER",
                ),
                details={
                    **details,
                    "ending_balance": ledger.get("printed_ending_balance")
                    or ledger.get("ending_balance"),
                    "opening_balance": ledger.get("opening_balance"),
                },
            )
        if total:
            return FinancialReconcileResult(
                disposition=FinancialDisposition.LINE_SUM_UNCORROBORATED,
                accepted_total=None,
                line_sum=str(total),
                reasons=("LEDGER_UNBALANCED",),
                details=details,
            )
        return FinancialReconcileResult(
            disposition=FinancialDisposition.EMPTY_FINANCIAL_INK,
            accepted_total=None,
            line_sum=None,
            reasons=("LEDGER_NO_SERVICE_CHARGES",),
            details=details,
        )

    if family in {"UB04", "EOB"}:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.INCOMPLETE_SERVICE_LINES,
            accepted_total=None,
            line_sum=None,
            reasons=(f"{family}_PARSER_REQUIRED", "NO_CMS_GEOMETRY"),
            details=details,
        )

    if family and family not in {"CMS1500", "CMS_1500", "UNKNOWN", ""}:
        return FinancialReconcileResult(
            disposition=FinancialDisposition.WRONG_FAMILY_NO_CMS_GEOMETRY,
            accepted_total=None,
            line_sum=None,
            reasons=("CMS_GEOMETRY_FORBIDDEN_FOR_FAMILY",),
            details=details,
        )

    # CMS1500 / UNKNOWN → existing Box 28 + 24F path.
    return reconcile_claim_total(
        box28_value=box28_value,
        service_lines=service_lines,
        charge_column_verified=charge_column_verified,
        all_service_rows_detected=all_service_rows_detected,
        independent_evidence_paths=independent_evidence_paths,
    )
