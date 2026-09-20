"""Reimbursement superbill ("Statement for Insurance Reimbursement") parser.

Locates fields by labels/headers — never CMS Box 28 / 24F geometry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .transactions import (
    FinancialTransaction,
    TransactionType,
    format_money,
    parse_money,
)


@dataclass
class ServiceLine:
    description: str | None = None
    procedure_code: str | None = None
    fee: Decimal | None = None
    quantity: Decimal | None = None
    line_total: Decimal | None = None
    service_date: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def derived_total(self) -> Decimal | None:
        if self.fee is None or self.quantity is None:
            return self.line_total
        return (self.fee * self.quantity).quantize(Decimal("0.01"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "procedure_code": self.procedure_code,
            "fee": format_money(self.fee) if self.fee is not None else None,
            "quantity": str(self.quantity) if self.quantity is not None else None,
            "line_total": (
                format_money(self.line_total) if self.line_total is not None else None
            ),
            "derived_total": (
                format_money(self.derived_total())
                if self.derived_total() is not None
                else None
            ),
            "service_date": self.service_date,
            "raw": dict(self.raw),
        }


@dataclass
class ReimbursementPageResult:
    patient_name: str | None = None
    patient_dob: str | None = None
    provider: str | None = None
    service_lines: list[ServiceLine] = field(default_factory=list)
    page_subtotal: Decimal | None = None
    printed_total: Decimal | None = None
    amount_paid: Decimal | None = None
    derived_total: Decimal | None = None
    totals_agree: bool = False
    amount_paid_confused_with_total: bool = False
    reasons: list[str] = field(default_factory=list)
    source_field_absent: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_name": self.patient_name,
            "patient_dob": self.patient_dob,
            "provider": self.provider,
            "service_lines": [s.to_dict() for s in self.service_lines],
            "page_subtotal": (
                format_money(self.page_subtotal)
                if self.page_subtotal is not None
                else None
            ),
            "printed_total": (
                format_money(self.printed_total)
                if self.printed_total is not None
                else None
            ),
            "amount_paid": (
                format_money(self.amount_paid) if self.amount_paid is not None else None
            ),
            "derived_total": (
                format_money(self.derived_total)
                if self.derived_total is not None
                else None
            ),
            "totals_agree": self.totals_agree,
            "amount_paid_confused_with_total": self.amount_paid_confused_with_total,
            "reasons": list(self.reasons),
            "source_field_absent": list(self.source_field_absent),
            "page_total_charges": (
                format_money(self.printed_total)
                if self.totals_agree and self.printed_total is not None
                else (
                    format_money(self.derived_total)
                    if self.derived_total is not None
                    else None
                )
            ),
        }

    def as_transactions(
        self, *, page_index: int = 0, patient_id: str | None = None, provider_id: str | None = None
    ) -> list[FinancialTransaction]:
        out: list[FinancialTransaction] = []
        for idx, line in enumerate(self.service_lines):
            amount = line.derived_total() or line.line_total
            if amount is None:
                continue
            out.append(
                FinancialTransaction(
                    date=line.service_date,
                    description=line.description,
                    procedure_code=line.procedure_code,
                    amount=amount,
                    fee=line.fee,
                    quantity=line.quantity,
                    line_total=amount,
                    transaction_type=TransactionType.SERVICE_CHARGE,
                    provider_id=provider_id or self.provider,
                    patient_id=patient_id or self.patient_name,
                    page_index=page_index,
                    row_index=idx,
                )
            )
        return out


_MONEY = re.compile(
    r"(?<![\d.])(-?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\(\$?\d+(?:\.\d{2})?\))(?![\d.])"
)
_QTY = re.compile(r"\b(\d+(?:\.\d+)?)\s*[xX×]\s*")
_LABEL_MONEY = re.compile(
    r"(?P<label>SUBTOTAL|TOTAL(?!\s+BALANCE)|AMOUNT\s+PAID|PAID)\s*[:#]?\s*"
    r"(?P<amount>\$?-?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
    re.IGNORECASE,
)


def parse_reimbursement_superbill(
    text: str,
    *,
    structured_lines: list[dict[str, Any]] | None = None,
) -> ReimbursementPageResult:
    """Parse a reimbursement statement from OCR text and/or structured rows.

    ``structured_lines`` entries may include fee, quantity, line_total,
    description, procedure_code, service_date — preferred when present.
    """
    result = ReimbursementPageResult()
    body = text or ""
    upper = body.upper()

    if "STATEMENT FOR INSURANCE REIMBURSEMENT" in upper or "REIMBURSEMENT" in upper:
        result.reasons.append("SUPERBILL_HEADING")

    # Patient / DOB label pulls (best-effort from text).
    name_m = re.search(
        r"(?:Patient(?:\s*Name)?|Member)\s*[:#]?\s*([A-Za-z][A-Za-z ,.'-]{2,60})",
        body,
        re.IGNORECASE,
    )
    if name_m:
        result.patient_name = name_m.group(1).strip(" :#")
    dob_m = re.search(
        r"(?:DOB|Date of Birth|Birth Date)\s*[:#]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        body,
        re.IGNORECASE,
    )
    if dob_m:
        result.patient_dob = dob_m.group(1)
    else:
        result.source_field_absent.append("patient_dob")

    provider_m = re.search(
        r"(?:Provider|Clinician|Therapist)\s*[:#]?\s*([A-Za-z][A-Za-z ,.'-]{2,60})",
        body,
        re.IGNORECASE,
    )
    if provider_m:
        result.provider = provider_m.group(1).strip(" :#")

    lines: list[ServiceLine] = []
    if structured_lines:
        for raw in structured_lines:
            fee = parse_money(raw.get("fee"))
            qty = None
            if raw.get("quantity") is not None:
                try:
                    qty = Decimal(str(raw.get("quantity")))
                except Exception:  # noqa: BLE001
                    qty = None
            line_total = parse_money(raw.get("line_total") or raw.get("amount"))
            if fee is not None and qty is not None:
                derived = (fee * qty).quantize(Decimal("0.01"))
                if line_total is None:
                    line_total = derived
                elif abs(line_total - derived) > Decimal("0.01"):
                    result.reasons.append("LINE_FEE_QTY_MISMATCH")
                    line_total = derived
            lines.append(
                ServiceLine(
                    description=raw.get("description"),
                    procedure_code=raw.get("procedure_code"),
                    fee=fee,
                    quantity=qty,
                    line_total=line_total,
                    service_date=raw.get("service_date"),
                    raw=dict(raw),
                )
            )
    else:
        # Pattern: fee × qty  or  qty × fee  with optional = total
        for match in re.finditer(
            r"(?P<a>\$?\d+(?:\.\d{2})?)\s*[xX×]\s*(?P<b>\d+(?:\.\d+)?)"
            r"(?:\s*=\s*(?P<c>\$?\d+(?:\.\d{2})?))?",
            body,
        ):
            a = parse_money(match.group("a"))
            b_raw = match.group("b")
            c = parse_money(match.group("c")) if match.group("c") else None
            try:
                qty = Decimal(b_raw)
            except Exception:  # noqa: BLE001, S112 -- malformed quantity is not a line item
                continue
            if a is None:
                continue
            # Prefer fee×qty where fee looks like a currency (>= 10) and qty small.
            fee, quantity = (a, qty) if a >= qty else (qty, a)
            if fee < quantity and quantity >= 10:
                fee, quantity = quantity, fee
            derived = (fee * quantity).quantize(Decimal("0.01"))
            lines.append(
                ServiceLine(
                    fee=fee,
                    quantity=quantity,
                    line_total=c or derived,
                    raw={"match": match.group(0)},
                )
            )
        # Also: "3 × 165.00 = 495.00" with qty first
        if not lines:
            for match in re.finditer(
                r"(?P<q>\d+)\s*[xX×]\s*(?P<fee>\$?\d+(?:\.\d{2})?)"
                r"(?:\s*=\s*(?P<tot>\$?\d+(?:\.\d{2})?))?",
                body,
            ):
                fee = parse_money(match.group("fee"))
                qty = Decimal(match.group("q"))
                tot = parse_money(match.group("tot")) if match.group("tot") else None
                if fee is None:
                    continue
                derived = (fee * qty).quantize(Decimal("0.01"))
                lines.append(
                    ServiceLine(
                        fee=fee,
                        quantity=qty,
                        line_total=tot or derived,
                        raw={"match": match.group(0)},
                    )
                )

    result.service_lines = lines
    derived = Decimal("0.00")
    for line in lines:
        piece = line.derived_total() or line.line_total
        if piece is not None:
            derived += piece
    result.derived_total = derived.quantize(Decimal("0.01")) if lines else None

    for match in _LABEL_MONEY.finditer(body):
        label = re.sub(r"\s+", " ", match.group("label").upper())
        amount = parse_money(match.group("amount"))
        if amount is None:
            continue
        if label.startswith("SUBTOTAL"):
            result.page_subtotal = amount
        elif label.startswith("AMOUNT") or label == "PAID":
            result.amount_paid = amount
        elif label.startswith("TOTAL"):
            result.printed_total = amount

    if (
        result.printed_total is not None
        and result.derived_total is not None
        and abs(result.printed_total - result.derived_total) <= Decimal("0.01")
        and lines
    ):
        result.totals_agree = True
        result.reasons.append("PRINTED_TOTAL_EQUALS_LINE_SUM")
    elif result.printed_total is not None and result.derived_total is not None:
        result.reasons.append("PRINTED_TOTAL_LINE_SUM_MISMATCH")

    if (
        result.amount_paid is not None
        and result.printed_total is not None
        and result.amount_paid == result.printed_total
    ):
        # Values may match, but roles stay separate — never collapse.
        result.amount_paid_confused_with_total = False
        result.reasons.append("AMOUNT_PAID_EQUALS_TOTAL_VALUE_BUT_DISTINCT_ROLES")

    if not lines:
        result.reasons.append("NO_SERVICE_LINES")
    return result
