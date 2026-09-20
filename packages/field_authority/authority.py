"""Field value authority — separate extraction candidates from accept decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuthorityDecision:
    accepted: bool
    reason: str
    gates: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "gates": dict(self.gates),
        }


def accept_field(
    *,
    field_name: str,
    valid_geometry: bool,
    valid_semantics: bool,
    valid_format: bool,
    calibrated_confidence: float | None,
    field_threshold: float,
    independent_evidence: int,
    required_evidence: int,
    unresolved_conflict: bool,
    llm_only: bool = False,
    critical: bool = True,
) -> AuthorityDecision:
    """Critical fields require all gates; LLM-only never accepts when critical."""
    gates = {
        "valid_geometry": bool(valid_geometry),
        "valid_semantics": bool(valid_semantics),
        "valid_format": bool(valid_format),
        "calibrated_confidence": (
            calibrated_confidence is not None
            and float(calibrated_confidence) >= float(field_threshold)
        ),
        "independent_evidence": int(independent_evidence) >= int(required_evidence),
        "no_unresolved_conflict": not bool(unresolved_conflict),
        "not_llm_only": not (critical and llm_only),
    }
    if critical and llm_only:
        return AuthorityDecision(
            accepted=False,
            reason="LLM_VLM_NOT_AUTHORITATIVE",
            gates=gates,
        )
    failed = [name for name, ok in gates.items() if not ok]
    if failed:
        return AuthorityDecision(
            accepted=False,
            reason="AUTHORITY_GATE_FAILED:" + ",".join(failed),
            gates=gates,
        )
    return AuthorityDecision(
        accepted=True,
        reason=f"AUTHORITY_ACCEPTED:{field_name}",
        gates=gates,
    )


def independent_evidence_count(
    candidates: list[Mapping[str, Any]],
    *,
    independence_groups: Mapping[str, str] | None = None,
) -> int:
    """Count distinct independence groups — same crop multi-read is not independent."""
    groups: set[str] = set()
    for cand in candidates:
        eng = str(cand.get("engine") or "").casefold()
        if independence_groups and eng in independence_groups:
            groups.add(independence_groups[eng])
            continue
        try:
            from packages.ocr.independence import independence_group

            groups.add(independence_group(eng))
        except Exception:  # noqa: BLE001
            groups.add(eng or "unknown")
    return len(groups)
