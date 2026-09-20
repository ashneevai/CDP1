"""Governed charge residual ladder from redesign-stack-v1.

Order is fail-closed: local OCR (ruling-aware windows) → tess digits → GPT
empty-finance sweep (local corroboration required) → optional Azure DI →
field-scoped HITL. OpenOCR is not on the ladder. GPT is never sole monetary
authority.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .redesign_stack import load_redesign_stack

DEFAULT_LADDER: tuple[str, ...] = (
    "local_paddle_rapid",
    "tesseract_digits_fill",
    "gpt4o_empty_finance_sweep",
    "azure_di_charge_optional",
    "field_scoped_hitl",
)


@dataclass(frozen=True)
class LadderStep:
    id: str
    index: int
    is_terminal_hitl: bool


def charge_residual_ladder() -> tuple[str, ...]:
    try:
        ladder = load_redesign_stack().charge_residual_ladder
    except Exception:  # noqa: BLE001 — config must not break charge path
        return DEFAULT_LADDER
    return ladder or DEFAULT_LADDER


def ladder_steps() -> tuple[LadderStep, ...]:
    return tuple(
        LadderStep(
            id=step_id,
            index=i,
            is_terminal_hitl=step_id == "field_scoped_hitl",
        )
        for i, step_id in enumerate(charge_residual_ladder())
    )


def next_step_after(completed: Iterable[str]) -> LadderStep | None:
    """Return the first ladder step not yet completed."""
    done = {str(x).strip() for x in completed if str(x).strip()}
    for step in ladder_steps():
        if step.id not in done:
            return step
    return None


def assert_ladder_order(attempted: list[str]) -> bool:
    """True if ``attempted`` is a subsequence of the governed ladder."""
    ladder = list(charge_residual_ladder())
    pos = 0
    for step in attempted:
        try:
            idx = ladder.index(step, pos)
        except ValueError:
            return False
        pos = idx + 1
    return True
