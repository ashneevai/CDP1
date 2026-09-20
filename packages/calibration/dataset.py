"""Leakage-safe calibration dataset builder (development split only)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CalibrationExample:
    field_name: str
    field_type: str
    document_class: str
    ocr_confidence: float
    engine_agreement: float
    geometry_score: float
    ink_quality: float
    preprocessing_type: str
    validation_ok: float
    reconciliation_strength: float
    source_independence: float
    conflict: float
    label_accepted_correct: int  # 1 = correct accept, 0 = should not accept
    split: str = "development"

    def feature_vector(self) -> dict[str, float]:
        return {
            "ocr_confidence": self.ocr_confidence,
            "engine_agreement": self.engine_agreement,
            "geometry_score": self.geometry_score,
            "ink_quality": self.ink_quality,
            "validation_ok": self.validation_ok,
            "reconciliation_strength": self.reconciliation_strength,
            "source_independence": self.source_independence,
            "conflict": self.conflict,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationDataset:
    examples: list[CalibrationExample] = field(default_factory=list)

    def add(self, example: CalibrationExample) -> None:
        if example.split != "development":
            raise ValueError("Frozen validation must not enter calibration training set")
        self.examples.append(example)

    def extend(self, rows: Iterable[CalibrationExample]) -> None:
        for row in rows:
            self.add(row)

    def by_field(self, field_name: str) -> list[CalibrationExample]:
        key = field_name.casefold()
        return [e for e in self.examples if e.field_name.casefold() == key]

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.examples]


def select_threshold_for_precision(
    scores: list[float],
    labels: list[int],
    *,
    target_precision: float = 0.995,
) -> float | None:
    """Pick the lowest threshold meeting precision; None if unmet.

    Does not maximise accuracy — precision constraint first.
    """
    if not scores or len(scores) != len(labels):
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    best: float | None = None
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / max(1, tp + fp)
        if prec >= target_precision and tp > 0:
            best = float(score)
    return best
