"""ROI-level image evidence analysis and empty-OCR disposition taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import cv2
import numpy as np
from PIL import Image


class InkDisposition(StrEnum):
    BLANK_CONFIRMED = "BLANK_CONFIRMED"
    INK_PRESENT_UNREADABLE = "INK_PRESENT_UNREADABLE"
    PIXELS_MISSING = "PIXELS_MISSING"
    ROI_MISALIGNED = "ROI_MISALIGNED"
    FIELD_NOT_APPLICABLE = "FIELD_NOT_APPLICABLE"
    OCR_EMPTY_UNCLASSIFIED = "OCR_EMPTY_UNCLASSIFIED"
    INK_OBSERVED = "INK_OBSERVED"


@dataclass(frozen=True)
class RoiImageEvidence:
    disposition: InkDisposition
    blank_probability: float
    ink_density: float
    blur_score: float
    contrast: float
    connected_components: int
    clipping_score: float
    estimated_dpi: float
    skew_deg: float
    one_bit_ink_loss_risk: float
    features: dict[str, float]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "blank_probability": self.blank_probability,
            "ink_density": self.ink_density,
            "blur_score": self.blur_score,
            "contrast": self.contrast,
            "connected_components": self.connected_components,
            "clipping_score": self.clipping_score,
            "estimated_dpi": self.estimated_dpi,
            "skew_deg": self.skew_deg,
            "one_bit_ink_loss_risk": self.one_bit_ink_loss_risk,
            "features": dict(self.features),
            "reasons": list(self.reasons),
        }


def analyze_roi(
    image: Image.Image,
    *,
    field_optional: bool = False,
    geometry_valid: bool = True,
    ocr_empty: bool = False,
    field_applicable: bool = True,
) -> RoiImageEvidence:
    """Classify ROI ink state. Confirmed optional blanks must not force HITL."""
    if not field_applicable:
        return _result(
            InkDisposition.FIELD_NOT_APPLICABLE,
            blank_probability=1.0,
            ink_density=0.0,
            blur_score=0.0,
            contrast=0.0,
            connected_components=0,
            clipping_score=0.0,
            estimated_dpi=0.0,
            skew_deg=0.0,
            one_bit_ink_loss_risk=0.0,
            features={},
            reasons=("FIELD_NOT_APPLICABLE",),
        )

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.size == 0:
        return _result(
            InkDisposition.PIXELS_MISSING,
            blank_probability=1.0,
            ink_density=0.0,
            blur_score=0.0,
            contrast=0.0,
            connected_components=0,
            clipping_score=1.0,
            estimated_dpi=0.0,
            skew_deg=0.0,
            one_bit_ink_loss_risk=1.0,
            features={"area": 0.0},
            reasons=("EMPTY_ROI_PIXELS",),
        )

    height, width = gray.shape
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())
    contrast = float(np.clip(gray.std() / 64.0, 0.0, 1.0))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_density = float(np.count_nonzero(ink) / ink.size)
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    connected = max(0, int(num_labels) - 1)
    # Form guideline dashes: many short horizontal components, almost no vertical mass.
    ruling_only = False
    glyph_density = ink_density
    if connected >= 3 and height > 0 and width > 0:
        horiz = cv2.morphologyEx(
            ink,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, width // 6), 1)),
        )
        vert = cv2.morphologyEx(
            ink,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, height // 3))),
        )
        glyph = cv2.subtract(cv2.subtract(ink, horiz), vert)
        glyph_density = float(np.count_nonzero(glyph) / glyph.size)
        horiz_ratio = float(np.count_nonzero(horiz) / max(1, np.count_nonzero(ink)))
        # Box-28 dashed underline: high horizontal ink share, tiny residual glyphs.
        if horiz_ratio >= 0.55 and glyph_density < 0.008 and connected >= 4:
            ruling_only = True
        # Also: components are all wide+short (dash aspect) — form underline.
        if not ruling_only and connected >= 5:
            dash_like = 0
            for i in range(1, num_labels):
                w_i = int(stats[i, cv2.CC_STAT_WIDTH])
                h_i = int(stats[i, cv2.CC_STAT_HEIGHT])
                if w_i >= 2 * max(1, h_i) and h_i <= max(3, height // 3):
                    dash_like += 1
            if dash_like / connected >= 0.7:
                ruling_only = True
    border = max(1, round(min(height, width) * 0.05))
    border_ink = np.concatenate(
        (
            ink[:border].ravel(),
            ink[-border:].ravel(),
            ink[:, :border].ravel(),
            ink[:, -border:].ravel(),
        )
    )
    clipping = float(np.count_nonzero(border_ink) / max(1, border_ink.size))
    estimated_dpi = float(round(max(width / 0.8, height / 0.25), 1))  # field crop heuristic
    # One-bit loss: near-binary with very low density but residual speckles.
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    hist /= max(1.0, hist.sum())
    extremes = float(hist[:8].sum() + hist[-8:].sum())
    one_bit_risk = float(np.clip(extremes * (1.0 - min(1.0, ink_density * 20.0)), 0.0, 1.0))
    blank_probability = float(
        np.clip(1.0 - (glyph_density * 40.0) - (0.02 * connected), 0.0, 1.0)
    )
    if ruling_only:
        blank_probability = max(blank_probability, 0.93)
    features = {
        "width": float(width),
        "height": float(height),
        "ink_density": ink_density,
        "glyph_density": float(glyph_density),
        "connected_components": float(connected),
        "blur_score": blur_score,
        "contrast": contrast,
        "clipping": clipping,
        "blank_probability": blank_probability,
        "one_bit_ink_loss_risk": one_bit_risk,
        "ruling_only": 1.0 if ruling_only else 0.0,
    }
    reasons: list[str] = []

    if not geometry_valid:
        reasons.append("GEOMETRY_INVALID")
        disposition = InkDisposition.ROI_MISALIGNED
    elif ruling_only:
        reasons.append("FORM_RULING_ONLY")
        disposition = InkDisposition.BLANK_CONFIRMED
    elif blank_probability >= 0.92 and connected <= 2 and glyph_density < 0.01:
        reasons.append("LOW_INK_DENSITY")
        disposition = InkDisposition.BLANK_CONFIRMED
    elif ocr_empty and glyph_density >= 0.012 and connected >= 3:
        reasons.append("INK_WITHOUT_OCR")
        if blur_score < 40 or contrast < 0.2 or one_bit_risk >= 0.55:
            reasons.append("DEGRADED_INK")
            disposition = InkDisposition.INK_PRESENT_UNREADABLE
        else:
            disposition = InkDisposition.INK_PRESENT_UNREADABLE
    elif ocr_empty and clipping > 0.35 and glyph_density < 0.01:
        reasons.append("EDGE_CLIPPING")
        disposition = InkDisposition.PIXELS_MISSING
    elif ocr_empty:
        reasons.append("OCR_EMPTY")
        disposition = InkDisposition.OCR_EMPTY_UNCLASSIFIED
    else:
        disposition = InkDisposition.INK_OBSERVED
        reasons.append("INK_OBSERVED")

    if field_optional and disposition == InkDisposition.BLANK_CONFIRMED:
        reasons.append("OPTIONAL_BLANK_NO_HITL")

    return _result(
        disposition,
        blank_probability=blank_probability,
        ink_density=ink_density,
        blur_score=blur_score,
        contrast=contrast,
        connected_components=connected,
        clipping_score=clipping,
        estimated_dpi=estimated_dpi,
        skew_deg=0.0,
        one_bit_ink_loss_risk=one_bit_risk,
        features=features,
        reasons=tuple(reasons),
    )


def requires_field_hitl(evidence: RoiImageEvidence, *, field_optional: bool) -> bool:
    """Optional confirmed blanks do not create unnecessary HITL."""
    if evidence.disposition == InkDisposition.FIELD_NOT_APPLICABLE:
        return False
    if field_optional and evidence.disposition == InkDisposition.BLANK_CONFIRMED:
        return False
    return evidence.disposition in {
        InkDisposition.INK_PRESENT_UNREADABLE,
        InkDisposition.PIXELS_MISSING,
        InkDisposition.ROI_MISALIGNED,
        InkDisposition.OCR_EMPTY_UNCLASSIFIED,
    }


def _result(
    disposition: InkDisposition,
    *,
    blank_probability: float,
    ink_density: float,
    blur_score: float,
    contrast: float,
    connected_components: int,
    clipping_score: float,
    estimated_dpi: float,
    skew_deg: float,
    one_bit_ink_loss_risk: float,
    features: dict[str, float],
    reasons: tuple[str, ...],
) -> RoiImageEvidence:
    return RoiImageEvidence(
        disposition=disposition,
        blank_probability=blank_probability,
        ink_density=ink_density,
        blur_score=blur_score,
        contrast=contrast,
        connected_components=connected_components,
        clipping_score=clipping_score,
        estimated_dpi=estimated_dpi,
        skew_deg=skew_deg,
        one_bit_ink_loss_risk=one_bit_ink_loss_risk,
        features=features,
        reasons=reasons,
    )
