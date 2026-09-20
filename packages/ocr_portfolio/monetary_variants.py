"""Controlled monetary crop variants for the OCR portfolio."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class CropVariant:
    variant_id: str
    image: Image.Image


def monetary_crop_variants(crop: Image.Image) -> list[CropVariant]:
    """Generate governed variants — not uncontrolled serial OCR spam."""
    base = crop.convert("RGB")
    out: list[CropVariant] = [CropVariant("original", base)]

    w, h = base.size
    for scale, vid in ((2, "scale_2x"), (4, "scale_4x")):
        out.append(
            CropVariant(
                vid,
                base.resize((max(1, w * scale), max(1, h * scale)), Image.Resampling.LANCZOS),
            )
        )

    out.append(CropVariant("inverted", ImageOps.invert(base.convert("RGB"))))
    gray = ImageOps.grayscale(base).convert("RGB")
    out.append(CropVariant("grayscale", gray))

    arr = np.asarray(ImageOps.grayscale(base), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    out.append(CropVariant("morph_close", Image.fromarray(closed).convert("RGB")))

    # Horizontal-line suppression (form rulings).
    horiz = cv2.morphologyEx(
        arr,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 4), 1)),
    )
    suppressed = cv2.subtract(arr, horiz)
    out.append(CropVariant("hline_suppressed", Image.fromarray(suppressed).convert("RGB")))

    # Slight expand is caller-side on page; here pad 4% canvas.
    pad_x, pad_y = max(1, w // 25), max(1, h // 25)
    padded = ImageOps.expand(base, border=(pad_x, pad_y), fill=(255, 255, 255))
    out.append(CropVariant("expanded_pad", padded))

    return out


def iter_variant_ids() -> Iterator[str]:
    yield from (
        "original",
        "scale_2x",
        "scale_4x",
        "inverted",
        "grayscale",
        "morph_close",
        "hline_suppressed",
        "expanded_pad",
        "column_context",  # produced by caller with wider bbox
    )
