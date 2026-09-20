"""Decode QR / barcode evidence from a page image for package mapping.

CMS-1500 pages in the Hackathon corpus typically encode the NUCC form URL
(``http://www.nucc.org/``). That is family-authenticity evidence, not member
identity. Separator barcodes may contain ``SEP`` tokens used by document-family
classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QrDecodeResult:
    texts: tuple[str, ...]
    detected: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "texts": list(self.texts),
            "detected": self.detected,
            "barcode_text": self.texts[0] if self.texts else None,
            "reasons": list(self.reasons),
        }


def decode_page_qr(image: Any) -> QrDecodeResult:
    """Decode QR codes from a PIL image or numpy array. Never invents payloads."""
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        return QrDecodeResult((), False, (f"OPENCV_UNAVAILABLE:{type(exc).__name__}",))

    if hasattr(image, "convert"):
        arr = np.asarray(image.convert("L"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    detector = cv2.QRCodeDetector()
    texts: list[str] = []
    reasons: list[str] = []
    try:
        ok, decoded, _pts, _ = detector.detectAndDecodeMulti(arr)
        if ok and decoded:
            texts.extend(str(t).strip() for t in decoded if str(t or "").strip())
            reasons.append("QR_MULTI_DECODE")
    except Exception:  # noqa: BLE001, S110 -- fall back to single-code decode
        pass
    if not texts:
        try:
            value, pts, _ = detector.detectAndDecode(arr)
            if value and str(value).strip():
                texts.append(str(value).strip())
                reasons.append("QR_SINGLE_DECODE")
            elif pts is not None:
                reasons.append("QR_DETECTED_UNDECODED")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"QR_DECODE_ERROR:{type(exc).__name__}")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return QrDecodeResult(tuple(ordered), bool(ordered), tuple(reasons) or ("QR_NONE",))


def qr_supports_cms1500_family(texts: tuple[str, ...] | list[str]) -> bool:
    """True when decoded QR is the NUCC CMS-1500 form marker."""
    for text in texts:
        key = str(text or "").strip().casefold()
        if "nucc.org" in key or "cms-1500" in key or "cms1500" in key:
            return True
    return False
