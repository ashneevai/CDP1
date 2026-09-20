"""CMS-1500 semantic geometry authority for Box 24B / 24F / 28.

Never accept a POS code (e.g. 11 / 11.00) as a service-line charge unless the
crop is proven inside Box 24F and outside Box 24B.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Reference dimensions from config/table_templates/cms1500_service_lines.yaml
_REF_W = 1712.0
_REF_H = 2214.0

# Column bands on the service-line strip (x0, x1) in reference pixels.
CMS1500_LINE_COLUMNS: dict[str, tuple[float, float]] = {
    "place_of_service": (418.0, 480.0),  # Box 24B
    "charges": (1030.0, 1207.0),  # Box 24F
}

# Dollars|cents vertical ruling inside Box 24F / Box 28 (reference pixels).
# Between the dollars column and the two-digit cents column on the printed form.
CMS1500_CHARGE_CENTS_X = 1155.0
# Box 24G units column start — glyphs here are not monetary.
CMS1500_UNITS_X0 = 1207.0

# Box 28 total charge on cms1500_v03 (absolute page coords).
# Prior (1335,1755,1465,1811) landed in Box 29 / NPI and cut off the value band.
CMS1500_BOX28: tuple[float, float, float, float] = (1045.0, 1805.0, 1248.0, 1875.0)

# Common CMS POS codes that OCR often emits as currency (11.00).
_POS_LIKE_AMOUNTS = frozenset(
    {
        "11.00",
        "12.00",
        "21.00",
        "22.00",
        "23.00",
        "24.00",
        "25.00",
        "26.00",
        "31.00",
        "32.00",
        "33.00",
        "34.00",
        "41.00",
        "42.00",
        "50.00",
        "51.00",
        "52.00",
        "53.00",
        "61.00",
        "62.00",
        "65.00",
        "71.00",
        "72.00",
        "81.00",
        "99.00",
    }
)


@dataclass(frozen=True)
class RegionVerdict:
    authorised: bool
    region: str
    overlap_pos: float
    overlap_charge: float
    reason: str


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(a0, b0)
    right = min(a1, b1)
    if right <= left:
        return 0.0
    width = max(1e-6, a1 - a0)
    return float((right - left) / width)


def scale_bbox_to_reference(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    w, h = image_size
    sx = _REF_W / max(1.0, float(w))
    sy = _REF_H / max(1.0, float(h))
    x0, y0, x1, y1 = bbox
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def charge_region_verdict(
    bbox: tuple[float, float, float, float],
    *,
    image_size: tuple[int, int] | None = None,
    already_reference: bool = False,
) -> RegionVerdict:
    """Decide whether a crop is inside Box 24F (charges) vs Box 24B (POS)."""
    if image_size and not already_reference:
        x0, _y0, x1, _y1 = scale_bbox_to_reference(bbox, image_size)
    else:
        x0, _y0, x1, _y1 = bbox
    pos0, pos1 = CMS1500_LINE_COLUMNS["place_of_service"]
    ch0, ch1 = CMS1500_LINE_COLUMNS["charges"]
    overlap_pos = _overlap_1d(x0, x1, pos0, pos1)
    overlap_charge = _overlap_1d(x0, x1, ch0, ch1)
    if overlap_charge >= 0.45 and overlap_charge >= overlap_pos:
        return RegionVerdict(
            authorised=True,
            region="BOX_24F",
            overlap_pos=overlap_pos,
            overlap_charge=overlap_charge,
            reason="CHARGE_GEOMETRY_OK",
        )
    if overlap_pos >= 0.35 and overlap_pos > overlap_charge:
        return RegionVerdict(
            authorised=False,
            region="BOX_24B",
            overlap_pos=overlap_pos,
            overlap_charge=overlap_charge,
            reason="CHARGE_GEOMETRY_POS_BLEED",
        )
    if overlap_charge < 0.2 and overlap_pos < 0.2:
        return RegionVerdict(
            authorised=False,
            region="UNKNOWN",
            overlap_pos=overlap_pos,
            overlap_charge=overlap_charge,
            reason="CHARGE_GEOMETRY_UNAUTHORISED",
        )
    return RegionVerdict(
        authorised=overlap_charge > overlap_pos,
        region="BOX_24F" if overlap_charge > overlap_pos else "AMBIGUOUS",
        overlap_pos=overlap_pos,
        overlap_charge=overlap_charge,
        reason="CHARGE_GEOMETRY_WEAK" if overlap_charge <= overlap_pos else "CHARGE_GEOMETRY_OK",
    )


def is_pos_like_currency(value: object) -> bool:
    text = str(value or "").strip().lstrip("$").replace(",", "")
    if text in _POS_LIKE_AMOUNTS:
        return True
    # Bare POS codes sometimes land in charge fields.
    return bool(text.isdigit() and 1 <= int(text) <= 99)


def reject_pos_as_charge(
    value: object,
    bbox: tuple[float, float, float, float] | None = None,
    *,
    image_size: tuple[int, int] | None = None,
    geometry_region: str | None = None,
) -> tuple[bool, str]:
    """Return (reject, reason). True means do not accept as charge/total."""
    if not is_pos_like_currency(value):
        return False, "NOT_POS_LIKE"
    if geometry_region and geometry_region.upper() in {"BOX_24B", "PLACE_OF_SERVICE", "POS"}:
        return True, "POS_CODE_IN_CHARGE_FIELD"
    if bbox is not None:
        verdict = charge_region_verdict(bbox, image_size=image_size)
        if not verdict.authorised or verdict.reason == "CHARGE_GEOMETRY_POS_BLEED":
            return True, verdict.reason
        # Even inside 24F, a lone POS-like amount needs other corroboration —
        # callers decide; we only hard-reject on POS geometry.
        if verdict.region == "BOX_24B":
            return True, "CHARGE_GEOMETRY_POS_BLEED"
    # No geometry: still reject POS-like as *total_charge* singleton (caller).
    return False, "POS_LIKE_NEEDS_CORROBORATION"


def box28_contains(
    bbox: tuple[float, float, float, float],
    *,
    image_size: tuple[int, int] | None = None,
) -> bool:
    if image_size:
        x0, y0, x1, y1 = scale_bbox_to_reference(bbox, image_size)
    else:
        x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = CMS1500_BOX28
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def any_authorised_charge_bbox(
    bboxes: Iterable[tuple[float, float, float, float]],
    *,
    image_size: tuple[int, int] | None = None,
) -> bool:
    return any(
        charge_region_verdict(b, image_size=image_size).authorised for b in bboxes
    )
