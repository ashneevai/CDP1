"""Geometry-based CMS dollars|cents reconstruction.

Digit identity still comes from an existing reader (character boxes). This
module only decides which glyphs are dollars, which are cents, and whether an
implied decimal is authorised by a validated vertical ruling. It does not
invent glyphs and it does not know claim identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GlyphBox:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class PageGlyph:
    """One digit glyph with crop-local, page, and canonical CMS centres."""

    text: str
    crop_x0: float
    crop_y0: float
    crop_x1: float
    crop_y1: float
    page_polygon: tuple[tuple[float, float], ...]
    canonical_cx: float
    canonical_cy: float
    zone: str  # dollar | cent | unit | outside

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "crop_bbox": [self.crop_x0, self.crop_y0, self.crop_x1, self.crop_y1],
            "page_polygon": [list(pt) for pt in self.page_polygon],
            "canonical_centre": [self.canonical_cx, self.canonical_cy],
            "zone": self.zone,
        }


@dataclass(frozen=True)
class MonetaryGeometryRead:
    raw_glyph_sequence: str
    dollars: str
    cents: str
    decimal_visible: bool
    geometry_candidate: str | None
    ambiguous: bool
    ruling_x: int | None
    reasons: tuple[str, ...]
    page_glyph_polygons: tuple[tuple[tuple[float, float], ...], ...] = ()
    canonical_glyph_centres: tuple[tuple[float, float], ...] = ()
    dollar_glyphs: tuple[str, ...] = ()
    cents_glyphs: tuple[str, ...] = ()
    unit_zone_glyphs: tuple[str, ...] = ()
    canonical_monetary_value: str | None = None

    def to_dict(self) -> dict:
        return {
            "raw_glyph_sequence": self.raw_glyph_sequence,
            "dollars": self.dollars,
            "cents": self.cents,
            "decimal_visible": self.decimal_visible,
            "geometry_candidate": self.geometry_candidate,
            "ambiguous": self.ambiguous,
            "ruling_x": self.ruling_x,
            "reasons": list(self.reasons),
            "page_glyph_polygons": [
                [list(pt) for pt in poly] for poly in self.page_glyph_polygons
            ],
            "canonical_glyph_centres": [list(c) for c in self.canonical_glyph_centres],
            "dollar_glyphs": list(self.dollar_glyphs),
            "cents_glyphs": list(self.cents_glyphs),
            "unit_zone_glyphs": list(self.unit_zone_glyphs),
            "canonical_monetary_value": self.canonical_monetary_value,
        }


def mask_form_lines(gray: np.ndarray) -> np.ndarray:
    """Whiten long horizontal and vertical rulings. Keep short glyph strokes."""
    import cv2

    if gray.ndim != 2 or gray.size == 0:
        return gray
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    height, width = ink.shape
    horiz = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 4), 1)),
    )
    vert = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, height // 2))),
    )
    rules = cv2.bitwise_or(horiz, vert)
    out = gray.copy()
    out[rules > 0] = 255
    return out


def _row_ink(gray: np.ndarray) -> np.ndarray:
    import cv2

    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return (ink > 0).sum(axis=1).astype(float)


def locate_value_band(gray: np.ndarray) -> tuple[int, int]:
    """Return (y0, y1) of the monetary glyph band, excluding caption and rules.

    The caption is the first dense text row. A full-width bottom rule is the
    lower form boundary. The value is the glyph run between those, not a
    hairline grid. One Box 28 fraction does not fit every renderer; the band
    is measured from this cell.
    """
    if gray.size == 0:
        return (0, 0)
    profile = _row_ink(gray)
    height = int(profile.shape[0])
    width = int(gray.shape[1]) if gray.ndim == 2 else 1
    if height < 4:
        return (0, height)
    bottom = None
    for y in range(height - 1, int(height * 0.62), -1):
        if profile[y] > 0.50 * width:
            bottom = y
            break
    cap_end = 0
    seen = False
    for y in range(int(height * 0.50)):
        if profile[y] > max(8.0, 0.15 * width):
            seen = True
            cap_end = y
        elif seen and profile[y] < 6:
            break
    y_start = min(height - 2, cap_end + 2)
    y_end = (bottom - 1) if bottom is not None else height
    glyph = [y for y in range(y_start, max(y_start, y_end)) if 4 <= profile[y] < 0.50 * width]
    if not glyph:
        # Single printed row (synthetic value under a caption, no bottom rule).
        dense = [y for y in range(height) if profile[y] >= max(3.0, float(profile.max()) * 0.25)]
        if len(dense) >= 3:
            return (dense[0], dense[-1] + 1)
        return (max(0, height // 3), height)
    spans: list[tuple[int, int]] = []
    start = prev = glyph[0]
    for y in glyph[1:] + [10**9]:
        if y <= prev + 2:
            prev = y
        else:
            spans.append((start, prev + 1))
            start = prev = y
    spans.sort(key=lambda span: span[1] - span[0], reverse=True)
    y0, y1 = spans[0]
    return (max(0, y0 - 1), min(height, y1 + 1))


def find_cents_ruling(gray: np.ndarray) -> int | None:
    """X of the dollars|cents dashed rule in the right half, if validated."""
    import cv2

    if gray.ndim != 2 or gray.shape[1] < 16 or gray.shape[0] < 6:
        return None
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    height, width = ink.shape
    vert = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, height // 3))),
    )
    col = vert.sum(axis=0).astype(float)
    x_lo, x_hi = int(width * 0.55), int(width * 0.92)
    region = col[x_lo:x_hi]
    if region.size == 0 or float(region.max()) < height * 40:
        return None
    # Rightmost strong column — the cents rule, not a digit stem.
    thresh = float(region.max()) * 0.55
    strong = np.where(region >= thresh)[0]
    if strong.size == 0:
        return None
    return int(strong[-1]) + x_lo


def reconstruct_from_glyphs(
    glyphs: list[GlyphBox],
    *,
    ruling_x: int | None,
    decimal_visible: bool = False,
) -> MonetaryGeometryRead:
    """Split glyphs into dollars and cents using a validated ruling."""
    ordered = sorted(glyphs, key=lambda g: g.cx)
    digits = [g for g in ordered if g.text.isdigit()]
    sequence = "".join(g.text for g in ordered if g.text.strip())
    reasons: list[str] = []
    if ruling_x is None and not decimal_visible:
        return MonetaryGeometryRead(
            sequence,
            "",
            "",
            False,
            None,
            True,
            None,
            ("RULING_NOT_VALIDATED", "AMBIGUOUS"),
        )
    if ruling_x is None and decimal_visible:
        # Visible decimal already encoded in glyph text; do not imply another.
        raw = "".join(g.text for g in ordered if g.text in set("0123456789."))
        parts = raw.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and len(parts[1]) == 2:
            candidate = f"{int(parts[0])}.{parts[1]}"
            return MonetaryGeometryRead(
                sequence,
                parts[0],
                parts[1],
                True,
                candidate,
                False,
                None,
                ("DECIMAL_VISIBLE",),
            )
        return MonetaryGeometryRead(
            sequence, "", "", True, None, True, None, ("DECIMAL_VISIBLE_AMBIGUOUS",)
        )

    left = [g.text for g in digits if g.cx < float(ruling_x)]
    right = [g.text for g in digits if g.cx >= float(ruling_x)]
    dollars = "".join(left)
    cents = "".join(right)
    reasons.append("RULING_VALIDATED")
    if not dollars:
        return MonetaryGeometryRead(
            sequence, dollars, cents, decimal_visible, None, True, ruling_x, ("NO_DOLLAR_GLYPHS", "AMBIGUOUS")
        )
    if len(cents) == 2:
        candidate = f"{int(dollars)}.{cents}"
        reasons.append("CENTS_COLUMN")
        return MonetaryGeometryRead(
            sequence, dollars, cents, decimal_visible, candidate, False, ruling_x, tuple(reasons)
        )
    if cents == "" or set(cents) <= {"0"} and len(cents) <= 2:
        candidate = f"{int(dollars)}.00"
        reasons.append("IMPLIED_DECIMAL_WHOLE_DOLLARS")
        return MonetaryGeometryRead(
            sequence, dollars, "", decimal_visible, candidate, False, ruling_x, tuple(reasons)
        )
    # 1 extra glyph is usually Box 24G units, not a third cent — ambiguous, do not strip.
    reasons.append("CENTS_WIDTH_UNEXPECTED")
    return MonetaryGeometryRead(
        sequence, dollars, cents, decimal_visible, None, True, ruling_x, tuple(reasons) + ("AMBIGUOUS",)
    )


def glyphs_from_tesseract_boxes(
    boxes_text: str,
    *,
    width: int,
    height: int,
) -> list[GlyphBox]:
    """Parse ``image_to_boxes`` output. Origin is bottom-left."""
    glyphs: list[GlyphBox] = []
    for line in (boxes_text or "").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        char, x0, y0, x1, y1 = parts[0], parts[1], parts[2], parts[3], parts[4]
        if char in {"~"}:
            continue
        try:
            left, bottom, right, top = int(x0), int(y0), int(x1), int(y1)
        except ValueError:
            continue
        glyphs.append(
            GlyphBox(
                char,
                left,
                max(0, height - top),
                right,
                max(0, height - bottom),
            )
        )
    return glyphs


def read_ruled_crop(image) -> MonetaryGeometryRead:
    """Label-free value band + ruling split using existing Tesseract character boxes."""
    import pytesseract
    from PIL import Image

    if not hasattr(image, "convert"):
        image = Image.fromarray(image)
    gray = np.asarray(image.convert("L"))
    y0, y1 = locate_value_band(gray)
    band = gray[y0:y1, :] if y1 > y0 else gray
    ruling = find_cents_ruling(band)
    cleaned = mask_form_lines(band)
    width = cleaned.shape[1]
    left_trim = int(width * 0.06)
    glyph_img = cleaned[:, left_trim:] if width > 20 else cleaned
    ruling_adj = None if ruling is None else ruling - left_trim
    if ruling_adj is not None and ruling_adj <= 0:
        ruling_adj = None
    pil = Image.fromarray(glyph_img)
    raw_boxes = pytesseract.image_to_boxes(
        pil,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.",
    )
    glyphs = glyphs_from_tesseract_boxes(
        raw_boxes, width=glyph_img.shape[1], height=glyph_img.shape[0]
    )
    decimal_visible = any(g.text == "." for g in glyphs)
    return reconstruct_from_glyphs(
        [g for g in glyphs if g.text != "."],
        ruling_x=ruling_adj,
        decimal_visible=decimal_visible,
    )


@dataclass(frozen=True)
class TextToken:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0


def template_signature(gray: np.ndarray) -> str:
    """Stable visual signature of a charge cell. Not a claim identifier.

    Quantized value-baseline and cents-column position. Renderers that share
    a signature share a value band; a new signature is not auto-trusted for
    an implied decimal until its ruling geometry validates.
    """
    if gray.size == 0:
        return "empty"
    height = max(1, gray.shape[0])
    width = max(1, gray.shape[1])
    y0, y1 = locate_value_band(gray)
    ruling = find_cents_ruling(gray[y0:y1, :] if y1 > y0 else gray)
    y_bin = round((y0 / height) * 20)
    r_bin = -1 if ruling is None else round((ruling / width) * 20)
    return f"cms-band-y{y_bin}-r{r_bin}"


def _clean_amount_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    # Ruling ticks and confusables that sit on a digit, not a new glyph.
    raw = raw.replace(",", "")
    if raw.startswith((":", ".")):
        raw = raw[1:]
    raw = raw.replace(":", ".").replace("$", "")
    raw = raw.replace("q", "0").replace("Q", "0").replace("g", "0").replace("G", "0")
    raw = raw.replace("O", "0").replace("o", "0")
    kept = []
    for ch in raw:
        if ch.isdigit() or ch == ".":
            kept.append(ch)
    return "".join(kept)


def _shaped_decimal(text: str) -> tuple[str, str] | None:
    parts = text.split(".")
    if (
        len(parts) == 2
        and parts[0]
        and parts[0].isdigit()
        and parts[1].isdigit()
        and len(parts[1]) == 2
    ):
        return parts[0], parts[1]
    return None


def assemble_printed_tokens(
    tokens: list[TextToken],
    *,
    ruling_x: int | None,
    geometry_authorised: bool,
    units_x: float | None = None,
) -> MonetaryGeometryRead:
    """Canonical dollars/cents from independently boxed printed tokens.

    An implied decimal is returned only when the cents column is spatially
    separated (ruling or a >=4px gap before exactly two cent glyphs). A third
    glyph in the cents/units column stays ambiguous — never stripped.
    """
    import re

    kept: list[tuple[str, TextToken]] = []
    for token in sorted(tokens, key=lambda item: item.cx):
        if units_x is not None and token.x0 >= float(units_x) - 1:
            continue
        text = _clean_amount_text(token.text)
        if not text or not re.search(r"\d", text):
            continue
        kept.append((text, token))
    sequence = " ".join(text for text, _ in kept)
    if not kept:
        return MonetaryGeometryRead(
            sequence, "", "", False, None, True, ruling_x, ("NO_GLYPHS", "AMBIGUOUS")
        )

    decimal_hits = [(text, token) for text, token in kept if _shaped_decimal(text)]
    if len(decimal_hits) == 1:
        text, token = decimal_hits[0]
        dollars, cents = _shaped_decimal(text)  # type: ignore[misc]
        reasons = ["DECIMAL_VISIBLE"]
        for other, other_token in kept:
            if other_token.cx >= token.cx:
                continue
            if not re.fullmatch(r"\d{1,3}", other):
                continue
            gap = token.x0 - other_token.x1
            if gap > 18:
                continue
            if dollars.startswith(other):
                continue
            dollars = other + dollars
            reasons.append("LEFT_DIGIT_PREPENDED")
        # Tokens to the right of a finished decimal are Box 24G / Box 29.
        right_digits = [
            other
            for other, other_token in kept
            if other_token.x0 > token.x1 - 1 and re.search(r"\d", other)
        ]
        if right_digits:
            return MonetaryGeometryRead(
                sequence,
                dollars,
                cents,
                True,
                None,
                True,
                ruling_x,
                tuple(reasons) + ("UNITS_OR_BOX29_BLEED", "AMBIGUOUS"),
            )
        candidate = f"{int(dollars)}.{cents}"
        return MonetaryGeometryRead(
            sequence, dollars, cents, True, candidate, False, ruling_x, tuple(reasons)
        )
    if len(decimal_hits) > 1:
        return MonetaryGeometryRead(
            sequence, "", "", True, None, True, ruling_x, ("MULTIPLE_DECIMALS", "AMBIGUOUS")
        )

    # Two printed columns: dollars token(s) and a 2-digit cents token.
    cents_at = None
    for index, (text, _token) in enumerate(kept):
        digits = re.sub(r"\D", "", text)
        if re.fullmatch(r"\d{2}", digits) and "." not in text:
            cents_at = index
    if (
        cents_at is not None
        and cents_at == len(kept) - 1
        and cents_at > 0
        and geometry_authorised
    ):
        cents = re.sub(r"\D", "", kept[cents_at][0])
        left_text = "".join(re.sub(r"\D", "", text) for text, _ in kept[:cents_at])
        gap = kept[cents_at][1].x0 - kept[cents_at - 1][1].x1
        dollar_token = kept[cents_at - 1][1]
        ruling_cuts_dollars = ruling_x is not None and (
            dollar_token.x0 + 4 < float(ruling_x) < dollar_token.x1 - 8
        )
        adjacent = gap <= 16
        if left_text and len(left_text) <= 5 and adjacent and not ruling_cuts_dollars:
            candidate = f"{int(left_text)}.{cents}"
            return MonetaryGeometryRead(
                sequence,
                left_text,
                cents,
                False,
                candidate,
                False,
                ruling_x,
                ("RULING_VALIDATED", "CENTS_COLUMN"),
            )

    # Single digit run. Implied decimal only when the gap before the last
    # two glyphs is a real column gap (>= 4px), not kerning.
    if len(kept) == 1 and geometry_authorised:
        digits = re.sub(r"\D", "", kept[0][0])
        token = kept[0][1]
        if len(digits) >= 4 and token.x1 > token.x0:
            # Character slots are not known; refuse to invent a split.
            return MonetaryGeometryRead(
                sequence,
                "",
                "",
                False,
                None,
                True,
                ruling_x,
                ("DIGIT_RUN_NEEDS_GLYPH_SPLIT", "AMBIGUOUS"),
            )

    return MonetaryGeometryRead(
        sequence,
        "",
        "",
        False,
        None,
        True,
        ruling_x,
        ("RULING_NOT_VALIDATED", "AMBIGUOUS") if not geometry_authorised else ("AMBIGUOUS",),
    )


def align_cents_column(
    blobs: list[tuple[int, int]],
    ticks: list[tuple[int, int]],
    rapid_digits: str,
) -> MonetaryGeometryRead:
    """Split a digit run on a validated cents ruling.

    ``blobs`` and ``ticks`` are ``(x0, x1)`` spans. A narrow ruling tick is not
    a digit. One inserted ``1`` sitting on that tick is dropped. Three-glyph
    runs and a third cents/units blob stay ambiguous.
    """
    ordered = sorted(blobs)
    sequence = rapid_digits
    if len(ordered) < 4:
        return MonetaryGeometryRead(
            sequence, "", "", False, None, True, None, ("GLYPHS_TOO_FEW", "AMBIGUOUS")
        )
    matches: list[tuple[float, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for x0, x1 in ticks:
        cx = (x0 + x1) / 2.0
        right = [span for span in ordered if (span[0] + span[1]) / 2.0 > cx]
        left = [span for span in ordered if (span[0] + span[1]) / 2.0 < cx]
        if len(right) != 2 or len(left) < 2:
            continue
        gap = right[0][0] - left[-1][1]
        if gap < 4:
            continue
        matches.append((cx, left, right))
    unique: list[tuple[float, list[tuple[int, int]], list[tuple[int, int]]]] = []
    seen: set[tuple[int, int]] = set()
    for cx, left, right in matches:
        key = (left[-1][0], right[0][0])
        if key in seen:
            continue
        seen.add(key)
        unique.append((cx, left, right))
    matches = unique
    if len(matches) != 1:
        reason = "UNITS_BLEED" if any(
            sum(1 for span in ordered if (span[0] + span[1]) / 2.0 > (x0 + x1) / 2.0) > 2
            for x0, x1 in ticks
        ) else "RULING_NOT_VALIDATED"
        return MonetaryGeometryRead(
            sequence, "", "", False, None, True, None, (reason, "AMBIGUOUS")
        )
    tick_x, left, right = matches[0]
    tick_x = int(tick_x)
    digits = "".join(ch for ch in rapid_digits if ch.isdigit())
    expected = len(left) + len(right)
    reasons = ["RULING_VALIDATED", "CENTS_GAP", "IMPLIED_DECIMAL"]
    if len(digits) == expected + 1 and digits[len(left) : len(left) + 1] == "1":
        digits = digits[: len(left)] + digits[len(left) + 1 :]
        reasons.append("RULING_TICK_DROPPED")
    if len(digits) != expected:
        return MonetaryGeometryRead(
            sequence, "", "", False, None, True, tick_x, ("GLYPH_TOKEN_MISMATCH", "AMBIGUOUS")
        )
    dollars, cents = digits[: len(left)], digits[len(left) :]
    if not dollars or len(cents) != 2:
        return MonetaryGeometryRead(
            sequence, dollars, cents, False, None, True, tick_x, ("AMBIGUOUS",)
        )
    return MonetaryGeometryRead(
        sequence,
        dollars,
        cents,
        False,
        f"{int(dollars)}.{cents}",
        False,
        tick_x,
        tuple(reasons),
    )


def _component_spans(gray: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Digit blobs and narrow ruling ticks inside the value band."""
    import cv2

    if gray.ndim != 2 or gray.size == 0:
        return [], []
    y0, y1 = locate_value_band(gray)
    band = gray[y0:y1, :] if y1 > y0 else gray
    masked = mask_form_lines(band)
    ink = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink, 8)
    digits: list[tuple[int, int]] = []
    ticks: list[tuple[int, int]] = []
    for index in range(1, count):
        x, _y, width, height, area = (int(v) for v in stats[index])
        if width <= 3 and height >= 6 and area >= 8:
            ticks.append((x, x + width))
            continue
        if 5 <= width <= 22 and height >= 8 and area >= 25 and x >= 8:
            digits.append((x, x + width))
    return digits, ticks


def split_digit_glyphs(glyphs: list[GlyphBox]) -> MonetaryGeometryRead:
    """Implied decimal from the gap before the last two digit glyphs.

    The gap must be a column gap (>= 4px). A wider gap elsewhere does not
    move the cents column. Three glyphs on the cents side stay ambiguous.
    """
    ordered = sorted((g for g in glyphs if g.text.isdigit()), key=lambda g: g.cx)
    sequence = "".join(g.text for g in ordered)
    if len(ordered) < 4:
        return MonetaryGeometryRead(
            sequence, "", "", False, None, True, None, ("GLYPHS_TOO_FEW", "AMBIGUOUS")
        )
    gaps = [ordered[i + 1].x0 - ordered[i].x1 for i in range(len(ordered) - 1)]
    cents_gap = gaps[-2]
    if cents_gap < 4:
        return MonetaryGeometryRead(
            sequence,
            "",
            "",
            False,
            None,
            True,
            None,
            ("CENTS_GAP_NOT_VALIDATED", "AMBIGUOUS"),
        )
    # A third glyph tight against the cents pair is units bleed — do not strip.
    if len(gaps) >= 3 and gaps[-1] >= 4 and cents_gap < 4:
        return MonetaryGeometryRead(
            sequence, "", "", False, None, True, None, ("UNITS_BLEED", "AMBIGUOUS")
        )
    dollars = "".join(g.text for g in ordered[:-2])
    cents = "".join(g.text for g in ordered[-2:])
    if not dollars or len(cents) != 2:
        return MonetaryGeometryRead(
            sequence, dollars, cents, False, None, True, None, ("AMBIGUOUS",)
        )
    ruling = int(ordered[-2].x0)
    return MonetaryGeometryRead(
        sequence,
        dollars,
        cents,
        False,
        f"{int(dollars)}.{cents}",
        False,
        ruling,
        ("RULING_VALIDATED", "CENTS_GAP", "IMPLIED_DECIMAL"),
    )


_RAPID = None


def _rapid_tokens(image) -> list[TextToken]:
    global _RAPID
    from rapidocr_onnxruntime import RapidOCR

    if _RAPID is None:
        _RAPID = RapidOCR()
    scale = 2 if image.width < 420 else 1
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale))
    import numpy as np

    result, _ = _RAPID(np.asarray(image.convert("RGB")))
    tokens: list[TextToken] = []
    for row in result or []:
        if not row or len(row) < 2:
            continue
        box, text = row[0], str(row[1])
        xs = [float(p[0]) / scale for p in box]
        ys = [float(p[1]) / scale for p in box]
        tokens.append(TextToken(text, min(xs), min(ys), max(xs), max(ys)))
    return tokens


def _tess_digit_glyphs(image) -> list[GlyphBox]:
    import pytesseract

    scale = 3
    up = image.resize((max(1, image.width * scale), max(1, image.height * scale)))
    raw = pytesseract.image_to_boxes(
        up,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.",
    )
    glyphs = glyphs_from_tesseract_boxes(raw, width=up.width, height=up.height)
    out: list[GlyphBox] = []
    width = image.width
    for glyph in glyphs:
        x0 = glyph.x0 / scale
        x1 = glyph.x1 / scale
        if x0 < 2:
            continue
        if x1 >= width - 1 and (x1 - x0) < 3:
            continue
        if (x1 - x0) > 24:
            continue
        if glyph.text == "." and (x1 - x0) > 6:
            continue
        out.append(GlyphBox(glyph.text, int(x0), int(glyph.y0 / scale), int(x1), int(glyph.y1 / scale)))
    return out


def page_to_canonical(
    page_x: float,
    page_y: float,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    """Map a page-pixel point into CMS-1500 reference coordinates."""
    from packages.geometry_authority.cms1500_regions import _REF_H, _REF_W

    width, height = image_size
    sx = _REF_W / max(1.0, float(width))
    sy = _REF_H / max(1.0, float(height))
    return (page_x * sx, page_y * sy)


def crop_local_to_page_polygon(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    crop_bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float], ...]:
    """Translate a crop-local box into a page-space rectangle polygon."""
    cx0, cy0, _cx1, _cy1 = crop_bbox
    return (
        (cx0 + x0, cy0 + y0),
        (cx0 + x1, cy0 + y0),
        (cx0 + x1, cy0 + y1),
        (cx0 + x0, cy0 + y1),
    )


def assign_monetary_zone(canonical_cx: float, *, cents_x: float, units_x: float) -> str:
    if canonical_cx >= float(units_x):
        return "unit"
    if canonical_cx >= float(cents_x):
        return "cent"
    ch0, _ch1 = (1030.0, 1207.0)
    try:
        from packages.geometry_authority.cms1500_regions import CMS1500_LINE_COLUMNS

        ch0, _ch1 = CMS1500_LINE_COLUMNS["charges"]
    except Exception:  # noqa: BLE001, S110 -- retain conservative default boundary
        pass
    if canonical_cx < ch0 - 12:
        return "outside"
    return "dollar"


def map_crop_glyphs_to_canonical(
    glyphs: list[GlyphBox],
    *,
    crop_bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    cents_x: float | None = None,
    units_x: float | None = None,
) -> list[PageGlyph]:
    """Map crop-local digit boxes onto canonical CMS centres and zones."""
    from packages.geometry_authority.cms1500_regions import (
        CMS1500_CHARGE_CENTS_X,
        CMS1500_UNITS_X0,
    )

    cents_boundary = CMS1500_CHARGE_CENTS_X if cents_x is None else float(cents_x)
    units_boundary = CMS1500_UNITS_X0 if units_x is None else float(units_x)
    mapped: list[PageGlyph] = []
    for glyph in glyphs:
        if not glyph.text.isdigit():
            continue
        polygon = crop_local_to_page_polygon(
            float(glyph.x0),
            float(glyph.y0),
            float(glyph.x1),
            float(glyph.y1),
            crop_bbox,
        )
        page_cx = sum(pt[0] for pt in polygon) / 4.0
        page_cy = sum(pt[1] for pt in polygon) / 4.0
        canon_cx, canon_cy = page_to_canonical(page_cx, page_cy, image_size)
        zone = assign_monetary_zone(canon_cx, cents_x=cents_boundary, units_x=units_boundary)
        mapped.append(
            PageGlyph(
                text=glyph.text,
                crop_x0=float(glyph.x0),
                crop_y0=float(glyph.y0),
                crop_x1=float(glyph.x1),
                crop_y1=float(glyph.y1),
                page_polygon=polygon,
                canonical_cx=canon_cx,
                canonical_cy=canon_cy,
                zone=zone,
            )
        )
    mapped.sort(key=lambda g: g.canonical_cx)
    return mapped


def reconstruct_from_canonical_glyphs(
    glyphs: list[PageGlyph],
    *,
    cents_x: float | None = None,
    units_x: float | None = None,
) -> MonetaryGeometryRead:
    """Assign dollars/cents from canonical CMS centres — never crop-local x."""
    from packages.geometry_authority.cms1500_regions import (
        CMS1500_CHARGE_CENTS_X,
        CMS1500_UNITS_X0,
    )

    cents_boundary = CMS1500_CHARGE_CENTS_X if cents_x is None else float(cents_x)
    units_boundary = CMS1500_UNITS_X0 if units_x is None else float(units_x)
    ordered = sorted(glyphs, key=lambda g: g.canonical_cx)
    units = [
        g
        for g in ordered
        if assign_monetary_zone(
            g.canonical_cx, cents_x=cents_boundary, units_x=units_boundary
        )
        == "unit"
    ]
    charge_digits = [
        g
        for g in ordered
        if assign_monetary_zone(
            g.canonical_cx, cents_x=cents_boundary, units_x=units_boundary
        )
        in {"dollar", "cent"}
    ]

    def _provenance(dollars: list[PageGlyph], cents: list[PageGlyph]) -> dict:
        zoned = list(dollars) + list(cents) + list(units)
        return {
            "page_glyph_polygons": tuple(g.page_polygon for g in zoned),
            "canonical_glyph_centres": tuple((g.canonical_cx, g.canonical_cy) for g in zoned),
            "dollar_glyphs": tuple(g.text for g in dollars),
            "cents_glyphs": tuple(g.text for g in cents),
            "unit_zone_glyphs": tuple(g.text for g in units),
        }

    def _split_at(boundary: float) -> tuple[list[PageGlyph], list[PageGlyph]]:
        left = [g for g in charge_digits if g.canonical_cx < boundary]
        right = [g for g in charge_digits if g.canonical_cx >= boundary]
        return left, right

    dollars, cents = _split_at(cents_boundary)
    # Fixed form ruling may sit a few px off a given registration. When the
    # fixed boundary does not yield exactly two cent glyphs, pick the split
    # among consecutive charge digits whose boundary is nearest the canonical
    # cents column — but only when that boundary is tight to the printed
    # cents ruling. A loose gap in the dollars column (3000 → 30.00) must not
    # win.
    if (not dollars or len(cents) != 2) and len(charge_digits) >= 4:
        candidates: list[tuple[float, float, list[PageGlyph], list[PageGlyph]]] = []
        for index in range(1, len(charge_digits) - 1):
            right = charge_digits[index:]
            left = charge_digits[:index]
            if len(right) != 2 or not left:
                continue
            gap = right[0].canonical_cx - left[-1].canonical_cx
            if gap < 4:
                continue
            boundary = (left[-1].canonical_cx + right[0].canonical_cx) / 2.0
            # Require the split to sit in the printed cents-column band.
            if abs(boundary - cents_boundary) > 12:
                continue
            if boundary > units_boundary:
                continue
            distance = abs(boundary - cents_boundary)
            candidates.append((distance, -gap, left, right))
        if candidates:
            candidates.sort()
            _dist, _gap, dollars, cents = candidates[0]
            cents_boundary = (dollars[-1].canonical_cx + cents[0].canonical_cx) / 2.0

    raw_digits = "".join(g.text for g in dollars + cents)
    provenance = _provenance(dollars, cents)
    if not dollars or len(cents) != 2:
        return MonetaryGeometryRead(
            raw_digits,
            "".join(g.text for g in dollars),
            "".join(g.text for g in cents),
            False,
            None,
            True,
            int(cents_boundary),
            ("CANONICAL_CENTS_UNRESOLVED", "AMBIGUOUS"),
            canonical_monetary_value=None,
            **provenance,
        )
    dollars_text = "".join(g.text for g in dollars)
    cents_text = "".join(g.text for g in cents)
    if not dollars_text.isdigit() or len(dollars_text) > 5:
        return MonetaryGeometryRead(
            raw_digits,
            dollars_text,
            cents_text,
            False,
            None,
            True,
            int(cents_boundary),
            ("CANONICAL_DOLLARS_UNRESOLVED", "AMBIGUOUS"),
            canonical_monetary_value=None,
            **provenance,
        )
    candidate = f"{int(dollars_text)}.{cents_text}"
    return MonetaryGeometryRead(
        raw_digits,
        dollars_text,
        cents_text,
        False,
        candidate,
        False,
        int(cents_boundary),
        ("CANONICAL_PAGE_CENTS", "RULING_VALIDATED", "CENTS_COLUMN"),
        canonical_monetary_value=candidate,
        **provenance,
    )


def _with_provenance(
    read: MonetaryGeometryRead,
    *,
    page_glyphs: list[PageGlyph] | None = None,
) -> MonetaryGeometryRead:
    if not page_glyphs:
        return read
    dollars = tuple(g.text for g in page_glyphs if g.zone == "dollar")
    cents = tuple(g.text for g in page_glyphs if g.zone == "cent")
    units = tuple(g.text for g in page_glyphs if g.zone == "unit")
    return MonetaryGeometryRead(
        read.raw_glyph_sequence,
        read.dollars,
        read.cents,
        read.decimal_visible,
        read.geometry_candidate,
        read.ambiguous,
        read.ruling_x,
        read.reasons,
        page_glyph_polygons=tuple(g.page_polygon for g in page_glyphs),
        canonical_glyph_centres=tuple((g.canonical_cx, g.canonical_cy) for g in page_glyphs),
        dollar_glyphs=dollars,
        cents_glyphs=cents,
        unit_zone_glyphs=units,
        canonical_monetary_value=read.geometry_candidate if not read.ambiguous else None,
    )


def read_monetary_crop(
    image,
    *,
    units_x: float | None = None,
    crop_bbox: tuple[float, float, float, float] | None = None,
    image_size: tuple[int, int] | None = None,
    cents_x: float | None = None,
) -> MonetaryGeometryRead:
    """Label-free monetary read with optional page→canonical cents assignment.

    When ``crop_bbox`` and ``image_size`` are provided, every digit glyph is
    mapped from crop-local coordinates to canonical CMS centres before
    dollars/cents assignment. Crop-local x is never used as the cents rule.
    """
    from PIL import Image

    from packages.geometry_authority.cms1500_regions import (
        CMS1500_CHARGE_CENTS_X,
        CMS1500_UNITS_X0,
    )

    if not hasattr(image, "convert"):
        image = Image.fromarray(image)
    gray = np.asarray(image.convert("L"))
    y0, y1 = locate_value_band(gray)
    band = gray[y0:y1, :] if y1 > y0 else gray
    ruling = find_cents_ruling(band)
    authorised = (y1 - y0) >= 8 and template_signature(gray) != "empty"
    page_context = crop_bbox is not None and image_size is not None
    cents_boundary = CMS1500_CHARGE_CENTS_X if cents_x is None else float(cents_x)
    units_boundary = (
        float(units_x)
        if units_x is not None
        else (CMS1500_UNITS_X0 if page_context else None)
    )

    # Preferred path: page glyphs → canonical CMS centres → dollars/cents.
    if page_context:
        try:
            local_glyphs = _tess_digit_glyphs(image)
        except Exception:  # noqa: BLE001
            local_glyphs = []
        rapid_digit_check = ""
        try:
            import re as _re_chk

            rapid_digit_check = _re_chk.sub(
                r"\D", "", " ".join(t.text for t in _rapid_tokens(image))
            )
        except Exception:  # noqa: BLE001
            rapid_digit_check = ""
        if len(local_glyphs) >= 3:
            mapped = map_crop_glyphs_to_canonical(
                local_glyphs,
                crop_bbox=crop_bbox,  # type: ignore[arg-type]
                image_size=image_size,  # type: ignore[arg-type]
                cents_x=cents_boundary,
                units_x=units_boundary if units_boundary is not None else CMS1500_UNITS_X0,
            )
            canonical_read = reconstruct_from_canonical_glyphs(
                mapped,
                cents_x=cents_boundary,
                units_x=units_boundary if units_boundary is not None else CMS1500_UNITS_X0,
            )
            if canonical_read.geometry_candidate and not canonical_read.ambiguous:
                # Tess under-reads (420 from 21200) must not beat a fuller Rapid
                # digit string on the same crop. A single trailing units digit on
                # Rapid (49721 vs 4972) is not an under-read.
                geo_digits = "".join(
                    ch for ch in canonical_read.raw_glyph_sequence if ch.isdigit()
                )
                accept_tess = False
                if not rapid_digit_check or rapid_digit_check == geo_digits or len(geo_digits) >= len(rapid_digit_check) or (
                    rapid_digit_check.startswith(geo_digits)
                    and len(rapid_digit_check) <= len(geo_digits) + 1
                ):
                    accept_tess = True
                elif not rapid_digit_check.endswith(geo_digits):
                    # Unrelated Rapid noise — keep the Tess canonical split.
                    accept_tess = True
                if accept_tess:
                    return canonical_read
            # Fall through to crop-local / token readers; keep mapped provenance.
            page_mapped = mapped
        else:
            page_mapped = []
    else:
        page_mapped = []

    rapid_read = None
    try:
        tokens = _rapid_tokens(image)
        # When page context exists, convert token centres to canonical before
        # assembly so cents are not taken from crop-local x.
        if page_context and tokens:
            canon_tokens: list[TextToken] = []
            crop_x0, crop_y0, _c1, _c2 = crop_bbox  # type: ignore[misc]
            for token in tokens:
                page_cx = crop_x0 + token.cx
                page_cy = crop_y0 + (token.y0 + token.y1) / 2.0
                canon_cx, _cy = page_to_canonical(page_cx, page_cy, image_size)  # type: ignore[arg-type]
                # Represent the token at its canonical x so assemble_printed_tokens
                # compares against a canonical ruling, not crop-local pixels.
                shift = canon_cx - token.cx
                canon_tokens.append(
                    TextToken(
                        token.text,
                        token.x0 + shift,
                        token.y0,
                        token.x1 + shift,
                        token.y1,
                    )
                )
            rapid_read = assemble_printed_tokens(
                canon_tokens,
                ruling_x=int(cents_boundary),
                geometry_authorised=authorised,
                units_x=units_boundary,
            )
            if rapid_read.geometry_candidate and not rapid_read.ambiguous:
                return _with_provenance(
                    MonetaryGeometryRead(
                        rapid_read.raw_glyph_sequence,
                        rapid_read.dollars,
                        rapid_read.cents,
                        rapid_read.decimal_visible,
                        rapid_read.geometry_candidate,
                        False,
                        int(cents_boundary),
                        tuple(rapid_read.reasons) + ("CANONICAL_TOKEN_CENTS",),
                        canonical_monetary_value=rapid_read.geometry_candidate,
                    ),
                    page_glyphs=page_mapped,
                )
        else:
            rapid_read = assemble_printed_tokens(
                tokens,
                ruling_x=ruling,
                geometry_authorised=authorised,
                units_x=units_x,
            )
    except Exception:  # noqa: BLE001 -- optional OCR backend failure is non-authoritative
        rapid_read = None
    rapid_digits = ""
    if rapid_read is not None:
        import re

        rapid_digits = re.sub(r"\D", "", rapid_read.raw_glyph_sequence)
    component_read = align_cents_column(*_component_spans(gray), rapid_digits)
    if (
        component_read.geometry_candidate
        and not component_read.ambiguous
        and "RULING_TICK_DROPPED" in component_read.reasons
    ):
        # A dashed cents rule read as ``1`` must not become an extra dollar.
        return _with_provenance(component_read, page_glyphs=page_mapped)
    if rapid_read and rapid_read.geometry_candidate and not rapid_read.ambiguous:
        return _with_provenance(rapid_read, page_glyphs=page_mapped)
    if component_read.geometry_candidate and not component_read.ambiguous:
        return _with_provenance(component_read, page_glyphs=page_mapped)
    glyph_read = split_digit_glyphs(_tess_digit_glyphs(image))
    if glyph_read.geometry_candidate and not glyph_read.ambiguous:
        rapid_digits = ""
        if rapid_read is not None:
            import re

            rapid_digits = re.sub(r"\D", "", rapid_read.raw_glyph_sequence)
        # Implied cents are authorised only when the glyph identities match
        # the token reader's digit string. A crop that drops the leading
        # digit must not invent a different amount.
        if rapid_digits and rapid_digits != glyph_read.raw_glyph_sequence:
            glyph_read = MonetaryGeometryRead(
                glyph_read.raw_glyph_sequence,
                "",
                "",
                False,
                None,
                True,
                glyph_read.ruling_x,
                ("GLYPH_TOKEN_MISMATCH", "AMBIGUOUS"),
            )
        else:
            return _with_provenance(glyph_read, page_glyphs=page_mapped)
    if page_mapped and not (rapid_read and rapid_read.geometry_candidate):
        # Prefer the canonical unresolved read so callers see page provenance.
        unresolved = reconstruct_from_canonical_glyphs(
            page_mapped,
            cents_x=cents_boundary,
            units_x=units_boundary if units_boundary is not None else CMS1500_UNITS_X0,
        )
        # Page-canonical Tess under-read: fall back to crop-local ruling/gap
        # logic that does not depend on page x (still no crop-local cents assign
        # when a page-canonical candidate already won above).
        try:
            local_only = read_monetary_crop(image, units_x=units_x)
        except Exception:  # noqa: BLE001
            local_only = None
        if (
            local_only is not None
            and local_only.geometry_candidate
            and not local_only.ambiguous
        ):
            return _with_provenance(local_only, page_glyphs=page_mapped)
        if rapid_read is not None and rapid_read.raw_glyph_sequence:
            return MonetaryGeometryRead(
                rapid_read.raw_glyph_sequence,
                unresolved.dollars,
                unresolved.cents,
                rapid_read.decimal_visible,
                unresolved.geometry_candidate,
                True,
                unresolved.ruling_x,
                unresolved.reasons,
                page_glyph_polygons=unresolved.page_glyph_polygons,
                canonical_glyph_centres=unresolved.canonical_glyph_centres,
                dollar_glyphs=unresolved.dollar_glyphs,
                cents_glyphs=unresolved.cents_glyphs,
                unit_zone_glyphs=unresolved.unit_zone_glyphs,
                canonical_monetary_value=None,
            )
        return unresolved
    if rapid_read is not None:
        return _with_provenance(rapid_read, page_glyphs=page_mapped)
    return _with_provenance(glyph_read, page_glyphs=page_mapped)
