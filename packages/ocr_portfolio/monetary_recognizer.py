"""Specialist monetary crop recognizer — restricted digit vocabulary."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from packages.ocr_portfolio.monetary_variants import CropVariant

_CURRENCY_RE = re.compile(
    r"\$?\d{1,3}(?:,\d{3})*\.\d{2}|\$?\d{1,6}(?:\.\d{2})?|\(\d+\.\d{2}\)|CR\s*\d+\.\d{2}"
)
_WHITELIST = set("0123456789,.$()-CR ")


@dataclass(frozen=True)
class MonetaryRead:
    value: str | None
    raw_text: str
    engine: str
    variant_id: str
    confidence: float
    reason: str


@dataclass
class MonetaryRecognizeResult:
    best: MonetaryRead | None
    attempts: list[MonetaryRead] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": None
            if self.best is None
            else {
                "value": self.best.value,
                "raw_text": self.best.raw_text,
                "engine": self.best.engine,
                "variant_id": self.best.variant_id,
                "confidence": self.best.confidence,
                "reason": self.best.reason,
            },
            "attempts": [
                {
                    "value": a.value,
                    "raw_text": a.raw_text,
                    "engine": a.engine,
                    "variant_id": a.variant_id,
                    "confidence": a.confidence,
                    "reason": a.reason,
                }
                for a in self.attempts
            ],
        }


def _env_on(name: str, default: str = "1") -> bool:
    return (os.environ.get(name) or default).strip().casefold() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def split_charge_at_vertical_ruling(
    crop: Image.Image,
) -> tuple[Image.Image, Image.Image] | None:
    """Split CMS-1500 $CHARGES cell at the dollars|cents dashed vertical ruling.

    Right-shifted windows often include the units column; OCR'ing the full crop
    yields digit soup (64000+1 → 64910). Dollars-only left of the ruling is the
    recoverable ink for whole-dollar typed amounts.
    """
    gray = np.asarray(crop.convert("L"), dtype=np.uint8)
    if gray.size == 0:
        return None
    height, width = gray.shape
    if width < 20 or height < 8:
        return None
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vert = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(4, height // 4))),
    )
    col_sum = vert.sum(axis=0).astype(float)
    # Decimal separator sits in the right half of a charge crop; avoid digit stems.
    x_lo, x_hi = int(width * 0.45), int(width * 0.90)
    if x_hi - x_lo < 4:
        return None
    region = col_sum[x_lo:x_hi]
    if float(region.max()) < float(height) * 255.0 * 0.08:
        col_sum = ink.sum(axis=0).astype(float)
        region = col_sum[x_lo:x_hi]
    if float(region.max()) <= 0:
        return None
    # Rightmost strong peak — dashed separator, not a digit vertical stroke.
    thresh = float(region.max()) * 0.55
    strong = np.where(region >= thresh)[0]
    peak = int(strong[-1]) + x_lo if strong.size else int(np.argmax(region)) + x_lo
    dollars = crop.crop((0, 0, max(1, peak - 1), height))
    cents = crop.crop((min(width - 1, peak + 2), 0, width, height))
    if dollars.width < 6 or cents.width < 2:
        return None
    return dollars, cents


def recover_dollars_from_split_raw(text: str) -> str | None:
    """Recover whole dollars when a space or long tail splits cents from the stem.

    ``640 .101`` and ``640.101`` are dollars ``640`` plus units/ruling noise, not
    ``101.00``. Tight two-digit cents (``640.00``, ``12.50``) are left alone.
    """
    compact = re.sub(r"\s+", " ", text or "").strip()
    match = re.search(r"(\d{2,4})\s*[. ]\s*(\d{2,4})", compact)
    if not match:
        return None
    dollars, tail = match.group(1), match.group(2)
    spanned = compact[match.start() : match.end()]
    if " " not in spanned and len(tail) == 2:
        return None
    try:
        amount = int(dollars)
    except ValueError:
        return None
    if not 1 <= amount <= 99999:
        return None
    return f"{amount}.00"


def _charge_dollars_digits(value: object) -> str:
    text = str(value or "").strip().lstrip("$").replace(",", "")
    if not text:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    return re.sub(r"\D", "", text)


def _charge_digit_tokens(text: object) -> list[str]:
    return re.findall(r"\d+", str(text or ""))


def _is_gpt_charge_candidate(cand: dict) -> bool:
    engine = str(cand.get("engine") or cand.get("producing_engine") or "").casefold()
    return "gpt4o" in engine or "gpt-4o" in engine


def _is_dollars_ruling_candidate(cand: dict) -> bool:
    return "dollars_ruling" in str(cand.get("preprocessing_variant") or "")


def is_ruling_tick_charge(
    candidates: list[dict] | None,
    current: str | None = None,
) -> bool:
    """True when every local digit is an isolated ``1`` (vertical form ruling).

    ``1\\n1\\n1`` and ``111`` from dashed rulings are not a typed charge.
    A real ``$111`` fails closed to HITL rather than becoming a false accept.
    A sibling digit other than 1 (rapid ``111`` beside tess ``4``) keeps the row.
    """
    tokens: list[str] = []
    saw_local = False
    for cand in candidates or []:
        if not isinstance(cand, dict) or _is_gpt_charge_candidate(cand):
            continue
        saw_local = True
        raw_tokens = _charge_digit_tokens(cand.get("raw_value"))
        tokens.extend(raw_tokens or _charge_digit_tokens(cand.get("value")))
    if not saw_local:
        tokens = _charge_digit_tokens(current)
    if not tokens:
        return False
    return all(set(tok) <= {"1"} for tok in tokens)


def ruling_geometry_supports_charge(
    candidates: list[dict] | None,
    target: object,
) -> bool:
    """Vision stem confirmed by the dollars|cents ruling, not by a threshold drop.

    Two geometries count:
    - full-window bleed is the stem plus one units/ruling digit, and the
      dollars-only crop is a prefix of that stem (``2001`` / ``20`` / ``200``);
    - the dollars-only crop dropped the trailing zero next to the dashed
      ruling and no full-window read proposes a different stem (``21`` / ``210``).
    """
    td = _charge_dollars_digits(target)
    if len(td) < 3 or not str(target or "").endswith(".00"):
        return False
    full: list[str] = []
    ruling: list[str] = []
    for cand in candidates or []:
        if not isinstance(cand, dict) or _is_gpt_charge_candidate(cand):
            continue
        digits = _charge_dollars_digits(cand.get("value"))
        if not digits:
            continue
        if _is_dollars_ruling_candidate(cand):
            ruling.append(digits)
        else:
            full.append(digits)
    if not ruling:
        return False
    clipped = td.endswith("0") and td[:-1] in ruling
    if not clipped:
        return False
    bleed = any(len(d) == len(td) + 1 and d.startswith(td) for d in full)
    if bleed:
        return True
    return bool(
        full
        and all(td.startswith(d) and 0 < (len(td) - len(d)) <= 1 for d in full)
    )


def resolve_service_charge(
    current: str | None,
    candidates: list[dict] | None,
) -> tuple[str | None, str]:
    """Pick the printed charge stem from local + vision candidates.

    Returns ``(value, tag)``. ``value is None`` with tag ``RULING_TICK_REJECTED``
    means the row is form ruling, not a service line. No claim-id branches.
    """
    cands = [c for c in (candidates or []) if isinstance(c, dict)]
    if is_ruling_tick_charge(cands, current):
        return None, "RULING_TICK_REJECTED"

    vision = None
    for cand in cands:
        if _is_gpt_charge_candidate(cand) and str(cand.get("value") or "").strip():
            vision = str(cand.get("value")).strip()
            break

    if vision and re.fullmatch(r"\d+\.\d{2}", vision):
        vd, vc = vision.split(".")
        vd_norm = str(int(vd))
        if len(vc) == 2 and vc != "00":
            for cand in cands:
                if _is_gpt_charge_candidate(cand):
                    continue
                groups = _charge_digit_tokens(cand.get("raw_value"))
                norms = []
                for group in groups:
                    try:
                        norms.append(str(int(group)))
                    except ValueError:
                        norms.append(group)
                if vd_norm in norms and vc in groups:
                    return vision, "RULED_CENTS_RECONSTRUCTED"

    if not vision:
        return current, ""

    vd = _charge_dollars_digits(vision)
    full: list[str] = []
    for cand in cands:
        if _is_gpt_charge_candidate(cand) or _is_dollars_ruling_candidate(cand):
            continue
        digits = _charge_dollars_digits(cand.get("value") or cand.get("raw_value"))
        if digits:
            full.append(digits)

    if vd and vd in full and any(
        d.startswith(vd) and len(d) == len(vd) + 1 for d in full
    ):
        shaped = vision if re.fullmatch(r"\d+\.\d{2}", vision) else f"{int(vd)}.00"
        return shaped, "LOCAL_STEM_OVER_BLEED"

    if ruling_geometry_supports_charge(cands, vision) or (
        vd
        and ruling_geometry_supports_charge(cands, f"{int(vd)}.00" if vd.isdigit() else vision)
    ):
        shaped = vision if re.fullmatch(r"\d+\.\d{2}", vision) else f"{int(vd)}.00"
        if _charge_dollars_digits(shaped) == vd:
            return shaped, "RULING_CLIPPED_ZERO_STEM"

    return current, ""


def apply_charge_line_resolution(lines: list[dict] | None) -> list[dict]:
    """Apply stem selection and drop ruling-tick rows. Marks unresolved cents glue.

    A dropped row whose vision read is the kept dollars plus two non-``10``
    cents digits (``49`` beside ``4972``) must not AUTO as whole dollars.
    """
    kept: list[dict] = []
    dropped_digits: list[str] = []
    for line in lines or []:
        if not isinstance(line, dict):
            continue
        value, tag = resolve_service_charge(
            line.get("charges") or line.get("charge_amount"),
            line.get("candidates"),
        )
        if tag == "RULING_TICK_REJECTED":
            for cand in line.get("candidates") or []:
                if not isinstance(cand, dict) or not _is_gpt_charge_candidate(cand):
                    continue
                digits = re.sub(
                    r"\D",
                    "",
                    str(cand.get("raw_value") or cand.get("value") or ""),
                )
                if digits:
                    dropped_digits.append(digits)
            continue
        updated = line
        if tag:
            updated = dict(line)
            if value and value != str(line.get("charges") or ""):
                updated["charges"] = value
                updated["charge_amount"] = value
            updated["router_reason"] = f"{line.get('router_reason') or ''}|{tag}".strip("|")
        kept.append(updated)

    for line in kept:
        amount = str(line.get("charges") or "")
        if not amount.endswith(".00"):
            continue
        td = _charge_dollars_digits(amount)
        if len(td) < 2:
            continue
        for digits in dropped_digits:
            if not digits.startswith(td) or len(digits) != len(td) + 2:
                continue
            tail = digits[len(td) :]
            if tail in {"00", "10"}:
                continue
            flagged = dict(line)
            flagged["cents_unresolved"] = True
            flagged["router_reason"] = (
                f"{line.get('router_reason') or ''}|CENTS_GLUE_UNRESOLVED"
            ).strip("|")
            kept[kept.index(line)] = flagged
            break
    return kept


def prefer_charge_ink_amount(a: str | None, b: str | None) -> str | None:
    """Prefer clean whole-dollar reads over units/ruling-bleed digit soup.

    Digit-drop twins (64 vs 640) keep the longer stem. Units bleed (640.00 vs
    649.10 / 260.10) keeps the .00 form when the dollar stem is shared.
    """
    if not a:
        return b
    if not b:
        return a
    if a == b:
        return a

    def _digits(text: str) -> str:
        return re.sub(r"\D", "", text)

    def _dollars(text: str) -> str:
        return _digits(text.split(".", 1)[0])

    da, db = _dollars(a), _dollars(b)
    full_a, full_b = _digits(a), _digits(b)

    # Ruling-tail before digit-drop: 260.00 vs 2605.00 (extra 1/4/5 from dash OCR).
    if (
        a.endswith(".00")
        and da
        and db.startswith(da)
        and len(db) == len(da) + 1
        and db[-1] in {"1", "4", "5"}
    ):
        return a
    if (
        b.endswith(".00")
        and db
        and da.startswith(db)
        and len(da) == len(db) + 1
        and da[-1] in {"1", "4", "5"}
    ):
        return b

    # Digit-drop twins on dollar stems: prefer longer (64 ⊂ 640).
    # Ruling-tail (+1/4/5) is already handled above; do not re-apply a shorter
    # preference here or untagged pairs like 25⊂251 collapse the real stem.
    if da and db and da != db and len(da) <= 4 and len(db) <= 4:
        if db.startswith(da) and len(db) > len(da):
            return b
        if da.startswith(db) and len(da) > len(db):
            return a

    # Same dollar stem → prefer whole dollars.
    if da and db and da == db:
        if a.endswith(".00") and not b.endswith(".00"):
            return a
        if b.endswith(".00") and not a.endswith(".00"):
            return b

    # Units/cents soup beyond a clean .00 stem: 640.00 vs 64910 / 649.10
    if (
        a.endswith(".00")
        and 2 <= len(da) <= 4
        and full_b.startswith(da)
        and len(full_b) >= len(da) + 2
        and not (2 <= len(db) <= 4 and db.startswith(da) and len(db) > len(da))
    ):
        return a
    if (
        b.endswith(".00")
        and 2 <= len(db) <= 4
        and full_a.startswith(db)
        and len(full_a) >= len(db) + 2
        and not (2 <= len(da) <= 4 and da.startswith(db) and len(da) > len(db))
    ):
        return b

    # Unrelated stems must not lose to a shorter ".00" fragment (640.01 vs 101.00).
    if da and db and da != db and not da.startswith(db) and not db.startswith(da):
        return a

    score_a = (2 if a.endswith(".00") else 0) + (1 if 2 <= len(da) <= 4 else 0)
    score_b = (2 if b.endswith(".00") else 0) + (1 if 2 <= len(db) <= 4 else 0)
    if score_a != score_b:
        return a if score_a > score_b else b
    return a


def shape_dollars_ruling_amount(text: str) -> str | None:
    """Shape a dollars|cents-ruling left crop as whole dollars only.

    The dollars-only crop has no cents column. Implied-decimal shaping
    (``21240`` → ``212.40``) invents cents from ruling/cents-column bleed and
    must not win over a full-window ``212.00``. Trailing ruling-tail digits
    ``1/4/5`` are stripped; five-digit leftovers are rejected as soup.
    """
    digits = re.sub(r"\D", "", text or "")
    if not digits or not (2 <= len(digits) <= 5):
        return None
    stem = digits
    if len(stem) >= 3 and stem[-1] in {"1", "4", "5"}:
        stem = stem[:-1]
    if not (2 <= len(stem) <= 4):
        return None
    try:
        amount = int(stem)
    except ValueError:
        return None
    if not 1 <= amount <= 99999:
        return None
    return f"{amount}.00"


def shape_monetary(text: str) -> str | None:
    recovered = recover_dollars_from_split_raw(text)
    if recovered:
        return recovered
    cleaned = "".join(ch for ch in (text or "") if ch in _WHITELIST).strip()
    if not cleaned:
        return None
    candidates: list[str] = []
    m = _CURRENCY_RE.search(cleaned.replace(" ", ""))
    if m:
        amount = m.group(0).lstrip("$").replace(",", "")
        if amount.upper().startswith("CR"):
            amount = amount[2:].strip()
        if amount.startswith("(") and amount.endswith(")"):
            amount = amount[1:-1]
        if "." in amount and re.fullmatch(r"\d+\.\d{2}", amount):
            # Already decimal-shaped — still strip units/ruling tails on .00 forms
            # (2701.00 from units "1" bleed → 270.00).
            if amount.endswith(".00"):
                dollars = amount.split(".", 1)[0]
                if (
                    len(dollars) >= 4
                    and dollars[-1] in {"1", "4", "5"}
                    and 2 <= len(dollars) - 1 <= 4
                ):
                    trimmed = f"{dollars[:-1]}.00"
                    try:
                        if 1.0 <= float(trimmed) <= 99999.99:
                            return trimmed
                    except ValueError:
                        pass
            return amount
        if "." not in amount and re.fullmatch(r"\d{2,6}", amount):
            # Defer to ranked digit-soup handling below (6404 → 640.00).
            cleaned = amount
    digits = re.sub(r"\D", "", cleaned)
    if not digits:
        return None
    if re.fullmatch(r"\d{2,6}", digits):
        candidates = [f"{digits}.00"]
        if len(digits) >= 4:
            candidates.append(f"{digits[:-2]}.{digits[-2:]}")
            # Ruling-tail noise (4/1/5) and units-column bleed (...10).
            if digits[-1] in {"4", "1", "5"}:
                candidates.append(f"{digits[:-1]}.00")
            if len(digits) >= 5 and digits.endswith("10"):
                candidates.append(f"{digits[:-2]}.00")
        ranked: list[tuple[float, str]] = []
        for cand in candidates:
            try:
                val = float(cand)
            except ValueError:
                continue
            if not re.fullmatch(r"\d+\.\d{2}", cand):
                continue
            if not (1.0 <= val <= 99999.99):
                continue
            dollars = cand.split(".", 1)[0]
            score = 2.0 if 2 <= len(dollars) <= 4 else 1.0
            if cand.endswith(".00"):
                score += 0.5
            if len(digits) >= 4 and digits[-1] in {"4", "1", "5"} and dollars == digits[:-1]:
                score += 2.0
            if dollars == digits and len(digits) >= 4 and digits[-1] in {"4", "1", "5"}:
                score -= 2.0
            if len(digits) >= 5 and digits.endswith("10") and dollars == digits[:-2] and cand.endswith(".00"):
                score += 2.5
            if cand.endswith(".10") and len(digits) >= 5 and digits.endswith("10"):
                score -= 1.5
            ranked.append((score, cand))
        if ranked:
            ranked.sort(key=lambda x: (-x[0], len(x[1])))
            return ranked[0][1]
    return None


def monetary_variants_extended(crop: Image.Image) -> list[CropVariant]:
    """Governed variants including nearest-neighbour, bicubic, adaptive threshold."""
    base = crop.convert("RGB")
    out: list[CropVariant] = [CropVariant("original", base)]
    w, h = base.size
    gray = ImageOps.grayscale(base)

    for scale in (2, 4):
        nw, nh = max(1, w * scale), max(1, h * scale)
        out.append(
            CropVariant(
                f"nn_{scale}x",
                gray.resize((nw, nh), Image.Resampling.NEAREST).convert("RGB"),
            )
        )
        out.append(
            CropVariant(
                f"bicubic_{scale}x",
                gray.resize((nw, nh), Image.Resampling.BICUBIC).convert("RGB"),
            )
        )

    out.append(CropVariant("inverted", ImageOps.invert(base.convert("RGB"))))
    arr = np.asarray(gray, dtype=np.uint8)
    adaptive = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    out.append(CropVariant("adaptive_threshold", Image.fromarray(adaptive).convert("RGB")))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    out.append(CropVariant("morph_close", Image.fromarray(closed).convert("RGB")))
    horiz = cv2.morphologyEx(
        arr,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 4), 1)),
    )
    out.append(
        CropVariant("line_removal", Image.fromarray(cv2.subtract(arr, horiz)).convert("RGB"))
    )
    pad_x, pad_y = max(1, w // 20), max(1, h // 20)
    out.append(
        CropVariant(
            "expanded_context",
            ImageOps.expand(base, border=(pad_x, pad_y), fill=(255, 255, 255)),
        )
    )
    split = split_charge_at_vertical_ruling(base)
    if split is not None:
        dollars_crop, _cents_crop = split
        out.append(CropVariant("dollars_left_of_ruling", dollars_crop.convert("RGB")))
        # Upscaled dollars-only — recovers 640 when full-cell OCR reads 64910.
        dw, dh = dollars_crop.size
        out.append(
            CropVariant(
                "dollars_left_of_ruling_3x",
                dollars_crop.resize(
                    (max(1, dw * 3), max(1, dh * 3)), Image.Resampling.NEAREST
                ).convert("RGB"),
            )
        )
    # Isolated connected components (largest ink blob).
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n_labels > 1:
        # Skip background label 0; pick largest component.
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        x, y, bw_w, bw_h, _ = stats[idx]
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(arr.shape[1], x + bw_w + pad), min(arr.shape[0], y + bw_h + pad)
        if x1 > x0 and y1 > y0:
            blob = gray.crop((x0, y0, x1, y1)).convert("RGB")
            out.append(CropVariant("connected_component", blob))
    return out


EngineFn = Callable[[Image.Image], tuple[str, float]]


def recognize_monetary_crop(
    crop: Image.Image,
    *,
    engines: dict[str, EngineFn] | None = None,
    max_variants: int | None = None,
) -> MonetaryRecognizeResult:
    """Run controlled variants × engines; return best shaped monetary value."""
    if not _env_on("CDP_MONETARY_VARIANTS", "1"):
        return MonetaryRecognizeResult(best=None, attempts=[])

    variants = monetary_variants_extended(crop)
    if max_variants is not None:
        variants = variants[: max(1, max_variants)]

    engine_map = engines or {}
    attempts: list[MonetaryRead] = []
    best: MonetaryRead | None = None

    for variant in variants:
        for eng_name, fn in engine_map.items():
            try:
                text, conf = fn(variant.image)
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    MonetaryRead(
                        value=None,
                        raw_text="",
                        engine=eng_name,
                        variant_id=variant.variant_id,
                        confidence=0.0,
                        reason=f"ERROR:{type(exc).__name__}",
                    )
                )
                continue
            shaped = shape_monetary(text or "")
            read = MonetaryRead(
                value=shaped,
                raw_text=text or "",
                engine=eng_name,
                variant_id=variant.variant_id,
                confidence=float(conf or 0.0),
                reason="SHAPED" if shaped else "UNSHAPED",
            )
            attempts.append(read)
            if shaped and (best is None or conf > best.confidence):
                best = read
            if shaped and _env_on("CDP_MONETARY_VARIANTS_EARLY_STOP", "1"):
                return MonetaryRecognizeResult(best=best, attempts=attempts)

    return MonetaryRecognizeResult(best=best, attempts=attempts)
