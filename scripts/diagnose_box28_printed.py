#!/usr/bin/env python3
"""Diagnostic bundles for printed CMS-1500 totals still blocked on Box 28.

Uses the locked-50 geometry transforms and the existing Tesseract character
boxes. Does not accept values and does not add an OCR engine.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from packages.geometry_authority.box28 import CMS1500_BOX28_FULL, box28_value_only_bbox
from packages.geometry_authority.monetary_geometry import (
    find_cents_ruling,
    glyphs_from_tesseract_boxes,
    locate_value_band,
    mask_form_lines,
    read_ruled_crop,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "evaluation_results/hackathon_50_box28_v3"
ZIP = ROOT / "data/Hackathon - 1000 Claims.zip"
OUT = ROOT / "evaluation_results/box28_diagnostics"

TARGETS = (
    "DJJM.002",
    "DJJM.005",
    "DJJM.009",
    "DJJM.019",
    "DJJM.023",
    "DJJM.024",
    "DJJM.026",
    "DJJM.027",
    "DJJF.015",
)

# Service-line charge windows already used by the cascade (not claim-specific).
LINE_WINDOWS = (
    ("primary", 1000, 1145),
    ("mid", 1050, 1165),
    ("right", 1100, 1220),
    ("left_bleed", 940, 1100),
)
ROW_Y0 = 1385
ROW_H = 55
HEADER = 55 // 3
MAX_ROWS = 6


def _claim_dir(stem: str) -> Path:
    return RUN / "claims" / f"Group A__M048{stem}"


def _warp(stem: str) -> Image.Image:
    claim = _claim_dir(stem)
    apps = list((claim / "application").glob("application-*"))
    app = apps[0]
    geom = json.loads((app / "GeometryResult.json").read_text())
    telem = json.loads((app / "geometry_telemetry.json").read_text())
    matrix = np.asarray(geom["source_to_geometry_transform"], dtype=float)
    size = (1712, 2214)
    for t in (telem.get("trace") or {}).get("traces") or []:
        for e in t.get("events") or []:
            obs = (e.get("data") or {}).get("coverage_observation")
            if obs and obs.get("stage") == "Input image":
                size = tuple(obs["reference"]["size"])
                break
    with zipfile.ZipFile(ZIP) as archive:
        payload = archive.read(telem["source"]["entry"])
    with Image.open(BytesIO(payload)) as tiff:
        tiff.seek(int(geom.get("page_number") or 1) - 1)
        source = np.asarray(tiff.convert("L"))
    warped = cv2.warpPerspective(source, matrix, size, borderValue=255)
    return Image.fromarray(warped)


def _components(gray: np.ndarray) -> list[dict]:
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 4 or h < 3:
            continue
        out.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "cx": round(float(centroids[i][0]), 1),
                "cy": round(float(centroids[i][1]), 1),
            }
        )
    return out


def _overlay_box(page: Image.Image, bbox, color=(220, 30, 30)) -> Image.Image:
    rgb = page.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    draw.rectangle(bbox, outline=color, width=3)
    return rgb


def _draw_glyphs(crop: Image.Image, glyphs) -> Image.Image:
    rgb = crop.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for g in glyphs:
        draw.rectangle((g.x0, g.y0, g.x1, g.y1), outline=(20, 90, 220), width=1)
    return rgb


def _tess_boxes(image: Image.Image) -> tuple[str, list]:
    import pytesseract

    raw = pytesseract.image_to_boxes(
        image,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.",
    )
    glyphs = glyphs_from_tesseract_boxes(raw, width=image.width, height=image.height)
    return raw, glyphs


def _row_ink_score(gray: np.ndarray) -> int:
    if gray.size == 0:
        return 0
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return int((ink > 0).sum())


def diagnose(stem: str) -> dict:
    page = _warp(stem)
    out = OUT / stem
    out.mkdir(parents=True, exist_ok=True)
    page.save(out / "registered_page.png")
    full = CMS1500_BOX28_FULL
    overlay = _overlay_box(page, full)
    # Box 29 starts ~1250; mark it.
    draw = ImageDraw.Draw(overlay)
    draw.line([(1248, 1780), (1248, 1900)], fill=(30, 120, 30), width=2)
    overlay.save(out / "box28_polygon_overlay.png")

    current = box28_value_only_bbox(full)
    page.crop(current).save(out / "box28_current_crop.png")
    cell = page.crop(full)
    gray = np.asarray(cell.convert("L"))
    y0, y1 = locate_value_band(gray)
    band = cell.crop((0, y0, cell.width, y1))
    band.save(out / "box28_value_band.png")
    cleaned = mask_form_lines(np.asarray(band.convert("L")))
    Image.fromarray(cleaned).save(out / "box28_value_band_masked.png")

    comps = _components(np.asarray(band.convert("L")))
    vis = band.convert("RGB")
    d = ImageDraw.Draw(vis)
    for c in comps:
        d.rectangle((c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]), outline=(200, 40, 40))
    vis.save(out / "box28_components.png")

    raw_boxes, glyphs = _tess_boxes(band)
    _draw_glyphs(band, glyphs).save(out / "box28_char_polygons.png")
    geom = read_ruled_crop(cell)
    ruling = find_cents_ruling(np.asarray(band.convert("L")))

    rejection = None
    for line in (RUN / "results.jsonl").read_text().splitlines():
        row = json.loads(line)
        if stem in row["claim_id"]:
            rejection = {
                "disposition": row["disposition"],
                "blockers": row["critical_blockers"],
                "reasons": (row.get("fields") or {}).get("total_charge", {}).get("reasons"),
                "value": (row.get("fields") or {}).get("total_charge", {}).get("value"),
                "gap": row.get("gap_classes"),
            }
            break

    ocr = json.loads((_claim_dir(stem) / "ocr/OCRCandidates.json").read_text())
    raw_ocr = []
    for field in ocr["fields"]:
        if field.get("field") in {"total_charge", "total_charges"}:
            raw_ocr.append(
                {
                    "region": field.get("ocr_region"),
                    "accept_reason": (field.get("cascade") or {}).get("accept_reason"),
                    "candidates": [
                        {
                            "value": c.get("value"),
                            "raw": c.get("raw_value"),
                            "variant": c.get("preprocessing_variant"),
                        }
                        for c in (field.get("candidates") or [])
                    ],
                }
            )

    line_rows = []
    table_y0 = ROW_Y0 + HEADER
    for i in range(MAX_ROWS):
        y_a = table_y0 + i * ROW_H
        y_b = y_a + ROW_H
        row_dir = out / f"line_{i+1}"
        row_dir.mkdir(exist_ok=True)
        best_ink = 0
        for name, x0, x1 in LINE_WINDOWS:
            crop = page.crop((x0, y_a, x1, y_b))
            ink = _row_ink_score(np.asarray(crop.convert("L")))
            best_ink = max(best_ink, ink)
            if ink < 40 and name != "primary":
                continue
            crop.save(row_dir / f"{name}.png")
            try:
                read = read_ruled_crop(crop)
                cand = read.to_dict()
            except Exception as exc:  # noqa: BLE001
                cand = {"error": str(exc)}
            line_rows.append(
                {
                    "row": i + 1,
                    "window": name,
                    "bbox": [x0, y_a, x1, y_b],
                    "ink": ink,
                    "geometry": cand,
                }
            )
        if best_ink < 80 and i > 0:
            break

    report = {
        "stem": stem,
        "box28_full": list(full),
        "current_crop": list(current),
        "value_band_y": [y0, y1],
        "ruling_x_in_band": ruling,
        "components": comps,
        "glyphs": [
            {"text": g.text, "x0": g.x0, "x1": g.x1, "cx": g.cx, "y0": g.y0, "y1": g.y1}
            for g in glyphs
        ],
        "tesseract_boxes": raw_boxes,
        "geometry": geom.to_dict(),
        "saved_ocr": raw_ocr,
        "authority_rejection": rejection,
        "line_windows": line_rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for stem in TARGETS:
        report = diagnose(stem)
        summary.append(
            {
                "stem": stem,
                "disposition": (report["authority_rejection"] or {}).get("disposition"),
                "value": (report["authority_rejection"] or {}).get("value"),
                "reasons": (report["authority_rejection"] or {}).get("reasons"),
                "geometry": report["geometry"],
                "glyphs": report["glyphs"],
                "ruling_x_in_band": report["ruling_x_in_band"],
                "value_band_y": report["value_band_y"],
                "lines": [
                    {
                        "row": r["row"],
                        "window": r["window"],
                        "candidate": (r["geometry"] or {}).get("geometry_candidate"),
                        "ambiguous": (r["geometry"] or {}).get("ambiguous"),
                        "raw": (r["geometry"] or {}).get("raw_glyph_sequence"),
                    }
                    for r in report["line_windows"]
                    if r["ink"] >= 40
                ],
            }
        )
        print(stem, report["geometry"].get("geometry_candidate"), report["geometry"].get("reasons"))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
