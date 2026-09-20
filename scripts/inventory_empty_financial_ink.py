#!/usr/bin/env python3
"""Build EMPTY_FINANCIAL_INK failure inventory + recoverability ceiling.

Reopens Hackathon ZIP pages using saved geometry ROIs from a 50-claim ops run.
Does not invent ground truth. Writes diagnostics under --out-dir.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def _load_rows(results_jsonl: Path) -> list[dict]:
    return [json.loads(line) for line in results_jsonl.read_text().splitlines() if line.strip()]


def _claim_dir(claims_root: Path, document: str) -> Path:
    slug = document.replace("/", "__")
    return claims_root / slug


def _find_geometry(claim_dir: Path) -> tuple[dict, dict] | None:
    apps = list((claim_dir / "application").glob("application-*"))
    if not apps:
        return None
    app = apps[0]
    geom = json.loads((app / "GeometryResult.json").read_text(encoding="utf-8"))
    telem = json.loads((app / "geometry_telemetry.json").read_text(encoding="utf-8"))
    return geom, telem


def _warp_page(zip_path: Path, telem: dict, geom: dict) -> Image.Image:
    matrix = np.asarray(
        geom.get("source_to_geometry_transform")
        or telem["registration_evidence"]["transform_matrix"],
        dtype=float,
    )
    size = None
    for t in (telem.get("trace") or {}).get("traces") or []:
        for e in t.get("events") or []:
            obs = (e.get("data") or {}).get("coverage_observation")
            if obs and obs.get("stage") == "Input image":
                size = tuple(obs["reference"]["size"])
                break
    if size is None:
        # Fallback from geometry fields image size if present.
        size = (1700, 2200)
    with zipfile.ZipFile(zip_path) as archive:
        payload = archive.read(telem["source"]["entry"])
    with Image.open(BytesIO(payload)) as tiff:
        tiff.seek(int(geom.get("page_number") or 1) - 1)
        source = np.asarray(tiff.convert("L"))
    warped = cv2.warpPerspective(source, matrix, size, borderValue=255)
    return Image.fromarray(warped)


def _total_bbox(geom: dict) -> tuple[int, int, int, int] | None:
    for field in geom.get("fields") or []:
        name = str(field.get("field") or field.get("field_name") or "").casefold()
        if name not in {"total_charge", "total_charges"}:
            continue
        result = field.get("result") or field
        roi = (
            (result.get("aligned_roi") if isinstance(result, dict) else None)
            or field.get("aligned_roi")
            or field.get("ocr_region")
            or field.get("canonical_region")
        )
        if isinstance(roi, dict):
            return (
                int(roi["x0"]),
                int(roi["y0"]),
                int(roi["x1"]),
                int(roi["y1"]),
            )
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            return tuple(int(v) for v in roi)  # type: ignore[return-value]
    return None


def _expand(bbox: tuple[int, int, int, int], page: Image.Image, frac: float = 0.35):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return (
        max(0, int(x0 - frac * w)),
        max(0, int(y0 - frac * h)),
        min(page.width, int(x1 + frac * w)),
        min(page.height, int(y1 + frac * h)),
    )


def _table_crop_box(page: Image.Image) -> tuple[int, int, int, int]:
    # Approximate CMS-1500 service table band in reference space.
    return (
        max(0, int(0.04 * page.width)),
        max(0, int(0.62 * page.height)),
        min(page.width, int(0.96 * page.width)),
        min(page.height, int(0.84 * page.height)),
    )


def _classify(
    *,
    evidence: dict,
    monetary_value: str | None,
    service_line_count: int,
    ocr_attempts: list,
) -> str:
    disp = str(evidence.get("disposition") or "")
    if disp == "BLANK_CONFIRMED" and service_line_count == 0 and not monetary_value:
        return "BLANK_CONFIRMED"
    if disp == "PIXELS_MISSING":
        return "PIXELS_MISSING"
    if disp == "ROI_MISALIGNED":
        return "ROI_MISALIGNED"
    if service_line_count > 0 and not monetary_value:
        return "SERVICE_LINES_AVAILABLE"
    if monetary_value:
        return "INK_PRESENT_READABLE"
    if disp == "INK_PRESENT_UNREADABLE":
        blur = float(evidence.get("blur_score") or 0)
        return "INK_PRESENT_DEGRADED" if blur < 60 else "INK_PRESENT_READABLE"
    if any((a.get("observation") or {}).get("text") for a in ocr_attempts):
        return "LABEL_PRESENT_VALUE_OUTSIDE_ROI"
    if float(evidence.get("blank_probability") or 0) >= 0.85:
        return "BLANK_CONFIRMED"
    return "OTHER_EXPLAINED"


def _contact_sheet(
    page: Image.Image,
    total: Image.Image,
    expanded: Image.Image,
    table: Image.Image,
    *,
    predicted: str,
    classification: str,
    out_path: Path,
) -> None:
    def fit(im: Image.Image, box: tuple[int, int]) -> Image.Image:
        im = im.convert("RGB")
        im.thumbnail(box)
        canvas = Image.new("RGB", box, (245, 245, 245))
        canvas.paste(im, ((box[0] - im.width) // 2, (box[1] - im.height) // 2))
        return canvas

    sheet = Image.new("RGB", (1100, 900), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 10), f"class={classification} predicted={predicted or 'NONE'}", fill=(0, 0, 0))
    sheet.paste(fit(page, (500, 650)), (20, 40))
    sheet.paste(fit(total, (250, 120)), (540, 40))
    sheet.paste(fit(expanded, (250, 120)), (540, 180))
    sheet.paste(fit(table, (520, 400)), (540, 320))
    draw.text((540, 740), "page | total ROI | expanded | table", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "evaluation_results/hackathon_50_blind_cascade_v12_3x/results.jsonl",
    )
    parser.add_argument(
        "--claims-root",
        type=Path,
        default=ROOT / "evaluation_results/hackathon_50_blind_cascade_v12_3x/claims",
    )
    parser.add_argument("--zip", type=Path, default=ROOT / "data/Hackathon - 1000 Claims.zip")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "evaluation_results/empty_financial_ink_inventory_v1",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from packages.image_evidence import analyze_roi
    from packages.ocr_portfolio import recognize_monetary_crop

    rows = _load_rows(args.results)
    empty_rows = []
    for r in rows:
        gaps = [
            (g.get("gap_class") if isinstance(g, dict) else g)
            for g in (r.get("gap_classes") or [])
        ]
        if "EMPTY_FINANCIAL_INK" in gaps:
            empty_rows.append(r)
    if args.limit:
        empty_rows = empty_rows[: args.limit]

    inventory = []
    taxonomy = Counter()
    recoverable = Counter()

    def _tess(img: Image.Image):
        try:
            import pytesseract

            raw = pytesseract.image_to_string(
                img, config="--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789,.$"
            ).strip()
            return raw, 0.6
        except Exception:  # noqa: BLE001 -- optional diagnostic OCR backend
            return "", 0.0

    for idx, row in enumerate(empty_rows):
        document = row.get("document") or row.get("claim_id")
        claim_dir = _claim_dir(args.claims_root, str(document))
        rec = {
            "claim_id": row.get("claim_id"),
            "document": document,
            "document_type": "CMS1500",
            "service_line_charges_reported": row.get("service_line_charges"),
            "primary_category": "OTHER_EXPLAINED",
        }
        try:
            found = _find_geometry(claim_dir)
            if not found:
                rec["error"] = "NO_GEOMETRY"
                inventory.append(rec)
                taxonomy["OTHER_EXPLAINED"] += 1
                recoverable["unresolved"] += 1
                continue
            geom, telem = found
            page = _warp_page(args.zip, telem, geom)
            bbox = _total_bbox(geom)
            if bbox is None:
                # Fallback from OCRCandidates region.
                ocr_p = claim_dir / "ocr" / "OCRCandidates.json"
                ocr = json.loads(ocr_p.read_text()) if ocr_p.exists() else {}
                for f in ocr.get("fields") or []:
                    if str(f.get("field") or "").casefold() in {"total_charge", "total_charges"}:
                        region = f.get("ocr_region") or f.get("canonical_region")
                        if region and len(region) == 4:
                            bbox = tuple(int(v) for v in region)
                        break
            if bbox is None:
                rec["error"] = "NO_TOTAL_ROI"
                inventory.append(rec)
                taxonomy["ROI_MISALIGNED"] += 1
                recoverable["unresolved"] += 1
                continue
            expanded = _expand(bbox, page)
            table_box = _table_crop_box(page)
            total_crop = page.crop(bbox)
            exp_crop = page.crop(expanded)
            table_crop = page.crop(table_box)
            evidence = analyze_roi(total_crop, ocr_empty=True).to_dict()
            # Lightweight monetary probe (tess variants only for inventory speed).
            monet = recognize_monetary_crop(
                total_crop,
                engines={"tesseract_digits": _tess},
                max_variants=5,
            )
            monet_exp = recognize_monetary_crop(
                exp_crop,
                engines={"tesseract_digits": _tess},
                max_variants=4,
            )
            predicted = (monet.best.value if monet.best else None) or (
                monet_exp.best.value if monet_exp.best else None
            )
            # Prior OCR attempts from saved artifacts.
            ocr = json.loads((claim_dir / "ocr" / "OCRCandidates.json").read_text())
            prior_attempts = []
            for f in ocr.get("fields") or []:
                if str(f.get("field") or "").casefold() in {"total_charge", "total_charges"}:
                    prior_attempts = list(f.get("attempts") or [])
            svc = [ln for ln in (ocr.get("service_lines") or []) if ln.get("charges")]
            category = _classify(
                evidence=evidence,
                monetary_value=predicted,
                service_line_count=len(svc) or int(row.get("service_line_charges") or 0),
                ocr_attempts=prior_attempts + monet.attempts,
            )
            taxonomy[category] += 1
            # Recoverability flags (diagnostic only — not acceptance).
            direct = bool(predicted)
            line_sum = len(svc) >= 1 or int(row.get("service_line_charges") or 0) >= 1
            irrecoverable = category in {"BLANK_CONFIRMED", "PIXELS_MISSING"} and not line_sum
            if direct:
                recoverable["direct_total"] += 1
            elif line_sum:
                recoverable["line_sum"] += 1
            elif irrecoverable:
                recoverable["irrecoverable"] += 1
            else:
                recoverable["unresolved"] += 1

            sheet_path = args.out_dir / "contact_sheets" / f"{idx:02d}_{Path(str(document)).name}.jpg"
            _contact_sheet(
                page,
                total_crop,
                exp_crop,
                table_crop,
                predicted=predicted or "",
                classification=category,
                out_path=sheet_path,
            )
            rec.update(
                {
                    "page_number": geom.get("page_number"),
                    "original_page_path": telem.get("source", {}).get("entry"),
                    "total_charge_roi": list(bbox),
                    "expanded_roi": list(expanded),
                    "financial_table_bbox": list(table_box),
                    "expected_semantic_field": "total_charge",
                    "ink_density": evidence.get("ink_density"),
                    "connected_components": evidence.get("connected_components"),
                    "clipping_score": evidence.get("clipping_score"),
                    "blur": evidence.get("blur_score"),
                    "blank_probability": evidence.get("blank_probability"),
                    "one_bit_loss": evidence.get("one_bit_ink_loss_risk"),
                    "image_evidence_disposition": evidence.get("disposition"),
                    "ocr_variant_attempts": monet.to_dict(),
                    "expanded_variant_best": None
                    if monet_exp.best is None
                    else monet_exp.best.value,
                    "predicted_value": predicted,
                    "service_line_candidates": [
                        {"charges": ln.get("charges"), "bbox": ln.get("canonical_region")}
                        for ln in svc
                    ],
                    "primary_category": category,
                    "recoverability": {
                        "direct_total": direct,
                        "line_sum": line_sum,
                        "cross_page": False,
                        "irrecoverable": irrecoverable,
                    },
                    "contact_sheet": str(sheet_path.relative_to(args.out_dir)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            taxonomy["OTHER_EXPLAINED"] += 1
            recoverable["unresolved"] += 1
        inventory.append(rec)
        print(
            f"[{idx+1}/{len(empty_rows)}] {document} -> {rec.get('primary_category')} "
            f"pred={rec.get('predicted_value')}",
            flush=True,
        )

    currently_stp = sum(1 for r in rows if r.get("true_stp"))
    total_claims = len(rows)
    safely = recoverable["direct_total"] + recoverable["line_sum"]
    # Ceiling uses currently STP + diagnostic recoverable EMPTY blockers only.
    ceiling = (currently_stp + safely) / max(1, total_claims)
    summary = {
        "source_results": str(args.results),
        "empty_financial_ink_count": len(empty_rows),
        "taxonomy": dict(taxonomy),
        "recoverability_counts": dict(recoverable),
        "currently_stp_claims": currently_stp,
        "total_claims": total_claims,
        "safely_recoverable_from_empty_blockers": safely,
        "recoverable_stp_ceiling": ceiling,
        "ceiling_below_94pct": ceiling < 0.94,
        "note": (
            "Recoverability is diagnostic from existing pixels via tess monetary "
            "variants + reported service lines. Not an acceptance decision. "
            "No invented ground truth."
        ),
    }
    (args.out_dir / "inventory.jsonl").write_text(
        "\n".join(json.dumps(r) for r in inventory) + "\n", encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
