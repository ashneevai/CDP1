#!/usr/bin/env python3
"""Side-by-side probe: stable PP-OCRv4 vs isolated PP-OCRv5 Server on charge crops.

Uses the system Python for v4 and ``.venv-ppocrv5`` for v5 server so the
production paddleocr 2.x stack stays untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ZIP = Path(os.environ.get("CDP_HACKATHON_ZIP") or ROOT / "data" / "Hackathon - 1000 Claims.zip")
VENV_PY = ROOT / ".venv-ppocrv5" / "bin" / "python"
SMOKE = ROOT / "evaluation_results" / "hackathon_charge_line_smoke_v12_3x"


_V5_WORKER = r"""
import json, os, sys
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import paddle
paddle.set_flags({"FLAGS_use_mkldnn": False})
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

path = sys.argv[1]
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="PP-OCRv5_server_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
    enable_mkldnn=False,
)
img = Image.open(path).convert("RGB")
out = list(ocr.predict(np.asarray(img)))
texts, scores = [], []
for result in out:
    payload = result.json if hasattr(result, "json") else result
    if callable(payload):
        payload = payload()
    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]
    texts = list(payload.get("rec_texts") or [])
    scores = list(payload.get("rec_scores") or [])
print(json.dumps({"texts": texts, "scores": scores}))
"""


def _v4_read(crop: Image.Image) -> dict:
    import numpy as np
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        use_angle_cls=False,
        lang="en",
        show_log=False,
        ocr_version="PP-OCRv4",
        enable_mkldnn=False,
        cpu_threads=2,
    )
    raw = engine.ocr(np.asarray(crop.convert("RGB")), cls=False)
    texts, scores = [], []
    for block in raw or []:
        for row in block or []:
            if not row or len(row) < 2:
                continue
            texts.append(str(row[1][0]))
            scores.append(float(row[1][1]))
    return {"texts": texts, "scores": scores}


def _v5_server_read(crop: Image.Image) -> dict:
    if not VENV_PY.is_file():
        return {"error": "missing .venv-ppocrv5"}
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
        crop.save(path)
    try:
        proc = subprocess.run(
            [str(VENV_PY), "-c", _V5_WORKER, path],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        return {"error": proc.stderr[-500:] or proc.stdout[-500:] or f"rc={proc.returncode}"}
    line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(line)


def _page_from_zip(document: str) -> Image.Image:
    with zipfile.ZipFile(ZIP) as zf:
        return Image.open(BytesIO(zf.read(document))).convert("RGB")


def _crop(page: Image.Image, bbox: list[int]) -> Image.Image:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    pad = 4
    return page.crop(
        (
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(page.width, x1 + pad),
            min(page.height, y1 + pad),
        )
    )


def main() -> int:
    rows = []
    for claim_dir in sorted((SMOKE / "claims").glob("Group A__*")):
        ocr_path = claim_dir / "ocr" / "OCRCandidates.json"
        if not ocr_path.exists():
            continue
        ocr = json.loads(ocr_path.read_text())
        document = claim_dir.name.replace("__", "/", 1)
        # Prefer result.json document key when present.
        result_path = claim_dir / "result.json"
        if result_path.exists():
            document = json.loads(result_path.read_text()).get("document") or document
        lines = ocr.get("service_lines") or []
        # Also probe box-28 total_charge ROI when present.
        fields = ocr.get("fields") or []
        targets = []
        for i, line in enumerate(lines):
            bbox = line.get("canonical_region") or line.get("ocr_region")
            if bbox:
                targets.append(("line", i, bbox, line.get("charges"), line.get("candidates")))
        for field in fields:
            if (field.get("field") or "").casefold() not in {"total_charge", "total_charges"}:
                continue
            bbox = field.get("ocr_region") or field.get("canonical_region")
            if bbox:
                targets.append(("box28", 0, bbox, field.get("value"), field.get("candidates")))
        if not targets:
            continue
        page = _page_from_zip(document)
        for kind, idx, bbox, prior, cands in targets:
            crop = _crop(page, bbox)
            v4 = _v4_read(crop)
            v5 = _v5_server_read(crop)
            rows.append(
                {
                    "document": document,
                    "kind": kind,
                    "index": idx,
                    "bbox": bbox,
                    "prior_selected": prior,
                    "prior_engines": [
                        (c.get("engine"), c.get("value")) for c in (cands or [])[:6]
                    ],
                    "ppocr_v4": v4,
                    "ppocr_v5_server": v5,
                }
            )
            print(
                f"{document} {kind}{idx} prior={prior!r} "
                f"v4={v4.get('texts')} v5={v5.get('texts') or v5.get('error')}",
                flush=True,
            )
        page.close()

    out = ROOT / "evaluation_results" / "ppocr_v5_server_charge_probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out / 'results.json'} n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
