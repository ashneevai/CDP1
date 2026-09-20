#!/usr/bin/env python3
"""Run field-cascade ops on the Hackathon Claims corpus (cascade strategy from YAML).

Operational completion / true-STP / HITL evaluation (no field-level GT).
Pipeline per claim: app.py (register+geometry+recovery ladder) → ocr → rank →
validate → assemble → complete (E3 from registration_report).

Resume-safe: JSONL ledger skips finished claim_ids.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packages.extraction_recovery.gap_taxonomy import classify_field_gap

DEFAULT_ZIP = ROOT / "data" / "Hackathon - 1000 Claims.zip"
DEFAULT_DATASET = ROOT / "dataset.yaml"
DEFAULT_OUT = ROOT / "evaluation_results" / "hackathon_1000_cascade_v9"

CRITICAL = (
    "patient_dob",
    "total_charge",
    "patient_name",
    "insured_id_number",
    "insured_name",
)
AUTO = {"AUTO_ACCEPTED", "REFERENCE_CONFIRMED"}
CASCADE_ENGINES = ("paddleocr", "rapidocr", "tesseract", "tesseract_digits")


def _ocr_pool_enabled() -> bool:
    raw = (os.environ.get("CDP_OCR_WORKER_POOL") or "1").strip().casefold()
    return raw not in {"0", "false", "no", "off"}


def _app_pool_enabled() -> bool:
    raw = (os.environ.get("CDP_APP_WORKER_POOL") or "1").strip().casefold()
    return raw not in {"0", "false", "no", "off"}


def _ocr_pool_job(geometry_dir: str, output_dir: str) -> tuple[int, str]:
    """Long-lived pool worker entry — keeps Paddle/Rapid warm across claims."""
    try:
        from scripts.ocr_from_geometry import run as ocr_run

        result = ocr_run(geometry_dir, output_dir)
        status = result.get("status")
        payload = json.dumps(
            {
                "status": status,
                "fields": len(result.get("fields") or []),
                "candidates": sum(
                    len(row.get("candidates") or [])
                    for row in (result.get("fields") or [])
                ),
            }
        )
        return (0 if status == "COMPLETED" else 1, payload)
    except Exception as exc:  # noqa: BLE001
        return (1, f"{type(exc).__name__}: {exc}")


def _app_pool_job(
    dataset_yaml: str,
    document: str,
    document_type: str,
    output_root: str,
) -> tuple[int, str]:
    """Long-lived registration worker — amortize imports + template SIFT cache."""
    import logging

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(message)s",
        force=True,
    )
    logging.getLogger("cdp.registration.telemetry").setLevel(logging.ERROR)
    try:
        from app import process_one

        _path, state = process_one(
            dataset_yaml,
            document=document,
            output_root=output_root,
            document_type=document_type or None,
        )
        status = state.get("status")
        return (0 if status == "SUCCESS" else 1, json.dumps({"status": status}))
    except Exception as exc:  # noqa: BLE001
        return (1, f"{type(exc).__name__}: {exc}")


def _shutdown_process_pool(pool: ProcessPoolExecutor | None) -> None:
    """Terminate pool workers — Paddle/ORT atexit finalizers can hang on join."""
    if pool is None:
        return
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        pool.shutdown(wait=False)
    processes = getattr(pool, "_processes", None) or {}
    for proc in list(processes.values()):
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:  # noqa: BLE001, S110
            pass


def _spawn_pool(n_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max(1, n_workers),
        mp_context=__import__("multiprocessing").get_context("spawn"),
        initializer=_pool_worker_silence_stdio,
    )


def _pool_worker_silence_stdio() -> None:
    """Keep ProcessPool workers from flooding the parent stdout pipe.

    TrOCR/transformers warnings on inherited stdout filled the tee pipe and
    blocked the cascade main thread (~27 claims finished but never ledgered).
    """
    import os
    import sys

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            # Dup over stdio fds so the redirect outlives this with-block.
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
    except OSError:
        pass
    # Spawn workers do not inherit the parent process's in-memory factories.
    # Without this, Azure DI charge residual raises AZURE_DI_FACTORY_UNCONFIGURED.
    from workers.ocr_engine_factories import wire_package_ocr_providers

    wire_package_ocr_providers()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _bundle_id(document: str) -> str:
    """Group + claim family (e.g. Group A/M048DJJM) — multipage claim bundle."""
    text = (document or "").replace("\\", "/")
    if "/" in text:
        group, name = text.split("/", 1)
    else:
        group, name = "", text
    family = name.split(".")[0] if name else ""
    return f"{group}/{family}" if group else family


def _group_id(document: str) -> str:
    text = (document or "").replace("\\", "/")
    return text.split("/", 1)[0] if "/" in text else "UNKNOWN"


def _probe_ocr_engines() -> dict[str, Any]:
    """One-shot live probe: paddle / rapid / tesseract must OBSERVE, not UNAVAILABLE."""
    from PIL import Image, ImageDraw, ImageFont

    from packages.ocr_router import OCRRouter, OCRRouteRequest
    from workers.ocr_engine_factories import wire_package_ocr_providers

    wire_package_ocr_providers()
    image = Image.new("L", (220, 64), 255)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001 -- optional font fallback
        font = None
    draw.text((12, 18), "HELLO 123", fill=0, font=font)
    router = OCRRouter(lambda _attempt: True)
    result = router.route(
        OCRRouteRequest(
            image,
            (0, 0, image.width, image.height),
            engine_order=("paddleocr", "rapidocr", "tesseract"),
        )
    )
    # Force tesseract even when confirmation already satisfied.
    tess_only = router.route(
        OCRRouteRequest(
            image,
            (0, 0, image.width, image.height),
            engine_order=("tesseract",),
        )
    )
    observed = {
        attempt.engine: attempt.reason for attempt in result.attempts
    }
    observed["tesseract"] = (
        tess_only.selected.reason
        if tess_only.selected is not None
        else next((a.reason for a in tess_only.attempts if a.engine == "tesseract"), "MISSING")
    )
    return {
        "paddleocr": observed.get("paddleocr", "MISSING"),
        "rapidocr": observed.get("rapidocr", "MISSING"),
        "tesseract": observed.get("tesseract", "MISSING"),
        "all_observed": all(
            observed.get(name) == "OBSERVED"
            for name in ("paddleocr", "rapidocr", "tesseract")
        ),
        "tesseract_text": (
            " ".join(line.text for line in tess_only.selected.observation.lines)
            if tess_only.selected and tess_only.selected.observation
            else ""
        ),
    }


def _ocr_engine_stats(claim_out: Path) -> dict[str, Any]:
    """Count cascade OCR attempt outcomes from OCRCandidates.json."""
    path = claim_out / "ocr" / "OCRCandidates.json"
    attempts: Counter[str] = Counter()
    observed: Counter[str] = Counter()
    unavailable: Counter[str] = Counter()
    if not path.exists():
        return {
            "engines_attempted": {},
            "engines_observed": {},
            "engines_unavailable": {},
            "cascade_healthy": False,
            "tesseract_observed": 0,
            "tesseract_unavailable": 0,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "engines_attempted": {},
            "engines_observed": {},
            "engines_unavailable": {},
            "cascade_healthy": False,
            "tesseract_observed": 0,
            "tesseract_unavailable": 0,
        }
    for field in payload.get("fields") or []:
        for attempt in field.get("attempts") or []:
            engine = str(attempt.get("engine") or "")
            if engine not in CASCADE_ENGINES:
                continue
            attempts[engine] += 1
            reason = str(attempt.get("reason") or "")
            if reason == "OBSERVED":
                observed[engine] += 1
            elif reason == "UNAVAILABLE":
                unavailable[engine] += 1
    # Healthy when primary paddle + confirmation rapid observed, and tesseract
    # never reported UNAVAILABLE (fill may be unused when first two succeed).
    healthy = (
        observed.get("paddleocr", 0) > 0
        and observed.get("rapidocr", 0) > 0
        and unavailable.get("tesseract", 0) == 0
    )
    return {
        "engines_attempted": dict(attempts),
        "engines_observed": dict(observed),
        "engines_unavailable": dict(unavailable),
        "cascade_healthy": healthy,
        "tesseract_observed": observed.get("tesseract", 0)
        + observed.get("tesseract_digits", 0),
        "tesseract_unavailable": unavailable.get("tesseract", 0),
    }

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _list_documents(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        docs = [name for name in zf.namelist() if not name.endswith("/")]
    return sorted(docs)


def _load_done(ledger: Path) -> set[str]:
    done: set[str] = set()
    if not ledger.exists():
        return done
    with ledger.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            claim_id = str(row.get("claim_id") or "").strip()
            if claim_id and row.get("finished"):
                done.add(claim_id)
    return done


def _append_ledger(ledger: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock, ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _claim_slug(document: str) -> str:
    return document.replace("/", "__")


def _prune_document_json(app_out: Path) -> None:
    """Drop oversized document.json; keep registration_trace for OCR."""
    for path in app_out.glob("**/document.json"):
        try:
            path.unlink()
        except OSError:
            pass


def _prune_trace(app_out: Path) -> None:
    for path in app_out.glob("**/registration_trace.json"):
        try:
            path.unlink()
        except OSError:
            pass


def _cms1500_template_version() -> str:
    """Pin finish/validate to the same CMS-1500 template the release registered."""
    try:
        import yaml

        from packages.release_selection import active_release_from_env, select_release

        manifest_path = select_release(active_release_from_env())
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        versions = manifest.get("template_versions") or {}
        version = str(versions.get("cms1500") or "").strip()
        if version:
            return version
    except Exception:  # noqa: BLE001, S110 -- fall back to explicit release name
        pass
    release = (os.environ.get("CDP_PIPELINE_RELEASE") or "").strip().casefold()
    if release in {"extraction-v3", "v3"}:
        return "03"
    return "02-12"


def _stage_env() -> dict[str, str]:
    """Limit nested BLAS/OpenMP threads so parallel claim workers do not thrash."""
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "FLAGS_num_threads",
    ):
        env.setdefault(key, "1")
    # Default STP eval to critical-field OCR only (~5× fewer ROIs). Override
    # with CDP_OCR_FIELD_SCOPE=all for full-form extraction.
    env.setdefault("CDP_OCR_FIELD_SCOPE", "stp_critical")
    # SELECTIVE_E2_ONLY: stop after field-shaped primary (skip dual-engine tax).
    env.setdefault("CDP_OCR_SELECTIVE_CONFIRM", "1")
    # Paddle warm inference is ~50× faster than Rapid on CPU; use as STP-eval
    # primary while Rapid remains confirmation when primary is unshaped.
    env.setdefault("CDP_OCR_PRIMARY_OVERRIDE", "paddleocr")
    # Prefer inference-scoped lock: unzip/warp/JSON overlap across workers while
    # Paddle/Rapid critical sections still serialize. Process scope was a
    # conservative fallback that inflated multi-worker claim wall.
    env.setdefault("CDP_OCR_LOCK", "1")
    env.setdefault("CDP_OCR_LOCK_SCOPE", "inference")
    # Skip multi-MB keypoint/match dumps + full-image SHA256 on registration.
    env.setdefault("CDP_REGISTRATION_VERBOSE_TELEMETRY", "0")
    # Long-lived workers amortize cold start (override with =0 for subprocess-per-claim).
    env.setdefault("CDP_OCR_WORKER_POOL", "1")
    env.setdefault("CDP_APP_WORKER_POOL", "1")
    # Latency bar: claim mean ≤30s. TrOCR DOB residual ON with process singleton
    # + skip-if-local-shaped (only handwriting/ambiguous gaps fire). Prefer
    # gpt-4o crop residuals over Azure DI (F0 is 1 analyze/min).
    env.setdefault("CDP_TROCR_DOB_RESIDUAL", "1")
    env.setdefault("CDP_AZURE_DI_DOB_RESIDUAL", "0")
    env.setdefault("CDP_AZURE_DI_CHARGE_RESIDUAL", "0")
    env.setdefault("CDP_AZURE_DI_CHARGE_CORROBORATE", "0")
    env.setdefault("CDP_AZURE_DI_CHARGE_ACCEPT", "0")
    env.setdefault("CDP_GPT4O_CROP_RESIDUAL", "1")
    env.setdefault("CDP_GPT4O_CROP_ACCEPT", "1")
    env.setdefault("CDP_DOB_RESIDUAL_SKIP_IF_LOCAL_SHAPED", "1")
    # SuperPoint+LightGlue for catastrophic REG — process-lifetime singleton
    # amortizes cold load; trail-aware near-miss recovers most STP regressions
    # without torch. Keep ON for STP; opt-out with =0 for pure latency smoke.
    env.setdefault("CDP_LEARNED_MATCHER", "1")
    # Name Rapid confirm gate (was 0.88 — nearly always confirmed).
    env.setdefault("CDP_OCR_NAME_CONFIRM_MIN_CONF", "0.80")
    env.setdefault("CDP_AZURE_DI_PAGE_CORNERS", "0")
    env.setdefault("CDP_PIPELINE_RELEASE", "extraction-v3")
    # Keep slot limiter available if DI is re-enabled; defaults do not call it.
    env.setdefault("CDP_AZURE_DI_MIN_INTERVAL_SECONDS", "60")
    env.setdefault("CDP_AZURE_DI_SLOT_PATH", "/tmp/cdp-azure-di-slot")
    env.setdefault("CDP_AZURE_DI_SERVICE_LINE_BUDGET", "1")
    env.setdefault(
        "CDP_AZURE_DI_METER_PATH",
        str(ROOT / "evaluation_results" / "azure_di_meter.jsonl"),
    )
    return env


@contextmanager
def _ocr_process_lock():
    """Whole-process OCR lock only when CDP_OCR_LOCK_SCOPE=process.

    Default cascade scope is ``inference`` — Paddle/Rapid critical sections take
    the flock inside ``packages.ocr_runtime_lock``, so unzip/warp/JSON overlap.
    """
    from packages.ocr_runtime_lock import ocr_process_lock

    with ocr_process_lock():
        yield


def _run_stage(cmd: list[str], log_path: Path) -> tuple[int, str]:
    """Stream stdout/stderr to a file to avoid pipe deadlocks with verbose app.py."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=_stage_env(),
            check=False,
        )
    tail = ""
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
    except OSError:
        pass
    return proc.returncode, tail


def _summarize_final(claim_out: Path) -> dict[str, Any]:
    final = json.loads((claim_out / "final" / "FinalClaim.json").read_text(encoding="utf-8"))
    decision = final.get("decision") or final
    fields: dict[str, Any] = {}
    for fd in decision.get("field_decisions") or []:
        name = fd.get("field_name")
        if name in CRITICAL:
            fields[name] = {
                "disp": fd.get("disposition"),
                "value": fd.get("selected_value"),
                "reasons": list(fd.get("reason_codes") or [])[:12],
            }
    ocr_path = claim_out / "ocr" / "OCRCandidates.json"
    service_line_charges = 0
    document_family = None
    allows_cms_geometry = None
    document_finance = None
    if ocr_path.exists():
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
        service_line_charges = sum(
            1
            for line in (ocr.get("service_lines") or [])
            if line.get("status") == "OBSERVED" and line.get("charges")
        )
        document_family = ocr.get("document_family")
        allows_cms_geometry = ocr.get("allows_cms_geometry")
        document_finance = ocr.get("document_finance")
        if document_family is None:
            pkg = ocr.get("package_intelligence") or {}
            pages = pkg.get("pages") or []
            if pages:
                document_family = pages[0].get("page_class")
                allows_cms_geometry = pages[0].get("allows_cms_geometry")

    blockers = list(decision.get("critical_blockers") or [])
    gaps = []
    for blocker in blockers:
        field_info = fields.get(blocker) or {}
        observed_text = str(field_info.get("value") or "")
        gap = classify_field_gap(
            blocker,
            observed_text=observed_text,
            accepted=False,
            service_line_charges=service_line_charges,
            reason_codes=field_info.get("reasons") or [],
        )
        if gap is not None:
            gaps.append(
                {
                    "field": gap.field_name,
                    "gap_class": gap.gap_class,
                    "action": gap.action,
                    "evidence": gap.evidence,
                }
            )
    completed = final.get("status") == "COMPLETED"
    review_required = bool(final.get("review_required") or decision.get("review_required"))
    engine_stats = _ocr_engine_stats(claim_out)
    return {
        "completed": completed,
        "review_required": review_required,
        "true_stp": completed and not review_required,
        "critical_blockers": blockers,
        "fields": fields,
        "gap_classes": gaps,
        "service_line_charges": service_line_charges,
        "claim_status": decision.get("claim_status") or final.get("claim_status"),
        "ocr_engine_stats": engine_stats,
        "document_family": document_family,
        "allows_cms_geometry": allows_cms_geometry,
        "document_finance": document_finance,
    }


def _process_one(
    *,
    document: str,
    out_dir: Path,
    dataset_yaml: Path,
    document_type: str,
    keep_heavy: bool,
    ocr_executor: Any = None,
    app_executor: Any = None,
) -> dict[str, Any]:
    claim_id = _claim_slug(document)
    claim_out = out_dir / "claims" / claim_id
    if claim_out.exists():
        shutil.rmtree(claim_out)
    claim_out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    logs = claim_out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    app_out = claim_out / "application"

    if app_executor is not None:
        try:
            rc, tail = app_executor.submit(
                _app_pool_job,
                str(dataset_yaml),
                document,
                document_type,
                str(app_out),
            ).result()
        except Exception as exc:  # noqa: BLE001
            rc, tail = 1, f"{type(exc).__name__}: {exc}"
        try:
            (logs / "app.log").write_text(tail or "", encoding="utf-8")
        except OSError:
            pass
    else:
        rc, tail = _run_stage(
            [
                sys.executable,
                str(ROOT / "app.py"),
                "--dataset",
                str(dataset_yaml),
                "--document",
                document,
                "--document-type",
                document_type,
                "--output-root",
                str(app_out),
            ],
            logs / "app.log",
        )

    geometry_hits = list(app_out.glob("**/GeometryResult.json"))
    if not geometry_hits:
        reason = "geometry_missing"
        for report in app_out.glob("**/registration_report.json"):
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
                reason = (
                    payload.get("reason")
                    or (payload.get("registration") or {}).get("reason")
                    or reason
                )
            except (OSError, json.JSONDecodeError):
                pass
            break
        for doc_json in app_out.glob("**/document.json"):
            try:
                payload = json.loads(doc_json.read_text(encoding="utf-8"))
                reg = payload.get("registration") or {}
                reason = reg.get("reason") or reason
            except (OSError, json.JSONDecodeError, ValueError):
                pass
            break
        if not keep_heavy:
            _prune_document_json(app_out)
            _prune_trace(app_out)
        disposition = "REGISTRATION_FAILED"
        if rc != 0 and reason == "geometry_missing":
            disposition = "APP_FAILURE"
        # Hard blind: some REG pages are freeform (no CMS grid). Template
        # geometry cannot recover — fall back to Azure DI page text (+ optional
        # gpt-4o text agent) for critical fields instead of a geometry VLM.
        unstructured_meta: dict[str, Any] | None = None
        if disposition == "REGISTRATION_FAILED":
            try:
                from packages.extraction_recovery.unstructured_reg_fallback import (
                    run_unstructured_reg_fallback,
                    unstructured_reg_fallback_enabled,
                )
            except ImportError:
                run_unstructured_reg_fallback = None  # type: ignore
                unstructured_reg_fallback_enabled = lambda: False  # type: ignore
            if unstructured_reg_fallback_enabled() and run_unstructured_reg_fallback is not None:
                page_image = None
                try:
                    # Prefer zip bytes via existing dataset helper if present.
                    from io import BytesIO
                    from zipfile import ZipFile

                    from PIL import Image as _PILImage

                    zip_path = Path(os.environ.get("CDP_HACKATHON_ZIP") or ROOT / "data" / "Hackathon - 1000 Claims.zip")
                    if zip_path.exists():
                        with ZipFile(zip_path) as zf:
                            page_image = _PILImage.open(BytesIO(zf.read(document))).convert("RGB")
                except (OSError, ValueError, KeyError, RuntimeError):
                    page_image = None
                if page_image is not None:
                    fb = run_unstructured_reg_fallback(page_image)
                    unstructured_meta = {
                        "attempted": fb.attempted,
                        "reason": fb.reason,
                        "agent_used": fb.agent_used,
                        "fields": dict(fb.fields),
                    }
                    # STP only when all critical blockers we care about shaped.
                    required = {"patient_name", "patient_dob", "insured_id_number", "total_charge"}
                    if required.issubset(fb.fields):
                        disposition = "TRUE_STP"
                    elif fb.fields:
                        disposition = "HITL"
                    with contextlib.suppress(OSError, AttributeError, ValueError):
                        page_image.close()
        row = {
            "finished": True,
            "claim_id": claim_id,
            "document": document,
            "bundle_id": _bundle_id(document),
            "group_id": _group_id(document),
            "registration_ok": False,
            "completed": disposition in {"TRUE_STP", "HITL"},
            "true_stp": disposition == "TRUE_STP",
            "disposition": disposition,
            "registration_reason": reason,
            "hitl_track": "UNSTRUCTURED_DI" if unstructured_meta and disposition == "HITL" else None,
            "critical_blockers": (
                [
                    f
                    for f in ("patient_name", "patient_dob", "insured_id_number", "total_charge")
                    if f not in (unstructured_meta or {}).get("fields", {})
                ]
                if disposition == "HITL" and unstructured_meta
                else None
            ),
            "unstructured_reg_fallback": unstructured_meta,
            "app_returncode": rc,
            "error": tail if disposition == "APP_FAILURE" else None,
            "elapsed_sec": round(time.time() - started, 3),
            "ts": _utc_now(),
            "strategy_id": "field-cascade-v12+unstructured-reg-fallback"
            if unstructured_meta and unstructured_meta.get("attempted")
            else "field-cascade-v12",
        }
        _write_json(claim_out / "result.json", row)
        return row

    geometry_dir = geometry_hits[0].parent
    if not keep_heavy:
        _prune_document_json(app_out)

    stages = [
        (
            "ocr",
            [
                sys.executable,
                "-m",
                "scripts.ocr_from_geometry",
                str(geometry_dir),
                str(claim_out / "ocr"),
            ],
        ),
        (
            "finish",
            [
                sys.executable,
                "-m",
                "scripts.finish_from_ocr",
                str(claim_out / "ocr" / "OCRCandidates.json"),
                str(claim_out),
                "--template-id",
                "cms1500",
                "--template-version",
                _cms1500_template_version(),
                "--document-family",
                "CMS1500",
            ],
        ),
    ]
    for stage_name, cmd in stages:
        if stage_name == "ocr" and ocr_executor is not None:
            logs.mkdir(parents=True, exist_ok=True)
            log_path = logs / "ocr.log"
            try:
                with _ocr_process_lock():
                    rc, tail = ocr_executor.submit(
                        _ocr_pool_job,
                        str(geometry_dir),
                        str(claim_out / "ocr"),
                    ).result()
            except Exception as exc:  # noqa: BLE001
                rc, tail = 1, f"{type(exc).__name__}: {exc}"
            try:
                log_path.write_text(tail or "", encoding="utf-8")
            except OSError:
                pass
        elif stage_name == "ocr":
            with _ocr_process_lock():
                rc, tail = _run_stage(cmd, logs / f"{stage_name}.log")
        else:
            rc, tail = _run_stage(cmd, logs / f"{stage_name}.log")
        if stage_name == "ocr" and not keep_heavy:
            # Trace is only required for OCR; free disk after that stage.
            _prune_trace(app_out)
        if rc != 0:
            row = {
                "finished": True,
                "claim_id": claim_id,
                "document": document,
                "bundle_id": _bundle_id(document),
                "group_id": _group_id(document),
                "registration_ok": True,
                "completed": False,
                "true_stp": False,
                "disposition": "STAGE_FAILURE",
                "failed_stage": stage_name,
                "error": tail,
                "ocr_engine_stats": _ocr_engine_stats(claim_out),
                "elapsed_sec": round(time.time() - started, 3),
                "ts": _utc_now(),
            }
            _write_json(claim_out / "result.json", row)
            return row

    try:
        summary = _summarize_final(claim_out)
    except Exception as exc:  # noqa: BLE001
        row = {
            "finished": True,
            "claim_id": claim_id,
            "document": document,
            "bundle_id": _bundle_id(document),
            "group_id": _group_id(document),
            "registration_ok": True,
            "completed": False,
            "true_stp": False,
            "disposition": "SUMMARY_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.time() - started, 3),
            "ts": _utc_now(),
        }
        _write_json(claim_out / "result.json", row)
        return row

    row = {
        "finished": True,
        "claim_id": claim_id,
        "document": document,
        "bundle_id": _bundle_id(document),
        "group_id": _group_id(document),
        "registration_ok": True,
        "completed": summary["completed"],
        "true_stp": summary["true_stp"],
        "review_required": summary["review_required"],
        "disposition": "TRUE_STP" if summary["true_stp"] else "HITL",
        "hitl_track": None if summary["true_stp"] else "FIELD_INK",
        "critical_blockers": summary["critical_blockers"],
        "fields": summary["fields"],
        "gap_classes": summary["gap_classes"],
        "service_line_charges": summary["service_line_charges"],
        "claim_status": summary["claim_status"],
        "ocr_engine_stats": summary.get("ocr_engine_stats") or {},
        "document_family": summary.get("document_family"),
        "allows_cms_geometry": summary.get("allows_cms_geometry"),
        "document_finance": summary.get("document_finance"),
        "strategy_id": "field-cascade-v12",
        "elapsed_sec": round(time.time() - started, 3),
        "ts": _utc_now(),
    }
    _write_json(claim_out / "result.json", row)
    return row


def _rollup_scope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    reg_ok = sum(1 for r in rows if r.get("registration_ok"))
    completed = sum(1 for r in rows if r.get("completed"))
    true_stp = sum(1 for r in rows if r.get("true_stp"))
    hitl = sum(1 for r in rows if r.get("disposition") == "HITL")
    reg_hitl = sum(1 for r in rows if r.get("disposition") == "REGISTRATION_FAILED")
    field_hitl = sum(1 for r in rows if r.get("disposition") == "HITL" and r.get("completed"))
    return {
        "n": n,
        "registration_ok": reg_ok,
        "registration_ok_rate": round(reg_ok / n, 6) if n else 0.0,
        "completed": completed,
        "completion_rate": round(completed / n, 6) if n else 0.0,
        "true_stp": true_stp,
        "true_stp_rate_of_all": round(true_stp / n, 6) if n else 0.0,
        "true_stp_rate_of_completed": round(true_stp / completed, 6) if completed else 0.0,
        "hitl": hitl,
        "hitl_rate_of_all": round(hitl / n, 6) if n else 0.0,
        "registration_hitl": reg_hitl,
        "registration_hitl_rate": round(reg_hitl / n, 6) if n else 0.0,
        "field_ink_hitl": field_hitl,
        "field_ink_hitl_rate_of_completed": (
            round(field_hitl / completed, 6) if completed else 0.0
        ),
    }


def _summarize(rows: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    n = len(rows)
    by_disp = Counter(str(r.get("disposition") or "UNKNOWN") for r in rows)
    blockers: Counter[str] = Counter()
    gaps: Counter[str] = Counter()
    auto = {name: 0 for name in CRITICAL}
    reg_reasons: Counter[str] = Counter()
    engine_attempted: Counter[str] = Counter()
    engine_observed: Counter[str] = Counter()
    engine_unavailable: Counter[str] = Counter()
    cascade_healthy = 0
    cascade_claims = 0
    tesseract_claims = 0
    by_bundle: dict[str, list[dict[str, Any]]] = {}
    by_group: dict[str, list[dict[str, Any]]] = {}
    completed = sum(1 for r in rows if r.get("completed"))
    for row in rows:
        for b in row.get("critical_blockers") or []:
            blockers[str(b)] += 1
        for g in row.get("gap_classes") or []:
            gaps[str(g.get("gap_class"))] += 1
        if row.get("disposition") == "REGISTRATION_FAILED":
            reg_reasons[str(row.get("registration_reason") or "unknown")] += 1
        fields = row.get("fields") or {}
        for name in CRITICAL:
            disp = (fields.get(name) or {}).get("disp")
            if disp in AUTO:
                auto[name] += 1
        stats = row.get("ocr_engine_stats") or {}
        if stats:
            cascade_claims += 1
            if stats.get("cascade_healthy"):
                cascade_healthy += 1
            if int(stats.get("tesseract_observed") or 0) > 0:
                tesseract_claims += 1
            for eng, count in (stats.get("engines_attempted") or {}).items():
                engine_attempted[str(eng)] += int(count)
            for eng, count in (stats.get("engines_observed") or {}).items():
                engine_observed[str(eng)] += int(count)
            for eng, count in (stats.get("engines_unavailable") or {}).items():
                engine_unavailable[str(eng)] += int(count)
        doc = str(row.get("document") or "")
        bundle = str(row.get("bundle_id") or _bundle_id(doc))
        group = str(row.get("group_id") or _group_id(doc))
        by_bundle.setdefault(bundle, []).append(row)
        by_group.setdefault(group, []).append(row)

    overall = _rollup_scope(rows)
    return {
        "dataset": "DEVELOPMENT_DATASET_V1 / Hackathon - 1000 Claims.zip",
        "document_count_requested": limit,
        "document_count_evaluated": n,
        "strategy_id": "field-cascade-v12",
        **overall,
        "disposition_counts": dict(by_disp),
        "registration_failure_reasons": dict(reg_reasons),
        "critical_blockers": dict(blockers),
        "gap_classes": dict(gaps),
        "field_auto_of_completed": {
            name: f"{auto[name]}/{completed}" for name in CRITICAL
        },
        "field_auto_rate_of_completed": {
            name: round(auto[name] / completed, 6) if completed else 0.0
            for name in CRITICAL
        },
        "accuracy": {
            "status": "UNAVAILABLE_NO_GROUND_TRUTH",
            "end_to_end_correct_completion_rate": None,
            "field_accuracy": None,
            "note": (
                "Hackathon 1000 ZIP has no field-level labels. Accuracy cannot be "
                "scored; report operational STP/HITL and field auto-accept rates only."
            ),
        },
        "ocr_cascade": {
            "claims_with_ocr": cascade_claims,
            "cascade_healthy_claims": cascade_healthy,
            "cascade_healthy_rate": (
                round(cascade_healthy / cascade_claims, 6) if cascade_claims else 0.0
            ),
            "claims_with_tesseract_observed": tesseract_claims,
            "engines_attempted": dict(engine_attempted),
            "engines_observed": dict(engine_observed),
            "engines_unavailable": dict(engine_unavailable),
            "required": (
                "paddleocr OBSERVED + rapidocr OBSERVED; tesseract fill on miss "
                "(never UNAVAILABLE)"
            ),
        },
        "by_group": {
            group: _rollup_scope(group_rows)
            for group, group_rows in sorted(by_group.items())
        },
        "by_bundle": {
            bundle: _rollup_scope(bundle_rows)
            for bundle, bundle_rows in sorted(
                by_bundle.items(), key=lambda item: (-len(item[1]), item[0])
            )
        },
        "bundle_count": len(by_bundle),
        "note": (
            "Operational metrics under field-cascade-v12 with paddle+rapid confirmation "
            "cascade, name/ID value-band-first, and label-contamination relief. No "
            "field-level GT on Hackathon corpus — accuracy marked unavailable."
        ),
        "generated_at": _utc_now(),
    }


def main() -> int:
    from workers.ocr_engine_factories import wire_package_ocr_providers

    wire_package_ocr_providers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--documents",
        default="",
        help="Comma-separated document paths to run (e.g. 'Group A/M048DJJM.036'). "
        "When set, offset/limit are ignored.",
    )
    parser.add_argument("--document-type", default="CMS1500")
    parser.add_argument("--keep-heavy", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()

    # Stamp STP product defaults onto os.environ before pools spawn. Stale shell
    # exports from a prior latency smoke (e.g. CDP_TROCR_DOB_RESIDUAL=0) would
    # otherwise win over ``_stage_env`` setdefault and suppress DOB/REG recovery.
    # Set CDP_CASCADE_RESPECT_ENV=1 to keep caller exports as-is.
    _respect = (os.environ.get("CDP_CASCADE_RESPECT_ENV") or "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    _product = {
        "CDP_TROCR_DOB_RESIDUAL": "1",
        "CDP_LEARNED_MATCHER": "1",
        # Prefer gpt-4o crop residuals; Azure DI F0 is 1 analyze/min.
        "CDP_AZURE_DI_DOB_RESIDUAL": "0",
        "CDP_AZURE_DI_CHARGE_RESIDUAL": "0",
        "CDP_AZURE_DI_CHARGE_CORROBORATE": "0",
        "CDP_AZURE_DI_CHARGE_ACCEPT": "0",
        "CDP_AZURE_DI_PAGE_CORNERS": "0",
        "CDP_AZURE_DI_MIN_INTERVAL_SECONDS": "60",
        "CDP_AZURE_DI_SLOT_PATH": "/tmp/cdp-azure-di-slot",
        "CDP_AZURE_DI_SERVICE_LINE_BUDGET": "1",
        "CDP_DOB_RESIDUAL_SKIP_IF_LOCAL_SHAPED": "1",
        "CDP_OCR_NAME_CONFIRM_MIN_CONF": "0.80",
        # Unstructured REG page-read path is DI-backed — off while DI is parked.
        "CDP_UNSTRUCTURED_REG_FALLBACK": "0",
        "CDP_UNSTRUCTURED_REG_AGENT": "0",
        # FIELD_INK DOB/ID/charge: crop-only gpt-4o after local(+TrOCR) miss.
        "CDP_GPT4O_CROP_RESIDUAL": "1",
        "CDP_GPT4O_CROP_ACCEPT": "1",
        "CDP_GPT4O_EMPTY_FINANCE": "1",
        # OpenOCR / Monkey / PaddleOCR-VL failed for charge recovery — keep off.
        "CDP_OPENOCR_SVTR": "0",
        "CDP_MONKEYOCR": "0",
        "CDP_PADDLEOCR_VL_TABLE": "0",
        # v02-12 identity boxes sit on the insurance-type row after alignment.
        # A clean shell must not fall back to that release.
        "CDP_PIPELINE_RELEASE": "extraction-v3",
    }
    for key, value in _product.items():
        if _respect:
            os.environ.setdefault(key, value)
        else:
            os.environ[key] = value

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "results.jsonl"
    lock = threading.Lock()

    try:
        engine_probe = _probe_ocr_engines()
    except Exception as exc:  # noqa: BLE001
        engine_probe = {
            "paddleocr": "ERROR",
            "rapidocr": "ERROR",
            "tesseract": "ERROR",
            "all_observed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write_json(out_dir / "engines_probe.json", engine_probe)
    print(f"engines_probe={json.dumps(engine_probe)}", flush=True)
    if not engine_probe.get("all_observed"):
        print(
            "WARNING: OCR engine probe incomplete — cascade fill may degrade",
            flush=True,
        )

    docs = _list_documents(args.zip)
    doc_filter = [
        part.strip().replace("\\", "/")
        for part in str(args.documents or "").split(",")
        if part.strip()
    ]
    if doc_filter:
        wanted = set(doc_filter)
        selected = [d for d in docs if d.replace("\\", "/") in wanted]
        missing = sorted(wanted - {d.replace("\\", "/") for d in selected})
        if missing:
            print(f"WARNING: documents not in zip: {missing}", flush=True)
    else:
        selected = docs[args.offset : args.offset + args.limit]
    done = _load_done(ledger) if args.resume else set()
    pending = [d for d in selected if _claim_slug(d) not in done]
    print(
        f"strategy=field-cascade-v12 selected={len(selected)} "
        f"already_done={len(selected) - len(pending)} pending={len(pending)} "
        f"workers={args.workers}",
        flush=True,
    )

    rows_new: list[dict[str, Any]] = []
    # Apply stage env to this process so OCR/app pool workers inherit knobs.
    for key, value in _stage_env().items():
        os.environ[key] = value
    pool_workers = max(1, min(int(args.workers), len(pending))) if pending else 1
    ocr_executor: ProcessPoolExecutor | None = None
    app_executor: ProcessPoolExecutor | None = None
    if _app_pool_enabled() and pending:
        print(
            f"app_worker_pool=on workers={pool_workers} "
            f"(amortize registration imports + template SIFT)",
            flush=True,
        )
        app_executor = _spawn_pool(pool_workers)
    if _ocr_pool_enabled() and pending:
        print(
            f"ocr_worker_pool=on workers={pool_workers} "
            f"(amortize Paddle/Rapid cold start across claims)",
            flush=True,
        )
        ocr_executor = _spawn_pool(pool_workers)
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    _process_one,
                    document=document,
                    out_dir=out_dir,
                    dataset_yaml=args.dataset,
                    document_type=args.document_type,
                    keep_heavy=args.keep_heavy,
                    ocr_executor=ocr_executor,
                    app_executor=app_executor,
                ): document
                for document in pending
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                document = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "finished": True,
                        "claim_id": _claim_slug(document),
                        "document": document,
                        "bundle_id": _bundle_id(document),
                        "group_id": _group_id(document),
                        "registration_ok": False,
                        "completed": False,
                        "true_stp": False,
                        "disposition": "WORKER_ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                        "ts": _utc_now(),
                    }
                _append_ledger(ledger, row, lock)
                rows_new.append(row)
                if i % 5 == 0 or i == len(futures):
                    stp = sum(1 for r in rows_new if r.get("true_stp"))
                    reg = sum(1 for r in rows_new if r.get("registration_ok"))
                    msg = (
                        f"progress {i}/{len(futures)} newest={row.get('claim_id')} "
                        f"batch_reg={reg}/{len(rows_new)} batch_true_stp={stp}/{len(rows_new)} "
                        f"disp={row.get('disposition')}"
                    )
                    # Always land progress on disk (survives stdout pipe stalls).
                    try:
                        (out_dir / "progress.txt").write_text(
                            msg + f"\n{ _utc_now() }\n", encoding="utf-8"
                        )
                    except OSError:
                        pass
                    print(msg, flush=True)
    finally:
        _shutdown_process_pool(ocr_executor)
        _shutdown_process_pool(app_executor)

    all_rows: list[dict[str, Any]] = []
    if ledger.exists():
        with ledger.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    all_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    by_id: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        cid = str(row.get("claim_id") or "")
        if cid:
            by_id[cid] = row
    final_rows = [by_id[_claim_slug(d)] for d in selected if _claim_slug(d) in by_id]
    summary = _summarize(final_rows, limit=len(selected))
    summary["workers"] = args.workers
    summary["engines_probe"] = engine_probe
    try:
        summary["out_dir"] = str(out_dir.relative_to(ROOT))
    except ValueError:
        summary["out_dir"] = str(out_dir)
    summary_path = out_dir / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
