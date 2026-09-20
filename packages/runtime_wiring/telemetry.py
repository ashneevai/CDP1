"""Runtime stage wiring ledger — proves redesign stages execute in ops path."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageInvocation:
    stage: str
    enabled: bool
    configured: bool
    claim_id: str | None = None
    page_id: str | None = None
    field_name: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None
    elapsed_ms: float = 0.0
    exception: str | None = None
    bypassed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StageTelemetry:
    """Thread-safe invocation ledger for one claim or process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.invocations: list[StageInvocation] = []

    def record(self, inv: StageInvocation) -> StageInvocation:
        with self._lock:
            self.invocations.append(inv)
        return inv

    @contextmanager
    def track(
        self,
        stage: str,
        *,
        enabled: bool = True,
        configured: bool = True,
        claim_id: str | None = None,
        page_id: str | None = None,
        field_name: str | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> Iterator[StageInvocation]:
        inv = StageInvocation(
            stage=stage,
            enabled=enabled,
            configured=configured,
            claim_id=claim_id,
            page_id=page_id,
            field_name=field_name,
            inputs=dict(inputs or {}),
        )
        if not enabled:
            inv.bypassed = True
            inv.fallback_reason = inv.fallback_reason or "STAGE_DISABLED"
            self.record(inv)
            yield inv
            return
        t0 = time.perf_counter()
        try:
            yield inv
        except Exception as exc:
            inv.exception = f"{type(exc).__name__}: {exc}"
            inv.fallback_reason = inv.fallback_reason or "STAGE_EXCEPTION"
            raise
        finally:
            inv.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.record(inv)

    def summary(self) -> dict[str, Any]:
        by_stage: dict[str, dict[str, Any]] = {}
        for inv in self.invocations:
            slot = by_stage.setdefault(
                inv.stage,
                {
                    "invocations": 0,
                    "bypassed": 0,
                    "exceptions": 0,
                    "enabled_runs": 0,
                    "total_ms": 0.0,
                },
            )
            slot["invocations"] += 1
            slot["total_ms"] += inv.elapsed_ms
            if inv.bypassed:
                slot["bypassed"] += 1
            elif inv.enabled:
                slot["enabled_runs"] += 1
            if inv.exception:
                slot["exceptions"] += 1
        required = [
            "package_intelligence",
            "image_evidence",
            "geometry_authority",
            "monetary_crop_variants",
            "ocr_portfolio",
            "candidate_evidence",
            "field_authority",
            "financial_reconciliation",
            "claim_decision",
        ]
        silent_bypass = [
            s
            for s in required
            if by_stage.get(s, {}).get("enabled_runs", 0) == 0
            and by_stage.get(s, {}).get("invocations", 0) == 0
        ]
        return {
            "by_stage": by_stage,
            "required_stages": required,
            "silent_bypass": silent_bypass,
            "wiring_ok": len(silent_bypass) == 0,
            "invocation_count": len(self.invocations),
        }

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "invocations": [i.to_dict() for i in self.invocations],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_THREAD = threading.local()


def get_telemetry() -> StageTelemetry:
    tel = getattr(_THREAD, "telemetry", None)
    if tel is None:
        tel = StageTelemetry()
        _THREAD.telemetry = tel
    return tel


def reset_telemetry() -> StageTelemetry:
    tel = StageTelemetry()
    _THREAD.telemetry = tel
    return tel


def stage_enabled(env_name: str, default: str = "1") -> bool:
    raw = (os.environ.get(env_name) or default).strip().casefold()
    return raw not in {"0", "false", "no", "off", ""}
