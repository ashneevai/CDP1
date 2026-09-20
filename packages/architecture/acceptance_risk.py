"""Acceptance-risk estimation from evidence features (redesign calibration role).

Uses existing Platt/isotonic field confidence when available. Optional LightGBM
model path via ``CDP_ACCEPTANCE_RISK_LGBM`` — fail-closed to heuristic when
missing. Does not auto-accept fields.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AcceptanceRiskEstimate:
    risk: float
    method: str
    features: dict[str, float]
    review_recommended: bool


def _env_on(name: str, default: str = "0") -> bool:
    return (os.environ.get(name) or default).strip().casefold() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def estimate_acceptance_risk(
    *,
    field_name: str,
    calibrated_confidence: float | None = None,
    engine_count: int = 0,
    dual_engine_agree: bool = False,
    gpt4o_only: bool = False,
    gap_class: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> AcceptanceRiskEstimate:
    """Higher risk → more likely to need HITL.

    GPT-only monetary candidates are forced high-risk (not monetary authority).
    """
    features = {
        "calibrated_confidence": float(calibrated_confidence or 0.0),
        "engine_count": float(engine_count),
        "dual_engine_agree": 1.0 if dual_engine_agree else 0.0,
        "gpt4o_only": 1.0 if gpt4o_only else 0.0,
    }
    if extra:
        for k, v in extra.items():
            if isinstance(v, (int, float)):
                features[str(k)] = float(v)

    method = "heuristic_v1"
    if gpt4o_only and (field_name or "").casefold() in {
        "total_charge",
        "total_charges",
        "charges",
        "charge_amount",
    }:
        risk = 0.95
        method = "gpt_not_monetary_authority"
    elif dual_engine_agree and (calibrated_confidence or 0) >= 0.95:
        risk = 0.08
    elif (calibrated_confidence or 0) >= 0.9 and engine_count >= 2:
        risk = 0.2
    elif gap_class in {"EMPTY_FINANCIAL_INK", "HANDWRITING_UNREADABLE"}:
        risk = 0.9
    elif gap_class == "LINE_SUM_UNCORROBORATED":
        risk = 0.75
    else:
        conf = float(calibrated_confidence or 0.5)
        risk = max(0.05, min(0.95, 1.0 - conf))

    if _env_on("CDP_ACCEPTANCE_RISK_LGBM"):
        # Optional LightGBM artifact — never required for STP.
        try:
            risk_lgbm = _try_lightgbm_risk(features)
            if risk_lgbm is not None:
                risk = risk_lgbm
                method = "lightgbm_acceptance_risk"
        except Exception:  # noqa: BLE001 — calibration must fail closed
            method = f"{method}|lgbm_unavailable"

    return AcceptanceRiskEstimate(
        risk=float(risk),
        method=method,
        features=features,
        review_recommended=risk >= 0.45,
    )


def _try_lightgbm_risk(features: dict[str, float]) -> float | None:
    path = (os.environ.get("CDP_ACCEPTANCE_RISK_LGBM_PATH") or "").strip()
    if not path:
        return None
    import lightgbm as lgb  # type: ignore
    import numpy as np

    booster = lgb.Booster(model_file=path)
    keys = sorted(features.keys())
    row = np.array([[features[k] for k in keys]], dtype=float)
    pred = booster.predict(row)
    return float(pred[0])
