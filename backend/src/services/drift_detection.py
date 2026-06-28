"""
PSI-based feature drift detection service for V4 production monitoring.

Numeric features — 7-bin quantile scheme derived from reference percentiles:
  Bin 0: (-inf, p10)       ref_pct = 10%
  Bin 1: [p10, p25)        ref_pct = 15%
  Bin 2: [p25, p50)        ref_pct = 25%
  Bin 3: [p50, p75)        ref_pct = 25%
  Bin 4: [p75, p90)        ref_pct = 15%
  Bin 5: [p90, p99)        ref_pct =  9%
  Bin 6: [p99, +inf)       ref_pct =  1%

  PSI = Σ (p_prod - p_ref) * ln(p_prod / p_ref)

Categorical features — each distinct category value is its own bin.
  PSI computed identically using reference value_pcts.

Small-value guard: any prod bin with 0 observations gets ε = 1e-4 to prevent ln(0).

Thresholds:
  PSI < 0.10  → severity = "low"     → Recommendation: Monitor
  0.10 ≤ PSI < 0.25 → severity = "medium" → Recommendation: Investigate
  PSI ≥ 0.25 → severity = "high"    → Recommendation: Consider Retraining

Read-only — does NOT modify champions, pipeline, or scheduler.
"""

import json
import math
import pathlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import Prediction

_REF_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "models" / "metadata" / "v4_reference_distributions.json"
)
_HISTORY_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "models" / "metadata" / "drift_history.json"
)

_DEFAULT_PSI_LOW  = 0.10
_DEFAULT_PSI_HIGH = 0.25
_EPS = 1e-4
_MIN_PROD_SAMPLE = 30          # skip PSI if fewer production rows than this
_HISTORY_MAX_ENTRIES = 90      # rotate history file at this many snapshots

# Fixed reference percentages for the 7-bin numeric scheme (sums to 100%)
_NUMERIC_REF_PCTS = [10.0, 15.0, 25.0, 25.0, 15.0, 9.0, 1.0]

_NUMERIC_BIN_LABELS = [
    "< p10", "p10-p25", "p25-p50", "p50-p75", "p75-p90", "p90-p99", "≥ p99"
]


# ─── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    feature: str
    feature_type: str           # "numeric" | "categorical"
    psi: float
    severity: str               # "low" | "medium" | "high" | "insufficient_data"
    ref_n: int
    prod_n: int
    ref_bins: List[float]       # reference percentages per bin
    prod_bins: List[float]      # production percentages per bin
    bin_labels: List[str]
    alert: Optional[str] = None


@dataclass
class DriftSnapshot:
    computed_at: str
    window_days: int
    prod_sample_n: int
    psi_low_threshold: float
    psi_high_threshold: float
    feature_results: List[FeatureDriftResult]
    overall_psi: float
    overall_severity: str       # "low" | "medium" | "high" | "no_data"
    recommendation: str         # "Monitor" | "Investigate" | "Consider Retraining" | "Insufficient Data"
    alerts: List[str]
    high_drift_features: List[str]
    medium_drift_features: List[str]
    features_computed: int
    features_skipped: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Reference distribution loader ───────────────────────────────────────────

def _load_reference() -> Dict[str, Any]:
    if not _REF_PATH.exists():
        raise FileNotFoundError(
            f"Reference distributions not found at {_REF_PATH}. "
            "Run: python backend/experiments/generate_v4_reference_distributions.py"
        )
    return json.loads(_REF_PATH.read_text(encoding="utf-8"))


# ─── PSI computation helpers ──────────────────────────────────────────────────

def _psi_value(ref_pcts: List[float], prod_pcts: List[float]) -> float:
    """Compute PSI given two lists of percentage values (0-100 scale)."""
    total = 0.0
    for r, p in zip(ref_pcts, prod_pcts):
        r_frac = max(r / 100.0, _EPS)
        p_frac = max(p / 100.0, _EPS)
        total += (p_frac - r_frac) * math.log(p_frac / r_frac)
    return round(total, 6)


def _severity(psi: float, psi_low: float, psi_high: float) -> str:
    if psi < psi_low:
        return "low"
    if psi < psi_high:
        return "medium"
    return "high"


def _alert_msg(feature: str, severity: str, psi: float) -> Optional[str]:
    if severity == "high":
        return f"HIGH drift detected in '{feature}': PSI={psi:.4f} (threshold ≥{_DEFAULT_PSI_HIGH})"
    if severity == "medium":
        return f"MEDIUM drift in '{feature}': PSI={psi:.4f} (threshold ≥{_DEFAULT_PSI_LOW})"
    return None


# ─── Numeric feature drift ────────────────────────────────────────────────────

def _compute_numeric_psi(
    feature: str,
    ref_stats: Dict[str, Any],
    prod_values: List[float],
    psi_low: float,
    psi_high: float,
) -> FeatureDriftResult:
    """
    Bin production values into the 7-bin quantile scheme derived from reference
    percentiles. Reference bin percentages are fixed by definition (10/15/25/25/15/9/1).
    """
    ref_n = int(ref_stats["count"])
    prod_n = len(prod_values)

    breakpoints = [
        ref_stats["p10"],
        ref_stats["p25"],
        ref_stats["p50"],
        ref_stats["p75"],
        ref_stats["p90"],
        ref_stats["p99"],
    ]

    if prod_n < _MIN_PROD_SAMPLE:
        return FeatureDriftResult(
            feature=feature,
            feature_type="numeric",
            psi=0.0,
            severity="insufficient_data",
            ref_n=ref_n,
            prod_n=prod_n,
            ref_bins=_NUMERIC_REF_PCTS[:],
            prod_bins=[0.0] * 7,
            bin_labels=_NUMERIC_BIN_LABELS[:],
            alert=f"Skipped: only {prod_n} production samples (min {_MIN_PROD_SAMPLE})",
        )

    # Count production observations per bin
    counts = [0] * 7
    for v in prod_values:
        if v < breakpoints[0]:
            counts[0] += 1
        elif v < breakpoints[1]:
            counts[1] += 1
        elif v < breakpoints[2]:
            counts[2] += 1
        elif v < breakpoints[3]:
            counts[3] += 1
        elif v < breakpoints[4]:
            counts[4] += 1
        elif v < breakpoints[5]:
            counts[5] += 1
        else:
            counts[6] += 1

    prod_pcts = [round(c / prod_n * 100.0, 4) for c in counts]
    psi = _psi_value(_NUMERIC_REF_PCTS, prod_pcts)
    sev = _severity(psi, psi_low, psi_high)

    return FeatureDriftResult(
        feature=feature,
        feature_type="numeric",
        psi=round(psi, 6),
        severity=sev,
        ref_n=ref_n,
        prod_n=prod_n,
        ref_bins=_NUMERIC_REF_PCTS[:],
        prod_bins=prod_pcts,
        bin_labels=_NUMERIC_BIN_LABELS[:],
        alert=_alert_msg(feature, sev, psi),
    )


# ─── Categorical feature drift ────────────────────────────────────────────────

def _compute_categorical_psi(
    feature: str,
    ref_stats: Dict[str, Any],
    prod_values: List[Any],
    psi_low: float,
    psi_high: float,
) -> FeatureDriftResult:
    """
    Each category value is a bin. Reference percentages come from value_pcts.
    New production categories not seen in reference are bucketed into an
    '<other>' bin with ref_pct=ε.
    """
    ref_n = int(ref_stats["count"])
    prod_n = len(prod_values)
    ref_pcts_map: Dict[str, float] = dict(ref_stats["value_pcts"])

    if prod_n < _MIN_PROD_SAMPLE:
        return FeatureDriftResult(
            feature=feature,
            feature_type="categorical",
            psi=0.0,
            severity="insufficient_data",
            ref_n=ref_n,
            prod_n=prod_n,
            ref_bins=[],
            prod_bins=[],
            bin_labels=list(ref_pcts_map.keys()),
            alert=f"Skipped: only {prod_n} production samples (min {_MIN_PROD_SAMPLE})",
        )

    # Count production observations per known category
    known_cats = list(ref_pcts_map.keys())
    prod_counts: Dict[str, int] = {cat: 0 for cat in known_cats}
    other_count = 0

    for v in prod_values:
        str_v = str(v) if v is not None else None
        if str_v in prod_counts:
            prod_counts[str_v] += 1
        else:
            other_count += 1

    ref_pcts_list  = [ref_pcts_map[cat] for cat in known_cats]
    prod_pcts_list = [round(prod_counts[cat] / prod_n * 100.0, 4) for cat in known_cats]
    bin_labels = known_cats[:]

    # If there are unseen categories, add an <other> bin
    if other_count > 0:
        ref_pcts_list.append(_EPS * 100.0)  # near-zero reference percentage
        prod_pcts_list.append(round(other_count / prod_n * 100.0, 4))
        bin_labels.append("<other>")

    psi = _psi_value(ref_pcts_list, prod_pcts_list)
    sev = _severity(psi, psi_low, psi_high)

    return FeatureDriftResult(
        feature=feature,
        feature_type="categorical",
        psi=round(psi, 6),
        severity=sev,
        ref_n=ref_n,
        prod_n=prod_n,
        ref_bins=ref_pcts_list,
        prod_bins=prod_pcts_list,
        bin_labels=bin_labels,
        alert=_alert_msg(feature, sev, psi),
    )


# ─── Main drift service ───────────────────────────────────────────────────────

class DriftDetectionService:
    """
    Read-only drift detection service.

    Usage:
        svc = DriftDetectionService(db)
        snapshot = await svc.compute_drift(window_days=30)
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def compute_drift(
        self,
        window_days: int = 30,
        psi_low: float = _DEFAULT_PSI_LOW,
        psi_high: float = _DEFAULT_PSI_HIGH,
    ) -> DriftSnapshot:
        """
        Extract production feature values from inference_features_json,
        compute per-feature PSI vs reference distributions, return DriftSnapshot.
        """
        ref = _load_reference()
        now = datetime.utcnow()
        since = now - timedelta(days=window_days)

        # Pull all inference_features_json records in the window
        rows = (await self._db.execute(
            select(Prediction.inference_features_json)
            .where(
                Prediction.inference_features_json.isnot(None),
                Prediction.created_at >= since,
            )
        )).scalars().all()

        prod_sample_n = len(rows)

        # Parse each JSONB row (asyncpg may return dict or str)
        parsed: List[Dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                parsed.append(row)
            elif isinstance(row, str):
                try:
                    parsed.append(json.loads(row))
                except (json.JSONDecodeError, TypeError):
                    pass

        feature_results: List[FeatureDriftResult] = []
        features_computed = 0
        features_skipped = 0

        # ── Numeric features ──────────────────────────────────────────────────
        for feat, stats in ref["numeric"].items():
            values: List[float] = []
            for record in parsed:
                v = record.get(feat)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

            result = _compute_numeric_psi(feat, stats, values, psi_low, psi_high)
            feature_results.append(result)
            if result.severity == "insufficient_data":
                features_skipped += 1
            else:
                features_computed += 1

        # ── Categorical features ──────────────────────────────────────────────
        for feat, stats in ref["categorical"].items():
            values_cat: List[Any] = []
            for record in parsed:
                v = record.get(feat)
                if v is not None:
                    values_cat.append(str(v))

            result = _compute_categorical_psi(feat, stats, values_cat, psi_low, psi_high)
            feature_results.append(result)
            if result.severity == "insufficient_data":
                features_skipped += 1
            else:
                features_computed += 1

        # ── Overall PSI (mean of computed features) ───────────────────────────
        computed_psis = [
            r.psi for r in feature_results if r.severity != "insufficient_data"
        ]
        if computed_psis:
            overall_psi = round(sum(computed_psis) / len(computed_psis), 6)
            overall_sev = _severity(overall_psi, psi_low, psi_high)
        else:
            overall_psi = 0.0
            overall_sev = "no_data"

        # ── Recommendation ────────────────────────────────────────────────────
        rec_map = {
            "low":     "Monitor",
            "medium":  "Investigate",
            "high":    "Consider Retraining",
            "no_data": "Insufficient Data — accumulate ≥30 V2 quotes",
        }
        recommendation = rec_map[overall_sev]

        # ── Alerts ────────────────────────────────────────────────────────────
        alerts: List[str] = []
        high_feats: List[str] = []
        medium_feats: List[str] = []

        for r in feature_results:
            if r.severity == "high":
                high_feats.append(r.feature)
                if r.alert:
                    alerts.append(r.alert)
            elif r.severity == "medium":
                medium_feats.append(r.feature)
                if r.alert:
                    alerts.append(r.alert)

        if len(high_feats) >= 3:
            alerts.insert(0,
                f"CRITICAL: {len(high_feats)} features show HIGH drift — "
                "population shift detected. Consider retraining V5."
            )
        elif overall_psi >= psi_high and overall_sev == "high":
            alerts.insert(0,
                f"Overall PSI={overall_psi:.4f} exceeds high threshold ({psi_high}). "
                "Investigate population shift."
            )

        return DriftSnapshot(
            computed_at=now.isoformat(),
            window_days=window_days,
            prod_sample_n=prod_sample_n,
            psi_low_threshold=psi_low,
            psi_high_threshold=psi_high,
            feature_results=feature_results,
            overall_psi=overall_psi,
            overall_severity=overall_sev,
            recommendation=recommendation,
            alerts=alerts,
            high_drift_features=high_feats,
            medium_drift_features=medium_feats,
            features_computed=features_computed,
            features_skipped=features_skipped,
        )


# ─── History persistence ──────────────────────────────────────────────────────

def save_drift_snapshot(snapshot: DriftSnapshot) -> None:
    """Append snapshot to drift_history.json (max _HISTORY_MAX_ENTRIES entries)."""
    history: List[Dict[str, Any]] = []
    if _HISTORY_PATH.exists():
        try:
            history = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            history = []

    history.append(snapshot.to_dict())

    # Keep only the last N entries
    if len(history) > _HISTORY_MAX_ENTRIES:
        history = history[-_HISTORY_MAX_ENTRIES:]

    _HISTORY_PATH.write_text(
        json.dumps(history, indent=2, default=str),
        encoding="utf-8",
    )


def load_drift_history(limit: int = 30) -> List[Dict[str, Any]]:
    """Return the most recent `limit` drift snapshots from history file."""
    if not _HISTORY_PATH.exists():
        return []
    try:
        history: List[Dict[str, Any]] = json.loads(
            _HISTORY_PATH.read_text(encoding="utf-8")
        )
        return history[-limit:] if len(history) > limit else history
    except (json.JSONDecodeError, IOError):
        return []
