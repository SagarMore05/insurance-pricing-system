"""
Phase 5B Governance Gates.

Seven gates must ALL pass before a champion promotion is allowed:

  1. human_approval      — ApprovalRequest with status="approved" exists
  2. shadow_completed    — shadow_predictions has COMPLETED records
  3. shadow_count        — at least MIN_SHADOW_OBSERVATIONS completed
  4. no_critical_drift   — no "high" drift severity in last 30 days (or bypassed)
  5. performance         — challenger outperforms champion on primary metric
  6. artifacts_exist     — candidate model files present on disk
  7. registry_valid      — ModelRegistry.validate_paths() passes

All gates return a GateResult. The caller collects them and proceeds only when
all gates have passed=True.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ApprovalRequest, CandidateModel, ShadowPrediction

logger = logging.getLogger("insurance_api.promotions.gates")

# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_SHADOW_OBSERVATIONS = 10          # minimum completed shadow records
DRIFT_LOOKBACK_DAYS     = 30          # window for critical drift check
FREQ_PERF_THRESHOLD     = 0.002       # challenger must beat champion by this delta
SEV_PERF_THRESHOLD      = 0.010       # on R²

_DRIFT_HISTORY_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "models" / "metadata" / "drift_history.json"
)


# ── Gate result ───────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    gate: str
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate":    self.gate,
            "passed":  self.passed,
            "reason":  self.reason,
            "details": self.details,
        }


@dataclass
class GatesReport:
    all_passed: bool
    gates: Dict[str, GateResult]
    passed_count: int
    failed_count: int
    failed_gates: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_passed":   self.all_passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "failed_gates": self.failed_gates,
            "gates":        {k: v.to_dict() for k, v in self.gates.items()},
        }


# ── Gate checker ──────────────────────────────────────────────────────────────

class GovernanceGates:
    """Runs all seven governance gates and returns a GatesReport."""

    def __init__(
        self,
        bypass_shadow: bool = False,
        bypass_drift: bool = False,
        min_shadow_observations: int = MIN_SHADOW_OBSERVATIONS,
    ) -> None:
        self.bypass_shadow = bypass_shadow
        self.bypass_drift  = bypass_drift
        self.min_shadow    = min_shadow_observations

    async def check_all(
        self,
        db: AsyncSession,
        *,
        approval_id: Optional[str],
        freq_candidate_id: str,
        sev_candidate_id: str,
        freq_candidate_metrics: Dict[str, Any],
        sev_candidate_metrics: Dict[str, Any],
        freq_champion_score: float,
        sev_champion_score: float,
    ) -> GatesReport:
        """
        Run all seven gates and return a GatesReport.

        Parameters
        ----------
        db                      : async DB session
        approval_id             : UUID string of the ApprovalRequest (required)
        freq_candidate_id       : UUID of the frequency CandidateModel
        sev_candidate_id        : UUID of the severity CandidateModel
        freq_candidate_metrics  : frequency candidate's metrics dict
        sev_candidate_metrics   : severity candidate's metrics dict
        freq_champion_score     : current champion frequency ROC-AUC
        sev_champion_score      : current champion severity R²
        """
        gates: Dict[str, GateResult] = {}

        # Gate 1 — Human approval
        gates["human_approval"] = await self._gate_human_approval(db, approval_id)

        # Gate 2 — Shadow deployment completed records exist
        gates["shadow_completed"] = await self._gate_shadow_completed(db)

        # Gate 3 — Minimum shadow observations
        gates["shadow_count"] = await self._gate_shadow_count(db)

        # Gate 4 — No critical drift
        gates["no_critical_drift"] = self._gate_no_critical_drift()

        # Gate 5 — Challenger outperforms champion
        gates["performance"] = self._gate_performance(
            freq_candidate_metrics, sev_candidate_metrics,
            freq_champion_score, sev_champion_score,
        )

        # Gate 6 — Artifact files exist on disk
        gates["artifacts_exist"] = await self._gate_artifacts_exist(
            db, freq_candidate_id, sev_candidate_id
        )

        # Gate 7 — Registry validation
        gates["registry_valid"] = self._gate_registry_valid()

        passed = {k: v for k, v in gates.items() if v.passed}
        failed = {k: v for k, v in gates.items() if not v.passed}

        return GatesReport(
            all_passed=len(failed) == 0,
            gates=gates,
            passed_count=len(passed),
            failed_count=len(failed),
            failed_gates=list(failed.keys()),
        )

    # ── Gate 1: Human approval ────────────────────────────────────────────────

    async def _gate_human_approval(
        self, db: AsyncSession, approval_id: Optional[str]
    ) -> GateResult:
        gate = "human_approval"
        if not approval_id:
            return GateResult(
                gate=gate,
                passed=False,
                reason="No approval_id provided. Human approval is mandatory for promotion.",
            )
        try:
            uid = uuid.UUID(approval_id)
        except ValueError:
            return GateResult(
                gate=gate, passed=False,
                reason=f"approval_id '{approval_id}' is not a valid UUID.",
            )

        result = await db.execute(
            select(ApprovalRequest).where(ApprovalRequest.request_id == uid)
        )
        req = result.scalar_one_or_none()

        if req is None:
            return GateResult(
                gate=gate, passed=False,
                reason=f"ApprovalRequest {approval_id} not found.",
            )
        if req.status != "approved":
            return GateResult(
                gate=gate, passed=False,
                reason=f"ApprovalRequest {approval_id} has status='{req.status}' (must be 'approved').",
                details={"status": req.status, "reviewed_by": req.reviewed_by},
            )
        return GateResult(
            gate=gate, passed=True,
            reason=f"ApprovalRequest {approval_id} approved by {req.reviewed_by}.",
            details={"reviewed_by": req.reviewed_by, "reviewed_at": str(req.reviewed_at)},
        )

    # ── Gate 2: Shadow deployment has completed records ───────────────────────

    async def _gate_shadow_completed(self, db: AsyncSession) -> GateResult:
        gate = "shadow_completed"
        if self.bypass_shadow:
            return GateResult(
                gate=gate, passed=True,
                reason="Shadow gate bypassed (bypass_shadow=True — dev/test mode).",
            )
        count_q = await db.execute(
            select(func.count()).select_from(ShadowPrediction)
            .where(ShadowPrediction.status == "COMPLETED")
        )
        count = count_q.scalar() or 0
        if count > 0:
            return GateResult(
                gate=gate, passed=True,
                reason=f"{count} completed shadow observations found.",
                details={"completed_shadow_count": count},
            )
        return GateResult(
            gate=gate, passed=False,
            reason="No completed shadow observations found. Shadow deployment must run first.",
            details={"completed_shadow_count": 0},
        )

    # ── Gate 3: Minimum shadow observations ──────────────────────────────────

    async def _gate_shadow_count(self, db: AsyncSession) -> GateResult:
        gate = "shadow_count"
        if self.bypass_shadow:
            return GateResult(
                gate=gate, passed=True,
                reason=f"Shadow count gate bypassed (bypass_shadow=True). Threshold={self.min_shadow}.",
            )
        count_q = await db.execute(
            select(func.count()).select_from(ShadowPrediction)
            .where(ShadowPrediction.status == "COMPLETED")
        )
        count = count_q.scalar() or 0
        if count >= self.min_shadow:
            return GateResult(
                gate=gate, passed=True,
                reason=f"{count} ≥ {self.min_shadow} minimum shadow observations.",
                details={"count": count, "threshold": self.min_shadow},
            )
        return GateResult(
            gate=gate, passed=False,
            reason=f"Only {count} completed shadow records; need ≥ {self.min_shadow}.",
            details={"count": count, "threshold": self.min_shadow},
        )

    # ── Gate 4: No critical drift ─────────────────────────────────────────────

    def _gate_no_critical_drift(self) -> GateResult:
        gate = "no_critical_drift"
        if self.bypass_drift:
            return GateResult(
                gate=gate, passed=True,
                reason="Drift gate bypassed (bypass_drift=True — dev/test mode).",
            )
        try:
            if not _DRIFT_HISTORY_PATH.exists():
                return GateResult(
                    gate=gate, passed=True,
                    reason="No drift history file found — assuming no drift detected.",
                )
            with open(_DRIFT_HISTORY_PATH) as fh:
                history = json.load(fh)

            cutoff = datetime.utcnow() - timedelta(days=DRIFT_LOOKBACK_DAYS)
            recent_high = [
                e for e in history.get("snapshots", [])
                if e.get("severity") == "high"
                and datetime.fromisoformat(e.get("timestamp", "2000-01-01")) >= cutoff
            ]
            if not recent_high:
                return GateResult(
                    gate=gate, passed=True,
                    reason=f"No high-severity drift detected in the last {DRIFT_LOOKBACK_DAYS} days.",
                    details={"lookback_days": DRIFT_LOOKBACK_DAYS},
                )
            return GateResult(
                gate=gate, passed=False,
                reason=f"{len(recent_high)} high-severity drift event(s) in last {DRIFT_LOOKBACK_DAYS} days.",
                details={"high_drift_count": len(recent_high)},
            )
        except Exception as exc:
            logger.warning("Drift gate check failed: %s — treating as passed.", exc)
            return GateResult(
                gate=gate, passed=True,
                reason=f"Drift history unreadable ({exc}) — gate treated as passed.",
            )

    # ── Gate 5: Challenger outperforms champion ───────────────────────────────

    def _gate_performance(
        self,
        freq_metrics: Dict[str, Any],
        sev_metrics: Dict[str, Any],
        freq_champion_score: float,
        sev_champion_score: float,
    ) -> GateResult:
        gate = "performance"
        issues = []
        details: Dict[str, Any] = {}

        # Frequency: ROC-AUC
        freq_auc = float(freq_metrics.get("roc_auc", 0.0))
        freq_delta = freq_auc - freq_champion_score
        details["frequency_roc_auc_challenger"] = round(freq_auc, 6)
        details["frequency_roc_auc_champion"]   = round(freq_champion_score, 6)
        details["frequency_delta"]              = round(freq_delta, 6)
        details["frequency_threshold"]          = FREQ_PERF_THRESHOLD
        if freq_delta <= FREQ_PERF_THRESHOLD:
            issues.append(
                f"Frequency ROC-AUC delta {freq_delta:+.4f} ≤ threshold {FREQ_PERF_THRESHOLD}"
            )

        # Severity: R²
        sev_r2 = float(sev_metrics.get("r2", -9999.0))
        sev_delta = sev_r2 - sev_champion_score
        details["severity_r2_challenger"] = round(sev_r2, 6)
        details["severity_r2_champion"]   = round(sev_champion_score, 6)
        details["severity_delta"]         = round(sev_delta, 6)
        details["severity_threshold"]     = SEV_PERF_THRESHOLD
        if sev_delta <= SEV_PERF_THRESHOLD:
            issues.append(
                f"Severity R² delta {sev_delta:+.4f} ≤ threshold {SEV_PERF_THRESHOLD}"
            )

        if issues:
            return GateResult(
                gate=gate, passed=False,
                reason="Challenger does not sufficiently outperform champion: " + "; ".join(issues),
                details=details,
            )
        return GateResult(
            gate=gate, passed=True,
            reason=(
                f"Challenger outperforms champion — "
                f"Freq ΔAUC={freq_delta:+.4f}, Sev ΔR²={sev_delta:+.4f}."
            ),
            details=details,
        )

    # ── Gate 6: Artifact files exist ─────────────────────────────────────────

    async def _gate_artifacts_exist(
        self,
        db: AsyncSession,
        freq_candidate_id: str,
        sev_candidate_id: str,
    ) -> GateResult:
        gate = "artifacts_exist"
        missing: List[str] = []
        details: Dict[str, Any] = {}

        for candidate_id, label in [
            (freq_candidate_id, "frequency"),
            (sev_candidate_id, "severity"),
        ]:
            try:
                uid = uuid.UUID(candidate_id)
            except ValueError:
                missing.append(f"{label}: invalid candidate ID")
                continue

            result = await db.execute(
                select(CandidateModel).where(CandidateModel.model_id == uid)
            )
            cand = result.scalar_one_or_none()
            if cand is None:
                missing.append(f"{label}: CandidateModel {candidate_id} not found")
                continue
            if cand.status != "COMPLETED":
                missing.append(f"{label}: candidate status={cand.status} (must be COMPLETED)")
                continue

            artifact_dir = cand.artifact_path
            if not artifact_dir or not os.path.isdir(artifact_dir):
                missing.append(f"{label}: artifact_dir '{artifact_dir}' not found on disk")
                continue

            model_pkl = os.path.join(artifact_dir, "model.pkl")
            if not os.path.exists(model_pkl):
                missing.append(f"{label}: model.pkl missing from {artifact_dir}")
            else:
                details[f"{label}_model_pkl"] = model_pkl

        if missing:
            return GateResult(
                gate=gate, passed=False,
                reason="Missing or invalid artifacts: " + "; ".join(missing),
                details=details,
            )
        return GateResult(
            gate=gate, passed=True,
            reason="All candidate artifact files verified on disk.",
            details=details,
        )

    # ── Gate 7: Registry validation ───────────────────────────────────────────

    def _gate_registry_valid(self) -> GateResult:
        gate = "registry_valid"
        try:
            from src.models.model_registry import ModelRegistry
            registry = ModelRegistry()
            paths = registry.validate_paths()
            failed_keys = [k for k, ok in paths.items() if not ok and "v1" not in k.lower()]
            if not failed_keys:
                return GateResult(
                    gate=gate, passed=True,
                    reason="Champion registry paths all valid.",
                    details=paths,
                )
            return GateResult(
                gate=gate, passed=False,
                reason=f"Registry path validation failed for: {', '.join(failed_keys)}",
                details=paths,
            )
        except Exception as exc:
            return GateResult(
                gate=gate, passed=False,
                reason=f"Registry validation raised exception: {exc}",
            )
