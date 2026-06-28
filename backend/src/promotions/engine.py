"""
Phase 5B PromotionEvaluationEngine.

Orchestrates the full promotion lifecycle:
  1. evaluate() — run governance gates, record ModelPromotion in DB
  2. execute()  — atomic promotion (requires status=APPROVED)
  3. rollback() — manual rollback of an ACTIVE promotion

Architecture:
  PromotionEvaluationEngine
    ├── GovernanceGates       (7 gates)
    ├── AtomicPromotion       (backup → install → registry → reload → health)
    └── DB persistence        (ModelPromotion + RollbackHistory rows)

This engine does NOT implement automatic promotion. Every promotion requires
an explicit call to execute() from the API route, which itself requires
status=APPROVED (meaning all gates passed in a prior evaluate() call).
"""
from __future__ import annotations

import logging
import os
import pathlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.models import CandidateModel, ModelPromotion, RollbackHistory, TrainingRun
from src.promotions.atomic import AtomicPromotion
from src.promotions.gates import GovernanceGates

logger = logging.getLogger("insurance_api.promotions.engine")

_BACKEND_ROOT   = pathlib.Path(__file__).resolve().parent.parent.parent
_CHALLENGERS_DIR = _BACKEND_ROOT / "models" / "challengers"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    promotion_id: str
    run_id: str
    status: str          # APPROVED | REJECTED
    all_gates_passed: bool
    gates_report: Dict[str, Any]
    freq_comparison: Dict[str, Any]
    sev_comparison: Dict[str, Any]
    message: str
    created_at: str


@dataclass
class PromotionResult:
    promotion_id: str
    status: str          # ACTIVE | ROLLED_BACK | FAILED
    new_frequency_champion: Optional[Dict[str, Any]]
    new_severity_champion: Optional[Dict[str, Any]]
    rollback_id: Optional[str]
    error: Optional[str]
    steps: list
    duration_seconds: float


@dataclass
class RollbackResult:
    rollback_id: str
    promotion_id: str
    rollback_status: str
    error: Optional[str]


# ── Engine ────────────────────────────────────────────────────────────────────

class Phase5BPromotionEngine:
    """
    Orchestrates promotion evaluation, atomic promotion, and rollback.

    Instantiate per-request (not singleton) since it holds a DB session.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Evaluate ──────────────────────────────────────────────────────────────

    async def evaluate(
        self,
        run_id: str,
        freq_candidate_id: str,
        sev_candidate_id: str,
        approval_id: Optional[str] = None,
        bypass_shadow: bool = False,
        bypass_drift: bool = False,
        notes: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run all governance gates and create a ModelPromotion record.

        Returns APPROVED if all gates pass, REJECTED otherwise.
        Promotion record is persisted regardless of outcome.
        """
        now = datetime.utcnow()

        # Load candidate models
        freq_cand = await self._get_candidate(freq_candidate_id)
        sev_cand  = await self._get_candidate(sev_candidate_id)

        freq_metrics = freq_cand.frequency_metrics or {} if freq_cand else {}
        sev_metrics  = sev_cand.severity_metrics   or {} if sev_cand else {}

        # Load champion scores from registry
        freq_champion_score, sev_champion_score = self._get_champion_scores()

        # Snapshot current champions
        old_freq_champ, old_sev_champ = self._snapshot_champions()

        # Build metric comparison reports
        freq_comparison = self._build_freq_comparison(freq_metrics, freq_champion_score)
        sev_comparison  = self._build_sev_comparison(sev_metrics, sev_champion_score)

        # Create DB record
        promotion_id = uuid.uuid4()
        db_rec = ModelPromotion(
            promotion_id=promotion_id,
            run_id=uuid.UUID(run_id),
            frequency_candidate_id=uuid.UUID(freq_candidate_id) if freq_candidate_id else None,
            severity_candidate_id=uuid.UUID(sev_candidate_id) if sev_candidate_id else None,
            frequency_approval_id=uuid.UUID(approval_id) if approval_id else None,
            old_frequency_champion=old_freq_champ,
            old_severity_champion=old_sev_champ,
            status="EVALUATING",
            notes=notes,
            created_at=now,
        )
        self._db.add(db_rec)
        await self._db.commit()

        # Run governance gates
        gates = GovernanceGates(bypass_shadow=bypass_shadow, bypass_drift=bypass_drift)
        gates_report = await gates.check_all(
            self._db,
            approval_id=approval_id,
            freq_candidate_id=freq_candidate_id,
            sev_candidate_id=sev_candidate_id,
            freq_candidate_metrics=freq_metrics,
            sev_candidate_metrics=sev_metrics,
            freq_champion_score=freq_champion_score,
            sev_champion_score=sev_champion_score,
        )

        # Classify gates
        gates_passed = {k: v.to_dict() for k, v in gates_report.gates.items() if v.passed}
        gates_failed = {k: v.to_dict() for k, v in gates_report.gates.items() if not v.passed}

        final_status = "APPROVED" if gates_report.all_passed else "REJECTED"
        message = (
            f"All {gates_report.passed_count} governance gates passed — ready for promotion."
            if gates_report.all_passed
            else (
                f"{gates_report.failed_count} gate(s) failed: "
                + ", ".join(gates_report.failed_gates)
                + ". Promotion denied."
            )
        )

        # Update DB record
        db_rec.status          = final_status
        db_rec.gates_passed    = gates_passed
        db_rec.gates_failed    = gates_failed
        db_rec.evaluation_report = {
            "gates_summary":    gates_report.to_dict(),
            "freq_comparison":  freq_comparison,
            "sev_comparison":   sev_comparison,
            "evaluated_at":     now.isoformat(),
        }
        db_rec.updated_at = datetime.utcnow()
        await self._db.commit()

        logger.info(
            "Promotion evaluation %s → %s (run=%s)",
            promotion_id, final_status, run_id,
        )

        return EvaluationResult(
            promotion_id=str(promotion_id),
            run_id=run_id,
            status=final_status,
            all_gates_passed=gates_report.all_passed,
            gates_report=gates_report.to_dict(),
            freq_comparison=freq_comparison,
            sev_comparison=sev_comparison,
            message=message,
            created_at=now.isoformat(),
        )

    # ── Execute promotion ─────────────────────────────────────────────────────

    async def execute(
        self,
        promotion_id: str,
        promoted_by: str = "admin",
    ) -> PromotionResult:
        """
        Execute an APPROVED promotion atomically.

        Requires:
          - ModelPromotion with status=APPROVED
          - All candidate artifacts on disk

        On failure: automatic rollback + DB status=ROLLED_BACK.
        """
        t_start = datetime.utcnow()

        db_rec = await self._get_promotion(promotion_id)
        if db_rec is None:
            raise ValueError(f"Promotion {promotion_id} not found.")
        if db_rec.status != "APPROVED":
            raise ValueError(
                f"Promotion {promotion_id} has status={db_rec.status}; must be APPROVED."
            )

        db_rec.status     = "PROMOTING"
        db_rec.updated_at = datetime.utcnow()
        await self._db.commit()

        # Gather artifact paths
        freq_cand = await self._get_candidate(str(db_rec.frequency_candidate_id))
        sev_cand  = await self._get_candidate(str(db_rec.severity_candidate_id))

        freq_artifact_dir = freq_cand.artifact_path if freq_cand else ""
        sev_artifact_dir  = sev_cand.artifact_path  if sev_cand else ""
        freq_algorithm    = freq_cand.algorithm      if freq_cand else "xgboost"
        sev_algorithm     = sev_cand.algorithm       if sev_cand else "xgboost"
        freq_metrics      = freq_cand.frequency_metrics or {}  if freq_cand else {}
        sev_metrics       = sev_cand.severity_metrics   or {}  if sev_cand else {}

        # Find run-level preprocessor path
        run = await self._db.get(TrainingRun, db_rec.run_id)
        preprocessor_path: Optional[str] = None
        if run:
            prep_candidate = _CHALLENGERS_DIR / run.run_tag / "preprocessor_v4.pkl"
            if prep_candidate.exists():
                preprocessor_path = str(prep_candidate)

        version_tag = f"v{t_start.strftime('%Y%m%d%H%M%S')}"

        # Execute atomic promotion
        ap = AtomicPromotion()
        result = ap.execute(
            freq_artifact_dir=freq_artifact_dir,
            sev_artifact_dir=sev_artifact_dir,
            preprocessor_path=preprocessor_path,
            freq_algorithm=freq_algorithm,
            sev_algorithm=sev_algorithm,
            freq_metrics=freq_metrics,
            sev_metrics=sev_metrics,
            promoted_by=promoted_by,
            version_tag=version_tag,
        )

        t_end = datetime.utcnow()
        duration = (t_end - t_start).total_seconds()

        if result.success:
            # Read updated champion state for DB record
            new_freq_champ, new_sev_champ = self._snapshot_champions()
            db_rec.status                 = "ACTIVE"
            db_rec.promoted_by            = promoted_by
            db_rec.promoted_at            = t_end
            db_rec.promotion_duration_seconds = duration
            db_rec.backup_path            = result.backup_path
            db_rec.new_frequency_champion = new_freq_champ
            db_rec.new_severity_champion  = new_sev_champ
            db_rec.updated_at             = t_end
            await self._db.commit()

            logger.info(
                "Promotion %s ACTIVE — freq=%s, sev=%s (%.2fs)",
                promotion_id,
                (new_freq_champ or {}).get("champion"),
                (new_sev_champ or {}).get("champion"),
                duration,
            )
            return PromotionResult(
                promotion_id=promotion_id,
                status="ACTIVE",
                new_frequency_champion=new_freq_champ,
                new_severity_champion=new_sev_champ,
                rollback_id=None,
                error=None,
                steps=result.to_dict()["steps"],
                duration_seconds=duration,
            )

        # Promotion failed — record rollback
        rollback_id = uuid.uuid4()
        rb_record = RollbackHistory(
            rollback_id=rollback_id,
            promotion_id=uuid.UUID(promotion_id),
            restored_frequency_champion=db_rec.old_frequency_champion,
            restored_severity_champion=db_rec.old_severity_champion,
            rollback_reason=result.error or "Promotion failed",
            rollback_trigger="auto",
            rollback_status="SUCCESS" if result.rollback_success else "FAILED",
            rolled_back_by="system",
            rolled_back_at=datetime.utcnow(),
            error_message=None if result.rollback_success else "Rollback itself failed",
        )
        self._db.add(rb_record)

        db_rec.status      = "ROLLED_BACK" if result.rollback_success else "FAILED"
        db_rec.error_message = result.error
        db_rec.updated_at  = datetime.utcnow()
        await self._db.commit()

        return PromotionResult(
            promotion_id=promotion_id,
            status=db_rec.status,
            new_frequency_champion=None,
            new_severity_champion=None,
            rollback_id=str(rollback_id) if result.rollback_triggered else None,
            error=result.error,
            steps=result.to_dict()["steps"],
            duration_seconds=duration,
        )

    # ── Manual rollback ───────────────────────────────────────────────────────

    async def rollback(
        self,
        promotion_id: str,
        reason: str,
        rolled_back_by: str = "admin",
    ) -> RollbackResult:
        """
        Manually roll back an ACTIVE promotion to the pre-promotion champion.

        Uses the backup_path stored at promotion time.
        """
        db_rec = await self._get_promotion(promotion_id)
        if db_rec is None:
            raise ValueError(f"Promotion {promotion_id} not found.")
        if db_rec.status != "ACTIVE":
            raise ValueError(
                f"Promotion {promotion_id} has status={db_rec.status}; "
                "only ACTIVE promotions can be rolled back."
            )

        backup_path = db_rec.backup_path
        ap = AtomicPromotion()
        step = ap.manual_rollback(backup_path or "")
        success = step.success

        rollback_id = uuid.uuid4()
        rb_record = RollbackHistory(
            rollback_id=rollback_id,
            promotion_id=uuid.UUID(promotion_id),
            restored_frequency_champion=db_rec.old_frequency_champion,
            restored_severity_champion=db_rec.old_severity_champion,
            rollback_reason=reason,
            rollback_trigger="manual",
            rollback_status="SUCCESS" if success else "FAILED",
            rolled_back_by=rolled_back_by,
            rolled_back_at=datetime.utcnow(),
            error_message=None if success else step.message,
        )
        self._db.add(rb_record)

        db_rec.status     = "ROLLED_BACK" if success else "FAILED"
        db_rec.updated_at = datetime.utcnow()
        await self._db.commit()

        return RollbackResult(
            rollback_id=str(rollback_id),
            promotion_id=promotion_id,
            rollback_status=rb_record.rollback_status,
            error=None if success else step.message,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _get_candidate(self, candidate_id: str) -> Optional[CandidateModel]:
        try:
            uid = uuid.UUID(candidate_id)
        except (ValueError, AttributeError):
            return None
        result = await self._db.execute(
            select(CandidateModel).where(CandidateModel.model_id == uid)
        )
        return result.scalar_one_or_none()

    async def _get_promotion(self, promotion_id: str) -> Optional[ModelPromotion]:
        try:
            uid = uuid.UUID(promotion_id)
        except ValueError:
            return None
        result = await self._db.execute(
            select(ModelPromotion)
            .options(selectinload(ModelPromotion.rollback_records))
            .where(ModelPromotion.promotion_id == uid)
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _get_champion_scores() -> tuple[float, float]:
        try:
            from src.models.model_registry import ModelRegistry
            reg = ModelRegistry()
            freq_score = float(reg.get_frequency_champion().get("score", 0.0))
            sev_score  = float(reg.get_severity_champion().get("score", 0.0))
            return freq_score, sev_score
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _snapshot_champions() -> tuple[Optional[Dict], Optional[Dict]]:
        try:
            from src.models.model_registry import ModelRegistry
            reg = ModelRegistry()
            return reg.get_frequency_champion(), reg.get_severity_champion()
        except Exception:
            return None, None

    @staticmethod
    def _build_freq_comparison(
        cand_metrics: Dict[str, Any], champion_score: float
    ) -> Dict[str, Any]:
        cand_auc = float(cand_metrics.get("roc_auc", 0.0))
        return {
            "metric":           "ROC-AUC",
            "challenger_value": round(cand_auc, 6),
            "champion_value":   round(champion_score, 6),
            "delta":            round(cand_auc - champion_score, 6),
            "challenger_pr_auc":   cand_metrics.get("pr_auc"),
            "challenger_f1":       cand_metrics.get("f1"),
            "challenger_recall":   cand_metrics.get("recall"),
            "challenger_precision": cand_metrics.get("precision"),
            "challenger_brier":    cand_metrics.get("brier_score"),
        }

    @staticmethod
    def _build_sev_comparison(
        cand_metrics: Dict[str, Any], champion_score: float
    ) -> Dict[str, Any]:
        cand_r2 = float(cand_metrics.get("r2", -9999.0))
        return {
            "metric":           "R²",
            "challenger_value": round(cand_r2, 6),
            "champion_value":   round(champion_score, 6),
            "delta":            round(cand_r2 - champion_score, 6),
            "challenger_rmse":  cand_metrics.get("rmse"),
            "challenger_mae":   cand_metrics.get("mae"),
            "challenger_mape":  cand_metrics.get("mape_pct"),
        }
