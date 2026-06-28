"""V5 Groundwork — Dataset extraction and readiness endpoints.

These are READ-ONLY analytics + export endpoints.  No model training.
No champion changes.  No promotion logic.

Endpoints:
    GET  /admin/v5/dataset-status   — JSON readiness report
    POST /admin/v5/export-dataset   — Build dataset and save CSV to disk;
                                      returns metadata (row count, filename, readiness)
"""
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import verify_admin_key
from src.api.schemas import V5DatasetStatus, V5ExportResponse
from src.database.session import get_db
from src.retraining.dataset_builder_v4 import V4DatasetBuilder

logger = logging.getLogger("v5_dataset")

router = APIRouter(dependencies=[Depends(verify_admin_key)])

# CSV exports are written here so admins can retrieve them from the server
_EXPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "experiments" / "outputs" / "data"


@router.get("/admin/v5/dataset-status", response_model=V5DatasetStatus)
async def get_v5_dataset_status(db: AsyncSession = Depends(get_db)):
    """
    Returns the current V4 dataset readiness report.

    Reports:
    - How many V2 predictions exist (inference_features_json stored)
    - How many have labels (model_feedback rows)
    - How many policies have matured (>= 90 days)
    - Per-feature fill rates across labelled matured rows
    - Whether the dataset meets the 500-row retraining threshold
    """
    builder = V4DatasetBuilder(db)
    result = await builder.build(include_dataset=False)
    return V5DatasetStatus(
        generated_at=result.extraction_ts,
        total_predictions=result.total_predictions,
        labelled_predictions=result.labelled_predictions,
        matured_predictions=result.matured_predictions,
        matured_labelled=result.matured_labelled,
        rows_valid=result.rows_valid,
        rows_dropped_missing_json=result.rows_dropped_missing_json,
        rows_dropped_validation=result.rows_dropped_validation,
        feature_fill_rates=result.feature_fill_rates,
        missing_features=result.missing_features,
        is_ready_for_retraining=result.is_ready_for_retraining,
        readiness_reason=result.readiness_reason,
    )


@router.post("/admin/v5/export-dataset", response_model=V5ExportResponse)
async def export_v5_dataset(db: AsyncSession = Depends(get_db)):
    """
    Build the training-ready dataset and save it as a CSV file on the server.

    Only matured (>= 90 days) labelled rows with all required fields present
    are included.  Returns metadata: row count, filename, and readiness status.

    The CSV is written to:
        backend/experiments/outputs/data/v4_dataset_export_<timestamp>.csv

    Note: returns status='empty' when no valid rows exist (Day 0 of production).
    """
    builder = V4DatasetBuilder(db)
    result = await builder.build(include_dataset=True)

    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"v4_dataset_export_{ts_str}.csv"

    rows_exported = 0
    if result.dataset is not None and not result.dataset.empty:
        # Drop internal tracking columns; keep features + labels only
        drop_cols = {"prediction_id"}
        export_cols = [c for c in result.dataset.columns if c not in drop_cols]
        try:
            _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            export_path = _EXPORT_DIR / filename
            result.dataset[export_cols].to_csv(export_path, index=False)
            rows_exported = len(result.dataset)
            logger.info("Dataset exported: %s (%d rows)", export_path, rows_exported)
        except OSError as exc:
            logger.warning("Could not write export CSV: %s", exc)

    return V5ExportResponse(
        status="ok" if rows_exported > 0 else "empty",
        rows_exported=rows_exported,
        filename=filename,
        generated_at=datetime.utcnow().isoformat(),
        is_ready_for_retraining=result.is_ready_for_retraining,
        readiness_reason=result.readiness_reason,
    )
