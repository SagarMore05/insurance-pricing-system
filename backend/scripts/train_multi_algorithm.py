"""
CLI entry point for Multi-Algorithm Model Selection Engine.

Usage:
    python scripts/train_multi_algorithm.py
    python scripts/train_multi_algorithm.py --dataset data/master/motor_insurance_master_dataset_50000.csv

Trains XGBoost, LightGBM, and CatBoost for both frequency and severity models,
automatically selects the winner, updates registries, and hot-reloads the predictor.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import pathlib

# Add backend root to path so src.* imports work
_BACKEND_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_multi_algorithm")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatic Multi-Algorithm Model Selection Engine",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to training CSV (default: data/master/motor_insurance_master_dataset_50000.csv)",
    )
    parser.add_argument(
        "--shap-samples",
        type=int,
        default=500,
        help="Number of test-set samples for SHAP explanation (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Enterprise Multi-Algorithm Model Selection Engine")
    logger.info("  Dataset : %s", args.dataset or "data/master/motor_insurance_master_dataset_50000.csv")
    logger.info("  Seed    : %d", args.seed)
    logger.info("  SHAP N  : %d", args.shap_samples)
    logger.info("=" * 60)

    from src.training.multi_algorithm_engine import MultiAlgorithmEngine

    engine = MultiAlgorithmEngine(
        dataset_path=args.dataset,
        random_state=args.seed,
        shap_sample_size=args.shap_samples,
    )

    try:
        record = engine.run()
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("  Run ID          : %s", record["run_id"])
    logger.info("  Run number      : %d", record["run_number"])
    logger.info("  Frequency winner: %s", record["frequency_winner"].upper())
    logger.info("  Severity winner : %s", record["severity_winner"].upper())
    logger.info("  Elapsed         : %.1fs", record["total_elapsed_sec"])
    logger.info("")

    # Print algorithm comparison tables
    logger.info("  Frequency Algorithm Comparison:")
    logger.info("  %-12s %-10s %-10s %-10s %-10s %-10s %s", "Algorithm", "ROC-AUC", "F1", "Precision", "Recall", "Brier", "Winner")
    for r in record["frequency_results"]:
        m   = r["metrics"]
        win = "*** WINNER ***" if r["is_winner"] else ""
        logger.info(
            "  %-12s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %s",
            r["algorithm"],
            m.get("roc_auc", 0) or 0,
            m.get("f1", 0) or 0,
            m.get("precision", 0) or 0,
            m.get("recall", 0) or 0,
            m.get("brier_score", 0) or 0,
            win,
        )

    logger.info("")
    logger.info("  Severity Algorithm Comparison:")
    logger.info("  %-12s %-10s %-10s %-10s %-10s %s", "Algorithm", "R²", "RMSE", "MAE", "MAPE%", "Winner")
    for r in record["severity_results"]:
        m   = r["metrics"]
        win = "*** WINNER ***" if r["is_winner"] else ""
        logger.info(
            "  %-12s %-10.4f %-10.0f %-10.0f %-10.2f %s",
            r["algorithm"],
            m.get("r2", 0) or 0,
            m.get("rmse", 0) or 0,
            m.get("mae", 0) or 0,
            m.get("mape_pct", 0) or 0,
            win,
        )

    logger.info("")
    logger.info("  Promotion reasons:")
    logger.info("  Frequency : %s", record["promotion_reason_frequency"])
    logger.info("  Severity  : %s", record["promotion_reason_severity"])
    logger.info("")
    logger.info("  Production artifacts updated:")
    logger.info("  Frequency : %s", record["frequency_production_path"])
    logger.info("  Severity  : %s", record["severity_production_path"])
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
