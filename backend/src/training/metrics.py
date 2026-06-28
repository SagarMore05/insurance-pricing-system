"""
Metric collection for Phase 5A V5 Training Engine.

Frequency metrics: ROC-AUC, PR-AUC, Precision, Recall, F1, Brier Score
Severity  metrics: RMSE, MAE, R², MAPE
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_frequency_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate a frequency (binary claim probability) model.

    Parameters
    ----------
    y_true  : 1-D int array of ground truth labels (0/1)
    y_prob  : 1-D float array of predicted claim probabilities
    threshold : classification threshold for precision/recall/f1
    """
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc  = float(average_precision_score(y_true, y_prob))

    prec   = float(precision_score(y_true, y_pred, zero_division=0))
    rec    = float(recall_score(y_true, y_pred, zero_division=0))
    f1     = float(f1_score(y_true, y_pred, zero_division=0))
    brier  = float(brier_score_loss(y_true, y_prob))

    return {
        "roc_auc":    round(roc_auc, 4),
        "pr_auc":     round(pr_auc, 4),
        "precision":  round(prec, 4),
        "recall":     round(rec, 4),
        "f1":         round(f1, 4),
        "brier_score": round(brier, 4),
        "threshold":  threshold,
        "n_samples":  int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 4),
    }


def compute_severity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate a severity (claim amount regression) model.

    Parameters
    ----------
    y_true : 1-D float array of actual claim amounts
    y_pred : 1-D float array of predicted claim amounts
    """
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    # MAPE — guard against zero actuals
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = None

    return {
        "rmse":     round(rmse, 2),
        "mae":      round(mae, 2),
        "r2":       round(r2, 4),
        "mape_pct": round(mape, 2) if mape is not None else None,
        "n_samples": int(len(y_true)),
        "mean_actual":    round(float(y_true.mean()), 2),
        "mean_predicted": round(float(y_pred.mean()), 2),
    }
