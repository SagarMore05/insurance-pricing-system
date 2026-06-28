import logging
from typing import Any

import numpy as np

logger = logging.getLogger("analytics.anomaly")

_MIN_ROWS = 4       # IQR is meaningless with fewer rows
_MAX_RESULTS = 5    # return at most the top N anomalies


def _try_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def detect_anomalies(data: list[dict]) -> list[dict]:
    """
    IQR-based outlier detection on all numeric columns in `data`.

    Returns up to _MAX_RESULTS anomaly dicts sorted by deviation_score descending.
    """
    if not data or len(data) < _MIN_ROWS:
        return []

    # Build column → float-values mapping
    columns: dict[str, list[float]] = {}
    for row in data:
        for col, val in row.items():
            f = _try_float(val)
            if f is not None:
                columns.setdefault(col, []).append(f)

    anomalies: list[dict] = []

    for col, values in columns.items():
        if len(values) < _MIN_ROWS:
            continue
        arr = np.array(values, dtype=float)
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        if iqr < 1e-9:
            continue  # constant column — skip
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        median = float(np.median(arr))

        for row in data:
            v = _try_float(row.get(col))
            if v is None:
                continue
            if v < lower or v > upper:
                direction = "high" if v > upper else "low"
                deviation_score = round(abs(v - median) / (iqr + 1e-9), 2)
                anomalies.append({
                    "column": col,
                    "value": v,
                    "row_data": {k: str(val) for k, val in row.items()},
                    "direction": direction,
                    "deviation_score": deviation_score,
                })

    anomalies.sort(key=lambda x: x["deviation_score"], reverse=True)
    return anomalies[:_MAX_RESULTS]
