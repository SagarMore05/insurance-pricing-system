import re

# No trailing \b — allow suffix matches: anomaly/anomalies, recommend/recommendations, etc.
_ANOMALY = re.compile(
    r"\b(anomal|outlier|unusual|suspicious|abnormal|spike|deviation|irregular|extreme|weird|fraud)",
    re.I,
)
_RECOMMENDATION = re.compile(
    r"\b(recommend|suggest|advic|advis|should we|what to do|action|improv|strateg|next step|priorit|how can|guidance)",
    re.I,
)
_COMPARISON = re.compile(
    r"\b(compar|vs\.?|versus|difference|contrast|better|worse|relative|rank|top|bottom)",
    re.I,
)


def classify_intent(question: str) -> str:
    """Return one of: anomaly | recommendation | comparison | general."""
    if _ANOMALY.search(question):
        return "anomaly"
    if _RECOMMENDATION.search(question):
        return "recommendation"
    if _COMPARISON.search(question):
        return "comparison"
    return "general"
