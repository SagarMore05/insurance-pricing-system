import json
import logging
from typing import Any

logger = logging.getLogger("analytics.recommendations")

_VALID_CATEGORIES = {"pricing", "risk_management", "claims", "customer_segment", "operations"}
_VALID_PRIORITIES = {"high", "medium", "low"}


async def generate_recommendations(
    groq_client: Any,
    question: str,
    answer: str,
    anomalies: list[dict],
) -> list[dict]:
    """Generate 3-4 actionable recommendations using Groq."""
    anomaly_context = ""
    if anomalies:
        items = ", ".join(
            f"{a['column']}={a['value']:.1f} ({a['direction']} outlier)"
            for a in anomalies[:3]
        )
        anomaly_context = f"\nStatistical anomalies found: {items}"

    prompt = f"""You are an expert insurance analytics consultant. Provide 3-4 specific, actionable recommendations based on this analysis.

User question: {question}
Analysis: {answer[:600]}{anomaly_context}

Respond with ONLY a valid JSON array. Each element must have exactly these keys:
- "category": one of "pricing", "risk_management", "claims", "customer_segment", "operations"
- "text": one or two sentences describing a concrete action
- "priority": one of "high", "medium", "low"

Example: [{{"category":"pricing","text":"Increase premiums by 8% for high-mileage vehicles in metro cities.","priority":"high"}}]

JSON array only, no other text:"""

    try:
        response = await groq_client.ainvoke(prompt)
        content = response.content.strip()

        # Strip markdown fences if present
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    content = part
                    break

        items = json.loads(content)
        result = []
        for item in items[:5]:
            cat = str(item.get("category", "operations"))
            pri = str(item.get("priority", "medium"))
            result.append({
                "category": cat if cat in _VALID_CATEGORIES else "operations",
                "text": str(item.get("text", ""))[:300],
                "priority": pri if pri in _VALID_PRIORITIES else "medium",
            })
        return result
    except Exception as exc:
        logger.warning("Recommendation generation failed: %s", exc)
        return []
