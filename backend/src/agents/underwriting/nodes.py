"""
LangGraph nodes for the Underwriting Agent.

Execution order (Phase 3B with HITL):
  fraud_signal_node → reasoning_node → decision_node
        → [needs_human_review] → human_review_node (interrupt_before) → END
                               → END  (if no review triggers fire)

Each node receives the full UnderwritingState, returns a partial state dict.
The node_log list uses an Annotated reducer (operator.add) — each node appends
one entry without overwriting previous nodes' entries.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from src.agents.underwriting.state import UnderwritingState
from src.agents.underwriting.tools import evaluate_fraud_signals

logger = logging.getLogger("underwriting.nodes")


# ─── Node 1: Fraud signal detection ───────────────────────────────────────────

async def fraud_signal_node(state: UnderwritingState) -> dict:
    t0 = time.monotonic()

    signals = evaluate_fraud_signals(
        state["request_data"],
        state["customer_history"],
    )

    duration_ms = round((time.monotonic() - t0) * 1000)
    high_count   = sum(1 for s in signals if s["severity"] == "high")
    medium_count = sum(1 for s in signals if s["severity"] == "medium")

    logger.info(
        "fraud_signal_node: %d signals (%d high, %d medium) in %dms",
        len(signals), high_count, medium_count, duration_ms,
    )

    return {
        "fraud_signals": signals,
        "node_log": [{
            "node": "fraud_signals",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "input_summary": (
                f"age={state['request_data'].get('age')}, "
                f"driving_score={state['request_data'].get('driving_score')}, "
                f"prev_claims={state['request_data'].get('previous_claims')}, "
                f"mileage={state['request_data'].get('annual_mileage_km')}"
            ),
            "output_summary": (
                f"{len(signals)} signals detected "
                f"({high_count} high, {medium_count} medium)"
            ),
        }],
    }


# ─── Node 2: LLM reasoning trace ──────────────────────────────────────────────

async def reasoning_node(state: UnderwritingState) -> dict:
    t0 = time.monotonic()

    trace = await _generate_reasoning(state)

    duration_ms = round((time.monotonic() - t0) * 1000)
    logger.info("reasoning_node: %d chars in %dms", len(trace), duration_ms)

    return {
        "reasoning_trace": trace,
        "node_log": [{
            "node": "reasoning",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "input_summary": (
                f"risk_level={state['pricing_result'].get('risk_level')}, "
                f"signals={len(state.get('fraud_signals', []))}"
            ),
            "output_summary": f"reasoning_trace generated ({len(trace)} chars)",
        }],
    }


async def _generate_reasoning(state: UnderwritingState) -> str:
    """Call Groq to produce the underwriting narrative.  Falls back to a
    rule-based summary if GROQ_API_KEY is absent or the call fails."""
    from src.config import settings

    req     = state["request_data"]
    ml      = state["ml_result"]
    pricing = state["pricing_result"]
    history = state["customer_history"]
    signals = state.get("fraud_signals", [])

    # Build signal summary for the prompt
    if signals:
        signal_lines = "\n".join(
            f"  [{s['severity'].upper()}] {s['signal']}: {s['description']}"
            for s in signals
        )
    else:
        signal_lines = "  None detected."

    prompt = f"""You are a senior insurance underwriter. Write a professional 3-4 sentence underwriting assessment.

APPLICANT:
- Age {req.get('age')}, {req.get('gender')}, {req.get('city')}
- {req.get('vehicle_brand', '').title()} {req.get('vehicle_segment', '')} ({req.get('fuel_type', '')}), {req.get('vehicle_age_years')}yr old, valued ₹{float(req.get('vehicle_value_inr', 0)):,.0f}
- Driving score {req.get('driving_score')}/100  |  {int(req.get('annual_mileage_km', 0)):,} km/yr  |  {req.get('previous_claims')} prior claims

ML RISK ASSESSMENT:
- Claim probability: {float(ml.get('claim_probability', 0)):.1%}
- Expected claim: ₹{float(ml.get('expected_claim_amount_inr', 0)):,.0f}
- Risk tier: {pricing.get('risk_level', 'unknown').upper()}
- Calculated premium: ₹{float(pricing.get('final_premium', 0)):,.0f}

SEGMENT CONTEXT ({req.get('city', 'this city')}):
- {history.get('similar_policies_count', 0)} comparable policies on books
- Segment avg premium: ₹{float(history.get('avg_premium_in_segment', 0)):,.0f}
- Segment claim frequency: {float(history.get('segment_claim_frequency', 0)):.1%}

RISK SIGNALS DETECTED:
{signal_lines}

Write a concise professional underwriting narrative covering: primary risk drivers, how this applicant compares to the segment, any concerns or positive factors, and a brief rationale. Be specific."""

    if not settings.GROQ_API_KEY:
        return _fallback_reasoning(req, ml, pricing, signals)

    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.GROQ_API_KEY,
            temperature=0.15,
            max_tokens=350,
        )
        response = await llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as exc:
        logger.warning("Groq call failed in reasoning_node: %s — using fallback", exc)
        return _fallback_reasoning(req, ml, pricing, signals)


def _fallback_reasoning(req: dict, ml: dict, pricing: dict, signals: list) -> str:
    """Deterministic fallback when Groq is unavailable."""
    risk = pricing.get("risk_level", "unknown")
    prob = float(ml.get("claim_probability", 0))
    premium = float(pricing.get("final_premium", 0))
    high_signals = [s for s in signals if s["severity"] == "high"]
    med_signals  = [s for s in signals if s["severity"] == "medium"]

    parts = [
        f"This {req.get('age')}-year-old applicant from {req.get('city')} presents a "
        f"{risk}-risk profile with a {prob:.1%} modelled claim probability and "
        f"a calculated annual premium of ₹{premium:,.0f}."
    ]
    if high_signals:
        parts.append(
            f"The application has {len(high_signals)} high-severity signal(s) — "
            f"{', '.join(s['signal'].replace('_', ' ') for s in high_signals)} — "
            "which require manual underwriter review before binding."
        )
    elif med_signals:
        parts.append(
            f"{len(med_signals)} medium-severity signal(s) were detected "
            f"({', '.join(s['signal'].replace('_', ' ') for s in med_signals)}) "
            "and should be noted in the policy file."
        )
    else:
        parts.append(
            "No significant risk signals were detected; "
            "the application aligns with standard underwriting criteria."
        )
    parts.append(
        "Premium has been set using the actuarial model output "
        "with standard business-rule adjustments applied."
    )
    return " ".join(parts)


# ─── Node 3: Underwriting decision ────────────────────────────────────────────

def decision_node(state: UnderwritingState) -> dict:
    t0 = time.monotonic()

    signals    = state.get("fraud_signals", [])
    high_count = sum(1 for s in signals if s["severity"] == "high")
    med_count  = sum(1 for s in signals if s["severity"] == "medium")

    # Decision matrix
    if high_count >= 2 or (high_count == 1 and med_count >= 2):
        decision   = "refer"
        confidence = 0.90
        adjustment = 1.25   # advisory: suggest 25% uplift if manually approved
    elif high_count == 1 or med_count >= 2:
        decision   = "flag"
        confidence = 0.78
        adjustment = 1.15   # advisory: suggest 15% uplift
    else:
        decision   = "approve"
        confidence = 0.95
        adjustment = 1.0

    duration_ms = round((time.monotonic() - t0) * 1000)
    logger.info(
        "decision_node: %s (confidence=%.2f, adjustment=%.2f) in %dms",
        decision, confidence, adjustment, duration_ms,
    )

    return {
        "underwriting_decision": decision,
        "confidence_score": confidence,
        "risk_adjustment_factor": adjustment,
        "node_log": [{
            "node": "decision",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "input_summary": (
                f"fraud_signals={len(signals)} "
                f"(high={high_count}, medium={med_count})"
            ),
            "output_summary": (
                f"decision={decision}, confidence={confidence:.2f}, "
                f"adjustment={adjustment}x"
            ),
        }],
    }


# ─── HITL: Conditional routing ────────────────────────────────────────────────

def _get_review_triggers(state: UnderwritingState) -> list:
    """Return list of trigger names that require human review."""
    triggers = []
    if state.get("confidence_score", 1.0) < 0.85:
        triggers.append("low_confidence")
    if state.get("fraud_signals"):
        triggers.append("fraud_signals_present")
    if float((state.get("ml_result") or {}).get("claim_probability", 0)) > 0.40:
        triggers.append("high_claim_probability")
    return triggers


def needs_human_review(state: UnderwritingState) -> str:
    """
    Conditional edge after decision_node.
    Returns 'human_review' when any trigger fires, 'end' otherwise.
    """
    return "human_review" if _get_review_triggers(state) else "end"


# ─── Node 4: Human review (HITL) ──────────────────────────────────────────────

async def human_review_node(state: UnderwritingState) -> dict:
    """
    Runs only after a human approves via POST /admin/review/{run_id}/approve.

    graph.py compiles with interrupt_before=["human_review"] so this node
    never executes automatically — the graph pauses here until the review
    endpoint injects human_decision via graph.aupdate_state() and then
    calls graph.ainvoke(None, config=config) to resume.
    """
    human_decision = state.get("human_decision", "approved")
    review_note = state.get("review_note") or ""

    logger.info("human_review_node: decision=%s", human_decision)

    return {
        "node_log": [{
            "node": "human_review",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": 0,
            "input_summary": f"human_decision={human_decision}",
            "output_summary": (
                f"Human review: {human_decision}"
                + (f" — {review_note[:100]}" if review_note else "")
            ),
        }],
    }
