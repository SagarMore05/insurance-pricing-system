from __future__ import annotations

from operator import add
from typing import Annotated, Optional, TypedDict


class UnderwritingState(TypedDict):
    """Full mutable state threaded through the LangGraph underwriting pipeline."""

    # ── Inputs (set by route handler before graph invocation) ──────────────────
    request_data: dict          # raw CustomerQuoteRequest.model_dump()
    session_id: str             # prediction_id used as session identifier

    # ── ML + pricing results (set by route handler) ────────────────────────────
    ml_result: dict             # {claim_probability, expected_claim_amount_inr, expected_loss_inr}
    pricing_result: dict        # PricingResult.to_dict()
    prediction_id: str

    # ── Pre-fetched segment context (set by route handler) ─────────────────────
    customer_history: dict      # ProfileRiskStats-shaped dict from tools.async_fetch_profile_stats

    # ── Node outputs (set by graph nodes, last-write-wins) ─────────────────────
    fraud_signals: list         # list of signal dicts from evaluate_fraud_signals
    reasoning_trace: str        # LLM-generated underwriting assessment text
    underwriting_decision: str  # "approve" | "flag" | "refer"
    confidence_score: float     # 0.0–1.0
    risk_adjustment_factor: float  # advisory multiplier (1.0 = no change)

    # ── Execution log (Annotated reducer: each node appends, never overwrites) ──
    node_log: Annotated[list, add]

    # ── Human review fields (Phase 3B HITL) ───────────────────────────────────
    human_decision: Optional[str]   # "approved" | "rejected" — injected by resume endpoint
    review_note: Optional[str]      # optional text from human reviewer
