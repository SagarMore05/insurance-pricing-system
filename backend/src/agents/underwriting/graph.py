"""
Underwriting Agent LangGraph graph definition — Phase 3B (HITL enabled).

Topology:

  START → fraud_detector → reasoning → decision
                                           │
                        ┌──────────────────┴──────────────────┐
                        │ needs_human_review()                 │
                        ▼                                      ▼
              human_review ← interrupt_before             END (auto-approve)
                        │
                        ▼
                       END

The graph is compiled with interrupt_before=["human_review"] and a persistent
checkpointer so it can pause mid-run and be resumed by the review endpoint.

setup_underwriting_graph() is called once during app lifespan (after
setup_checkpointer() has run). If called before that, get_underwriting_graph()
falls back to a checkpointer-free build so tests and dev scripts still work.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from src.agents.underwriting.nodes import (
    decision_node,
    fraud_signal_node,
    human_review_node,
    needs_human_review,
    reasoning_node,
)
from src.agents.underwriting.state import UnderwritingState

logger = logging.getLogger("underwriting.graph")

_graph = None


def _build(checkpointer=None):
    """Construct and compile the underwriting StateGraph."""
    builder = StateGraph(UnderwritingState)

    builder.add_node("fraud_detector", fraud_signal_node)
    builder.add_node("reasoning",      reasoning_node)
    builder.add_node("decision",       decision_node)
    builder.add_node("human_review",   human_review_node)

    builder.add_edge(START,            "fraud_detector")
    builder.add_edge("fraud_detector", "reasoning")
    builder.add_edge("reasoning",      "decision")
    builder.add_conditional_edges(
        "decision",
        needs_human_review,
        {"human_review": "human_review", "end": END},
    )
    builder.add_edge("human_review", END)

    compiled = builder.compile(
        interrupt_before=["human_review"],
        checkpointer=checkpointer,
    )
    logger.info(
        "Underwriting graph compiled (checkpointer=%s)",
        type(checkpointer).__name__ if checkpointer else "None — HITL disabled",
    )
    return compiled


async def setup_underwriting_graph():
    """Build the graph with the persistent checkpointer. Called during app lifespan."""
    global _graph
    from src.agents.underwriting.checkpointer import get_checkpointer
    _graph = _build(checkpointer=get_checkpointer())
    return _graph


def get_underwriting_graph():
    """Return the compiled graph singleton, building without a checkpointer if needed."""
    global _graph
    if _graph is None:
        # lifespan hasn't run (e.g. pytest, dev scripts) — HITL won't work but graph executes
        _graph = _build(checkpointer=None)
    return _graph
