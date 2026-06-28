"""AsyncPostgresSaver singleton for LangGraph HITL checkpointing."""
from __future__ import annotations

import logging

logger = logging.getLogger("underwriting.checkpointer")

_checkpointer = None


async def setup_checkpointer():
    """
    Create and setup the LangGraph checkpointer.

    Tries AsyncPostgresSaver (persistent, survives restarts).
    Falls back to MemorySaver (in-process only) if Postgres setup fails so that
    the rest of the app still starts.
    """
    global _checkpointer
    from src.config import settings

    # AsyncPostgresSaver expects plain postgresql:// — strip the +asyncpg dialect
    conn_string = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        cp = AsyncPostgresSaver.from_conn_string(conn_string)
        await cp.setup()   # creates langgraph checkpoint tables if they don't exist
        _checkpointer = cp
        logger.info("LangGraph AsyncPostgresSaver initialised")
        return cp
    except Exception as exc:
        logger.warning(
            "AsyncPostgresSaver setup failed (%s) — "
            "falling back to MemorySaver (HITL state is non-persistent across restarts)",
            exc,
        )
        from langgraph.checkpoint.memory import MemorySaver

        _checkpointer = MemorySaver()
        return _checkpointer


def get_checkpointer():
    """Return the module-level checkpointer singleton (raises if not yet initialised)."""
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer not initialised — call setup_checkpointer() during app lifespan"
        )
    return _checkpointer
