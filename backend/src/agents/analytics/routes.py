import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import verify_admin_key
from src.api.schemas import AgentAnalyticsRequest, AgentAnalyticsResponse
from src.database.session import get_db
from src.database.models import AgentRun
from src.config import settings

logger = logging.getLogger("analytics.routes")

router = APIRouter(dependencies=[Depends(verify_admin_key)])


@router.post("/agent/analytics/query", response_model=AgentAnalyticsResponse)
async def analytics_query(
    request: AgentAnalyticsRequest,
    db: AsyncSession = Depends(get_db),
):
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    from src.agents.analytics.agent import get_analytics_agent

    session_id = request.session_id or str(uuid.uuid4())
    agent = get_analytics_agent()

    run = AgentRun(
        run_id=uuid.uuid4(),
        session_id=session_id,
        agent_type="analytics",
        input_text=request.question,
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(run)
    await db.commit()

    try:
        result = await agent.run(session_id, request.question)

        run.status = "completed"
        run.intent = result.intent
        run.output_text = result.answer[:2000]
        run.completed_at = datetime.utcnow()
        await db.commit()

        return result

    except Exception as exc:
        run.status = "failed"
        run.output_text = str(exc)[:500]
        run.completed_at = datetime.utcnow()
        await db.commit()
        logger.error("Analytics agent failed for session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Agent query failed. Check server logs.")


@router.delete("/agent/analytics/session/{session_id}", status_code=204)
async def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    from src.agents.analytics.memory import get_session_memory
    memory = get_session_memory(settings.REDIS_URL)
    if memory:
        memory.clear(session_id)
