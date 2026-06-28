from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import AssistantQueryRequest, AssistantQueryResponse
from src.api.dependencies import verify_admin_key
from src.nlp_assistant.assistant import get_assistant
from src.config import settings

router = APIRouter(dependencies=[Depends(verify_admin_key)])


@router.post("/assistant/query", response_model=AssistantQueryResponse)
async def assistant_query(request: AssistantQueryRequest):
    if not settings.GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")

    assistant = get_assistant()
    result = await assistant.query(request.question)

    return AssistantQueryResponse(
        answer=result["answer"],
        sql_used=result.get("sql_used"),
        data=result.get("data"),
        chart_suggestion=result.get("chart_suggestion", "none"),
    )
