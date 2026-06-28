import uuid
import logging
from typing import Optional

from src.api.schemas import AgentAnalyticsResponse, AnomalyItem, RecommendationItem
from src.config import settings
from src.agents.analytics.memory import get_session_memory
from src.agents.analytics.intent import classify_intent
from src.agents.analytics.anomaly import detect_anomalies
from src.agents.analytics.recommendations import generate_recommendations

logger = logging.getLogger("analytics.agent")


class AnalyticsAgent:
    def __init__(self):
        self._groq = None

    def _get_groq(self):
        if self._groq is None:
            from langchain_groq import ChatGroq
            self._groq = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=settings.GROQ_API_KEY,
                temperature=0.1,
                max_tokens=512,
            )
        return self._groq

    async def run(self, session_id: str, question: str) -> AgentAnalyticsResponse:
        from src.nlp_assistant.assistant import get_assistant
        assistant = get_assistant()

        memory = get_session_memory(settings.REDIS_URL)
        history = memory.get_history(session_id) if memory else []
        intent = classify_intent(question)

        # Inject recent context into the question so the SQL agent is aware of the conversation
        if history:
            recent = history[-4:]  # last 2 turns
            ctx = " | ".join(f"{t['role']}: {t['content'][:120]}" for t in recent)
            contextualized = f"[Prior context: {ctx}] Current question: {question}"
        else:
            contextualized = question

        # Delegate to the existing SQL assistant
        try:
            sql_result = await assistant.query(contextualized)
        except Exception as exc:
            logger.error("SQL assistant failed: %s", exc)
            sql_result = {
                "answer": f"I encountered an error processing your query: {exc}",
                "sql_used": None,
                "data": None,
                "chart_suggestion": "none",
            }

        raw_data: Optional[list[dict]] = sql_result.get("data")

        # Anomaly detection — run for anomaly/comparison/general intents when data present
        anomalies: list[dict] = []
        if raw_data and intent in ("anomaly", "comparison", "general"):
            anomalies = detect_anomalies(raw_data)

        # Recommendation generation — run for recommendation intent or when anomalies exist
        recommendations: list[dict] = []
        if intent == "recommendation" or (anomalies and intent in ("anomaly", "general")):
            if settings.GROQ_API_KEY:
                recommendations = await generate_recommendations(
                    self._get_groq(),
                    question,
                    sql_result["answer"],
                    anomalies,
                )

        # Persist turn to session memory
        if memory:
            memory.append(session_id, "user", question)
            memory.append(session_id, "assistant", sql_result["answer"])

        return AgentAnalyticsResponse(
            answer=sql_result["answer"],
            sql_used=sql_result.get("sql_used"),
            data=raw_data,
            chart_suggestion=sql_result.get("chart_suggestion", "none"),
            session_id=session_id,
            intent=intent,
            anomalies=[AnomalyItem(**a) for a in anomalies] if anomalies else None,
            recommendations=[RecommendationItem(**r) for r in recommendations] if recommendations else None,
        )


_agent: Optional[AnalyticsAgent] = None


def get_analytics_agent() -> AnalyticsAgent:
    global _agent
    if _agent is None:
        _agent = AnalyticsAgent()
    return _agent
