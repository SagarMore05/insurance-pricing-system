import json
import logging
from datetime import datetime
from typing import Optional

import redis as redis_lib

logger = logging.getLogger("analytics.memory")

MAX_TURNS = 10  # max conversation turns to keep per session


class SessionMemory:
    """Redis-backed conversation memory for the Analytics Agent."""

    def __init__(self, client: redis_lib.Redis, ttl: int = 3600):
        self._r = client
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"agent:analytics:{session_id}"

    def get_history(self, session_id: str) -> list[dict]:
        raw = self._r.get(self._key(session_id))
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:
            return []

    def append(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        history = self.get_history(session_id)
        history.append({
            "role": role,
            "content": content[:800],  # truncate very long responses
            "timestamp": datetime.utcnow().isoformat(),
        })
        history = history[-(MAX_TURNS * 2):]  # rolling window: N user + N assistant
        self._r.setex(key, self._ttl, json.dumps(history))

    def clear(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))


_memory: Optional[SessionMemory] = None


def get_session_memory(redis_url: str) -> Optional[SessionMemory]:
    global _memory
    if _memory is not None:
        return _memory
    try:
        client = redis_lib.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        _memory = SessionMemory(client)
        logger.info("Redis session memory connected at %s", redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — agent runs in stateless mode", exc)
        _memory = None
    return _memory
