import json
import logging
import threading
from datetime import datetime
from typing import Union

logger = logging.getLogger("analytics.memory")

MAX_TURNS = 10  # max conversation turns kept per session (N user + N assistant)

# ── Optional Redis import ─────────────────────────────────────────────────────
# Redis is NOT a required dependency.  If the package is not installed, or if
# the server is unreachable, the agent falls back to in-process session storage
# automatically — no configuration change required.

try:
    import redis as _redis_lib
    _REDIS_IMPORTABLE = True
except ImportError:
    _redis_lib = None          # type: ignore[assignment]
    _REDIS_IMPORTABLE = False


# ── Redis-backed implementation ───────────────────────────────────────────────

class _RedisSessionMemory:
    """Session memory persisted in Redis (survives restarts, shared across workers)."""

    def __init__(self, client, ttl: int = 3600) -> None:
        self._r = client
        self._ttl = ttl

    @staticmethod
    def _key(session_id: str) -> str:
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
            "content": content[:800],
            "timestamp": datetime.utcnow().isoformat(),
        })
        history = history[-(MAX_TURNS * 2):]
        self._r.setex(key, self._ttl, json.dumps(history))

    def clear(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))


# ── In-memory fallback ────────────────────────────────────────────────────────

class _InMemorySessionMemory:
    """Thread-safe in-process session memory used when Redis is unavailable.

    Memory is scoped to the current process lifetime.  TTL is accepted for
    interface compatibility but is not enforced (sessions are small and bounded
    by MAX_TURNS so unbounded growth is not a concern for this workload).
    """

    def __init__(self, ttl: int = 3600) -> None:
        self._store: dict[str, list[dict]] = {}
        self._ttl = ttl          # stored for interface parity; not actively enforced
        self._lock = threading.Lock()

    @staticmethod
    def _key(session_id: str) -> str:
        return f"agent:analytics:{session_id}"

    def get_history(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._store.get(self._key(session_id), []))

    def append(self, session_id: str, role: str, content: str) -> None:
        key = self._key(session_id)
        with self._lock:
            history = list(self._store.get(key, []))
            history.append({
                "role": role,
                "content": content[:800],
                "timestamp": datetime.utcnow().isoformat(),
            })
            self._store[key] = history[-(MAX_TURNS * 2):]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(self._key(session_id), None)


# Public type alias (agent.py only imports get_session_memory, but keep the
# name available in case other modules reference it via type annotations).
SessionMemory = Union[_RedisSessionMemory, _InMemorySessionMemory]

# ── Factory ───────────────────────────────────────────────────────────────────

_memory: Union[_RedisSessionMemory, _InMemorySessionMemory, None] = None


def get_session_memory(
    redis_url: str,
) -> Union[_RedisSessionMemory, _InMemorySessionMemory]:
    """Return the active session-memory backend.

    Priority:
      1. Redis — if the package is installed *and* the server responds to PING.
      2. In-memory — automatic fallback in all other cases.

    The returned object is always usable; callers do not need to guard for None.
    """
    global _memory
    if _memory is not None:
        return _memory

    if _REDIS_IMPORTABLE:
        try:
            client = _redis_lib.Redis.from_url(   # type: ignore[union-attr]
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            client.ping()
            _memory = _RedisSessionMemory(client)
            logger.info("Analytics agent: Redis session memory connected at %s", redis_url)
            return _memory
        except Exception as exc:
            logger.warning(
                "Analytics agent: Redis unavailable (%s) — falling back to in-memory session store",
                exc,
            )
    else:
        logger.info(
            "Analytics agent: redis package not installed — using in-memory session store"
        )

    _memory = _InMemorySessionMemory()
    return _memory
