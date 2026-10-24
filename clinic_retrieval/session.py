"""The only thing this system remembers between questions.

A clarification that has been asked and not yet answered. Nothing else: no
question text, no answer history, no learned preferences. The reason is in
ORCHESTRATION.md, and the short version is that a clinic tablet is shared and
anything remembered about one patient is a leak waiting for the next one.

In process and bounded. For a pilot this moves to Redis with the same shape,
which is the point of keeping the shape this small.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

TTL_SECONDS = 600          # ten minutes, roughly one counter conversation
MAX_SESSIONS = 500         # a clinic has a handful of tablets, not thousands


@dataclass(frozen=True)
class Pending:
    """A question the system asked and is waiting on."""
    kind: str                       # only "drug_choice" today
    options: tuple[str, ...]        # what the user was offered, in order
    codes: tuple[str, ...]          # the formulary codes behind the options


@dataclass
class Session:
    id: str
    state: str
    created: float
    touched: float
    turns: int = 0
    pending: Pending | None = None


class Sessions:
    def __init__(self, ttl: float = TTL_SECONDS, limit: int = MAX_SESSIONS):
        self.ttl = ttl
        self.limit = limit
        self._live: dict[str, Session] = {}

    def start(self, state: str, now: float | None = None) -> Session:
        now = now if now is not None else time.monotonic()
        self._evict(now)
        session = Session(id=secrets.token_urlsafe(9), state=state.lower(),
                          created=now, touched=now)
        self._live[session.id] = session
        return session

    def get(self, session_id: str | None, state: str,
            now: float | None = None) -> Session | None:
        """None for an unknown, expired, or wrong-clinic id. A session is not a
        way to reach another clinic's data."""
        now = now if now is not None else time.monotonic()
        self._evict(now)
        session = self._live.get(session_id or "")
        if session is None or session.state != state.lower():
            return None
        session.touched = now
        return session

    def _evict(self, now: float) -> None:
        for key, session in list(self._live.items()):
            if now - session.touched > self.ttl:
                del self._live[key]
        while len(self._live) > self.limit:
            oldest = min(self._live.values(), key=lambda s: s.touched)
            del self._live[oldest.id]

    def forget_unless_pending(self, session: Session | None) -> Session | None:
        """Kept only if there is a question outstanding. A one-off query leaves
        nothing behind, which is both smaller and the privacy answer."""
        if session is None:
            return None
        if session.pending is None:
            self._live.pop(session.id, None)
            return None
        return session

    def __len__(self) -> int:
        return len(self._live)
