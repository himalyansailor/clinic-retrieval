"""Who may ask, and about which clinic.

One control, applied in one place: an API key maps to the states it may query.
That is deliberately the same boundary the retrieval filter already enforces,
so a mistake in either one is caught by the other. Defence in depth, not
duplication.

Keys come from the environment, never from a file in the repo:

    CLINIC_API_KEYS="secret-one:br-cl-114:bihar,secret-two:jh-cl-207:jharkhand"

With none set the API runs open and says so at /health. That is fine for a
laptop and not fine for a clinic, which is why it is reported rather than
silent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

HEADER = "x-api-key"
ANONYMOUS = "anonymous"


@dataclass(frozen=True)
class Caller:
    id: str                         # a label, never the key itself
    states: tuple[str, ...]         # empty means every configured clinic

    def may_query(self, state: str) -> bool:
        return not self.states or state.lower() in self.states


def load_keys(raw: str | None = None) -> dict[str, Caller]:
    raw = raw if raw is not None else os.environ.get("CLINIC_API_KEYS", "")
    keys: dict[str, Caller] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        secret, _, rest = entry.partition(":")
        label, _, states = rest.partition(":")
        if not secret:
            continue
        keys[secret] = Caller(
            id=label or "unlabelled",
            states=tuple(s.strip().lower() for s in states.split("|") if s.strip()),
        )
    return keys


class Gate:
    def __init__(self, keys: dict[str, Caller] | None = None):
        self.keys = keys if keys is not None else load_keys()

    @property
    def open(self) -> bool:
        return not self.keys

    def caller(self, presented: str | None) -> Caller | None:
        """None means the key was required and wrong. An open gate gives an
        anonymous caller scoped to nothing in particular."""
        if self.open:
            return Caller(ANONYMOUS, ())
        return self.keys.get(presented or "")
