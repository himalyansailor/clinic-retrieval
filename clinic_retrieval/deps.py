"""Process-wide objects. The index is built once at import, not per request."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Header, HTTPException

from clinic_retrieval import security
from clinic_retrieval.agents.drugs import DrugResolver
from clinic_retrieval.agents.orchestrate import Orchestrator
from clinic_retrieval.analytics import Log
from clinic_retrieval.llm import Llm
from clinic_retrieval.retrieval.index import Hybrid
from clinic_retrieval.session import Sessions

STATES = ("bihar", "jharkhand")

GATE = security.Gate()
SESSIONS = Sessions()

LOG = Log()
RESOLVER = DrugResolver()
LLM = Llm()


@lru_cache(maxsize=1)
def get_index() -> Hybrid:
    return Hybrid.from_corpus()


@lru_cache(maxsize=8)
def get_orchestrator(state: str) -> Orchestrator:
    """One per state, because the stock table is per clinic."""
    return Orchestrator(get_index(), state, log=LOG, resolver=RESOLVER, llm=LLM)


def caller(presented: str | None = Header(default=None, alias=security.HEADER)
           ) -> security.Caller:
    """One gate, applied by dependency so no endpoint can forget it."""
    who = GATE.caller(presented)
    if who is None:
        raise HTTPException(401, "missing or unknown API key")
    return who


def clinic_caller(state: str, who: security.Caller) -> None:
    if state not in STATES:
        raise HTTPException(404, f"no clinic configured for {state!r}")
    if not who.may_query(state):
        raise HTTPException(403, f"{who.id} may not query {state}")
