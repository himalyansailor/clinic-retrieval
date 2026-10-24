"""One question, one of three outcomes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from clinic_retrieval import security
from clinic_retrieval.core import Answer, Clarify
from clinic_retrieval.deps import LOG, SESSIONS, caller, clinic_caller, get_orchestrator
from clinic_retrieval.models import AskRequest, AskResponse, Source, TraceStep

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest,
        who: security.Caller = Depends(caller)) -> AskResponse:
    state = body.state.lower()
    clinic_caller(state, who)

    # A session only exists to answer a question the system asked. One is
    # offered on every turn and thrown away unless something is outstanding.
    session = SESSIONS.get(body.session, state) or SESSIONS.start(state)
    response = get_orchestrator(state).handle(body.question, session=session,
                                              actor=who.id)
    session = SESSIONS.forget_unless_pending(session)
    result = response.result
    latency = LOG.records[-1].ms if LOG.records else 0.0

    common = dict(
        state=state,
        route=list(response.route.agents),
        trace=[TraceStep(agent=s.agent, detail=s.detail) for s in response.trace],
        latency_ms=round(latency, 2),
        session=session.id if session else None,
    )
    if isinstance(result, Answer):
        return AskResponse(
            answered=True, text=result.text, score=round(result.score, 4),
            citations=result.citations,
            sources=[Source(ref=ref, dated=when) for ref, when in result.sources],
            flags=list(result.flags), **common,
        )
    if isinstance(result, Clarify):
        return AskResponse(answered=False, reason=result.reason.value,
                           asks=result.asks, options=list(result.options), **common)
    return AskResponse(answered=False, reason=result.reason.value,
                       note=result.note or None, **common)
