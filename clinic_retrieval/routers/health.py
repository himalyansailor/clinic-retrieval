"""What is loaded, and what the clinics look like."""
from __future__ import annotations

from fastapi import APIRouter

from clinic_retrieval import deps
from clinic_retrieval.deps import GATE, SESSIONS, STATES, get_index, get_orchestrator
from clinic_retrieval.models import ToolRow

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    index = get_index()
    return {
        "status": "ok",
        "chunks": len(index.chunks),
        "retrieval": index.name,
        "floors": {"dense": index.embedder.min_score,
                   "lexical": index.lexical_floor},
        "model": deps.LLM.name,
        "model_reachable": deps.LLM.reachable(),
        "formulary": len(get_orchestrator(STATES[0]).resolver.drugs),
        "auth": "open" if GATE.open else f"{len(GATE.keys)} key(s)",
        "sessions": len(SESSIONS),
        "clinics": [
            {
                "state": state,
                "code": get_orchestrator(state).stock.clinic,
                "counted": get_orchestrator(state).stock.counted,
                "drugs": len(get_orchestrator(state).stock.drugs),
            }
            for state in STATES
        ],
    }


@router.get("/tools", response_model=list[ToolRow])
def tools() -> list[ToolRow]:
    """What the agents can reach, and which data each tool can see.

    The same list a function-calling model would be handed, which is why it is
    an endpoint rather than a comment.
    """
    toolbox = get_orchestrator(STATES[0]).tools
    return [ToolRow(name=tool.name, description=tool.description,
                    reads=tool.reads, parameters=tool.schema()["parameters"])
            for tool in toolbox.tools.values()]
