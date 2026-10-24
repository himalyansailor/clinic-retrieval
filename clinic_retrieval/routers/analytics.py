"""Per-form rollups, the RAGAS metrics, and names nothing matched."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from clinic_retrieval import deps
from clinic_retrieval.analytics import ragas_scores
from clinic_retrieval.deps import LOG, caller, get_index
from clinic_retrieval.models import FormRow, RagasRow, UnresolvedRow

# Clinic data, so the gate applies to the whole group.
router = APIRouter(prefix="/analytics", tags=["analytics"],
                   dependencies=[Depends(caller)])


@router.get("/live", response_model=list[FormRow])
def live() -> list[FormRow]:
    """Rollup of what this process has actually been asked, per input form."""
    return [
        FormRow(form=form, asked=s.asked, answered=s.answered,
                abstain_rate=round(s.abstain_rate, 3),
                mean_score=round(s.mean_score, 4), p95_ms=round(s.p95_ms, 2),
                flagged=s.flagged, redacted=s.redacted, reasons=s.reasons)
        for form, s in sorted(LOG.rollup().items())
    ]


# The gold set and the model do not change inside a process, and a run is
# about forty model calls. Recomputed per request, it competed with the
# clinic's own questions for the one model every time the page refreshed.
_ragas: tuple[object, list[RagasRow]] | None = None


@router.get("/ragas", response_model=list[RagasRow])
def ragas() -> list[RagasRow]:
    """The four metrics over the gold set, per form, never averaged. Computed
    once per process and model."""
    global _ragas
    if _ragas is None or _ragas[0] is not deps.LLM:
        _ragas = (deps.LLM, _score())
    return _ragas[1]


def _score() -> list[RagasRow]:
    return [
        RagasRow(form=form, n=s.n, answered=s.answered,
                 context_recall=round(s.context_recall, 3),
                 context_precision=round(s.context_precision, 3),
                 faithfulness=round(s.faithfulness, 3),
                 answer_relevance=round(s.answer_relevance, 3))
        for form, s in sorted(ragas_scores(get_index(), llm=deps.LLM).items())
    ]


@router.get("/unresolved", response_model=list[UnresolvedRow])
def unresolved() -> list[UnresolvedRow]:
    """Drug names nothing in the formulary matched. This list is the input to
    curating aliases, which is how a closed vocabulary actually gets good."""
    return [UnresolvedRow(question=question, count=count, nearest=nearest)
            for question, count, nearest in LOG.unresolved_counts()]
