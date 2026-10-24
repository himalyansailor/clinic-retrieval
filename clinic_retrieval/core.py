"""Types everything passes around, and where the data lives."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The sample data ships next to the package; a deployment points this at its own.
DATA = Path(os.environ.get("CLINIC_DATA", ROOT / "data"))
CORPUS = DATA / "corpus"
STOCK = DATA / "stock" / "clinic-stock.json"
FORMULARY = DATA / "formulary" / "formulary.json"
GOLD = DATA / "gold" / "gold.jsonl"
DRUG_GOLD = DATA / "gold" / "drugs.jsonl"


class Reason(str, Enum):
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    BELOW_FLOOR = "below_floor"
    WRONG_STATE = "wrong_state"
    UNGROUNDED_CITATION = "ungrounded_citation"
    OUT_OF_SCOPE = "out_of_scope"
    NOT_IN_FORMULARY = "not_in_formulary"
    CLINICAL_ADVICE = "clinical_advice"
    EMERGENCY = "emergency"
    UNSAFE_INPUT = "unsafe_input"
    STALE_SOURCE = "stale_source"
    AMBIGUOUS_DRUG = "ambiguous_drug"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_ABSTAINED = "model_abstained"
    UNCLEAR_INTENT = "unclear_intent"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    text: str
    state: str          # "bihar", "jharkhand", ... or "all"
    kind: str           # "policy" for scheme documents, "clinic" for protocols
    updated: str        # source document date


@dataclass(frozen=True)
class Hit:
    """Two retrievers, two scales. A cosine and a normalised BM25 sum cannot be
    compared with each other or share a threshold, so each keeps its own number
    and its own floor, and `cleared` is the one question a caller has to ask."""
    chunk: Chunk
    dense: float
    lexical: float = 0.0
    cleared: bool = False
    retriever: str = ""          # which floor it cleared, for the trace

    @property
    def score(self) -> float:
        """The cosine. Reported as the relevance figure because that is the
        number the rest of the world quotes; the lexical score is on its own
        scale and is reported next to it, never averaged with it."""
        return self.dense


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]
    sources: list[tuple[str, str]]   # (chunk id or stock table, date)
    state: str
    score: float
    flags: tuple[str, ...] = ()      # guards that warned without blocking


@dataclass(frozen=True)
class Handoff:
    reason: Reason
    state: str
    best_score: float = 0.0
    note: str = ""
    considered: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Clarify:
    """Neither an answer nor a refusal. The question was understood and more
    than one drug fits, so the caller has to put the choice back to the user."""
    asks: str
    options: tuple[str, ...]
    state: str
    reason: Reason = Reason.AMBIGUOUS_DRUG


# Three outcomes, as equals. Not answering is a return value, so every caller
# has to handle it.
Result = Answer | Handoff | Clarify
