"""Request and response bodies for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    state: str = Field(default="bihar", examples=["bihar", "jharkhand"])
    session: str | None = Field(default=None,
                                description="returned by a previous answer that "
                                            "asked a question back")


class Source(BaseModel):
    ref: str
    dated: str


class TraceStep(BaseModel):
    agent: str
    detail: str


class AskResponse(BaseModel):
    answered: bool
    text: str | None = None
    asks: str | None = None          # set when more than one drug fits the name
    options: list[str] = []
    reason: str | None = None
    note: str | None = None
    state: str
    score: float | None = None
    citations: list[str] = []
    sources: list[Source] = []
    flags: list[str] = []
    route: list[str]
    trace: list[TraceStep]
    latency_ms: float
    session: str | None = None       # set only while a question is outstanding


class FormRow(BaseModel):
    form: str
    asked: int
    answered: int
    abstain_rate: float
    mean_score: float
    p95_ms: float
    flagged: int
    redacted: int
    reasons: dict[str, int]


class RagasRow(BaseModel):
    form: str
    n: int
    answered: int
    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevance: float


class UnresolvedRow(BaseModel):
    question: str
    count: int
    nearest: str


class ToolRow(BaseModel):
    name: str
    description: str
    reads: str
    parameters: dict
