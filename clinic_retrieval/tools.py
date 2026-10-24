"""Every capability an agent can reach, declared instead of called directly.

Three reasons this is a registry and not three method calls. Each tool declares
which data it touches, which is what the governance table in ORCHESTRATION.md is
built from. Each call is recorded with its arguments, so the trace is an audit
trail rather than a log line. And the declared parameters are the same shape a
function-calling model would be handed, so putting a model in front of this is a
new caller rather than a rewrite.

An agent here is a tool with a decision inside it. The orchestrator is the only
thing that calls tools; tools never call each other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from clinic_retrieval.agents.docs import Clinic
from clinic_retrieval.agents.drugs import DrugResolver, StockTable
from clinic_retrieval.core import Answer
from clinic_retrieval.llm import Llm
from clinic_retrieval.retrieval.index import Hybrid


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict[str, str]
    reads: str                      # the data this tool can see, for governance
    run: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """What a function-calling model would be given."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string", "description": d}
                               for p, d in self.params.items()},
                "required": list(self.params),
            },
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    summary: str

    def __str__(self) -> str:
        shown = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.name}({shown}) -> {self.summary}"


class Toolbox:
    """Built once per clinic, because two of the three tools are clinic-scoped."""

    def __init__(self, index: Hybrid, state: str,
                 resolver: DrugResolver | None = None, llm: Llm | None = None):
        self.state = state.lower()
        self.resolver = resolver or DrugResolver()
        self.stock = StockTable(state, self.resolver)
        self.docs = Clinic(index, state, llm)
        self.calls: list[ToolCall] = []

        self.tools = {t.name: t for t in (
            Tool(
                name="resolve_drug",
                description="Match a drug name as typed against the national "
                            "formulary. Returns one drug, several candidates, "
                            "or nothing.",
                params={"question": "the question as the user typed it"},
                reads="national formulary",
                run=self._resolve_drug,
            ),
            Tool(
                name="check_stock",
                description="This clinic's own counted stock for one formulary "
                            "code. Says nothing about any other clinic.",
                params={"code": "a formulary code, for example D-AMX-125"},
                reads="this clinic's stock count",
                run=self._check_stock,
            ),
            Tool(
                name="answer_from_documents",
                description="State-scoped hybrid retrieval, then a generated "
                            "answer from the retrieved text only. Returns the "
                            "answer with its citation, or a refusal with a reason.",
                params={"query": "the question to retrieve against"},
                reads="corpus documents for this state and the national ones",
                run=self._answer_from_documents,
            ),
        )}

    def schema(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def permissions(self) -> list[tuple[str, str]]:
        return [(tool.name, tool.reads) for tool in self.tools.values()]

    def call(self, name: str, **kwargs: Any) -> Any:
        tool = self.tools[name]
        result, summary = tool.run(**kwargs)
        self.calls.append(ToolCall(name, kwargs, summary))
        return result

    # ---- the tools themselves -------------------------------------------

    def _resolve_drug(self, question: str):
        found = self.resolver.resolve(question)
        if found.outcome == "resolved":
            summary = f"{found.best.code} via {found.best.how}"
        elif found.outcome == "ambiguous":
            summary = f"ambiguous between {len(found.candidates)}"
        else:
            summary = "no drug in the formulary fits"
        return found, summary

    def _check_stock(self, code: str):
        record = self.stock.record(code)
        if record is None:
            return None, f"{code} in the formulary, not carried at {self.stock.clinic}"
        return record, f"{code}, {record.on_hand} on hand at {record.clinic}"

    def _answer_from_documents(self, query: str):
        found = self.docs.ask(query)
        if isinstance(found, Answer):
            return found, (f"generated from {','.join(found.citations)} "
                           f"at {found.score:.2f}x floor")
        return found, f"nothing returned: {found.reason.value}"
