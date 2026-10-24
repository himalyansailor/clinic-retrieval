"""The orchestrator: guards, memory, intent, structured lookup or retrieval,
generation, validation, escalation.

Hand-written and sequential. Two model calls per question at most, one to read
the question and one to write the answer, both to the same model. Everything
between them is a decision this file makes, every data source it reaches is a
declared tool, and the trace it returns is the list of those calls with their
arguments. See ORCHESTRATION.md.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date

from clinic_retrieval import guardrails as guard
from clinic_retrieval.agents.drugs import DrugResolver
from clinic_retrieval.agents.intent import Route, plan, read
from clinic_retrieval.analytics import Log, QueryRecord
from clinic_retrieval.core import Answer, Clarify, Handoff, Reason, Result
from clinic_retrieval.lang import detect_form
from clinic_retrieval.llm import Llm
from clinic_retrieval.retrieval.index import Hybrid
from clinic_retrieval.session import Pending, Session
from clinic_retrieval.tools import Toolbox

STOCK_OUT_QUERY = "What should staff do when a listed drug is not available in clinic stock?"


@dataclass(frozen=True)
class Step:
    agent: str
    detail: str


@dataclass(frozen=True)
class Response:
    result: Result
    trace: tuple[Step, ...]
    route: Route
    session: Session | None = None


class Orchestrator:
    def __init__(self, index: Hybrid, state: str, log: Log | None = None,
                 today: date | None = None, resolver: DrugResolver | None = None,
                 llm: Llm | None = None):
        self.index = index
        self.state = state.lower()
        self.llm = llm or Llm()
        self.tools = Toolbox(index, state, resolver, self.llm)
        self.resolver = self.tools.resolver
        self.stock = self.tools.stock
        self.docs = self.tools.docs
        self.log = log
        self.today = today

    def handle(self, question: str, session: Session | None = None,
               actor: str = "anonymous") -> Response:
        started = time.perf_counter()
        trace: list[Step] = []
        self.tools.calls.clear()

        screened = guard.screen_input(question)
        for trip in screened.trips:
            trace.append(Step("guard:in", f"{trip.guard} {trip.action}, {trip.detail}"))
        if screened.blocked is not None:
            handoff = Handoff(screened.blocked, self.state,
                              note=screened.trips[0].detail)
            return self._stop(question, handoff, trace, _no_route(screened.blocked),
                              started, screened, session, actor)

        question = screened.text
        if session is not None:
            session.turns += 1

        # Asking for a person is answered by handing over, not by a model call.
        if guard.WANTS_HUMAN.search(question):
            trace.append(Step("escalation", "asked for a person"))
            return self._stop(question, Handoff(Reason.OUT_OF_SCOPE, self.state,
                                                note="asked for a person"),
                              trace, _no_route(Reason.OUT_OF_SCOPE), started,
                              screened, session, actor)

        route = self._route(question, session, trace)
        if route is None:
            down = not self.llm.reachable()
            reason = Reason.MODEL_UNAVAILABLE if down else Reason.UNCLEAR_INTENT
            note = (f"no reply from {self.llm.name}" if down else
                    f"{self.llm.name} did not return one of the three labels")
            trace.append(Step("escalation", reason.value))
            return self._stop(question, Handoff(reason, self.state, note=note),
                              trace, _no_route(reason), started, screened,
                              session, actor)

        if route.reason is Reason.AMBIGUOUS_DRUG:
            return self._clarify(question, route, trace, started, screened,
                                 session, actor)

        if route.reason is not None:
            if route.reason is Reason.NOT_IN_FORMULARY and self.log is not None:
                nearest = (route.resolution.names[0]
                           if route.resolution and route.resolution.candidates else "")
                self.log.add_unresolved(question, self.state, nearest)
                trace.append(Step("analytics", "unresolved drug name recorded"))
            trace.append(Step("escalation", route.reason.value))
            return self._stop(question, Handoff(route.reason, self.state, note=route.why),
                              trace, route, started, screened, session, actor)

        agents = list(route.agents)
        answered_by: set[str] = set()
        docs_query = question
        parts: list[str] = []
        citations: list[str] = []
        dated: list[tuple[str, str]] = []
        sources: list[str] = []
        refused: Handoff | None = None
        score: float | None = None

        if "stock" in agents and route.drug is not None:
            record = self.tools.call("check_stock", code=route.drug)
            trace.append(Step("stock", str(self.tools.calls[-1])))
            if record is None:
                # Known drug, no count here. That is an answer, and it needs the
                # indent procedure with it, so one agent's finding changes the plan.
                parts.append(self.stock.not_carried(route.drug))
                if route.owner == "stock":
                    # Only an availability question may be answered with the
                    # procedure instead. Rewriting any other question would put
                    # the retrieval floor on a question nobody asked.
                    docs_query = STOCK_OUT_QUERY
                    if "docs" not in agents:
                        agents.append("docs")
                        trace.append(Step("router", "re-planned: adding the procedure"))
            else:
                parts.append(self.stock.line(record))
            answered_by.add("stock")
            sources.append(parts[-1])
            dated.append((f"stock:{self.stock.clinic}", self.stock.counted))

        if "docs" in agents:
            found = self.tools.call("answer_from_documents", query=docs_query)
            trace.append(Step("docs", str(self.tools.calls[-1])))
            if isinstance(found, Answer):
                parts.append(found.text)
                answered_by.add("docs")
                citations += found.citations
                dated += found.sources
                sources += [c.text for c in self.index.chunks
                            if c.id in set(found.citations)]
                score = found.score if score is None else min(score, found.score)
            else:
                refused = found

        # One agent's finding is not an answer to a question another agent owns.
        # A procedure question whose procedure nobody can find is a refusal,
        # even with a stock count in hand.
        if route.owner not in answered_by:
            handoff = refused or Handoff(Reason.NO_RELEVANT_CONTEXT, self.state)
            trace.append(Step("escalation", handoff.reason.value))
            return self._stop(question, handoff, trace, route, started, screened,
                              session, actor)

        failed = _verify(self.index, self.state, citations, dated)
        if failed is not None:
            trace.append(Step("verifier", f"rejected: {failed.value}"))
            trace.append(Step("escalation", failed.value))
            return self._stop(question, Handoff(failed, self.state,
                                                note="failed the verifier"),
                              trace, route, started, screened, session, actor)
        trace.append(Step("verifier", f"passed, {len(dated)} dated source(s)"))

        text = " ".join(parts)
        checked = guard.screen_output(text, sources, dated, self.today)
        for trip in checked.trips:
            trace.append(Step("guard:out", f"{trip.guard} {trip.action}, {trip.detail}"))
        if checked.blocked is not None:
            trace.append(Step("escalation", checked.blocked.value))
            return self._stop(question, Handoff(checked.blocked, self.state,
                                                note=checked.trips[0].detail),
                              trace, route, started, screened, session, actor)

        # A structured lookup is exact, so an answer with no retrieval in it
        # reports 1.0 rather than borrowing a retrieval score it never had.
        answer = Answer(text=text, citations=citations, sources=dated,
                        state=self.state, score=1.0 if score is None else score,
                        flags=tuple(checked.flags))
        return self._stop(question, answer, trace, route, started, screened,
                          session, actor)

    # ---- routing, including the one thing that is remembered -------------

    def _route(self, question: str, session: Session | None,
               trace: list[Step]) -> Route | None:
        pending = session.pending if session is not None else None
        if pending is not None:
            session.pending = None
            chosen = self._chosen(question, pending)
            if chosen is not None:
                trace.append(Step("memory", f"answers the pending choice: {chosen}"))
                return Route(("stock",), chosen, "the drug the clinic picked",
                             "inventory", owner="stock")
            trace.append(Step("memory", "not an answer to the pending choice, "
                                        "treating it as a new question"))

        intent = read(question, self.llm)
        if intent is None:
            return None
        trace.append(Step("llm:intent",
                          f"{intent.label}" + (f", drug {intent.drug!r}"
                                               if intent.drug else "")))

        found = self.tools.call("resolve_drug", question=intent.drug or question)
        trace.append(Step("resolver", str(self.tools.calls[-1])))
        route = plan(intent, found)
        trace.append(Step("router", f"{'+'.join(route.agents)}: {route.why}"))
        return route

    def _clarify(self, question: str, route: Route, trace: list[Step],
                 started: float, screened: guard.Screen, session: Session | None,
                 actor: str) -> Response:
        options = tuple(route.resolution.names)
        codes = tuple(c.code for c in route.resolution.candidates)
        trace.append(Step("clarify", ", ".join(options)))
        if session is not None:
            session.pending = Pending("drug_choice", options, codes)
            trace.append(Step("memory", "holding the choice for the next turn"))
        asks = ("More than one drug matches that name. Which one: "
                + ", or ".join(options) + "?")
        return self._stop(question, Clarify(asks, options, self.state), trace,
                          route, started, screened, session, actor)

    def _chosen(self, question: str, pending: Pending) -> str | None:
        """Match a reply against the options already offered, and nothing else."""
        asked = question.strip().lower()
        if asked.isdigit() and 1 <= int(asked) <= len(pending.codes):
            return pending.codes[int(asked) - 1]
        for option, code in zip(pending.options, pending.codes):
            if option.lower() in asked or asked in option.lower():
                return code
        found = self.resolver.resolve(question)
        if found.outcome == "resolved" and found.best.code in pending.codes:
            return found.best.code
        return None

    def _stop(self, question: str, result: Result, trace: list[Step], route: Route,
              started: float, screened: guard.Screen, session: Session | None,
              actor: str) -> Response:
        response = Response(result, tuple(trace), route, session)
        if self.log is not None:
            answered = isinstance(result, Answer)
            self.log.add(QueryRecord(
                form=detect_form(question),
                state=self.state,
                intent=route.label,
                agents=route.agents,
                answered=answered,
                reason="" if answered else result.reason.value,
                score=result.score if answered else 0.0,
                ms=(time.perf_counter() - started) * 1000,
                flags=result.flags if answered else (),
                redactions=tuple(t.guard for t in screened.trips
                                 if t.action == "redact"),
                actor=actor,
                tools=tuple(call.name for call in self.tools.calls),
            ))
        return response


def _verify(index: Hybrid, state: str, citations: list[str],
            dated: list[tuple[str, str]]) -> Reason | None:
    """State and dating re-checked at the boundary where answers are composed.

    The document agent already filters by state, but once two agents share one
    answer no single agent owns the whole of it.
    """
    by_id = {c.id: c for c in index.chunks}
    for cid in citations:
        chunk = by_id.get(cid)
        if chunk is None:
            return Reason.UNGROUNDED_CITATION
        if chunk.state != "all" and chunk.state.lower() != state.lower():
            return Reason.WRONG_STATE
    if any(not when or when == "unknown" for _, when in dated):
        return Reason.NO_RELEVANT_CONTEXT
    return None


def _no_route(reason: Reason) -> Route:
    return Route(("guard",), None, reason.value, "", "guard", reason)
