"""The query log, the RAGAS metrics, and the gold-set runner.

Two rules this module exists to enforce: never average across input forms,
and never add a wrong answer to a handoff.

RAGAS scores its four metrics with a judge model. The judge here would be the
same 3B that wrote the answer, which is marking its own homework, so all four
are computed against the retrieved set and the gold label instead. The judge
belongs on a reviewer's machine with a larger model, not in the clinic."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from clinic_retrieval.core import DRUG_GOLD, GOLD, Answer, Hit
from clinic_retrieval.guardrails import figures
from clinic_retrieval.lang import words
from clinic_retrieval.llm import Llm
from clinic_retrieval.retrieval.embed import cosine
from clinic_retrieval.retrieval.index import Hybrid


@dataclass(frozen=True)
class QueryRecord:
    form: str
    state: str
    intent: str
    agents: tuple[str, ...]
    answered: bool
    reason: str
    score: float
    ms: float
    flags: tuple[str, ...] = ()
    redactions: tuple[str, ...] = ()
    actor: str = "anonymous"         # which API key asked, for the audit trail
    tools: tuple[str, ...] = ()      # which tools ran, in order


@dataclass
class FormStats:
    asked: int = 0
    answered: int = 0
    abstained: int = 0
    flagged: int = 0
    redacted: int = 0
    scores: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def abstain_rate(self) -> float:
        return self.abstained / self.asked if self.asked else 0.0

    @property
    def mean_score(self) -> float:
        answered = [s for s in self.scores if s]
        return sum(answered) / len(answered) if answered else 0.0

    @property
    def p95_ms(self) -> float:
        return percentile(self.latencies, 95)


@dataclass(frozen=True)
class Unresolved:
    """A drug name nobody could match. The list of these is how a formulary
    gets fixed: curate the frequent ones into aliases, weekly."""
    question: str
    state: str
    nearest: str = ""


class Log:
    """One row per question. Grouped by input form, never averaged across them."""

    def __init__(self) -> None:
        self.records: list[QueryRecord] = []
        self.unresolved: list[Unresolved] = []

    def add(self, record: QueryRecord) -> None:
        self.records.append(record)

    def add_unresolved(self, question: str, state: str, nearest: str = "") -> None:
        self.unresolved.append(Unresolved(question, state, nearest))

    def unresolved_counts(self) -> list[tuple[str, int, str]]:
        counts: dict[str, int] = {}
        nearest: dict[str, str] = {}
        for item in self.unresolved:
            counts[item.question] = counts.get(item.question, 0) + 1
            nearest.setdefault(item.question, item.nearest)
        return sorted(((q, n, nearest[q]) for q, n in counts.items()),
                      key=lambda row: -row[1])

    def rollup(self) -> dict[str, FormStats]:
        out: dict[str, FormStats] = {}
        for r in self.records:
            stats = out.setdefault(r.form, FormStats())
            stats.asked += 1
            stats.latencies.append(r.ms)
            if r.answered:
                stats.answered += 1
                stats.scores.append(r.score)
            else:
                stats.abstained += 1
                stats.reasons[r.reason] = stats.reasons.get(r.reason, 0) + 1
            if r.flags:
                stats.flagged += 1
            if r.redactions:
                stats.redacted += 1
        return out


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    i = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[i]


def context_recall(hits: list[Hit], expected_doc: str) -> float:
    """Did the expected document reach the context at all."""
    return 1.0 if any(h.chunk.doc == expected_doc for h in hits) else 0.0


def context_precision(hits: list[Hit], expected_doc: str) -> float:
    """Reciprocal rank of the expected document in the context."""
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.doc == expected_doc:
            return 1.0 / rank
    return 0.0


def faithfulness(answer: str, sources: list[str]) -> float:
    """Share of the answer's content words that appear in its sources.

    A model writes the sentence now, so this no longer sits at 1.0 by
    construction and it is the number that moves first when a prompt changes.
    It undercounts legitimate paraphrase, which is the price of not using a
    judge; the guard that actually blocks output is the digit check, which is
    exact.
    """
    claim = words(answer)
    if not claim:
        return 0.0
    supported = set()
    for text in sources:
        supported |= set(words(text))
    return sum(1 for w in claim if w in supported) / len(claim)


def answer_relevance(question: str, answer: str, index: Hybrid) -> float:
    """Cosine between question and answer in the retrieval space."""
    return cosine(index.embedder.embed(question), index.embedder.embed(answer))


def figure_recall(answer: str, expected: list[int]) -> float:
    """Share of the figures a gold case asks for that the answer states, by
    value. The digit guard proves every figure in an answer is in a source; this
    is the other direction, and it is what catches an answer that cites the
    right chunk and reports the top-up instead of the cover."""
    if not expected:
        return 1.0
    stated = figures(answer)
    return sum(1 for n in expected if n in stated) / len(expected)


def load_gold(path: Path = GOLD) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class Scores:
    """Context metrics are over every answerable question. Faithfulness and
    relevance are over the ones that produced an answer, because neither is
    defined for an abstention. The gap between n and answered is the abstain
    rate, and it is reported rather than folded into the other four."""
    n: int = 0
    answered: int = 0
    context_recall: float = 0.0
    context_precision: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0


@dataclass
class Tally:
    """Wrong answers and handoffs are separate columns and never summed."""
    right: int = 0
    wrong: int = 0         # answered from a document about something else
    leaked: int = 0        # right scheme, wrong state. Must stay at zero
    missed: int = 0        # refused when it should have answered
    held: int = 0          # refused, correctly

    @property
    def total(self) -> int:
        return self.right + self.wrong + self.leaked + self.missed + self.held


def retrieval_tally(index: Hybrid, dense: float | None = None,
                    lexical: float | None = None,
                    gold: list[dict] | None = None) -> dict[str, Tally]:
    """Retrieval alone, with no model in the path.

    Scores the same gold set on the only question retrieval can answer: did the
    expected document clear a floor, and was it the first chunk that did. This
    is the half of the eval that stays measurable when the model changes or is
    not served at all, and it is what the floors were swept against.
    """
    dense = index.embedder.min_score if dense is None else dense
    lexical = index.lexical_floor if lexical is None else lexical
    by_form: dict[str, Tally] = {}
    for case in gold or load_gold():
        tally = by_form.setdefault(case["form"], Tally())
        hits = index.search(case["q"], case["state"], 4)
        cleared = [h for h in hits if h.dense >= dense or h.lexical >= lexical]

        if case["expect"] == "handoff":
            if cleared:
                tally.wrong += 1
            else:
                tally.held += 1
            continue
        if not cleared:
            tally.missed += 1
        elif cleared[0].chunk.doc == case["expect"]:
            tally.right += 1
        elif cleared[0].chunk.doc.split("-")[0] == case["expect"].split("-")[0]:
            tally.leaked += 1
        else:
            tally.wrong += 1
    return by_form


def gold_tally(index: Hybrid, llm: Llm | None = None,
               gold: list[dict] | None = None) -> dict[str, Tally]:
    """The gold set through the whole stack: guards, intent, retrieval,
    generation and every check. Keyed by input form and never averaged across
    them. Needs a model; `retrieval_tally` is the half that does not."""
    from clinic_retrieval.agents.orchestrate import Orchestrator

    llm = llm or Llm()
    by_form: dict[str, Tally] = {}
    for case in gold or load_gold():
        tally = by_form.setdefault(case["form"], Tally())
        result = Orchestrator(index, case["state"], llm=llm).handle(case["q"]).result

        if case["expect"] == "handoff":
            if isinstance(result, Answer):
                tally.wrong += 1
            else:
                tally.held += 1
            continue
        if not isinstance(result, Answer):
            tally.missed += 1
            continue
        doc = result.citations[0].split("#")[0] if result.citations else ""
        if doc == case["expect"]:
            tally.right += 1
        elif doc.split("-")[0] == case["expect"].split("-")[0]:
            tally.leaked += 1
        else:
            tally.wrong += 1
    return by_form


def run_with_log(index: Hybrid, gold: list[dict] | None = None,
                 llm: Llm | None = None) -> Log:
    """The gold set through the whole stack, with every query logged."""
    from clinic_retrieval.agents.orchestrate import Orchestrator  # the orchestrator logs here

    llm = llm or Llm()
    log = Log()
    for case in gold or load_gold():
        Orchestrator(index, case["state"], log=log, llm=llm).handle(case["q"])
    return log


def ragas_scores(index: Hybrid, gold: list[dict] | None = None,
                 llm: Llm | None = None) -> dict[str, Scores]:
    """The four metrics per input form. Answerable cases only.

    The two context metrics need no model and are always measured. Faithfulness
    and relevance need an answer, so with no model reachable they stay at zero
    and `answered` says why.
    """
    from clinic_retrieval.agents.orchestrate import Orchestrator

    llm = llm or Llm()
    by_id = {c.id: c for c in index.chunks}
    out: dict[str, Scores] = {}
    for case in gold or load_gold():
        if case["expect"] == "handoff":
            continue
        scores = out.setdefault(case["form"], Scores())
        hits = index.search(case["q"], case["state"], 4)
        scores.n += 1
        scores.context_recall += context_recall(hits, case["expect"])
        scores.context_precision += context_precision(hits, case["expect"])

        result = Orchestrator(index, case["state"], llm=llm).handle(case["q"]).result
        if isinstance(result, Answer):
            scores.answered += 1
            sources = [by_id[c].text for c in result.citations if c in by_id]
            scores.faithfulness += faithfulness(result.text, sources)
            scores.answer_relevance += answer_relevance(
                case["q"], result.text, index)

    for scores in out.values():
        if scores.n:
            scores.context_recall /= scores.n
            scores.context_precision /= scores.n
        if scores.answered:
            scores.faithfulness /= scores.answered
            scores.answer_relevance /= scores.answered
    return out


@dataclass
class DrugTally:
    resolved: int = 0          # right code
    misresolved: int = 0       # a code, and the wrong one. Must stay at zero
    asked: int = 0             # correctly ambiguous
    over_asked: int = 0        # ambiguous when one answer was right
    held: int = 0              # correctly unresolved
    over_held: int = 0         # unresolved when one answer was right
    leaked: int = 0            # resolved when it should have asked or held


def load_drug_gold(path: Path = DRUG_GOLD) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def drug_scores(accept: float, margin: float,
                gold: list[dict] | None = None) -> DrugTally:
    from clinic_retrieval.agents.drugs import DrugResolver

    gold = gold or load_drug_gold()
    resolver = DrugResolver(accept=accept, margin=margin)
    tally = DrugTally()
    for case in gold:
        want = case["expect"]
        got = resolver.resolve(case["q"])
        if want.startswith("D-"):
            if got.outcome == "resolved":
                if got.best.code == want:
                    tally.resolved += 1
                else:
                    tally.misresolved += 1
            elif got.outcome == "ambiguous":
                tally.over_asked += 1
            else:
                tally.over_held += 1
        elif got.outcome == want:
            tally.asked += 1 if want == "ambiguous" else 0
            tally.held += 1 if want == "unresolved" else 0
        elif got.outcome == "resolved":
            tally.leaked += 1
        else:
            tally.over_asked += 1 if got.outcome == "ambiguous" else 0
            tally.over_held += 1 if got.outcome == "unresolved" else 0
    return tally


def drug_sweep(accepts: list[float], margins: list[float]) -> None:
    gold = load_drug_gold()
    print(f"{'accept':>7} {'margin':>7} | {'right':>6} {'WRONG':>6} {'asked':>6} "
          f"{'held':>5} {'over-ask':>9} {'over-hold':>10} {'leaked':>7}")
    for accept in accepts:
        for margin in margins:
            t = drug_scores(accept, margin, gold)
            print(f"{accept:>7.2f} {margin:>7.2f} | {t.resolved:>6} {t.misresolved:>6} "
                  f"{t.asked:>6} {t.held:>5} {t.over_asked:>9} {t.over_held:>10} "
                  f"{t.leaked:>7}")
