"""The gold set against a real served model, one pass, every case recorded.

    ollama pull qwen2.5:3b-instruct
    .venv/bin/python -m scripts.live_eval            # writes docs/results/

The test suite runs generation against a scripted stand-in, which makes
faithfulness true by construction. This is the run that is not: the same
tally rules and the same four metrics as `/analytics`, but with the model the
clinic would actually serve, and every answer and refusal kept so the numbers
can be checked against the text that produced them.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinic_retrieval.agents.orchestrate import Orchestrator  # noqa: E402
from clinic_retrieval.analytics import (  # noqa: E402
    Log,
    answer_relevance,
    context_precision,
    context_recall,
    faithfulness,
    figure_recall,
    load_gold,
    percentile,
)
from clinic_retrieval.core import Answer, Clarify  # noqa: E402
from clinic_retrieval.llm import Llm  # noqa: E402
from clinic_retrieval.retrieval.index import Hybrid  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "results"
FORMS = {"en": "English", "hi": "Devanagari", "hinglish": "Romanised Hindi"}


def outcome(case: dict, result) -> str:
    """The same rules as `analytics.gold_tally`: a wrong answer and a handoff
    are different failures and are never summed."""
    if case["expect"] == "handoff":
        return "wrong" if isinstance(result, Answer) else "held"
    if not isinstance(result, Answer):
        return "missed"
    doc = result.citations[0].split("#")[0] if result.citations else ""
    if doc == case["expect"]:
        return "right"
    if doc.split("-")[0] == case["expect"].split("-")[0]:
        return "leaked"
    return "wrong"


def main() -> int:
    llm = Llm()
    if not llm.reachable():
        print(f"no model served at {llm.name}; nothing to measure")
        return 1

    index = Hybrid.from_corpus()
    by_id = {c.id: c for c in index.chunks}
    log = Log()
    cases = []
    for case in load_gold():
        hits = index.search(case["q"], case["state"], 4)
        started = time.perf_counter()
        response = Orchestrator(index, case["state"], log=log, llm=llm).handle(case["q"])
        ms = (time.perf_counter() - started) * 1000
        result = response.result
        row = {
            "q": case["q"], "form": case["form"], "state": case["state"],
            "expect": case["expect"], "outcome": outcome(case, result),
            "ms": round(ms),
            "trace": [f"{s.agent}: {s.detail}" for s in response.trace],
        }
        if isinstance(result, Answer):
            sources = [by_id[c].text for c in result.citations if c in by_id]
            row |= {"answer": result.text, "citations": result.citations,
                    "faithfulness": round(faithfulness(result.text, sources), 3),
                    "relevance": round(answer_relevance(case["q"], result.text, index), 3)}
            if case.get("figures"):
                row["figure_recall"] = figure_recall(result.text, case["figures"])
        elif isinstance(result, Clarify):
            row |= {"clarify": result.asks}
        else:
            row |= {"reason": result.reason.value, "note": result.note}
        if case["expect"] != "handoff":
            row |= {"context_recall": context_recall(hits, case["expect"]),
                    "context_precision": round(context_precision(hits, case["expect"]), 3)}
        cases.append(row)
        print(f"{row['outcome']:<7} {row['ms']:>6}ms  {case['form']:<9} {case['q'][:60]}")

    summary = {}
    for form in FORMS:
        rows = [r for r in cases if r["form"] == form]
        answerable = [r for r in rows if r["expect"] != "handoff"]
        answered = [r for r in answerable if "answer" in r]
        counts = {k: sum(r["outcome"] == k for r in rows)
                  for k in ("right", "wrong", "leaked", "missed", "held")}
        mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None   # noqa: E731
        summary[form] = counts | {
            "n": len(rows),
            "context_recall": mean([r["context_recall"] for r in answerable]),
            "context_precision": mean([r["context_precision"] for r in answerable]),
            "faithfulness": mean([r["faithfulness"] for r in answered]),
            "answer_relevance": mean([r["relevance"] for r in answered]),
            "figure_recall": mean([r["figure_recall"] for r in answered if "figure_recall" in r]),
            "p50_ms": round(percentile([r["ms"] for r in rows], 50)),
            "p95_ms": round(percentile([r["ms"] for r in rows], 95)),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "model": llm.model, "endpoint": llm.base_url, "date": date.today().isoformat(),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "summary": summary, "cases": cases,
    }
    slug = llm.model.replace(":", "-").replace("/", "-")
    path = OUT / f"live-{slug}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'form':<16} {'n':>2} {'right':>5} {'wrong':>5} {'leak':>4} {'miss':>4} "
          f"{'held':>4} {'recall':>6} {'prec':>5} {'faith':>5} {'relev':>5} {'figs':>5} {'p95ms':>6}")
    for form, s in summary.items():
        f = lambda v: "  -  " if v is None else f"{v:.2f}"   # noqa: E731
        print(f"{FORMS[form]:<16} {s['n']:>2} {s['right']:>5} {s['wrong']:>5} "
              f"{s['leaked']:>4} {s['missed']:>4} {s['held']:>4} {f(s['context_recall']):>6} "
              f"{f(s['context_precision']):>5} {f(s['faithfulness']):>5} "
              f"{f(s['answer_relevance']):>5} {f(s['figure_recall']):>5} {s['p95_ms']:>6}")
    print(f"\nwrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
