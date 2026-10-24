"""Floor calibration and the retrieval report.

    .venv/bin/python -m scripts.evaluate

Three thresholds in this system were measured rather than chosen, and this is
where all three come from. Two retrieval floors, one per retriever, because a
cosine and a normalised BM25 sum are not the same quantity and cannot share a
threshold. And the drug resolver's accept score, where the constraint is
different: nothing may resolve to the wrong drug at any setting, so the sweep
is read for the WRONG column being zero before anything else.

Each table is swept against a gold set that contains out-of-scope questions as
well as answerable ones, so what a candidate threshold costs in missed answers
and what it buys in refusals are both visible in the same row.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinic_retrieval.analytics import drug_sweep, load_gold, retrieval_tally  # noqa: E402
from clinic_retrieval.retrieval.index import Hybrid  # noqa: E402


def hits_table(index: Hybrid) -> None:
    gold = load_gold()
    print(f"{'form':<9} {'expected':<16} {'rank':>4} {'dense':>6} {'lex':>6} "
          f"{'by':<8} question")
    for case in gold:
        hits = index.search(case["q"], case["state"], 4)
        want = case["expect"]
        rank = next((n for n, h in enumerate(hits, 1)
                     if h.chunk.doc == want), 0) if want != "handoff" else 0
        top = hits[0] if hits else None
        print(f"{case['form']:<9} {want:<16} {rank:>4} "
              f"{top.dense if top else 0:>6.3f} {top.lexical if top else 0:>6.3f} "
              f"{top.retriever if top else '-':<8} {case['q'][:46]}")


def sweep(index: Hybrid, dense_floors: list[float],
          lexical_floors: list[float]) -> None:
    gold = load_gold()
    print(f"\n{'dense':>6} {'lex':>6} | {'right':>6} {'WRONG':>6} {'leaked':>7} "
          f"{'missed':>7} {'held':>5}")
    for dense in dense_floors:
        for lex in lexical_floors:
            by_form = retrieval_tally(index, dense, lex, gold)
            t = {k: sum(getattr(v, k) for v in by_form.values())
                 for k in ("right", "wrong", "leaked", "missed", "held")}
            print(f"{dense:>6.3f} {lex:>6.3f} | {t['right']:>6} {t['wrong']:>6} "
                  f"{t['leaked']:>7} {t['missed']:>7} {t['held']:>5}")


if __name__ == "__main__":
    index = Hybrid.from_corpus()
    print(index.name, f"| {len(index.chunks)} chunks\n")
    hits_table(index)

    print("\nretrieval floors: dense x lexical, over 25 gold questions")
    sweep(index, [0.840, 0.855, 0.870], [0.011, 0.055, 0.102])
    print(f"\nshipped: dense {index.embedder.min_score}, "
          f"lexical {index.lexical_floor}")

    print("\ndrug resolution: accept x margin, over 35 gold drug names")
    print("WRONG must be zero in every row that could ship")
    drug_sweep([0.80, 0.84, 0.86, 0.88], [0.03, 0.05, 0.08])
