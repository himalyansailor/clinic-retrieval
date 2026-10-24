import pytest

from clinic_retrieval.agents.orchestrate import Orchestrator
from clinic_retrieval.analytics import (
    Log,
    faithfulness,
    gold_tally,
    percentile,
    ragas_scores,
    run_with_log,
)
from tests.conftest import FakeLlm


@pytest.fixture(scope="module")
def scores(index):
    return ragas_scores(index, llm=FakeLlm())


@pytest.fixture(scope="module")
def rollup(index):
    return run_with_log(index, llm=FakeLlm()).rollup()


def test_retrieval_finds_the_right_document_in_every_form(scores):
    for form, s in scores.items():
        assert s.context_recall == pytest.approx(1.0), form


def test_every_answerable_question_is_answered_in_every_form(scores):
    # Before the lexical retriever, two of the four romanised questions
    # abstained with the right document already in the context.
    for form, s in scores.items():
        assert s.answered == s.n, form


def test_the_right_document_ranks_first_in_the_context(scores):
    for form, s in scores.items():
        assert s.context_precision > 0.85, form


def test_forms_are_never_pooled(scores):
    assert set(scores) == {"en", "hi", "hinglish"}


def test_the_two_context_metrics_need_no_model(index):
    # The question a reviewer asks: how much of this is measurable with the
    # model server switched off.
    off = ragas_scores(index, llm=FakeLlm(up=False))
    for form, s in off.items():
        assert s.context_recall == pytest.approx(1.0), form
        assert s.answered == 0
        assert s.faithfulness == 0.0


def test_log_rolls_up_per_form(rollup):
    assert set(rollup) == {"en", "hi", "hinglish"}
    assert all(s.p95_ms > 0 for s in rollup.values())
    assert all(s.mean_score > 0.8 for s in rollup.values())


def test_abstentions_are_counted_by_reason(rollup):
    for form, stats in rollup.items():
        assert sum(stats.reasons.values()) == stats.abstained, form


def test_out_of_scope_questions_are_the_only_abstentions(rollup):
    # Five in the gold set: four below the floor, one asking for a person.
    assert sum(s.abstained for s in rollup.values()) == 5
    reasons = {r for s in rollup.values() for r in s.reasons}
    assert reasons == {"below_floor", "out_of_scope"}


def test_the_intent_is_recorded_on_every_row(index):
    log = run_with_log(index, llm=FakeLlm())
    assert all(r.intent for r in log.records if r.reason != "out_of_scope")


def test_the_gold_set_answers_from_the_right_document_in_every_form(index):
    by_form = gold_tally(index, FakeLlm())
    assert set(by_form) == {"en", "hi", "hinglish"}
    for form, t in by_form.items():
        assert t.wrong == 0, form
        assert t.leaked == 0, form          # right scheme, wrong state
        assert t.missed == 0, form
    assert sum(t.right for t in by_form.values()) == 20
    assert sum(t.held for t in by_form.values()) == 5


def test_percentile_picks_a_real_sample():
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    assert percentile([], 95) == 0.0


def test_faithfulness_drops_when_the_answer_adds_words():
    assert faithfulness("pays fourteen hundred",
                        ["the scheme pays fourteen hundred"]) == 1.0
    assert faithfulness("pays fourteen hundred rupees monthly forever",
                        ["the scheme pays fourteen hundred"]) < 1.0


def test_log_records_redactions(index, llm):
    log = Log()
    Orchestrator(index, "bihar", log=log, llm=llm).handle(
        "call 9888888888, is ORS in stock")
    assert log.records[-1].redactions == ("phone",)


def test_figure_recall_catches_the_right_chunk_reporting_the_wrong_figure():
    # Seen against a served 3B: asked for the cover, it reported the top-up.
    # Cited correctly and every figure grounded, so only this metric sees it.
    from clinic_retrieval.analytics import figure_recall
    top_up_only = "बिहार में पीएम-जय कवर प्रति परिवार प्रति वर्ष पचास हज़ार रुपये का अतिरिक्त टॉप-अप देता है।"
    assert figure_recall(top_up_only, [500000]) == 0.0
    assert figure_recall("Cover is up to five lakh rupees per family per year.", [500000]) == 1.0
    assert figure_recall("कवर पाँच लाख रुपये तक है", [500000]) == 1.0
