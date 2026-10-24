import pytest

from clinic_retrieval.agents.docs import Clinic
from clinic_retrieval.analytics import load_gold
from clinic_retrieval.core import Answer, Handoff, Reason
from clinic_retrieval.lang import keys
from clinic_retrieval.retrieval.embed import SentenceEmbedder
from clinic_retrieval.retrieval.index import load
from clinic_retrieval.retrieval.lexical import Bm25, terms

ROMANISED = "ayushman bharat yojana mein kitna cover milta hai"


@pytest.fixture
def clinic(index, llm):
    return Clinic(index, "bihar", llm)


# ---- the state boundary --------------------------------------------------

def test_same_question_two_states_two_answers(index, llm):
    question = "What is the Janani Suraksha Yojana amount for a rural delivery?"
    bihar = Clinic(index, "bihar", llm).ask(question)
    jharkhand = Clinic(index, "jharkhand", llm).ask(question)

    assert isinstance(bihar, Answer) and isinstance(jharkhand, Answer)
    assert bihar.citations[0].startswith("jsy-bihar")
    assert jharkhand.citations[0].startswith("jsy-jharkhand")
    assert "fourteen hundred" in bihar.text
    assert "seventeen hundred" in jharkhand.text


def test_candidate_set_excludes_other_states(index):
    for state, forbidden in [("bihar", "jharkhand"), ("jharkhand", "bihar")]:
        for chunk in (index.chunks[i] for i in index.visible(state)):
            assert forbidden not in chunk.id


def test_state_top_up_does_not_leak(index, llm):
    # Bihar has a PM-JAY top-up, Jharkhand does not.
    result = Clinic(index, "jharkhand", llm).ask("Is there a state top-up on PM-JAY?")
    assert isinstance(result, Answer)
    assert "jharkhand" in result.citations[0]
    assert "fifty thousand" not in result.text


def test_filter_has_something_to_hold_back(clinic):
    # Without this the other tests could pass by accident.
    kept_out = clinic.would_have_matched_elsewhere(
        "Janani Suraksha Yojana amount for a rural delivery")
    assert any("jharkhand" in cid for cid in kept_out)


def test_answer_carries_source_date(clinic):
    result = clinic.ask("What is the PM-JAY cover limit per family per year?")
    assert isinstance(result, Answer)
    (chunk_id, updated), = result.sources
    assert updated == "2026-03-14"


# ---- the floor -----------------------------------------------------------

def test_out_of_scope_hands_off(clinic):
    result = clinic.ask("What is the weather in Patna tomorrow?")
    assert isinstance(result, Handoff)
    assert result.reason is Reason.BELOW_FLOOR
    assert result.considered                  # it says what it looked at


def test_an_out_of_scope_question_clears_neither_floor(index):
    for question in ("What is the weather in Patna tomorrow?",
                     "petrol ka rate kya hai aaj"):
        for hit in index.search(question, "bihar", 4):
            assert not hit.cleared, (question, hit.chunk.id)


def test_nothing_in_the_gold_set_is_answered_from_the_wrong_document(index, llm):
    for case in load_gold():
        result = Clinic(index, case["state"], llm).ask(case["q"])
        if case["expect"] == "handoff":
            assert isinstance(result, Handoff), case["q"]
        else:
            assert isinstance(result, Answer), case["q"]
            assert result.citations[0].split("#")[0] == case["expect"], case["q"]


def test_every_figure_in_an_answer_comes_from_its_cited_chunk(index, llm):
    import re
    by_id = {c.id: c for c in index.chunks}
    for case in load_gold():
        result = Clinic(index, case["state"], llm).ask(case["q"])
        if isinstance(result, Answer):
            source = " ".join(by_id[c].text for c in result.citations)
            assert set(re.findall(r"\d+", result.text)) <= set(
                re.findall(r"\d+", source)), case["q"]


# ---- the two retrievers --------------------------------------------------

def test_romanised_hindi_is_answered_by_the_lexical_retriever(index, llm):
    hit = index.search(ROMANISED, "bihar", 4)[0]
    assert hit.chunk.doc == "pmjay-bihar"
    # The encoder puts it below its own floor; BM25 over keys is what carries it.
    assert hit.dense < SentenceEmbedder.min_score
    assert hit.lexical >= index.lexical_floor
    assert hit.retriever == "lexical"
    assert isinstance(Clinic(index, "bihar", llm).ask(ROMANISED), Answer)


def test_the_encoder_alone_would_refuse_it(index):
    # What the hybrid buys, stated as a test: on the dense score alone this
    # question is indistinguishable from an out-of-scope one.
    romanised = max(h.dense for h in index.search(ROMANISED, "bihar", 4))
    petrol = max(h.dense for h in index.search("petrol ka rate kya hai aaj",
                                               "bihar", 4))
    assert romanised < SentenceEmbedder.min_score
    assert abs(romanised - petrol) < 0.02


def test_the_two_floors_are_separately_measured(index):
    # A cosine and a normalised BM25 sum are not the same quantity, so one
    # number could not serve both.
    assert index.lexical_floor != SentenceEmbedder.min_score
    assert 0 < index.lexical_floor < 1


def test_a_hit_reports_the_cosine_and_keeps_the_lexical_score_apart(index):
    hit = index.search("What is the PM-JAY cover limit per family per year?",
                       "bihar", 1)[0]
    assert hit.score == hit.dense           # the cosine is the quoted figure
    assert hit.lexical != hit.dense         # two scales, never averaged
    assert hit.cleared and hit.retriever in ("dense", "lexical")


def test_bm25_scores_stay_inside_the_unit_range(index):
    for question in ("cold chain", "What is the PM-JAY cover limit?",
                     "कोल्ड चेन सामग्री की जाँच कब होती है?", "petrol"):
        for i in range(len(index.chunks)):
            assert 0.0 <= index.lexical.score(question, i) <= 1.0


def test_the_three_ways_a_word_is_typed_share_a_term():
    devanagari = {key for key, _ in keys("आयुष्मान")}
    romanised = {key for key, _ in keys("ayushman")}
    assert devanagari & romanised              # they meet on the Latin key


def test_a_term_absent_from_the_corpus_scores_nothing():
    chunks = load()
    bm25 = Bm25(chunks)
    assert all(bm25.score("xylophone", i) == 0.0 for i in range(len(chunks)))


def test_titles_are_indexed_with_their_text():
    chunks = load()
    bm25 = Bm25(chunks)
    cold = next(i for i, c in enumerate(chunks) if c.title == "Cold chain checks")
    assert "s:cold" in bm25.counts[cold]
    assert "s:cold" in terms("Cold chain checks")
