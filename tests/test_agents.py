import pytest

from clinic_retrieval.agents.drugs import ACCEPT, MARGIN, DrugResolver
from clinic_retrieval.agents.intent import Intent, plan, read
from clinic_retrieval.agents.orchestrate import Orchestrator
from clinic_retrieval.analytics import Log, drug_scores, load_drug_gold
from clinic_retrieval.core import Answer, Clarify, Handoff, Reason
from clinic_retrieval.llm import Llm
from clinic_retrieval.session import Sessions
from tests.conftest import FakeLlm


@pytest.fixture(scope="module")
def resolver():
    return DrugResolver()


# ---- resolving a drug name -----------------------------------------------

@pytest.mark.parametrize("question, code", [
    ("is amoxicillin available", "D-AMX-125"),          # listed name
    ("amoxycilin stock", "D-AMX-125"),                  # misspelling
    ("amoxibest hai kya", "D-AMX-125"),                 # brand name
    ("क्या ऑक्सीटोसिन है", "D-OXY-010"),                  # Devanagari
    ("oxytocine kitna bacha hai", "D-OXY-010"),         # romanised, misspelled
    ("oral rehydration salts stock", "D-ORS-001"),      # three-word phrase
    ("iron sucrose stock", "D-FES-100"),                # phrase beats its word
    ("iron tablet hai kya", "D-IFA-060"),               # the other phrase
])
def test_resolves(resolver, question, code):
    found = resolver.resolve(question)
    assert found.outcome == "resolved"
    assert found.best.code == code


@pytest.mark.parametrize("question", ["is iron available", "आयरन है क्या"])
def test_a_shared_word_is_ambiguous_rather_than_first_wins(resolver, question):
    found = resolver.resolve(question)
    assert found.outcome == "ambiguous"
    assert len(found.candidates) == 2


@pytest.mark.parametrize("question", [
    "is insulin available", "is aspirin in stock", "petrol ka rate kya hai aaj",
    "What is the PM-JAY cover limit?",
])
def test_unresolved(resolver, question):
    assert resolver.resolve(question).outcome == "unresolved"


def test_nothing_resolves_to_the_wrong_drug_at_the_chosen_threshold():
    # The constraint that picked the threshold. A misspelling matched to the
    # wrong drug is the one failure here worse than asking.
    assert drug_scores(ACCEPT, MARGIN).misresolved == 0


def test_the_whole_drug_gold_set_lands_where_it_should():
    gold = load_drug_gold()
    tally = drug_scores(ACCEPT, MARGIN, gold)
    assert tally.resolved + tally.asked + tally.held == len(gold)


def test_nothing_resolves_wrongly_anywhere_in_the_usable_band():
    for accept in (0.84, 0.86, 0.88):
        for margin in (0.03, 0.05, 0.08):
            assert drug_scores(accept, margin).misresolved == 0


def test_a_lower_threshold_is_what_starts_guessing():
    loose = drug_scores(0.80, MARGIN)
    assert loose.leaked > drug_scores(ACCEPT, MARGIN).leaked


def test_adding_the_full_formulary_does_not_change_query_cost(resolver):
    # Lookup is blocked by trigram, so the candidate set stays small.
    assert len(resolver._fuzzy("amoxycilin")) < len(resolver.drugs)


# ---- reading the question ------------------------------------------------

@pytest.mark.parametrize("label", ["inventory", "policy", "clinic"])
def test_a_label_the_model_returns_is_used(label, llm):
    llm.intents["anything"] = {"intent": label, "drug": "iron sucrose"}
    found = read("anything", llm)
    assert found == Intent(label, "iron sucrose")


@pytest.mark.parametrize("reply", [
    {"intent": "stock_check"},         # not one of the three
    {"drug": "amoxicillin"},           # no label at all
    {},
])
def test_a_label_the_system_does_not_have_is_not_guessed(reply, llm):
    llm.intents["q"] = reply
    assert read("q", llm) is None


def test_an_unreachable_model_reads_nothing():
    assert read("is amoxicillin available", FakeLlm(up=False)) is None


@pytest.mark.parametrize("label, outcome, agents", [
    ("inventory", "resolved", ("stock",)),
    ("inventory", "ambiguous", ("clarify",)),
    ("inventory", "unresolved", ("escalation",)),
    ("policy", "resolved", ("docs",)),         # a scheme question that names a drug
    ("policy", "unresolved", ("docs",)),
    ("clinic", "resolved", ("stock", "docs")),
    ("clinic", "unresolved", ("docs",)),
])
def test_the_branch_follows_the_label_and_the_resolution(
        resolver, label, outcome, agents):
    # plan is pure: a label and an already-resolved drug in, a branch out.
    question = {"resolved": "is amoxicillin available",
                "ambiguous": "is iron available",
                "unresolved": "is insulin available"}[outcome]
    found = resolver.resolve(question)
    assert found.outcome == outcome
    assert plan(Intent(label), found).agents == agents


# ---- the whole pipeline --------------------------------------------------

def test_stock_question_is_not_answered_from_documents(index, llm):
    # The corpus says what to do about a stock-out. It cannot say whether this
    # clinic has the drug, so the document agent must not be the one to answer.
    response = Orchestrator(index, "bihar", llm=llm).handle("is amoxicillin available")
    assert [s.agent for s in response.trace] == [
        "llm:intent", "resolver", "router", "stock", "verifier"]
    assert isinstance(response.result, Answer)
    assert "BR-CL-114" in response.result.text
    assert llm.calls == ["intent"]          # no generation call for a table lookup


def test_unlisted_drug_escalates_instead_of_retrieving(index, llm):
    response = Orchestrator(index, "bihar", llm=llm).handle("is insulin available")
    assert isinstance(response.result, Handoff)
    assert response.result.reason is Reason.NOT_IN_FORMULARY
    assert "docs" not in [s.agent for s in response.trace]


def test_an_unresolved_name_is_recorded_for_the_formulary_owner(index, llm):
    log = Log()
    Orchestrator(index, "bihar", log=log, llm=llm).handle("is insulin available")
    assert log.unresolved_counts()[0][0] == "is insulin available"


def test_an_ambiguous_name_asks_instead_of_guessing(index, llm):
    response = Orchestrator(index, "bihar", llm=llm).handle("is iron available")
    result = response.result
    assert isinstance(result, Clarify)
    assert result.reason is Reason.AMBIGUOUS_DRUG
    assert len(result.options) == 2
    assert "docs" not in [s.agent for s in response.trace]


def test_a_known_drug_this_clinic_does_not_carry_is_answered(index, llm):
    """Three outcomes, not two: unknown drug, known and stocked, known and not
    carried here. The third is an answer, with the indent procedure attached."""
    response = Orchestrator(index, "bihar", llm=llm).handle("is zinc available")
    result = response.result
    assert isinstance(result, Answer)
    assert "no stock record" in result.text
    assert "raise an indent" in result.text.lower()
    assert [s.agent for s in response.trace] == [
        "llm:intent", "resolver", "router", "stock", "router", "docs", "verifier"]
    refs = [ref for ref, _ in result.sources]
    assert refs == ["stock:BR-CL-114", "drugs#1"]


def test_a_misspelled_drug_still_reaches_its_stock_record(index, llm):
    response = Orchestrator(index, "bihar", llm=llm).handle("amoxycilin kitna bacha hai")
    assert isinstance(response.result, Answer)
    assert "D-AMX-125" in response.result.text


def test_a_drug_name_the_model_extracts_is_what_gets_resolved(index, llm):
    # The question says nothing a resolver could match; the label carries it.
    llm.intents["the usual one for coughs in children"] = {
        "intent": "inventory", "drug": "amoxicillin"}
    response = Orchestrator(index, "bihar", llm=llm).handle(
        "the usual one for coughs in children")
    assert isinstance(response.result, Answer)
    assert "D-AMX-125" in response.result.text


def test_composite_answer_carries_both_sources(index, llm):
    response = Orchestrator(index, "bihar", llm=llm).handle(
        "amoxicillin stock mein nahi hai to kya karna chahiye")
    assert [s.agent for s in response.trace] == [
        "llm:intent", "resolver", "router", "stock", "docs", "verifier"]
    assert isinstance(response.result, Answer)
    kinds = [src for src, _ in response.result.sources]
    assert "stock:BR-CL-114" in kinds
    assert any(src.startswith("drugs#") for src in kinds)


def test_same_stock_question_differs_by_clinic(index, llm):
    bihar = Orchestrator(index, "bihar", llm=llm).handle("is amoxicillin available")
    jharkhand = Orchestrator(index, "jharkhand", llm=llm).handle(
        "is amoxicillin available")
    assert isinstance(bihar.result, Answer) and isinstance(jharkhand.result, Answer)
    assert "none on hand" in bihar.result.text
    assert "90 on hand" in jharkhand.result.text


def test_verifier_catches_a_cross_state_citation(index, llm):
    """The document agent filters by state, but once two agents compose one
    answer no single agent owns all of it."""
    orch = Orchestrator(index, "bihar", llm=llm)
    wrong = next(c for c in index.chunks if c.state == "jharkhand")
    orch.docs.ask = lambda q: Answer(
        text=wrong.text, citations=[wrong.id],
        sources=[(wrong.id, wrong.updated)], state="bihar", score=1.2,
    )
    response = orch.handle("What is the Janani Suraksha Yojana amount?")
    assert isinstance(response.result, Handoff)
    assert response.result.reason is Reason.WRONG_STATE
    assert [s.agent for s in response.trace][-2:] == ["verifier", "escalation"]


def test_handoff_still_wins_when_no_agent_has_an_answer(index, llm):
    response = Orchestrator(index, "bihar", llm=llm).handle(
        "Who won the cricket match?")
    assert isinstance(response.result, Handoff)
    assert [s.agent for s in response.trace] == [
        "llm:intent", "resolver", "router", "docs", "escalation"]


def test_asking_for_a_person_spends_no_model_call(index, llm):
    response = Orchestrator(index, "bihar", llm=llm).handle(
        "I want to speak to a human about my claim")
    assert isinstance(response.result, Handoff)
    assert response.result.reason is Reason.OUT_OF_SCOPE
    assert llm.calls == []


# ---- what happens when the model misbehaves ------------------------------

def test_a_model_that_is_down_escalates_rather_than_answering(index):
    response = Orchestrator(index, "bihar", llm=FakeLlm(up=False)).handle(
        "What is the PM-JAY cover limit per family per year?")
    assert isinstance(response.result, Handoff)
    assert response.result.reason is Reason.MODEL_UNAVAILABLE


def test_a_model_that_declines_is_not_overridden(index, llm):
    # Retrieval clears the floor and the model still says the sources do not
    # answer it. That is its own reason, not a retrieval miss.
    orch = Orchestrator(index, "bihar", llm=llm)
    orch.docs.llm = FakeLlm()
    orch.docs.llm.json = lambda s, u, max_tokens=300: {"insufficient": True}
    response = orch.handle("What is the PM-JAY cover limit per family per year?")
    assert isinstance(response.result, Handoff)
    assert response.result.reason is Reason.MODEL_ABSTAINED


def test_a_citation_the_model_was_not_shown_is_refused(index, llm):
    orch = Orchestrator(index, "bihar", llm=llm)
    orch.docs.llm = FakeLlm()
    orch.docs.llm.json = lambda s, u, max_tokens=300: {
        "answer": "Cover is five lakh rupees.", "sources": [9]}
    response = orch.handle("What is the PM-JAY cover limit per family per year?")
    assert isinstance(response.result, Handoff)
    assert response.result.reason is Reason.UNGROUNDED_CITATION


def test_a_figure_the_sources_do_not_contain_is_blocked(index, llm):
    orch = Orchestrator(index, "bihar", llm=llm)
    orch.docs.llm = FakeLlm()
    orch.docs.llm.json = lambda s, u, max_tokens=300: {
        "answer": "The cover limit is 700000 rupees per family.", "sources": [1]}
    response = orch.handle("What is the PM-JAY cover limit per family per year?")
    assert isinstance(response.result, Handoff)
    assert response.result.reason is Reason.UNGROUNDED_CITATION
    assert "700000" in response.trace[-2].detail


def test_a_dose_the_model_writes_is_blocked(index, llm):
    orch = Orchestrator(index, "bihar", llm=llm)
    orch.docs.llm = FakeLlm()
    orch.docs.llm.json = lambda s, u, max_tokens=300: {
        "answer": "Give 500 mg twice daily.", "sources": [1]}
    response = orch.handle("What happens to records when there is no connectivity?")
    assert isinstance(response.result, Handoff)
    assert response.result.reason is Reason.CLINICAL_ADVICE


# ---- tools ---------------------------------------------------------------

def test_every_capability_is_a_declared_tool(index, llm):
    toolbox = Orchestrator(index, "bihar", llm=llm).tools
    assert set(toolbox.tools) == {"resolve_drug", "check_stock",
                                  "answer_from_documents"}
    for tool in toolbox.tools.values():
        assert tool.description and tool.reads
        assert tool.schema()["parameters"]["required"]


def test_tool_calls_are_recorded_with_their_arguments(index, llm):
    orch = Orchestrator(index, "bihar", llm=llm)
    orch.handle("is amoxicillin available")
    calls = orch.tools.calls
    assert [c.name for c in calls] == ["resolve_drug", "check_stock"]
    assert calls[1].args == {"code": "D-AMX-125"}
    assert "on hand" in calls[1].summary


def test_each_tool_declares_the_data_it_can_see(index, llm):
    reads = dict(Orchestrator(index, "bihar", llm=llm).tools.permissions())
    assert reads["check_stock"] == "this clinic's stock count"
    assert reads["resolve_drug"] == "national formulary"


def test_the_log_records_the_intent_and_which_tools_ran(index, llm):
    log = Log()
    Orchestrator(index, "bihar", log=log, llm=llm).handle("is amoxicillin available")
    assert log.records[-1].tools == ("resolve_drug", "check_stock")
    assert log.records[-1].intent == "inventory"
    assert log.records[-1].actor == "anonymous"


# ---- memory --------------------------------------------------------------

def test_a_clarification_is_answered_on_the_next_turn(index, llm):
    orch = Orchestrator(index, "bihar", llm=llm)
    sessions = Sessions()
    session = sessions.start("bihar")

    first = orch.handle("is iron available", session=session)
    assert isinstance(first.result, Clarify)
    assert session.pending.kind == "drug_choice"

    second = orch.handle("iron sucrose", session=session)
    assert isinstance(second.result, Answer)
    assert "D-FES-100" in second.result.text
    assert session.pending is None          # asked once, not again


def test_answering_a_choice_spends_no_model_call(index, llm):
    orch = Orchestrator(index, "bihar", llm=llm)
    session = Sessions().start("bihar")
    first = orch.handle("is iron available", session=session)
    wanted = first.result.options[1]
    llm.calls.clear()

    second = orch.handle("2", session=session)
    assert isinstance(second.result, Answer)
    assert wanted.split()[0].lower() in second.result.text.lower()
    assert llm.calls == []


def test_an_unrelated_reply_starts_over_instead_of_guessing(index, llm):
    orch = Orchestrator(index, "bihar", llm=llm)
    session = Sessions().start("bihar")
    orch.handle("is iron available", session=session)

    after = orch.handle("What is the PM-JAY cover limit per family per year?",
                        session=session)
    assert isinstance(after.result, Answer)
    assert after.result.citations[0].startswith("pmjay-bihar")
    assert session.pending is None


def test_nothing_is_remembered_without_an_outstanding_question(index, llm):
    sessions = Sessions()
    session = sessions.start("bihar")
    Orchestrator(index, "bihar", llm=llm).handle("is amoxicillin available",
                                                 session=session)
    assert sessions.forget_unless_pending(session) is None
    assert len(sessions) == 0


def test_a_session_does_not_reach_another_clinic():
    sessions = Sessions()
    session = sessions.start("bihar")
    assert sessions.get(session.id, "jharkhand") is None
    assert sessions.get(session.id, "bihar") is session


def test_sessions_expire():
    sessions = Sessions(ttl=60)
    session = sessions.start("bihar", now=0.0)
    assert sessions.get(session.id, "bihar", now=30.0) is session
    assert sessions.get(session.id, "bihar", now=120.0) is None


# ---- reading what the model sent back ------------------------------------

def test_json_is_recovered_from_a_model_that_wraps_it_in_prose():
    # Small instruct models do this often enough to plan for.
    client = Llm()
    client.chat = lambda s, u, max_tokens=300: (
        'Sure! Here is the label:\n{"intent": "policy", "drug": null}\nHope that helps.')
    assert client.json("s", "u") == {"intent": "policy", "drug": None}


@pytest.mark.parametrize("reply", ["", "I cannot help with that.", "[1, 2]", None])
def test_a_reply_with_no_object_in_it_is_treated_as_no_reply(reply):
    client = Llm()
    client.chat = lambda s, u, max_tokens=300: reply
    assert client.json("s", "u") is None


def test_an_unreachable_endpoint_answers_nothing_rather_than_raising():
    client = Llm(base_url="http://127.0.0.1:9/v1", timeout=0.5)
    assert client.json("s", "u") is None
    assert client.reachable() is False
