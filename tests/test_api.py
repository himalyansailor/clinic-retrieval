import pytest
from fastapi.testclient import TestClient

from clinic_retrieval import deps
from clinic_retrieval.app import app
from tests.conftest import FakeLlm


@pytest.fixture(scope="module")
def client():
    """The scripted model stands in for the clinic's. Everything else in the
    app is the real thing: the index, the guards, the gate and the log."""
    deps.LLM = FakeLlm()
    deps.get_orchestrator.cache_clear()
    with TestClient(app) as c:
        yield c
    deps.get_orchestrator.cache_clear()


def test_health_reports_a_loaded_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 16
    assert body["floors"] == {"dense": 0.855, "lexical": 0.055}
    assert "bm25" in body["retrieval"]


def test_health_says_whether_the_model_can_be_reached(client):
    body = client.get("/health").json()
    assert body["model"] and body["model_reachable"] is True


def test_a_model_that_is_down_is_an_escalation_not_an_error(client):
    deps.LLM = FakeLlm(up=False)
    deps.get_orchestrator.cache_clear()
    try:
        response = client.post("/ask", json={
            "question": "What is the PM-JAY cover limit per family per year?",
            "state": "bihar"})
        assert response.status_code == 200
        body = response.json()
        assert body["answered"] is False
        assert body["reason"] == "model_unavailable"
        assert client.get("/health").json()["model_reachable"] is False
    finally:
        deps.LLM = FakeLlm()
        deps.get_orchestrator.cache_clear()


def test_ask_returns_an_answer_with_sources(client):
    body = client.post("/ask", json={
        "question": "What is the Janani Suraksha Yojana amount for a rural delivery?",
        "state": "bihar"}).json()
    assert body["answered"] is True
    assert "fourteen hundred" in body["text"]
    assert body["sources"][0]["dated"] == "2026-03-14"
    assert body["route"] == ["docs"]


def test_the_same_question_differs_by_state(client):
    def amount(state):
        return client.post("/ask", json={
            "question": "What is the Janani Suraksha Yojana amount for a rural delivery?",
            "state": state}).json()["text"]
    assert "fourteen hundred" in amount("bihar")
    assert "seventeen hundred" in amount("jharkhand")


def test_stock_question_hits_the_stock_agent(client):
    body = client.post("/ask", json={"question": "is amoxicillin available",
                                     "state": "bihar"}).json()
    assert body["route"] == ["stock"]
    assert body["sources"][0]["ref"] == "stock:BR-CL-114"


def test_composite_question_returns_both_agents_in_the_trace(client):
    body = client.post("/ask", json={
        "question": "amoxicillin stock mein nahi hai to kya karna chahiye",
        "state": "bihar"}).json()
    assert [s["agent"] for s in body["trace"]] == [
        "llm:intent", "resolver", "router", "stock", "docs", "verifier"]


def test_guarded_question_is_refused_without_retrieval(client):
    body = client.post("/ask", json={"question": "what dose should i give",
                                     "state": "bihar"}).json()
    assert body["answered"] is False
    assert body["reason"] == "clinical_advice"
    assert body["route"] == ["guard"]


def test_unknown_state_is_404(client):
    assert client.post("/ask", json={"question": "anything",
                                     "state": "kerala"}).status_code == 404


def test_overlong_question_is_rejected_by_validation(client):
    assert client.post("/ask", json={"question": "x" * 501,
                                     "state": "bihar"}).status_code == 422


def test_live_analytics_are_grouped_by_form(client):
    client.post("/ask", json={"question": "kya ORS stock mein hai", "state": "bihar"})
    client.post("/ask", json={"question": "What is the PM-JAY cover limit?",
                              "state": "bihar"})
    rows = client.get("/analytics/live").json()
    forms = {row["form"] for row in rows}
    assert {"en", "hinglish"} <= forms
    assert all(row["asked"] >= 1 for row in rows)


def test_ragas_endpoint_reports_four_metrics_per_form(client):
    rows = client.get("/analytics/ragas").json()
    assert {row["form"] for row in rows} == {"en", "hi", "hinglish"}
    for row in rows:
        assert row["context_recall"] == 1.0
        assert row["answered"] == row["n"]
        assert row["faithfulness"] > 0.9


def test_root_redirects_to_the_client(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_client_is_served(client):
    body = client.get("/ui/").text
    assert "<title>Clinic retrieval</title>" in body
    assert "app.js" in body


def test_client_assets_are_served(client):
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/styles.css").status_code == 200


def test_health_lists_each_clinic(client):
    clinics = client.get("/health").json()["clinics"]
    assert {c["state"] for c in clinics} == {"bihar", "jharkhand"}
    assert {c["code"] for c in clinics} == {"BR-CL-114", "JH-CL-207"}


def test_ambiguous_drug_comes_back_as_a_question_with_options(client):
    body = client.post("/ask", json={"question": "is iron available",
                                     "state": "bihar"}).json()
    assert body["answered"] is False
    assert body["reason"] == "ambiguous_drug"
    assert body["asks"].startswith("More than one drug")
    assert len(body["options"]) == 2


def test_choosing_an_option_resolves_it(client):
    body = client.post("/ask", json={"question": "is Iron sucrose injection available",
                                     "state": "bihar"}).json()
    assert body["reason"] != "ambiguous_drug"
    assert "D-FES-100" in body["text"]


def test_a_drug_the_clinic_does_not_carry_is_answered_not_refused(client):
    body = client.post("/ask", json={"question": "is zinc available",
                                     "state": "bihar"}).json()
    assert body["answered"] is True
    assert "no stock record" in body["text"]
    assert len(body["sources"]) == 2


def test_unresolved_names_are_listed_for_curation(client):
    client.post("/ask", json={"question": "is insulin available", "state": "bihar"})
    rows = client.get("/analytics/unresolved").json()
    assert any("insulin" in row["question"] for row in rows)


def test_health_counts_the_formulary(client):
    assert client.get("/health").json()["formulary"] == 14


# ---- sessions over HTTP --------------------------------------------------

def test_a_clarification_round_trips_through_a_session(client):
    first = client.post("/ask", json={"question": "is iron available",
                                      "state": "bihar"}).json()
    assert first["reason"] == "ambiguous_drug"
    assert first["session"]

    second = client.post("/ask", json={"question": first["options"][0],
                                       "state": "bihar",
                                       "session": first["session"]}).json()
    assert second["answered"] is True
    assert second["session"] is None          # nothing left outstanding


def test_a_plain_answer_leaves_no_session_behind(client):
    body = client.post("/ask", json={"question": "is amoxicillin available",
                                     "state": "bihar"}).json()
    assert body["answered"] is True
    assert body["session"] is None


def test_a_session_from_another_clinic_is_ignored(client):
    first = client.post("/ask", json={"question": "is iron available",
                                      "state": "bihar"}).json()
    second = client.post("/ask", json={"question": "1", "state": "jharkhand",
                                       "session": first["session"]}).json()
    # Treated as a fresh question in the other clinic, not as an answer here.
    assert second["reason"] != "ambiguous_drug" or second["session"] != first["session"]


# ---- tools ---------------------------------------------------------------

def test_the_tools_endpoint_describes_what_agents_can_reach(client):
    rows = client.get("/tools").json()
    assert {row["name"] for row in rows} == {"resolve_drug", "check_stock",
                                             "answer_from_documents"}
    for row in rows:
        assert row["reads"]
        assert row["parameters"]["type"] == "object"


# ---- access control ------------------------------------------------------

def test_health_reports_the_auth_mode(client):
    assert client.get("/health").json()["auth"] == "open"


def test_with_keys_configured_a_key_is_required(client, monkeypatch):
    from clinic_retrieval import deps, security
    monkeypatch.setattr(deps, "GATE",
                        security.Gate(security.load_keys("k1:br-cl-114:bihar")))

    assert client.post("/ask", json={"question": "is amoxicillin available",
                                     "state": "bihar"}).status_code == 401
    assert client.get("/analytics/live").status_code == 401

    headers = {"x-api-key": "k1"}
    ok = client.post("/ask", json={"question": "is amoxicillin available",
                                   "state": "bihar"}, headers=headers)
    assert ok.status_code == 200
    assert client.get("/analytics/live", headers=headers).status_code == 200


def test_a_key_cannot_reach_another_clinic(client, monkeypatch):
    from clinic_retrieval import deps, security
    monkeypatch.setattr(deps, "GATE",
                        security.Gate(security.load_keys("k1:br-cl-114:bihar")))
    denied = client.post("/ask", json={"question": "is amoxicillin available",
                                       "state": "jharkhand"},
                         headers={"x-api-key": "k1"})
    assert denied.status_code == 403


def test_the_audit_record_names_the_caller(client, monkeypatch):
    from clinic_retrieval import deps, security
    monkeypatch.setattr(deps, "GATE",
                        security.Gate(security.load_keys("k1:br-cl-114:bihar")))
    client.post("/ask", json={"question": "is amoxicillin available",
                              "state": "bihar"}, headers={"x-api-key": "k1"})
    assert deps.LOG.records[-1].actor == "br-cl-114"
