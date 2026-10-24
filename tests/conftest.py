"""A scripted model, so the tests measure the pipeline and not a 3B's mood.

The real model is reached over HTTP and is not part of a test run. This stand-in
answers the two prompts the system sends: it labels a question from a keyword
table, and it writes an answer by quoting the first sentence of the first source
it was given and citing it. Both are deterministic, so a failing test is a
change in the pipeline.

For the answer it quotes the whole of the first source back. Choosing which
sentence matters is the real model's job and cannot be faked without writing a
second retriever, so the double does the one thing that keeps the test honest:
it stays inside the source it was given. Everything downstream of generation,
the citation check, the digit guard, the dosing guard and the state verifier,
then runs against text that is grounded by construction, so a failure there is
a failure in the pipeline.
"""
from __future__ import annotations

import re

import pytest

from clinic_retrieval.agents import intent as intent_module
from clinic_retrieval.retrieval.index import Hybrid

_AVAILABILITY = re.compile(
    r"available|in stock|stock mein|on hand|how many|bacha|hai kya"
    r"|उपलब्ध|स्टॉक में|कितनी बची", re.IGNORECASE)
_PROCEDURE = re.compile(
    r"what should|what do|what happens|procedure|protocol|how often|when does"
    r"|karna chahiye|kya karna|क्या कर|कब होती|कब रेफर|क्या होता", re.IGNORECASE)
_SCHEME = re.compile(
    r"pm-?jay|ayushman|jsy|janani|yojana|yojna|suraksha|cover|claim|scheme"
    r"|योजना|कवर|पीएम-जय|आयुष्मान|सुरक्षा", re.IGNORECASE)



class FakeLlm:
    def __init__(self, intents: dict[str, dict] | None = None,
                 answers: dict[str, dict] | None = None, up: bool = True):
        self.intents = intents or {}        # question -> reply, overriding the table
        self.answers = answers or {}
        self.up = up
        self.calls: list[str] = []

    name = "scripted"

    def reachable(self) -> bool:
        return self.up

    def json(self, system: str, user: str, max_tokens: int = 300) -> dict | None:
        if not self.up:
            return None
        if system is intent_module.SYSTEM:
            self.calls.append("intent")
            return (self.intents[user] if user in self.intents
                    else self._label(user))
        self.calls.append("generate")
        return self.answers[user] if user in self.answers else self._write(user)

    def _label(self, question: str) -> dict:
        if _PROCEDURE.search(question):
            label = "clinic"
        elif _AVAILABILITY.search(question):
            label = "inventory"
        elif _SCHEME.search(question):
            label = "policy"
        else:
            label = "clinic"
        return {"intent": label, "drug": None}

    def _write(self, prompt: str) -> dict:
        _, _, rest = prompt.partition("\n\nSources:\n[1] ")
        first = rest.split("\n[2]")[0]
        _, _, body = first.partition(". ")          # drop the chunk title
        return {"answer": body.strip(), "sources": [1], "insufficient": False}


@pytest.fixture
def llm():
    return FakeLlm()


@pytest.fixture(scope="module")
def index():
    return Hybrid.from_corpus()
