"""Writing the answer.

The model sees the question and the retrieved sources, numbered, and nothing
else: no conversation, no clinic identity, no earlier answer. Numbers rather
than chunk ids, because a 3B instruct model copies "[2]" back reliably and
"pmjay-bihar#1" it does not, and the mapping back to ids happens here where it
can be checked.

Nothing in this file decides whether the answer is usable. It returns what the
model produced, including the model's own refusal, and the checks run after.
"""
from __future__ import annotations

from dataclasses import dataclass

from clinic_retrieval.core import Hit
from clinic_retrieval.lang import detect_form
from clinic_retrieval.llm import Llm

SYSTEM = """You answer questions from staff at rural health clinics in India, \
using only the numbered sources you are given.

Rules:
- Use only facts stated in the sources. Never add a figure, a scheme name or a \
condition that is not written there.
- Write the answer in the language named on the "Answer in" line, even when \
the sources are in another language.
- Two sentences at most. No preamble.
- Never state a drug dose, a diagnosis, or what to give a patient, even if a \
source mentions one.
- A source that says something does not apply, does not exist or is not \
allowed answers the question: say so. That is an answer, not insufficient.
- If the sources do not answer the question, set "insufficient" to true and \
leave the answer empty.

Reply with JSON only, in this shape:
{"answer": "...", "sources": [1], "insufficient": false}"""


@dataclass(frozen=True)
class Written:
    text: str
    citations: tuple[str, ...]
    insufficient: bool = False


# Named per request rather than left to "the language of the question", which
# a 3B model follows unreliably. Hindi typed in English letters is answered in
# English, because that is the script the reader typed in.
LANGUAGE = {"hi": "Hindi, in Devanagari script", "en": "English", "hinglish": "English"}


def context(hits: list[Hit]) -> str:
    return "\n".join(f"[{n}] {h.chunk.title}. {h.chunk.text}"
                     for n, h in enumerate(hits, start=1))


def write(question: str, hits: list[Hit], llm: Llm) -> Written | None:
    """None means the model could not be reached or did not return an object.
    That is a different outcome from the model declining to answer."""
    reply = llm.json(SYSTEM, prompt(question, hits))
    if reply is None:
        return None

    if reply.get("insufficient") is True:
        return Written("", (), True)

    text = str(reply.get("answer") or "").strip()
    if not text:
        return Written("", (), True)

    return Written(text, _cited(reply.get("sources"), hits))


def prompt(question: str, hits: list[Hit]) -> str:
    return (f"Question: {question}\nAnswer in: {LANGUAGE[detect_form(question)]}"
            f"\n\nSources:\n{context(hits)}")


def _cited(numbers: object, hits: list[Hit]) -> tuple[str, ...]:
    """Source numbers back to chunk ids. Anything that is not a number of a
    source it was given is dropped here, and the grounding check downstream
    then sees an answer with no citation and refuses it."""
    if not isinstance(numbers, list):
        numbers = [numbers]
    out: list[str] = []
    for item in numbers:
        try:
            n = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(hits) and hits[n - 1].chunk.id not in out:
            out.append(hits[n - 1].chunk.id)
    return tuple(out)
