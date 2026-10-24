"""Guardrails on the way in and on the way out."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from clinic_retrieval.core import Reason

WANTS_HUMAN = re.compile(
    r"\b(talk|speak) to (a |an )?(human|person|someone|doctor|supervisor)\b"
    r"|\bcall (me )?back\b|किसी से बात|सुपरवाइज़र से बात",
    re.IGNORECASE,
)

# Something is wrong with the question itself.
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above)|disregard (the|all|any)"
    r"|system prompt|your instructions|you are now|act as if|pretend (to be|you)"
    r"|reveal your|repeat the prompt",
    re.IGNORECASE,
)

# Red flags go to a person immediately. The system does not triage.
_EMERGENCY = re.compile(
    r"bleeding heavily|heavy bleeding|unconscious|not breathing|convulsion"
    r"|fitting|seizure|chest pain"
    r"|बेहोश|दौरा|ख़ून बह|खून बह|सांस नहीं|साँस नहीं",
    re.IGNORECASE,
)

# Dosing and diagnosis are a prescriber's decision, which is also what the
# protocol document says. Asked either way, this system escalates.
_CLINICAL = re.compile(
    r"\bdose\b|\bdosage\b|how (much|many) (should|do) (i|we)\b|\bmg\b|\bml\b"
    r"|prescrib|diagnos|treatment for|should i give|safe for a child"
    r"|खुराक|कितनी दवा दें|इलाज क्या"
    r"|kitni dose|dose kitni|kitna dena",
    re.IGNORECASE,
)

_PHONE = re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")
_AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")

# Dosing language in an answer, which no document here contains.
_DOSE_IN_OUTPUT = re.compile(r"\b\d+\s?(mg|ml|mcg|g)\b|\btake \d+", re.IGNORECASE)

# Figures are compared as values, because a model writes "1200" for a source
# that says "twelve hundred", and "5 lakh" for one that says "five lakh".
_TOKEN = re.compile(r"[0-9०-९][0-9०-९,]*|[a-z]+|[ऀ-ॣ॰-ॿ]+")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_SMALL = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_SMALL |= {w: 10 * n for n, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_SMALL |= {"एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6,
           "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
           "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16, "सत्रह": 17,
           "अठारह": 18, "उन्नीस": 19, "बीस": 20, "पच्चीस": 25, "तीस": 30,
           "चालीस": 40, "पचास": 50, "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90}
_SCALE = {"hundred": 100, "thousand": 1000, "lakh": 10**5, "crore": 10**7,
          "सौ": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 10**5, "करोड़": 10**7}

MAX_CHARS = 500
STALE_AFTER_DAYS = 365


@dataclass(frozen=True)
class Trip:
    guard: str
    action: str          # block, redact or flag
    detail: str


@dataclass(frozen=True)
class Screen:
    text: str
    trips: tuple[Trip, ...] = ()
    blocked: Reason | None = None

    @property
    def flags(self) -> list[str]:
        return [t.guard for t in self.trips if t.action == "flag"]


def screen_input(question: str) -> Screen:
    """Order matters: an emergency is still an emergency if it also asks a dose."""
    if len(question) > MAX_CHARS:
        return Screen(question, (Trip("length", "block", f"{len(question)} chars"),),
                      Reason.UNSAFE_INPUT)

    if _EMERGENCY.search(question):
        return Screen(question, (Trip("emergency", "block", "red flag in the question"),),
                      Reason.EMERGENCY)

    if _INJECTION.search(question):
        return Screen(question, (Trip("injection", "block", "instruction override"),),
                      Reason.UNSAFE_INPUT)

    if _CLINICAL.search(question):
        return Screen(question, (Trip("clinical", "block", "dosing or diagnosis"),),
                      Reason.CLINICAL_ADVICE)

    trips: list[Trip] = []
    text = question
    for name, pattern in (("aadhaar", _AADHAAR), ("phone", _PHONE)):
        if pattern.search(text):
            text = pattern.sub("[redacted]", text)
            trips.append(Trip(name, "redact", "identifier removed before retrieval"))
    return Screen(text, tuple(trips))


def screen_output(answer: str, sources: list[str], dated: list[tuple[str, str]],
                  today: date | None = None) -> Screen:
    """Checks that hold whether the answer was quoted or generated."""
    today = today or date.today()

    if not dated:
        return Screen(answer, (Trip("citation", "block", "no cited source"),),
                      Reason.UNGROUNDED_CITATION)

    if _DOSE_IN_OUTPUT.search(answer):
        return Screen(answer, (Trip("dosing", "block", "dosing language in the answer"),),
                      Reason.CLINICAL_ADVICE)

    if _PHONE.search(answer) or _AADHAAR.search(answer):
        return Screen(answer, (Trip("pii_echo", "block", "identifier in the answer"),),
                      Reason.UNSAFE_INPUT)

    # Every figure in the answer has to come from a source, compared by value.
    # Sources count every number they spell; the answer is held to figures
    # written with a digit or a scale word, because "one of" and एक are prose.
    in_sources: set[int] = set()
    for text in sources:
        in_sources |= figures(text, bare_words=True)
    invented = sorted(figures(answer) - in_sources)
    if invented:
        return Screen(answer, (Trip("numbers", "block", f"not in source: {invented}"),),
                      Reason.UNGROUNDED_CITATION)

    trips: list[Trip] = []
    for source, when in dated:
        age = _age_days(when, today)
        if age is not None and age > STALE_AFTER_DAYS:
            trips.append(Trip("staleness", "flag", f"{source} is {age} days old"))
    return Screen(answer, tuple(trips))


def figures(text: str, bare_words: bool = False) -> set[int]:
    """The numbers a text states, as values: "1,200", "twelve hundred", "बारह
    सौ" and "१२००" are all 1200. Two digit runs never merge, so a date stays
    three numbers. A number spelled with small words alone ("five", "एक")
    counts only when `bare_words` is set."""
    found: set[int] = set()
    total = current = 0
    started, scaled, last = False, False, ""

    def flush() -> None:
        nonlocal total, current, started, scaled, last
        if started and (scaled or bare_words or last == "digit"):
            found.add(total + current)
        total = current = 0
        started, scaled, last = False, False, ""

    for token in _TOKEN.findall(text.lower()):
        if token[0].isdigit():
            flush()
            digits = token.translate(_DEVANAGARI_DIGITS).replace(",", "")
            if digits:
                current, started, last = int(digits), True, "digit"
        elif token in _SMALL:
            value = _SMALL[token]
            # "twenty five" is one number; "five six" and "5 five" are two.
            joins = last == "scale" or (last == "small" and current % 10 == 0
                                        and current >= 20 and value < 10)
            if started and not joins:
                flush()
            current += value
            started, last = True, "small"
        elif token in _SCALE and started:
            scale = _SCALE[token]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            scaled, last = True, "scale"
        elif token == "and" and started:
            continue
        else:
            flush()
    flush()
    return found


def _age_days(when: str, today: date) -> int | None:
    try:
        stamp = date.fromisoformat(when.split()[0])
    except ValueError:
        return None
    return (today - stamp).days
