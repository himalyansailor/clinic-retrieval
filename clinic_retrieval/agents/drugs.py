"""What drug was meant, and whether this clinic has it.

The formulary is national and says what a drug is. The stock table is per
clinic and says how many are on the shelf. Keeping them apart is what lets
the system tell "we do not know that drug" from "we know it and this
clinic has no record", which are different answers to a health worker."""
from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from clinic_retrieval.core import FORMULARY, STOCK
from clinic_retrieval.lang import latin_key, skeleton, words

_GENERIC = {"tablet", "tablets", "injection", "sachet", "salts", "oral", "solution",
            "dispersible", "acid", "and", "vaccine", "syrup", "suspension", "drops"}

# Measured with clinic.analytics.evaluate.drug_sweep over data/gold/drugs.jsonl.
# Nothing resolves to the wrong drug at any setting; what moves is how often it
# asks or refuses. Below 0.84 it starts resolving a term that should have been
# a question, and above 0.88 it stops recognising misspellings, because a
# transliteration-key hit scores 0.88. So the band is 0.84 to 0.88 and this is
# the middle. The margin band is 0.03 to 0.08; past that it over-asks.
ACCEPT = 0.86
MARGIN = 0.05

# A whole listed phrase is certain. One word out of a phrase is not, because
# several drugs can share it: "iron" belongs to two, and that has to surface as
# a question rather than as whichever drug was loaded first.
_PHRASE = 1.00
_PART = 0.90
_KEY = 0.88

# Transliteration keys below this length are not indexed. They exist to survive
# a misspelling or a change of script, and a two-letter key does neither: the
# formulary lists MR, TT and IFA as abbreviations, whose keys collide with
# ordinary Hindi words ("मेरे" skeletonises to "mr"). Those abbreviations still
# match exactly through the part index, which is where an abbreviation belongs.
_MIN_KEY = 3


@dataclass(frozen=True)
class Drug:
    code: str
    generic: str
    cold_chain: bool
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    code: str
    name: str
    score: float
    how: str           # alias, key, or fuzzy
    term: str          # what in the question matched


@dataclass(frozen=True)
class Resolution:
    outcome: str       # resolved, ambiguous, unresolved
    best: Candidate | None
    candidates: tuple[Candidate, ...]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.candidates]


def load_formulary(path: Path = FORMULARY) -> list[Drug]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Drug(code=d["code"], generic=d["generic"], cold_chain=d["cold_chain"],
             aliases=tuple(d["aliases"]))
        for d in raw["drugs"]
    ]


def _trigrams(text: str) -> set[str]:
    padded = f"  {text} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


class DrugResolver:
    """Built once per process. Lookup is blocked by trigram, so adding the
    full national formulary changes the index size and not the query cost."""

    def __init__(self, drugs: list[Drug] | None = None,
                 accept: float = ACCEPT, margin: float = MARGIN):
        self.drugs = drugs if drugs is not None else load_formulary()
        self.by_code = {d.code: d for d in self.drugs}
        self.accept = accept
        self.margin = margin

        self._phrase: dict[str, set[str]] = {}
        self._part: dict[str, set[str]] = {}
        self._key: dict[str, set[str]] = {}
        self._trigram: dict[str, set[str]] = {}
        self.longest = 1

        for drug in self.drugs:
            for phrase in (drug.generic, *drug.aliases):
                norm = " ".join(phrase.lower().split())
                self._phrase.setdefault(norm, set()).add(drug.code)
                self.longest = max(self.longest, len(norm.split()))
                parts = [p for p in norm.split() if p not in _GENERIC]
                for part in parts:
                    self._part.setdefault(part, set()).add(drug.code)
                    for key in (skeleton(part), latin_key(part)):
                        if len(key) >= _MIN_KEY:
                            self._key.setdefault(key, set()).add(drug.code)
                    for gram in _trigrams(part):
                        self._trigram.setdefault(gram, set()).add(drug.code)

    def terms(self, question: str) -> list[str]:
        """Windows up to the longest listed phrase, so a phrase can outrank one
        of its own words. Windows are built before generic words are dropped:
        "iron tablet" is a listed phrase while "iron" alone is two drugs, and
        "oral rehydration salts" is three words of which two are generic."""
        found = [w.lower() for w in words(question)]
        out = [w for w in found if w not in _GENERIC]
        for size in range(2, self.longest + 1):
            out += [" ".join(found[i:i + size])
                    for i in range(len(found) - size + 1)]
        return out

    def _fuzzy(self, term: str) -> list[tuple[str, float]]:
        grams = _trigrams(term)
        blocked: set[str] = set()
        for gram in grams:
            blocked |= self._trigram.get(gram, set())
        scored = []
        for code in blocked:
            drug = self.by_code[code]
            best = max(
                SequenceMatcher(None, term, phrase.lower()).ratio()
                for phrase in (drug.generic, *drug.aliases)
            )
            scored.append((code, best))
        return scored

    def resolve(self, question: str) -> Resolution:
        best: dict[str, Candidate] = {}

        def offer(code: str, score: float, how: str, term: str) -> None:
            drug = self.by_code[code]
            current = best.get(code)
            if current is None or score > current.score:
                best[code] = Candidate(code, drug.generic, score, how, term)

        for term in self.terms(question):
            if term in self._phrase:
                for code in self._phrase[term]:
                    offer(code, _PHRASE, "phrase", term)
                continue
            if term in self._part:
                for code in self._part[term]:
                    offer(code, _PART, "part", term)
                continue
            keyed = set()
            for key in (skeleton(term), latin_key(term)):
                if len(key) >= _MIN_KEY:
                    keyed |= self._key.get(key, set())
            if keyed:
                for code in keyed:
                    offer(code, _KEY, "key", term)
                continue
            for code, score in self._fuzzy(term):
                if score >= self.accept - self.margin:
                    offer(code, score, "fuzzy", term)

        ranked = sorted(best.values(), key=lambda c: (-c.score, c.code))
        if not ranked or ranked[0].score < self.accept:
            return Resolution("unresolved", None, tuple(ranked[:3]))
        close = [c for c in ranked if ranked[0].score - c.score < self.margin]
        if len(close) > 1:
            return Resolution("ambiguous", None, tuple(close[:3]))
        return Resolution("resolved", ranked[0], (ranked[0],))


@dataclass(frozen=True)
class StockRecord:
    drug: Drug
    on_hand: int
    clinic: str
    counted: str

    @property
    def available(self) -> bool:
        return self.on_hand > 0


class StockTable:
    def __init__(self, state: str, resolver: DrugResolver | None = None,
                 path: Path = STOCK):
        self.state = state.lower()
        self.resolver = resolver or DrugResolver()
        raw = json.loads(path.read_text(encoding="utf-8"))

        self.clinic = ""
        self.counted = ""
        self._on_hand: dict[str, int] = {}
        for entry in raw["clinics"]:
            if entry["state"].lower() == self.state:
                self.clinic = entry["code"]
                self.counted = entry["counted"]
                self._on_hand = dict(entry["stock"])

    @property
    def drugs(self) -> list[StockRecord]:
        return [self.record(code) for code in sorted(self._on_hand)]

    def record(self, code: str) -> StockRecord | None:
        """None means the formulary knows this drug and this clinic does not
        carry it, which is an answer rather than a failure."""
        if code not in self._on_hand:
            return None
        return StockRecord(drug=self.resolver.by_code[code],
                           on_hand=self._on_hand[code],
                           clinic=self.clinic, counted=self.counted)

    def line(self, record: StockRecord) -> str:
        count = f"{record.on_hand} on hand" if record.available else "none on hand"
        cold = ", cold chain item" if record.drug.cold_chain else ""
        return (f"{record.drug.generic} ({record.drug.code}): {count} at "
                f"{record.clinic}, counted {record.counted}{cold}.")

    def not_carried(self, code: str) -> str:
        drug = self.resolver.by_code[code]
        return (f"{drug.generic} ({drug.code}) is in the national formulary and "
                f"{self.clinic} has no stock record for it as of {self.counted}.")
