"""BM25 over transliteration keys.

Plain BM25 over words does not work here. A clinic types Hindi on an English
keyboard, and "kitna bacha hai" shares no token with "स्टॉक" or with
"remaining stock". So every word is expanded to three terms, its surface form,
its normalised Latin form and its consonant skeleton, and all three are
indexed: the three ways one word gets typed meet in the same posting list.

The query weights those three (1.0, 0.6, 0.5) so a surface match outranks a
skeleton collision. Documents do not, because a weight on a posting is a
ranking signal with no meaning.

Scores are divided by the best score the query could have achieved, which is
what makes one floor usable across questions of different lengths.
"""
from __future__ import annotations

import math
from collections import Counter

from clinic_retrieval.core import Chunk
from clinic_retrieval.lang import keys, words

K1 = 1.2        # term-frequency saturation
B = 0.75        # length normalisation


def terms(text: str) -> list[str]:
    """Every indexable term in a piece of text, surface forms included."""
    out: list[str] = []
    for word in words(text):
        out += [key for key, _ in keys(word)]
    return out


def weighted(text: str) -> dict[str, float]:
    """Query side: one entry per distinct term, at its highest weight."""
    out: dict[str, float] = {}
    for word in words(text):
        for key, weight in keys(word):
            out[key] = max(out.get(key, 0.0), weight)
    return out


class Bm25:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.counts = [Counter(terms(f"{c.title}. {c.text}")) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.df: Counter[str] = Counter()
        for counted in self.counts:
            self.df.update(counted.keys())
        self.n = len(chunks)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: str, i: int) -> float:
        """Between 0 and 1: the BM25 sum over the ceiling the query could reach."""
        asked = weighted(query)
        if not asked:
            return 0.0
        ceiling = sum(w * self.idf(t) * (K1 + 1) for t, w in asked.items())
        if ceiling <= 0:
            return 0.0
        counted, length = self.counts[i], self.lengths[i]
        total = 0.0
        for term, weight in asked.items():
            tf = counted.get(term, 0)
            if not tf:
                continue
            denominator = tf + K1 * (1 - B + B * length / (self.average or 1))
            total += weight * self.idf(term) * (tf * (K1 + 1)) / denominator
        return total / ceiling
