"""Corpus loading, the state filter, and hybrid search.

One index, two retrievers. Dense embeddings handle paraphrase in English and
Devanagari; BM25 over transliteration keys handles romanised Hindi, exact
scheme names and formulary codes, none of which an encoder is reliable on. A
chunk is a candidate when either retriever clears its own measured floor, and
the ranking is reciprocal rank fusion over the two orderings.

The state filter runs before either retriever scores anything, so another
state's document is never a candidate rather than merely never a winner.
"""
from __future__ import annotations

from pathlib import Path

from clinic_retrieval.core import CORPUS, Chunk, Hit
from clinic_retrieval.retrieval.embed import SentenceEmbedder, cosine
from clinic_retrieval.retrieval.lexical import Bm25

RRF_K = 60          # the usual constant; it flattens the tail, not the top


def _front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    _, block, body = raw.split("---", 2)
    meta = {}
    for line in block.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def load(path: Path = CORPUS) -> list[Chunk]:
    """One chunk per markdown heading."""
    chunks: list[Chunk] = []
    for md in sorted(path.glob("*.md")):
        meta, body = _front_matter(md.read_text(encoding="utf-8"))
        doc = meta.get("doc", md.stem)
        for n, section in enumerate(body.split("\n# ")):
            section = section.strip().lstrip("# ").strip()
            if not section:
                continue
            title, _, text = section.partition("\n")
            text = text.strip()
            if not text:
                continue
            chunks.append(Chunk(
                id=f"{doc}#{n}",
                doc=doc,
                title=title.strip(),
                text=text,
                state=meta.get("state", "all"),
                kind=meta.get("kind", "clinic"),
                updated=meta.get("updated", "unknown"),
            ))
    return chunks


class Hybrid:
    def __init__(self, chunks: list[Chunk], embedder: SentenceEmbedder | None = None,
                 lexical_floor: float = 0.055):
        self.chunks = chunks
        self.embedder = embedder or SentenceEmbedder()
        self.lexical = Bm25(chunks)
        # Measured the same way the dense floor was, in its own space: the
        # highest out-of-scope question reaches 0.011 here and the lowest
        # correct chunk reaches 0.102, so the band is 0.011 to 0.102 and this
        # is the middle of it. scripts/evaluate.py prints the sweep.
        self.lexical_floor = lexical_floor
        self.vectors = [self.embedder.embed_document(f"{c.title}. {c.text}")
                        for c in chunks]

    @classmethod
    def from_corpus(cls, path: Path = CORPUS) -> "Hybrid":
        return cls(load(path))

    @property
    def name(self) -> str:
        return (f"hybrid: {self.embedder.name} at {self.embedder.min_score} "
                f"+ bm25 over transliteration keys at {self.lexical_floor}")

    def visible(self, state: str) -> list[int]:
        """A clinic sees its own state's documents and the national ones."""
        state = state.lower()
        return [i for i, c in enumerate(self.chunks)
                if c.state == "all" or c.state.lower() == state]

    def search(self, query: str, state: str, k: int = 4) -> list[Hit]:
        allowed = self.visible(state)
        if not allowed:
            return []

        q = self.embedder.embed(query)
        dense = {i: cosine(self.vectors[i], q) for i in allowed}
        lexical = {i: self.lexical.score(query, i) for i in allowed}

        # Each retriever contributes its own shortlist. A retriever that
        # matched nothing contributes nothing: fusing its arbitrary ordering of
        # zeroes would reorder the other one's results for no reason.
        shortlists = (
            sorted(allowed, key=lambda i: -dense[i])[:k],
            [i for i in sorted(allowed, key=lambda i: -lexical[i])[:k]
             if lexical[i] > 0],
        )
        fused: dict[int, float] = {}
        for shortlist in shortlists:
            for rank, i in enumerate(shortlist, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)

        order = sorted(fused, key=lambda i: (-fused[i], -dense[i]))
        return [self._hit(i, dense[i], lexical[i]) for i in order[:k]]

    def search_unfiltered(self, query: str, k: int = 4) -> list[Hit]:
        """Diagnostics only: shows what the state filter kept out."""
        q = self.embedder.embed(query)
        hits = [self._hit(i, cosine(self.vectors[i], q),
                          self.lexical.score(query, i))
                for i in range(len(self.chunks))]
        hits.sort(key=lambda h: -h.dense)
        return hits[:k]

    def _hit(self, i: int, dense: float, lexical: float) -> Hit:
        """A chunk is usable when either retriever reached its own floor. The
        floor test is made here, once, so no caller has to know there are two.
        """
        over_dense = dense >= self.embedder.min_score
        over_lexical = lexical >= self.lexical_floor
        which = ("dense" if over_dense else "lexical" if over_lexical else "")
        return Hit(self.chunks[i], dense, lexical,
                   over_dense or over_lexical, which)
