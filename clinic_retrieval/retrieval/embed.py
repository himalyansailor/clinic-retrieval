"""The dense half of retrieval.

multilingual-e5-small, 384 dimensions, ~120MB on disk, CPU inference. Chosen
over a larger multilingual encoder because a clinic box is capped at 16GB
with a generative model already resident; the measured cost of the choice is in
TRADEOFFS.md.

The floor travels with the model. A cosine of 0.83 means something different in
every embedding space, so the number lives next to the weights that produced it
and `retrieval.index` reads it from here.
"""
from __future__ import annotations

from functools import lru_cache

MODEL = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class SentenceEmbedder:
    # Measured on the gold set, not carried over from another corpus. The
    # highest an out-of-scope question reaches here is 0.840 and the lowest
    # correct English or Devanagari answer is 0.871, so the band is 0.840 to
    # 0.871 and this is the middle of it. Romanised Hindi lands at 0.797 to
    # 0.839, inside the out-of-scope range, which is the whole reason the
    # lexical retriever exists: no value here separates it.
    min_score = 0.855
    name = f"{MODEL} (cpu)"

    def __init__(self, model: str = MODEL):
        self.model_name = model

    # e5 was trained with these prefixes and loses accuracy without them.
    def embed(self, text: str) -> list[float]:
        return self._encode(f"query: {text}")

    def embed_document(self, text: str) -> list[float]:
        return self._encode(f"passage: {text}")

    def _encode(self, text: str) -> list[float]:
        vector = _model(self.model_name).encode(
            text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
