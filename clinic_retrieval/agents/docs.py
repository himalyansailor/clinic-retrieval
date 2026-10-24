"""The document agent: retrieve, check the floor, generate, check grounding.

Four steps in that order, and the last two exist because the third one is a
language model. Retrieval decides whether there is anything to say; the model
decides how to say it; the grounding check decides whether what came back is
allowed out.
"""
from __future__ import annotations

from clinic_retrieval.core import Answer, Handoff, Reason, Result
from clinic_retrieval.generation import write
from clinic_retrieval.llm import Llm
from clinic_retrieval.retrieval.index import Hybrid


class Clinic:
    def __init__(self, index: Hybrid, state: str, llm: Llm | None = None,
                 top_k: int = 4):
        self.index = index
        self.state = state.lower()
        self.llm = llm or Llm()
        self.top_k = top_k

    def ask(self, question: str) -> Result:
        hits = self.index.search(question, self.state, self.top_k)
        if not hits:
            return Handoff(Reason.NO_RELEVANT_CONTEXT, self.state)

        # Either retriever clearing its own floor makes a chunk usable. The
        # model is shown only those: padding the context with chunks nothing
        # matched is how a small model ends up answering from the wrong one.
        cleared = [h for h in hits if h.cleared]
        if not cleared:
            best = max(hits, key=lambda h: h.dense)
            return Handoff(
                Reason.BELOW_FLOOR, self.state, best_score=best.dense,
                considered=[h.chunk.id for h in hits],
                note=f"best was {best.dense:.3f} dense and "
                     f"{max(h.lexical for h in hits):.3f} lexical, "
                     f"against floors of {self.index.embedder.min_score} "
                     f"and {self.index.lexical_floor}",
            )

        written = write(question, cleared, self.llm)
        if written is None:
            return Handoff(Reason.MODEL_UNAVAILABLE, self.state,
                           best_score=cleared[0].dense,
                           note=f"no usable reply from {self.llm.name}")
        if written.insufficient:
            return Handoff(Reason.MODEL_ABSTAINED, self.state,
                           best_score=cleared[0].dense,
                           considered=[h.chunk.id for h in cleared],
                           note="retrieval cleared the floor, the model said "
                                "the sources do not answer it")

        # A citation outside the set the model was shown means the sentence came
        # from somewhere else, and there is no way to tell from here where.
        shown = {h.chunk.id for h in cleared}
        if not written.citations or not set(written.citations) <= shown:
            return Handoff(Reason.UNGROUNDED_CITATION, self.state,
                           best_score=cleared[0].dense,
                           note="cited something it was not shown")

        by_id = {h.chunk.id: h for h in cleared}
        return Answer(
            text=written.text,
            citations=list(written.citations),
            sources=[(cid, by_id[cid].chunk.updated) for cid in written.citations],
            state=self.state,
            score=min(by_id[cid].dense for cid in written.citations),
        )

    def would_have_matched_elsewhere(self, question: str) -> list[str]:
        """Diagnostics only: chunk ids the state filter kept out."""
        allowed = {self.index.chunks[i].id for i in self.index.visible(self.state)}
        return [h.chunk.id for h in self.index.search_unfiltered(question, 5)
                if h.chunk.id not in allowed]
