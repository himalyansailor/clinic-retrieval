# Decisions

> Module paths are relative to [`clinic_retrieval/`](../clinic_retrieval); `tests/`, `scripts/` and `data/` are at the repo root.

Every technical choice in this build, what else was on the table, and what
would change my mind. Entries marked **reasoning only** are decisions I made
without implementing, because they sit outside the slice that was worth
building first.

What a deadline would change is kept out of this file. That is in
[TRADEOFFS.md](TRADEOFFS.md), so the engineering argument and the deadline
argument can be read separately and disagreed with separately.

The problem: a multi-agent system answering questions on health schemes, drug
availability and clinic protocols, in Hindi and English, across three states,
on 2 GPUs and 16GB per clinic with unreliable connectivity.

---

## 1. Agent boundaries, and what decides the branch

**Chosen:** a small model reads the question into one of three intents, a pure
function turns that into a branch over two specialist agents, and a verifier
and an escalation path sit behind them (`agents/intent.py`,
`agents/orchestrate.py`).

| Option | Verdict |
|---|---|
| One agent, one retrieval path | Rejected, and this is what the first version was. It cannot answer "is amoxicillin in stock here", because that is not in any document. Asked anyway, it returns the stock-out procedure, which reads like an answer and is not one. |
| Router plus a document agent and a stock agent | **Used.** The split follows the data, not the topic: one agent reads versioned documents that change every few months, the other reads the clinic's own stock count from this morning. Different source, different staleness, different failure mode. |
| One agent per domain: schemes, drugs, protocols | Rejected. Schemes and protocols are the same tool over the same index with the same contract. Splitting there buys nothing and costs a handoff, and every handoff loses context and doubles the cost. |
| LLM intent classification | **Used**, and this replaced a page of regular expressions. Those rules needed a pattern per phrasing per language per script, and the list only ever grows: `kya karna chahiye`, `क्या करना चाहिए`, `what should staff do`, all meaning the same thing. A 3B model reads all three without a table. What it costs is reproducibility, which is bought back by validating the label against a fixed set and by keeping every other decision in code. |
| Regular expressions over phrasing | **Rejected, and this is what it was.** It worked on the gold set and it worked because I wrote both. The failure mode is quiet: a phrasing nobody listed routes to documents, retrieval answers with the stock-out procedure, and nothing looks wrong. |
| LangGraph / LlamaIndex orchestration | **The production answer** once there is state worth persisting: retries, interrupted human escalation, a resumable conversation. Here the graph is five nodes with no cycles, so a framework would be the largest dependency in a project that otherwise has none. The boundaries are what matter, and they port. |

**The verifier is the reason the split is safe.** The document agent filters by
state, but once two agents compose a single answer no agent owns all of it any
more, so state and dating are re-checked at the boundary
(`agents/orchestrate.py`). There is a test that makes the document agent return a
Jharkhand citation to a Bihar clinic and asserts the verifier stops it.

**Three labels, and only one of them changes where the answer comes from.**
`inventory` leaves the corpus for a structured lookup; `policy` and `clinic`
both go to the same hybrid retrieval. So a confusion between the last two costs
nothing, and the only cell of the confusion matrix worth watching is the one
between `inventory` and the others. That is a deliberate shape: the model's
mistakes are concentrated where they are cheap.

**A label the system does not have is not guessed.** `read` validates against
`LABELS` and returns `None` otherwise, which ends the turn at a person. The
same for no reply at all, with a different reason recorded, because an outage
and a confused model are different operational problems.

**Why not two labels, or five.** Two would merge `inventory` into "everything
else" and lose the one distinction that matters. Five would need a reason per
label, and a label earns its place by changing the data source or the
permissions, not by naming a topic. Schemes and protocols are the same tool
over the same index with the same contract.

**Routing is still asserted, not sampled.** `plan` is pure, so
`tests/test_agents.py` asserts the branch for seven label-and-resolution
combinations without a model in the way, and asserts the exact trace for six
whole questions against a scripted model. The model's own accuracy is the part
that needs a labelled set of real questions, and that is named in LLD.md as
missing rather than claimed.

---

## 2. Tool use, and why it is a registry

**Chosen:** every capability declared in `tools.py` with its parameters and the
data it can see, called through one method that records the call.

| Option | Verdict |
|---|---|
| Direct method calls from the orchestrator | **Rejected, and this is what it was.** It works, and it gives you nothing to audit, nothing to hand a model, and no single place where data access is declared. |
| A tool registry with declared params and scope | **Used.** Three tools. Each declares what it reads, which is where the governance table comes from; each call is recorded with its arguments, which is what makes the trace an audit trail and the agent-sequence tests possible. |
| An LLM function-calling loop over the same tools | **Reasoning only**, and deliberately a half-step away. The model picks a branch from three labels; it does not emit `check_stock(code=...)`. If it did, a code the model invented would reach the stock table, and the thing that makes drug lookup safe here is that the code always comes out of the formulary via the resolver. `GET /tools` returns the schema and `Toolbox.call` records every call, so the day a model drives the registry it is a new caller around the same code, with the resolver still standing between a typed name and a lookup. |
| An MCP server per tool | Rejected here. MCP earns its keep when tools are shared between applications or run out of process. These are in-process functions over local files, so it would add a protocol and a process for nothing. Worth it the day another A4I project wants the same formulary tool. |
| LangChain or LlamaIndex tool abstractions | Rejected. A dependency to get a dataclass and a dict. |

**Tools never call each other.** The orchestrator is the only caller, so the
call order in the trace is the whole story and there is no hidden fan-out.

**An agent is a tool with a decision inside it.** `answer_from_documents`
retrieves in two spaces, fuses them, applies a floor per space, sends the
chunks that cleared to the model, and refuses any citation it did not show the
model, all before it returns. `check_stock` is a lookup. Keeping one shape at
the boundary is what makes the trace uniform and the tests simple.

**The generation call sits inside that tool, not in the orchestrator.** The
thing that chose the context is the thing that has to check the citation
against it. Moving the call up a level would mean passing the cleared set
upward just so something else could validate against it.

---

## 3. State and memory

**Chosen:** one piece of state, a clarification that has been asked and not yet
answered (`session.py`, 40 lines).

| Option | Verdict |
|---|---|
| No state at all | **Rejected**, and it is what the first version did. The clarification was unanswerable: the client had to restate the whole question, so "which iron did you mean" could only be answered by typing the full drug name, which defeats asking. |
| Pending clarification only, server side, TTL ten minutes | **Used.** It is the smallest state that makes asking useful, and it is bound to one clinic so a session cannot reach another clinic's data. |
| Full conversation history | **Rejected on privacy, not on cost.** The tablet is shared across a queue of patients. Anything remembered about one is a leak waiting for the next, and no feature here is worth that. |
| Summary or vector memory over past turns | Rejected. The same leak with extra steps, and scheme lookups gain nothing from it: the questions are independent. |
| Follow-ups like "and zinc?" | **Rejected as a consequence.** Resolving pronouns and ellipsis needs the previous question kept, which is the thing ruled out above. Typing the drug name again is cheap. |
| Keeping the state in the client | **Rejected, and this one is a security answer.** If the options and codes live in a cookie, a reply can name a drug that was never offered. Holding them server side means a choice can only be made from what was actually asked. |
| Redis | **The pilot answer**, same shape and same TTL. Keeping the shape this small is what makes that a storage change rather than a redesign. |

**A one-off question leaves nothing behind.** A session is offered on every turn
and discarded unless something is outstanding, so the store holds conversations
mid-question and nothing else. Bounded at 500, evicted by last touch.

**Nothing in the pipeline reads session state except the routing step**, which
is what keeps retrieval reproducible: the eval and the floor are measured
against behaviour that does not depend on what was asked before.

---

## 4. Guardrails

**Chosen:** deterministic rules in one leaf module, screening on the way in and
on the way out (`guardrails.py`).

| Option | Verdict |
|---|---|
| Named rules, three actions: block, redact, flag | **Used.** Every trip appears in the trace and in the query log, so a refusal can be explained to a supervisor afterwards. The clinic network can read the rule set, which matters when the rules encode somebody's clinical policy. |
| An LLM moderation pass | **Reasoning only.** It would catch phrasing these patterns miss. It also spends the one model slot, adds latency to every question, and a classifier that is 95% right about an emergency is not a thing to put in front of a clinic. Revisit as a second opinion that can only escalate, never clear. |
| Guardrails AI / NeMo Guardrails | Rejected here. The framework would be larger than the project, and the value is in which rules exist rather than in what runs them. The rules port. |
| Output guarding only | Rejected. A dosing question should not reach retrieval at all, and a red flag should not wait for an answer to be composed before it goes to a person. |
| Block anything uncertain | Rejected. Staleness is a flag, not a block: an old document is still the best available answer, and its date is already in the output. |

**On the way in.** An emergency red flag stops everything and goes to a person.
Then instruction override, then dosing or diagnosis, then an oversized
question. Phone and Aadhaar-shaped numbers are redacted before retrieval and
before the log, so an identifier a patient typed is never stored.

**Order is a decision, not an accident.** "She is unconscious, what dose do I
give" trips two guards. The one that reaches a human wins, and there is a test
pinning that precedence.

**On the way out.** A missing citation, dosing language in the answer, an
identifier echoed back, and any digit that does not appear in a cited source.

**Three of the output guards were written before there was a model to need
them**, and all three can fire now: a citation outside the set the model was
shown, a digit that appears in no cited source, dosing language in a written
sentence. The digit guard is the one that matters most, because an entitlement
amount is both the most harmful thing to get wrong and the easiest thing to
check exactly. It is what catches fourteen hundred becoming 1500.

**What it does not do:** no toxicity classifier, no jailbreak coverage beyond
the obvious patterns, no rate limiting, which belongs at the gateway rather
than in the pipeline.

---

## 5. Drug resolution

**Chosen:** a three-layer resolver over the national formulary, with measured
thresholds and three outcomes (`agents/drugs.py`).

| Option | Verdict |
|---|---|
| One exact lookup against the clinic's own stock list | **Rejected, and this is what it replaced.** A boolean over per-clinic aliases conflates three different situations, and the worst of them tells a health worker the clinic does not stock something it does. |
| Alias, then transliteration key, then trigram-blocked fuzzy match | **Used.** Cheapest layer first. A listed phrase scores 1.00, one word out of a phrase 0.90, a transliteration-key hit 0.88, and a fuzzy match its own ratio. Candidates come from a character-trigram index, so the formulary can grow without the per-query cost growing. |
| Embedding similarity over drug names | **Rejected for now.** Drug names are a closed vocabulary, which is the one case where a lexicon beats a model: an embedding model is good at meaning and unreliable at spelling, and the measurement in decision 4 shows what it does with romanised Indian text. It would also put a model in the path of a lookup that has to work offline. |
| An LLM to extract the drug name | Rejected. A model call per question to search fourteen rows, non-reproducible routing, and the failure mode is inventing a plausible drug name, which is the worst outcome available here. |
| Take the closest match and answer | Rejected. Resolving a misspelling to the wrong drug is the only failure in this build that is worse than refusing. |

**Three outcomes, never a boolean.** Resolved, ambiguous, unresolved. Then the
stock agent splits "resolved" again: this clinic has a count for it, or the
formulary knows the drug and this clinic carries no record. That second case is
an answer with the indent procedure attached, not a refusal, and the stock
agent's finding is what adds the document agent to the plan.

**Ambiguity asks.** "Is iron available" fits two drugs, so the system returns a
question with the options rather than whichever row loaded first. `Clarify`
sits beside `Answer` and `Handoff` in the return type, so a caller cannot
accidentally treat it as either.

**The thresholds are measured**, with `analytics.drug_sweep`
over a 35-case gold set of misspellings, brand names, both scripts, shared
words and non-drugs.

| Accept | Right | Wrong | Asked | Held | Over-asks | Over-holds | Guessed |
|---|---|---|---|---|---|---|---|
| 0.80 | 28 | 0 | 2 | 4 | 0 | 0 | 1 |
| 0.84 | 28 | 0 | 2 | 5 | 0 | 0 | 0 |
| 0.86 | 28 | 0 | 2 | 5 | 0 | 0 | 0 |
| 0.88 | 28 | 0 | 2 | 5 | 0 | 0 | 0 |
| 0.90 | 23 | 0 | 2 | 5 | 0 | 5 | 0 |

Nothing resolves to the wrong drug at any setting, which is the constraint that
had to hold. What moves is how often it asks or refuses. Below 0.84 it starts
answering a term that should have been a question; above 0.88 it stops
recognising misspellings, because a key hit scores 0.88. The floor sits at 0.86,
the middle of that band, and there is a test asserting zero misresolutions
across the whole band rather than at one point.

**The unresolved list is the production loop.** Every availability question
that matches nothing is recorded with its nearest near-miss and served at
`/analytics/unresolved`. A closed vocabulary gets good by someone reading that
list weekly and curating aliases, which is a person's half hour rather than a
model. It is also the honest answer to "what happens when a new drug arrives":
the system refuses, says why, and tells you what to add.

**Why the formulary is separate data from the stock count.** Different owners
and different clocks. The formulary is national and revised rarely; counts are
per clinic and change every shift. They join on the code, so a clinic's file
never carries a drug name that someone could misspell into a different drug.

**What it does not do.** No strength or dosage-form disambiguation, so a real
formulary with 125mg and 250mg of the same drug needs another level in the key.
No supplier or batch codes. Both are data problems rather than design ones, and
both are why the resolver returns candidates rather than a single code.

---

## 6. Embedding model

**Chosen:** `multilingual-e5-small` as the dense half of a hybrid retriever,
with its floor measured in its own space (`retrieval/embed.py`).

| Option | Verdict |
|---|---|
| `multilingual-e5-small` | **Used.** 384 dimensions, about 120MB of weights, CPU inference, downloaded once and then offline. It is the smallest encoder that handles Devanagari and English properly, which matters because a 3B generative model is resident in the same 16GB. |
| A larger multilingual encoder (`e5-base`, `e5-large`) | Rejected for this box. The gain is real and the memory is not available next to a generative model. This is the clearest place the hardware constraint actually bit. |
| `MuRIL` / IndicBERT | **The next thing to benchmark**, and now for a specific reason: both are trained on transliterated Indian text, which is the exact gap measured below. If MuRIL closes it, the lexical retriever gets simpler rather than disappearing. |
| `LaBSE` | Heavier than needed and tuned for sentence equivalence rather than query-to-document retrieval. |
| OpenAI `text-embedding-3-*` | Rejected outright. A clinic with an unreliable link cannot depend on a network call to answer whether a drug is substitutable. |
| Dense retrieval alone | **Rejected on measurement.** See the next section: it cannot separate romanised Hindi from out-of-scope at any threshold. |

**What the encoder can and cannot separate**, measured on the 25-question gold
set with `scripts/evaluate.py`:

| | highest out-of-scope | lowest correct answer | separable? |
|---|---|---|---|
| English and Devanagari | 0.840 | 0.871 | yes, floor 0.855 |
| romanised Hindi | 0.840 | 0.797 | **no** |

**Why it fails on romanised Hindi.** That form is not in its training
distribution. Latin-script Hindi tokenises into subwords that carry no meaning
for the model, so similarities collapse towards the corpus mean, and they land
*below* where an out-of-scope question lands. There is no threshold that admits
the first and refuses the second. `tests/test_retrieval.py` pins it: the
romanised Ayushman Bharat question and a question about the petrol price sit
within 0.02 of each other on the encoder.

**That is a measurement, not a preference, and it is what the hybrid is for.**
The honest reading is not "the encoder is bad", it is "one third of the traffic
is out of its distribution and a second retrieval signal is cheaper than a
model that covers it".

**The honest caveat.** The corpus is 16 chunks of distinctive scheme names,
which flatters exact lexical matching, so the relative strength of the two
retrievers will not transfer unchanged to a real corpus. What transfers is the
out-of-distribution finding, because it is about the shape of the score
distribution rather than about this gold set. At pilot scale I expect the
encoder to win outright on Devanagari and English, and the romanised path to
still need its own signal.

**One configuration ships, not three.** The app builds the hybrid index and
nothing else: no environment variable, no choice at startup. A switch would
only be somewhere for a clinic to be misconfigured.
`scripts/evaluate.py` prints each retriever's scores side by side, which is
where evidence belongs rather than in a runtime flag.

---

## 7. Cross-lingual strategy

**Chosen:** three weighted keys per token, surface form plus normalised Latin
plus consonant skeleton, indexed as BM25 terms (`lang.py`,
`retrieval/lexical.py`).

| Option | Verdict |
|---|---|
| A shared multilingual embedding space, alone | **Rejected on measurement.** See section 6: romanised Hindi is out of distribution and no threshold separates it from noise. |
| Transliteration keys as BM25 terms, beside the encoder | **Used.** Every word is expanded to its surface form, its normalised Latin form and its consonant skeleton, and all three are indexed, so the three ways one word gets typed meet in the same posting list: `आयुष्मान` and `ayushman` land on the same Latin key. |
| Translate the query to English, retrieve against an English corpus | Rejected. More predictable, but it discards Hindi-language source documents and puts a translation model in the latency path of every question, on hardware with no room for one. |
| Two indexes, one per language | Rejected. A romanised query belongs to neither, so the form most clinic staff type in is the form with no home. |
| Ask the generative model to normalise the query first | Rejected. A third model call, and it makes retrieval depend on the model's spelling choices, so the same question can retrieve differently on two runs. Transliteration is a deterministic string problem and belongs in deterministic code. |

**Why the skeleton key exists:** romanised Hindi has no agreed spelling.
*yojana* and *yojna* are the same word, *ayushman* and *aayushmaan* are the
same word, *dawa* and *dava* are the same word. Dropping vowels after the
first character makes them collide. It over-matches by design, which is why it
carries 0.5 against 1.0 for an exact same-script match.

**Weights are on the query side only.** BM25 takes term frequencies from a
document; a weight on a posting would be a ranking signal with no meaning. So
the query asks for `s:ayushman` at 1.0 and `k:ysmn` at 0.5, and the documents
just hold counts.

**Keys shorter than three characters are not indexed**, and this came out of a
real failure. The formulary lists MR, TT and IFA as abbreviations, and `मेरे`
skeletonises to `mr`, so "मेरे बेटे का स्कूल दाखिला कैसे होगा?" resolved to the
measles and rubella vaccine and was answered with the indent procedure. A
two-letter key carries no information: it cannot survive a misspelling because
there is nothing to misspell. Abbreviations still match exactly through the
part index, which is where an abbreviation belongs.

**Keep this even with a better multilingual model.** Transliteration variance
is a spelling problem, not a semantic one, and embedding models are not
reliably good at it.

---

## 8. Chunking

**Chosen:** one chunk per markdown heading (`retrieval/index.py`).

| Option | Verdict |
|---|---|
| Heading / structural | **Used.** Scheme documents are already written in the unit a question is asked about: eligibility, entitlement, process. The author's structure beats any splitter I would write. |
| Fixed token window with overlap | Rejected here. It splits an entitlement away from the scheme it belongs to, and then a retrieved chunk holds an amount with no indication which scheme or state it belongs to. |
| Semantic / embedding-based splitting | Rejected. It needs the embedding model at ingest time, it is hard to explain to a partner organisation, and it buys nothing when the documents are already sectioned. |
| Sentence windows, or parent-document retrieval | **The upgrade.** Retrieve small for precision, return the parent section for context. This build approximates the second half by returning two sentences instead of one. |

**Chunk size is bounded by the source, not by a constant.** A scheme document
with a 900-word section gets a 900-word chunk here, and retrieval goes vague.
That is a real limitation and the trigger for parent-document retrieval.

---

## 9. Retrieval

**Chosen:** hybrid. Dense cosine and BM25 over transliteration keys, each
scanned flat, fused by reciprocal rank, with a floor per retriever
(`retrieval/index.py`).

| Option | Verdict |
|---|---|
| Dense alone | **Rejected on measurement.** Section 6: it cannot separate romanised Hindi from out-of-scope. |
| BM25 alone | Rejected. It has no answer for paraphrase, and the gold set has questions that share no token with their document. |
| Hybrid dense plus BM25, reciprocal rank fusion | **Used.** Scheme names, drug codes and clinic codes are exact-match tokens that dense retrieval is bad at and BM25 is good at; paraphrase is the reverse. RRF needs no score calibration between the two, which matters because a cosine and a normalised BM25 sum are not the same quantity. |
| Weighted score fusion | Rejected. It needs the two scores on a comparable scale, and they are not. Any weighting would be a constant fitted to 16 chunks. |
| Routing by input form, one retriever per form | **Rejected, and this is what it was.** It worked, and it made the form detector a single point of failure: a question misread as English went to the encoder and was refused. Running both retrievers always and accepting either floor removes that branch entirely. |
| Flat exhaustive scan for both | **Used.** 16 chunks now, maybe a few thousand per state at full scope. Exhaustive is exact and instant at that size, and an ANN index here is a dependency bought for nothing. |
| FAISS / `sqlite-vec` / Chroma | **The production answer** once the corpus is big enough to matter. `sqlite-vec` plus an FTS5 table, because two tables in one file on disk suits a clinic better than a service that has to be running. |
| Milvus, pgvector, any hosted store | Rejected for the clinic. A vector service alongside the model in 16GB is the wrong use of the memory. pgvector is reasonable at a district tier, if one ever exists. |

**A retriever that matched nothing contributes nothing.** Only hits a retriever
actually scored above zero enter its shortlist, because fusing an arbitrary
ordering of zeroes reorders the other retriever's results for no reason. That
was a real bug: out-of-scope questions came back in ascending relevance.

**BM25 is normalised against the ceiling the query could reach**, the sum of
`weight × idf × (k1+1)` over the query's terms. Without that the score grows
with query length and one floor cannot serve questions of different lengths.
It is an imperfect normalisation: a long question with one matched term scores
low even when that term is decisive. The alternative, dividing by only the
terms that matched, makes any single match look certain, which is worse.

---

## 10. The state filter

**Chosen:** hard metadata filter applied before scoring (`Hybrid.visible`).

| Option | Verdict |
|---|---|
| Pre-filter the candidate set | **Used.** Another state's document is not a candidate at all. |
| Post-filter the results | Rejected. It works until the filter is forgotten in one code path, and then nothing fails loudly. |
| State as a ranking signal or a boost | **Rejected, and this is the most important decision in the build.** As a ranking signal a neighbouring state's document is merely outranked, which means one day it is not. The system then tells a Bihar clinic Jharkhand's entitlement amount, with a citation attached, at full confidence, with nothing in the output indicating anything went wrong. |

The same scheme pays different amounts in different states, and rural differs
from urban again. `tests/test_retrieval.py` asserts both that the filter holds and
that it has something to hold back, so the other tests cannot pass by accident.

---

## 11. Re-ranking

**Chosen:** none, and the fusion does part of its job.

| Option | Verdict |
|---|---|
| No re-ranker | **Used**, with both floors measured high enough to pay for it. |
| Reciprocal rank fusion over two retrievers | **Used, and it is not a re-ranker.** It reorders, but it adds no model and no new signal; it just refuses to let either retriever's ordering be the only one. On the gold set it puts the right document first in every form but one, where it is second. |
| Cross-encoder re-ranker (`bge-reranker`, MiniLM) | **The upgrade**, and the first thing I would add with headroom. A large precision gain on exactly the near-miss cases in the eval. Costs a second model in 16GB that is already holding a generative one, which is why it is not here. |
| LLM re-ranking | Rejected. A model call per candidate, on hardware with one model, to reorder four results, when the same model is about to be called to write the answer anyway. |

Without re-ranking, precision drops. The response is not to ship something
wrong more often, it is to keep the floors where they are and ship something
that says "ask the supervisor" more often.

---

## 12. The retrieval floors

**Chosen:** one floor per retriever, each measured by sweeping the gold set,
each owned by the thing that produces the score. 0.855 for the encoder, 0.055
for BM25. A chunk is usable when either one clears.

| Option | Verdict |
|---|---|
| A floor per retriever, measured | **Used.** A cosine and a normalised BM25 sum are not the same quantity, and no single number can be a threshold for both. |
| One constant in the pipeline | Rejected. It silently becomes wrong the day a retriever or a model is swapped, and nothing fails: the system just starts answering from noise. |
| One fused score with one floor | Rejected after trying it. Expressing each score as a multiple of its own floor does give one comparable threshold, and it also makes the magnitudes incomparable: the lexical ratio reaches 11x where the dense ratio reaches 1.09x, so the reported confidence was always the lexical one. Two numbers and one boolean is less clever and says more. |
| Require both retrievers to clear | Rejected. It is the intersection of two different kinds of confidence, and it would refuse every romanised question, which the encoder cannot score. |
| Relative margin between the top two hits | **Worth adding.** Catches several chunks matching weakly and equally, which an absolute floor lets through if the scores happen to sit high. In the dense space the top-1 to top-2 gap is 0.001 to 0.014, so that space in particular wants a margin rather than only a threshold. |
| Per-language floors | **Rejected, and I want to be careful here.** Lowering the bar for the language that is failing is the opposite of a fix. The hybrid addresses the same problem by adding a signal rather than by relaxing a standard. |

**Both numbers sit in the middle of a measured band**, which is the whole
method: find the highest score an out-of-scope question reaches in that space,
find the lowest score a correct answer reaches, and take the middle.

| Retriever | highest out-of-scope | lowest correct | band | floor |
|---|---|---|---|---|
| dense cosine | 0.840 | 0.871 | 0.840 – 0.871 | **0.855** |
| BM25 over keys | 0.011 | 0.102 | 0.011 – 0.102 | **0.055** |

At those two values the gold set answers 20 of 20 answerable questions from the
right document and holds all 5 out-of-scope ones, and that result is stable
across the whole band rather than at a point: `scripts/evaluate.py` prints the
sweep. A floor at an edge of a band is a floor that moves the first time the
corpus does.

**What the sweep also showed.** A *lower* lexical floor produced a wrong
answer, which is the opposite of the intuition. Admitting more chunks changes
which cleared chunk ranks first, and a weakly-matched one can take the top
spot. A floor is not only a refusal threshold, it is also a filter on what the
model is shown.

---

## 13. Answer generation

**Chosen:** the same local model that read the question writes the answer, shown
only the chunks that cleared a floor, and asked for the source numbers it used
(`generation.py`).

| Option | Verdict |
|---|---|
| Abstractive, local model, constrained to retrieved text | **Used.** Clinic staff ask in two languages and three scripts; quoting cannot answer "what should I do" in the language it was asked in. The constraint does the safety work: the model sees only cleared chunks, and what it writes is checked afterwards. |
| Extractive quoting | **Rejected, and this is what it was.** Safe and unreadable. It answered a Hindi question with an English sentence, because the sentence it quoted was the one in the document. It also cannot compose two sources into one answer, which the stock-plus-procedure case needs. |
| Hybrid: quote protocol and figures verbatim, generate the connective language | **The better version, and not built.** It needs a way to tell which spans are quotable, which means either marking them in the corpus or an extraction step. Marking them in the corpus is the honest route and it is curation work. |
| Template-filled responses | Rejected. Needs a schema per scheme, and schemes change faster than the templates would. |
| A larger model for generation, a smaller one for intent | Rejected. Two sets of weights in 16GB. The intent prompt is short and cheap against the same model. |

**Source numbers, not chunk ids.** The model is shown `[1]`, `[2]`, and asked
which it used. A 3B model copies a small integer back reliably and
`pmjay-bihar#1` it does not. The mapping back to ids happens in code, where a
number outside the range it was shown is dropped, and the citation check then
refuses an answer with nothing left.

**The prompt forbids four things**, and each is also enforced in code
afterwards: adding a figure, answering in the wrong language, a dose, and
answering when the sources do not. Prompt instructions are a first line, not a
control. The ones that matter are checked.

**"Insufficient" is a first-class reply.** The model can say the sources do not
answer the question, and that becomes `MODEL_ABSTAINED`, which is counted
separately from a retrieval miss. One says the corpus had nothing; the other
says the corpus had something and the model would not use it. Averaging them
would hide whichever is actually happening.

**Temperature 0, one attempt.** The same question gets the same answer, which
is what makes the eval mean anything. A retry at a higher temperature is a
different answer to the same question, which in a clinic is worse than a
refusal.

---

## 14. The grounding gate

**Chosen:** three checks after generation, all deterministic, none of them the
model's own opinion.

| Option | Verdict |
|---|---|
| Citation must be one of the chunks the model was shown | **Used, and it fires now.** Under extraction this check could not fail; with a model writing, it is the thing standing between a confident paragraph and a clinic. |
| Every digit in the answer must appear in a cited source | **Used, then changed to values.** An entitlement amount is the single most harmful thing to get wrong here, and a wrong number is also the easiest thing to check exactly. Written before there was a model to need it. |
| Compare figures as values, digits and words alike | **Used**, after a served 3B wrote "1200" for a source that says "twelve hundred" and the string match refused a correct answer. "1,200", "twelve hundred", "बारह सौ" and "१२००" are all 1200 (`guardrails.figures`). The answer side counts only figures with a digit or a scale word, since "one of" and एक are prose. It also closes a hole: "six lakh" against "five lakh" used to pass because neither side had a digit. |
| Dosing language blocked in output | **Used**, even though no current document contains any. The day one does, the guard is already there. |
| NLI entailment between answer and source | **The stronger version**, and the right upgrade. It catches the harder failure: a correct citation attached to a claim the source does not support, in words the digit check cannot see. Costs a second model, which is the same 16GB argument as the re-ranker. |
| LLM self-check | Rejected. The model that wrote the answer is the worst available judge of whether the answer is grounded, and it is the only model in the box. |
| Asking the model for a confidence score | Rejected. It is a number that correlates with fluency rather than with correctness, and it would be reported next to real measured scores as though it were one. |

**What is still not caught.** A fluent paraphrase that changes a condition
without changing a digit: "cover is cashless at empanelled hospitals" becoming
"cover is cashless at any hospital". No digit moved, the citation is right, and
the digit guard is blind to it. That is the NLI gap, it is named here rather
than hidden, and in the meantime the prompt's two-sentence limit keeps the
surface area small.

---

## 15. Evaluation, and where a judge model belongs

**Chosen:** a deterministic gold set, scored per input form, with no judge
model in the loop (`analytics.py`, `gold/gold.jsonl`).

| Option | Verdict |
|---|---|
| Deterministic gold set | **Used** as the gate. A test passes or it does not, and a failure points at a line of code rather than at a mood. |
| RAGAS's four metrics, computed without a judge | **Used** (`analytics.py`, served at `/analytics/ragas`). Context recall, context precision, faithfulness and answer relevance, each derived from the retrieved set and the gold label rather than from an LLM. Proxies, and they move in the same direction as the judged versions. |
| RAGAS with a judge model | **One layer out, not in the gate, and not this model.** The only model in the box is the 3B that wrote the answer, so asking it to grade its own faithfulness is marking its own homework. A judge belongs on a reviewer's machine with a larger model, where judged faithfulness and relevance over a corpus is genuinely useful for drift and for the long tail no gold set covers. What it cannot be is the thing that blocks a deploy: a flaky grader and a real regression look identical from the failure message, and the afternoon goes on the wrong one. |
| LangSmith / tracing-based eval | Complementary, not an alternative. It is where the production traces go; it does not decide whether the build is good. |
| Human review by clinic staff | **Not optional at pilot.** The gold set was written by me, and I do not know what a health worker actually asks. The clinic network's own staff should write the questions, which is a scheduling item rather than a tooling one. |

**Three rules the module enforces.**

Never average across languages. English traffic carries the mean while Hindi
fails quietly, and the single number reads healthy throughout. That is not a
hypothetical here: the version before the hybrid scored 18 of 20 overall and 2
of 4 on romanised Hindi, and only the per-form table said so.

Never sum wrong answers and handoffs. A handoff costs time; an answer from the
wrong state's document costs more than time. Separate columns, never added.

Count the cross-state leak separately and assert it at zero, not low.

**What the metrics say that the tally does not.** Context recall was 1.00 in
every input form back when romanised Hindi abstained on half its questions.
Retrieval was finding the right document every time and the similarity score
was not clearing the floor, so the gap was in the scoring rather than in the
search. That is what pointed at a second retrieval signal instead of a bigger
corpus or a lower floor, and the tally alone would not have said which.

**Which half of this is still measurable with no model served.** Context recall
and context precision need no generated answer, so they are measured either
way. Faithfulness and relevance need one, and with nothing served they report
zero answered rather than a flattering number. The test suite measures them against
the scripted stand-in, which quotes its source back, so faithfulness is 1.00 by
construction there and says nothing about a 3B. The published figures come
from a separate run against a served model ([EVALUATION.md](EVALUATION.md)),
which is where the lexical faithfulness metric also showed its limit: it
cannot see a faithful translation, and scored English answers written from
Hindi chunks near zero. That is why `figure_recall` sits next to it, checking
by value that the answer states the figure the question asked for.

---

## 16. Fallback behaviour

**Chosen:** abstain and hand off to a human. No retry, no degradation, no
hedged answer, and no path that answers without the model.

| Option | Verdict |
|---|---|
| Abstain and escalate | **Used.** Refusing is a designed outcome, not an error path, which is why `Handoff` sits alongside `Answer` in the return type instead of being raised. |
| Answer anyway with a confidence caveat | Rejected. "I think the amount is fourteen hundred rupees, but please verify" is read as fourteen hundred rupees. A hedge is not a safety mechanism, it is a liability transfer. |
| Retry with a relaxed threshold | Rejected. A relaxed retry is a wrong answer on a delay, and the floor sweep shows what gets admitted. |
| Retry the model call | Rejected. One attempt, 30 second timeout. A clinic waiting on a second attempt is worse off than a clinic told to ask the supervisor, and a model that timed out once at temperature 0 will time out again. |
| Fall back to extraction when the model is down | **Rejected, and this was tempting.** It keeps the demo working, and it means a clinic cannot tell from the answer whether the system is healthy. Two answer paths also means two sets of behaviour to evaluate and one of them is never exercised. A model that cannot be reached is an outage, and an outage should look like one. |
| Ask a clarifying question | **Used, for the one case where the system can tell ambiguity from ignorance:** a drug name that fits two formulary entries. `Clarify` is a third return type beside `Answer` and `Handoff`. |

An explicit request for a person short-circuits before the model is called at
all, and so does answering an outstanding clarification. Both are asserted in
tests rather than left to a prompt.

**The refusal reasons are deliberately granular.** Six distinct ways to not
answer, each pointing at a different owner: a guard trip is a policy question,
`below_floor` is a corpus gap, `not_in_formulary` is a curation task,
`model_abstained` is a prompt or model problem, `model_unavailable` is an
operations problem, and `unclear_intent` is a prompt problem. One
"I don't know" bucket would make the weekly review useless.

---

## 17. Serving the generative model

**Chosen:** not served in this process. One client in `llm.py` against an
OpenAI-compatible `/chat/completions` endpoint, configured by environment.

| Option | Verdict |
|---|---|
| An OpenAI-compatible endpoint, set by URL | **Used.** Ollama exposes it on a laptop, vLLM exposes it on the clinic's GPUs, and a hosted endpoint exposes it for a reviewer who wants to see the thing run. One code path, and deployment stops being a code decision. |
| `llama.cpp` with a 4-bit quantised 7-8B instruct model | **What I would run in a clinic.** 16GB of system RAM is the binding constraint, not the GPUs, and a clinic has a handful of concurrent users rather than hundreds. Memory headroom beats batching throughput. It also serves the same API, so nothing here changes. |
| vLLM | Better engineering, aimed at a problem a single clinic does not have. Continuous batching and paged attention earn their keep under concurrency one clinic will never generate. Right answer at a district tier. |
| Loading the model in-process with `transformers` | Rejected. It couples the API's lifetime to a multi-gigabyte load, makes the test suite depend on weights, and means a model upgrade is a redeploy of the pipeline. |
| Hosted API as the clinic's answer | Rejected. Unreliable connectivity is the premise. Fine as a reviewer's convenience, which is why the URL is configurable rather than fixed. |

**The model family, if I had to name one:** Qwen2.5 instruct, 3B for a laptop
and 7B quantised for the clinic. It is the strongest small open family on Hindi
I have used, and both sizes take the same prompts, so the demo and the
deployment are the same system at two sizes. That is a defensible default and
not a measured result: I have not benchmarked it against Gemma or Llama on this
corpus, and a pilot should.

**What happens with nothing served.** Every question that needs a written
answer returns `model_unavailable` and goes to a person; retrieval, drug
resolution, the guards and the two context metrics all still run and are still
measured. The defaults point at a local Ollama, so one `ollama pull` makes the
whole thing live with no code change. The cost of that choice is that the
test suite never exercises a real model, so generation quality is measured
separately, by `scripts/live_eval.py` against a served
`qwen2.5:3b-instruct` ([EVALUATION.md](EVALUATION.md)).

---

## 18. Staleness and sync — **reasoning only**

An offline corpus goes stale and government schemes change. Two things in the
build already account for that: every chunk carries its source document's date,
and every answer carries that date through to the output.

Sync is pull-based delta when the link returns, and a sync never overwrites a
local record. Conflicts go to a supervisor instead of being resolved
automatically, which is also what the protocol document says should happen.
Last-write-wins would silently discard a clinic's own register entry.

---

## 19. Access control and governance

**Chosen:** an API key maps to the states it may query, applied as one
dependency (`security.py`, `deps.caller`).

| Option | Verdict |
|---|---|
| Key to clinic scope, one dependency | **Used.** 401 for an unknown key, 403 for the wrong clinic. `/analytics/*` carries it at the router so an endpoint cannot forget it. |
| No authentication | **Rejected for a pilot**, and it is the default on a laptop. With no keys configured the API runs open and `/health` reports `"auth": "open"`. A demo that needs setup before it runs is a demo nobody runs, so the posture is reported rather than silent. |
| Fail closed with a generated key at startup | The right default at pilot, and wrong for a demo, for the reason above. It is a line of configuration, not a redesign. |
| mTLS per device | **The pilot answer for tablets**, since a device certificate is harder to copy than a header. Needs a certificate authority somebody runs. |
| OAuth, or ABDM identity | Right when this has to join a national identity stack. Overkill now, and token refresh assumes connectivity that is unreliable by premise. |
| Relying on the retrieval state filter alone | **Rejected as the only control.** It would probably hold. The cross-state answer is the worst failure in this system, so it gets two independent controls. |

**The duplication is the design.** The authorisation boundary and the retrieval
boundary are the same line: a Bihar key asking about Jharkhand is refused by the
gate, and would have been refused by the filter. A mistake in either is caught
by the other.

**The audit record names the caller.** Every query logs actor, outcome, which
tools ran, guard trips and redactions. It is in process today and dies on
restart, which is honest for a demo and the first thing to fix for a pilot,
along with a retention period agreed with the clinic network.

**No patient identifier is a key anywhere.** Stock is keyed by drug code,
documents by chunk id, sessions by a random token. Phone and Aadhaar-shaped
digits are redacted before retrieval and before the log, so the stored question
already has them removed.

The rest, including the data classification table and what a pilot still needs,
is in [ORCHESTRATION.md](ORCHESTRATION.md).

---

## 20. The HTTP layer, and where the dependency line sits

**Chosen:** FastAPI over the same orchestrator, with the pipeline itself
keeping zero dependencies (`routers/`).

| Option | Verdict |
|---|---|
| FastAPI | **Used.** Typed request and response bodies, validation at the edge (a 501-character question is a 422 before any guard runs), and generated OpenAPI so the clinic network's own app has something to read. |
| Flask | Workable. No typed models and no schema without extra pieces, which is most of what the API layer is doing here. |
| A library with no API | Rejected. Something has to call this: a clinic tablet, a supervisor's dashboard, eventually an IVR front end. |
| gRPC | Rejected for this deployment. Harder to debug on a clinic box, and there is no streaming or cross-service traffic to justify it. |

**The dependency line is the point.** Everything below `routers/` imports
only the standard library, so the pipeline runs on a clinic machine with
nothing but python3 and a folder of markdown. `requirements.txt` exists for
the HTTP layer and the tests. If FastAPI is not installed, `demo.py` and the
whole retrieval stack still run.

**Handlers are synchronous.** The work is CPU-bound: hashing, cosine, sentence
selection. An async handler around CPU-bound work buys nothing and blocks the
loop, so it goes in a threadpool, and concurrency per clinic is a handful of
users anyway.

**One orchestrator per state, cached, and the index built once at startup.**
The stock table is per clinic, so the state belongs in the object rather than
in each call. Building the index per request would be the easy mistake here.

**The client is one page of plain JavaScript, served by the same process**
(`static/`). No build step, no framework, no CDN: a page that
fetches a font from a CDN is a page that breaks in the clinic it was written
for. React would be a toolchain and a node_modules for four fetches and two
tables. What the page is actually for is making the refusals legible, since
"it declined, and here is the guard that declined it, and here is what it
would have cited" is the part that is hard to convey in prose.

**`/analytics/live` is in-process and resets on restart**, which is honest for
a demo and wrong for a pilot. In production the same records go to
OpenTelemetry spans and a table, which is also where per-clinic abstain rates
become something a supervisor can watch.

---

## 21. Smaller things

**Language detection** is script detection, not a classifier. Devanagari
present means Hindi. A statistical detector on a six-word query is unreliable
and would be a dependency bought for a problem a range check solves.

**Stopwords** are three curated lists: English, Hindi, and romanised Hindi.
IDF over the corpus would be more principled and needs a corpus large enough to
estimate from. The romanised list exists because a Hinglish query carries Hindi
function words in Latin script, and without it they dominate the hash.

**BM25 at k1=1.2 and b=0.75.** The standard values. Fitting them to 16 chunks
would be fitting noise, and the normalisation that actually mattered was
dividing by the ceiling the query could reach, not the saturation constant.

**Reciprocal rank fusion at k=60.** Also the usual constant. It flattens the
tail rather than the top, which is the property wanted here: the fusion should
decide between a retriever's second and third result, not overrule its first.

**The model client uses `urllib`, not `httpx`.** One POST and one GET, so the
standard library is enough and the runtime dependency list stays at three
things. `httpx` is already present for the tests, which is a different budget.

**Scheme aliases are curated into the corpus.** The documents say "PM-JAY is
also called Ayushman Bharat". Before that line existed, a romanised question
about Ayushman Bharat had no lexical anchor at all, because the document only
ever said PM-JAY. That is corpus work, not a retrieval trick, and it is exactly
what a weekly curation pass is for: the unresolved-names endpoint exists to
generate the same kind of task for drugs.

**No package install.** A venv, `demo.py` and `pytest` from the project root.
The embedding model is the only download, and it is cached after the first run.
