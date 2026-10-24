# Scoping under a compressed timeline

This file holds the decisions a deadline changes. The engineering reasoning
is in [DECISIONS.md](DECISIONS.md), deliberately kept apart, because a cut made
for a deadline is a different kind of argument from a choice made on the
merits, and the two should be reviewable separately. If the date moves, this
file changes and that one mostly does not.

The scenario I scoped against is the common one: a pilot timeline shortened by
a third, a stakeholder demo due before the build is finished, and three cuts
on the table that every AI project gets offered at that point: smaller
embedding models, no re-ranking, and minimal monitoring.

## Where I agree

Two of the three are fine.

**Smaller embedding models.** `multilingual-e5-small` is 384 dimensions and
about 120MB of weights, runs on CPU, and is in the build. The cut I would not
accept is a smaller *English* model, because that is not a size decision, it is
dropping Hindi. Measuring the small model also turned up the one thing worth
knowing early: Hindi typed in Latin script is out of its distribution and
scores *below* where an out-of-scope question scores, so no threshold separates
them. That cost a second retrieval signal, which is cheaper than the model that
would have covered it. Size is negotiable, the shared Hindi-English space is
not.

**No re-ranking.** A cross-encoder is a real precision gain and it is the first
thing I would cut. It gets paid for by keeping the floors high: fewer answers,
not more wrong ones. The memory argument makes it easy anyway, since the box is
already holding a generative model.

## Where I don't

**Minimal monitoring, if monitoring means the behaviour tests or the guards.**
The tests are assertions in a text file and they take an afternoon. The guards
are a single module of patterns. They are also the only thing that catches the
cross-state leak, and that failure is invisible from the output: a Bihar clinic
gets a confident, cited, correctly formatted answer carrying Jharkhand's
amount. Without the test nobody finds it until a patient is turned away.

The case got stronger once a model started writing the answers. Three of those
guards could not fire under extractive quoting: a citation outside the
retrieved set, a digit with no source, dosing language in the output. All three
can fire now. Cutting them would have meant shipping generation with nothing
between a fluent paragraph and a clinic.

Dashboards, tracing and judge-based drift scoring can wait. The gate cannot.

## What the early demo is for

Stakeholders need to see it work, not see it finished. A demo that answers
confidently and wrongly on one of their own questions costs more than a
demo that abstains twice and says why. So the demo build keeps every
correctness property and spends the saved time on nothing else.

| | Demo build (this repo) | Full pilot |
|---|---|---|
| Orchestration | hand-written, eight stages, two of them model calls | LangGraph once there is resumable state to persist |
| Reading the question | one model call, three labels, validated against a fixed set | measured against a labelled set of real questions, with a confusion matrix |
| Tool use | a declared registry, the orchestrator calling the tools | the model calling them natively, with the resolver still between a typed name and a lookup |
| Guardrails | deterministic rules, in and out | plus an LLM second opinion that can escalate but never clear |
| Answers | one model call, shown only chunks that cleared a floor, then checked | plus NLI entailment, and quotable spans marked in the corpus so figures are quoted and only the connective language is written |
| Retrieval | hybrid dense plus BM25 over transliteration keys, flat scan | `sqlite-vec` plus an FTS5 table, same two signals |
| Re-ranking | none, floors kept high | cross-encoder, floors lowered back to what the precision earns |
| Embeddings | `e5-small` for the dense half | benchmark MuRIL, which is trained on transliterated text |
| Model serving | an OpenAI-compatible URL: Ollama on a laptop, vLLM on the clinic GPUs | `llama.cpp` packaged per clinic box, with a model upgrade path |
| Analytics | per-form log, RAGAS metrics computed without a judge | judged scoring on a nightly corpus run against a larger model, traces in OpenTelemetry |
| States | two, with the filter proven | all three |
| Eval | deterministic gate, per form, plus a scripted model so the pipeline is testable without weights | the same gate against a served model, plus a staff-written gold set and a prompt regression set |
| Sync | manual corpus drop | pull-based delta with supervisor conflict review |
| Drug names | resolver plus the curated alias list, unresolved names logged | strengths and dosage forms, and a curation screen instead of a JSON endpoint |
| Memory | the pending clarification, in process, ten minutes | Redis, same shape and same TTL |
| Access control | key to clinic scope, open by default on a laptop | fail closed, keys per device, mTLS, durable audit |
| Monitoring | the behaviour tests | tracing, dashboards, per-clinic abstain rates |

## Non-negotiable at any schedule

These ten are not features, so there is no version of the timeline where they
get descoped. All ten are asserted in tests.

1. State is a hard pre-filter. Three states, three entitlement amounts.
2. Every answer cites one of the chunks the model was shown, every figure in it
   appears in a cited source, and it carries that source's date.
3. Below both floors, abstain and hand off. No hedged answer, no relaxed retry.
4. A live-stock question is never answered from a document.
5. Emergencies and dosing questions go to a person without reaching the model,
   and identifiers are redacted before anything is stored.
6. A drug name never resolves to the wrong drug. Ambiguity asks.
7. Nothing about a patient is remembered between questions.
8. The clinic a caller may query is enforced twice, at the gate and in
   retrieval.
9. Evaluation is reported per language and never averaged.
10. A model that cannot be reached, or that answers with something the system
    does not recognise, escalates. There is no path that answers without it.

## Decisions that cannot be deferred

A compressed build is mostly a question of which mistakes are cheap to fix
after the demo. These are the ones that are not, so they get decided now even though
nothing forces them to be.

| Decision | Why it cannot wait |
|---|---|
| State is a filter, not a ranking signal | Chunks carry state because retrieval needs it before scoring. Changing this later is a retrieval rewrite plus re-validating every answer already given to a clinic |
| Refusing is a return value, not an error | Every caller is written against the response shape. A system that always answers cannot learn to abstain without changing all of them |
| Generation is constrained and checked, never trusted | The checks decide what the corpus has to look like: chunks need stable ids to cite, figures need to sit in the text that gets retrieved. Adding the checks after a corpus is built around an unchecked generator is a corpus migration |
| Every answer carries its source's date | Provenance that was not captured on day one cannot be added to answers already given, and an offline corpus is stale by definition |
| Evaluation is per language from the first run | Weeks of averaged numbers are weeks with no per-language history to compare against. The baseline has to be shaped right before there is a baseline |
| There is no answer path that skips the model | A fallback that answers when the model is down is a second behaviour to evaluate, and the one nobody exercises. Removing it later means removing something a clinic has already come to rely on |
| Guard precedence, emergency first | Cheap to decide now. After an incident it is a policy review with the clinic network, not a code change |
| The embedding family is multilingual, not English-only | The specific model is a swap. A corpus and a floor built around an English-only space is not |
| Retrieval has a lexical signal, not only a dense one | This came from measurement, not taste: one of the three input forms cannot be scored by an encoder at all. It sets the index layout, which is two structures over the same chunks, and the corpus work that feeds it: scheme aliases written into the documents so the lexical side has something to match |
| The formulary is separate data from the stock count | Different owners and different clocks: a national list revised rarely, counts that change every shift. Joining them by code later means re-keying every clinic file that by then has names typed into it |
| A drug name resolves to one of three outcomes | A boolean that cannot say "known, not carried here" is not a threshold to tune later, it is a different return type through the router, the orchestrator and the API |
| Each branch has an owning agent | Which agent's refusal refuses the whole answer is not a tuning knob. Get it wrong and a procedure question nobody can find an answer to comes back with a stock count attached, which reads like an answer |
| What the system may remember | A conversation store added later is a schema, a retention period and a consent conversation. Deciding now that only an outstanding question is kept means none of those exist to get wrong |
| The authorisation boundary is the retrieval boundary | Two controls on the same line, chosen now because retrofitting authorisation around a filter that already shipped means re-testing every answer path |
| Capabilities are declared, not called | The declaration is what the audit trail and the agent-sequence tests are built on. Adding it later means rewriting the orchestrator and the tests together |
| The model never dispatches a tool call directly | The reason drug lookup is safe is that the code always comes from the formulary via the resolver. A registry the model can call directly, added later, has to re-establish that guarantee against every tool at once |

And the mirror of that list, the things that are one-file swaps and therefore
deliberately not decided yet: which embedding model, which generative model,
which serving engine, whether a re-ranker, which vector store, whether an
orchestration framework, where traces go.

One of the rows above used to read differently. An earlier version of this file
said answers had to quote their source, and argued that quoted-to-generated was
additive later while generated-back-to-quoted would be a visible regression.
That is the move that then happened, in the direction the file said was safe,
and it happened because quoting could not answer a Hindi question in Hindi. The
checks it predicted would be needed were already built.

## If the timeline halved again

The cut is the corpus, not the machinery. One state, one scheme family, and the
formulary trimmed to the drugs those clinics actually dispense, which is a list
the clinic network can produce in a day. Everything in the non-negotiable list above stays,
because none of it is more than a few days of work and all of it is expensive to
retrofit. What goes is the second state, the tuning for the third input form,
and the curation screen, which stays a JSON endpoint somebody reads.

The thing I would not compress is the week with field staff writing questions and
naming drugs the way they actually name them. That week is what makes the alias
list real, and no amount of engineering substitutes for it. It is also what
would turn the intent step from an argued design into a measured one.

## What I would do first with four more weeks

1. Re-measure on the target GPUs, and with a 7B for the Hindi path. The first
   served-model run ([EVALUATION.md](EVALUATION.md)) found three failure
   classes in an afternoon that the scripted stand-in could never have shown.
2. Have clinic staff write the gold set, and label intents while they are at it.
   Mine is a guess at what gets asked.
3. A prompt regression set, so a prompt edit cannot change behaviour quietly.
4. Add the cross-encoder re-ranker, then lower both floors to what the better
   precision earns back.
5. NLI entailment behind the digit guard, which is the gap the digit guard
   cannot see: a paraphrase that changes a condition without changing a number.

## The honest part

Romanised Hindi used to be 2 of 4 in the eval and I did not tune it away; the
overall 18 of 20 read healthy while the form most clinic staff type in sat at
half. The hybrid retriever fixed it properly and it is 4 of 4 now, and the
reason the fix was the right one rather than a lower floor for Hindi is that
the per-form table said where the problem was.

The equivalent honest number today is figure recall in Devanagari: 0.67.
Running the gold set against a served 3B fixed two real problems, Hindi
questions answered in English and a digit guard that blocked "1200" because the
source said "twelve hundred", and then showed what the language fix cost: on one
question in nine the model, writing Hindi, reported the top-up instead of the
cover. It cites the right chunk and passes every guard. The tally calls it
right. Only a check on the figure itself sees it, which is why that check now
exists, and why the details are in [EVALUATION.md](EVALUATION.md) rather than
averaged into a headline. The intent step is still the same shape of gap as
before: correct on the gold set, argued from the design, not yet measured
against questions a health worker wrote.
