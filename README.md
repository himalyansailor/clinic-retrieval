# Clinic retrieval

[![ci](https://github.com/himalyansailor/clinic-retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/himalyansailor/clinic-retrieval/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A multi-agent question-answering system for rural health clinics in India.
Staff ask about health-scheme entitlements, whether a drug is in stock, and
clinic protocols, in English, Hindi, or Hindi typed in English letters. It
answers offline, on a box with 16GB of memory and one small model, and when it
can't answer safely it says so and hands the question to a person.

![The client answering a romanised-Hindi stock question: the stock agent's dated count, the document agent's procedure, both cited, and the trace of every agent that ran](docs/assets/ui.png)

## Why this is harder than RAG over some PDFs

**The worst failure looks like success.** Bihar and Jharkhand run the same
schemes with different amounts. An answer carrying Jharkhand's figure to a
Bihar clinic is fluent, cited and formatted correctly, and it turns a patient
away. So state is a hard filter applied before scoring, and the verifier checks
it again after two agents have composed an answer, because at that point no
single agent owns all of it.

**"Is amoxicillin in stock?" isn't in any document.** It's a number counted
this morning. A single retrieval pipeline answers it with the stock-out
procedure, which reads like an answer and isn't one. That's what the agent
split is for: an LLM reads the question into one of three intents, and
`inventory` leaves the corpus for a structured lookup.

**One of the three input forms breaks the encoder.** Romanised Hindi ("dawa
stock mein nahi hai") is out of distribution for `multilingual-e5-small`; its
similarity to the right answer lands *below* an off-topic question's. No cosine
threshold can separate them, which a test pins. The fix is a second retriever,
BM25 over transliteration keys, rather than a bigger encoder that wouldn't fit
next to the generative model.

**A 3B model writes the answers, so nothing it writes is trusted.** It only sees
chunks that cleared a measured floor. Afterwards, its citation has to be one of
those chunks, every figure has to appear in a cited source (compared by value,
so "1200" matches "twelve hundred"), dosing language is blocked outright, and
the state and source dates are re-checked. If the model is down, the system
refuses rather than falls back.

## Architecture

```mermaid
flowchart LR
    Q([question]) --> G{input guards}
    G -- "emergency, dosing,<br/>injection" --> H([hand off to a person])
    G --> I["LLM call 1<br/>intent + drug name"]
    I --> R[drug resolver]
    R -- "two drugs fit" --> C([ask which one])
    R -- "unknown name" --> H
    R -- inventory --> S["stock agent<br/>formulary code → clinic count"]
    R -- "policy / clinic" --> D["document agent<br/>hybrid retrieval, state pre-filter"]
    S -- "known, not carried" --> D
    D --> W["LLM call 2<br/>write from cleared chunks only"]
    D -- "below both floors" --> H
    S --> V
    W --> V{"verifier + output guards<br/>citation · figures · dosing · state · dates"}
    V -- pass --> A([answer, dated sources, agent trace])
    V -- fail --> H
```

Two model calls per question at most, both to the same model, because a clinic
box holds one set of weights. Everything between them is plain code: the
resolver, both retrieval floors, the citation check, the figure guard, the
state verifier. That's what keeps the failure modes enumerable, and it's why
the routing is asserted in tests rather than sampled. There's no orchestration
framework; the graph has eight stages and no cycles, so LangGraph would be the
largest dependency in a project that otherwise has three.

The model sits behind an OpenAI-compatible URL: Ollama on a laptop, vLLM on the
clinic's GPUs. Deployment is a config value, not a code path.

## Results

**Retrieval** (no generative model involved), 25-question gold set at the
measured floors:

| form | n | right | wrong | leaked | missed | held |
|---|---|---|---|---|---|---|
| English | 11 | 8 | 0 | 0 | 0 | 3 |
| Devanagari | 9 | 8 | 0 | 0 | 0 | 1 |
| Romanised Hindi | 5 | 4 | 0 | 0 | 0 | 1 |

**End to end, with `qwen2.5:3b-instruct` actually served** (details and every
answer in [docs/EVALUATION.md](docs/EVALUATION.md)):

| form | right | wrong | leaked | missed | held | faithfulness | figure recall |
|---|---|---|---|---|---|---|---|
| English | 7 | 0 | 0 | 1 | 3 | 0.97 | 1.00 |
| Devanagari | 7 | 0 | 0 | 1 | 1 | 0.96 | 0.67 |
| Romanised Hindi | 4 | 0 | 0 | 0 | 1 | 0.86 | 1.00 |

Metrics are reported per input form and never averaged, because an average
hides the form that fails. Wrong answers and handoffs are never summed, because
in a clinic a handoff costs time and a wrong answer costs more.

The first served-model run surfaced three failure classes that the test suite's
scripted model never could: Hindi questions answered in English, a figure guard
that refused "1200" because the source said "twelve hundred", and a stated "no"
read as "the sources don't say". All three are fixed and tested. Fixing the
first exposed a fourth: writing in Hindi, the 3B once reported the ₹50,000
top-up instead of the ₹5 lakh cover. The answer cited the right chunk and
passed every guard. `figure_recall` exists because of that answer, and the 0.67
above is it, reported rather than tuned away.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
ollama pull qwen2.5:3b-instruct              # optional: without it, every question escalates
.venv/bin/uvicorn clinic_retrieval.app:app   # http://127.0.0.1:8000
```

Or the whole stack in containers:

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:3b-instruct
docker compose restart api                   # it probes for the model once at startup
```

```bash
.venv/bin/python -m pytest -q                # 165 tests, ~25s, no model needed
.venv/bin/python demo.py                     # a narrated walkthrough of every path
.venv/bin/python -m scripts.evaluate         # the threshold sweeps both floors came from
.venv/bin/python -m scripts.live_eval        # the gold set against the served model
```

| env | default | |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | any OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen2.5:3b-instruct` | |
| `CLINIC_API_KEYS` | unset (open, and `/health` says so) | `key:label:state`, comma separated; scopes each key to its clinic |
| `CLINIC_DATA` | `./data` | corpus, formulary, stock and gold sets |

## The API

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"amoxicillin stock mein nahi hai to kya karna chahiye","state":"bihar"}'
```

Every response carries the answer or the refusal reason, citations with source
dates, any guard that fired, and the trace of tool calls with their arguments:

```
llm:intent  clinic, drug 'amoxicillin'
resolver    resolve_drug(question='amoxicillin') -> D-AMX-125 via phrase
router      stock+docs: procedure question naming Amoxicillin dispersible tablet via phrase
stock       check_stock(code='D-AMX-125') -> D-AMX-125, 0 on hand at BR-CL-114
docs        answer_from_documents(query='...') -> generated from drugs#2,drugs#1 at 0.79x floor
verifier    passed, 3 dated source(s)
```

| endpoint | |
|---|---|
| `POST /ask` | answer, clarifying question, or refusal with one of 14 reasons |
| `GET /health` | index size, both floors, the model and whether it's actually served |
| `GET /analytics/live` | this process's questions per input form: abstain rate, p95, refusals by reason |
| `GET /analytics/ragas` | context recall and precision, faithfulness, answer relevance; no judge model |
| `GET /analytics/unresolved` | drug names nothing matched, with the nearest miss, for curating aliases |
| `GET /tools` | each agent tool, the data it may see, and its function-calling schema |
| `GET /docs` | OpenAPI |

## Design decisions worth arguing about

**The intent model picks a data source and nothing else.** It doesn't rank,
retrieve, or set a threshold. Three labels, and only `inventory` changes where
the answer comes from, so confusing `policy` with `clinic` costs nothing.

**Every branch has an owning agent.** On a procedure question the document agent
owns the answer; if nothing clears its floor, the system refuses even though the
stock agent had a number to offer.

**Three outcomes for a drug name, because two is a lie.** "We don't know that
drug" and "we know it and this clinic has none" are different answers, and a
boolean lookup reports the second as the first. A name that fits two drugs asks
which, because resolving a misspelling to the wrong drug is the one failure
worse than refusing. The resolver's thresholds are swept against 35 misspellings,
brand names and shared words, and nothing resolves to the wrong drug anywhere in
the usable band.

**Both retrieval floors are measured.** A cosine and a normalised BM25 sum can't
share a threshold, so each floor sits in the gap between the highest
out-of-scope score and the lowest correct one in its own space: 0.840 to 0.871
for the encoder, 0.011 to 0.102 for BM25.

**One thing is remembered between questions**: a clarification that was asked
and not yet answered, bound to one clinic, for ten minutes. No history and
nothing about a patient, so there's no conversation store to secure.

**A judge model is deliberately absent.** The only model on the box is the 3B
that wrote the answer, and grading its own faithfulness is marking its own
homework. All four RAGAS metrics are computed against the gold labels and the
retrieved set instead.

Every one of these, with the alternatives and what would change my mind, is in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Documentation

| | |
|---|---|
| [EVALUATION.md](docs/EVALUATION.md) | the served-model run: results, the failures it found, what the fixes cost, and where I stopped tuning |
| [ORCHESTRATION.md](docs/ORCHESTRATION.md) | how the orchestrator works, what's remembered, how agents reach tools, the security and governance model |
| [LLD.md](docs/LLD.md) | components, data contracts, every invariant with the test that pins it, the failure taxonomy |
| [DECISIONS.md](docs/DECISIONS.md) | 21 decisions, each with the options considered and what would change it |
| [TRADEOFFS.md](docs/TRADEOFFS.md) | what a compressed timeline would cut, what it can't, and the decisions that can't be deferred |
| [SOURCES.md](data/SOURCES.md) | the real public sources a production corpus would come from, and the gap that matters |

## Layout

```
clinic_retrieval/
  app.py              FastAPI app, lifespan, routers, the client mount
  agents/             intent.py, drugs.py (resolver + stock), docs.py, orchestrate.py
  retrieval/          embed.py (dense), lexical.py (BM25), index.py (hybrid + floors)
  generation.py       the prompt, and mapping citations back to chunks
  guardrails.py       input and output rules, including the figure guard
  llm.py              the one model, over an OpenAI-compatible URL
  tools.py            the declared tools and the data each may see
  analytics.py        query log, RAGAS without a judge, figure recall, gold runners
  lang.py             Devanagari to Latin, transliteration keys, input-form detection
  security.py         API keys, clinic scope, audit actor
  session.py          the one pending clarification
  static/             the client: one page, plain JS, no CDN
data/                 synthetic corpus, formulary, stock counts, gold sets
scripts/              evaluate.py, live_eval.py, ingest.py
tests/                one file per area, plus the scripted model in conftest.py
docs/                 design docs and evaluation results
```

## Scope

Built: two states (Bihar and Jharkhand) plus national documents, two schemes
(PM-JAY and Janani Suraksha Yojana), a 14-drug formulary, and three clinic
protocols. A third state adds corpus, not design; the filter, the stock table
and the evaluation are already per state.

Not built, on purpose: a re-ranker, corpus sync, a judge model, and serving
the model inside this process. [TRADEOFFS.md](docs/TRADEOFFS.md) explains which
of those I'd add first.

The corpus, formulary and stock counts are synthetic, written for this project.
None of the figures in them are real scheme, clinical or dispensing guidance.

## License

MIT
