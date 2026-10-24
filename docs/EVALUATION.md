# Evaluation against a served model

The test suite runs generation against a scripted stand-in (`tests/conftest.py`)
that quotes its source back, so faithfulness there is 1.00 by construction and
says nothing about a 3B model. This file is the run that does say something:
the 25-question gold set through the whole stack, with `qwen2.5:3b-instruct`
served by Ollama, every answer and refusal kept.

```
ollama pull qwen2.5:3b-instruct
.venv/bin/python -m scripts.live_eval        # writes docs/results/
```

Raw output, one record per question with the answer text, the trace and the
latency: [`results/live-qwen2.5-3b-instruct.json`](results/live-qwen2.5-3b-instruct.json),
and the first run before any fixes:
[`results/live-qwen2.5-3b-instruct.baseline.json`](results/live-qwen2.5-3b-instruct.baseline.json).

Conditions worth stating up front: one run each, temperature 0, 20 answerable
and 5 out-of-scope questions, on an 8GB Apple M2 laptop rather than the clinic
GPUs. That's enough to find failure classes. It isn't enough to rank models.

## Results

| form | n | right | wrong | leaked | missed | held | faithfulness | figure recall | p50 |
|---|---|---|---|---|---|---|---|---|---|
| English | 11 | 7 | 0 | 0 | 1 | 3 | 0.97 | 1.00 | 3.2s |
| Devanagari | 9 | 7 | 0 | 0 | 1 | 1 | 0.96 | 0.67 | 4.1s |
| Romanised Hindi | 5 | 4 | 0 | 0 | 0 | 1 | 0.86 | 1.00 | 3.5s |

18 of 20 answerable questions came back from the right document, and none from
the wrong one or the wrong state. All five out-of-scope questions went to a
person. Context recall is 1.00 in every form; context precision is 1.00, 1.00
and 0.88.

*Right* means the answer cited the expected document. *Figure recall* is the
share of the figures a gold case asks for that the answer actually states,
compared by value (`analytics.figure_recall`). It was added after this run
turned up an answer that cited the right chunk and gave the wrong number,
which is described below.

## What the first run found

The baseline was 17 of 20, zero wrong, and faithfulness of 0.47 for Devanagari
and 0.50 for romanised Hindi. Reading the 25 records turned that into three
separate failures, and none of them was a wrong answer reaching a clinic.

**Hindi in, English out.** Five of the six Devanagari questions the model
answered, it answered in English. The prompt said "answer in the language of
the question", which a 3B model treats as a suggestion. The low faithfulness
followed from that: an English sentence written from a Hindi chunk shares no
words with it, so a lexical metric scores it near zero. The fix names the
language per request, derived from the detected input form ("Answer in: Hindi,
in Devanagari script"), in `generation.prompt`. After it, 7 of 7 Devanagari
answers are in Devanagari and faithfulness is 0.96.

**The digit guard blocked a correct answer.** The Jharkhand source says
"twelve hundred rupees"; the model wrote "1200". The guard compared digit
strings, found "1200" in no source, and refused. That's the safe direction to
fail in, but it was still the guard misreading a correct figure. It now
compares values: "1,200", "twelve hundred", "बारह सौ" and "१२००" are all 1200
(`guardrails.figures`). Two things stayed deliberately strict. The answer side
is only held to figures written with a digit or a scale word, because "one of"
and एक (also the Hindi article) are prose. And the change also catches
something the old guard never could: "six lakh" against a source that says
"five lakh" is now blocked, where before a figure in words passed unchecked.

**A stated "no" read as "the sources don't say".** Asked "Does Jharkhand run a
state top-up on PM-JAY?", with a source saying "Jharkhand does not run a state
top-up", the model set `insufficient` and abstained, in both English and
Hindi. One prompt rule ("a source that says something does not apply answers
the question") fixed the Hindi case. The English case still abstains.

## What the fixes cost

Forcing Hindi output has a price, and figure recall is the metric that shows it.
Asked "आयुष्मान भारत में प्रति परिवार कितना कवर मिलता है?" (how much cover per
family), the model, now writing in Hindi, answered:

> बिहार में पीएम-जय कवर प्रति परिवार प्रति वर्ष पचास हज़ार रुपये का अतिरिक्त टॉप-अप देता है।

That's the ₹50,000 cardiac and oncology top-up, not the ₹5 lakh cover. It
cites the right chunk and every figure in it is grounded, so the tally, the
citation check and the digit guard all pass it. In the baseline the same
question was answered correctly, in English. The same 3B is measurably worse
at picking the right fact when it also has to write Hindi.

I kept the Hindi fix. A reader who asked in Devanagari and gets English back
has been failed on every question, while this is one question in nine, and
it's now visible in a metric rather than hidden in a "right" count. One
regression also came with the prompt change: a Hindi referral question that
the baseline answered now abstains. Safe, and noted rather than chased.

## Found by using it, not by the gold set

The gold set has no inventory questions, because availability is answered from
the stock table and the retrieval tally can't score it. Clicking through the
client against the served model found the gap that leaves. "is insulin
available" (insulin isn't in the formulary) should come back
`not_in_formulary` and land on the curation list. Instead the 3B labelled it a
`policy` question and extracted no drug name, so it went to document retrieval
and the model abstained. It still refused, so nothing unsafe reached a clinic,
but the refusal reason was wrong and the unknown name never reached the list
that exists to fix it. The intent step needs its own labelled set, with
availability questions in all three forms, before its accuracy is a measured
claim rather than an argued one.

Two operational bugs only showed up with a real model behind the client, and
both are fixed. The page recomputed the gold-set metrics after every question,
which is about forty model calls competing with the clinic's next question on a
box built for one model; they're now computed once per process and loaded
separately from the live tables. And one request that timed out while the
model was busy flagged it unreachable for the rest of the process, so `/health`
reported it down while it was answering; a successful reply now clears that.

## Where I stopped, and why

The obvious next move is another round of prompt edits until all 20 pass.
I didn't make it. With 20 answerable questions, a prompt tuned until they all
pass is fitted to the gold set, and the next question a health worker types
is the one it wasn't fitted to. What would actually move these numbers, in
order:

1. **A larger gold set written by clinic staff**, with expected figures, so
   figure recall stops being eight cases.
2. **A 7B model for the Hindi path**, measured with this same script. The
   cover-versus-top-up miss is a reading failure, and reading is what scale buys.
3. **Figures copied, not written.** Mark the quotable spans in the corpus and
   have the model place them rather than restate them, so a figure can't drift.
4. **A prompt regression set**, so a prompt edit that fixes one question and
   breaks another (as the referral question did here) fails CI instead of
   shipping.

Latency is from a laptop sharing 8GB between the model and everything else,
so it says little about the GPU box. Medians sit at 3 to 4 seconds; p95 reached
11 seconds for Devanagari and 15 for romanised Hindi, where p95 over five
questions is really the single slowest one. I haven't profiled either. My
guess for Devanagari is token count, since Hindi script costs more tokens per
word, but that's a guess, and latency is the first number to re-measure on the
target hardware.
