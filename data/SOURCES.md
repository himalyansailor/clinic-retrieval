# Where real data would come from

Everything in `data/` today is synthetic and labelled as such. This is what I
checked for real sources, what each one actually gives, and the gap that matters.
Checked 20 Oct 2026.

## Usable now

**National List of Essential Medicines 2022** (Ministry of Health and Family
Welfare). 384 drugs in 27 therapeutic categories, tiered P, S and T for
primary, secondary and tertiary care. This is the right source for
`data/formulary/formulary.json`, and the P tier is close to what a rural clinic
actually holds. Published as a PDF, so the table needs extracting once and
checking by a person; after that `scripts/ingest.py drugs` takes a CSV of
`code, generic, cold_chain, aliases`. No official CSV exists that I could find.

**Indian Public Health Standards 2022** (MoHFW, four volumes, released 16 April
2022). Volume III covers Health and Wellness Centres and Primary Health
Centres, 128 pages. This is the right source for the protocol documents:
referral criteria, cold chain, record keeping. PDF again.

**data.gov.in**, with a free API key and JSON or CSV responses. Real and
maintained, and it has PM-JAY state-wise authorised admissions and
hospital-empanelment counts. Worth being clear about what that is: aggregate
statistics for dashboards, not the entitlement text a health worker asks about.
Useful for a different product, not for this retrieval corpus. Check the
Government Open Data Licence terms per resource.

## Useful for testing, not as the corpus

**`OdiaGenAI/health_hindi_200`** on Hugging Face. Public, no auth, 200 Hindi
medical question-answer pairs in one JSON file. Verified it exists and is
ungated. Two caveats: it states no licence at all, so it does not get vendored
into this repo, and 200 general medical pairs are not scheme questions. What it
is good for is a smoke test that Devanagari retrieval behaves on text somebody
else wrote.

**`ai4bharat` IndicQA (Hindi)**. Extractive QA in Devanagari, not health
specific. Same use: an independent check that the tokenisation and the
transliteration keys hold up on text I did not write.

## The gap, which is the point

There is no public, structured, bilingual corpus of **per-state scheme
entitlements**. Janani Suraksha Yojana pays different amounts in different
states, and that difference lives in state health society circulars and NHM
state PDFs, not in any dataset. The same is true of state top-ups on PM-JAY.

Three consequences, and all three are already in the design.

The per-state variation has to be curated per state, by someone who can read a
circular. That is why `TRADEOFFS.md` says the corpus is the cost and the
machinery is not, and why a compressed timeline takes a state out rather than
taking quality out.

Provenance has to survive ingestion, because the only way anyone trusts a
curated answer is the document and date behind it. Every chunk carries `source`
and `updated` from the front matter, and every answer carries the date out.

PDF extraction is a human-reviewed step, not a pipeline stage. `scripts/ingest.py`
starts after `pdftotext`, on purpose: a silently mangled entitlement amount is
exactly the failure this whole system is built to avoid.

## Sources

- NLEM 2022 announcement and contents: https://www.biospectrumindia.com/public/article/pdf/21947
- NLEM 2022 summary: https://biovoicenews.com/384-drugs-included-in-nlem-2022-34-new-drugs-added/
- IPHS 2022 Volume III (mirror with description): https://ruralindiaonline.org/en/library/resource/indian-public-health-standards-2022-volume-iii-health-and-wellness-centre-primary-health-centre/
- PM-JAY authorised admissions by state: https://www.data.gov.in/resource/stateuts-wise-detail-authorized-hospital-admissions-under-ayushman-bharat-pradhan-mantri-0
- Hospitals empanelled under AB-PMJAY: https://odisha.data.gov.in/resource/state-wise-details-total-number-public-and-private-hospitals-empanelled-north-east-and
- Hindi medical QA set: https://huggingface.co/datasets/OdiaGenAI/health_hindi_200
- Hindi extractive QA: https://huggingface.co/datasets/sepiatone/ai4bharat-indicqa-hi-202410
