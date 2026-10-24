import json

from clinic_retrieval.core import Chunk
from clinic_retrieval.retrieval.index import load
from scripts.ingest import read_sections, write_doc, write_drugs


def test_headings_become_sections():
    sections = read_sections("# Cover limit\nFive lakh.\n\n# Eligibility\nSECC listed.")
    assert [title for title, _ in sections] == ["Cover limit", "Eligibility"]
    assert sections[0][1] == "Five lakh."


def test_paragraphs_are_used_when_there_are_no_headings():
    sections = read_sections("Cover limit.\nFive lakh per family.\n\nEligibility.\nSECC.")
    assert len(sections) == 2
    assert sections[0][0] == "Cover limit"


def test_an_ingested_document_loads_as_chunks(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# Amount in Kerala\nThe scheme pays eight hundred rupees.")
    out = tmp_path / "corpus"
    write_doc(source, "jsy-kerala", "kerala", "2026-04-01",
              "SAMPLE for a test", out_dir=out)

    chunks = load(out)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert chunk.doc == "jsy-kerala"
    assert chunk.state == "kerala"
    assert chunk.updated == "2026-04-01"
    assert "eight hundred" in chunk.text


def test_a_drug_csv_merges_into_the_formulary(tmp_path):
    csv_file = tmp_path / "drugs.csv"
    csv_file.write_text("code,generic,cold_chain,aliases\n"
                        "D-NEW-001,Ciprofloxacin tablet,no,cipro|सिप्रो\n")
    out = tmp_path / "formulary.json"
    out.write_text(json.dumps({"source": "test", "drugs": [
        {"code": "D-OLD-001", "generic": "Old drug", "cold_chain": False,
         "aliases": []}]}))

    write_drugs(csv_file, out=out)
    drugs = json.loads(out.read_text())["drugs"]
    assert [d["code"] for d in drugs] == ["D-NEW-001", "D-OLD-001"]
    assert "सिप्रो" in drugs[0]["aliases"]


def test_the_ingested_formulary_is_resolvable(tmp_path):
    from clinic_retrieval.agents.drugs import DrugResolver, load_formulary

    csv_file = tmp_path / "drugs.csv"
    csv_file.write_text("code,generic,cold_chain,aliases\n"
                        "D-NEW-001,Ciprofloxacin tablet,no,cipro|सिप्रो\n")
    out = tmp_path / "formulary.json"
    write_drugs(csv_file, out=out)

    resolver = DrugResolver(load_formulary(out))
    assert resolver.resolve("is cipro available").best.code == "D-NEW-001"
    assert resolver.resolve("क्या सिप्रो है").best.code == "D-NEW-001"
