"""Turn a downloaded document or drug list into the formats this repo reads.

    python3 scripts/ingest.py doc --in notes.md --doc jsy-bihar \
        --state bihar --updated 2026-03-14 --source "https://..."

    python3 scripts/ingest.py drugs --in nlem.csv

Deliberately no PDF handling and no network. Government sources are PDFs, and
turning a PDF into trustworthy text is a step a person has to look at: run
pdftotext, read what came out, keep the page number. This script is what
happens after that, and it is the boring half on purpose.

CSV columns for `drugs`: code, generic, cold_chain, aliases (pipe-separated).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinic_retrieval.core import CORPUS, FORMULARY  # noqa: E402

HEADING = re.compile(r"^#{1,3}\s+(.+)$")


def read_sections(text: str) -> list[tuple[str, str]]:
    """Markdown headings if there are any, otherwise blank-line paragraphs."""
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        found = HEADING.match(line.strip())
        if found:
            sections.append((found.group(1).strip(), []))
        elif sections:
            sections[-1][1].append(line)

    if sections:
        return [(title, "\n".join(body).strip())
                for title, body in sections if "\n".join(body).strip()]

    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if block:
            first, _, rest = block.partition("\n")
            out.append((first.strip().rstrip(".")[:80], (rest or first).strip()))
    return out


def write_doc(source_file: Path, doc: str, state: str, updated: str,
              source: str, out_dir: Path = CORPUS) -> Path:
    sections = read_sections(source_file.read_text(encoding="utf-8"))
    if not sections:
        raise SystemExit(f"nothing to ingest from {source_file}")

    lines = ["---", f"doc: {doc}", f"state: {state}", f"updated: {updated}",
             f"source: {source}", "---", ""]
    for title, body in sections:
        lines += [f"# {title}", "", body, ""]

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{path}: {len(sections)} sections")
    return path


def write_drugs(source_file: Path, out: Path = FORMULARY) -> Path:
    existing = (json.loads(out.read_text(encoding="utf-8"))
                if out.exists() else {"source": "", "drugs": []})
    by_code = {d["code"]: d for d in existing["drugs"]}

    added = 0
    with source_file.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("code") or "").strip()
            generic = (row.get("generic") or "").strip()
            if not code or not generic:
                continue
            aliases = [a.strip() for a in (row.get("aliases") or "").split("|")
                       if a.strip()]
            added += code not in by_code
            by_code[code] = {
                "code": code,
                "generic": generic,
                "cold_chain": (row.get("cold_chain") or "").strip().lower()
                in ("1", "true", "yes"),
                "aliases": aliases,
            }

    existing["drugs"] = [by_code[code] for code in sorted(by_code)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"{out}: {len(by_code)} drugs, {added} new")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="what", required=True)

    doc = sub.add_parser("doc", help="a scheme or protocol document")
    doc.add_argument("--in", dest="source_file", type=Path, required=True)
    doc.add_argument("--doc", required=True, help="corpus id, e.g. jsy-bihar")
    doc.add_argument("--state", required=True, help="a state, or 'all'")
    doc.add_argument("--updated", required=True, help="the source's own date")
    doc.add_argument("--source", required=True, help="where it came from")

    drugs = sub.add_parser("drugs", help="a formulary CSV")
    drugs.add_argument("--in", dest="source_file", type=Path, required=True)

    args = parser.parse_args()
    if args.what == "doc":
        write_doc(args.source_file, args.doc, args.state, args.updated, args.source)
    else:
        write_drugs(args.source_file)


if __name__ == "__main__":
    main()
