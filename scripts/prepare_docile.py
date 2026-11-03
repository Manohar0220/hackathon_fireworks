"""Prepare the fixed 50-document development pilot without changing originals."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from invoiceroute.preparation import VERSION, convert_document, read_json, training_example


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_pages(records, output):
    # Inspection-only tools were installed locally during the initial PDF sample.
    sys.path.insert(0, str(ROOT / "tmp/docile_tools"))
    import pymupdf
    from PIL import Image, ImageDraw, ImageFont
    previews = output / "previews"
    previews.mkdir(exist_ok=True)
    thumbnails = []
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    for index, record in enumerate(records, 1):
        with pymupdf.open(record["source"]["paths"]["pdf"]) as document:
            if len(document) != 1:
                raise ValueError("Pilot contains a document outside single-page scope")
            page = document[0]
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6))
            path = previews / f"{index:02d}-{record['document_id']}.png"
            pixmap.save(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((900, 1250))
            tile = Image.new("RGB", (940, 1320), "white")
            draw = ImageDraw.Draw(tile)
            draw.text((12, 8), f"{index:02d}  {record['document_id']}", font=font, fill="black")
            tile.paste(image, ((940 - image.width) // 2, 55))
            draw.rectangle((0, 0, 939, 1319), outline="#888888", width=2)
            thumbnails.append(tile)
    for start in range(0, len(thumbnails), 4):
        sheet = Image.new("RGB", (1880, 2640), "#dddddd")
        for index, thumbnail in enumerate(thumbnails[start:start + 4]):
            sheet.paste(thumbnail, ((index % 2) * 940, (index // 2) * 1320))
        sheet.save(previews / f"sheet-{start // 4 + 1:02d}.png")


def review_html(records, reviews, output):
    sections = []
    for index, record in enumerate(records, 1):
        identifier = record["document_id"]
        rows = []
        for field, label in record["header_labels"].items():
            raw = " | ".join(dict.fromkeys(candidate["raw_text"] for candidate in label["candidates"]))
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in
                        (field, label["candidate_value"], raw, label["status"], ", ".join(label["issues"]))) + "</tr>")
        decision = reviews.get(identifier, {"status": "pending"})
        sections.append(f'<article id="doc-{index}"><h2>{index:02d} - {identifier}</h2>'
                        f'<p>Development data only. Review: {html.escape(decision["status"])}</p>'
                        f'<div class="pair"><a href="previews/{index:02d}-{identifier}.png"><img src="previews/{index:02d}-{identifier}.png" alt="Original invoice page"></a>'
                        f'<div><table><thead><tr><th>Field</th><th>Candidate</th><th>Raw annotation</th><th>Status</th><th>Issues</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
                        f'<h3>Review notes</h3><pre>{html.escape(json.dumps(decision,indent=2,ensure_ascii=False))}</pre>'
                        f'<h3>OCR supplied to the model</h3><pre>{html.escape(record["input"]["text"])}</pre>'
                        f'<h3>Line-item candidates</h3><pre>{html.escape(json.dumps(record["line_items"],indent=2,ensure_ascii=False))}</pre></div></div></article>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>InvoiceRoute pilot review</title><style>body{font:15px Arial;margin:24px;color:#17212b}article{border-top:2px solid #64748b;margin:32px 0;padding-top:12px}.pair{display:grid;grid-template-columns:minmax(400px,48%) 1fr;gap:20px}img{width:100%;position:sticky;top:10px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #aaa;padding:6px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f4f6;padding:12px}h1{color:#164e63}</style><h1>InvoiceRoute - 50-document development pilot</h1><p>Original PDFs, annotations, and OCR are unchanged. Candidate labels require explicit review. Partial-header training exports request only approved fields. This is not a benchmark.</p>' + "".join(sections) + '</html>'
    (output / "review.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data/prepared/docile_pilot_v1")
    args = parser.parse_args()
    output = args.output.resolve()
    source = ROOT / "data/docile"
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Derived output must be separate from the original data")
    selection = read_json(output / "selection.json")
    training_ids = set(read_json(source / "train.json"))
    documents = selection["documents"]
    if len({document["document_id"] for document in documents}) != len(documents):
        raise ValueError("Duplicate pilot IDs")
    if any(document["document_id"] not in training_ids for document in documents):
        raise ValueError("Pilot must use original training documents only")
    records = [convert_document(source, document["document_id"]) for document in documents]
    reviews_path = output / "review_decisions.json"
    reviews = read_json(reviews_path) if reviews_path.exists() else {}
    if set(reviews) - {record["document_id"] for record in records}:
        raise ValueError("Review references documents outside the pilot")
    examples, provenance = [], []
    for record in records:
        decision = reviews.get(record["document_id"], {})
        if decision:
            if decision.get("source_sha256") != record["source"]["sha256"]:
                raise ValueError("Review is stale or missing source hashes")
            record["review_status"] = decision.get("status", "pending")
            result = training_example(record, decision)
            if result:
                example, audit = result
                examples.append(example)
                provenance.append(audit)
                record["training_eligible"] = True
                record["approved_header_fields"] = audit["fields"]
                record["reviewed_header_targets"] = json.loads(example["messages"][2]["content"])
        record["full_schema_training_eligible"] = False
    for name, rows in (("pilot_candidates.jsonl", records), ("pilot_header_sft.jsonl", examples), ("pilot_header_sft_provenance.jsonl", provenance)):
        with (output / name).open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = {}
    for field in records[0]["header_labels"]:
        fields[field] = dict(Counter(record["header_labels"][field]["status"] for record in records))
    issues = Counter(issue for record in records for label in record["header_labels"].values() for issue in label["issues"])
    summary = {"version": VERSION, "pilot_documents": len(records), "unique_layout_clusters": len({record["source"]["cluster_id"] for record in records}),
               "selection_strata": dict(Counter(document["stratum"] for document in documents)),
               "header_candidate_status_counts": fields, "header_issues": dict(issues),
               "reviewed_documents": sum(record["review_status"] == "reviewed" for record in records),
               "partial_header_sft_examples": len(examples),
               "approved_field_targets": sum(len(row["fields"]) for row in provenance),
               "full_schema_sft_examples": 0,
               "line_item_rows": sum(len(record["line_items"]) for record in records),
               "limitations": ["Development pilot, not a held-out benchmark.", "Unannotated fields remain unknown, not absent or zero.", "SFT exports request only explicitly reviewed fields; full-schema and line-item exports are pending coverage/completeness review.", "No model was called and no training was launched."]}
    write_json(output / "audit_summary.json", summary)
    if args.render:
        render_pages(records, output)
    review_html(records, reviews, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
