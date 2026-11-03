"""Offline token-volume estimates; does not upload data or launch training."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from invoiceroute.preparation import read_json, source_segments
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def main():
    tokenizer_repo = "zai-org/GLM-5.3-Flash"
    tokenizer_path = hf_hub_download(tokenizer_repo, "tokenizer.json", local_files_only=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    catalog = [json.loads(line) for line in (ROOT / "data/docile_catalog.jsonl").read_text(encoding="utf-8").splitlines()]
    train = [row for row in catalog if row["official_split"] == "train"]
    groups = defaultdict(list)
    prompt = "Extract the invoice annotations from the following OCR text. Return JSON.\n\n"
    for index, row in enumerate(train, 1):
        annotation = read_json(ROOT / "data" / row["annotation_path"])
        ocr = read_json(ROOT / "data" / row["ocr_path"])
        source = source_segments(ocr)["text"]
        full_target = json.dumps(annotation, ensure_ascii=False, separators=(",", ":"))
        # Alternative estimate only: retains native field names and values,
        # without geometry, header labels, or annotation-only metadata.
        labels = {key: [{field: value for field, value in label.items() if field in {"fieldtype", "text", "line_item_id"}} for label in annotation[key]]
                  for key in ("field_extractions", "line_item_extractions")}
        labels_target = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
        source_tokens = len(tokenizer.encode(prompt + source, add_special_tokens=False).ids)
        full_tokens = len(tokenizer.encode(full_target, add_special_tokens=False).ids)
        labels_tokens = len(tokenizer.encode(labels_target, add_special_tokens=False).ids)
        observation = {"input_tokens": source_tokens, "full_annotation_target_tokens": full_tokens,
                       "native_extraction_labels_target_tokens": labels_tokens}
        groups["all_official_training_documents"].append(observation)
        if row["document_type"] == "tax_invoice":
            groups["all_training_tax_invoices"].append(observation)
        if row["candidate_english_single_page_tax_invoice"]:
            groups["single_page_english_training_tax_invoices"].append(observation)
        if index % 1000 == 0:
            print(f"Counted {index:,}/{len(train):,} original training documents", flush=True)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "tokenizer": tokenizer_repo,
              "tokenizer_revision": Path(tokenizer_path).parent.name,
              "assumptions": ["One epoch; no validation or test documents included.",
                              "Content token counts only; provider chat rendering, packing, evaluation, retries and checkpoints add overhead.",
                              "Original annotation values are unchanged; full annotation JSON is minified for counting.",
                              "Native extraction labels variant is an estimate, not a prepared or approved training file.",
                              "Throughput scenarios are assumptions, not measured Fireworks performance or provider ETAs.",
                              "No training, inference, upload or external API requests were performed by this estimator."], "groups": {}}
    for name, rows in groups.items():
        fields = {field: sum(row[field] for row in rows) for field in rows[0]}
        full_sequences = [row["input_tokens"] + row["full_annotation_target_tokens"] for row in rows]
        labels_sequences = [row["input_tokens"] + row["native_extraction_labels_target_tokens"] for row in rows]
        full_total, labels_total = sum(full_sequences), sum(labels_sequences)
        report["groups"][name] = {"documents": len(rows), **fields,
            "full_annotation_total_tokens": full_total,
            "labels_only_total_tokens": labels_total,
            "full_annotation_sequence_max_tokens": max(full_sequences),
            "full_annotation_sequence_median_tokens": statistics.median(full_sequences),
            "full_annotation_sequences_over_65536": sum(value > 65536 for value in full_sequences),
            "labels_only_sequence_max_tokens": max(labels_sequences),
            "full_annotation_compute_minutes_at_assumed_tokens_per_second": {str(rate): round(full_total / rate / 60, 1) for rate in (1000, 2000, 5000, 10000)},
            "labels_only_compute_minutes_at_assumed_tokens_per_second": {str(rate): round(labels_total / rate / 60, 1) for rate in (1000, 2000, 5000, 10000)},
        }
    output = ROOT / "data/fireworks/full_dataset_token_estimate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
