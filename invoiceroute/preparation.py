"""Conservative, offline conversion of DocILE labels into review candidates.

This module never writes into the source dataset. A candidate is not a reviewed
answer. Unknown annotation coverage is kept distinct from verified absence.
"""

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import unicodedata


VERSION = "docile-pilot-v1"
HEADER_MAP = {
    "vendor": "vendor_name", "invoice_number": "document_id",
    "invoice_date": "date_issue", "due_date": "date_due",
    "subtotal": "amount_total_net", "tax": "amount_total_tax",
    "invoice_total": "amount_total_gross", "amount_due": "amount_due",
    "amount_paid": "amount_paid",
}
ITEM_MAP = {
    "description": "line_item_description", "quantity": "line_item_quantity",
    "unit_price_net": "line_item_unit_price_net",
    "unit_price_gross": "line_item_unit_price_gross",
    "line_total_net": "line_item_amount_net",
    "line_total_gross": "line_item_amount_gross",
}
MONEY_FIELDS = {"subtotal", "tax", "invoice_total", "amount_due", "amount_paid",
                "unit_price_net", "unit_price_gross", "line_total_net", "line_total_gross"}
DATE_FIELDS = {"invoice_date", "due_date"}
SYSTEM_INSTRUCTION = (
    "Extract only the requested invoice header fields from the source OCR. "
    "Return one JSON object with exactly those requested keys. Use source-supported "
    "values, decimal strings for money, and ISO dates only when resolved. "
    "Do not invent amounts or repair printed arithmetic. This is a partial-header "
    "training task; fields not requested are outside this example's annotation coverage."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_key(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def normalize_decimal(raw, quantity=False):
    """Parse supported formats without replacing OCR letters or rounding values."""
    value = unicodedata.normalize("NFKC", raw).strip()
    negative = value.startswith("(") and value.endswith(")")
    if negative:
        value = value[1:-1]
    value = re.sub(r"\b(?:USD|EUR|GBP|CAD|AUD|JPY|CHF)\b", "", value, flags=re.I)
    value = re.sub(r"[\s$€£¥'’]", "", value)
    # A final sentence dot after an already complete two-decimal amount is
    # punctuation, e.g. OCR '78,95.'; never substitute letters for digits.
    if re.fullmatch(r"[+-]?[\d,.]+[.,]\d{2}\.", value):
        value = value[:-1]
    if not re.fullmatch(r"[+-]?(?:\d[\d.,]*|[.,]\d+)", value):
        return None, "invalid_number"
    if "," in value and "." in value:
        decimal_mark = "," if value.rfind(",") > value.rfind(".") else "."
        group_mark = "." if decimal_mark == "," else ","
        integer, fraction = value.rsplit(decimal_mark, 1)
        if not fraction.isdigit() or not re.fullmatch(r"[+-]?\d{1,3}(?:" + re.escape(group_mark) + r"\d{3})+", integer):
            return None, "ambiguous_number_format"
        value = integer.replace(group_mark, "") + "." + fraction
    elif "," in value:
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", value):
            if quantity:
                return None, "ambiguous_quantity_separator"
            value = value.replace(",", "")
        elif value.count(",") == 1 and len(value.rsplit(",", 1)[1]) in (1, 2):
            value = value.replace(",", ".")
        else:
            return None, "ambiguous_number_format"
    elif value.count(".") > 1:
        return None, "ambiguous_number_format"
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None, "invalid_number"
    if negative:
        number = -number
    if not number.is_finite():
        return None, "invalid_number"
    if quantity:
        normalized = format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else format(number, "f")
        return normalized or "0", None
    return format(number, ".2f") if number.as_tuple().exponent >= -2 else format(number, "f"), None


def normalize_date(raw):
    value = unicodedata.normalize("NFKC", raw).strip()
    value = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", value, flags=re.I)
    for pattern in ("%Y-%m-%d", "%B %d, %Y", "%B %d,%Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%b-%d-%Y"):
        try:
            parsed = datetime.strptime(value, pattern).date()
            return parsed.isoformat(), None
        except ValueError:
            pass
    numeric = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2}|\d{4})", value)
    if numeric:
        first, second, year = map(int, numeric.groups())
        if len(numeric[3]) == 2:
            return None, "two_digit_year_requires_review"
        if first <= 12 and second <= 12 and first != second:
            return None, "ambiguous_day_month"
        month, day = (second, first) if first > 12 else (first, second)
        try:
            return date(year, month, day).isoformat(), None
        except ValueError:
            return None, "invalid_date"
    return None, "date_requires_review"


def normalize_value(field, raw):
    if field in MONEY_FIELDS or field == "quantity":
        return normalize_decimal(raw, quantity=field == "quantity")
    if field in DATE_FIELDS:
        return normalize_date(raw)
    value = re.sub(r"\s+", " ", raw).strip()
    return (value, None) if value else (None, "empty_annotation")


def source_segments(ocr):
    """Build line IDs and word geometry using OCR only, retaining table spacing."""
    segments = []
    words = []
    for page_index, page in enumerate(ocr["pages"]):
        native_lines = []
        for block_index, block in enumerate(page["blocks"]):
            for line_index, line in enumerate(block["lines"]):
                if not line["words"]:
                    continue
                native_lines.append(line)
        native_lines.sort(key=lambda line: (sum(p[1] for p in line["geometry"]) / 2, line["geometry"][0][0]))
        rows = []
        for line in native_lines:
            top, bottom = line["geometry"][0][1], line["geometry"][1][1]
            center, height = (top + bottom) / 2, bottom - top
            if rows and abs(center - rows[-1]["center"]) <= 0.4 * min(height, rows[-1]["height"]):
                rows[-1]["words"].extend(line["words"])
            else:
                rows.append({"center": center, "height": height, "words": list(line["words"])})
        for line_index, row in enumerate(rows, 1):
            identifier = f"P{page_index + 1:02d}-L{line_index:03d}"
            row_words = sorted(row["words"], key=lambda word: word["geometry"][0][0])
            text = ""
            for word_index, word in enumerate(row_words, 1):
                position = max(0, round(word["geometry"][0][0] * 160))
                if text:
                    text += " " * max(1, position - len(text))
                else:
                    text = " " * position
                text += word["value"]
                words.append({"id": f"{identifier}-W{word_index:03d}", "line_id": identifier,
                              "text": word["value"], "bbox": [*word["geometry"][0], *word["geometry"][1]], "page": page_index})
            segments.append({"id": identifier, "page": page_index, "text": text.rstrip(), "word_ids": [w["id"] for w in words if w["line_id"] == identifier]})
    text = "\n".join(f"[{segment['id']}] {segment['text']}" for segment in segments)
    return {"text": text, "segments": segments, "words": words}


def overlap(word_box, annotation_box):
    left, top = max(word_box[0], annotation_box[0]), max(word_box[1], annotation_box[1])
    right, bottom = min(word_box[2], annotation_box[2]), min(word_box[3], annotation_box[3])
    area = max(0, word_box[2] - word_box[0]) * max(0, word_box[3] - word_box[1])
    return max(0, right - left) * max(0, bottom - top) / max(area, 1e-12)


def annotation_evidence(annotation, source):
    words = [word for word in source["words"] if word["page"] == annotation["page"] and overlap(word["bbox"], annotation["bbox"]) >= 0.45]
    line_ids = list(dict.fromkeys(word["line_id"] for word in words))
    return {"source_lines": line_ids, "source_word_ids": [word["id"] for word in words],
            "ocr_text": " ".join(word["text"] for word in words),
            "annotation_bbox": annotation["bbox"], "annotation_page": annotation["page"]}


def supported(field, raw, normalized, evidence):
    if not evidence["ocr_text"]:
        return False
    if field in MONEY_FIELDS or field == "quantity":
        if normalized is None:
            return False
        tokens = evidence["ocr_text"].split()
        for start in range(len(tokens)):
            if start and tokens[start - 1] in {"-", "(", "($", "-$"}:
                continue
            for width in range(1, min(5, len(tokens) - start + 1)):
                if start + width < len(tokens) and tokens[start + width] == ")":
                    continue
                candidate, issue = normalize_decimal(" ".join(tokens[start:start + width]), quantity=field == "quantity")
                if not issue and Decimal(candidate) == Decimal(normalized):
                    return True
        return False
    # Dates unresolved by policy may still have exact OCR support for review.
    expected = text_key(raw)
    actual = text_key(evidence["ocr_text"])
    if field == "invoice_number":
        characters = re.sub(r"\s+", "", raw)
        pattern = r"(?<!\w)" + r"\s*".join(re.escape(character) for character in characters) + r"(?!\w)"
        return re.search(pattern, evidence["ocr_text"], re.I) is not None
    return bool(expected) and expected in actual


def convert_field(field, annotations, source):
    candidates = []
    for annotation in annotations:
        normalized, issue = normalize_value(field, annotation["text"])
        evidence = annotation_evidence(annotation, source)
        issues = [issue] if issue else []
        if not supported(field, annotation["text"], normalized, evidence):
            issues.append("ocr_value_not_verified")
        if field == "subtotal":
            # DocILE net totals may be taxable bases, net after commission, or
            # one of several subtotals. Never auto-approve that semantic mapping.
            issues.append("net_total_mapping_requires_manual_review")
        candidates.append({"raw_text": annotation["text"], "normalized_value": normalized, "evidence": evidence, "issues": issues})
    if not candidates:
        return {"candidate_value": None, "status": "unannotated", "issues": ["annotation_coverage_unknown"], "candidates": []}
    values = {text_key(c["normalized_value"]) if field == "vendor" and c["normalized_value"] else c["normalized_value"] for c in candidates}
    usable = [candidate for candidate in candidates if not candidate["issues"]]
    issues = sorted({issue for candidate in candidates for issue in candidate["issues"]})
    if len(values) > 1:
        issues.append("conflicting_annotation_values")
        status = "ambiguous"
    elif usable:
        status = "candidate"
    else:
        status = "needs_review"
    value = usable[0]["normalized_value"] if usable and len(values) == 1 else None
    return {"candidate_value": value, "status": status, "issues": issues, "candidates": candidates}


def convert_currency(source):
    matches = []
    pattern = re.compile(r"\b(?:USD|EUR|GBP|CAD|AUD|JPY|CHF|CZK|NOK)\b|\bU\.?S\.?\s+DOLLARS?\b|\bU\.?S\.?\s+CURRENCY\b|US\$", re.I)
    for segment in source["segments"]:
        for match in pattern.finditer(segment["text"]):
            raw = match.group()
            if re.fullmatch(r"[A-Za-z]{3}", raw):
                context = segment["text"]
                explicit_context = re.search(r"\b(?:currency|dollars?|payable|amount|total|price)\b", context, re.I)
                attached_symbol = re.search(r"(?:[$€£¥]\s*" + re.escape(raw) + r"|" + re.escape(raw) + r"\s*[$€£¥])", context, re.I)
                if not explicit_context and not attached_symbol:
                    continue
            value = raw.upper() if re.fullmatch(r"[A-Za-z]{3}", raw) else "USD"
            matches.append({"raw_text": raw, "normalized_value": value,
                            "evidence": {"source_lines": [segment["id"]], "ocr_text": raw}, "issues": []})
    values = {match["normalized_value"] for match in matches}
    return {"candidate_value": next(iter(values)) if len(values) == 1 else None,
            "status": "candidate" if len(values) == 1 else "ambiguous" if values else "unannotated",
            "issues": [] if len(values) == 1 else ["multiple_explicit_currencies" if values else "no_explicit_currency_verified"],
            "candidates": matches}


def convert_document(root, document_id):
    root = Path(root)
    paths = {kind: root / folder / f"{document_id}.{extension}" for kind, folder, extension in
             (("annotation", "annotations", "json"), ("ocr", "ocr", "json"), ("pdf", "pdfs", "pdf"))}
    annotation, ocr = read_json(paths["annotation"]), read_json(paths["ocr"])
    source = source_segments(ocr)
    labels = {field: convert_field(field, [a for a in annotation["field_extractions"] if a["fieldtype"] == native], source) for field, native in HEADER_MAP.items()}
    labels["currency"] = convert_currency(source)
    grouped = defaultdict(list)
    for field in annotation["line_item_extractions"]:
        grouped[field["line_item_id"]].append(field)
    items = []
    for identifier, fields in sorted(grouped.items()):
        items.append({"source_line_item_id": identifier,
                      "cells": {field: convert_field(field, [a for a in fields if a["fieldtype"] == native], source) for field, native in ITEM_MAP.items()}})
    return {"schema_version": VERSION, "document_id": document_id,
            "source": {"paths": {key: path.as_posix() for key, path in paths.items()},
                       "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
                       "official_split": "train", "cluster_id": annotation["metadata"]["cluster_id"]},
            "input": source, "header_labels": labels, "line_items": items,
            "line_items_status": "annotated_rows_require_completeness_review" if items else "annotation_coverage_unknown",
            "review_status": "pending", "training_eligible": False}


def validate_approved_value(field, value):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Approved {field} must be a nonempty string")
    if field in MONEY_FIELDS:
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            raise ValueError(f"Invalid decimal string for {field}")
        Decimal(value)
    if field in DATE_FIELDS:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError(f"Invalid ISO date for {field}")
    if field == "currency" and not re.fullmatch(r"[A-Z]{3}", value):
        raise ValueError("Invalid currency code")


def training_example(record, review):
    """Only explicitly reviewed, source-grounded header fields enter pilot SFT."""
    if review.get("status") != "reviewed" or not review.get("pdf_inspected") or not review.get("ocr_inspected"):
        return None
    approved = {}
    provenance = {}
    for field in review.get("approve_candidates", []):
        label = record["header_labels"][field]
        if label["status"] != "candidate":
            raise ValueError(f"Cannot approve unresolved candidate {record['document_id']}/{field}")
        approved[field] = label["candidate_value"]
        provenance[field] = next(candidate["evidence"]["source_lines"] for candidate in label["candidates"] if not candidate["issues"])
    for field, override in review.get("corrections", {}).items():
        if field not in record["header_labels"]:
            raise ValueError(f"Unknown header field {field}")
        value = override["value"]
        validate_approved_value(field, value)
        lines = {segment["id"]: segment["text"] for segment in record["input"]["segments"]}
        if not override.get("reason") or not override.get("source_lines") or any(identifier not in lines for identifier in override["source_lines"]):
            raise ValueError("Corrections require a reason and valid OCR evidence")
        evidence = {"ocr_text": " ".join(lines[identifier] for identifier in override["source_lines"])}
        raw_support = override.get("raw_text", value)
        if field in DATE_FIELDS:
            if not text_key(raw_support) or text_key(raw_support) not in text_key(evidence["ocr_text"]) or not override.get("date_interpretation"):
                raise ValueError("Date corrections require source text and an explicit interpretation")
            resolved, issue = normalize_date(raw_support)
            if resolved and resolved != value:
                raise ValueError("Date correction contradicts an unambiguous source date")
        elif field == "currency":
            # Currency overrides must be one of the explicit OCR-derived candidates.
            if value not in {candidate["normalized_value"] for candidate in record["header_labels"][field]["candidates"]}:
                raise ValueError("Currency correction lacks explicit source support")
        elif not supported(field, raw_support, value, evidence):
            raise ValueError(f"Correction lacks OCR value support: {record['document_id']}/{field}")
        approved[field] = value
        provenance[field] = override["source_lines"]
    for field, value in approved.items():
        validate_approved_value(field, value)
    if not approved or review.get("exclude_from_training"):
        return None
    fields = sorted(approved)
    return {"messages": [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": "Requested fields: " + ", ".join(fields) + "\n\nInvoice OCR:\n" + record["input"]["text"]},
        {"role": "assistant", "content": json.dumps({field: approved[field] for field in fields}, ensure_ascii=False)},
    ]}, {"document_id": record["document_id"], "fields": fields, "evidence": provenance,
          "source_sha256": record["source"]["sha256"], "task": "partial_header_extraction"}
