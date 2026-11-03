# Local DocILE data

The local source collection is the complete `annotated-trainval.zip` archive.
It contains 5,680 real annotated business documents: 5,180 in the official
training split and 500 in the official validation split. These are document
counts, not counts of eligible invoices for the first experiment.

The synthetic, unlabeled, and official test archives are separate collections
and are outside this download. The original three-document inspection sample
is retained independently in `data/docile_sample/`.

## Downloaded inventory

The completed download has 17,043 CRC-verified files. The archive occupies
1,140,247,430 bytes; its extracted source files occupy 1,680,362,824 bytes.
The archive SHA-256 is
`59e0a4211fa190f44159435a65038a9a3505b94a95527bfe5d1a100d2c5026fc`.

The metadata identifies 3,850 `tax_invoice` documents overall. Of those,
3,008 are English and single-page: 2,723 in official training and 285 in
official validation. These are inspection candidates, not audited training
examples. The remaining business documents include orders and receipts.

The official splits share 144 layout cluster IDs and 27 combinations of
source and original filename. A layout group is not the same as a vendor,
and shared source filenames are not proof of identical document content.
These observations require grouped splitting and duplicate checks before our
own generalization evaluation.

Header labels are sparse: `amount_total_net` occurs in 792 documents and
`amount_total_tax` in 610 across the collection. Lack of an annotation does
not establish that a value is absent from the source. Audit coverage before
choosing training targets and the benchmark's required fields.

## Files

| Path | Purpose |
| --- | --- |
| `data/docile/pdfs/<document_id>.pdf` | Original document |
| `data/docile/ocr/<document_id>.json` | Precomputed word-level OCR and geometry |
| `data/docile/annotations/<document_id>.json` | Original field labels, line-item groups, and metadata |
| `data/docile/train.json` | Original training document IDs |
| `data/docile/val.json` | Original validation document IDs |
| `data/docile/trainval.json` | Combined original document IDs |
| `data/downloads/annotated-trainval.zip` | Retained source archive |
| `data/downloads/annotated-trainval.download.json` | Remote size, ETag, and modification date; no credentials |
| `data/docile_inventory.json` | Integrity results, split counts, metadata distributions, and annotation presence |
| `data/docile_catalog.jsonl` | One record per document, with local paths and selection metadata |
| `data/docile_file_manifest.json` | Original member names, sizes, ZIP CRCs, and archive SHA-256 |

Catalog paths are relative to `data/`. All source data and download metadata
remain local under the ignored `data/` directory.

## Read an example offline

Run from the project root with Python 3.12; no third-party packages are needed:

```python
import json
from pathlib import Path

root = Path("data/docile")
document_id = json.loads((root / "train.json").read_text(encoding="utf-8"))[0]
annotation = json.loads((root / "annotations" / f"{document_id}.json").read_text(encoding="utf-8"))
ocr = json.loads((root / "ocr" / f"{document_id}.json").read_text(encoding="utf-8"))

print(annotation["metadata"])
for field in annotation["field_extractions"]:
    print(field["fieldtype"], field["text"], field["page"], field["bbox"])
```

Annotations contain `field_extractions`, `line_item_extractions`,
`line_item_headers`, and `metadata`. OCR contains pages, blocks, lines, and words.
Word confidence describes OCR recognition; it is not a calibrated invoice
extraction correctness probability.

## Download and verification

`scripts/download_docile.py` reads `DOCILE_ACCESS_TOKEN` from its process
environment. Supply the token through the session environment; do not save it
in this documentation, source files, commands committed to Git, or dataset
manifests. Start or resume the download with:

```powershell
py -3.12 -u scripts/download_docile.py
```

The downloader pins the remote ETag, resumes partial downloads with verified
HTTP ranges, checks the final byte count, records the archive SHA-256, and
reads every ZIP member to EOF to verify its CRC during extraction. It checks
that each document has all three source files, parses all annotations and OCR,
compares OCR page counts with annotation metadata, and validates the official
split manifests. The SHA-256 is a local fingerprint, not an independently
published publisher checksum.

To repeat extraction and inventory from the retained archive while offline:

```powershell
py -3.12 -u scripts/download_docile.py --verify-local
```

## Preparation still required before fine-tuning

The download is source data, not an approved training dataset. The metadata
candidate flag in the catalog selects English, single-page `tax_invoice`
documents for inspection; it does not replace manual quality and scope checks.

1. Audit representative documents, label coverage, and OCR recoverability.
2. Finalize our field semantics and map native labels into the versioned schema.
3. Aggregate repeated annotations conservatively. Preserve distinct gross/net
   amount meanings and line-item associations.
4. Treat missing annotations as unknown until coverage is established. Do not
   turn every missing label into a null training target, or use document
   metadata as input evidence unavailable in the invoice text.
5. Group related source documents, duplicates, vendors, and layouts before
   making experiment splits. The official train/validation split is preserved
   for provenance; it is not automatically our unseen-layout final evaluation.
6. Freeze reviewed evaluation labels independently of model outcomes, then
   export training and validation JSONL. No custom test set or training targets
   are created by the downloader.

Source documentation: https://github.com/rossumai/docile and
https://docile.rossum.ai/. Follow the dataset access terms supplied with access.
