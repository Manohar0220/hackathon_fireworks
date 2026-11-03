# Prepared DocILE pilot

Run `py -3.12 scripts/prepare_docile.py` from the project root to rebuild the
derived pilot in `data/prepared/docile_pilot_v1/`. It reads the fixed selection
and source-hash-bound review decisions. Original PDFs, OCR and annotations are
never modified. The completed preservation check matched all 17,043 original
files against the downloaded archive's member sizes and CRCs.

The pilot covers 50 training documents. A Codex visual/OCR audit excluded five
documents from training and approved 201 header targets across 45 examples.
This is an agent-reviewed development sample, not independently human-reviewed
benchmark gold. Selection was stratified and is not representative prevalence.

- `review.html`: offline source images, OCR, candidates and review decisions.
- `pilot_candidates.jsonl`: all 50 records, with unknown/conflicting labels retained.
- `review_decisions.json`: exclusions, corrections, reasons and OCR evidence.
- `pilot_header_sft.jsonl`: 45 conversational JSONL examples in Fireworks SFT format.
- `pilot_header_sft_provenance.jsonl`: IDs, approved fields, evidence and source hashes.
- `audit_summary.json`: counts and unresolved preparation issues.
- `source_integrity.json`: original-source preservation result.

The SFT file is a **partial-header extraction task**: every input explicitly
requests only the fields approved for that record. Unannotated values are not
turned into null or zero training targets. It does not yet cover the full
application schema, absent-field behavior, or line-item extraction. The 193
grouped line-item rows remain review candidates with distinct net/gross prices.

Dates with unresolved day/month or century ambiguity are withheld. Source
currency metadata is not a target; bare $ does not establish USD. Native net
totals require explicit review before becoming subtotal. Printed arithmetic
inconsistencies are preserved rather than repaired.

The two-hour hackathon plan uses Fireworks managed training. Further manual
audit expansion and local training setup were stopped in favor of lightweight
conversion, a supported tunable model, and a small independent evaluation set.
No external training job, upload, deployment, or model API call has been made.
