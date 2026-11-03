"""Download, verify, extract, and inventory the annotated DocILE train/val set.

Usage: set DOCILE_ACCESS_TOKEN in the process environment, then run this file.
Only the annotated train/validation archive is downloaded. No credentials or
credential-bearing URLs are written to disk or included in output.
"""

import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/downloads/annotated-trainval.zip"
OUTPUT = ROOT / "data/docile"


def download(token):
    url = (
        "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/"
        + token
        + "/annotated-trainval.zip"
    )
    with urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD"), timeout=60
    ) as response:
        total = int(response.headers["Content-Length"])
        etag = response.headers["ETag"]
        modified = response.headers.get("Last-Modified")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE.with_suffix(".zip.partial")
    identity_path = ARCHIVE.with_suffix(".download.json")
    identity = {"bytes": total, "etag": etag, "last_modified": modified}
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Remote archive changed; preserve and inspect the existing download")
    elif partial.exists() or ARCHIVE.exists():
        raise RuntimeError("Existing archive has no download identity; inspect it before resuming")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size != total:
            raise RuntimeError("Existing archive size differs from remote size")
        print("Archive already downloaded; proceeding to verification", flush=True)
        return identity
    print(f"Archive size: {total:,} bytes", flush=True)
    for attempt in range(1, 6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == total:
            partial.replace(ARCHIVE)
            return identity
        if offset > total:
            raise RuntimeError("Partial download exceeds expected archive size")
        if shutil.disk_usage(ARCHIVE.parent).free < total - offset + 6 * 1024**3:
            raise RuntimeError("Insufficient free disk space for download and extraction")
        request = urllib.request.Request(url, headers={"If-Match": etag})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and (
                    response.status != 206
                    or response.headers.get("Content-Range") != f"bytes {offset}-{total - 1}/{total}"
                ):
                    raise RuntimeError("Server did not honor the requested resume range")
                if response.headers.get("ETag") != etag:
                    raise RuntimeError("Archive identity changed during download")
                start = time.monotonic()
                last_report = start
                received = offset
                with partial.open("ab") as target:
                    while chunk := response.read(4 * 1024**2):
                        target.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            speed = (received - offset) / max(now - start, 0.001) / 1024**2
                            print(f"Download: {received / total:.1%} ({received / 1024**2:.0f} MiB), {speed:.1f} MiB/s", flush=True)
                            last_report = now
            if partial.stat().st_size != total:
                raise OSError("Incomplete download")
            partial.replace(ARCHIVE)
            print("Download complete", flush=True)
            return identity
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            status = f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError) else type(error).__name__
            print(f"Download interrupted ({status}), resumable attempt {attempt}/5", flush=True)
            if attempt == 5:
                raise RuntimeError("Download interrupted; rerun to resume the partial archive") from None
            time.sleep(2)


def extract_and_inventory(identity):
    with ARCHIVE.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    print(f"Archive SHA-256: {digest}", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    extracted_bytes = 0
    members = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = archive.infolist()
        for index, info in enumerate(infos, 1):
            target = (OUTPUT / info.filename).resolve()
            if not target.is_relative_to(OUTPUT.resolve()):
                raise RuntimeError("Unsafe archive member path")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            # Reading each member to EOF verifies its ZIP CRC before committing it.
            with archive.open(info) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024**2)
            if temporary.stat().st_size != info.file_size:
                raise RuntimeError("Extracted member size mismatch")
            temporary.replace(target)
            extracted_bytes += info.file_size
            members.append({"path": info.filename, "bytes": info.file_size, "crc32": f"{info.CRC:08x}"})
            if index % 3000 == 0:
                print(f"Extracted and CRC-verified {index:,}/{len(infos):,} files", flush=True)
    splits = {name: json.loads((OUTPUT / f"{name}.json").read_text(encoding="utf-8")) for name in ("train", "val", "trainval")}
    train, val, combined = (set(splits[name]) for name in ("train", "val", "trainval"))
    if train & val or train | val != combined:
        raise RuntimeError("Inconsistent official split manifests")
    if any(len(splits[name]) != len(set(splits[name])) for name in splits):
        raise RuntimeError("Duplicate document IDs within an official split")
    for folder, extension in (("pdfs", ".pdf"), ("ocr", ".json"), ("annotations", ".json")):
        present = {path.stem for path in (OUTPUT / folder).glob(f"*{extension}")}
        if present != combined:
            raise RuntimeError(f"Document IDs in {folder} do not match the split manifests")
    counts = {key: collections.Counter() for key in ("document_type", "language", "currency", "page_count", "source")}
    header_presence = collections.Counter()
    eligible = collections.Counter()
    catalog = []
    clusters = {"train": set(), "val": set()}
    source_groups = {"train": set(), "val": set()}
    total_line_items = 0
    for identifier in splits["trainval"]:
        annotation = json.loads((OUTPUT / "annotations" / f"{identifier}.json").read_text(encoding="utf-8"))
        ocr = json.loads((OUTPUT / "ocr" / f"{identifier}.json").read_text(encoding="utf-8"))
        metadata = annotation["metadata"]
        if len(ocr["pages"]) != metadata["page_count"]:
            raise RuntimeError(f"OCR page count mismatch for document {identifier}")
        split = "train" if identifier in train else "val"
        fieldtypes = sorted({field["fieldtype"] for field in annotation["field_extractions"]})
        item_rows = len({field["line_item_id"] for field in annotation["line_item_extractions"]})
        total_line_items += item_rows
        for key, counter in counts.items():
            counter[str(metadata.get(key))] += 1
        header_presence.update(fieldtypes)
        clusters[split].add(metadata.get("cluster_id"))
        source_groups[split].add((metadata.get("source"), metadata.get("original_filename")))
        candidate = metadata.get("language") == "eng" and metadata.get("page_count") == 1 and metadata.get("document_type") == "tax_invoice"
        if candidate:
            eligible[split] += 1
        catalog.append({
            "document_id": identifier, "official_split": split,
            "document_type": metadata.get("document_type"), "language": metadata.get("language"),
            "page_count": metadata.get("page_count"), "currency_metadata": metadata.get("currency"),
            "cluster_id": metadata.get("cluster_id"), "source": metadata.get("source"),
            "original_filename": metadata.get("original_filename"),
            "header_fieldtypes": fieldtypes, "line_item_rows": item_rows,
            "candidate_english_single_page_tax_invoice": candidate,
            "pdf_path": f"docile/pdfs/{identifier}.pdf", "ocr_path": f"docile/ocr/{identifier}.json",
            "annotation_path": f"docile/annotations/{identifier}.json",
        })
    inventory = {
        "retrieved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "https://github.com/rossumai/docile", "archive": "annotated-trainval.zip",
        "remote_identity": identity, "sha256": digest, "zip_members_crc_verified": len(members),
        "extracted_bytes": extracted_bytes,
        "official_split_counts": {name: len(ids) for name, ids in splits.items()},
        "document_counts": {key: dict(counter.most_common()) for key, counter in counts.items()},
        "header_annotation_presence_documents": dict(header_presence.most_common()),
        "annotated_line_item_rows": total_line_items,
        "candidate_english_single_page_tax_invoices": dict(eligible),
        "official_train_val_shared_layout_clusters": len(clusters["train"] & clusters["val"]),
        "official_train_val_shared_source_documents": len(source_groups["train"] & source_groups["val"]),
        "status": "Source data downloaded and integrity-checked; semantic audit and training conversion pending.",
        "notes": [
            "Official splits are preserved; no custom frozen test split has been created.",
            "Candidate flags use metadata only and are not a manual eligibility or quality audit.",
            "Annotation presence counts do not establish complete annotation coverage.",
            "Currency metadata is not automatically a supported text-extraction target.",
            "No synthetic, unlabeled, or official test archives are included in this download.",
        ],
    }
    data_dir = OUTPUT.parent
    (data_dir / "docile_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    with (data_dir / "docile_catalog.jsonl").open("w", encoding="utf-8") as target:
        for record in catalog:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    (data_dir / "docile_file_manifest.json").write_text(json.dumps({"archive_sha256": digest, "members": members}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(inventory, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-local", action="store_true", help="Re-extract and inventory the local archive without network access or a token")
    args = parser.parse_args()
    token = os.environ.get("DOCILE_ACCESS_TOKEN")
    if not args.verify_local and not token:
        raise SystemExit("Set DOCILE_ACCESS_TOKEN for this process")
    try:
        if args.verify_local:
            identity = json.loads(ARCHIVE.with_suffix(".download.json").read_text(encoding="utf-8"))
            if ARCHIVE.stat().st_size != identity["bytes"]:
                raise RuntimeError("Local archive size differs from the download identity")
            extract_and_inventory(identity)
        else:
            extract_and_inventory(download(token))
    except urllib.error.HTTPError as error:
        raise SystemExit(f"DocILE request failed: HTTP {error.code}") from None
    except urllib.error.URLError:
        raise SystemExit("DocILE request failed: network error") from None
