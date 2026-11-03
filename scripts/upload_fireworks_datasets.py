"""Upload selected unchanged JSONL files to Fireworks; never starts training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid

from check_fireworks import ROOT, ReadOnlyFireworks, configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "val_eval"), default=["train", "val"])
    args = parser.parse_args()
    config = configuration()
    key = os.environ.get("FIREWORKS_API_KEY") or config.get("FIREWORKS_API_KEY") or config.get("api")
    if not key:
        raise RuntimeError("Missing Fireworks API key")
    client = ReadOnlyFireworks(key)
    accounts = client.list_all("accounts", "accounts")
    requested = os.environ.get("FIREWORKS_ACCOUNT_ID") or config.get("FIREWORKS_ACCOUNT_ID")
    if requested:
        accounts = [a for a in accounts if a["name"] == "accounts/" + requested.removeprefix("accounts/")]
    if len(accounts) != 1:
        raise RuntimeError("Select exactly one Fireworks account")
    account = accounts[0]["name"]
    existing = {d["name"]: d for d in client.list_all(account + "/datasets", "datasets")}
    prepared = []
    for split in args.splits:
        path = ROOT / (split + ".jsonl")
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
        for row in rows:
            messages = row["messages"]
            if [m["role"] for m in messages] != ["system", "user", "assistant"]:
                raise RuntimeError("Unexpected message roles in " + path.name)
            if not all(isinstance(m["content"], str) and m["content"].strip() for m in messages):
                raise RuntimeError("Empty or non-string content in " + path.name)
            if not isinstance(json.loads(messages[-1]["content"]), dict):
                raise RuntimeError("Non-object annotation target in " + path.name)
        sha = hashlib.sha256(data).hexdigest()
        prepared.append((split, path, data, len(rows), sha))

    def request(method, resource, payload=None, content_type="application/json"):
        if not re.fullmatch(re.escape(account) + r"/datasets(?:/[a-z0-9-]+(?::(?:upload|validateUpload|getDownloadEndpoint))?)?", resource):
            raise RuntimeError("Refusing a non-dataset operation")
        body = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        req = urllib.request.Request("https://api.fireworks.ai/v1/" + resource, data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": content_type}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = response.read()
                return json.loads(result) if result else {}
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read()).get("message", "")
            except (ValueError, AttributeError):
                message = ""
            message = re.sub(r"https?://\S+", "[URL]", str(message).replace(key, "[REDACTED]"))[:400]
            raise RuntimeError(f"Dataset operation failed with HTTP {error.code}: {message}") from None
        except urllib.error.URLError:
            raise RuntimeError("Dataset operation failed: network error") from None

    report_path = ROOT / "data/fireworks/dataset_uploads.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"account": account, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "training_started": False, "datasets": []}
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("account") == account:
            selected_files = {split + ".jsonl" for split in args.splits}
            report["datasets"] = [d for d in previous.get("datasets", []) if d["file"] not in selected_files]

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for split, path, data, count, sha in prepared:
        dataset_id = "invoiceroute-" + split.replace("_", "-") + "-" + sha[:10]
        resource = account + "/datasets/" + dataset_id
        entry = {"file": path.name, "name": resource, "example_count": count,
                 "bytes": len(data), "sha256": sha, "state": "PENDING"}
        report["datasets"].append(entry)
        save()
        current = existing.get(resource)
        if current is None:
            current = request("POST", account + "/datasets", {
                "datasetId": dataset_id, "dataset": {"displayName": "InvoiceRoute DocILE " + split,
                "exampleCount": str(count), "userUploaded": {}, "format": "CHAT"}})
            print("Created " + dataset_id, flush=True)
        if current.get("state") != "READY":
            boundary = "----InvoiceRoute" + uuid.uuid4().hex
            prefix = ("--" + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="'
                      + path.name + '"\r\nContent-Type: application/jsonl\r\n\r\n').encode()
            result = request("POST", resource + ":upload", prefix + data + ("\r\n--" + boundary + "--\r\n").encode(),
                             "multipart/form-data; boundary=" + boundary)
            entry["upload_acknowledged_bytes"] = result.get("bytes")
            save()
            print("Uploaded " + path.name, flush=True)
            current = request("GET", resource)
            # The streamlined :upload endpoint starts validation automatically.
            deadline = time.monotonic() + 300
            while current.get("state") != "READY":
                if current.get("status", {}).get("code") not in (None, "OK", 0, "0"):
                    raise RuntimeError("Fireworks dataset validation reported an error for " + dataset_id)
                if time.monotonic() >= deadline:
                    raise RuntimeError("Dataset validation still pending for " + dataset_id)
                time.sleep(3)
                current = request("GET", resource)
        entry["state"] = current["state"]
        entry["provider_example_count"] = int(current.get("exampleCount", 0))
        if entry["provider_example_count"] != count:
            raise RuntimeError("Provider example count differs from local file")
        entry["local_file_unchanged"] = hashlib.sha256(path.read_bytes()).hexdigest() == sha
        if not entry["local_file_unchanged"]:
            raise RuntimeError("Local dataset changed during upload")
        save()
        # Signed storage URLs are used only in memory, without the API key header.
        try:
            urls = request("GET", resource + ":getDownloadEndpoint").get("filenameToSignedUrls", {})
        except RuntimeError as error:
            if "dataset downloads are temporarily disabled" not in str(error):
                raise
            entry["remote_hash_verification"] = "Unavailable: Fireworks has disabled dataset downloads"
            save()
            print(json.dumps(entry), flush=True)
            continue
        if len(urls) != 1:
            raise RuntimeError("Expected one downloadable source file")
        signed_url = next(iter(urls.values()))
        if not isinstance(signed_url, str) or not signed_url.startswith("https://"):
            raise RuntimeError("Unexpected dataset download endpoint")
        try:
            with urllib.request.urlopen(signed_url, timeout=60) as response:
                remote_sha = hashlib.sha256(response.read()).hexdigest()
        except Exception:
            raise RuntimeError("Could not verify uploaded file by downloading it") from None
        entry["remote_sha256"] = remote_sha
        entry["remote_bytes_match"] = remote_sha == sha
        entry["local_file_unchanged"] = hashlib.sha256(path.read_bytes()).hexdigest() == sha
        save()
        if not entry["remote_bytes_match"] or not entry["local_file_unchanged"]:
            raise RuntimeError("Dataset integrity check failed")
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from None
