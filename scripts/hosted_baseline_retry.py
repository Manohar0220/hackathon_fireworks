"""Submit/monitor inference-only Fireworks batch retries for failed baseline calls."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid

from check_fireworks import ROOT, configuration, ReadOnlyFireworks

ACCOUNT = "accounts/korikanamanohar2-me6"
RUN = ROOT / "data/evals/baseline_20261102"
OUT = RUN / "hosted_retry"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Read status of the already submitted job")
    args = parser.parse_args()
    cfg = configuration()
    key = cfg.get("FIREWORKS_API_KEY") or cfg.get("api")
    if not key:
        raise RuntimeError("Missing Fireworks API key")
    reader = ReadOnlyFireworks(key)

    def call(method, resource, body=None, content_type="application/json"):
        base = resource.split("?", 1)[0]
        pattern = re.escape(ACCOUNT) + r"/(?:datasets|batchInferenceJobs)(?:/[a-z0-9-]+(?::(?:upload|getDownloadEndpoint))?)?"
        if not re.fullmatch(pattern, base):
            raise RuntimeError("Refusing an operation outside datasets and batch inference")
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        tries = 4 if method == "GET" else 1
        for attempt in range(tries):
            request = urllib.request.Request("https://api.fireworks.ai/v1/" + resource, data=data,
                headers={"Authorization": "Bearer " + key, "Content-Type": content_type}, method=method)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as error:
                if method == "GET" and error.code in (429, 500, 502, 503, 504) and attempt + 1 < tries:
                    time.sleep(2 ** attempt)
                    continue
                try:
                    message = json.loads(error.read()).get("message", "")
                except (ValueError, AttributeError):
                    message = ""
                message = re.sub(r"https?://\S+", "[URL]", str(message).replace(key, "[REDACTED]"))[:400]
                raise RuntimeError(f"HTTP {error.code}: {message}") from None
            except urllib.error.URLError:
                raise RuntimeError("Fireworks API network error") from None

    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "job.json"
    if args.refresh:
        report = json.loads(report_path.read_text())
    else:
        snapshot = (RUN / "eval_snapshot.jsonl").read_bytes()
        snapshot_sha = hashlib.sha256(snapshot).hexdigest()
        if (ROOT / "val_eval.jsonl").read_bytes() != snapshot:
            raise RuntimeError("Current evaluation file differs from the frozen baseline snapshot")
        rows = [json.loads(line) for line in snapshot.decode("utf-8-sig").splitlines() if line.strip()]
        protocol = json.loads((RUN / "protocol.json").read_text())
        records = {r["example_index"]:r for r in map(json.loads, (RUN / "nemotron_predictions.jsonl").read_text(encoding="utf-8").splitlines())}
        indexes = [i for i in range(len(rows)) if i not in records or records[i]["status"] != "ok"]
        if not indexes:
            print("No failed Nemotron requests remain")
            return
        requests, mapping = [], {}
        for index in indexes:
            row = rows[index]
            messages = row["messages"][:-1]
            if [m["role"] for m in messages] != ["system", "user"]:
                raise RuntimeError("An inference request must contain only system and user messages")
            custom_id = row.get("id") or "example-" + str(index)
            mapping[custom_id] = index
            requests.append({"custom_id":custom_id,"body":{"messages":messages,
                "temperature":protocol["temperature"],"max_tokens":protocol["max_tokens"],
                "reasoning_effort":protocol["model_reasoning"]["nemotron"],
                "context_length_exceeded_behavior":"error"}})
        payload = ("\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        (OUT / "nemotron_requests.jsonl").write_bytes(payload)
        (OUT / "request_map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        dataset_id = "invoiceroute-nemotron-retry-" + digest[:10]
        dataset_name = ACCOUNT + "/datasets/" + dataset_id
        job_id = "invoiceroute-nemotron-eval-" + digest[:10]
        report = {"operation":"Hosted batch inference only; no training", "source_eval_sha256":snapshot_sha,
            "reference_dataset":ACCOUNT + "/datasets/invoiceroute-val-eval-" + snapshot_sha[:10],
            "request_count":len(requests),"request_sha256":digest,"input_dataset":dataset_name,
            "output_dataset":ACCOUNT + "/datasets/invoiceroute-nemotron-out-" + digest[:10],
            "job_name":ACCOUNT + "/batchInferenceJobs/" + job_id,
            "model":"accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
            "assistant_references_sent_as_model_inputs":False,"training_started":False,
            "submitted_at_utc":datetime.now(timezone.utc).isoformat()}
        report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        existing = {d["name"]:d for d in reader.list_all(ACCOUNT + "/datasets", "datasets")}
        dataset = existing.get(dataset_name)
        if not dataset:
            dataset = call("POST", ACCOUNT + "/datasets", {"datasetId":dataset_id,
                "dataset":{"displayName":"InvoiceRoute held-out Nemotron retry prompts",
                    "exampleCount":str(len(requests)),"userUploaded":{}}})
        if dataset.get("state") != "READY":
            boundary = "----InvoiceRoute" + uuid.uuid4().hex
            preamble = ("--" + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="nemotron_requests.jsonl"\r\nContent-Type: application/jsonl\r\n\r\n').encode()
            call("POST",dataset_name+":upload",preamble+payload+("\r\n--"+boundary+"--\r\n").encode(),
                 "multipart/form-data; boundary="+boundary)
            deadline = time.monotonic() + 120
            while True:
                dataset = call("GET", dataset_name)
                if dataset.get("state") == "READY":
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Batch input dataset validation is still pending")
                if dataset.get("status",{}).get("code") not in (None,"OK",0,"0"):
                    raise RuntimeError("Batch input dataset validation failed")
                time.sleep(3)
        report["input_dataset_state"] = dataset["state"]
        report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        existing_jobs = {j["name"]:j for j in reader.list_all(ACCOUNT + "/batchInferenceJobs", "batchInferenceJobs")}
        if report["job_name"] not in existing_jobs:
            job = call("POST",ACCOUNT+"/batchInferenceJobs?batchInferenceJobId="+job_id,
                {"displayName":"InvoiceRoute Nemotron baseline retry (evaluation only)",
                 "model":report["model"],"inputDatasetId":dataset_name,
                 "outputDatasetId":report["output_dataset"],
                 "inferenceParameters":{"maxTokens":protocol["max_tokens"],"temperature":0,
                    "extraBody":json.dumps({"reasoning_effort":"none","context_length_exceeded_behavior":"error"})},
                 "maxJobDuration":"43200s"})
            report["job_name"] = job["name"]
    job = call("GET", report["job_name"])
    report["last_checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["state"] = job.get("state")
    report["status"] = job.get("status")
    report["progress"] = job.get("jobProgress")
    if job.get("outputDatasetId"):
        report["output_dataset"] = job["outputDatasetId"]
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
