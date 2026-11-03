"""Read-only Fireworks account/model/quota check using the local .env key.

No inference, uploads, jobs, deployments, or billing mutations are performed.
Credentials remain in .env and are never included in the saved report.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    values = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*?)\s*$", line)
            if not match:
                continue
            value = match[2]
            if value[:1] in ('"', "'") and value[-1:] == value[:1]:
                value = value[1:-1]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
            values[match[1]] = value
    return values


class ReadOnlyFireworks:
    def __init__(self, key):
        self.key = key

    def get(self, path):
        if not path.startswith("accounts") or ":" in path:
            raise ValueError("Only account GET endpoints are allowed")
        request = urllib.request.Request(
            "https://api.fireworks.ai/v1/" + path,
            headers={"Authorization": "Bearer " + self.key, "Accept": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Fireworks GET failed with HTTP {error.code}") from None
        except urllib.error.URLError:
            raise RuntimeError("Fireworks GET failed: network error") from None

    def list_all(self, path, collection):
        values, token = [], None
        while True:
            query = {"pageSize": 200}
            if token:
                query["pageToken"] = token
            payload = self.get(path + "?" + urllib.parse.urlencode(query))
            values.extend(payload.get(collection, []))
            token = payload.get("nextPageToken")
            if not token:
                return values


def main():
    config = configuration()
    key = os.environ.get("FIREWORKS_API_KEY") or config.get("FIREWORKS_API_KEY") or config.get("api")
    if not key:
        raise RuntimeError("Fireworks API key missing from environment or .env")
    client = ReadOnlyFireworks(key)
    accounts = client.list_all("accounts", "accounts")
    requested = os.environ.get("FIREWORKS_ACCOUNT_ID") or config.get("FIREWORKS_ACCOUNT_ID")
    if requested:
        requested = requested.removeprefix("accounts/")
        accounts = [account for account in accounts if account["name"] == "accounts/" + requested]
    if len(accounts) != 1:
        raise RuntimeError("Set FIREWORKS_ACCOUNT_ID to select exactly one accessible account")
    account = accounts[0]
    name = account["name"]
    tasks = {
        "quotas": lambda: client.list_all(name + "/quotas", "quotas"),
        "jobs": lambda: client.list_all(name + "/supervisedFineTuningJobs", "supervisedFineTuningJobs"),
    }
    for model in ("glm-5p3-flash", "deepseek-v4p1-flash-1017", "nemotron-lightning-3p5-30b-a3b"):
        tasks[model] = lambda model=model: client.get("accounts/fireworks/models/" + model)
    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {label: pool.submit(task) for label, task in tasks.items()}
        for label, future in futures.items():
            try:
                results[label] = future.result()
            except RuntimeError as error:
                errors[label] = str(error)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "authenticated": True,
              "account": {field: account.get(field) for field in ("name", "displayName", "state", "suspendState", "productivityOnly")},
              "models": {}, "quotas": [], "existing_sft_jobs": [], "read_errors": errors,
              "operations_performed": "GET requests only; no training or inference started"}
    model_fields = ("name", "displayName", "state", "tunable", "supportsLora", "supervisedLoraTunable",
                    "supervisedFullParameterTunable", "supportsServerless", "trainingContextLength", "contextLength", "huggingFaceUrl")
    for label, result in results.items():
        if label == "quotas":
            report["quotas"] = [{field: item.get(field) for field in ("name", "value", "maxValue", "usage")} for item in result]
        elif label == "jobs":
            for item in result:
                report["existing_sft_jobs"].append({field: item.get(field) for field in ("name", "baseModel", "state", "epochs", "createTime", "completedTime", "jobProgress")})
        else:
            report["models"][label] = {field: result.get(field) for field in model_fields}
    # Defense against accidental credential persistence, including API error text.
    serialized = json.dumps(report, indent=2)
    if key in serialized:
        raise RuntimeError("Refusing to write a report containing credentials")
    output = ROOT / "data/fireworks/account_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    report["quotas"] = [quota for quota in report["quotas"] if re.search(r"train|spend|concurrent|credit", quota["name"], re.I)]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
