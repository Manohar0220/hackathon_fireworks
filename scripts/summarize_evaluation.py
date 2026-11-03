"""Score saved inference outputs without making API calls or changing labels."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

from evaluate_fireworks import ROOT, counts, metrics, normalized, score, sha


def content_pairs(obj, normalize=False):
    """Empty arrays contribute no values; malformed value types are false positives."""
    out = Counter()
    if not isinstance(obj, dict):
        return out
    for field, values in obj.items():
        if not isinstance(values, list):
            out[(field, ("INVALID_VALUE_TYPE", json.dumps(values, sort_keys=True)))] += 1
            continue
        for value in values:
            if isinstance(value, str):
                out[(field, normalized(value) if normalize else value)] += 1
            else:
                out[(field, ("INVALID_VALUE_TYPE", json.dumps(value, sort_keys=True)))] += 1
    return out


def pair_counts(pred, gold):
    tp = sum((pred & gold).values())
    return Counter(tp=tp, fp=sum(pred.values()) - tp, fn=sum(gold.values()) - tp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="data/evals/baseline_20261102")
    args = parser.parse_args()
    folder = ROOT / args.run_dir
    protocol = json.loads((folder / "protocol.json").read_text())
    snapshot = (folder / "eval_snapshot.jsonl").read_bytes()
    if sha(snapshot) != protocol["eval_sha256"]:
        raise RuntimeError("Evaluation snapshot does not match the frozen protocol")
    examples = [json.loads(line) for line in snapshot.decode("utf-8-sig").splitlines() if line.strip()]
    summary = {"evaluation_sha256": protocol["eval_sha256"], "evaluation_examples": len(examples),
        "scoring_definitions": {
            "field_value_micro": "Exact field-name/string multiset matching. List order ignored. Empty arrays contribute no values; unknown fields and malformed value types are false positives. Malformed JSON/non-object outputs are empty predictions.",
            "normalized_field_value_micro": "Same matching after Unicode NFKC, casefolding and whitespace collapse. Digits and punctuation are retained.",
            "contract_field_value_micro": "Original strict metric: an invalid schema makes the whole document an empty prediction.",
            "schema": "Only the 34 named keys; nonempty arrays of nonempty strings; missing fields omitted.",
            "coverage": "Only attempted examples are scored; completion requires every example to have a successful response. Transport failures remain empty predictions until retried.",
            "cost": "Token-rate estimates for recorded attempts, not a billing statement. Qwen assumes no cache discount; probes and timed-out requests may add charges.",
            "latency": "Successful attempt wall time, including network/service time. Qwen uses Training SDK sampling; others use chat inference. Not a controlled hardware comparison."},
        "models": {}}
    csv_rows = []
    for path in sorted(folder.glob("*_predictions.jsonl")):
        label = path.name.removesuffix("_predictions.jsonl")
        attempts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        latest = {r["example_index"]: r for r in attempts}
        totals, norm, contract = Counter(tp=0, fp=0, fn=0), Counter(tp=0, fp=0, fn=0), Counter(tp=0, fp=0, fn=0)
        field_counts = {}
        json_ok = schema_ok = exact_docs = errors = truncated = 0
        latencies = []
        for index, result in latest.items():
            row = examples[index]
            if result["input_sha256"] != sha(row["messages"][:-1]):
                raise RuntimeError("Saved prediction input does not match the evaluation snapshot")
            reference = json.loads(row["messages"][-1]["content"])
            scored = score(result.get("content", ""), reference, set(protocol["schema_fields"]))
            obj = scored["parsed"]
            pred = content_pairs(obj)
            gold = content_pairs(reference)
            totals.update(pair_counts(pred, gold))
            norm.update(pair_counts(content_pairs(obj, True), content_pairs(reference, True)))
            contract.update(scored["strict"])
            json_ok += scored["json_valid"]
            schema_ok += scored["schema_valid"]
            exact_docs += pred == gold
            for field in set(reference) | (set(obj) if isinstance(obj, dict) else set()):
                p = Counter({k:v for k,v in pred.items() if k[0] == field})
                g = Counter({k:v for k,v in gold.items() if k[0] == field})
                field_counts.setdefault(field, Counter(tp=0, fp=0, fn=0)).update(pair_counts(p, g))
            if result["status"] == "ok":
                latencies.append(result["latency_seconds"])
            else:
                errors += 1
            truncated += result.get("finish_reason") in ("length", "max_tokens")
        n = len(latest)
        latencies.sort()
        record = {"model_resource": next(iter(latest.values()))["model_resource"] if latest else None,
            "route": next(iter(latest.values()))["route"] if latest else None,
            "attempted_examples": n, "successful_examples": n-errors, "request_errors": errors,
            "complete": n == len(examples) and not errors, "field_value_micro": metrics(totals),
            "normalized_field_value_micro": metrics(norm), "contract_field_value_micro": metrics(contract),
            "valid_json_rate": json_ok/n if n else None, "valid_schema_rate": schema_ok/n if n else None,
            "document_value_exact_match_rate": exact_docs/n if n else None, "truncated_outputs": truncated,
            "latency_p50_seconds": statistics.median(latencies) if latencies else None,
            "latency_p95_seconds": latencies[min(len(latencies)-1, int(.95*(len(latencies)-1)))] if latencies else None,
            "estimated_recorded_attempt_cost_usd": round(sum(r.get("estimated_cost_usd",0) for r in attempts),6),
            "per_field": {field:metrics(count) for field,count in sorted(field_counts.items())}}
        summary["models"][label] = record
        csv_rows.append({"model":label, "successful_examples":n-errors, "expected_examples":len(examples),
            "field_value_f1":record["field_value_micro"]["f1"],
            "normalized_field_value_f1":record["normalized_field_value_micro"]["f1"],
            "contract_field_value_f1":record["contract_field_value_micro"]["f1"],
            "valid_json_rate":record["valid_json_rate"], "valid_schema_rate":record["valid_schema_rate"],
            "latency_p50_seconds":record["latency_p50_seconds"],
            "estimated_recorded_attempt_cost_usd":record["estimated_recorded_attempt_cost_usd"], "complete":record["complete"]})
    (folder/"comparison.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (folder/"comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]))
        writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps({"evaluation_examples":len(examples),"models":csv_rows},indent=2))


if __name__ == "__main__":
    main()
