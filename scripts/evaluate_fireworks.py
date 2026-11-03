"""Resumable extraction evaluation. Inference only: no training/deployment APIs."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request

from check_fireworks import ROOT, configuration

MODELS = {
    "glm": {"id": "glm-5p3-flash", "effort": "low", "rates": [0.15, 0.03, 0.50]},
    "qwen": {"id": "qwen3p8-27b", "effort": "none", "route": "sampler", "rates": [1.86, 0.372, 5.595]},
    "nemotron": {"id": "nemotron-lightning-3p5-30b-a3b", "effort": "none", "rates": [0.05, 0.01, 0.20]},
    "deepseek": {"id": "deepseek-v4p1-flash", "effort": "none", "rates": [0.30, 0.006, 1.20]},
}


def sha(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def pairs(obj, normalize=False):
    return Counter((key, normalized(value) if normalize else value)
                   for key, values in obj.items() for value in values)


def counts(prediction, reference, normalize=False):
    pred, gold = pairs(prediction, normalize), pairs(reference, normalize)
    tp = sum((pred & gold).values())
    return {"tp": tp, "fp": sum(pred.values()) - tp, "fn": sum(gold.values()) - tp}


def metrics(count):
    tp, fp, fn = (count.get(key, 0) for key in ("tp", "fp", "fn"))
    return {**dict(count), "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}


def score(text, reference, allowed):
    try:
        obj = json.loads(text)
        json_valid = True
    except (ValueError, TypeError):
        obj, json_valid = None, False
    schema_valid = (isinstance(obj, dict) and all(
        key in allowed and isinstance(values, list) and len(values) > 0
        and all(isinstance(v, str) and v.strip() for v in values)
        for key, values in obj.items()))
    pred = obj if schema_valid else {}
    return {"json_valid": json_valid, "schema_valid": schema_valid,
            "strict": counts(pred, reference), "normalized": counts(pred, reference, True),
            "document_exact": schema_valid and pairs(pred) == pairs(reference),
            "parsed": obj}


def summarize(records, expected):
    strict, norm = Counter(tp=0, fp=0, fn=0), Counter(tp=0, fp=0, fn=0)
    successful = [r for r in records if r.get("status") == "ok"]
    # API failures count as missing predictions, never silently disappear.
    for record in records:
        strict.update(record["scores"]["strict"])
        norm.update(record["scores"]["normalized"])
    latencies = sorted(r["latency_seconds"] for r in successful)
    n = len(records)
    return {"expected_examples": expected, "attempted_examples": n,
            "successful_requests": len(successful), "request_errors": n - len(successful),
            "complete": n == expected and len(successful) == expected,
            "strict_field_value_micro": metrics(strict), "normalized_field_value_micro": metrics(norm),
            "valid_json_rate": sum(r["scores"]["json_valid"] for r in records) / n if n else None,
            "valid_schema_rate": sum(r["scores"]["schema_valid"] for r in records) / n if n else None,
            "document_exact_match_rate": sum(r["scores"]["document_exact"] for r in records) / n if n else None,
            "latency_p50_seconds": statistics.median(latencies) if latencies else None,
            "latency_p95_seconds": latencies[min(len(latencies)-1, int(.95*(len(latencies)-1)))] if latencies else None,
            "estimated_inference_cost_usd": round(sum(r.get("estimated_cost_usd", 0) for r in records), 6),
            "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in records),
            "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in records)}


def chat_call(key, model, messages, max_tokens, timeout):
    payload = {"model": model["resource"], "messages": messages, "temperature": 0,
               "max_tokens": max_tokens, "reasoning_effort": model["effort"],
               "stream": False, "context_length_exceeded_behavior": "error"}
    req = urllib.request.Request("https://api.fireworks.ai/inference/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key,
        "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        # Do not persist credentials, request headers, or arbitrary server bodies.
        try:
            msg = json.loads(error.read()).get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            msg = ""
        msg = re.sub(r"https?://\S+", "[URL]", str(msg).replace(key, "[REDACTED]"))[:300]
        raise RuntimeError(f"HTTP {error.code}: {msg}") from None
    choice = result["choices"][0]
    return {"content": choice["message"].get("content") or "", "finish_reason": choice.get("finish_reason"),
            "usage": result.get("usage", {}), "response_id": result.get("id"),
            "provider_model": result.get("model"), "reasoning_content": choice["message"].get("reasoning_content")}


def sampler_setup(key):
    sdk_path = ROOT / "tmp/fireworks_eval_sdk"
    if sdk_path.exists():
        sys.path.insert(0, str(sdk_path))
    from fireworks.training.sdk import FiretitanServiceClient, types
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", trust_remote_code=False)
    service = FiretitanServiceClient(api_key=key, base_url="https://api.fireworks.ai/training/v1/serverless")
    # This creates only a sampler. There is deliberately no training client.
    sampler = service.create_sampling_client(base_model="accounts/fireworks/models/qwen3p8-27b", tokenizer=tokenizer)
    return service, sampler, tokenizer, types


def sampler_call(context, messages, max_tokens, timeout):
    service, sampler, tokenizer, types = context
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False, return_dict=False)
    result = sampler.sample(prompt=types.ModelInput.from_ints(prompt_ids), num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0)).result(timeout=timeout)
    seq = result.sequences[0]
    generated = list(seq.tokens)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"content": text, "finish_reason": getattr(seq, "stop_reason", None),
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(generated)},
            "token_usage_source": "local_token_counts; no cache discount assumed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--model-resource", help="Override model resource for one-model fine-tuned comparison (chat route)")
    parser.add_argument("--eval-file", default="val_eval.jsonl")
    parser.add_argument("--run-dir", default="data/evals/baseline_20261102")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.model_resource and len(args.models) != 1:
        parser.error("--model-resource requires exactly one --models value")
    config = configuration()
    key = os.environ.get("FIREWORKS_API_KEY") or config.get("FIREWORKS_API_KEY") or config.get("api")
    if not key:
        raise RuntimeError("Missing Fireworks API key")
    eval_bytes = (ROOT / args.eval_file).read_bytes()
    rows = [json.loads(line) for line in eval_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    for row in rows:
        if [m["role"] for m in row["messages"]] != ["system", "user", "assistant"]:
            raise RuntimeError("Expected system/user input and one held-out assistant reference")
    allowed = set(re.search(r"Keys must come from this list, in this order: (.*?)\. Each value", rows[0]["messages"][0]["content"])[1].split(", "))
    eval_inputs = {sha(row["messages"][:-1]) for row in rows}
    for split in ("train.jsonl", "val.jsonl"):
        split_rows = [json.loads(line) for line in (ROOT / split).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if eval_inputs & {sha(row["messages"][:-1]) for row in split_rows}:
            raise RuntimeError("Evaluation inputs overlap " + split)
    out = ROOT / args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"eval_sha256": sha(eval_bytes), "evaluation_examples": len(rows), "temperature": 0,
                "max_tokens": args.max_tokens, "schema_fields": sorted(allowed), "structured_decoding": False,
                "scoring": "field/value multiset exact micro-F1; NFKC/casefold/whitespace variant secondary; invalid schema scores as empty; API errors count as empty; list order ignored",
                "request_policy": "system and user messages only; assistant reference never sent",
                "model_reasoning": {m: MODELS[m]["effort"] for m in MODELS},
                "latency_policy": "wall time of successful inference attempt; routes differ and are not a controlled hardware benchmark"}
    manifest_path = out / "protocol.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != protocol:
        raise RuntimeError("Existing evaluation protocol differs; choose a new run directory")
    manifest_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "eval_snapshot.jsonl").write_bytes(eval_bytes)
    selected = rows[:args.limit] if args.limit else rows
    for label in args.models:
        model = dict(MODELS[label])
        model["resource"] = args.model_resource or "accounts/fireworks/models/" + model["id"]
        if args.model_resource:
            model["route"] = "chat"
            model["rates"] = None  # Dedicated inference is not priced like serverless tokens.
        path = out / (label + "_predictions.jsonl")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        existing = {r["example_index"]: r for r in records}
        if any(r["model_resource"] != model["resource"] for r in records):
            raise RuntimeError("Saved predictions are for a different model")
        todo = [(i, row) for i, row in enumerate(selected) if i not in existing or (args.retry_errors and existing[i]["status"] != "ok")]
        context = None
        setup_error = None
        if todo and model.get("route") == "sampler":
            try:
                context = sampler_setup(key)
            except Exception as error:
                setup_error = type(error).__name__ + ": " + str(error).replace(key, "[REDACTED]")[:300]
                print(label + " sampler unavailable: " + setup_error, flush=True)
        def one(item):
            i, row = item
            reference = json.loads(row["messages"][-1]["content"])
            record = {"example_index": i, "document_id": row.get("id"), "model_resource": model["resource"],
                      "route": model.get("route", "chat"), "input_sha256": sha(row["messages"][:-1]),
                      "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
            start = time.perf_counter()
            try:
                if setup_error:
                    raise RuntimeError(setup_error)
                response = (sampler_call(context, row["messages"][:-1], args.max_tokens, args.timeout)
                            if context else chat_call(key, model, row["messages"][:-1], args.max_tokens, args.timeout))
                record.update(response)
                record["status"] = "ok"
                usage = response.get("usage", {})
                rates = model.get("rates")
                if rates:
                    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0
                    record["estimated_cost_usd"] = ((usage.get("prompt_tokens", 0)-cached)*rates[0] + cached*rates[1]
                        + usage.get("completion_tokens", 0)*rates[2]) / 1e6
            except Exception as error:
                record.update(status="error", error=type(error).__name__ + ": " + str(error).replace(key, "[REDACTED]")[:400], content="")
            record["latency_seconds"] = round(time.perf_counter() - start, 4)
            record["scores"] = score(record["content"], reference, allowed)
            return record
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(one, item) for item in todo]
                with path.open("a", encoding="utf-8") as handle:
                    processed_this_run = 0
                    for future in as_completed(futures):
                        record = future.result()
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        existing[record["example_index"]] = record
                        processed_this_run += 1
                        if processed_this_run % 10 == 0 or record["status"] != "ok" or processed_this_run == len(todo):
                            successful = sum(r["status"] == "ok" for r in existing.values())
                            print(f"{label}: {successful}/{len(rows)} successful; retry/run progress {processed_this_run}/{len(todo)}", flush=True)
                        summary = summarize(list(existing.values()), len(rows))
                        (out / (label + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        finally:
            if context:
                context[1].close()
        summary = summarize(list(existing.values()), len(rows))
        (out / (label + "_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(label + " SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
