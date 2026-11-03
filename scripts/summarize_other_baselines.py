"""Rescore saved GLM, Qwen and Nemotron outputs with the DeepSeek comparison rules."""
from collections import Counter
import csv
import json

from compare_deepseek import fuzzy_counts, percentile
from evaluate_fireworks import ROOT, metrics, score, sha
from summarize_evaluation import content_pairs, pair_counts


def main():
    folder = ROOT / 'data/evals/baseline_20261102'
    protocol = json.loads((folder / 'protocol.json').read_text())
    snapshot = (folder / 'eval_snapshot.jsonl').read_bytes()
    assert sha(snapshot) == protocol['eval_sha256']
    rows = [json.loads(s) for s in snapshot.decode('utf-8-sig').splitlines() if s.strip()]
    definitions = json.loads((ROOT / 'data/evals/deepseek_comparison_20261102/comparison.json').read_text())
    report = {k: definitions[k] for k in ('fuzzy_rule', 'exact_rule', 'document_success_rule')}
    report.update(evaluation_examples=len(rows), evaluation_sha256=sha(snapshot), training_performed=False,
                  source='Previously saved baseline outputs; no new inference calls.',
                  latency_rule='Successful final attempt wall time; p50/p95 linear interpolation. Qwen uses Training SDK sampling, GLM and Nemotron use chat inference. Retries excluded; routes and reasoning settings differ.', models={})
    table = []
    for key, name in [('glm', 'GLM 5.3 Flash'), ('qwen', 'Qwen3.8 27B'), ('nemotron', 'Nemotron Lightning 3.5')]:
        attempts = [json.loads(s) for s in (folder / (key+'_predictions.jsonl')).read_text(encoding='utf-8').splitlines() if s.strip()]
        latest = {r['example_index']: r for r in attempts}
        assert set(latest) == set(range(len(rows))), 'Incomplete coverage: '+key
        exact, fuzzy = Counter(tp=0, fp=0, fn=0), Counter(tp=0, fp=0, fn=0)
        perfect = errors = json_ok = schema_ok = 0
        latencies = []
        for index, row in enumerate(rows):
            result = latest[index]
            assert result['input_sha256'] == sha(row['messages'][:-1])
            gold = json.loads(row['messages'][-1]['content'])
            scored = score(result.get('content', ''), gold, set(protocol['schema_fields']))
            p, g = content_pairs(scored['parsed']), content_pairs(gold)
            exact.update(pair_counts(p, g)); fuzzy.update(fuzzy_counts(p, g))
            perfect += bool(scored['schema_valid'] and p == g)
            json_ok += scored['json_valid']; schema_ok += scored['schema_valid']
            if result['status'] == 'ok': latencies.append(result['latency_seconds'])
            else: errors += 1
        assert errors == 0, 'Unresolved API errors: '+key
        n = len(rows)
        record = dict(resource=latest[0]['model_resource'], route=latest[0]['route'], examples=n,
                      successful_responses=n-errors, request_errors=errors, exact=metrics(exact), fuzzy=metrics(fuzzy),
                      document_success_count=perfect, document_success_rate=perfect/n,
                      valid_json_rate=json_ok/n, valid_schema_rate=schema_ok/n,
                      latency_p50_seconds=percentile(latencies, .5), latency_p95_seconds=percentile(latencies, .95),
                      reasoning_effort=protocol['model_reasoning'][key], recorded_api_attempts=len(attempts))
        report['models'][key] = record
        table.append(dict(model=name, responses=f'{n-errors}/{n}',
                          field_f1_fuzzy_percent=round(record['fuzzy']['f1']*100, 2),
                          field_f1_exact_percent=round(record['exact']['f1']*100, 2),
                          precision_exact_percent=round(record['exact']['precision']*100, 2),
                          recall_exact_percent=round(record['exact']['recall']*100, 2),
                          doc_success_percent=round(perfect/n*100, 2),
                          latency_p50_seconds=round(record['latency_p50_seconds'], 3),
                          latency_p95_seconds=round(record['latency_p95_seconds'], 3)))
    (folder / 'other_baselines_table.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (folder / 'other_baselines_table.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
