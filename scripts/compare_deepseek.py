"""Compare frozen baseline/tuned DeepSeek predictions; no model API calls."""

import argparse
from collections import Counter
import csv
from difflib import SequenceMatcher
import json
import math
from pathlib import Path

from evaluate_fireworks import ROOT, normalized, score, sha
from summarize_evaluation import content_pairs, pair_counts
from evaluate_fireworks import metrics


def fuzzy_counts(pred, gold, threshold=0.9):
    """Maximum one-to-one matching within each field; malformed values cannot match."""
    matched = 0
    fields = {key[0] for key in pred} | {key[0] for key in gold}
    for field in fields:
        pv = [value for (key, value), count in pred.items() if key == field for _ in range(count)]
        gv = [value for (key, value), count in gold.items() if key == field for _ in range(count)]
        edges = []
        for value in pv:
            candidates = []
            if isinstance(value, str):
                value = normalized(value)
                for index, ref in enumerate(gv):
                    if not isinstance(ref, str):
                        continue
                    ratio = SequenceMatcher(None, value, normalized(ref), autojunk=False).ratio()
                    if ratio >= threshold:
                        candidates.append((ratio, index))
            edges.append([index for _,index in sorted(candidates, reverse=True)])
        owner = [-1] * len(gv)
        def augment(index, seen):
            for target in edges[index]:
                if target in seen:
                    continue
                seen.add(target)
                if owner[target] == -1 or augment(owner[target], seen):
                    owner[target] = index
                    return True
            return False
        for index in range(len(pv)):
            matched += augment(index, set())
    return Counter(tp=matched, fp=sum(pred.values())-matched, fn=sum(gold.values())-matched)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values)-1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi]-values[lo]) * (pos-lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="data/evals/deepseek_comparison_20261102")
    args = parser.parse_args()
    folder = ROOT / args.run_dir
    baseline_protocol = json.loads((folder/'baseline/protocol.json').read_text())
    tuned_protocol = json.loads((folder/'finetuned/protocol.json').read_text())
    if baseline_protocol != tuned_protocol:
        raise RuntimeError("Generation/scoring protocols differ")
    snapshot = (folder/'baseline/eval_snapshot.jsonl').read_bytes()
    if snapshot != (folder/'finetuned/eval_snapshot.jsonl').read_bytes() or sha(snapshot) != baseline_protocol['eval_sha256']:
        raise RuntimeError("Evaluation snapshots differ")
    rows = [json.loads(line) for line in snapshot.decode('utf-8-sig').splitlines() if line.strip()]
    report = {'evaluation_examples':len(rows),'evaluation_sha256':sha(snapshot),
        'input':'OCR text with source positions','temperature':0,'max_tokens':baseline_protocol['max_tokens'],
        'reasoning_effort':'none','fuzzy_rule':'Maximum one-to-one matches within the same field; difflib.SequenceMatcher(autojunk=False) ratio >=0.90 after NFKC, casefolding, and whitespace collapse. Digits/punctuation retained.',
        'exact_rule':'Exact field-name/string multiset matching; list order ignored. Empty arrays contribute no values. Unknown fields/malformed value types are false positives; invalid JSON/non-object output is an empty prediction.',
        'document_success_rule':'Valid schema and all field/value instances exactly match the reference, with no extra or missing values.',
        'latency_rule':'Successful request wall-clock latency, p50/p95 by linear interpolation. Baseline uses serverless inference; tuned uses a dedicated four-B300 FP4 deployment. Hardware/routes differ; this is not a controlled hardware comparison.',
        'training_performed':False,'models':{}}
    csv_rows=[]
    for label in ['baseline','finetuned']:
        attempts=[json.loads(line) for line in (folder/label/'deepseek_predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        latest={r['example_index']:r for r in attempts}
        if set(latest) != set(range(len(rows))):
            raise RuntimeError(label+' has incomplete example coverage')
        exact=Counter(tp=0,fp=0,fn=0);fuzzy=Counter(tp=0,fp=0,fn=0)
        latencies=[];perfect=json_valid=schema_valid=errors=0
        per_doc=[]
        for index in range(len(rows)):
            row,r=rows[index],latest[index]
            if r['input_sha256']!=sha(row['messages'][:-1]):raise RuntimeError('Input mismatch')
            gold=json.loads(row['messages'][-1]['content'])
            scored=score(r.get('content',''),gold,set(baseline_protocol['schema_fields']))
            p,g=content_pairs(scored['parsed']),content_pairs(gold)
            e=pair_counts(p,g);f=fuzzy_counts(p,g)
            exact.update(e);fuzzy.update(f)
            doc_success=bool(scored['schema_valid'] and p==g)
            perfect+=doc_success;json_valid+=scored['json_valid'];schema_valid+=scored['schema_valid']
            if r['status']=='ok':latencies.append(r['latency_seconds'])
            else:errors+=1
            per_doc.append({'id':row.get('id'),'exact':dict(e),'fuzzy':dict(f),'document_success':doc_success,'status':r['status']})
        n=len(rows)
        record={'resource':latest[0]['model_resource'],'examples':n,'successful_responses':n-errors,
            'request_errors':errors,'complete':errors==0,'exact':metrics(exact),'fuzzy':metrics(fuzzy),
            'document_success_count':perfect,'document_success_rate':perfect/n,
            'valid_json_rate':json_valid/n,'valid_schema_rate':schema_valid/n,
            'latency_p50_seconds':percentile(latencies,.5),'latency_p95_seconds':percentile(latencies,.95),
            'recorded_api_attempts':len(attempts)}
        report['models'][label]=record
        (folder/(label+'_per_document_scores.json')).write_text(json.dumps(per_doc,indent=2),encoding='utf-8')
        csv_rows.append({'model':'DeepSeek V4.1 Flash ('+label+')','responses':str(n-errors)+'/'+str(n),
            'field_f1_fuzzy_percent':round(record['fuzzy']['f1']*100,2),
            'field_f1_exact_percent':round(record['exact']['f1']*100,2),
            'precision_exact_percent':round(record['exact']['precision']*100,2),
            'recall_exact_percent':round(record['exact']['recall']*100,2),
            'doc_success_percent':round(record['document_success_rate']*100,2),
            'latency_p50_seconds':round(record['latency_p50_seconds'],3) if latencies else None,
            'latency_p95_seconds':round(record['latency_p95_seconds'],3) if latencies else None})
    report['exact_f1_change_percentage_points']=(report['models']['finetuned']['exact']['f1']-report['models']['baseline']['exact']['f1'])*100
    (folder/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps({'table':csv_rows,'exact_f1_change_percentage_points':report['exact_f1_change_percentage_points']},indent=2))


if __name__=='__main__':
    main()
