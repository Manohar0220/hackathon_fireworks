# Fine-tuned DeepSeek deployment

The deployment belongs to `korikanamanohar2-me6` and references the shared fine-tuned model:

- Model: `accounts/siakatakhan-pgaz2u78/models/inv-deepseek`
- Base: `accounts/fireworks/models/deepseek-v4p1-flash`
- Adapter: rank-16 LoRA (`HF_PEFT_ADDON`)
- Deployment: `accounts/korikanamanohar2-me6/deployments/inv-deepseek-demo`

The model remains owned by the source account. Fireworks rejected the adapter export with `model downloading is restricted`; no adapter weights were downloaded or imported into the destination account. The CLI successfully validated and created a deployment referencing the shared model directly. No new training was performed.

## Calling it

Use the destination account's API key and this deployment identifier as the `model` parameter:

```json
{
  "model": "accounts/korikanamanohar2-me6/deployments/inv-deepseek-demo",
  "messages": [
    {"role": "system", "content": "Extract invoice fields and return JSON."},
    {"role": "user", "content": "YOUR OCR INPUT"}
  ],
  "temperature": 0,
  "max_tokens": 4096,
  "reasoning_effort": "none"
}
```

POST to `https://api.fireworks.ai/inference/v1/chat/completions`. For comparisons, use the exact system/user messages from the frozen evaluation snapshot instead of the abbreviated system prompt above. See [querying dedicated deployments](https://docs.fireworks.ai/guides/querying-text-models#querying-dedicated-deployments).

## Capacity and idle behavior

The validated shape is `deepseek-v4p1-flash-lora-b300-fp4-w4-p1`, version `xh9sq3zz`: four B300 GPUs, FP4, one maximum replica. Published on-demand pricing is $15 per B300 GPU-hour, or $60/hour while one replica is active. This is serving cost, not training cost. [Fireworks pricing](https://fireworks.ai/pricing)

Minimum replicas are zero; maximum replicas are one. The scale-to-zero idle window is 300 seconds. After idle scale-down, requests may receive HTTP 503 with `DEPLOYMENT_SCALING_UP` while the server warms up; callers must retry. [Autoscaling behavior](https://docs.fireworks.ai/deployments/autoscaling#scaling-from-zero-behavior)

## Status and evidence

```powershell
.\tmp\firectl\firectl.exe -p invoiceroute-source -a korikanamanohar2-me6 deployment get inv-deepseek-demo -o json
```

Local evidence is under `data/fireworks/deepseek_transfer/`:

- `source_model.json`: source model and base/adapter metadata.
- `deployment.json`: actual creation response.
- `deployment_status.json`: most recent monitored state.
- `inference_config.json`: endpoint and ownership details.
- `invoice_smoke.json`: created only after an inference request succeeds.

The existence of the deployment resource alone does not prove inference is ready. Check its state and the smoke-test evidence.
