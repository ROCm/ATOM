# Exact ATOM tokenizer ingress

Use the deployment commands in the existing
[shared Store recipe](../../recipes/DeepSeek-V4.1-6P16D-C1228-Share-Cache.md#router-and-encoding-ingress).
This directory supplies the CPU ingress; it does not implement routing or KV transfer.

The default final text encoder is fastokens 0.3.6 with both input and segment
caches disabled. ATOM's checkpoint-specific chat renderer, thinking controls,
tool normalization, and decoder remain authoritative. The optional
`--encode-backend huggingface` changes only final text encoding.

Run these CPU checks from the repository root after installing the requirements:

```bash
python3 examples/pd_tokenizer_ingress/verify_encoding.py --model "$MODEL_PATH"
python3 examples/pd_tokenizer_ingress/tests/http_lifecycle_probe.py \
  --source examples/pd_tokenizer_ingress --output /tmp/ingress-http.json
python3 examples/pd_tokenizer_ingress/tests/cli_budget_probe.py \
  --launcher examples/pd_tokenizer_ingress/launch.py --output /tmp/ingress-cli.json
```

`verify_encoding.py` loads the real checkpoint tokenizer and custom ATOM encoder.
It compares complete forwarded bodies, chat/tool/thinking cases, supplied token
IDs, Unicode/special-token behavior, and unsupported-mode rejection. Pass
`--requests requests.json` to include an array of real workload requests.

The HTTP probe executes the real ingress with controlled encoder/upstream
boundaries: native work retains its slot after disconnect/timeout; submission
failure and completion release it; raw streaming/status/headers and upstream
closure survive cancellation races. It does not establish model encoding
equivalence. The CLI probe intercepts only Uvicorn startup and checks total
budgets and invalid arguments; it does not establish process startup or GPU performance.
