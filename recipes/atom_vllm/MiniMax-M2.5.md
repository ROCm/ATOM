# MiniMax-M2.5 with ATOM vLLM Plugin Backend

This recipe shows how to run `MiniMaxAI/MiniMax-M2.5` (HF `architectures[0]`: `MiniMaxM2ForCausalLM`, MoE + FP8 weights) with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

The checkpoint uses custom modeling code; keep `--trust-remote-code` on the server command line.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

The following matches the internal benchmark entry (`--kv_cache_dtype fp8 -tp 2 --trust-remote-code` in `.github/benchmark/models.json`). On multi-GPU hosts, use tensor parallel size 2 or adjust to your topology.

```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --gpu_memory_utilization 0.9 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --no-enable-prefix-caching
```

Caveat: the upstream `config.json` may advertise MTP-related fields; the current ATOM `MiniMaxM2ForCausalLM` path targets the main transformer. If you hit load or shape errors around MTP modules, compare with native ATOM server behavior and upstream vLLM release notes.

## Step 3: Performance Benchmark

```bash
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model MiniMaxAI/MiniMax-M2.5 \
    --dataset-name random \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --random-range-ratio 0.8 \
    --max-concurrency 64 \
    --num-prompts 640 \
    --trust_remote_code \
    --percentile-metrics ttft,tpot,itl,e2el
```

## Step 4: Accuracy Validation

Nightly OOT accuracy uses `gsm8k` with **3-shot** in `.github/scripts/atom_oot_test.sh` (same as other full-validation models). For a local check:

```bash
lm_eval --model local-completions \
        --model_args model=MiniMaxAI/MiniMax-M2.5,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 3 \
        --output_path ./lm_eval_minimax_m25_gsm8k
```

Reference metric (tracking baseline for this model family; replace with your run output and keep the raw JSON path next to the table):

- Internal tracking: `accuracy_baseline` **0.9401** for `MiniMaxAI/MiniMax-M2.5` in `.github/benchmark/models_accuracy.json` (see `_baseline_note` there for HF card context).
- OOT gate: `accuracy_test_threshold` **0.92** on `exact_match,flexible-extract` (see `atom-vllm-oot-test.yaml` nightly matrix).

Example table shape after `lm_eval` (fill `Value` / `Stderr` from your console or `${output_path}` JSON):

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.9287|±  |0.0071|
|     |       |strict-match    |     3|exact_match|↑  |0.9272|±  |0.0072|
```

Raw results JSON: `<path-to-lm_eval-output-*.json>` (from `--output_path` above).
