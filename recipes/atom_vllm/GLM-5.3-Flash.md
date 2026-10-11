# GLM-5.3-Flash with ATOM vLLM Plugin Backend

This recipe shows how to run GLM-5.3-Flash (`Glm5NextForConditionalGeneration`,
text only) with the ATOM vLLM plugin backend. For background on the plugin
backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

```bash
vllm serve zai-org/GLM-5.3-Flash \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching
```

### GLM-5.3-Flash MTP (TP=8, MI355X)

```bash
vllm serve zai-org/GLM-5.3-Flash \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

### GLM-5.3-Flash with LMCache KV offload (TP=8, MI355X)

KV is offloaded to an LMCache CPU tier through `AtomLMCacheOffloadConnector`.

```bash
export PYTHONHASHSEED=0                 # on the client too
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=96    # GiB per TP rank
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_CACHE_POLICY=ATOM_SLRU
export LMCACHE_TRACK_USAGE=false
export OFFLOAD_MIN_LOAD_TOKENS=256      # default 8192 skips chat-sized prompts

vllm serve zai-org/GLM-5.3-Flash \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
```

Prefix caching must stay on: it selects `--mamba-cache-mode align`, the only
mode the connector accepts for this hybrid model. `LMCACHE_CHUNK_SIZE` must be a
multiple of the block size vLLM settles on, which the log line
`Setting attention block size to N tokens` reports (1024 at TP8 with an fp8 KV
cache).

To check that the connector is on:

```bash
grep -E "recurrent state leg on group|registered [0-9]+ layers" server.log
```

```text
ATOM LMCache offload: recurrent state leg on group(s) 0,1,2 (mamba_block=1024, hash_block=1024, chunk=1024)
ATOM LMCache offload: registered 12 layers, num_blocks=N (leading dim 16N, block_size=1024)
```

12 layers is the 11 MLA layers plus the index proxy, 13 with MTP. A count of 11
means the pooled index rows are not being moved.

## Step 3: Performance Benchmark
Users can use the default vllm bench commands for performance benchmarking.
```bash
ISL=1000
OSL=100
CONC=4

vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model zai-org/GLM-5.3-Flash \
    --dataset-name random \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --random-range-ratio 0.0 \
    --max-concurrency "${CONC}" \
    --num-prompts "$(( CONC * 8 ))" \
    --trust_remote_code \
    --num-warmups "${CONC}" \
    --request-rate inf \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --percentile-metrics ttft,tpot,itl,e2el
```

## Step 4: Accuracy Validation

GLM-5.3-Flash answers in markdown, so GSM8K is scored in chat mode with the
few-shot examples as turns.

```bash
lm_eval --model local-chat-completions \
        --model_args model=zai-org/GLM-5.3-Flash,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 20 \
        --gen_kwargs max_gen_toks=512 \
        --apply_chat_template \
        --fewshot_as_multiturn
```

| configuration | flexible-extract | strict-match |
| --- | ---: | ---: |
| TP8, fp8 KV | 0.9674 | 0.9682 |
| TP8, fp8 KV, MTP 3 | 0.9697 | 0.9697 |
