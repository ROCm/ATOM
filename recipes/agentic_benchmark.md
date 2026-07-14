# This is a guide on how to run the InferenceX agentic benchmark.

## Install aiperf
InferenceX agentic benchmark uses a patched version of aiperf maintained in [https://github.com/SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX).
Install the patched version of aiperf from source:
```bash
git clone https://github.com/SemiAnalysisAI/InferenceX.git
cd InferenceX
git submodule update --init utils/aiperf

python -m pip install -r utils/agentic-benchmark/requirements.txt -e utils/aiperf
```

## Launch the server
Launch the server with ATOM/VLLM-ATOM/SGLANG-ATOM. Using ATOM as an example:
```bash
#!/bin/bash
model_path=amd/GLM-5.2-MXFP4
model_name=$(basename ${model_path})

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

PORT=8001
TP=4

python -m atom.entrypoints.openai_server \
    --model ${model_path} \
    --trust-remote-code \
    --server-port ${PORT} \
    --kv_cache_dtype fp8 \
    -tp ${TP} \
    --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate", "model.layers.[0-9].mlp.*expert*","model.layers.[1-6][0-9].mlp.*expert*","model.layers.7[0-7].mlp.*expert*"]}' \
    --num-speculative-tokens 3 \
    --method mtp \
    2>&1 | tee "logs/${model_name}-atom-server.log"

```

## Run aiperf benchmark
Run benchmark using real agentic conversation traces, `--scenario inferencex-agentx-mvp`:
```bash
#!/bin/bash
model=amd/GLM-5.2-MXFP4
PORT=8001

aiperf profile --scenario inferencex-agentx-mvp --url http://localhost:${PORT} --endpoint /v1/chat/completions --endpoint-type chat --streaming --model ${model} --concurrency 16 --benchmark-duration 1800 --random-seed 42 --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --agentic-cache-warmup-duration 600 --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 --output-artifact-dir ./results/aiperf_artifacts --public-dataset semianalysis_cc_traces_weka_062126
```
