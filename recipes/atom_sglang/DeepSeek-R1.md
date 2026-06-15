# DeepSeek-R1 with ATOM SGLang Backend

This recipe shows how to run `deepseek-ai/DeepSeek-R1-0528` or `amd/DeepSeek-R1-0528-MXFP4-v2` with the SGLang-ATOM backend. For background on the SGLang-ATOM integration, see [Introduce ATOM as external model package of SGLang](https://github.com/ROCm/ATOM/issues/359).

## Step 1: Pull the SGLang-ATOM Docker

```bash
docker pull rocm/atom-dev:sglang-latest
```

Launch a container from this image and run the remaining commands inside the container.

## Step 2: Launch SGLang-ATOM Server

The SGLang-ATOM backend keeps the standard SGLang CLI, server APIs, and general usage flow compatible with upstream SGLang. For general server options and API usage, users can refer to the [official SGLang documentation](https://docs.sglang.ai/).

### DeepSeek with FP8 (TP=8)

Users can use this command to launch the FP8 server with the same settings as the SGLang benchmark workflow.

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export SGLANG_USE_AITER=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
# Introduce ATOM as external model package of SGLang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_ENABLE_TORCH_COMPILE=1

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

### DeepSeek with FP8 (TP=4)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_FP8_PREFILL_ATTN=0
export SGLANG_USE_AITER=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
# Introduce ATOM as external model package of SGLang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_ENABLE_TORCH_COMPILE=1
TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

### DeepSeek with MXFP4 (TP=8)

AMD Instinct MI355X GPU supports MXFP4 computation instructions, and users can use the following command to launch the MXFP4 server on MI355X. For MXFP4 model weights, we suggest using the checkpoint quantized from AMD Quark.

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_FP8_PREFILL_ATTN=1
export SGLANG_USE_AITER=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
# Introduce ATOM as external model package of SGLang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_ENABLE_TORCH_COMPILE=1
TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-0528-MXFP4-v2 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

### DeepSeek with MXFP4 (TP=4)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_FP8_PREFILL_ATTN=1
export SGLANG_USE_AITER=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
# Introduce ATOM as external model package of SGLang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_ENABLE_TORCH_COMPILE=1
TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-0528-MXFP4-v2 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

### DeepSeek with MXFP4 (TP=4, DP=4, EP=4)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_FP8_PREFILL_ATTN=1
export SGLANG_USE_AITER=1
export SGLANG_ENABLE_TORCH_COMPILE=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-0528-MXFP4-v2 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --expert-parallel-size 4 \
    --data-parallel-size 4 \
    --enable-dp-attention \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```

### DeepSeek with MXFP4 (TP=8, DP=8, EP=8)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_AITER_FP8_PREFILL_ATTN=1
export SGLANG_USE_AITER=1
export SGLANG_ENABLE_TORCH_COMPILE=1
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-0528-MXFP4-v2 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --expert-parallel-size 8 \
    --data-parallel-size 8 \
    --enable-dp-attention \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache
```
In Addition, To align with mori prefill, you need to add --chunked-prefill-size 65536. This is also included in mori's startup command.


### DeepSeek with MXFP4 + MTP (TP=8, MTP=3)

Use checkpoint `amd/DeepSeek-R1-0528-MXFP4-v2` and `SGLang/DeepSeek-R1-NextN`, which includes MTP weights. To enable MTP decoding, launch SGLang with the `NEXTN` speculative decoding options. The example below shows three draft step.

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export SGLANG_AITER_FP8_PREFILL_ATTN=1
export SGLANG_USE_AITER=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_TORCH_COMPILE=1
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
TP_SIZE=8
MTP=${MTP:-3}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-$((MTP + 1))}

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-0528-MXFP4-v2 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size "${TP_SIZE}" \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache \
    --speculative-draft-model-path SGLang/DeepSeek-R1-NextN \
    --speculative-algorithm NEXTN \
    --speculative-num-steps "${MTP}" \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}" \
    --max-running-requests 256 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 160 192 224 256 
```

For a 4-GPU run, set `CUDA_VISIBLE_DEVICES` to the target devices and change `TP_SIZE=4`.

### DeepSeek with MXFP4 + MTP (MTP=3, TP=4, DP=4, EP=4)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export SGLANG_AITER_FP8_PREFILL_ATTN=1
export SGLANG_USE_AITER=1
export SGLANG_ENABLE_SPEC_V2=1
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_ENABLE_TORCH_COMPILE=1

export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

export CUDA_VISIBLE_DEVICES="0,1,2,3"
MTP=${MTP:-3}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-$((MTP + 1))}

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-0528-MXFP4-v2 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --expert-parallel-size 4 \
    --data-parallel-size 4 \
    --enable-dp-attention \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache \
    --speculative-draft-model-path SGLang/DeepSeek-R1-NextN \
    --speculative-algorithm NEXTN \
    --speculative-num-steps "${MTP}" \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}" \
    --max-running-requests 4096 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 160 192 224 256 
```

### DeepSeek with MXFP4 + MTP (MTP=3, TP=8, DP=8, EP=8)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1
export SGLANG_AITER_FP8_PREFILL_ATTN=1
export SGLANG_USE_AITER=1
export SGLANG_ENABLE_SPEC_V2=1
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_ENABLE_TORCH_COMPILE=1

export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
MTP=${MTP:-3}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-$((MTP + 1))}

TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path amd/DeepSeek-R1-0528-MXFP4-v2 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --expert-parallel-size 8 \
    --data-parallel-size 8 \
    --enable-dp-attention \
    --attention-backend aiter \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.8 \
    --page-size 1 \
    --disable-radix-cache \
    --speculative-draft-model-path SGLang/DeepSeek-R1-NextN \
    --speculative-algorithm NEXTN \
    --speculative-num-steps "${MTP}" \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}" \
    --max-running-requests 4096 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 160 192 224 256 
```
In addition, to align with MORI prefill, you need to add --chunked-prefill-size 65536. This is also included in mori's startup command.



## Step 3: Performance Benchmark

The SGLang benchmark workflow uses the `bench_serving` client for performance benchmarking. The following example matches the workflow command pattern for an MXFP4 TP4 case.

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

ISL=8192
OSL=1024
CONC=64
RANDOM_RANGE_RATIO=0.8
RESULT_DIR=./benchmark-results
RESULT_FILENAME=deepseek-r1-fp4-tp4-${ISL}-${OSL}-${CONC}-${RANDOM_RANGE_RATIO}.json

python3 /tmp/bench_serving/benchmark_serving.py \
    --model=amd/DeepSeek-R1-0528-MXFP4-v2 \
    --backend=sglang \
    --base-url=http://127.0.0.1:8000 \
    --dataset-name=random \
    --random-input-len="${ISL}" \
    --random-output-len="${OSL}" \
    --random-range-ratio "${RANDOM_RANGE_RATIO}" \
    --num-prompts="$(( CONC * 10 ))" \
    --max-concurrency="${CONC}" \
    --trust-remote-code \
    --num-warmups="$(( 2 * CONC ))" \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir="${RESULT_DIR}" \
    --result-filename="${RESULT_FILENAME}"
```

For FP8 or TP8 cases, keep the same benchmark command and replace `--model` with the checkpoint used in Step 2.

### Optional: Enable Profiling
If you want to collect profiling traces, set the SGLang profiling environment variables before launching the server, and add `--profile` to the benchmark client command.

```bash
export SGLANG_PROFILE_RECORD_SHAPES=1
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_TORCH_PROFILER_DIR=./profile_sglang/
```

Then append `--profile` to the `benchmark_serving.py` command in Step 3.

## Step 4: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=amd/DeepSeek-R1-0528-MXFP4-v2,base_url=http://localhost:8000/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 3
```
