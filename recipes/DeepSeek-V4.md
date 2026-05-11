# DeepSeek-V4 Usage Guide

[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) is a million-token-context Mixture-of-Experts (MoE) large language model from DeepSeek. It builds on the V3.2 architecture with hash-based expert routing (3 hash layers + sigmoid + bias), a Compressed Sparse Attention (CSA) indexer that selects top-1024 prior tokens per query, and Multi-Latent Attention (MLA) with LoRA-compressed QKV projections. Weights are stored natively in FP8 (E4M3) with UE8M0 block-scaled scales. ATOM ships built-in support via the `DeepseekV4ForCausalLM` architecture — no `--trust-remote-code` is needed.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below will be executed inside the container.

## Launching server

### FP8 on 8xMI355X GPUs (TP8 + FP8 KV Cache)

```bash
ATOM_USE_TRITON_MOE=1 \
python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --kv_cache_dtype fp8 -tp 8
```

Tips on server configuration:
- **`ATOM_USE_TRITON_MOE=1` is required.** V4-Pro routes 6 experts out of 384 with hash-based selection; the triton MoE backend is the only path that handles the FP8 E4M3 + UE8M0 block-scaled weights correctly. Launching without this env silently falls back to a numerically incorrect path and GSM8K accuracy drops from ~0.95 to ~0.6.
- Use `--kv_cache_dtype fp8` for memory efficiency. The CSA indexer's compressed K cache is stored separately in FP8 regardless.
- Set `AITER_LOG_LEVEL=WARNING` before starting to suppress aiter kernel log noise.
- Clear compile cache before restarting after code changes: `rm -rf /root/.cache/atom/*`
- V4-Pro reuses the DeepSeek-V3 config schema; V4-specific fields (compress ratios, hash layers, index head dims) are read from the HF config automatically.

## Performance baseline

The following script can be used to benchmark the performance:

```bash
python -m atom.benchmarks.benchmark_serving \
  --model=deepseek-ai/DeepSeek-V4-Pro --backend=vllm --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=${ISL} --random-output-len=${OSL} \
  --random-range-ratio=1.0 \
  --num-prompts=$(( $CONC * 10 )) \
  --max-concurrency=$CONC \
  --request-rate=inf --ignore-eos \
  --save-result --percentile-metrics="ttft,tpot,itl,e2el"
```

Performance on 8xMI355X GPUs with the following environment:
- Date measured: 2026-05-07.
- Docker image: rocm/atom:latest.
- ATOM: main branch (commit 33c54649).
- `ATOM_USE_TRITON_MOE=1`, `--kv_cache_dtype fp8`.

The numbers below are a snapshot. For the latest data tracked across commits, see [rocm.github.io/ATOM/benchmark-dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/).

### FP8 (TP8, FP8 KV Cache)

| ISL  | OSL  | Concurrency | Num Prompts | Output Throughput (tok/s) | Total Throughput (tok/s) | Mean TPOT (ms) |
| ---- | ---- | ----------- | ----------- | ------------------------- | ------------------------ | -------------- |
| 1024 | 1024 | 4           | 40          | 111.03                    | 222.06                   | 35.67          |
| 1024 | 1024 | 8           | 80          | 196.19                    | 392.39                   | 40.25          |
| 1024 | 1024 | 16          | 160         | 369.79                    | 739.59                   | 42.36          |
| 1024 | 1024 | 32          | 320         | 660.31                    | 1320.62                  | 46.85          |
| 1024 | 1024 | 64          | 640         | 1138.68                   | 2277.37                  | 53.64          |
| 1024 | 1024 | 128         | 1280        | 1888.45                   | 3776.90                  | 63.41          |
| 1024 | 1024 | 256         | 2560        | 2926.71                   | 5853.41                  | 79.66          |

Here are the steps to reinstall ATOM/AITER in the docker, if you are trying to verify with other specific commits:
```bash
# uninstall existing ATOM/AITER
pip uninstall -y atom amd-aiter

cd PATH_TO_ATOM
# normally ATOM is already installed in develop mode
# you may just do checkout without reinstall
git checkout specific_branch_or_commit
pip install -e .

cd PATH_TO_AITER
rm -rf aiter/jit/build aiter/jit/*.so
git checkout specific_branch_or_commit
git submodule sync && git submodule update --init --recursive
python setup.py develop
```

### Accuracy test

We verified the lm_eval accuracy on gsm8k dataset with command:
```bash
lm_eval \
  --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-V4-Pro,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
```

Reference accuracy on 8xMI355X GPUs (FP8, FP8 KV Cache, `ATOM_USE_TRITON_MOE=1`):
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9530|±  |0.0058|
|     |       |strict-match    |     5|exact_match|↑  |0.9538|±  |0.0058|
```
