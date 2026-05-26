# ATOM vLLM Benchmark Guide

Use this guide when the user asks to benchmark ATOM vLLM Plugin performance, compare with the upstream vLLM (optional), or validate a performance claim.

## What this guide should do

1. Confirm benchmark goal and variables (throughput, TTFT, TPOT, E2EL, etc.).
2. Use plugin-on as default setup; optionally benchmark upstream vLLM for performance comparison.
3. Run repeatable measurements and summarize median results.
4. Report reproducible commands, key metrics, and known risks.
5. Prefer model-specific launch settings from `recipes/atom_vllm/` when available; otherwise use the standard workflow in this guide.

## Inputs To Collect First

Before running any benchmark, confirm:

- Model path (HF id or local path)
- Serving backend (`vllm`)
- Hardware shape (GPU type/count)
- `tensor_parallel_size`, KV cache dtype, and scheduler flags
- Request mix (ISL/OSL/concurrency/request rate)
- Comparison target:
  - default candidate (plugin on)
  - optional baseline/control (plugin off)
  - optional runtime baseline (upstream vLLM nightly)

If any of these are missing, ask before execution.

## Environment Checklist

For each benchmark round:

- Kill old server processes and verify VRAM is released.
- Clear cache when startup behavior is inconsistent:
  - `rm -rf /root/.cache/atom/*`
  - `rm -rf /root/.cache/vllm/*`
- Set stable logging:
  - `export AITER_LOG_LEVEL=WARNING`
- Confirm server is actually ready:
  - `curl -sf http://localhost:8000/v1/models`
  - `rocm-smi --showmemuse` and verify allocated memory is greater than 0.

## Benchmark Workflow (Plugin On by Default)

Run plugin-on by default.

### Recipe-first execution policy

When running ATOM vLLM benchmark:

1. First check whether the target model has a matching recipe under `recipes/atom_vllm/`.
2. If a matching recipe exists, use its model-specific environment variables and `vllm serve` launch args as the default benchmark setup.
3. If no matching recipe exists, follow the standard launch template and workflow in this guide.

Keep benchmark methodology unchanged (smoke check, concurrency isolation, and fixed comparison variables) regardless of whether the launch config comes from a recipe or from the default template.

- **Default (candidate):** plugin enabled
- **Optional control (baseline):** plugin disabled (only when A/B comparison is requested)
- **Default ATOM vLLM Plugin image:** `rocm/atom-dev:vllm-latest`

If both conditions are run, only one variable is allowed to change between them.

### 1) Launch server

Use the same launch command template. Keep plugin-on as default, and only change plugin toggle env vars when running the optional plugin-off control.

```bash
vllm serve <model_path> \
  --host localhost \
  --port 8000 \
  --tensor-parallel-size <tp> \
  --attention-backend ROCM_AITER_FA \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --gpu_memory_utilization 0.9 \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --no-enable-prefix-caching
```

Common plugin toggles:

- Default plugin-on: do not set `ATOM_DISABLE_VLLM_PLUGIN` or `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION` (or set them to `0`)
- Optional plugin-off control: set `ATOM_DISABLE_VLLM_PLUGIN=1`
- Optional attention-off control: set `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1`
- Always set `VLLM_ROCM_USE_AITER=1` in both ATOM plugin and upstream vLLM runs.

### 2) Run smoke validation

```bash
curl -sf http://localhost:8000/v1/models
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --model <model> \
  --dataset-name random \
  --random-input-len 32 \
  --random-output-len 32 \
  --temperature 0.0 \
  --max-concurrency <smoke_conc> \
  --num-prompts <smoke_conc_x10> \
  --num-warmups <smoke_conc_x2> \
  --request-rate 1 \
  --disable-tqdm
```

Use `smoke_conc_x10 = smoke_conc * 10` and `smoke_conc_x2 = smoke_conc * 2`.

If smoke test fails, do not continue with performance comparison.

### 3) Run benchmark workload

```bash
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --model=<model> \
  --dataset-name=random \
  --random-input-len=<isl> \
  --random-output-len=<osl> \
  --random-range-ratio=0.8 \
  --temperature=0.0 \
  --num-prompts=<conc_x10> \
  --num-warmups=<conc_x2> \
  --max-concurrency=<conc> \
  --trust-remote-code \
  --request-rate=inf \
  --ignore-eos \
  --disable-tqdm \
  --save-result \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

### 4) Concurrency sweep isolation (required)

When sweeping multiple concurrency points, each point must run in a fresh container lifecycle:

1. Start a fresh container.
2. Start vLLM server in that container.
3. Run smoke + benchmark for one `--max-concurrency=<conc>`.
4. Stop vLLM server.
5. Exit and stop/remove the container.
6. Start a new container for the next concurrency point.

Do not reuse the same running server/container across different concurrency points.

Example pattern:

```bash
for conc in 4 8 16 32 64 128; do
  docker run -dt --name "atom_bench_c${conc}" <image>
  docker exec "atom_bench_c${conc}" bash -lc 'vllm serve <model_path> ...'
  docker exec "atom_bench_c${conc}" bash -lc "vllm bench serve ... --max-concurrency=${conc} ..."
  docker exec "atom_bench_c${conc}" bash -lc 'pkill -f "vllm serve" || true'
  docker stop "atom_bench_c${conc}" && docker rm "atom_bench_c${conc}"
done
```

### 5) Minimum benchmark matrix

| Scenario | ISL | OSL | Concurrency |
|---|---:|---:|---:|
| Short in / short out | 1024 | 1024 | 4, 8, 16, 32, 64, 128 |
| Short in / long out | 1024 | 8192 | 4, 8, 16, 32, 64, 128 |
| Long in / short out | 8192 | 1024 | 4, 8, 16, 32, 64, 128 |

### 6) Repetition and statistics

- Run each scenario at least 1 times.
- Use median values for comparison.
- If variance is large, increase repeat count.

## Optional Upstream vLLM Benchmark

Use this optional path when you need ATOM vs upstream runtime comparison.

Reference images:

- ATOM vLLM plugin default image: `rocm/atom-dev:vllm-lastest`
- Upstream vLLM comparison image: `vllm/vllm-openai-rocm:nightly`

### 1) Pull latest upstream ROCm image

```bash
docker pull vllm/vllm-openai-rocm:nightly
```

### 2) Run the same workflow and config

For upstream runs, keep everything identical to ATOM runs:

- Same model path
- Same GPU set and `tensor_parallel_size`
- Same KV cache dtype and scheduler flags
- Same request mix (ISL/OSL/concurrency/request rate)
- Same concurrency sweep isolation policy (restart container per concurrency point)

Only runtime image is allowed to change.

### 3) Upstream execution note

- Use the same smoke and benchmark commands from this guide.
- Do not set ATOM plugin toggle env vars for upstream runs.
- Always set `VLLM_ROCM_USE_AITER=1` for upstream runs.
- Keep result naming explicit (for example, include `upstream-nightly` in filenames).

## Regression Profiling Workflow

When regression is observed:

```bash
# 1) Start (or restart) server with profiler enabled
vllm serve <model_path> \
  --host localhost \
  --port 8000 \
  --tensor-parallel-size <tp> \
  --attention-backend ROCM_AITER_FA \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --profiler-config '<profiler_config_json>'

# 2) Run profiled benchmark workload
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --model <model> \
  --dataset-name random \
  --random-input-len <isl> \
  --random-output-len 20 \
  --temperature 0.0 \
  --max-concurrency <conc> \
  --num-prompts <conc_x2> \
  --num-warmups <conc_x2> \
  --request-rate inf \
  --profile \
  --save-result
```

- `--profile` requires server-side `--profiler-config`.
- Keep profiler config identical between plugin-off and plugin-on runs.
- For reproducibility, keep benchmark client semantics fixed to completion mode:
  `--backend vllm --endpoint /v1/completions` (this path uses `max_tokens`).

## Reporting Template

Use this format in final response:

```markdown
### ATOM vLLM Benchmark Result
- Goal: <what was measured and why>
- Setup:
  - Model: <model>
  - Hardware: <gpu and count>
  - Fixed args: <tp, kv dtype, scheduler flags>
  - Isolation: <restart container per concurrency point: yes/no>
- Validation: <smoke test pass/fail>
- Benchmark:
  - Default (plugin on): <throughput/latency median>
  - Optional control (plugin off): <throughput/latency median, if run>
  - Optional runtime baseline (upstream nightly): <throughput/latency median, if run>
  - Delta: <percent change, if both runs exist>
- Confidence:
  - Runs: <repeat count>
  - Variance: <low/medium/high>
- Risk:
  - <known risks or unknowns>
- Repro:
  - <exact default launch command (plugin on)>
  - <exact optional control launch command (plugin off), if run>
  - <exact benchmark command>
```

## Guardrails

- Never compare runs with different model, TP, KV dtype, or request mix.
- For concurrency sweep, stop server and restart container between concurrency points.
- For ATOM vs upstream comparison, only runtime image may differ.
- Keep `VLLM_ROCM_USE_AITER=1` fixed for both ATOM and upstream runs.
- If GPU memory is not allocated, treat benchmark results as invalid.
- If plugin-on and plugin-off outputs differ semantically, run accuracy checks before claiming performance gains.
