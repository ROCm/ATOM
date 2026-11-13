> ⚠️ **EXAMPLE**: a made-up repro that only illustrates the format.

# Qwen-35B-A3B: c64 fails with HIP out of memory (455-CI 2026-11-14)

## Environment

- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114` (digest `sha256:0000mock`)
- ATOM `main@000000000`, aiter `main@111111111`; last good session 2026-11-07
- Machine: e15-06, 1 x MI455X; weights: `<HF repo> @ <revision>` (mock)

## Steps

1. Start the test container as in 455-CI-recipe.md.
2. Start the server:

   ```bash
   python3 -m atom.entrypoints.openai_server --model /models/<Qwen-35B-A3B> --kv_cache_dtype fp8    # (mock)
   ```

3. Run the benchmark at concurrency 64, as in `bench_cmd.sh` of this run with `CONC=64`.

## Expected vs actual

- Expected: c64 completes, about 6,345 tok/s/GPU as on 2026-11-07.
- Actual: the server exits while capturing CUDA graphs with
  `RuntimeError: HIP error: out of memory (mock)`. 2 of 2 runs; c4 passes.

## Logs

[error.log](error.log), the core excerpt of the server log.

## What we checked

- Open and recently merged PRs in ROCm/ATOM and ROCm/aiter for "out of memory",
  "cudagraph" and "Qwen": none about this.
- Not tried yet: a lower `--gpu-memory-utilization`.
