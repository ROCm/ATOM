# DeepSeek-V4.1-Flash DEP4: idle DP ranks raise in the LM head GEMM, engine stops (455-CI 2026-11-28)

## Environment
- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261128` (digest `sha256:58d7808f5fbfdf020616ebfc421327e095e839d2b59d1b296861ff09dca8d2fd`), ROCm 10.2.0 base
- ATOM `main@380aeb8a0`, aiter `main@9da592818`; last good: aiter `2989e550f` (2026-11-24/25, DEP4 GSM8K completes)
- Machine: one MI455X node, 4 GPUs; weights: `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a`

## Steps
1. Start the test container with the ROCm 10.2 base's venv on `PATH` (as in [../deepseek-v4-pro/repro.md](../deepseek-v4-pro/repro.md)).
2. Start the server with the 455-CI recipe's DEP4 command, as run: [server_cmd_dep4.sh](server_cmd_dep4.sh).
3. Send GSM8K (recipe §3.3) or any concurrent traffic; or, on one GPU:
   ```python
   import torch
   from aiter.tuned_gemm import tgemm
   w = torch.randn(129280, 5120, device="cuda").bfloat16()
   tgemm.mm(torch.empty(0, 5120, device="cuda").bfloat16(), w)  # with the recipe's AITER_CONFIG_GEMM_BF16
   ```

## Expected vs actual
- Expected: an empty `[0, 129280]` result; the server serves GSM8K and the c64 bench.
- Actual: `ValueError: dimensions, tiles, stages, wave counts and WMMA sizes must be positive integers` from `validate_gemm_gfx1250_config` (FlyDSL TDM HGEMM). Three of four DP ranks raise at the first GSM8K batch and the engine stops; reproduced 2 of 2 server runs. The logged values: `M=0, N=129280, K=5120`, tile 16x256x128, 2 stages.

## Logs
[error.log](error.log)

## What we checked
- One GPU, `M=0`: aiter `2989e550f` OK; `9da592818` fails; `9da592818` with ROCm/aiter#8031 reverted OK. #8031 moved the gfx1250 BF16 rows of this shape to the TDM kernel.
- Fix: return the empty result before the tuned lookup in `aiter/tuned_gemm.py:_gemm_a16w16`:
  ```python
  if m == 0:
      return _into(out, inp_view.new_empty((*A.shape[:-1], n), dtype=otype))
  ```
  With it mounted into this image: GSM8K completes and MLPerf Endpoint c64 reaches 2270.56 tok/s/GPU, 10962/10962 samples successful (verification only, not in perf.csv). An aiter PR with this change and an `op_tests/test_gemm_a16w16.py -t zero` case is prepared, not opened yet.
- PRs searched (ROCm/aiter): `hgemm M=0`, `TDM empty`, `tdm hgemm`, `zero rows gemm`; none about this.
