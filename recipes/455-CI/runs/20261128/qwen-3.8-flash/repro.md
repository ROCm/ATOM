# Qwen-3.8 Flash: 8K/1K throughput down 17-20% since 2026-11-21 (455-CI 2026-11-28)

## Environment
- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261128` (digest `sha256:58d7808f5fbfdf020616ebfc421327e095e839d2b59d1b296861ff09dca8d2fd`), ROCm 10.2.0 base
- ATOM `main@380aeb8a0`, aiter `main@9da592818`; last good session: 2026-11-21, ATOM `cccc96a03`, aiter `df3e0b2f6` (c4 8359.97, c64 35214.47 tok/s/GPU)
- Machine: one MI455X GPU (TP1); weights: `Qwen3.8-Flash-Next-Quark-MXFP4`

## Steps
1. Start the test container with the ROCm 10.2 base's venv on `PATH` (as in [../deepseek-v4-pro/repro.md](../deepseek-v4-pro/repro.md)).
2. Start the server with the 455-CI recipe's Qwen-3.8 Flash command, as run: [server_cmd_perf.sh](server_cmd_perf.sh) (MTP, `--spec-decode-acceptance-length 2.32`).
3. Run the recipe's 8K/1K bench at concurrency 4 and 64: [bench_cmd.sh](bench_cmd.sh).

## Expected vs actual
- Expected: c4 about 8400-8900, c64 about 35000-37000 tok/s/GPU.
- Actual: c4 6683.5, c64 29197.2 (-20%, -17%). ITL p50 is unchanged (7.44 vs 7.48 ms at c4) while TPOT mean grows (4.43 vs 3.76 ms); the acceptance length is fixed by the flag, so the slowdown is per step outside the measured ITL.

## What we checked
All points below are c4 tok/s/GPU on the same ROCm 10.2 b1123 base (e15-06 and e15-07 agree within 1%: the same image gave 8483.1 and 8407.2).

| ATOM | aiter | c4 |
|---|---|---|
| `cccc96a03` (2026-11-21) | `df3e0b2f6` (2026-11-21) | 8928.8 (c64 37052.5) |
| `cccc96a03` | `9da592818` | 8483.1 |
| `380aeb8a0` | `df3e0b2f6` | 7497.8 |
| `380aeb8a0` | `9da592818` (this session) | 6683.5 (c64 29197.2) |

So the base image is not the cause; ATOM carries most of the drop and aiter a smaller part. Bisect over ATOM's 100 first-parent commits `cccc96a03..380aeb8a0` (aiter `9da592818`):

| ATOM commit (index in the range) | c4 |
|---|---|
| #3 `15708c6c7` | 8599.2 |
| #7 `be13d1cf6` | 8431.5 |
| #8 `032441916` | 8594.7 |
| #9 `9f98bffc8` | 8448.0 |
| #10 `93f31c640` | 8638.0 |
| **#11 `811b02bb4`** ROCm/ATOM#2636 | **7334.2** |
| #22, #33, #66, #77, #88 | 7042.2, 7343.6, 7377.0, 7545.6, 7156.4 |
| #92, #93, #94, #95 | 7353.9, 7351.5, 7079.1, 7355.9 |
| #96 `c910eb235` | 6994.4 |
| #100 `380aeb8a0` | 6683.5 |

- First bad: ROCm/ATOM#2636 "[Qwen 3.8] LMCache offload for QSA + GDN + PLE state" (merged 2026-11-23), -15% at c4. The recipe does not configure KV transfer, so the PR's PAGE checkpoints stay off (`_wants_paged_checkpoints` returns False); the cost comes from its other changes on the decode path (`atom/model_ops/attentions/qwen4_exp_attn.py`, `qwen4_exp/qsa_attention.py`). Not profiled yet.
- A second, smaller drop (about 7350 -> 6683) is not pinned: single c4 runs vary by about ±3%. A bisect with a 7030 threshold stopped at #96 (ROCm/ATOM#3838), but that commit only touches the DeepSeek-V4.1 indexer and `atom/model_ops/indexer/`, which Qwen-3.8 does not import, so it is noise. Most points after #2636 sit near 7350 and the later ones are lower, so the drop is more likely in #96-#100; #97 (ROCm/ATOM#3672, an idle EngineCore waits for input) is a candidate, untested.
- PRs searched (ROCm/ATOM, open and merged): `2636`, `Qwen3.8 perf regression`, `qwen4_exp decode`, `qwen4_exp_attn`; none about this regression.
