# DeepSeek-V4-Pro: MegaMoE stage-1 (S1F) GEMM1 asserts `slot_width <= 128` in warmup after ROCm/aiter#5899 (455-CI 2026-11-14)

- Written: 2026-11-14 (Beijing time), 455-CI 2026-11-14 session.
- Status as of 2026-11-18: fixed on main by [ROCm/aiter#7973](https://github.com/ROCm/aiter/pull/7973) (merged 2026-11-16), which gives the S1F 256x256 tile a 2x2 wave layout. Verified in the 2026-11-18 session: server up, no assert, GSM8K 0.9507.

## Environment
- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114` (digest `sha256:9c38aef6ad030885b5309ad90b30ff778da4f764161aecdf7b7c7d8ae3ea1eb9`)
- ATOM `main@9fec99786b088168aec7729858c1176e99d818e6`, aiter `main@b6f1ff6b7f3754aca8cc8c403b5fcd7a258eba47`. That aiter contains ROCm/aiter#5899 (`56121ed0b`, merged 2026-11-13 20:27 +0800) and the earlier V4-Pro fixes ROCm/aiter#7920 + #7924.
- Last good: 2026-11-13, aiter `main@40565597` + #7920 + #7924, which is before #5899. Results: GSM8K 0.9522, 8K/1K c4 271.24 / c64 6,581.46 tok/s/GPU ([LLM-Booster#490](https://github.com/wuhuikx/LLM-Booster/issues/490)).
- Machine: e15-07 (CI run) and e15-06 (revert check), 4 x MI455X each. Weights: `deepseek-ai/DeepSeek-V4-Pro` @ `b5968e9190ef611bbf34a7229255be88a0e937c1`.

## Steps
1. Start the test container (model dir mounted at `/models/DeepSeek-V4-Pro`):
   ```bash
   docker run -d --name v4pro --network host --ipc host --shm-size 128g --device /dev/kfd --device /dev/dri \
     --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
     -v <DeepSeek-V4-Pro dir>:/models/DeepSeek-V4-Pro:ro rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114
   ```
2. Start the server: the 455-CI recipe's DeepSeek-V4-Pro command, [`../server_cmd.sh`](../server_cmd.sh). It runs DP4 attention + EP4 MegaMoE with `ATOM_MEGA_STAGE1_FUSED=1 ATOM_MEGA_STAGE2_FUSED=1`, MXFP4 wire, and `ATOM_MOE_GU_ITLV=1`.
3. No client is needed: the server dies in model warmup, about 13 minutes after launch (782 s).

## Expected vs actual
- Expected: the server comes up, as it did on 2026-11-13 before #5899.
- Actual: all 4 ranks fail in the first warmup forward while compiling the S1F stage-1 GEMM1 (full excerpt in [`../error.log`](../error.log)):
  ```
  File ".../mega_moe_gfx1250/s1f_stage1.py", line 1990, in __while_after_36   (g1_task)
  File ".../mxfp4_preshuffle_gfx1250_tdm.py", line 1579, in gemm_task
  File ".../mxfp4_preshuffle_gfx1250_tdm.py", line 1560, in make_rmem_slot
      assert slot_width <= 128
  AssertionError
  ```
- Reproduced 1 of 1 runs on main.
- **Same image with only #5899 reverted** (`git -C /app/aiter revert --no-commit 56121ed0b`, 14 files): the server is ready after 682 s, and GSM8K 3-shot gives 0.9477 flexible / 0.9484 strict on e15-06. #5899 is therefore the regression.

## What we checked
- PRs searched: `slot_width`, `make_rmem_slot`, `s1f_stage1`, `mxfp4_preshuffle_gfx1250_tdm`, `mega_moe gfx1250 stage1 assert` in ROCm/aiter and ROCm/ATOM. None is about this.
- Why it asserts: S1F builds GEMM1 with `m_warp=1, n_warp=4` (`s1f_stage1.py`, `build_gemm_a8w4_tdm_task`). With FP4 A (`ACT_NDW=8`) and `WMMA_VECTOR_DWORDS=8`, the explicit-VGPR-partition slot is `wmma_m_rep*8 + wmma_n_rep*8 = tile_m/2 + tile_n/8` dwords. That stays within 128 for 64x256 (64) but not for `tile_m >= 256` at `tile_n = 256` (160).
- What #5899 changed in the gate (reading the code, not instrumented):
  - Before #5899, `s1f_ctl._gemm1_reasons` kept S1F off for a (align_m, recv_bound) key until GEMM1 had run once ("probing the GEMM1 configuration") and recorded why its config did not fit.
  - #5899 removes the probe. GEMM1's tile is now resolved per step (`_compact_gemm_tiles`), and `step_reasons` only checks `g1_tile_m == align_m`.
  - V4-Pro is EP4, h7168, topk 9 (topk 8 plus the fused shared expert), the same geometry as the new `_DISPATCH_COMPACT_EP4_TOPK9` table. With the probe gone, a step with a large `tile_m` (inferred: 256) now enters S1F and hits the slot limit.
  - `_S1F_MAX_RECV` for (4, 9) is still 4096, so the receive cap did not change.
- Not tried: `ATOM_MEGA_STAGE1_FUSED=0` (it should avoid S1F), or a `tile_reasons` check for `tile_m/2 + tile_n/8 > 128`. Both would hide the question of which tile V4-Pro should get, so the fix is left to the S1F owner.
