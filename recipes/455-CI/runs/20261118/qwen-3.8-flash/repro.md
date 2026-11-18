# Qwen-3.8 Flash: GPU page fault at the first multi-request prefill batch on gfx1250 (455-CI 2026-11-18)

- Written: 2026-11-18 (Beijing time), 455-CI 2026-11-18 session.
- Status as of 2026-11-18: open, no fix PR. Owner: mengdie (minmengdie).

## Environment
- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261118` (digest `sha256:d17a28d3ea228c67c25fd1d15ae7138162287a7002ba13e2bae3ee33ed14cb69`)
- ATOM `main@9b2e67b7f6f5841b0d32db9589e989ff52673761`, aiter `main@b1d849ff36907402d7ea6a3483e6f0dafa995aca`. Both include ROCm/ATOM#3225 and ROCm/aiter#7847, the Qwen3.8 gfx1250 fixes and tunings.
- Last good: none on main. On 2026-11-14 (before #3225) the server failed earlier, in warmup (BF16 MTP head MoE).
- Machine: e15-07, one MI455X (TP1, `HIP_VISIBLE_DEVICES=0`); amdgpu `gpu_recovery=0 halt_if_hws_hang=1 noretry=1`.
- Weights: `amd/Qwen3.8-Flash-Next-Quark-MXFP4` @ `b71e28afcd4b43add751159862235035613a58ab`.

## Steps
1. Start the test container (weights mounted at `/models`):
   ```bash
   docker run -d --name q38 --network host --ipc host --shm-size 128g --device /dev/kfd --device /dev/dri \
     --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
     -v <dir with Qwen3.8-Flash-Next-Quark-MXFP4>:/models:ro rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261118
   ```
2. Start the server: [`server_cmd.sh`](server_cmd.sh). This is the 455-CI recipe's Qwen-3.8 Flash GSM8K server: TP1, MTP with 3 draft tokens and the MTP experts quantized online to MXFP4, FP8 KV, BF16 index cache, block 16, prefix caching, `--max-num-batched-tokens 16384`, without `--spec-decode-acceptance-length`. It comes up after about 441 s.
3. Run GSM8K: [`gsm8k_cmd.sh`](gsm8k_cmd.sh) (lm_eval `local-completions`, 3-shot, 65 concurrent requests).

## Expected vs actual
- Expected: GSM8K completes.
- Actual: the first prefill batch (1 request, ~1,000 tokens) runs. The first multi-request batch, 18 requests and ~16,350 tokens (`Scheduled prefill batch: 18 reqs, 16331 new tokens`), faults the GPU with a page fault at a small address:
  - **CI run (19:46 BJ):** `Memory Fault Error [... GPU index: 0, faulting addr: 0x4e2000, kernel: Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x16x32_MI16x16x1_...]`, i.e. a hipBLASLt BF16 GEMM. ModelRunner0 dies (exitcode -6). See [`ci_run_fault.txt`](ci_run_fault.txt).
  - **Rerun with `AITER_LOG_LEVEL=INFO` (20:27 BJ):** same point (`18 reqs, 16364 new tokens`, then `19 reqs, 16349`). The kernel log shows `no-retry page fault ... in page starting at address 0x0000000000cc1000` (GC_UTCL2, TCP client, PERMISSION_FAULTS 0x3), then `MES(0, 0) failed to respond to msg=REMOVE_QUEUE`. The engine stops producing output, the processes become defunct, and the GPU stays wedged (with `gpu_recovery=0`) until the node is rebooted. See [`debug_run_hang.txt`](debug_run_hang.txt).
- Reproduced 2 of 2 runs at the first multi-request prefill.

## What we checked
- PRs searched: "Qwen3.8 memory fault", "qwen4_exp gfx1250", "Qwen3.8 MI455" and "hipblaslt bias gfx1250" in ROCm/ATOM and ROCm/aiter. Nothing about this; ROCm/ATOM#3225 and ROCm/aiter#7847 (both merged) are the bring-up PRs.
- **Not the #7847 tuned table.** The 74 hipBLASLt rows of `qwen38_flash_next_bf16_tuned_gemm.csv` are gfx942 (ISA942, 80 CUs). The 206 gfx1250 rows are opus, FlyDSL, torch and Triton, so the faulting kernel is a hipBLASLt default (heuristic) pick, not a tuned row.
- **Not the GEMM in isolation.** The fault kernel launched 81,680 workgroups of a 32x16 tile, so M x N is about 41.8M, i.e. N ~ 2560 at M ~ 16331. On one gfx1250 GPU with this image, each of these runs cleanly:
  - `F.linear` at M=16331, N=2560, K in {256, 320, 512, 640, 2560, 4096, 6144, 10240}, with and without bias;
  - `torch` and `aiter tgemm.mm` at M=16331 for N in {10240, 13312, 16480}, K=2560 (the shapes whose tuned rows stop at M=8192).
- Both faulting addresses are small (0x4e2000, 0xcc1000). For a BF16 output with N=2560, 0x4e2000 is exactly row 1000 from a zero base, which suggests a null or unallocated buffer on the large-batch prefill path rather than a bad GEMM kernel.
- Candidate (not proven): `HyperConnection._mix_fused` (`atom/model_ops/qwen4_exp/hyperconnection.py`) runs `tgemm.mm(normed, fused_w_cat)` then `tgemm.mm(scaled_silu(d[:, :rank]), w_up)` for every batch above `HC_ROWS_MAX = 2`. The 1,000-token prefill uses the same path and passes, so the trigger is more likely something that only the multi-request batch uses, for example the QSA prefill attention (the log says it falls back to the native kernel: KV cache not BF16, and a per-rank head shape of 24 q / 2 kv heads x 256), the MTP drafter's prefill, or the PLE layer.
- Next steps for the owner: run with `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1` and an ATOM per-op trace on a 2-request prefill of ~16k tokens, to name the op whose input or output pointer is null. Use a node you can reboot: the fault can wedge MES.
