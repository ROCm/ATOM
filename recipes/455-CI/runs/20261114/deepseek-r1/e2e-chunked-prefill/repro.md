# DeepSeek R1: GPU memory fault in `_triton_gather_kv_b_proj` with chunked prefill at concurrency 64 (455-CI 2026-11-14)

## Environment
- Image: `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114` (digest `sha256:9c38aef6ad030885b5309ad90b30ff778da4f764161aecdf7b7c7d8ae3ea1eb9`)
- aiter `main@b6f1ff6b7` (as in the image). ATOM: `main@ed372e0ab` + ROCm/ATOM#3345 (`626b2d235`), mounted over `/app/ATOM`. #3345 only changes how history tool-call arguments reach the chat template; without it most agentic turns fail with HTTP 500 before reaching this load.
- Machine: e15-06, 4 x MI455X (gfx1250); amdgpu `gpu_recovery=0 halt_if_hws_hang=1`.
- Weights: `amd/Deepseek-S3_sq_a05_v2_mlperf6_1` @ `1415311f5a898bc5fb0579612edacf4d4eb75bed` (`kv_b_proj` unquantized).
- Last good: the same server at concurrency 4 (127/127 samples, contexts up to 33k tokens) and GSM8K, minutes earlier.

## Steps
1. Start the test container:
   ```bash
   docker run -d --name r1-repro --network host --ipc host --shm-size 128g \
     --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE \
     --security-opt seccomp=unconfined -v <model dir>:/models:ro \
     -v <ATOM with #3345>:/app/ATOM rocm/fw-bringup-mlperf:gfx1250-atom-ci455-20261114
   ```
2. Start the server: [`server_cmd.sh`](server_cmd.sh). It is the 455-CI recipe's DeepSeek R1 server (DP4 attention + EP4 MoE, FP8 KV, block 64, `ATOM_MLA_KV_SHUFFLED=1`, `--gpu-memory-utilization 0.975`, no prefix caching) with one change: `--no-enable-chunked-prefill` removed, so chunked prefill is on with the default `max_num_batched_tokens=16384`.
3. Run the client: MLPerf Endpoints (`mlcommons/endpoints` @ `4b385eb`) agentic bench at concurrency 64, [`bench_cmd.sh`](bench_cmd.sh) with `CONC=64`. The fault happens in the warmup phase, config [`warmup.yaml`](warmup.yaml): 64 trajectories, prompts of ~26.6k tokens each.

## Expected vs actual
- Expected: the server serves the c64 warmup, as it served c4.
- Actual: about 1 s after the c64 warmup starts, rank DP2 (GPU index 2) faults and the server shuts down:
  ```
  :0:rocdevice.cpp :3896: Memory Fault Error [host: ctheliosr-rck-g02-e15-06, GPU index: 2, faulting addr: 0x6fcf17598000, kernel: _triton_gather_kv_b_proj]
  Memory access fault by GPU node-4 (Agent handle: 0x139b3a20) on address 0x6fcf17598000. Reason: Page not present or supervisor privilege.
  AsyncIOProcManager(ModelRunner): [ModelRunner0/1] proc died unexpectedly (exitcode=-6), shutting down.
  ```
- Reproduced 1 of 1 runs. The node was healthy afterwards (4 GPUs, no KFD processes left, no MES errors in dmesg).

## Logs
- [`fault_excerpt.txt`](fault_excerpt.txt): the fault and the hang analysis (dispatch: workgroup 128, grid 384x1x1).
- [`server_log_before_fault.txt`](server_log_before_fault.txt): the 60 server log lines before the fault, with the prefill batches the scheduler issued.

## What we checked
- PRs searched: `gather_kv_b_proj`, `shuffle_kv_block`, `kv_shuffled chunked prefill`, `MLA chunked prefill memory fault` in ROCm/ATOM and ROCm/aiter. None is about this. ROCm/aiter#7952 (open) fixes 32-bit offset overflow in a different MLA kernel (`fused_qk_rope_cat_and_cache_mla`).
- Code path: chunked prefill with context goes through `MLAAttention._gather_cached_kv_b_proj` (`atom/model_ops/attention_mla.py`). With `ATOM_MLA_KV_SHUFFLED=1` it calls aiter `gather_kv_b_proj(..., shuffled_kv_cache=True)` with the block-granular `shuffle_kv_block_indptr` / `shuffle_kv_block_indices` built in `atom/model_ops/attentions/aiter_mla.py` (around line 2346).
- 32-bit overflow on the KV cache offset looks unlikely: in aiter `b6f1ff6b7`, both `_triton_gather_kv_b_proj_impl` and `_triton_gather_kv_b_proj_fp4_impl` multiply the block index by an int64 `stride_k_buffer`.
- What differs from the runs that worked (from the scheduler log; not verified):
  - Before the fault, 668 prefill batches ran, 298 of them with context. That includes single-request batches with up to 33,152 context tokens (c4) and 3 batches mixing a request with context and fresh requests, but with small contexts (192-576 tokens).
  - The fault comes right after the first batches that mix a request with a large context and fresh requests with none, e.g. `3 reqs, done: [16384, 0, 0], new: [10237, 939, 5184]` and `2 reqs, done: [14272, 0], new: [12312, 4032]`.
  - Candidates: the shuffled-KV block indptr/indices or the context cumsum (`cu_seqlens_k`) for such batches, or the sizing of the `k_out` / `v_out` buffers.
- Not tried yet: `ATOM_MLA_KV_SHUFFLED=0`, a larger `--max-num-batched-tokens`, or a lower `--gpu-memory-utilization`.
