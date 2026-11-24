# MiniMax-M3.1-Flash P/D on one node: PP4 prefill + TP4 or DEP4 decode with a Mooncake Store

One MI355X node runs 1P1D:
- **Prefill:** PP4 x TP1 on GPUs 0-3 (NUMA0, rdma0-3).
- **Decode:** 4 GPUs on GPUs 4-7 (NUMA1, rdma4-7).
- **Store:** a Mooncake Store on the node's DRAM, used as the prefill's prefix tier (ATOM's native `mooncake_store` offload, no LMCache).
- **Router:** atomesh.

Two decode variants:

| | Decode A: TP4 + indexer CP | Decode B: DEP4 |
| --- | --- | --- |
| Parallelism | `-tp 4`, `ATOM_M3_INDEXER_CP=1` | `-tp 4 --enable-dp-attention --enable-expert-parallel`: 4 DP ranks (TP1 attention each), MoE EP4 over MoRI |
| Index K | replicated on the 4 ranks | stored once, on the request's rank |
| MoE communication | 2 all-reduces per layer | dispatch + combine per MoE layer; `--moe-backend mega` fuses them with the experts (aiter MegaMoEV2) |
| Code | ATOM main | ATOM main + ROCm/ATOM#3563 + ROCm/ATOM#3564 (both open) |

> Status: written against ATOM main `a62134f99` (2026-11-24) from main's code and earlier M3.1 P/D + Store runs on experiment branches. The commands have not yet been run end to end on this main; please report what breaks.

## Numerics

- NVFP4 KV and NVFP4 index cache on both roles (`KV_DTYPE=nvfp4`, the default): the M3.1 production setting, as in the InferenceX vLLM MI355X recipe. `KV_DTYPE=fp8` switches both roles to FP8. Main carries what this needs: ROCm/ATOM#3471 and #3472 (NVFP4 over Mooncake P/D and the Store), and ROCm/aiter#8285 in the image (the NVFP4 decode scorer under indexer CP).
- Native MXFP8 linears. No `--online_quant_config ptpc_fp8`.
- Index top-k on every layer: no `index_topk_freq` override.
- Shared expert stays its own MXFP8 MLP: `ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=0`.
- No `AITER_QUICK_REDUCE_QUANTIZATION`: all-reduce stays bf16.
- DSpark draft as shipped: `--method dspark --draft-model $MODEL/dspark --num-speculative-tokens 7`.
- Benchmarks force the InferenceX golden acceptance length on both roles: `FORCE_AL=4.04` (DSpark k=7). Serving and accuracy runs leave it empty, i.e. real acceptance.

## Prerequisites

- A whole pit2-p03 node (gpool `-g 8`).
- Image `rocm/atom-dev:nightly_202611240650`, with this branch mounted over it at `/workspace/ATOM`.
- Model `/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4`; the DSpark draft is in its `dspark/` subdir.
- For decode B: merge the two open PRs into your tree first.

  ```bash
  git fetch origin pull/3563/head:pr-3563 pull/3564/head:pr-3564
  git merge --no-edit pr-3563 pr-3564
  ```

  ROCm/ATOM#3563 launches the sparse decode index only for requests that have rows. ROCm/ATOM#3564 gives DSpark block attention several KV heads per rank, which TP1 under DP needs.
- Free DRAM for the Store owners on each NUMA node. `store.sh` checks it (`numa_memory_budget.py --compact`) and refuses owners that do not fit.

## Steps

All scripts are in [`minimax_m31_pd_store/`](minimax_m31_pd_store/).

1. On the host, start three containers: `prefill` (GPUs 0-3), `decode` (GPUs 4-7) and `store` (no GPU). They are named `$PFX-<role>` (`PFX` defaults to `<user>-m31pd`). Run logs go to `RUNS` (default `./runs`).

   ```bash
   recipes/mesh/minimax_m31_pd_store/containers.sh [ATOM_TREE] [IMAGE]
   ```

2. Start the Store master and owners in the store container. `OWNERS` is `numa:GiB:nic;...`, on NICs the prefill stages do not use.

   ```bash
   docker exec -e OWNERS="0:128:rdma4;0:128:rdma5;1:128:rdma6;1:128:rdma7" "$PFX-store" \
     bash /workspace/ATOM/recipes/mesh/minimax_m31_pd_store/store.sh start
   ```

3. Start the prefill, then wait until it is ready.

   ```bash
   docker exec -d "$PFX-prefill" bash /workspace/ATOM/recipes/mesh/minimax_m31_pd_store/prefill.sh
   ```

4. Start the decode. `MODE=tp4` is decode A, `MODE=dep4` decode B, `MODE=mega` decode B with MegaMoE.

   ```bash
   docker exec -d -e MODE=dep4 "$PFX-decode" bash /workspace/ATOM/recipes/mesh/minimax_m31_pd_store/decode.sh
   ```

5. Start the router. `POLICY=random` lets ATOM place requests on decode DP ranks (`--dp-load-balance least_requests`). `POLICY=dp_sticky` keeps a session (`X-Session-ID`) on one rank, for multi-turn traffic.

   ```bash
   docker exec -d "$PFX-decode" bash /workspace/ATOM/recipes/mesh/minimax_m31_pd_store/router.sh
   ```

6. Check it:
   - `wait_ready.sh <port> <log>` for each server (prefill 8010, decode 8020).
   - `gsm8k.sh <label> 32 200` for a 200-question GSM8K through the router (20-shot chat, real acceptance).
   - `store.sh status` for Store capacity, keys and evictions.

## Benchmark point

Run 1P1D with AgentX traces at **C140** (the concurrency of the whole 1P1D), with `FORCE_AL=4.04` on both roles. Earlier 1P1D + Store runs on experiment branches peaked at C140, and C160 was 4% lower. Report the decode running batch and KV usage next to tok/s: at C140 a TP4 decode runs about 40-50 requests at a time.

## Settings and why

- Prefill:
  - `VLLM_PP_LAYER_PARTITION=16,16,16,12`: the last stage also holds the DSpark draft's KV.
  - `--max-num-batched-tokens 8192`: a TP1 step must stay at or below 16383 tokens.
  - `--enforce-eager`.
  - Chunked P->D transfer, with `chunked_admission_timeout_s` 1500.
- Store:
  - Owners use transparent huge pages (`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`): 4 KiB pages exhaust the NIC's registration budget.
  - Eviction watermark 0.88 / 0.02.
  - The page-cache dropper only touches the model directory.
- Decode consumer:
  - `ib_device` rdma4-7.
  - `pd_admission` with `max_pending_kv_fraction` 0.15.
  - TP4: `--max-num-seqs 256`, MNBT 16384, CUDA graphs up to 256.
  - DEP4: 64 requests per DP rank, because under DP plus a drafter the decode token graphs stop at `ATOM_TOKEN_GRAPH_DP_MAX_TOKENS` (512 rows = 64 requests x 8 verify rows). MNBT 4096, `MORI_SHMEM_HEAP_SIZE=8G`.
  - `MODE=mega` uses MNBT 512 so MegaMoE's combine wire stays bf16.
- Both servers share the host network. Each gets its own `ATOM_DP_MASTER_PORT` / `ATOM_DP_BASE_PORT`. RCCL runs without IB (`NCCL_IB_DISABLE=1`), and the P/D and Store clients use one QP per endpoint (`MC_NUM_QP_PER_EP=1`).

## Known limits

- Under DEP4 a request lives on one rank, so its context is capped by one rank's KV pool, not by the instance's.
- DEP4 replicates attention, shared/dense MLPs, embedding, `lm_head` and the draft on every rank: about 13 GB more weights per GPU than TP4.
- At C140 the decode is bound by KV capacity rather than step time: the running batch stays at about 40-50 requests while the rest wait for KV. Compare the two decodes at C140 on the same AgentX workload.
