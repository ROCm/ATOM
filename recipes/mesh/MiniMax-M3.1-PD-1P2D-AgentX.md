# MiniMax-M3.1-Flash P/D on two nodes: 1P2D AgentX, current best point (2026-11-26)

This is our best measured AgentX point so far: one PP4 x TP1 prefill and two TP4 decodes on two MI355X nodes, with one Mooncake Store shared by both nodes, at concurrency 240. Over a full CI hour it gives 125,717 tok/s/GPU at P90 TPOT 6.55 ms; the InferenceX vLLM MI355X bar is 112,932 at 8.64 ms. The single-node 1P1D recipe in [MiniMax-M3.1-PD-PP4-MooncakeStore.md](MiniMax-M3.1-PD-PP4-MooncakeStore.md) is unchanged.

The best point needs code that is not on main yet. It therefore runs from two reproduction branches and their own launch path:

| | |
|---|---|
| ATOM | ROCm/ATOM [`Jasen/m31-agentx-repro-v2`](https://github.com/ROCm/ATOM/tree/Jasen/m31-agentx-repro-v2): the tested build `76b83860b`, the recipe directory `recipes/m31_agentx_1p2d/`, and three code commits: Store R1 `198bbb854`, O1 `bda43bdfc` and the NIC fix `d0a2d9d8b`. The hour runs `d0a2d9d8b`. |
| aiter | ROCm/aiter [`Jasen/m31-agentx-repro`](https://github.com/ROCm/aiter/tree/Jasen/m31-agentx-repro) (`c2f06e01`), unchanged, baked into the image |
| Image | `rocm/atom-dev:m31-agx-local-c2f06e01`: the team share's tarball, or rebuild it from aiter `c2f06e01` |
| Model | `amd/MiniMax-M3.1-Flash-Quark-MXFP4`, pinned to revision `1442d09d`. HF main has moved since, and its chat template changes the prompts. |
| Full steps | [`recipes/m31_agentx_1p2d/README.md`](https://github.com/ROCm/ATOM/blob/Jasen/m31-agentx-repro-v2/recipes/m31_agentx_1p2d/README.md) on the ATOM branch |

## Topology and concurrency

- Two MI355X nodes, 12 GPUs in use:
  - **Node A:** prefill PP4 x TP1 on GPUs 0-3 (rdma0-3), with one send process per stage; decode D0 TP4 on GPUs 4-7 (rdma4-7); the Store masters and the router.
  - **Node B:** decode D1 TP4 on GPUs 4-7 (rdma4-7). GPUs 0-3 stay idle.
- **Store:** one Mooncake Store for both nodes, 4 per-NIC pools, with owners on both NUMA nodes of both nodes.
- **Load:** AgentX, i.e. an aiperf agentic replay of `semianalysis_cc_traces_weka_062126`, at **concurrency 240**.
  - Warmup: 10 requests per lane.
  - DSpark k=7, with the InferenceX golden acceptance length 4.04 forced on both roles.

## Best configuration (the hour)

06d's configuration plus three prefill code changes and a larger Store:

- **Prefill:**
  - Chunked P->D transfer **off**.
  - `--max-num-seqs 24`. Without a cap, Store-load requests crowd out the prefill's prefix cache when chunked transfer is off.
  - `OFFLOAD_LOAD_WORKERS=2`.
  - **New, Store R1** (`ATOM_STORE_LOAD_PIPELINE=1`): a Store load thread starts the next window's get while the last one unpacks, instead of waiting for the unpack.
  - **New, O1 send process** (`ATOM_PD_SEND_PROCESS=1`): each PP stage sends its KV to the decodes from its own process, so the stage's main thread no longer competes for the GIL with its 16 send threads. A version for main is in review as ROCm/ATOM#3864 -> #3865 -> #3866 (drafts).
  - **New, P/D NIC from PCI topology** (always on, no switch): each stage registers on the NIC behind its GPU's PCIe switch, not `rdma<HIP index>`. HIP numbers the GPUs in KFD order, not PCI order, so before the fix pp0, pp1 and pp3 sent through another GPU's NIC. On g52 the stages now use rdma3, rdma0, rdma2 and rdma1.
  - `VLLM_PP_LAYER_PARTITION=16,16,16,12`, MNBT 8192.
  - NVFP4 KV and NVFP4 index cache.
- **Decode:** unchanged from 06d.
  - TP4 with the C17 decode kernels.
  - amd-kernels MoE (`ATOM_M3_AK_MOE=1`): a4w4 at every batch size, the same MoE numerics as the InferenceX vLLM MI355X image.
  - lm_head AK + ring.
  - `pd_admission` fraction 0.15, reserve 512.
- **Router:** atomesh `cache_aware`, unchanged.
- **Store:** 3,392 GiB, 848 GiB per pool (06a and 06d: 3,200 GiB, 800 per pool).
  - The size varies per run. Each owner needs 2 MiB pages bound to its NUMA node, so the plan follows each NUMA node's free 2 MiB blocks after the host-memory prep. Those depend on other users' memory and on fragmentation: 3,776 GiB did not fit the hour's nodes.
  - Leave room on node A NUMA0 for the four send processes, 3.5-3.7 GiB each.
- The three code changes do not touch numerics: they change how the KV moves and which NIC carries it.

## Results

| | Hour (best) | 06d (previous best) | 06a (chunked on, AK MoE off) | vLLM MI355X agg TP4 C64 (InferenceX) |
|---|---:|---:|---:|---:|
| Window | 1 hour | 25-min screen | 1 hour | 1 hour |
| Total tok/s/GPU | 125,717 (126,320 tail-free) | 97,355 (98,152 tail-free) | 111,887 (112,285 tail-free) | 112,932 |
| req/s/GPU | 0.842 | 0.768 | 0.764 | 0.780 |
| P90 TPOT | 6.55 ms | 6.45 ms | 6.51 ms | 8.64 ms |
| TTFT p50 / p90 | 0.81 / 4.19 s | 4.4 / 17.9 s | 2.0 / 15.0 s | 0.48 (mean) / 0.88 s |
| Steady part | min 20-60: 133,240 (134,230 tail-free) | min 20-25: 120,707 (126,619 tail-free) | min 20-60: 126,873 (127,627 tail-free) | -- |
| Completed requests | 36,641 | 14,046 | 33,214 | -- |

- **Against the vLLM bar:** +11.3% tok/s/GPU at a lower P90 TPOT. vLLM keeps the lower TTFT. Our requests that start outside the two arrival waves (minutes 5-10, and a smaller one at minutes 55-60) have TTFT p90 2.3 s.
- **Against 06a, the previous hour:** +12.4% at about the same P90 TPOT, and TTFT p90 4.2 s instead of 15.0 s. The start-up wave clears by minute 9 instead of minute 20.
- **Against 06d:** every AgentX run starts with ~5 trace-idle minutes and then an arrival wave of ~200 lanes, so a 25-min screen reads lower than an hour. The hour's own first 25 minutes give 108,960 tok/s/GPU (113,679 tail-free), +12% (+16% tail-free) over 06d.
  - R1 and O1 carry most of that: a 25-min screen of 06d plus R1 and O1, on other nodes, gave 111,739 tail-free.
  - The NIC fix adds ~1-2% over the first 25 minutes, all in the start-up wave.
- 0 errors in 36,641 requests; no P/D landing drops, Store failures or NIC errors.
- **Next steps:**
  - **C280.** P90 TPOT 6.55 ms leaves room to the bar's 8.64 ms, but C280 still collapses without more P->D send capacity. In the C280 screen S2 (R1 and O1, before the NIC fix), pp3's send process saturated at ~17 cores, P->D stayed flat at ~57 GB/s and the prefill queue never drained: 82K tok/s/GPU tail-free over the first 25 minutes. The NIC fix lowers pp0's and pp3's send CPU per GB by ~40%, to the other stages' level. C280 with it is the next screen.
  - **1P3D** waits for its gate: the prefill computing >= 95K tok/s under load. It peaks at ~74K today, and the two decodes are not the limit (batch ~26 each). One prefill feeding 16 GPUs instead of 12 would lower tok/s/GPU.

## Run it

Check out the ATOM branch, set `RECIPE=<checkout>/recipes/m31_agentx_1p2d` and follow its README for:
- the image;
- the model revision;
- host prerequisites, the Store host-memory prep and the Store owners for your nodes;
- how the numbers are computed;
- how to check that your containers got the same environment.

The hour's launch, from the driver host, with node A first:

```bash
setsid nohup bash $RECIPE/harness/bin/agx_run.sh \
  --tag p2d-c240-hour-o1-nic --commit d0a2d9d8b --topo 1p2d --nodes <nodeA>,<nodeB> --lease <id> \
  --conc 240 --duration 3600 --warmup 10 --image rocm/atom-dev:m31-agx-local-c2f06e01 \
  --p-env $RECIPE/env/p_env_nochunk_lw2_r1_o1.sh --p-seqs 24 \
  --d-env $RECIPE/env/d_env_akmoe_lmhead.sh \
  --overlay $RECIPE/overlays/tmpdir.sh \
  --overlay $RECIPE/overlays/store_3392_hour.sh \
  --store-caps 1024/768/1152 --keep-model-cache --label FULLSTACK-NOCHUNK-AK-LMH-CAP24-LW2-R1-O1-NICLOCAL-1H \
  > $AGX_RUNS/p2d-c240-hour-o1-nic.out 2>&1 < /dev/null &
```

- The hour ran with node A g52 and node B g27.
- `--commit d0a2d9d8b`, the NIC-fix commit, gives the server the hour's code: `76b83860b` plus the three changes. `p_env_nochunk_lw2_r1_o1.sh` turns on R1 and O1.
- `store_3392_hour.sh` holds the hour's Store owners. It fits only nodes with at least the hour's free 2 MiB blocks. Elsewhere, size the owners from probes taken after the prep, or leave the overlay out to take the harness's automatic plan.
- The NIC fix needs one ACTIVE NIC behind each prefill GPU's PCIe switch; without it the prefill refuses to start.
- The run takes about 1 h 50 min. Hold both nodes at least 2 h 10 min past the launch.
- Count it as reproduced when the CI tok/s/GPU and minutes 20-60 land within 5% of the table, with the README's health lines.
- 06d and 06a run from the same branch with `--commit 76b83860b`, because the NIC fix has no switch. Their commands are in the README.
