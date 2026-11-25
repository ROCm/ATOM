# MiniMax-M3.1-Flash P/D on two nodes: 1P2D AgentX, current best point (2026-11-25)

This is our best measured AgentX point so far: one PP4 x TP1 prefill and two TP4 decodes on two MI355X nodes, with one Mooncake Store shared by both nodes, at concurrency 240. The single-node 1P1D recipe in [MiniMax-M3.1-PD-PP4-MooncakeStore.md](MiniMax-M3.1-PD-PP4-MooncakeStore.md) is unchanged.

The best point needs code that is not on main yet. It therefore runs from two reproduction branches and their own launch path:

| | |
|---|---|
| ATOM | ROCm/ATOM [`Jasen/m31-agentx-repro`](https://github.com/ROCm/ATOM/tree/Jasen/m31-agentx-repro): the tested build `76b83860b` plus one commit that only adds `recipes/m31_agentx_1p2d/` |
| aiter | ROCm/aiter [`Jasen/m31-agentx-repro`](https://github.com/ROCm/aiter/tree/Jasen/m31-agentx-repro) (`c2f06e01`), baked into the image |
| Image | `rocm/atom-dev:m31-agx-local-c2f06e01`: the team share's tarball, or rebuild it from aiter `c2f06e01` |
| Model | `amd/MiniMax-M3.1-Flash-Quark-MXFP4`, pinned to revision `1442d09d`. HF main has moved since, and its chat template changes the prompts. |
| Full steps | [`recipes/m31_agentx_1p2d/README.md`](https://github.com/ROCm/ATOM/blob/Jasen/m31-agentx-repro/recipes/m31_agentx_1p2d/README.md) on the ATOM branch |

## Topology and concurrency

- Two MI355X nodes, 12 GPUs in use:
  - **Node A:** prefill PP4 x TP1 on GPUs 0-3 (rdma0-3); decode D0 TP4 on GPUs 4-7 (rdma4-7).
  - **Node B:** decode D1 TP4 on GPUs 4-7 (rdma4-7); the Store owners.
- **Store:** one Mooncake Store for both nodes, 4 per-NIC pools x 800 GiB = 3,200 GiB.
- **Load:** AgentX, i.e. an aiperf agentic replay of `semianalysis_cc_traces_weka_062126`, at **concurrency 240**.
  - Warmup: 10 requests per lane.
  - DSpark k=7, with the InferenceX golden acceptance length 4.04 forced on both roles.

## Best configuration (06d)

- **Prefill:**
  - Chunked P->D transfer **off**.
  - `--max-num-seqs 24`. Without a cap, Store-load requests crowd out the prefill's prefix cache when chunked transfer is off.
  - `OFFLOAD_LOAD_WORKERS=2`.
  - `VLLM_PP_LAYER_PARTITION=16,16,16,12`, MNBT 8192.
  - NVFP4 KV and NVFP4 index cache.
- **Decode:**
  - TP4 with the C17 decode kernels.
  - amd-kernels MoE (`ATOM_M3_AK_MOE=1`): a4w4 at every batch size, the same MoE numerics as the InferenceX vLLM MI355X image.
  - lm_head AK + ring.
  - `pd_admission` fraction 0.15, reserve 512.
- **Router:** atomesh `cache_aware`.

## Results

| | 06d (best configuration) | 06a (chunked on, AK MoE off) | vLLM MI355X agg TP4 C64 (InferenceX) |
|---|---:|---:|---:|
| Window | 25-min screen | 1 hour | 1 hour |
| Total tok/s/GPU | 97,355 (98,152 tail-free) | 111,887 | 112,932 |
| req/s/GPU | 0.768 | 0.764 | 0.780 |
| P90 TPOT | 6.45 ms | 6.51 ms | 8.64 ms |
| TTFT p50 / p90 | 4.4 / 17.9 s | 2.0 / 15.0 s | 0.48 (mean) / 0.88 s |
| Steady part | min 20-25: 120,707 (126,619 tail-free) | min 20-60: 126,873 | -- |

- **Why the screen reads lower than the hour:** every AgentX run starts with ~5 trace-idle minutes and then an arrival wave of ~200 lanes, so a 25-min screen scores lower than a full hour.
  - Compare 06d with 06a's own first 25 minutes: 90,711 tok/s/GPU tail-free, which 06d beats by 8.2%.
  - 06d's configuration has not run a full hour yet. Splicing 06d's first 25 minutes with 06a's last 35 gives an estimate of ~115K tok/s/GPU.
- **Next steps:**
  - Raise concurrency toward X 116 (P90 TPOT 8.6 ms): C280-C340 on 1P2D.
  - Prefill fixes in progress: producer-side admission control for chunked-off, the P->D sender contention on the prefill host threads, and Store load pipelining. Today the prefill reaches about 55% of its kernel throughput in steady state.

## Run it

Check out the ATOM branch, set `RECIPE=<checkout>/recipes/m31_agentx_1p2d` and follow its README for:
- the image;
- the model revision;
- host prerequisites and the Store host-memory prep;
- how the numbers are computed;
- how to check that your containers got the same environment.

The 06d launch, from the driver host, with node A first:

```bash
setsid nohup bash $RECIPE/harness/bin/agx_run.sh \
  --tag p2d-c240-nochunk-cap24-lw2-screen --commit 76b83860b --topo 1p2d --nodes <nodeA>,<nodeB> --lease <id> \
  --conc 240 --duration 1500 --warmup 10 --image rocm/atom-dev:m31-agx-local-c2f06e01 \
  --p-env $RECIPE/env/p_env_nochunk_lw2.sh --p-seqs 24 \
  --d-env $RECIPE/env/d_env_akmoe_lmhead.sh \
  --overlay $RECIPE/overlays/tmpdir.sh \
  --overlay $RECIPE/overlays/store_3200_06c.sh \
  --keep-model-cache --label FULLSTACK-NOCHUNK-AK-LMH-CAP24-LW2 \
  > $AGX_RUNS/p2d-c240-nochunk-cap24-lw2-screen.out 2>&1 < /dev/null &
```

- `--duration 3600` runs the full hour.
- `--commit 76b83860b` keeps the server snapshot identical to what ran.
