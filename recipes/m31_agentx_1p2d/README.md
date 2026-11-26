# MiniMax-M3.1-Flash AgentX, 1P2D on two MI355X nodes

This directory reproduces our best MiniMax-M3.1 AgentX P/D results: one PP4 x TP1
prefill and two TP4 decodes with NVFP4 KV, a DSpark k=7 draft and one shared
Mooncake Store across both nodes. It runs ATOM's CI launcher
(`.github/scripts/atomesh`) through a small harness, so the numbers come out in
CI format.

The server code is exactly what ran. This branch is ATOM `76b83860b`, a commit
that adds this directory, three code commits (Store R1, O1 and the NIC fix) and
a commit that adds the hour point here. The hour point runs `d0a2d9d8b`, the
last code commit; 06a and 06d ran `76b83860b`. The kernels come from the aiter
branch `Jasen/m31-agentx-repro` on ROCm/aiter (`c2f06e01`), baked into the image
below, for all three points.

## The hour point (best)

`p2d-c240-hour-o1-nic` ran the full CI hour on 2026-11-26 (UTC), with node A g52 and node B g27. It is 06d's deployment plus three code changes that cut the prefill's P->D send and Store load costs, a 1-hour window and a larger Store. Concurrency 240, forced acceptance length 4.04, a warmup of 10 requests per lane and the replay are as for 06d.

| | **Hour** |
|---|---|
| Window | 1 hour (`--duration 3600`), 04:27:07-05:27:07; records span 3,628 s |
| CI total tok/s/GPU (12 GPUs) | **125,717** |
| req/s/GPU | 0.842 |
| P90 TPOT | 6.55 ms |
| TTFT p50 / p90 | 0.81 / 4.19 s |
| Completed requests, cache hit | 36,641 (0 errors), 96.6% |
| Tail-free tok/s/GPU | 126,320 (0.846 req/s/GPU) |
| Steady part | minutes 20-60: 133,240 tok/s/GPU (134,230 tail-free), 0.83 req/s/GPU, P90 TPOT 6.30 ms, TTFT 0.79 / 3.07 s |

Against 06a, the previous full hour (111,887): +12.4% at about the same P90 TPOT (6.55 vs 6.51 ms), and TTFT p90 4.2 s instead of 15.0 s. Its first 25 minutes give 108,960 tok/s/GPU over their records span (113,679 tail-free), against 06d's 97,355 (98,152).

It differs from 06d only in these. The container env differences show in `reference/06d/docker-rank-0.env` against `reference/hour/docker-rank-0.env`:

| Setting | 06d | Hour |
|---|---|---|
| ATOM code | `76b83860b` | `d0a2d9d8b` = `76b83860b` + R1 + O1 + NIC fix |
| Prefill `ATOM_STORE_LOAD_PIPELINE`, `ATOM_PD_SEND_PROCESS` | unset (0), unset (0) | 1, 1 |
| Store owners, GiB (node A NUMA0 + NUMA1, node B NUMA0 + NUMA1) | 4x240 + 4x176, 4x240 + 4x144 (3,200; 800 per pool) | 4x224 + 4x272, 4x144 + 4x208 (3,392; 848 per pool) |
| Window | 25 min | 1 hour |
| Launcher ports | +1000 | as main |
| Nodes A, B | g27, g23 | g52, g27 |

The rest is 06d's, including the lossy settings (see "Previous points"). None of the three changes the model's math: they move KV and pick NICs. No accuracy run covered them; the hour checked transport health only (0 errors, no landing drops, Store failures or NIC errors).

**Code.** Three commits on this branch, each one of the hour's `--patch` files applied unchanged, in the hour's order:

| Commit | What it does | Switch (prefill) | On main |
|---|---|---|---|
| `198bbb854` Store R1 | A Store load thread queues each window's unpack behind an event and starts the next get at once, instead of waiting for the unpack first | `ATOM_STORE_LOAD_PIPELINE=1` | a later revision with tests, branch `Jasen/m31-store-load-pipeline` (`e2b5d15d6`), not on GitHub as of 2026-11-26 |
| `bda43bdfc` O1 | Each prefill stage sends its KV from a process of its own, `pd-send-ppK`, so the stage's main thread no longer waits on the GIL behind its 16 send threads | `ATOM_PD_SEND_PROCESS=1` | ROCm/ATOM#3864 -> #3865 -> #3866 (drafts). They lack one part of the code that ran: a transfer that failed before it posted any RDMA write is reported as unwritten (`posted`) |
| `d0a2d9d8b` NIC fix | A P/D rank registers on the NIC behind its GPU's PCIe switch, not `rdma<HIP index>` | none, always on; an explicit `ib_device` wins | ROCm/ATOM#3884 (draft), the same changed lines |

The tree of `d0a2d9d8b` is the hour's snapshot (`76b83860b` + the three patches) plus this directory, which neither the launcher nor the server reads. None of the three touches `.github/`, so the launcher is still 76b83860b's own (see Code). aiter is unchanged: ROCm/aiter `Jasen/m31-agentx-repro` (`c2f06e01`), baked into `rocm/atom-dev:m31-agx-local-c2f06e01`; load or build it as in Image.

**Before the run.** The host prerequisites, model, dataset and image are unchanged from 06d: follow "Before a run" steps 1-5. The NIC fix also needs one ACTIVE NIC behind each prefill GPU's PCIe switch, as on these nodes; without it the prefill refuses to start and names `ib_device` / `ATOM_MOONCAKE_IB_DEVICE`. Then the Store, which took about 45 min before the launch (03:01-03:46 UTC).

*Host memory:* three rounds of `prep/prep_hour.sh`. A round runs memory pressure (`evict_cache.py`) and then compaction (the launcher's `compact_huge_pages.py`) on each NUMA node it is given, all in parallel, then probes both nodes with `node_probe.py --drop`, as the harness does, and plans the Store owners (below). The hour's rounds:
```bash
export EXTRA_RESERVE=g27:0:400   # the planner's margin for other users' jobs on g27 NUMA0 (see Store owners)
# 1. All four NUMA nodes: pressure to a 96 GiB floor, then compaction (420 s cap).
bash $RECIPE/prep/prep_hour.sh g52 g27
# 2. Node A: pressure to a 24 GiB floor, then compaction; node B: compaction only (600 s cap).
FLOOR_A=24 FLOOR_B=- TA1=1300 COMPACT_TIMEOUT=600 bash $RECIPE/prep/prep_hour.sh g52 g27
# 3. Node A: compaction only (300 s cap).
FLOOR_A=- FLOOR_B=- TA0=1370 TA1=1350 TB0=- TB1=- COMPACT_TIMEOUT=300 bash $RECIPE/prep/prep_hour.sh g52 g27
```
- Round 1 is 06d's method with other targets. It left free 2 MiB blocks of 978 / 848 GiB on g52 and 1,130 / 1,047 GiB on g27. On g52 the kernel moved the page cache to its active list instead of reclaiming it, so the pressure reached its floor with 153 / 375 GiB of page cache left.
- Round 2 took g52 to 1,184 / 1,231 GiB; g27 did not change. A 24 GiB floor reclaims almost all page cache and can swap out other processes' memory. It was safe on g52 only because g52 ran no other users' processes and has about 1.7 TiB of swap. On a node with other users' jobs, keep the 96 GiB floor or no pressure, as for g27.
- Round 3 took g52 to 1,213 / 1,231 GiB.

*Store owners:* `prep/plan_hour.py`, the hour's planner, ends every round and writes the overlay `<OUT>/store_hour.sh`. After round 3 it gave `overlays/store_3392_hour.sh`: 3,392 GiB, 848 per pool. It sizes 4 equal pools toward 3,776 GiB. Per NUMA node it takes the smallest of five ceilings; all four NUMA nodes ended at SPARE:
- THP: 0.97 x the free 2 MiB blocks.
- SPARE: the free 2 MiB blocks minus a margin left free: 266 GiB on node A NUMA0 (250, plus 16 for the four `pd-send-ppK` processes of 3.5-3.7 GiB each), 120 on A NUMA1, 96 on B NUMA0, 160 on B NUMA1.
- MEM: free memory minus 272, 192, 64 and 192 GiB.
- LAUNCH: MemTotal - 128 GiB, the launcher's budget check.
- OWNER: 304 GiB per owner.

`--extra-reserve NODE:NUMA:GiB` (`EXTRA_RESERVE`) lowers SPARE and MEM there. The hour kept 400 GiB on g27 NUMA0, where other users' memory grew by about 190 GiB during the hour, to about 380 GiB.
- Pass `overlays/store_3392_hour.sh` when the plan's printed ceilings reach its owners: 4 x 224 and 4 x 272 GiB on node A, 4 x 144 and 4 x 208 GiB on node B. Otherwise pass the plan's own `store_hour.sh`. On larger nodes, `PLAN_TOTAL=3392` stops the plan at the hour's size.
- Do not leave the Store overlay out. The harness then sizes the owners itself (`harness/bin/store_plan.py`) with smaller margins: on the hour's own launch probes that gave 3,648 GiB and would have left about 76 GiB free on g27 NUMA0 (42 GiB of 2 MiB blocks).
- To probe again without a prep round, run `python3 $RECIPE/harness/bin/node_probe.py --drop /share_nfs/models` on each node, or read `probe-<node>.json` in the run dir of a harness `--dry-run`. Both drop the model's page cache first.

**Launch** from the driver host, with `RECIPE` and `AGX_RUNS` set as in "Before a run"; node A first:
```bash
setsid nohup bash $RECIPE/harness/bin/agx_run.sh \
  --tag p2d-c240-hour-o1-nic --commit d0a2d9d8b --topo 1p2d --nodes g52,g27 --lease <id> \
  --conc 240 --duration 3600 --warmup 10 --image rocm/atom-dev:m31-agx-local-c2f06e01 \
  --p-env $RECIPE/env/p_env_nochunk_lw2_r1_o1.sh --p-seqs 24 \
  --d-env $RECIPE/env/d_env_akmoe_lmhead.sh \
  --overlay $RECIPE/overlays/tmpdir.sh \
  --overlay $RECIPE/overlays/store_3392_hour.sh \
  --store-caps 1024/768/1152 --keep-model-cache --label FULLSTACK-NOCHUNK-AK-LMH-CAP24-LW2-R1-O1-NICLOCAL-1H \
  > $AGX_RUNS/p2d-c240-hour-o1-nic.out 2>&1 < /dev/null &
```
- `--commit d0a2d9d8b` is this branch's NIC-fix commit, `[P/D] Mooncake: register on the GPU's PCIe-local NIC, not rdma<HIP index>`. There is no `--patch`. The hour itself ran `--commit 76b83860b` with the three changes as `--patch` files, which gives the same tree.
- `env/p_env_nochunk_lw2_r1_o1.sh` is 06d's prefill role plus the two switches. The decode role and the tmpdir overlay are 06d's.
- The hour also had `AGX_PREFLIGHT_WARN_ONLY=1` in front. Its probes found no launcher port taken and no GPU memory in use, so it changed nothing.
- The automatic plan from the hour's probes (3,648 GiB) was larger than the overlay, so `--keep-model-cache` never stopped the model page-cache droppers. They were stopped outside the harness at 03:56:52 UTC, once every pool counted 848 GiB, as Pitfalls describes. Do the same: until then they slow the weight load.
- It took 1 h 49 min: 5 min of snapshot, probes and cell, 18 min from rank start to serving, 17 min of warmup, the hour, and 9.5 min of drain and summary. Let the lease run at least 2 h 10 min past the launch, and extend it before you launch (see "Before a run" step 4); the hour ran `gpool touch -t 2h15m` right before.
- Read-only samplers ran beside the hour: on node A's host, a 1-s per-thread CPU sampler, `rocm-smi --showpids` every 60 s and `docker stats` every 10 s; from the driver host, a per-node load and memory poll over ssh every 60 s. A reproduction does not need them.

**Check it is the same deployment:**
```bash
diff <(sort $RECIPE/reference/hour/docker-rank-0.env) <(sort <run>/logs/slurm_job-*/docker-rank-0.env)
```
The same lines may differ as for 06d (see Launch), except the port offset, which the hour did not use. A `--dry-run` of the command above on 2026-11-26 gave the hour's `cell.json` and container env exactly, apart from the run dir, the job id, the run token and the harness's automatic Store plan, which the overlay replaces.

Then check in the logs that the three changes are on:
- R1: the prefill's `[OFFLOAD-LOAD-PROF]` lines carry `settle_ms`.
- O1: `[pd-send pp0]` to `[pd-send pp3]` each log `Mooncake write listener bound`.
- NIC fix: the prefill logs `Auto-selecting RDMA devices rdmaN for GPU <PCI address>` once per stage, each NIC the one behind that GPU's switch. On g52: pp0 rdma3, pp1 rdma0, pp2 rdma2, pp3 rdma1, as `[pd-send ppK] Mooncake TransferEngine initialized: ... ib_device=` also shows. The decodes keep `ib_device=rdma4,rdma5,rdma6,rdma7`.

**Compare:**
```bash
python3 $RECIPE/analysis/subwindow.py <run> 0 3600 12
python3 $RECIPE/analysis/subwindow.py <run> 1200 3600 12
```
Over 0-3600 the hour gave `tok_s_per_gpu` 125717 (the CI tok/s/GPU over the records span, as in `point.txt`), `tail_free_tok_s_per_gpu` 126320, `p90_tpot_ms` 6.55 and `ttft_p50_s` / `ttft_p90_s` 0.81 / 4.19. Over minutes 20-60 (1200-3600) it gave `tok_s_per_gpu` 133240, `tail_free_tok_s_per_gpu` 134230 and `p90_tpot_ms` 6.3. Count the hour as reproduced when its CI tok/s/GPU and minutes 20-60 are within 5% of these, with 06d's health lines (see "How the numbers are computed") and `cap 848.0` on all four Store pools. The "/912" next to it in `reference/hour/point.txt` is the automatic plan that the overlay replaced.

Things to know:
- Besides the start-up wave at minutes 5-10 (the prefill queue was down to 6 by minute 9), the hour had a smaller wave at minutes 55-60: the Store hit rose to 0.73 of the prompt, and the prefill queue to 22 on average.
- g52's rdma0-3 retransmitted 38 GB over the hour (`retx` in `point.txt`), 0.02-0.03% of what they sent, with no errors or CNPs. Node A of 06a and 06d (g27) had 3.1 and 0.9 GB. This does not make g52 a bad node: at the run's start its lifetime retransmits were 0.06-0.13% of what its NICs had sent, the lowest of the five nodes the campaign measured, among them g27 (0.25-0.40%) and g23 (0.84-1.10%).
- As for 06a and 06d, other users' containers ran beside it: 3 on node A and 12-13 on node B. Node B's NUMA0 carried their CPU jobs and up to about 380 GiB of their memory.

## Previous points: 06d and 06a

Both points ran on 2026-11-25 (UTC) on 1P2D at concurrency 240, with forced
acceptance length 4.04 on both roles. The window starts after a warmup of 10
requests per lane. "CI" is the launcher's `process_result.py`.

| | **06d (previous best)** | **06a (full hour)** |
|---|---|---|
| Window | 25-min screen (`--duration 1500`), 09:21:40-09:46:40; records span 1,525 s | 1 hour (`--duration 3600`), 04:02:36-05:02:36; records span 3,625 s |
| CI total tok/s/GPU (12 GPUs) | **97,355** | **111,887** |
| req/s/GPU | 0.768 | 0.764 |
| P90 TPOT | 6.45 ms | 6.51 ms |
| TTFT p50 / p90 | 4.41 / 17.91 s | 2.00 / 15.01 s |
| Completed requests, cache hit | 14,046, 95.7% | 33,214, 96.4% |
| Tail-free tok/s/GPU | 98,152 (0.776 req/s/GPU) | -- |
| Steady part | minutes 20-25: 120,707 tok/s/GPU (126,619 tail-free), 0.84 req/s/GPU, P90 TPOT 6.54 ms, TTFT 1.41 / 4.18 s | minutes 20-60: 126,873 tok/s/GPU, 0.80 req/s/GPU, P90 TPOT 6.42 ms, TTFT 1.70 / 4.37 s |

For comparison, 06a's own first 25 minutes gave 81,052 tok/s/GPU over its
records span and 90,711 tail-free. Every run starts with a few trace-idle
minutes and then an arrival wave that queues on the prefill; it cleared by
about minute 20 in 06a and by minute 9 in the hour. So compare a 25-min screen
only with another run's first 25 minutes.

The two points differ only in these settings (compare `reference/06a/docker-rank-0.env`
with `reference/06d/docker-rank-0.env`):

| Setting | 06a | 06d |
|---|---|---|
| Chunked P->D transfer (prefill kv-transfer-config) | on, `chunked_admission_timeout_s` 1500 | off |
| Prefill `max_num_seqs` | 480 (2 x C) | 24 (`--p-seqs 24`) |
| Prefill Store load threads `OFFLOAD_LOAD_WORKERS` | 1 | 2 |
| Decode amd-kernels MoE (`ATOM_M3_AK_MOE` + 4 tiling flags) | off | on |
| Decode lm_head AK + ring (`ATOM_DG_LMHEAD_AK`, `ATOM_P1_LMHEAD_RING`) | off | on |
| Store owners, GiB (node A NUMA0 + NUMA1, node B NUMA0 + NUMA1) | 4x240 + 4x192, 4x224 + 4x144 | 4x240 + 4x176, 4x240 + 4x144 |
| Launcher ports | as main | +1000 (see Launch) |

Both use 3,200 GiB of Store in total: 800 GiB per pool, 4 pools.

Both also share these settings:
- **KV caches:** NVFP4 KV and NVFP4 index cache. The DSpark draft's own KV pool is FP8 (`--spec-decode-kv-cache-dtype fp8`).
- **Prefill:** `VLLM_PP_LAYER_PARTITION=16,16,16,12`, `--enforce-eager`, MNBT 8192. It uses the PP stage boundary over HIP IPC (`ATOM_PP_IPC=1`, 8 batches in flight) and the TP1 layer kernels. The Store keeps only each saved prompt's draft read window (`ATOM_MOONCAKE_STORE_DRAFT_WINDOW=1`); the eviction watermark is 0.89 / 0.02.
- **Decode:** ganyi's C17 decode kernels, configured in `env/d_env.sh`. Each decode runs at most 256 sequences with MNBT 16384 and CUDA graphs up to 256. The decode admission is `pd_admission` {fraction 0.15, reserve 512}.
- **Router:** atomesh `cache_aware`, `balance_abs_threshold` 2.

Two settings are lossy: the NVFP4 KV and index caches, and in 06d the AK MoE, which runs MXFP4 activations (a4w4) at every batch size. That is the MoE numerics of the InferenceX vLLM MI355X image. Nothing else is lossy: no PTPC FP8, no quantized all-reduce, no `index_topk_freq` > 1, and the DSpark draft as shipped. lm_head AK changes only the fp32 summation order.

## Code

| Repo | Ref | What it is |
|---|---|---|
| ROCm/ATOM | `76b83860b` (06a, 06d); this branch `Jasen/m31-agentx-repro-v2` adds this directory and the hour point's three commits | See the merge list below; the three commits are in "The hour point" |
| ROCm/aiter | `Jasen/m31-agentx-repro` = `c2f06e01` | See the merge list below; baked into the image |

ATOM `76b83860b` is main `49e3aee9a` (2026-11-24) with these merges:
- ganyi's p1-c17 decode delta from `origin/p1-c17-reproducible`, merged as `0756e1223`. It includes the vendored `amd_kernels/`, `p1k2/`, `p1l2/`, `l4c_y/` and `df_draft_ar/`, plus the tuned GEMM tables in `recipe/configs/`.
- The gpu_guard fix `96d1280e9`.
- ROCm/ATOM#3536 (open), head `f3c09c61c`, merged as `139754bf9`.
- The Store draft-window stack ROCm/ATOM#3599-#3602 (closed without merging), top head `aadf2a967`, merged as `81b88602d`.
- The harness's launcher change `bd78aa2f4`, merged as `31535702a`.
- The test alignment commit `66c21a920`.
- The PP4 x TP1 prefill PRs ROCm/ATOM#3791-#3801.

aiter `c2f06e01` is main `414230ff` with these merges:
- `origin/p1-integration` `b5fe771f` (ROCm/aiter#8465, #8472, #8473).
- The 9 PP4 kernel PRs, in this order: #8588 `8d508da5`, #8592 `1d69109e`, #8593 `67e56028`, #8589 `a6699349`, #8590 `f4a72574`, #8591 `96e88573`, #8594 `ff484b61`, #8595 `98bc7428` and #8602 `41dd6ff0`.

Its only submodule pin is `3rdparty/composable_kernel` `f93c029e`. That commit is an ancestor of ROCm/composable_kernel `develop`.

**The launcher is 76b83860b's own.** The harness carries a launcher change: the Store eviction knobs `MOONCAKE_STORE_EVICTION_HIGH_WATERMARK` / `_RATIO`, and a `${WORKER_RDMA_DEVICES}` placeholder that a role's kv-transfer-config fills with the NICs of that worker's GPUs. 76b83860b already contains it, so no launcher patch is needed:
- The change is `bd78aa2f4`, merged in 76b83860b as `31535702a`.
- `git diff bd78aa2f4 76b83860b -- .github/scripts/atomesh .github/benchmark` is empty.
- 06a, 06d and the hour logged `launcher_overlay=false`.
- 06a's snapshot diff is empty. 06d's is only `patches/combined_port_offset.patch`. The hour's is its three patches; with `--commit d0a2d9d8b` it is empty.

## Image

The runs used `rocm/atom-dev:m31-agx-local-c2f06e01`, a local tag that no registry has. Load the team's copy or rebuild it, then run the checks below on both nodes.

**Load the team's copy.** The team share keeps it as a `docker save` archive of 19.7 GB, `m31agx-images/c2f06e01/image-c2f06e01.tar`; ask the campaign owner for the full path. Run `docker load -i <tar>` on both nodes, which takes about 5 minutes. `docker image inspect --format '{{.Id}}'` then shows `sha256:ba5ab678acea436d0e556ca67c0dceb3a2c40768d14cccf304d1a1691fa10a42` with docker's containerd image store, as on our nodes. With the classic store it shows the config digest, `sha256:6be4d4d7b604c01eca65aa66242196057f01861c3aa48c882f4570ed365a5ee1`.

**Rebuild it.** This is the build the archive came from, without its two detours (see "How the archive was built"): stage 1 compiles `c2f06e01` directly, and stage 2 runs the FlyDSL pre-warm. It runs on one node, CPU only, with Docker 23 or newer (BuildKit: the Dockerfile uses `RUN --mount`), room for the base image and about 10 GB free in `/var/tmp`. The commands run from the driver host; the checkout and `AGX_IMAGE_ART` must be on storage that the build node mounts at the same path.
```bash
RECIPE=$PWD/recipes/m31_agentx_1p2d            # run from the root of the shared checkout
export AGX_IMAGE_ART=/shared/m31agx-images     # bundle and tars, on shared storage
NODE=g27 HOST=pit2-p03-g27                     # the build node; the scripts add AGX_HOST_PREFIX (see Before a run)
# The base, pinned by digest (nightly tags get pruned), under the tag the build uses:
ssh $HOST 'D=rocm/atom-dev@sha256:bb85c0194fb02bde4d3cebc55aff3710f2314c6c7530d3fef7aff46b4544d5f0; docker pull $D && docker tag $D rocm/atom-dev:nightly_202611240650'
# aiter c2f06e01 as a local branch, in a clone that also holds 82b6b4f19 (the base image's aiter commit):
AITER=/path/to/aiter                           # any directory on the driver host
git clone https://github.com/ROCm/aiter.git $AITER
git -C $AITER fetch origin Jasen/m31-agentx-repro
git -C $AITER branch m31-agentx-repro c2f06e016a2315b8576478ec5ae5491ce012282c
# Stage 1: a detached GPU-less container writes /var/tmp/m31agx-aiter/aiter-tree-c2f06e01.tar on the node.
AITER_REPO=$AITER bash $RECIPE/image/build_aiter_tree.sh $NODE m31-agentx-repro 48
# Follow it with the "docker logs -f" command it prints; the last line is "[agx] wrote ...". Then remove the container.
# Stage 2 on the same node, then save the image to AGX_IMAGE_ART (about 20 GB) and load it on the other node:
bash $RECIPE/image/build_image_node.sh $NODE c2f06e016a2315b8576478ec5ae5491ce012282c
bash $RECIPE/image/distribute_image.sh $NODE rocm/atom-dev:m31-agx-local-c2f06e01 g23
```
- **Stage 1** (`build_aiter_tree.sh` -> `aiter_prebuild_inner_v2.sh`) takes over an hour at 48 CPUs; the campaign's run, which had a seed, took 69 min. In the container it:
  - fetches the commit from a thin git bundle whose prerequisites (`82b6b4f19`, `18f8fdc0`) are in the base's `/app/aiter-test`, checks it out there, and runs `git submodule update --init --recursive` (CK `f93c029e` is already in the base) and `git clean -ffdxq`;
  - runs `pip install -r requirements.txt`, which installs nothing: every package there is pinned with `==`, and the base already has it;
  - builds with `AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 python3 setup.py develop` and writes `/app/aiter-test` as a tar.

  It needs neither a seed nor `AGX_TOLERATE_AOT_8WAVE_OVERFLOW`: on `c2f06e01` the AOT replay of the table that failed in the trial passes (0 failures in the CSV-fix layer).
- **Stage 2** (`build_image_node.sh` -> `Dockerfile`) replaces the base's `/app/aiter-test` with the stage-1 tree and re-registers the editable install. Its context's `aot/` holds the decode's two tuned GEMM tables from this checkout's `recipe/configs/`, renamed for aiter's table matching.
  - With no `aot/prewarm.tar`, the build AOT-compiles their FlyDSL kernels itself; the campaign's separate pre-warm of them took 394 s at 32 jobs. This only pre-warms, so failures are logged, not fatal: look for `[FAIL]` in `/var/tmp/m31agx-aiter/build-c2f06e01.log` on the node.
  - To pre-warm separately instead, as the campaign did, run `c17_aot_prewarm_inner.sh` in a GPU-less container of the base, seeded with the stage-1 tree's `aiter/jit/flydsl_cache` and `flydsl_aot`, and put its tar at `$AGX_IMAGE_ART/c2f06e01/c17-aot-c2f06e01.tar`.
- We have not run this rebuild end to end. Its image ID differs from the archive's, so judge it by the checks below.

**Checks.** Run them on each node, with `RECIPE` set to the same checkout. The last one also needs the base image on the node:
```bash
IMG=rocm/atom-dev:m31-agx-local-c2f06e01
docker image inspect --format '{{index .Config.Labels "com.amd.m31agx.aiter.commit"}}' $IMG
docker run --rm --network none --entrypoint bash -v $RECIPE/image:/agx:ro $IMG /agx/image_fingerprint_inner.sh
docker run --rm --network none --entrypoint bash $IMG -c 'cd /app/aiter-test && echo $(cat AGX_AITER_COMMIT) $(git rev-parse HEAD^{tree}) $(git status --short --untracked-files=no | wc -l)'
for i in rocm/atom-dev:nightly_202611240650 $IMG; do docker run --rm --network none --entrypoint python3 $i -m pip list --format=freeze --disable-pip-version-check; done | sort | uniq -u
```
The archive's image gave, on 2026-11-25:
- Label and `AGX_AITER_COMMIT` `c2f06e016a2315b8576478ec5ae5491ce012282c`, tree `35640aef015d8b0ccb7bdd6175605bebfb965c96`, 0 changed tracked files.
- Fingerprint: `so_count=154`, `versions torch=2.10.0+rocm7.2.4.lw.git3d3aa833 triton=3.8.0+amd.rocm7.2.0.git7cb7b059 flydsl=0.5.3 amd-aiter=0.0.0` and `mooncake imports ok`. Two lines are per build and differ in a rebuild: `so_sha=a205f88bfb4f7fe5` and `flydsl_cache_files=15701 flydsl_aot_files=12972`.
- The `pip list` line prints nothing: the image has exactly the base's 179 Python distributions, at the same versions.

**What is inside:**
- **Base:** `rocm/atom-dev:nightly_202611240650`, digest `sha256:bb85c0194fb02bde4d3cebc55aff3710f2314c6c7530d3fef7aff46b4544d5f0`. It carries ROCm 7.2.4, torch 2.10.0+rocm7.2.4, triton 3.8.0 and flydsl 0.5.3, which is aiter's pin. It also has Mooncake `206d634` with the Store (main's pin) and agentx-harness 1.0.9.
- **aiter:** `/app/aiter-test` is replaced by aiter `c2f06e01`, and `/app/aiter-test/AGX_AITER_COMMIT` holds that SHA. It is an editable install with its JIT modules prebuilt for gfx950 (`PREBUILD_KERNELS=1`), the FlyDSL AOT caches, and the FlyDSL kernels of the decode's tuned GEMM tables.
- **ATOM is not baked in.** The launcher mounts the run's snapshot of `--commit` at `/workspace/ATOM` and runs from it. The only ATOM part the image supplies is the atomesh router binary, built from the base's `/app/ATOM` (`dc6cd23bb`): the harness keeps it (`ATOMESH_BUILD_MESH=false`), and `atom/mesh` is the same in `dc6cd23bb` and `76b83860b`.

**How the archive was built** (2026-11-24, CPU only, on one node, with the scripts in `image/`):
1. **Stage 1** on aiter `0d26c2a1`, the trial merge of the same heads; its tree equals `c2f06e01` except for one CSV row. It ran with `AGX_TOLERATE_AOT_8WAVE_OVERFLOW=1` because the trial's 8-wave GEMM rows (from `2c757685`) failed 2 FlyDSL AOT probes; the flag patches the build container only and restores the file afterwards. `41dd6ff0` (ROCm/aiter#8602) later changed one row of that table. The stage was seeded with the FlyDSL caches of a first attempt that had failed on those probes.
2. **Pre-warm:** `c17_aot_prewarm_inner.sh`, in parallel in a GPU-less container of the base with the same seed, compiled 5,734 rows with 0 failures.
3. **Stage 2** with `AITER_SHA=0d26c2a1d4b108a70e61782e366c4b741d97807d` and the pre-warm tar gave `rocm/atom-dev:m31-agx-local-0d26c2a1`.
4. **CSV-fix layer:** `build_csvfix_node.sh` (`Dockerfile.csvfix`) on that image with `FIX_SHA=c2f06e016a2315b8576478ec5ae5491ce012282c` and an aiter bundle. The build refuses any change that is not a CSV under `aiter/configs/`. It checks out `c2f06e01`, writes `AGX_AITER_COMMIT`, and re-runs the FlyDSL AOT of the changed table (485 rows, 0 failed). The result is `rocm/atom-dev:m31-agx-local-c2f06e01`.
5. **Distribution:** `distribute_image.sh` ran `docker save` on the build node and `docker load` on the others, which gives the same ID everywhere.

## Hardware and placement

You need two nodes. Each node has:
- 8 x MI355X (288 GB each).
- 2 x AMD EPYC 9575F (128 cores) and 2 NUMA nodes with about 1.5 TiB of DRAM each.
- 8 RDMA NICs (ionic): `rdma0-3` on NUMA0 and `rdma4-7` on NUMA1. In PCI order (amd-smi's), GPU k shares a PCIe switch with `rdma k`. HIP numbers the GPUs in KFD order, which differs: on g52, HIP GPUs 0-7 sit behind rdma3, rdma0, rdma2, rdma1, rdma7, rdma4, rdma6, rdma5.

Hold both nodes, all 16 GPUs, for the whole run.

| Node | GPUs 0-3 (NUMA0, rdma0-3) | GPUs 4-7 (NUMA1, rdma4-7) |
|---|---|---|
| A (06a/06d: g27; hour: g52) | P: PP4 x TP1, stage k on HIP GPU k, sending to the decodes on rdma k (06a/06d) or on the NIC behind its GPU (hour); Store masters (4 pools), router | D0: TP4, `ib_device` rdma4-7 |
| B (06a/06d: g23; hour: g27) | idle (its NUMA0 DRAM and rdma0-3 carry Store owners) | D1: TP4, `ib_device` rdma4-7 |

The Store has 4 per-NIC pools; pool k uses `rdma k` on both nodes. The owners sit on both NUMA nodes of both nodes. They are THP-backed (`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`) and bound to their NUMA node. The CI number counts 12 GPUs: P plus 2 decodes. Never put a decode on node A's rdma0-3; those NICs carry the prefill stages' traffic.

**Host software.** 06a's and 06d's nodes (g27, g23) have the same stack, read with `prep/host_facts.sh` on 2026-11-25. They have run since a boot on 2026-08-28, so it also matches the team's 2026-10-27 audit of these nodes, which is where the IFWI version comes from. The hour's node A, g52, was not read on 2026-11-26: its versions and its boot (2026-08-28 19:23) come from that audit, which found all 32 pit2-p03 nodes it covered on the same versions. The stack:
- Ubuntu 22.04.5, kernel 6.8.0-60-generic, BIOS 2.0a.
- Host ROCm 7.2.1 and amdgpu 6.16.13. GPU IFWI 00175784 (113-M355-01-1K1-030A), PM firmware 04.86.16.03, SPX / NPS1.
- ionic driver 26.03.3.001 with NIC firmware 1.117.5-a-77.
- THP `madvise`, cgroup v2, Docker 29.1.3 with the containerd image store, host Python 3.10.12.

A container cannot change any of these, so the numbers compare only on the same driver and firmware. Run `bash $RECIPE/prep/host_facts.sh <node> <node>` for your nodes and compare.

## Prerequisites

The driver host, where you start the harness, needs:
- Python 3.10 or newer, with PyYAML. `cellgen.py` imports `yaml`, and `summarize.py` and `closed_loop.py` use `itertools.pairwise`.
- A full git clone of this branch, not `--depth 1` and not an archive. The harness checks out `--commit` (`76b83860b`, or `d0a2d9d8b` for the hour) as a worktree and resolves `bd78aa2f4`.
- Passwordless ssh (`BatchMode`) to both nodes.

Each node needs:
- `amd-smi` on `PATH`. Without it the preflight counts 0 GPUs and refuses the node.
- Docker usable without sudo. The launcher starts its containers with `--device` `/dev/kfd`, `/dev/dri` and `/dev/infiniband`, `--network host`, `--ipc host`, `--cap-add IPC_LOCK` and `--ulimit memlock=-1`.
- Host `python3` 3.8 or newer, `ss` for the port check, and `ip`.
- The 8 ionic NICs named `rdma0` to `rdma7`, and the host's ionic verbs provider, which the launcher mounts into the containers. `store_plan.py`, the overlays and the env defaults use these names.
- THP set to `madvise` or `always`, for the Store owners' 2 MiB pages.
- Outbound HTTPS to huggingface.co: the benchmark container downloads the dataset.

## Model and dataset

- **Model:** `amd/MiniMax-M3.1-Flash-Quark-MXFP4` on Hugging Face at revision `1442d09d4926cb286fa2dc6646c83c4af2b3898c` (2026-10-23), with its `dspark/` draft; about 224 GiB, no token needed:
  ```bash
  hf download amd/MiniMax-M3.1-Flash-Quark-MXFP4 --revision 1442d09d4926cb286fa2dc6646c83c4af2b3898c \
    --local-dir <MODEL_ROOT>/amd/MiniMax-M3.1-Flash-Quark-MXFP4
  ```
  - Do not take `main`. Its head (`8543e8e6`, 2026-11-23) has a chat template that ignores `thinking_mode`, which both roles set to `enabled` (`--default-chat-template-kwargs`), so every prompt changes without any error. It also drops 7 tokenizer files from `dspark/`.
  - The runs read it from `/share_nfs/models/amd/MiniMax-M3.1-Flash-Quark-MXFP4`.
  - Besides the source snapshot and the job dir, the launcher mounts only `/share_nfs/models`, `/data`, `/mnt`, `/shared_nfs` and `/it-share` into the containers, so keep the model under one of them.
  - If you keep it elsewhere, set `AGX_MODEL_PATH`, `AGX_MODEL_ROOT` and `AGX_PROBE_DROP`, and edit `MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS` in `env/p_env.sh`.
- **Dataset:** aiperf public dataset `semianalysis_cc_traces_weka_062126`, which is Hugging Face `semianalysisai/cc-traces-weka-062126` and needs no token.
  - The runs resolved revision `23f152f6f0f9399a85901b89a6458def0ef16729`. The launcher passes only the dataset name, so aiperf takes the dataset's current head. Check that your run replayed the same trace: `grep -o 'revision/[0-9a-f]\{40\}' <run>/logs/slurm_job-*/benchmark_results/aiperf-*/aiperf.log` must print `revision/23f152f6f0f9399a85901b89a6458def0ef16729`.
  - The replay uses 393 entries, a 1M max context, trajectory start 0.25-0.75, an idle-gap cap of 300 s and `--random-seed 42`. Runs with the same C, warmup and seed replay the same trace, so their windows compare minute by minute.

## Files

| Path | What |
|---|---|
| `harness/bin/agx_run.sh` | Runs one point: lease check, node preflight, Store plan, cell, both ranks (`pd_slurm_job.sh --spur-worker`), scraper, NIC sampler, watchdog, teardown, summary. `-h` lists its options; set `AGX_RUNS` first, because the script checks it before it reads its arguments |
| `harness/bin/` (other files) | `cellgen.py` (cell via the snapshot's `pd_matrix.py`), `store_plan.py`, `node_probe.py`, `scrape.py`, `nic_sampler.sh`, `watchdog.py`, `agx_teardown.sh <run>`, `agx_status.sh <run>`, `summarize.py --run <run>`, `docker-wrap/docker` (skips `docker pull` of an image already on the node) |
| `env/` | Role env files: `p_env.sh` (06a prefill), `p_env_nochunk.sh` -> `p_env_nochunk_lw2.sh` (06d prefill) -> `p_env_nochunk_lw2_r1_o1.sh` (hour prefill), `d_env.sh` (06a decode), `d_env_akmoe.sh` -> `d_env_akmoe_lmhead.sh` (06d and hour decode) |
| `overlays/` | Bash files sourced last in each rank env. `tmpdir.sh` (all runs) sets `TMPDIR=/dev/shm`. `store_3392_hour.sh` (hour) and `store_3200_06c.sh` (06d) hold the explicit Store owners; the hour's is an example for nodes like its own. `store_06a_plan.sh` holds 06a's owners, which the run got from its automatic plan. `port_offset_1000.sh` (06d) shifts the launcher ports |
| `patches/combined_port_offset.patch` | 06d: lets `ATOMESH_COMBINED_PORT_OFFSET` shift the launcher ports (`pd_slurm_job.sh` only) |
| `prep/` | Store host-memory prep: `prep_store_mem_1p2d.sh` (06a), `prep_final.sh` + `evict_cache.py` (06d), `prep_hour.sh` + `evict_cache.py` and the Store planner `plan_hour.py` (hour), `numa_snapshot.sh`. `host_facts.sh` reads a node's host software, read-only |
| `analysis/` | `subwindow.py` (slices, tail-free), `closed_loop.py` (req/s/GPU), `pd_balance.py` and `screen_compare.py` (screen vs a base run's first 25 min) |
| `image/` | The two Dockerfiles; the in-container build scripts `aiter_prebuild_inner_v2.sh` (stage 1) and `c17_aot_prewarm_inner.sh` (pre-warm); the host scripts that run a build on a node, `build_aiter_tree.sh` (stage 1), `build_image_node.sh` (stage 2), `build_csvfix_node.sh` (CSV-fix layer) and `distribute_image.sh` (save and load); `image_fingerprint_inner.sh` (image check) |
| `reference/hour`, `reference/06a`, `reference/06d` | Each run's `point.txt`, `ci-summary.md` and the env the rank-0 containers got (`docker-rank-0.env`, run token removed). Rank 1 differs only by `ATOMESH_DECODE_ENV_ROCR_VISIBLE_DEVICES=4,5,6,7`. The full run dirs are `p2d-c240-hour-o1-nic-20261126034734` (hour), `p2d-c240-nochunk-cap24-lw2-screen-20261125084047` (06d) and `p2d-c240-20261125032033` (06a) under the campaign's run root on the team share |

## Before a run

1. Put a checkout of this branch on storage that the driver host and both nodes mount at the same path. The harness runs its scripts on the nodes over ssh, and the containers mount the per-run source snapshot. Do not put it under `/tmp`.
2. Set `AGX_RUNS` to a directory on the same shared storage. It holds the run logs and source snapshots. Do not use `$HOME`: an aiperf export can exceed 1 GB.
   ```bash
   RECIPE=$PWD/recipes/m31_agentx_1p2d     # run from the root of the shared checkout
   export AGX_RUNS=/shared/m31agx-runs     # shared storage
   ```
3. Check the Prerequisites above. The harness turns a node name like `g27` into the host name `pit2-p03-g27`. Set `AGX_HOST_PREFIX` to change the prefix, or `AGX_HOST_PREFIX=` to pass full host names.
4. Hold both nodes exclusively for the whole run, prep included. To have the harness check your lease on the team's GPU pool, `export AGX_GPOOL_STATE=<pool>/state.json` and pass `--lease <id>`. Otherwise export `AGX_NO_LEASE_CHECK=1`. With the lease check, the watchdog stops the run 10 min before the shortest lease's estimated end. The harness reads that end once, at launch, so extending the lease later does not move it. Let the lease run at least 1 h 30 min (06d) or 2 h 10 min (06a, the hour) past the launch, plus the prep before it: 10-15 min for 06d, about 45 min for the hour.
5. Load or build the image on both nodes and run its checks (see Image).
6. **Prepare host memory for the Store.** Store owners need 2 MiB pages bound to their NUMA node. On our nodes, other users' jobs had left up to about 800 GiB of page cache per NUMA node, in files we could not read, so `fadvise` freed nothing. Much of the free memory was fragmented. Without prep, two things can happen:
   - The automatic plan shrinks; one attempt got 2,752 GiB instead of 3,200.
   - The owners compact memory page by page, and the Store start stalls for up to `MOONCAKE_STORE_COMPACT_TIMEOUT` (1,800 s). This looks like a hang.

   Run the prep only on nodes you hold, because it allocates about 1.3 TiB per NUMA node for a few minutes. For the hour, follow "Before the run" under "The hour point".

   For 06d (memory pressure, then compaction):
   ```bash
   NODE_A=g27 NODE_B=g23 COMPACT_TIMEOUT=360 bash $RECIPE/prep/prep_final.sh 1p2d
   ```
   `evict_cache.py pressure` faults and frees 4 KiB anonymous pages bound to each NUMA node. It stops when MemFree + Inactive(file) nears 96 GiB, so the kernel reclaims clean page cache without swapping. Then the launcher's `compact_huge_pages.py` compacts each NUMA node. On 06d's nodes this left free 2 MiB blocks of 1,168 / 1,008 GiB (node A) and 1,276 / 1,172 GiB (node B), so the explicit 3,200 GiB plan needed no compaction at start. The script also proposes a bigger plan in `OUT/store_final.sh`; 06d did not use it.

   For 06a (compaction only):
   ```bash
   bash $RECIPE/prep/prep_store_mem_1p2d.sh g27 g23
   ```
   It drops the model's page cache and compacts toward A0 1060 / A1 800 / B0 1190 / B1 800 GiB, with a 900 s cap. It picks as node A the node with more free 2 MiB blocks; set `FORCE_A=<node>` to choose.

## Launch

Run these from the driver host, with `RECIPE` and `AGX_RUNS` set as above. Replace `g27,g23` with your nodes, node A first (P + D0), then node B (D1 + Store owners). Each run writes `$AGX_RUNS/<tag>-<UTC stamp>/` and prints `done: <run>/point.txt` at the end. 06d took 1 h 14 min and 06a took 1 h 52 min:
- About 7 min of snapshot, node probes and cell generation.
- About 16 min from rank start to serving: Store start, weight load and graph capture.
- About 18-19 min of warmup.
- The window.
- 8-11 min of drain and summary.

**06d (best, 25-min screen):**
```bash
setsid nohup bash $RECIPE/harness/bin/agx_run.sh \
  --tag p2d-c240-nochunk-cap24-lw2-screen --commit 76b83860b --topo 1p2d --nodes g27,g23 --lease <id> \
  --conc 240 --duration 1500 --warmup 10 --image rocm/atom-dev:m31-agx-local-c2f06e01 \
  --p-env $RECIPE/env/p_env_nochunk_lw2.sh --p-seqs 24 \
  --d-env $RECIPE/env/d_env_akmoe_lmhead.sh \
  --overlay $RECIPE/overlays/tmpdir.sh \
  --overlay $RECIPE/overlays/store_3200_06c.sh \
  --keep-model-cache --label FULLSTACK-NOCHUNK-AK-LMH-CAP24-LW2 \
  > $AGX_RUNS/p2d-c240-nochunk-cap24-lw2-screen.out 2>&1 < /dev/null &
```
06d as run had three more settings, carried over from the screen before it (06c): `AGX_PREFLIGHT_WARN_ONLY=1` in front of the command, and `--overlay $RECIPE/overlays/port_offset_1000.sh --patch $RECIPE/patches/combined_port_offset.patch`. 06c needed them because another user's container held the launcher ports (see Pitfalls). 06d's own probes found none of those ports taken and no GPU memory in use, so in 06d the three changed only the port numbers. Add them only when a foreign process holds the ports.

**06a (full hour):**
```bash
setsid nohup bash $RECIPE/harness/bin/agx_run.sh \
  --tag p2d-c240 --commit 76b83860b --topo 1p2d --nodes g27,g23 --lease <id> \
  --conc 240 --duration 3600 --warmup 10 --image rocm/atom-dev:m31-agx-local-c2f06e01 \
  --p-env $RECIPE/env/p_env.sh --d-env $RECIPE/env/d_env.sh \
  --overlay $RECIPE/overlays/tmpdir.sh \
  --keep-model-cache --label FULLSTACK \
  > $AGX_RUNS/p2d-c240.out 2>&1 < /dev/null &
```

Notes on these commands:
- They are the runs' commands with paths made relative, and for 06d without the three port settings above.
- The runs took `--image` from the harness default, which resolved to this tag; `--warmup 10` is also the default.
- The other options (`--al 4.04`, `--store-caps 1024/768/1152`, `--store-pools per_nic`, draft window on, `--pp-inflight 8`, `--mem 0.9`) are defaults.
- Use `--commit 76b83860b`, not this branch's tip, so that the snapshot tree is the one that ran. The tip also carries the NIC fix, which has no switch.
- 06a's Store came from the harness's automatic plan, which gave exactly the owners in `overlays/store_06a_plan.sh`. Add `--overlay $RECIPE/overlays/store_06a_plan.sh` to pin them on other nodes.
- To stop a run: `bash $RECIPE/harness/bin/agx_teardown.sh <run dir>`. It stops only this run's containers and processes.

Check the run is the same deployment by comparing container envs:
```bash
diff <(sort $RECIPE/reference/06d/docker-rank-0.env) <(sort <run>/logs/slurm_job-*/docker-rank-0.env)
```
Only these lines should differ:
- `ATOMESH_RUN_TOKEN`.
- The job id inside the two `AITER_CONFIG_*` paths.
- `ATOMESH_COMBINED_PORT_OFFSET`, if you left out the port offset (06d).
- The Store owners, if you changed them.
- `DISPLAY_TOPOLOGY`, if you passed another `--label`.
- `DOCKER_IMAGE`, if your image has another tag.
- `MODEL_PATH`, `DRAFT_MODEL_PATH` and `ATOMESH_PREFILL_ENV_MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS`, if the model is elsewhere.

## How the numbers are computed

- **CI line in `point.txt`:** `summarize.py` runs the snapshot's `.github/scripts/atomesh/process_result.py` over the job dir, as CI does.
  - Total tok/s/GPU = (input + output tokens of every profiling request) / records span / 12 GPUs. The records span runs from the first request start to the last request end, so it includes the drain tail after the window.
  - It also gives P90 TPOT, P90 E2E-normalized interactivity and TTFT p50/p90.
- **25-min screen vs 1 hour:** the screen is the same CI cell with `--duration 1500` instead of 3600. Same warmup, same replay order.
- **req/s/GPU** = completed / records span / 12 (`analysis/closed_loop.py --run <run>`).
- **Tail-free and slices** come from `analysis/subwindow.py <run> FROM_S TO_S 12`. It takes the requests that started in the slice and reports tok/s/GPU over their own span. Tail-free counts only the requests that also ended inside the slice, divided by the slice length.
  - 06d's numbers come from `subwindow.py <run> 0 1500 12` and `subwindow.py <run> 1200 1500 12`.
  - 06a's minutes 20-60 come from `subwindow.py <run> 1200 3600 12`.
  - The hour's numbers come from `subwindow.py <run> 0 3600 12`, `0 1500 12` (first 25 min) and `1200 3600 12`.
- **Screen against a base run:** `analysis/screen_compare.py <new run> 12 <base run> 12 1500` compares a screen with a base run's first 25 min, client side and server side (`pd_balance.py`). It writes `screen_compare.json` and `pd_balance_first1500s.json` into the run dirs.

Re-running `summarize.py`, `subwindow.py`, `closed_loop.py` and `pd_balance.py` from this directory on both runs' data reproduces every number above exactly. On the hour's data, `subwindow.py`, `closed_loop.py` and `pd_balance.py` reproduce its numbers exactly.

What to compare, with 06d as the example (06a in `reference/06a/point.txt`):
- **Headline:** the CI line, req/s/GPU, tail-free tok/s/GPU and minutes 20-25.
- **Health:** both ranks `rc 0` and the watchdog did not stop the run. The events line shows 0 `cq_completion_timeouts`, 0 `store_taint` and 0 `decode_kv_receive_failed`. Shutdown tracebacks are benign; 06d logged 22.
- **Store:** each of the 4 pools reports 800 GiB (`storeK ... cap 800.0`). The "/928" next to it in 06d's `point.txt` is the automatic plan that the overlay replaced.
- **Decodes:** `toks/fwd 4.04` (forced acceptance), `ib_device rdma4-7` in the decode logs, running about 23 per decode and KV use about 0.33.
- **Prefill:** about 51K computed tok/s, prefix hit HBM 0.33 + Store 0.63, and stage busy 0.70 / 0.85 / 0.90 / 0.87.
- **The hour** (`reference/hour/point.txt`): each pool `cap 848.0`; decodes running about 28.5 each with KV use about 0.43; the prefill at about 51K computed tok/s, prefix hit HBM 0.49 + Store 0.47, stage busy 0.53 / 0.72 / 0.83 / 0.84.

Count a point as reproduced when its CI tok/s/GPU and its steady slice (minutes 20-25 for 06d, 20-60 for 06a) are within 5% of the reference, with the same health lines. The 5% is a working tolerance, not a measured noise band: each point ran once, and the start-up wave is the most variable part. Run a point twice before you read a gap as real.

The reference nodes were not clean. Other users' containers ran beside both runs: 14-15 on node A and 2 on node B. The preflight found no GPU memory in use, but they shared the CPUs and DRAM. During 06d, and not during 06a, one of them was a CPU-only simulation that held 48-442 GiB of anonymous memory on node A. So the two points differ only in the settings above inside the containers, not necessarily on the nodes, and clean nodes may land differently.

## Pitfalls

- **Another process on the launcher ports.** In the 06c screen, another user's CPU-only container on node A listened on ports 29474-29601. The prefill's TCPStore failed with EADDRINUSE on port 29600. The fix is `patches/combined_port_offset.patch` plus `overlays/port_offset_1000.sh` (all ports +1000), together with `AGX_PREFLIGHT_WARN_ONLY=1`. 06d kept all three (see Launch):
  - The preflight does not know about the offset. It still checks the base ports, which the other process holds, so it refuses the node.
  - Warn-only also turns its GPU-VRAM and RDMA-port checks into warnings, so check those yourself.
- **`--keep-model-cache` with an explicit Store overlay.** After the Store starts, the launcher's page-cache dropper re-evicts the model's page cache every 45 s for 30 min, which slows weight loading. `--keep-model-cache` stops it once every Store pool counts its planned owners. The harness compares the pools with its own automatic plan (`store_plan.json`), not with the overlay:
  - When the overlay is smaller, as in 06d and the hour, the check never passes, and the droppers run their full 30 min.
  - When the overlay is larger, the check passes as soon as each pool reaches the automatic plan's size, before the overlay's last owners pin their memory. Those owners then compete with the model's page cache (see THP).
  - Either way, stop the droppers by hand once every pool reports the overlay's capacity, as 06d and the hour did. The masters' `master_total_capacity_bytes` in `scrape.jsonl` gives it:
    ```bash
    python3 -c 'import json, sys
    cap = {}
    for line in open(sys.argv[1]):
        try: row = json.loads(line)
        except ValueError: continue
        if row["target"].startswith("store"): cap[row["target"]] = round(row["m"].get("master_total_capacity_bytes", 0) / 2**30)
    print(cap)' <run>/scrape.jsonl    # 06d: {'store0': 800, 'store1': 800, 'store2': 800, 'store3': 800}
    ```
    Then run this on each node; `<jid>` is in `<run>/run.json`:
    ```bash
    for c in $(docker ps --format '{{.Names}}' | grep -E '^atomesh-.*-<jid>-[01](-benchmark|-eval)?$'); do
      docker exec $c sh -c 'for p in $(pgrep -f "[d]rop_page_cache.py"); do kill $p; done'; done
    ```
- **THP.** See host-memory prep above. Owners larger than the free 2 MiB blocks of their NUMA node can stall the Store start. `store_plan.py` keeps its plan under 0.97 x those blocks; nothing checks an explicit overlay.
- **Image `/tmp` permissions.** The image has a root-owned `/tmp/aiter_configs`, and the launcher's containers run as your uid. `overlays/tmpdir.sh` is required for this image; without it the prefill dies at import with a PermissionError on `/tmp/aiter_configs/*.lock`.
- **Preflight refusals.** The preflight refuses a node when:
  - any GPU holds more than 2 GB of VRAM;
  - a launcher port is already taken;
  - an RDMA port is not ACTIVE;
  - the image is missing. A local tag cannot be pulled, so load or build it first.

  It only warns about other users' containers and never touches them.
- **Cold start.** A node with a cold page cache loads the weights in up to about 26 min. The first server start on a node also compiles FlyDSL kernels, which are then cached per node and image.
- **Editing during a run.** Never edit `agx_run.sh` while a driver runs: bash reads the file as it goes.
- **Leftover snapshot.** If the driver's last step prints `failed to delete .../src/<tag>-<stamp>: Directory not empty`, delete that directory and run `git worktree prune`. A preflight refusal (exit 4) leaves the snapshot too: run `git worktree remove --force $AGX_RUNS/src/<tag>-<stamp>`.

## Changes from the scripts as run

Everything here is a copy of the campaign's harness, role env, overlay, prep, analysis and image files, except the new `prep/host_facts.sh` and `prep/prep_hour.sh`. Most changes below only make a path relative or overridable, and the others are listed with them. The commands above behave as the runs did, apart from 06d's port settings (see Launch).

The changes, by file:
- **`harness/bin/agx_run.sh`:**
  - `H` comes from the script's location (`AGX_HARNESS`).
  - The `REPO` default is this checkout (`AGX_REPO`).
  - `AGX_RUNS` is required; the old default was a personal directory.
  - The `AGX_LAUNCHER_REF` default is the SHA `bd78aa2f4`; it was a local branch name that pointed at that SHA.
  - The lease check reads `AGX_GPOOL_STATE`, which is required unless `AGX_NO_LEASE_CHECK=1`. It checks `GPOOL_USER`/`USER` instead of a fixed user. The YIELD poll uses `AGX_GPOOL`, else `<state dir>/bin/gpool`.
  - Without `--image`, the image comes from `AGX_IMAGE_CHOICE`, else `rocm/atom-dev:m31-agx-local-c2f06e01`. It used to come from the campaign's choice file, which named this tag during both runs, else the nightly.
  - The node-name prefix comes from `AGX_HOST_PREFIX` (default `pit2-p03-`).
  - Two comment lines changed.
- **`harness/bin/agx_teardown.sh`:** the node prefix comes from `AGX_HOST_PREFIX`.
- **`harness/bin/cellgen.py`:** `AGX_MODEL_PATH` and `AGX_MODEL_ROOT` override the model path and the model root; the defaults are unchanged.
- **`harness/bin/summarize.py`:** when the run's snapshot is gone, it falls back to this checkout's `.github/scripts/atomesh` (`AGX_CI_SCRIPTS`) instead of the launcher worktree. The files are identical.
- **`env/*`:**
  - The files source each other by relative path.
  - `d_env.sh` takes the two tuned GEMM tables from this branch's `recipe/configs/` (`TUNED_CONFIGS`). They are byte-identical to the copies the runs mounted (blobs `027bb88a` and `efc14790`).
  - One comment in `p_env.sh` lost an internal path.
- **`prep/*`:**
  - Paths are relative.
  - `prep_final.sh` takes `NODE_A`/`NODE_B` (they were fixed to g27/g23).
  - `AGX_HOST_PREFIX` and `AGX_PROBE_DROP` are honored.
  - The default output directory is `$AGX_RUNS` (else `$PWD`).
- **`analysis/screen_compare.py`, `analysis/pd_balance.py`:** script paths are relative. `pd_balance.py`'s `[OFFLOAD-LOAD-PROF]` pattern also accepts the `settle_ms` field that R1 adds, as the hour's own copy did; on 06a and 06d it gives the same output as before.
- **`image/build_aiter_tree.sh`, `build_image_node.sh`, `build_csvfix_node.sh`, `distribute_image.sh`:**
  - The aiter clone, the artifact directory and the node-local directory come from `AITER_REPO`, `AGX_IMAGE_ART` and `AGX_IMAGE_NODE_DIR` (default `/var/tmp/m31agx-aiter`, as run). `AGX_BASE_IMAGE` and `AGX_IMAGE_TAG` override the base and the tag, and `AGX_HOST_PREFIX` the node prefix.
  - The Dockerfiles and the in-container scripts come from this directory. `build_image_node.sh` copies the two decode tables from this checkout's `recipe/configs/`, which hold the same bytes as the build's copies, and writes the same `aot/SOURCE.txt`.
  - The stage-1 container mounts this directory and the artifact directory instead of the whole share. Container names start with your user name.
  - ssh runs with `BatchMode`, and `docker build` with `DOCKER_BUILDKIT=1`. `build_aiter_tree.sh` refuses a `BRANCH` that is not a local branch, and `distribute_image.sh` creates the node-local directory.
- **`image/Dockerfile`:** two comments name the scripts by their names in this directory.
- **`prep/host_facts.sh`** is new: a read-only subset of the team's host audit script.

The hour ran with the campaign's copies of the harness and of 06d's decode role and tmpdir overlay, which differ from the ones here as listed above. Its own files are here with these changes:
- `env/p_env_nochunk_lw2_r1_o1.sh` is its prefill role with a relative `source` path and a new comment. Sourced, it gives the same env.
- `overlays/store_3392_hour.sh` is its Store overlay with a new comment and the same two exports.
- `prep/plan_hour.py` is its Store planner with the license header, a new docstring and a shorter comment line in the overlay it writes; the code is unchanged. On the probes taken after each prep round and at the launch, its printout and plan equal the campaign copy's; after round 3 they give `store_3392_hour.sh`'s exports.
- `prep/prep_hour.sh` is new. Rounds 1 and 2 ran as two campaign scripts, round 3 by hand. With the knobs in "The hour point", it sends the nodes the same commands as rounds 1 and 2 (round 1's now also print the compaction's exit code) and runs round 3's recorded steps. Its probes skip the image check.
- Its three `--patch` files are this branch's three code commits.

All other files are unchanged. That includes `watchdog.py`, `scrape.py`, `store_plan.py`, `node_probe.py`, `nic_sampler.sh`, `agx_status.sh`, `docker-wrap/docker`, the overlays, the patch, `evict_cache.py`, `subwindow.py`, `closed_loop.py`, `image/Dockerfile.csvfix`, `image/aiter_prebuild_inner_v2.sh`, `image/c17_aot_prewarm_inner.sh` and `image/image_fingerprint_inner.sh`.

Check: `cellgen.py` with these env files and each run's Store plan regenerates both runs' `cell.json` exactly. Only the log root and the source path of the two CSVs differ, and the harness copies those CSVs into the job dir as before. For the hour, the `--dry-run` in "The hour point" did the same check end to end.
