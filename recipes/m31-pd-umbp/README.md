# MiniMax-M3.1 1P1D with a MORI-UMBP tier on the prefill node (MI455X)

The MiniMax-M3.1 1P1D of `rocm/fw-bringup:gfx1250-minimax-m3-20261125-v2` runs a prefill node (P) and a decode node (D)
on two MI455X trays in one UALink vPOD:
- P is PP4 x TP1;
- D is DP4/EP4 with MegaMoE and DSpark K=7;
- the KV moves over MoRIIO fabric.

This recipe replaces P's LMCache CPU tier with MORI-UMBP: the `umbp_store` connector over one UMBP standalone server
per tray. Everything else is unchanged: the kit (`/app/m31_pd_recipe`), the P/D flags, D and the router.

On branch `zejun/m31-pd-umbp-pool` it can also hand the paged HBM pools to UMBP (`umbp_pool`, the L1): section 7.

Status:
- `umbp_store`: CPU-tested (`tests/test_umbp_store_*.py`); on GPU, check V1 passed (section 5), and V2 and A1 are
  pending. Tracking: LLM-Booster #826.
- `umbp_pool`: CPU-tested (`tests/test_umbp_pool_*.py`, `tests/test_umbp_l1_pool.py`); not run on GPU yet. Tracking:
  LLM-Booster #827.

## 1. What runs

| | LMCache arm (today) | UMBP arm (this recipe) |
|---|---|---|
| P connector | `multi[moriio fabric producer, lmcache_offload save_policy=lazy]` | `multi[moriio fabric producer, umbp_store save_policy=lazy]` |
| Tier | 4 in-process LMCache pools (one per PP stage) | 1 UMBP standalone server in the P container, shared by the 4 stages |
| Save | a block is stored when HBM is about to evict it (lazy) | the same: `LazySaveSchedulerMixin` over UMBP's keys |
| Keys | LMCache chunk hashes | `atomumbp1-<namespace>/w<stage>of4/<digest>`: chained blake2b digests, one 128-token chunk per KV block |
| Lookup | past the HBM hit, LMCache RPC | past the HBM hit, `batch_exists` on a lookup thread, prefetched for the head of the waiting queue |
| HBM prefix policy | SLRU, half protected | the same |
| Eviction | in-process, LMCache | in the server at its watermarks; a chunk's four stage keys are evicted together (mori#1023) |

A prefix hit needs every stage's key of each chunk: the lookup takes the shortest run over the four stages.

## 2. Image

```bash
cd recipes/m31-pd-umbp
docker build -t <you>/m31-pd-umbp:<date> .     # needs network: apt (gRPC), github (mori PR refs, this branch)
```

The Dockerfile starts from the v2 image and changes three things:
- **mori**: rebuilds the image's own mori c5bdc391 with `BUILD_UMBP=ON`, plus six UMBP-only fixes (#929, #1049, #1048,
  #1086, #1023, #941). The MORI-IO and EP code is unchanged, so the fabric P/D and D's MegaMoE are what v2 ran.
- **mori, L1**: adds UMBP's L1 service and its Python client (`mori_umbp_l1`) for `umbp_pool`:
  - #1045, #1082 and #1089 from main;
  - #1102 (draft: multi-part pools, one part per GPU of an engine), vendored at c93838be
    (`mori-pr1102-c93838be.patch`).

  All under `src/umbp`. The L1 service runs inside the same `umbp_standalone_server`.
- **ATOM**: checks out this branch (`zejun/m31-pd-umbp-pool`).
- **kit**: applies `kit.patch` to `/app/m31_pd_recipe/kit`.

Do not add mori#1050 / #1078. They make a lookup lease the keys it finds instead of renewing them, so a conversation's
head chunk, which always hits in HBM, would age out of UMBP first.

## 3. Host

- As the v2 recipe: two trays in one vPOD, cold-booted and `modprobe amdgpu` together, and `kernel.numa_balancing=0`.
- P's host memory must have the tier plus 5% and 16 GiB available (`pd_up.sh` checks).
  - 1 TB trays: `PD_UMBP_GB` 600–700.
  - 2 TB trays: up to about 1,600.
- Hugepages are optional (`UMBP_HUGEPAGES=1`). With them a 740 GB tier starts in 10–13 s instead of 64–81 s, and the
  GPU driver does not re-check 4 KiB registrations when pages move. Reserve 2 MiB pages on both NUMA nodes first.
- A container created before this patch lacks `--cap-add SYS_NICE` (NUMA binding of the tier): remove it once
  (`docker rm -f $PD_CONTAINER`) and let `prep_container.sh` create it again.

## 4. Run

The v2 README's flow, with `PD_UMBP_GB` set instead of `PD_LMCACHE_GB` for P:

```bash
# in the P container's kit, through the control machine's ctl/ scripts or by hand:
PD_UMBP_GB=640 LAZY=1 TP=1 PP=4 PD_CP=0 GPU_MEM=0.92 AL=<0|4.04> ./fab_up.sh p <tag> <conc>
DPEP=1 PD_CP=0 DELAYER=0 AL=<0|4.04> ./fab_up.sh d <tag> <conc>
```

What `PD_UMBP_GB` does:
- `pd_up.sh` starts `umbp_server.sh` in the P container before ATOM, with:
  - a DRAM tier of `PD_UMBP_GB` GiB over NUMA nodes 0 and 1;
  - the same `ROCR_VISIBLE_DEVICES` as ATOM;
  - the socket at `unix:///tmp/umbp/server.sock` and metrics on `:9092`;
  - a log at `$WORK/logs/<tag>/umbp-server.log`.
- P's kv-transfer-config is `P_KV_UMBP` (`fab_env.sh`): `umbp_store.chunk_tokens=128`, `save_policy=lazy`,
  `lookup_threads=1`.
- `ATOM_PREFIX_CACHE_POLICY=slru`, `ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5`, `ATOM_OFFLOAD_MISS_ACCOUNTING=1`.
- `down.sh` also stops the UMBP server, which frees its DRAM.

D, the router and AgentX are run exactly as in the v2 recipe.

The `umbp_store` settings (all under `kv_connector_extra_config`, prefixed `umbp_store.`):

| Setting | Default | Here |
|---|---|---|
| `address` | required | `unix:///tmp/umbp/server.sock` |
| `chunk_tokens` | 256 | 128: lazy stores one chunk per KV block |
| `save_policy` | `eager` | `lazy` |
| `lookup_threads` | 1 | 1 (0: lookups on the scheduler thread) |
| `tier_index_chunks` | 0 | 0 (> 0: `ATOM_PREFILL_ORDER=cached_first` also ranks by an index of what the tier holds) |
| `lookup_timeout_ms`, `startup_timeout_ms`, `load_pool_mib`, `save_pool_mib`, `save_abandon_timeout_s` | 1000, 10000, 1024, 256, 300 | defaults |

## 5. Checks before any performance run

Each check runs only after the previous one passed.

| # | Where | What | Pass |
|---|---|---|---|
| V1 | P tray alone, aggregated PP4 (no DSpark on a single-node PP4) + `umbp_store`, DRAM only (`kit/v1_up.sh`) | gate, then the read-back (`kit/needle.py`): 10 prompts of 32K tokens with a code near their head; a flood twice HBM's KV pushes them out to UMBP; asked again, they must load from UMBP and still name the code | GATE PASS, READBACK PASS |
| V2 | 1P1D | VERIFY C16 (`ATOM_MORIIO_VERIFY=1`, incl. prefixes restored from UMBP), then GSM8K through the router | moriio-verify `bad=0`, GSM8K ≥ 0.945 |
| A1 | 1P1D | AgentX C192 900 s: the LMCache arm vs the UMBP arm, same DRAM budget | report as below |
| A2 | 1P1D | the better arm at C256 and C300; C300 for 1 h (30-minute decay) | hit ≥ 94% at C300 |

V1 on one tray (11-27 on e15-05: GATE PASS; READBACK PASS, 10 loads of 328,704 tokens from UMBP, 10/10 right):

```bash
# in the P tray's kit; GPU_MEM=0.25 leaves HBM about 570K tokens of KV, LRU makes the flood evict every needle prompt
GPU_MEM=0.25 UMBP_GB=256 V1_ENV=ATOM_PREFIX_CACHE_POLICY=lru ./v1_up.sh v1
docker exec "$PD_CONTAINER" python3 /kit/gate.py http://127.0.0.1:8010 /models/MiniMax-M3.1-Flash-Quark-MXFP4
docker exec "$PD_CONTAINER" python3 /kit/needle.py http://127.0.0.1:8010 /models/MiniMax-M3.1-Flash-Quark-MXFP4 \
  --flood-concurrency 2 --max-tokens 1024
CONTAINER="$PD_CONTAINER" ./down.sh
```

GSM8K runs in V2 instead: an aggregated PP4 decodes eagerly, a request at about 10 tokens/s.

What to read:
- **P log:**
  - `[OFFLOAD-MISS-ACCOUNT]`: of each prompt's reusable prefix, the share served from HBM, loaded from the tier, held
    but not loaded, and evicted from both;
  - `[OFFLOAD-LOOKUP-STATS]`: lookups, uneven ones (a stage missing a chunk), stranded objects;
  - `[Cache Tiers]`: HBM- and tier-served shares of all reuse;
  - `UMBP lazy:` warnings: unsaved blocks HBM evicted, failed demotions.
- **ATOM Prometheus (`/metrics` on the API port):** the offload counters, named `atom:lmcache_*` whatever the tier:
  `load_requests`, `loaded_tokens`, `load_failures`, `save_requests`, `saved_tokens`, `lazy_evicted_unsaved_blocks`,
  `lazy_demotions_failed`, `lazy_held_blocks`. The miss split is in the log only (`[OFFLOAD-MISS-ACCOUNT]`), and the
  `umbp_*` lookup counters are not exported.
- **UMBP server `:9092/metrics`:**
  - `mori_umbp_client_capacity_{total,used,available}_bytes`;
  - `mori_umbp_client_local_evict_total`;
  - `mori_umbp_backend_ops_total{op="allocate",status="no_space"}`.

## 6. Known limits

- **A hung UMBP server blocks P's save and load threads.** Mori's put/get have no deadline; only lookups are bounded.
  A lazy demotion that does not report in `save_abandon_timeout_s` (300 s) is unpinned unsaved, but a load in flight
  keeps its request parked.
- **A server restart loses its tier.** Restart P after it: the kit stops and starts both together.
- **No exclusive tiers.** `ATOM_OFFLOAD_LAZY_EXCLUSIVE` is refused, because UMBP has no call that makes an object the
  next it evicts. A prefix loaded back stays in both tiers.
- **No `ATOM_M3_INDEX_SHARD`.** A sharded index region does not fit the store's fixed-size chunks. P at TP1 has no
  index shard anyway.
- **Not covered here:**
  - the D tray's DRAM over UALink (mori#1024);
  - an SSD tier.
  - `umbp_pool` has its own section (7).

## 7. `umbp_pool` (L1): `PD_UMBP_POOL=1`

`umbp_pool` hands each half's paged HBM pool to UMBP. The UMBP server keeps the free list, the content index and the
eviction order of the pool, and copies a prefix one engine rank lacks from another rank's pool on the same server
(HBM to HBM).

| | P | D |
|---|---|---|
| Connector | `multi[moriio producer, umbp_pool]`, plus `umbp_store` (eager) with `PD_UMBP_GB` | `multi[moriio consumer, umbp_pool]` |
| Pool | one pool of 4 parts, one per PP stage; the head's scheduler owns it | 4 pools, one per DP rank |
| Shares with | nobody: the PP4 engine fills the tray, so the server has no other pool | the other DP ranks |
| UMBP server | in the P container, DRAM tier `PD_UMBP_GB` (or `UMBP_POOL_DRAM_GB`) | in the D container, DRAM tier `UMBP_POOL_DRAM_GB` (1) |

On 1P1D M3.1 this is a functional bring-up: P has no peer pool, and D gains only a shorter P→D push.

```bash
# either half, through fab_up.sh (on top of section 4's flow)
PD_UMBP_POOL=1 ./fab_up.sh p <tag> <conc>                  # P: producer + umbp_pool
PD_UMBP_POOL=1 PD_UMBP_GB=640 ./fab_up.sh p <tag> <conc>   # P: producer + umbp_pool + umbp_store (eager)
PD_UMBP_POOL=1 ./fab_up.sh d <tag> <conc>                  # D: consumer + umbp_pool, incremental push
```

| Knob | Default | What |
|---|---|---|
| `UMBP_POOL_INCREMENTAL` | 1 | D's `ATOM_MORIIO_INCREMENTAL_PUSH`.<br>1: a prefix D holds, or copies in from another DP rank, is not pushed again.<br>0: every receive writes the whole block table, and such a request takes no prefix hit from the shared pool. |
| `UMBP_POOL_DRAM_GB` | 1 | The server's DRAM tier when only the L1 runs. |
| `UMBP_L1_GATHER_COPY` | 0 | Server side. This mori's copy engine gathers host ↔ device only, so 0 batches the GPU-to-GPU copies as `hipMemcpyAsync`. |

**Checks, in order:**
1. **The pool can be adopted.** MoRIIO fabric allocates each pool in fabric (VMM) memory. UMBP adopts it through a HIP
   IPC handle (`adopt_exportable_pool`), and whether gfx1250 gives one for that memory is unverified.
   - Success: every stage of P, and every DP rank of D, logs `UMBP pool atom:dp<r> part <p> of <n>: adopted ...`.
   - Failure: startup stops with `no HIP IPC handle ... cannot adopt it (it was allocated in exportable memory by a P/D
     connector ...)`. The pool then has to be allocated by UMBP in fabric memory instead (mori#1074 + ATOM#3828,
     single-part today).
2. **P/D correctness.** Section 5's V2 verify (`bad=0`) and GSM8K, with the pool on both halves.
3. **D-side sharing.** With DP4 sessions spread over D's ranks, D logs `UMBP pool ...: fetched <k> of <n> blocks from
   peer pools`, and P's push skips those blocks (MoRIIO's push statistics).

**Limits:**
- **`umbp_store` beside `umbp_pool` is eager only.** Lazy is refused at startup: it saves the blocks ATOM's pool is
  about to evict, and UMBP chooses those now.
- **`share_ratio` 1 only on M3 / M3.1.** Whole-block copies for MHA, which a split pool needs, are not implemented.
- **No `ATOM_M3_INDEX_SHARD`.** UMBP grants the block ids, so they cannot be kept in lanes.
- **No cross-tray L1.** A PP pool's parts must all be on one server: P = PP8 across two trays (Kimi-K3.1) needs more
  work.

