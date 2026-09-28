# KV Connectors, MultiConnector, and P/D Disaggregation

This document explains how KV connectors relate to the inference engine, the
structure of the `multi` connector (run P/D disaggregation **and** LMCache
offload at once), how prefilled KV blocks are safely freed when both a P/D send
and an offload save read them, and an end-to-end runbook for bringing up and
verifying P/D disaggregation.

Code: `atom/kv_transfer/disaggregation/` (`base.py`, `factory.py`,
`multi/multi_connector.py`), engine side `atom/model_engine/scheduler.py`,
`model_runner.py`, `utils/forward_context.py`.

---

## 1. Connector vs Inference Engine

- **Inference engine** computes tokens and **owns** the KV cache in HBM. It is
  the main body of the system and runs fine with no connector at all.
- **KV connector** does not compute; it only **moves** KV bytes — to a remote
  decode node (P/D disaggregation) or to CPU/NVMe (offload), and back when
  needed. It is pluggable; the factory selects exactly one by the
  `kv_connector` config name.

They are decoupled by a narrow set of hooks in `base.py`. Each step the engine
calls the connector at fixed points; the connector reads/writes the HBM KV
tensors it was handed, then reports "transfer finished" so the engine can free
blocks / wake sequences.

```
                        +------------------+
                        |    API Server    |
                        +------------------+
                                 |
                                 v
+=================================================+
| INFERENCE ENGINE - computes tokens, owns KV     |
| EngineCore -> Scheduler -> ModelRunner (GPU)    |
| BlockManager -> [ HBM KV Cache = real KV bytes ]|
+=================================================+
                 ^               |
                 |   hooks (engine calls connector)
                 |               v
+=================================================+
| KV CONNECTOR - moves KV bytes, no compute       |
| KVConnectorBase (7 hooks) -> pick ONE:          |
| moriio | mooncake | lmcache_offload | multi     |
+=================================================+
            |                          |
            v                          v
     +-------------+            +-------------+
     | Decode node |            |  CPU / NVMe |
     +-------------+            +-------------+
       (P/D xfer)                (LMCache)
```

### The eight hooks (`base.py`)

| Side | Hook | Purpose |
|---|---|---|
| scheduler | `get_num_new_matched_tokens(seq)` | Does this seq have reusable KV (remote already computed / cached on CPU)? If so, park it and wait for the transfer. |
| scheduler | `update_state_after_alloc(seq)` | After HBM blocks are allocated, record the "to recv / to save" intent. |
| scheduler | `build_connector_meta()` | Pack this step's transfer requests into a `meta`. |
| scheduler | `request_finished(seq)` | Clean up when a request finishes. |
| scheduler | `source_blocks_released(seq)` | The finished request's HBM is back in the pool — drop the state whose lifetime was those blocks, or explicitly do nothing if none is owned. |
| worker | `register_kv_caches(tensors)` | Once at init: hand the HBM KV tensor addresses to the connector (it reads/writes through these). |
| worker | `start_load_kv(meta)` | Kick off the async transfers (load in / save out). |
| worker | `get_finished()` | Report which req IDs finished sending / recving / saving / failed. |

> Offload-specific scheduler methods (`should_park_*`, `save_finished`,
> `load_failed`, ...) are all guarded with `hasattr` on the engine side, so a
> connector that does not implement them works fine.

---

## 2. Two layers: scheduler vs worker

The connector is split into two objects living in two processes:
- **scheduler-side** (main / EngineCore process, no GPU) — built by
  `get_kvconnector("scheduler")`. Decides / bookkeeps / packs / cleans up.
- **worker-side** (GPU TP-worker process, one per rank, **holds HBM KV
  pointers**) — built by `get_kvconnector("worker")`. Actually moves bytes.

`meta` flows down (scheduler → workers), `KVConnectorOutput` flows up
(workers → scheduler, merged across TP ranks by `KVOutputAggregator`). Left
column = what the engine does that tick; right column = the connector hook fired
on the same tick.

```
+=====================================================================+
| SCHEDULER LAYER   -   main process  (no GPU)                        |
| Scheduler  +  scheduler-side connector object                       |
+=====================================================================+
| WORK / SCHEDULE                | CONNECTOR HOOKS  (*)               |
|                                                                     |
| [1] Scheduler.schedule()       |                                    |
|       reuse remote / CPU KV ?  | * get_num_new_matched_tokens       |
|       alloc HBM blocks         |                                    |
|       record transfer intent   | * update_state_after_alloc         |
|       snapshot this step       | * build_connector_meta  => meta    |
|                                                                     |
+=====================================================================+

        |  meta   (IPC:  scheduler  -->  workers)
        v

+=====================================================================+
| WORKER LAYER   -   GPU TP-worker process  (per rank)                |
| ModelRunner  +  worker-side connector  (holds HBM KV)               |
+=====================================================================+
| GPU WORK                       | CONNECTOR HOOKS  (*)               |
|                                                                     |
| (init, once)                   | * register_kv_caches(tensors)      |
|                                                                     |
| [2] receive meta               | * start_load_kv(meta)              |
|       async copy HBM <-> ext   |   (load / save)                    |
|                                                                     |
| [3] GPU forward prefill/dec    |   compute (no connector)           |
|       connector copy parallel  |   (side stream)                    |
|                                                                     |
| [4] poll transfer status       | * get_finished()                   |
|                                |   => {sent,recv,saved,fail}        |
|                                                                     |
+=====================================================================+

        ^  KVConnectorOutput
        |  (IPC:  workers --> scheduler,  merged across TP)

+=====================================================================+
| SCHEDULER LAYER   -   back in main process                          |
+=====================================================================+
| APPLY RESULTS                  | CONNECTOR HOOKS  (*)               |
|                                                                     |
| [5] recv done -> wake seq      |                                    |
|         -> start its decode    |                                    |
|     sent/saved -> check free   |                                    |
|     finished                   | * request_finished(seq)            |
|     blocks actually freed      | * source_blocks_released(seq)      |
|                                                                     |
|     loop  -->  back to [1]     |                                    |
+=====================================================================+
```

The scheduler is the clock. The worker only "moves + reports"; block freeing and
sequence waking are decided by the scheduler in `[5]` based on what was reported.
`[3]` GPU forward and the worker-side copy run on different streams in parallel,
so transfers do not block compute.

---

## 3. MultiConnector structure

To the engine, `multi` is just one connector (implements the same `base.py`
interface; the engine does not know it is multi). Internally it holds a list of
real sub-connectors and fans out / merges per hook. Three classes:
`MultiConnector` (worker), `MultiConnectorScheduler` (scheduler),
`MultiConnectorMetadata`.

```
            engine  -->  base interface  (unchanged, doesn't know it's multi)
                                |
                                v
+=============================================================+
| MultiConnectorScheduler   (SCHEDULER layer)                 |
| one connector, holds a list of sub-connectors               |
+=============================================================+
| HOOK                       | MERGE POLICY                   |
|                                                             |
| get_num_new_matched        | first-hit-wins (1 sub owns it) |
| update_state_after_alloc   | fan-out to ALL subs            |
| build_connector_meta       | pack -> MultiConnectorMetadata |
| request_finished           | fan-out to ALL subs            |
| source_blocks_released     | fan-out to ALL subs            |
+=============================================================+
            |                                       |
            v                                       v
   +--------------------------+      +------------------------------+
   | mooncake (sched sub)     |      | lmcache_offload (sched sub)  |
   | kv_role=kv_producer      |      | kv_role=offload              |
   +--------------------------+      +------------------------------+

   build_connector_meta()  =>  MultiConnectorMetadata(
                                  metas = [ meta_mooncake , meta_offload ] )
                                |   IPC down  (scheduler -> workers)
                                v
+=============================================================+
| MultiConnector            (WORKER layer, per TP rank)       |
| holds worker-side sub-connectors (touch HBM KV)             |
+=============================================================+
| HOOK                       | MERGE POLICY                   |
|                                                             |
| register_kv_caches         | fan-out to ALL subs            |
| start_load_kv(meta)        | route metas[i] -> subs[i]      |
| get_finished()             | UNION independent completions  |
+=============================================================+
            |                                       |
            v                                       v
   +--------------------------+      +------------------------------+
   | mooncake (worker sub)    |      | lmcache_offload (worker sub) |
   |   -> Decode node (P/D)   |      |   -> CPU / NVMe (LMCache)    |
   +--------------------------+      +------------------------------+
```

**How subs are built (no recursion):** `_build_subconnectors` does
`copy.copy(config)`, swaps in the sub `kv_transfer_config`, and goes through the
factory again. Each sub dict carries its own `kv_connector`
(moriio / mooncake / lmcache_offload), never `multi`, so there is no recursion.
List index is strictly aligned: `metas[i] <-> subs[i]`, which is how the worker
de-multiplexes.

### Priority when more than one sub could match

`get_num_new_matched_tokens` is **first-hit-wins by `connectors` list order**:

```python
result = (0, False)
for c in self._connectors:            # iterate in config order
    toks, needs_load = c.get_num_new_matched_tokens(seq)
    if result[0] == 0 and toks > 0:   # first sub returning >0 owns the request
        result = (toks, needs_load)
return result
```

- Priority = list order. Reorder `connectors` to change it.
- **In the actual producer topology, offload always wins regardless of order:**
  `mooncake (kv_producer)` only returns `>0` when
  `seq.kv_transfer_params["do_remote_prefill"]` is set — and that flag is a
  *consumer-side* flag (set by the proxy on the decode node), never on the
  producer. So `mooncake-producer` always returns `(0, False)`, and only offload
  can match.
- Order only matters on a node where two subs could match the same request
  (e.g. a hypothetical `multi=[mooncake-consumer, offload]` on the decode node).
  Then prefer putting cheaper local offload first.

> Caveat: the loop calls `get_num_new_matched_tokens` on every sub (only the
> first non-zero is returned). For offload this method has side effects (records
> `_load_specs`, pins LMCache, appends `_lookup_in_step`). Harmless in the
> producer topology (mooncake no-ops, offload is the winner). If you ever run a
> node where both could match the same request, the non-winning sub still leaves
> state; vLLM's MultiConnector avoids this with a `_requests_to_connector` owner
> map — the current ATOM version does not track an owner.

---

## 4. Block-free correctness: every owner must let go

On a producer node a finished request's KV blocks must survive while **either**
the P/D send **or** offload still reads them — including chunks that are
computed but whose save has not been dispatched yet. Both only *read* the
blocks, so there is no write conflict; the hazard is freeing them under a live
reader.

### Report each operation as it finishes

`MultiConnector.get_finished()` normalizes each sub's output and unions the
completion sets. Send and save progress are forwarded **immediately and
independently** — there is no worker-side pairing state.

This is load-bearing, not a simplification: the chunked offload scheduler keeps
at most one save per request in flight, so chunk *k*'s `finished_saving` is what
lets chunk *k+1* be dispatched. Withholding it until the P/D send completes —
and mooncake only sends after the **last** chunk — deadlocks the two against
each other, and every chunk after the first is silently dropped.

### The scheduler asks the connectors, once

`Scheduler` parks a finished request in `deferred_free_blocks` whenever
`should_defer_free(seq)` is true, and re-asks on every completion report:

```python
# scheduler.py  _maybe_release_deferred
if (
    seq.id not in self.deferred_free_blocks
    or getattr(seq, "_awaiting_aborted_load_cleanup", False)
    or self._connector_should_defer_free(seq)
):
    return
```

There is no producer special case. `MultiConnectorScheduler.should_defer_free`
ORs its subs, so each owner vetoes the free on its own:

| Sub | Claims the source while… | Claim taken | Claim dropped |
|---|---|---|---|
| mooncake / moriio | an advertised send is outstanding | `request_finished` | `send_finished(req_id)` |
| offload | a save is in flight, or a computed range is still undispatched | its own bookkeeping | its own completions |

`Scheduler._update_from_kv_xfer_finished` drops the send claim first
(`_connector_send_finished`) and then re-asks, so `finished_sending` and
`finished_saving` both funnel through the same one-line check, in any order.

| State of a finished producer request | Release decision |
|---|---|
| Saves complete, send still pending | retain — the send still claims it |
| Send complete, save still in flight | retain — offload still claims it |
| Send complete, suffix save not dispatched | retain — offload still claims it |
| Nobody claims it | free |

The claim is taken in `request_finished` — the call that hands the peer this
request's block addresses — and not at allocation. Before it the allocation is
nobody's but ours, and `Scheduler._is_preemptable` is the negation of this same
predicate: claiming at alloc would pin every running request on a prefill node
and leave `_preempt_one_running` nothing to choose under memory pressure.

An aborted request is never advertised, so it is never claimed — which is why
`postprocess` settles `seq.leave_reason` *before* the single
`request_finished` call that reads it, and why `Scheduler` then retires the
claim itself (`_connector_send_finished`) rather than trusting each backend to
re-derive "this will never send". Nothing else could: `send_finished` is driven
by a completion report that is not coming.

> **Scope:** completion reports name a *request*, not a connector, so two send
> sub-connectors would be indistinguishable in `finished_sending` and the first
> report would retire both claims. `_build_subconnectors` therefore refuses more
> than one producer sub, and `_offload_subconfig` refuses more than one offload
> sub. Supporting either needs per-child completion identities first.

### PP quorum and idle suffix saves

`KVOutputAggregator` aggregates across TP ranks. Under PP the head forwards
mooncake's `finished_sending` straight to the scheduler — it already represents
PP-wide send completion — while offload load/save operations go through
`PPKVAggregator`, which requires a terminal from every stage, per save
generation. The head keeps no pairing state of its own.

After the last forward batch there can still be computed KV left to save.
`Scheduler.is_finished()` stays false while blocks are deferred, so
`_head_busy_loop` never reaches its own idle-drain branch; `_pp_head_step`
calls `_advance_idle_kv_transfer()` when it cannot build a batch, which
dispatches the suffix through the shared drain throttle. A 192-token prompt
computed in 64-token steps saves `[0,64)` and `[64,128)` during prefill, then
`[128,192)` while idle.

`_reconcile_stalled_deferred_saves` abandons a save whose report never arrives,
then re-asks `should_defer_free`: if the send still claims the source, the
blocks stay put and `finished_sending` remains their only releaser. The abandon
happens once (`seq._save_abandoned`); the release attempt repeats every
interval, so a late report still collects the blocks.

Abandonment also retires the PP aggregator's save tallies. It remembers the
most recent 4096 abandoned native request IDs (the TP aggregator uses the same
default bound for its terminal history), so late save, store-terminal and
source-safe reports cannot recreate those tallies. The terminal is recorded
even if no stage has reported yet. Load tallies and independent state-store
events keep their own quorum; abandoning a save does not cancel a load.

### The other half of `request_finished`

`request_finished` runs while `should_defer_free` is still true — that is the
whole point of the park — so a connector cannot use it to drop state whose
lifetime is the *blocks* rather than the request. The offload schedulers keep
exactly that: `_save_tracker` holds the `Sequence` and is what the save loop
iterates, reading `seq.block_table` out of it.

`source_blocks_released(seq)` is the terminal for it, called by
`_maybe_release_deferred` right after `block_manager.deallocate`. It is
deliberately not `request_finished` a second time: that call also *takes* the
send claim, so re-invoking it would re-arm the claim the release just cleared.
`KVConnectorSchedulerBase` requires explicit implementations of
`should_defer_free`, `send_finished`, and `source_blocks_released`; omitting
one prevents a backend from being instantiated. P/D backends implement
`source_blocks_released` as a no-op, while offload backends implement
`send_finished` as a no-op. `MultiConnectorScheduler` fans out the calls.
The vLLM plugin's `_collect_releases` makes the same split at its own release
point.

### Regression coverage

`tests/test_multi_connector.py` — send and save reported independently in
either order, save operation IDs preserved, >1 producer refused, ownership
ORed across subs.
`tests/test_lmcache_offload_connector.py` — chunks 2..N dispatch with no send,
each dispatched save getting a full retention window, and a deferred request's
save state being dropped at `source_blocks_released` (identity-guarded).
`tests/test_pd_source_claim.py` — the claim's timing on both P/D backends:
taken at publication, not at alloc; dropped on `send_finished`; never taken for
a local-only, aborted, or consumer-side request.
`tests/test_scheduler.py` — the joint barrier in both completion orders, the
stall reclaimer retrying rather than giving up under a pending send, a running
producer request staying preemptable, an aborted producer request getting its
blocks back (both because it is never claimed and because the scheduler retires
a claim taken anyway), and the release reaching `source_blocks_released` without
a second `request_finished`.
`tests/test_pp_kv_status.py` — per-operation PP quorum, send not held for it.
`tests/test_pd_pp.py`, `tests/test_kv_drain_liveness.py` — idle suffix dispatch
through the shared drain throttle, and engine quiescence.

### HBM-hit + send + offload at the same time

These are three independent things; an HBM prefix-cache hit only skips the
redundant offload **reload**, it does not affect the send or the save:

| Action | Owner | Effect of HBM hit |
|---|---|---|
| send to decode (P/D) | mooncake sub | none — sends the HBM blocks regardless |
| offload save to CPU | offload sub | none — saves whatever is not yet in LMCache |
| reload from offload | offload sub | correctly skipped (`need = lmcache_hit - num_cached <= 0`) |

---

## 5. P/D disaggregation runbook (recorded run)

The results below were recorded on MI325X, container `yhl_kvoff_009`,
`Llama-3.1-8B-Instruct`, mooncake `protocol=tcp`. They predate the scheduler-owned
release checks described in section 4; use that section's regression coverage
for the current completion-ordering behavior.

### 5.1 Prerequisites / gotchas

- **mori is unusable on this host** (container ionic RoCE driver ABI 1 vs
  libibverbs needs 4 -> MORI finds no usable RDMA device). Use mooncake with
  `"protocol":"tcp"`, which needs no RDMA.
- Container deps: `mooncake`, `msgpack`, `msgspec`, `quart`; offload also needs
  `lmcache`. Verify:
  ```bash
  python -c "import mooncake,lmcache,msgpack,msgspec,quart,zmq; print('deps ok')"
  ```
- `--network host` ⇒ ports are host-global. Rapid restarts hit
  `Address already in use` (ROUTER socket TIME_WAIT); use a fresh
  `handshake_port` or wait ~60s.
- Shared GPU box: pick 0%-VRAM cards via `rocm-smi --showmemuse`. Repeated `-9`
  kills leak `psm_*` segments in host-shared `/dev/shm` and can exhaust aiter's
  broadcast pool (NCCL `Failed to CUDA calloc`). **Do not blindly delete
  `/dev/shm` — it is shared with other users' jobs.**

### 5.2 Topology

```
Client --> Proxy(:10003, discovery :36367)
              |--> Prefill node (kv_producer)  GPU a  API :8000  handshake 6700
              '--> Decode  node (kv_consumer)  GPU b  API :8001  handshake 6800
```
Plain P/D: both ends use `kv_connector:"mooncake"`. P/D + offload: switch only
the **prefill** node to `kv_connector:"multi"`; the decode node stays a plain
mooncake consumer.

### 5.3 Launch

Proxy:
```bash
python -m atom.kv_transfer.disaggregation.proxy --port 10003
```

Prefill / producer (plain P/D):
```bash
HIP_VISIBLE_DEVICES=5 python -m atom.entrypoints.openai_server \
  --model /shared/data/amd_int/models/Llama-3.1-8B-Instruct \
  --kv_cache_dtype fp8 --block-size 16 -tp 1 --trust-remote-code \
  --max-model-len 8192 --gpu-memory-utilization 0.40 \
  --port 8009 --server-port 8000 \
  --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","protocol":"tcp","proxy_ip":"127.0.0.1","proxy_ping_port":36367,"http_port":8000,"handshake_port":6700}'
```

Prefill / producer (P/D + offload via multi) — add LMCache env +
`--enable_prefix_caching`, swap the connector config:
```bash
HIP_VISIBLE_DEVICES=5 \
LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=16.0 LMCACHE_CHUNK_SIZE=256 LMCACHE_USE_GDS=False PYTHONHASHSEED=0 \
python -m atom.entrypoints.openai_server \
  --model /shared/data/amd_int/models/Llama-3.1-8B-Instruct \
  --kv_cache_dtype fp8 --block-size 16 -tp 1 --trust-remote-code \
  --enable_prefix_caching --max-model-len 8192 --gpu-memory-utilization 0.40 \
  --port 8009 --server-port 8000 \
  --kv-transfer-config '{"kv_connector":"multi","connectors":[{"kv_connector":"mooncake","kv_role":"kv_producer","protocol":"tcp","proxy_ip":"127.0.0.1","proxy_ping_port":36367,"http_port":8000,"handshake_port":6700},{"kv_connector":"lmcache_offload","kv_role":"offload"}]}'
```

Decode / consumer (always plain mooncake):
```bash
HIP_VISIBLE_DEVICES=6 python -m atom.entrypoints.openai_server \
  --model /shared/data/amd_int/models/Llama-3.1-8B-Instruct \
  --kv_cache_dtype fp8 --block-size 16 -tp 1 --trust-remote-code \
  --max-model-len 8192 --gpu-memory-utilization 0.40 \
  --port 8019 --server-port 8001 \
  --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","protocol":"tcp","proxy_ip":"127.0.0.1","proxy_ping_port":36367,"http_port":8001,"handshake_port":6800}'
```

> Config-key note: moriio uses `http_prt`; **mooncake uses `http_port`** plus
> `handshake_port` (producer/consumer must differ on a single host: 6700 vs
> 6800). `http_port` must equal `--server-port`.

### 5.4 Readiness (do not `tail -f` forever)

Poll `until curl /v1/models` with a 6-minute cap and fail-fast on
`Traceback|HIP out of memory|proc died|NCCL error|Failed to CUDA|Address already in use`.

### 5.5 End-to-end test (send only to the proxy)

```bash
curl -s http://127.0.0.1:10003/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```
For an offload save, use a prompt >= chunk_size (256 tokens).

### 5.6 How to verify it works

**P/D transfer happened:**
- response: `kv_transfer_params.do_remote_prefill == true`
- producer log: `Received write_request`, `_execute_transfer`,
  `All 1 consumers served`, `get_finished: sending={0}`
- consumer log: `Queued req ... for remote KV recv`, `Sending write_request ...`,
  `Write-done received ... done_recving now: {0}`
- output is coherent (greedy temp=0 can be diffed token-for-token vs single-node)

**offload save happened (P/D + multi):**
- producer log: `LMCache ... Stored <N> tokens` (needs prompt >= 256). Measured:
  2509-token prompt -> `Stored 2304` (= chunk_floor(2509)), concurrent with
  `All 1 consumers served`.

**offload reload (optional, needs HBM eviction first):**
- a second identical request right away will NOT reload (prefix still in HBM;
  `Retrieved=0` is correct). Forcing a real reload needs evicting HBM (hard on a
  256 GB card with an 8B model — use MiniMax + low util + long context and
  `scripts/offload_ttft_micro.py`). Marker: `LMCache ... Retrieved`.

**block-free diagnostics:**
- a clean run shows 0 hard errors:
  ```bash
  grep -acE "Traceback|assert|use.after.free|MEMORY_VIOLATION|Memory access fault" <prefill log>
  # ignore the benign tokenizer warning that contains the word "corrupt"
  ```
- An error-free log alone does not establish completion-ordering safety; the
  lifecycle regressions in section 4 check when blocks are retained and freed.

### 5.7 Cleanup

```bash
pkill -9 -f "atom.entrypoints.openai_server|disaggregation.proxy"
rocm-smi --showmemuse | grep -E "GPU\[(5|6)\].*VRAM"   # confirm VRAM freed
```
Do not delete `psm_*` in host-shared `/dev/shm` (would break other containers).

### 5.8 Validated results (8B / mooncake-tcp)

| Item | Result |
|---|---|
| plain mooncake P/D | OK — end-to-end, log handshake closed, coherent output |
| MultiConnector as P/D producer | OK — P/D transfer works through multi |
| producer offload save (via multi) | OK — `Stored 2304`, concurrent with P/D send |
| HBM-hit second request | OK — send proceeds, reload correctly skipped, no re-store |
| block release diagnostics | Recorded run: 0 errors / 0 asserts |
| unit tests at the time of the run | `tests/test_multi_connector.py` 16/16; current coverage is listed in section 4 |

> Not separately isolated: the exact "prefix in HBM but not in LMCache" case of
> HBM-hit + send + a NEW save (mechanism supports it, see §4); a real reload demo
> needs tight HBM (hard on this host — use MiniMax).
