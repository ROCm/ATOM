# Mori UMBP as ATOM's KV offload tier (`umbp_store`)

`umbp_store` stores prefix KV in a Mori UMBP standalone server and loads it
back when a later request shares the prefix. The server owns one DRAM pool per
node that every TP rank (and, with DP, every DP rank) shares; ATOM's HBM prefix
cache stays in front of it.

This page is the model-independent how-to. Model walkthroughs:

- [MiniMax-M3-UMBP.md](MiniMax-M3-UMBP.md): MiniMax-M3, aggregated TP4

PD is not supported: a `multi` connector with a P/D connector beside
`umbp_store` is refused at startup.

Design notes are in
[`atom/kv_transfer/offload/README.md`](../atom/kv_transfer/offload/README.md#mori-umbp-offload-umbp_store).

## How it works

```text
ATOM server (TP N)
  scheduler process  ── exists(keys) ───────────────┐   control plane:
  (UMBPStoreOffloadScheduler)                        │   gRPC over a unix socket
                                                     ▼
  worker, one per rank ── put / get ──────►  umbp_standalone_server
  (UMBPStoreOffloadConnector)                 shared DRAM pool (L2), SSD optional
  HBM staging pool, registered  ◄── HIP IPC ── the server copies to / from it
```

- **Keys**: the prompt is cut into chunks (`umbp_store.chunk_tokens`, 256 by
  default) and every chunk gets a chained content hash. Each rank stores its
  own shard of a chunk under its own key:
  `atomumbp1-<namespace>/w<rank>of<world>/<digest>`. The namespace folds in
  the model, the KV layout and the codec, so incompatible bytes never meet.
- **Lookup**: the scheduler asks the server, for every rank, how many leading
  chunks exist, and loads the minimum (a prefix is restorable only if every
  rank holds its shard).
- **Save**: after prefill, each worker packs its new chunks into the staging
  pool and puts them. Chunks the server already holds are skipped (one
  batched `exists` per save window).
- **Load**: a hit is read into the staging pool and unpacked into the KV blocks
  the scheduler allocated, before the request's prefill continues.
- Objects are immutable. Putting a key that exists is a no-op that keeps the
  old bytes, which is why keys carry the content hash.

Supported layouts: dense KV and MiniMax-M3/M3.1 PAGE (K, V, scales and the NSA
index cache). Stateful layouts and DCP > 1 are refused at startup.

Copies between the KV cache and the staging pool are staged on the forward's
stream, as `mooncake_store`'s are; no side copy stream waits on the device for
the forward.

## 1. Requirements

**Mori with UMBP, including ROCm/mori#791 and ROCm/mori#949.** In the ATOM
container:

```bash
python3 -c "import mori.umbp; print('umbp ok')"
python3 -c "import mori.umbp as u; u.UMBPStandaloneProcessConfig().lookup_timeout_ms = 1; print('#949 ok')"
pip show amd_mori_nightly 2>/dev/null | grep Version     # want 1.2.6.dev20261112 or later
UMBP_BIN=$(python3 -c 'import mori.umbp,pathlib;print(pathlib.Path(mori.umbp.__file__).resolve().parents[1]/"umbp_standalone_server")')
ls -l "$UMBP_BIN"
```

ROCm/mori#791 ("a lookup renews the keys it finds") makes `exists` refresh
the LRU. Without it, a prefix that keeps hitting in HBM is never read from
UMBP, so it ages out first. A conversation's shared head chunk is hit
hardest, and a lookup that misses chunk 0 misses the whole prefix.
ROCm/mori#949 adds the lookup deadline (`lookup_timeout_ms`) that
`umbp_store.lookup_timeout_ms` sets; with an older Mori the connector does not
start. Nightly wheels name only a date; the nightly build of a given date is
cut from Mori main at 20:00 UTC that day. The ATOM image installs the newest
Mori nightly when it is built, so check the version in the image you run.

**Building Mori from source** (when the container's Mori is older):

```bash
git clone https://github.com/ROCm/mori && cd mori        # then check out a known commit
git submodule update --init --recursive
BUILD_UMBP=ON MORI_GPU_ARCHS=gfx950 pip wheel --no-deps --no-build-isolation -w /work/mori_wheel .
pip install --no-deps --target /work/mori_main /work/mori_wheel/*.whl
export PYTHONPATH=/work/mori_main:$PYTHONPATH            # for the server and ATOM
python3 -c "import mori, mori.umbp; print(mori.__file__)"   # must point into /work/mori_main
```

Set `MORI_GPU_ARCHS` to your GPU (`gfx950` for MI355X, `gfx1250` for MI450/MI455).
`setup.py` silently drops UMBP when gRPC or Protobuf development files are
missing, so always check that `import mori.umbp` works and that
`umbp_standalone_server` comes from the new tree.

## 2. Start the server

One server per node, in the same container as the ATOM server it serves,
started **before** ATOM and stopped after it.

```bash
export UMBP_ADDR=unix:///tmp/umbp/server.sock
mkdir -p /tmp/umbp && rm -f /tmp/umbp/server.sock*

ROCR_VISIBLE_DEVICES=0,1,4,5 \
UMBP_DRAM_CAPACITY=$((1024 * (1 << 30))) \
UMBP_DRAM_NUMA_NODE=0,1 \
UMBP_DRAM_USE_HUGEPAGES=0 \
UMBP_DRAM_PREFAULT_THREADS=16 \
UMBP_SSD_ENABLED=0 \
UMBP_STANDALONE_METRICS_PORT=9877 \
MORI_UMBP_LOG_LEVEL=INFO \
  nohup "$UMBP_BIN" "$UMBP_ADDR" > umbp_server.log 2>&1 &
```

| Setting | What to set |
|---|---|
| `ROCR_VISIBLE_DEVICES` | **The same list, in the same order, as the ATOM server.** The server opens each worker's staging pool through HIP IPC by device ordinal. |
| `UMBP_DRAM_CAPACITY` | Bytes for the whole node. The pool is shared, so size it for the node, not per rank (e.g. what you would have given an LMCache CPU tier per rank, times the TP size). Only the server's setting counts; client-side capacity settings are ignored. |
| `UMBP_DRAM_NUMA_NODE` | The NUMA nodes of the GPUs in use, so the pool is spread over both sockets when the ranks are. |
| `UMBP_DRAM_PREFAULT_THREADS` | Prefaulting a large pool takes minutes; more threads shorten it. |
| `UMBP_SSD_*` | Optional L3 tier: `UMBP_DISTRIBUTED_MEDIUM=SSD UMBP_SSD_ENABLED=1 UMBP_SSD_DIR=… UMBP_SSD_CAPACITY=…`. Not validated with ATOM yet. |
| `UMBP_STANDALONE_METRICS_PORT` | Prometheus-style server metrics (resident bytes, commits, evictions, reads). `0` disables. |

These `UMBP_*` variables belong to the Mori server; ATOM adds none.

**Wait until the server is ready.** The socket appears long before the pool is
prefaulted and the host memory registered with the GPUs. Connect with a real
client:

```bash
until python3 - <<PY 2>/dev/null
import mori.umbp as u
c = u.UMBPConfig(); p = u.UMBPStandaloneProcessConfig()
p.address = "$UMBP_ADDR"; p.auto_start = False; p.startup_timeout_ms = 10000
c.standalone_process = p
u.UMBPClient(c)
PY
do sleep 5; done; echo "umbp ready"
```

Start ATOM only after this loop succeeds. The server may accept clients while
it is still registering the rest of its host memory in the background, and
answers puts that arrive before that finishes with False. Each worker's first
put is its startup probe: it is asked again until
`umbp_store.startup_timeout_ms` has passed, and if the server still does not
store it the engine does not start. Saves that fail later are retried by the
scheduler. Wait for the server log to go quiet (or give it a minute after
`umbp ready`) before sending load.

Restart the server whenever you restart ATOM for a new measurement, so each run
starts from an empty pool.

## 3. Point ATOM at it

```bash
  --kv-transfer-config '{"kv_connector":"umbp_store","kv_role":"offload",
    "kv_connector_extra_config":{"umbp_store.address":"unix:///tmp/umbp/server.sock"}}'
```

Offload only, on an aggregated instance: a `multi` connector with a P/D
connector beside `umbp_store` is refused at startup.

Keep the HBM prefix cache on (`--enable-prefix-caching`); UMBP sits behind it.
No `PYTHONHASHSEED` is needed: the object keys are blake2b chain hashes of the
prompt's tokens, and the HBM prefix cache hashes with xxhash.

| Key | Default | Meaning |
|-----|:-------:|---------|
| `umbp_store.address` | required | The server's address, e.g. `unix:///tmp/umbp/server.sock`. |
| `umbp_store.startup_timeout_ms` | 10000 | How long a client waits for the server when it connects, and how long a worker's startup probe asks again while the server answers its put with False. |
| `umbp_store.lookup_timeout_ms` | 1000 | Deadline of one lookup RPC. Past it every key counts as absent, the request prefills, and the scheduler stops asking the server for a while (at least 10 s). Must be > 0: lookups run on the scheduler thread. |
| `umbp_store.chunk_tokens` | 256 | Tokens per object; a multiple of the KV block size. Also the routing granularity. |
| `umbp_store.load_pool_mib` | 1024 | HBM load staging per worker, MiB. |
| `umbp_store.save_pool_mib` | 256 | HBM save staging per worker, MiB. |
| `umbp_store.save_abandon_timeout_s` | 300 | Seconds before the engine reclaims the source blocks of a save that never reported. |
| `max_pending_saves` | unbounded | Optional cap on saves in flight across requests (one per request at most either way); unprefixed, as for `mooncake_store`. `OFFLOAD_MAX_PENDING_SAVES` sets it too. |

An unknown `umbp_store.*` key is an error.

Like `mooncake_store`, `umbp_store` loads only hits of at least 512 tokens
(two chunks); the global default for other tiers is 8192. The existing
`OFFLOAD_MIN_LOAD_TOKENS` setting overrides it.

One server serves one node. A tensor-parallel group over several nodes is
refused at startup, and with DP over several nodes each node's ranks reach
only their own server, so shared-prefix DP routing (`--dp-load-balance
share_cache` / `prefix_spread`) is refused across nodes; on one node it is
supported.

## 4. Check that it works

- **Startup**: every TP rank selects the `umbp_store` connector, registers its
  staging pool and writes one probe object. The warning
  `UMBP startup probe: UMBP cannot remove by key; the probe object … stays on the server until it is evicted`
  is expected. The server log shows the clients registering.
- **Traffic**: each worker periodically logs
  `[OFFLOAD-STORE-STATS] rank=… puts=… put_keys=… save_exists=… save_put_chunks=… save_held_chunks=… gets=… get_keys=… failures=…`.
  Under a prefix-heavy workload `puts` and `gets` grow. `save_held_chunks` counts
  chunks a save found already stored and skipped. `failures` should stay empty
  once the server is fully ready.
- **Lookups**: the scheduler logs
  `[OFFLOAD-LOOKUP-STATS] lookups=… uneven_lookups=… stranded_objects=…`.
  `uneven_lookups` counts lookups where the ranks held different prefix
  lengths. It stays low in a healthy run; a high value means the ranks' shards
  are being evicted independently (see Limitations).
- **Per-save timing**: `OFFLOAD_PROFILE=1` (an existing offload knob) adds an
  `[OFFLOAD-SAVE-PROF]` line per save, with `held_chunks` and `put_chunks`,
  `windows` (the windows the save reached, one lookup each) and `segments`
  (the runs of missing chunks it packed and put), and `exists_ms`, `pack_ms`,
  `put_ms`.
- **Read-back correctness**: send a few long greedy prompts with facts spread
  through them, push them out of the HBM prefix cache with unrelated long
  requests, then resend them. Confirm that the log shows a load for them and
  that the answers about every fact are still right. Run accuracy checks
  without `--spec-decode-acceptance-length` / `--spec-decode-acceptance-rate`;
  with those flags every output is synthetic.

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `umbp_store` refuses to start, "server is not ready" | Wrong address, or the server is still prefaulting | Check `umbp_store.address`; wait for the readiness loop |
| `UMBP startup probe: ... did not store` | The server was still registering host memory past `umbp_store.startup_timeout_ms`, or its pool is full | Start ATOM after the readiness loop and a quiet server log; raise `umbp_store.startup_timeout_ms` |
| `put_failed` right after startup | The server is still registering host memory with the GPUs | Wait for the server to finish; the scheduler retries failed saves |
| `took ... ms to answer a lookup` warnings | The server is overloaded or stalled | Lookups to it pause and those requests prefill; check the server |
| Hits fall off over a long run although the pool is large | Mori without #791: hot prefixes age out | Use a Mori build that includes ROCm/mori#791 |
| Server cannot open a worker's buffer | GPU ordinals differ between the server and ATOM | Same `ROCR_VISIBLE_DEVICES` for both |

## Limitations

These are open with the Mori team; the connector works around each of them:

- **No grouped eviction.** UMBP evicts each rank's object of a chunk on its
  own, so a chunk can survive on some ranks only. Mooncake Store groups them
  (`group_ids`) and evicts the group whole. `umbp_store` relies on the lookup
  renewing every rank's key at once (#791) and skips held chunks on save.
- **No read lease.** A key found by `exists` can be evicted before the load
  reads it. The load then fails cleanly and the request recomputes.
- **No per-key remove, no per-key error codes.** A failed RPC looks like a
  miss, and stale objects can only age out.
- **`deregister_memory` releases every region of a client.** The connector
  registers one staging region per worker and deregisters it only at close.
- **Server restarts lose the pool.** There is no reconnect; restart ATOM with
  the server.
- **No deadline on put and get.** Lookups are bounded by
  `umbp_store.lookup_timeout_ms`, but Mori has no deadline for
  `batch_put_from_ptr` / `batch_get_into_ptr`: a server that stops answering
  stalls the worker's save and load threads, and with them a preemption that
  waits for its saves and the engine's shutdown. A deadline could not free the
  staging slots anyway, since the server may still be copying into them.
  Stop ATOM before the server.

## Stop

```bash
pkill -f 'atom.entrypoints.openai_server' || true
pkill -f umbp_standalone_server || true
```
