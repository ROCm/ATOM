# Independent P/D ranks on a rail-isolated RDMA fabric

Some multi-NIC GPU deployments connect corresponding NICs across hosts:
`ionic_2` on a prefill host can reach `ionic_2` on a decode host, but cannot
reach `ionic_6`. GPU ranks and network rails are separate choices. P2 can
serve D6 if the source GPU's memory is registered on the producer's
`ionic_6` and both transfer endpoints use that rail.

Registering memory on every NIC in one Mooncake engine does not force a
reachable source/destination pair. Mooncake may still select a disconnected
rail. ATOM's opt-in matched-rail mode uses one single-HCA engine per rail.

## Configuration

On both prefill and decode servers, enable automatic discovery before startup:

```bash
export ATOM_MOONCAKE_MATCHED_RAILS=auto
```

Auto mode finds HCAs in the primary GPU-local HCA's numbered name family and
includes those with at least one ACTIVE RDMA port in local sysfs. For example,
a primary `ionic_2` discovers active `ionic_*` HCAs and excludes an unrelated
`mlx5_0`. Families such as `rdmaN` and `mlx5_N` are also supported; neither
the names nor the number of HCAs are hardcoded. Startup logs the resolved list.
A missing/inactive primary, unavailable sysfs, or an unnumbered primary name
produces an actionable error instead of silently disabling matched rails.

To restrict the allowed rails, or use other naming schemes, an explicit list
remains supported:

```bash
export ATOM_MOONCAKE_MATCHED_RAILS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
```

Use `protocol=rdma`. Keep `ib_enable_alternate_hca` disabled. Leave
`ib_device` / `ATOM_MOONCAKE_IB_DEVICE` unset to select each GPU's local
primary HCA automatically, or select a single primary HCA explicitly.
All allowlisted HCAs must exist locally, and the primary must be in the list.
Combining this mode with a multi-HCA primary engine or TCP is rejected.

The same name must identify corresponding, mutually reachable rails across
P/D hosts; matching is by name, not list position. Auto mode discovers local
HCAs and link state, not cross-host reachability or memory-registration support.
It does not infer a mapping between differently named remote HCAs.
The source GPU memory must support registration on every selected
HCA; the feature does not change driver, GID, routing, or DMA-BUF support.

Configure the router to choose P/D ranks independently:

```bash
atomesh launch --pd-disaggregation --backend atom --dp-aware \
  --prefill "$PREFILL_URL" --decode "$DECODE_URL" \
  --atom-pd-rank-mapping-policy none
```

Rank mapping is independent of the chosen routing policy. This transport
feature does not require cache-aware routing.

## Request and memory lifetime

The consumer advertises its single selected HCA with the write request.
The producer selects the matching engine and passes that same engine through
block, slot, staged-index writes, and retries. It never changes a shared
`transfer_engine` while concurrent requests are in flight.

The primary engine is reused. Extra engines are initialized lazily under a
lock and cached for the connector's lifetime, with at most one engine per
allowlisted HCA. Every extra engine registers the same memory ranges,
including slot and index staging buffers, using its own memory keys. This
adds registration and connection resources; it does not allocate another KV
cache. KV buffers must outlive the pool, and replacing live registrations is
rejected.

A failed primary memory registration aborts connector initialization.
If an extra engine fails to initialize or register memory, successfully
registered regions are rolled back. That rail remains failed until restart.
A missing, ambiguous, or unlisted consumer HCA fails the transfer; there is no
fallback to an arbitrary rail. The connector's existing failure notification
and completion/release handling remain in use.

With the environment variable unset, the existing single-engine behavior is
retained, including TCP and explicitly configured multi-HCA engines. Updated
consumers add an optional write-request field that older producers ignore.
A matched-rail producer requires a consumer that advertises a single HCA.

## Validation scope

The original implementation completed all 64 P/D GPU-rank combinations on
two eight-GPU MI355X hosts, including repeated cross-rank requests,
long-prefix reuse with a different D rank, concurrent requests, and a
stream-disconnect recovery check. The same mechanism was also used for
completed C128 agentic runs. These tests establish transport functionality;
they do not establish a throughput improvement from independent pairing.

The pool in this main-branch port was also checked directly with a GPU2 to
GPU6 RDMA byte-integrity probe. The sender reused its primary `ionic_2`
engine and created an `ionic_6` engine from a worker thread. Six writes
(64 B, 64 KiB, and 1 MiB at different offsets) passed full-buffer and
guard-byte checks, with successful deregistration and GPU-buffer cleanup.
This is a pool-level check, not a new end-to-end model run of this port.

CPU regression tests cover concurrent engine creation, registration failure
rollback, peer-HCA validation, request-local engine selection, retries, and
staged-index propagation. The main-branch port retains the current DCP,
FP4 data/scale region, and staging paths.
