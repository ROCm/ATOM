# Packed metadata publication

Set `ATOM_H2D_BACKEND=packed` before starting the runner to combine small
metadata uploads. The default, `direct`, copies each member separately.
Packing is selected once for eligible groups; noncontiguous bindings use direct
copies. A single active member also uses a direct copy to avoid an extra kernel.
It neither borrows the packed arena nor releases an earlier packed read.
A producer group fully covered by an eligible larger group keeps checked
direct copies for standalone callers, without a duplicate packed arena.
Producers fill the final group's counts directly; compression plans use
`set_count(buffer, count)` without accessing buffer binding internals.

## Consumer boundaries

| Group | Data | First GPU consumer |
| --- | --- | --- |
| `token_inputs` | Sampling parameters, padded query prefix, input IDs, deferred source indices and speculative verification indices | Token assembly, then draft ID gather |
| `prefill_inputs` | Prefill attention mirrors and ordinary positions | Attention preparation |
| `mha_decode` | Slots, context lengths, block tables, KV prefix and positions | KV-index generation |
| `v41_metadata` | Compression/write plans, state slots, positions, token batch IDs, visibility and changed block tables | Step indptr generation |

V4.1 ordinary DSpark decode uses two H2Ds and two scatters: token inputs and
V4.1 metadata. All graph padding is included. Other attention builders, MRoPE
and TBO can require additional groups at their own consumer boundaries.
The shared table state records publication only after the upload succeeds.

Engram tentative preparation writes cursor candidates together with its
snapshot, preserving every accepted prefix. Committed in-place cursor updates
remain ordered after the snapshot. The sampler corrects greedy rows without
reading a GPU mask back to the CPU. Uniform top-k/top-p filters stay as CPU
scalars through ordinary and speculative sampling; only per-request filters
are uploaded. This preserves AITER scalar dispatch without a GPU readback.
The runner restores an implicit CPU default after GPU initialization to avoid
per-call DeviceContext dispatch. MRoPE decode planes use the running token
width as their axis stride before their first publication, for every backend.
V4 TBO consumers retain views of their own ubatch buffers; CPU query lengths
also supply the maximum query width without a device reduction or readback.

## Lifetime and publication contract

Each existing runner slot owns its pinned sources, packed arenas and completion
event. `begin()` waits for source reuse before any producer writes and starts
one forward epoch. `finish()` records completion after preparation; it does not
wait for GPU execution. PP rotates the existing slots. Late TBO preparation
resumes the same epoch and records completion again after its uploads.

Buffers are declared with a group and count unit (`rows`, `elements` or `bytes`)
at allocation. Initialization binds their fixed source/destination storage and
checks overlaps. Producer counts include semantic padding. `None` omits a
member; zero is an explicit empty publication. A group validates all counts,
owner thread/device/stream and duplicate publications before submitting work.
Sampling, input IDs, query prefixes, speculative indices and common attention
producers declare their source groups with `@h2d_producer("group", ...)`.
The shared decorator checks the current slot before entering the producer, including when its upload uses a
combined group. Rejected producer reentry leaves earlier sources intact.
A deliberate second publication supplies a reason and acquires its source
before rewriting it. Enqueue failures invalidate the owner and retain storage.

The packed arena contains int64 offset/count pairs and byte payloads in one DMA.
The scatter preserves bit patterns and writes only the published ranges into
the original GPU addresses. It reuses its compiled kernel and fixed destination
table. No new synchronization event or GPU scalar read is added by packing.

Publication runs before actual graph capture/replay on the consumer stream.
Bound metadata storage stays fixed; rebuilding those buffers requires the
caller's existing consumer drain and graph rebuild. This API does not manage
checkpoint pools, bulk embeddings, model weights or KV offload storage.

## Verification

Run GPU publication and producer tests with `RUN_H2D_GPU_TESTS=1`.
The suites are organized by contract and consumer, with reentry regressions
kept beside the producers they protect:

| Test file | Coverage |
| --- | --- |
| `tests/test_h2d_publication.py` | Ownership, atomic validation, stream/device identity and graph capture |
| `tests/test_packed_h2d.py` | Byte-preserving DMA/scatter, arena reuse and direct fallback |
| `tests/test_h2d_runner_publication.py` | Sampling, token inputs, query prefixes and speculative indices |
| `tests/test_h2d_attention_publication.py` | Common attention consumers, page maps and PP/TBO buffers |
| `tests/test_h2d_v4_publication.py` | V4/V4.1 plans, state, step metadata and dummy isolation |
| `tests/test_h2d_v4_indexer_publication.py` | V4 indexer, PCP, TBO and Opus |
| `tests/test_h2d_draft_publication.py` | Draft and GDN consumers |

These suites check delayed source reuse, exact values, untouched tails, changing
batch sizes, graph padding and first consumers. CPU page-table state transitions
remain in `tests/test_shared_block_tables.py`, runnable without GPU opt-in.

Use the server's torch profiler endpoints for model traces. On ROCm, correlate
copies with HIP runtime `kind=1`; displayed memcpy names can mislabel H2D as DtoD.
Measure from preparation GPU work through the first model kernel as well as
counting copies. Fewer copies alone do not establish a universal TTFT benefit.
Experiment logs and one-off profiling tools are maintained outside this PR.

## Shared block-table preparation

`block_table_state(buffer)` attaches page-map state to the existing physical
`CpuGpuBuffer`. Common attention, MHA/MLA, V4 and V4.1 share this preparation
and publication policy. Derived attention backends inherit it. Each PP slot
and each TBO buffer has its own state; this optimization is independent of TP,
DP or PP topology. The worker RPC decoder preserves versioned rows for the
same reason, rather than rebuilding page-map change information in attention.

A source `BlockTable` carries an append-lineage version. Ordered version and
length vectors identify the batch mapping without reading page IDs. Appending retains the version; modifying or deleting existing IDs draws
a new version. The decoder copies a growing row while preserving that lineage,
so older batches remain unchanged. Independent copies draw new versions.
Unversioned array/list callers compare against the pinned snapshot instead.

The pinned page table is the CPU snapshot: preparation retains no source
array views or intermediate ndarray/tuple copies. A hit reads only row versions
and lengths. An append checks and copies the added IDs; a changed mapping
updates only changed rows. Full-width rows are copied without clearing the
bytes they replace; fully replaced batches clear exposed tails in one bytewise
pass. The fixed destination view is reused, and no source view is retained.
Optional page-range validation follows each append lineage across reordering,
with the pool limit as part of the proof. Single-page suffixes use scalar bounds
checks. Request spans still check their required number of pages every step. All rows
are validated and source ownership acquired before any host write.

CPU preparation and successful GPU publication have separate revisions.
Prefill can prepare CPU slots without uploading a table. Publication reuses the
GPU table only when its destination, prepared revision and requested row count
match. Changed tables still use one bulk publication; this does not add partial
H2D transfers. A combined group omits an unchanged table but consumers retain
its GPU view. Failed publication never marks the new revision as published.
Padding, TBO slices and capture table preparation use the same state. Direct
writes to the CPU/GPU table outside these entry points invalidate this contract.

V4.1 `RequestSpan` describes only the request interval and state slot;
`begin_step(..., block_tables=rows)` receives the page mappings separately.
Runtime dummy PAGE/state storage remains private. Graph capture continues to
bind serving storage, as required by its fixed addresses. This change does not
alter either cache-allocation policy.
