# Cached-prefill K/V workspace budget

Cached prefill can gather each request's full context from the paged KV cache
into dense K/V tensors before variable-length attention. Long cached contexts
can make these temporary tensors much larger than the new-token batch:
`max_num_batched_tokens` limits queries, not the sum of full context lengths.

To bound this expansion, add the global engine option:

```bash
--gqa-prefill-kv-workspace-mb 8192
```

The default is `0` (disabled). Values are in MiB, so `8192` allows **8 GiB per
rank, per attention call** for the materialized dense K+V tensors. Grouping
currently requires prefill context parallel size 1; a positive budget with
PCP greater than 1 is rejected during configuration.

## Which layers group

The decision is made after the current attention layer selects its backend,
at `PagedAttentionImpl.prefill_attention()`. Only cached prefill through the
dense MHA/GQA gather + FMHA path is eligible, for target and draft layers alike.
This includes MQA and equal Q/KV head counts (MHA), as used by some Eagle drafts;
checking only whether query heads outnumber KV heads would miss those drafts.

Sparse attention, MLA, linear attention, and direct paged attention bypass this
entry point. A model can mix eligible and ineligible layers. Model family and
speculative decoding method do not determine whether a particular layer groups.
Uncached prefill, decode and batches already within budget take the existing
path. Under-budget batches do not allocate an extra output buffer.

## Request groups and memory accounting

Contiguous complete requests are grouped in their original order. Each group
runs the existing gather and attention on the caller's stream; outputs return
to their original query-token positions. Every query retains its visible context
and bottom-right causal alignment. The grouping itself never splits or truncates
a request's context.

Each layer calculates its token limit from its actual local KV shape and the
dtype used by gather (both dense tensors use `k.dtype`):

```text
max_group_kv_tokens = floor(workspace_bytes /
    (2 * local_num_kv_heads * head_dim * dtype_bytes))
```

For 16 local KV heads, head dimension 128 and BF16, 8 GiB fits 1,048,576 KV
tokens. Both cached and newly added tokens count. A recorded MiniMax-M3 batch
contained 15 requests, 32,721 queries and 4,030,289 full-context KV tokens.
Ungrouped dense K+V would occupy 30.75 GiB per rank; the 8 GiB budget creates six
complete-request groups, with 6.64 GiB in the largest group. These are calculated
tensor sizes, not measured process peak memory.

A single request exceeding the budget raises an error asking for a larger budget
or `0` to disable grouping. Grouping adds attention launches and output copies,
so the budget should accommodate the intended maximum context.

The limit covers dense K/V materialization only. It excludes Q/output tensors,
persistent KV pools, cache-layout conversions, attention scratch/conversions,
and allocator-reserved memory. Concurrent TBO calls each have their own budget;
this is not a rank-wide aggregate or a bound on total model memory. It does not
change allocator/IPC settings or perform LMCache host-memory cleanup.

## Reserve workspace before sizing persistent KV

A short or uncached warmup may not allocate long-context gather tensors. When
this option is enabled, `ModelRunner.get_num_blocks()` therefore reserves the
largest configured dense-gather bound among the rank's local target and draft
layers before planning persistent KV pools. Sequential layers and request groups
reuse the workspace; their bounds are not added together. With TBO enabled,
the reservation is twice the largest local bound for two overlapping batches.

Sparse/MLA/linear implementations do not incur this dense-gather reservation.
A rank without eligible local layers reserves zero. Dense implementations are
conservatively counted even if some requests dispatch to a direct-paged path.
For example, MiniMax-M3 PP4 has target dense GQA layers on the first stage and
an EAGLE MHA draft on the last stage; those stages reserve workspace while the
sparse-only middle stages do not.

The reservation is deducted independently from both the utilization-based KV
budget and the physically free memory limit. The smaller resulting budget sizes
the KV pool; the two deductions are not added together. Existing activation,
CUDA graph, safety and connector budgets remain in place. Per-rank budgets can
differ; PP ultimately uses the minimum supported KV block count across ranks.
Startup logs report the workspace reserve and resulting KV budget. This is
memory accounting, not an eager allocation of a dedicated workspace tensor.

Increasing the workspace limit can reduce persistent KV capacity. At TP1, an
EAGLE layer with 64 local KV heads, head dimension 128 and BF16 gather output
needs 32 GiB for one 1,048,576-token request; 8 GiB accommodates only 262,144
tokens for that layer. Choose the budget for the intended context and local
head geometry. Setting the option to zero disables grouping and this explicit
reservation; it does not make an unbounded gather consume zero memory.

## Shared metadata and Eagle

The common prefill metadata builder captures owned CPU cumulative query and
context lengths from real requests, excluding padded rows. These immutable
snapshots do not alias the runner's reusable staging buffers and require no
GPU-to-CPU readback. Grouping creates metadata views without modifying the
caller's batch or scheduling state.

Eagle's first prefill pass and standalone `compute_draft_kv()` reuse these
request lengths. The target and draft may have different KV geometries, so each
layer plans its own groups; no target-derived grouping is imposed on the draft.
When Eagle reuses the metadata object for later decode steps, it clears the
prefill snapshot.

TBO slices the snapshot together with query/request metadata. If a token cut
makes a prior micro-batch's tokens into a cached prefix, the dense backend updates
the snapshot to the final visible context before gathering. Grouping then keeps
each request within that metadata view intact. MLA's separate chunk workspace
path does not consume the dense grouping snapshot.
