# Batch-sharded sampling on four ranks: target logits exchange, sampler-output gather, Gumbel Markov draft (vocab 200064, hidden 6144, Markov rank 256)

## What it does

Batch-sharded sampling on four TP ranks: each rank verifies, and drafts for, the requests it owns (batch position i
on rank i % 4, local row i // 4), so every logits exchange moves a quarter of the rows. `BatchShardedSampling` holds
the communicators, the owner tables of every request count and the scratch buffers; the caller passes plain
tensors and ints. Shapes: vocabulary 200064 (lm_head shards of 50016, this rank's shard in `shuffle_weight`'s
layout), hidden 6144, Markov rank 256 (`markov_w1` / `markov_w2` bf16 [200064, 256] contiguous); target 1-2048 logits
rows, 4-256 requests, 1-8 logits rows per request; the Markov draft 1-7 steps, up to 64 requests per rank. Caller:
the MiniMax-M3.1 TP4 sampler and DSpark Markov draft. Contract (`reference.py`): the per-kernel entries' references;
the block equals the per-kernel chain bit for bit (draft ids, every draft-cache byte, the logits), cosine ~1 against
the fp32 torch reference.

## Entry points

- `BatchShardedSampling`: the per-process object, built collectively (every rank, the same order) before any CUDA
  graph capture; it builds and warms up the target exchange and gather with real exchanges and compiles the lm_head's
  publish tiles.
- `BatchShardedSampling.close`: collective teardown after every graph that uses it is retired.
- `BatchShardedSampling.target_logits`: the `[owned rows, 200064]` bf16 logits of this rank's requests from the
  owner-sorted hidden rows (the same row count on every rank, none: no exchange); the rows per owner are host ints,
  staged through pinned memory without a sync.
- `BatchShardedSampling.gather_sampler_output`: the sampled tokens and the sampled / rejected / NaN counts of every
  request in batch order.
- `sampling_blocker`: why the block cannot serve a batch shape (None: it can).
- `BatchShardedSampling.setup_markov`: collective, before capture: the draft's exchange, its id gather and the owner
  tables of every request count.
- `markov_blocker`: why a draft does not fit (None: it fits), naming the draft features the chain lacks.
- `BatchShardedSampling.markov_draft`: the draft ids of every request in batch order and this rank's draft-cache rows
  (bf16 `[slots, steps, 200064]`); draws at the sample position - 1; graph safe.
- `owner_tables`, `OwnerTables`: the index tables of one request count.

Every rank makes every call with the same counts. `target_logits` and `markov_draft` are entries
(`amd_kernels.warm_up`): the first call of each mode compiles every lm_head tile and every Markov variant of
1..max_reqs requests; their warm-up calls publish into a private slot, skip the exchanges and give the Gumbel kernel
no request slot, so they never write a caller's buffer or a communicator. No kernel of its own: `host.py` sequences
the per-kernel entries (its module docstring lists each chain).

## Performance

Not timed as a block. The launched entries' READMEs have their numbers.

## Tests

```bash
python3 -m pytest tests/sampling/batch_sharded_markov_v200064_h6144_r256_ranks4 -q  # 4 GPUs
```
