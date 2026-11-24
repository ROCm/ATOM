# Four-rank all-gather of int64 rows [requests, 7]

## What it does

`I64C7AllGather` gathers the DSpark draft-token rows of MiniMax-M3.1-Flash TP4 from four ranks straight into a
contiguous int64 `[requests, 7]` output, in one FlyDSL launch, in place of a padded all-gather plus `index_select`
in the batch-sharded draft sampler's draft-id gather. Contract (`reference.py`): integer-exact, every int64 bit pattern
kept; `src[i]` decodes as `(rank, local_row) = divmod(src[i], src_stride)`; only the referenced local rows must be
valid. The communicator is `SamplerAllGather`'s (`collectives/all_gather_sampler_i64_c8_ranks4`): the same IPC pool of
two 22528-byte slots (this entry uses 256 x 56 bytes of each), the same two-slot device sequence and system-scope store
/ load rules, with its own kernel and its own sequence counter. Atomics only poll the sequence flags; output rows use
plain stores.

## Entry points

- `I64C7AllGather`: a collective on the four ranks of a gloo group; allocates and maps the pool.
- `I64C7AllGather.warmup`: a collective before graph capture; compiles the one variant (0-256 local and output rows).
- `I64C7AllGather.__call__`: gathers the referenced int64 `[*, 7]` rows (any row stride) into a contiguous output on
  the communicator's device; returns it. The caller passes `local_draft_ids[:num_local_reqs]`,
  `gathered_src[:out.shape[0]]` and `num_reqs_padded` as the source stride.
- `I64C7AllGather.close`: a collective, after every graph that uses the communicator is retired.

Every rank calls in the same order, empty senders and receivers included; one communicator per stream. Shapes are
runtime arguments, fixed inside a captured graph; payloads and maps may change on every replay. The call allocates
nothing.

## Performance

Four GPUs of one MI355X node. Graphs of 100 calls on 100 distinct payloads, ten replays per round, five
alternating rounds, MAX over ranks, median us. Baseline: the padded all-gather + `index_select` (measured separately;
the benchmark times this entry only). Every round wins, ranges disjoint.

| requests | dense owners: baseline | this entry | random owners: baseline | this entry |
|---:|---:|---:|---:|---:|
| 10 | 15.1 | 4.4 | 15.3 | 4.4 |
| 16 | 16.5 | 4.5 | 16.4 | 4.6 |
| 32 | 12.8 | 4.6 | 12.7 | 4.6 |
| 64 | 14.4 | 4.6 | 14.4 | 4.6 |
| 128 | 15.0 | 4.8 | 15.3 | 4.7 |
| 256 | 14.1 | 5.3 | 14.2 | 5.3 |

## Tests

```bash
python3 -m pytest tests/collectives/all_gather_i64_c7_ranks4 -q                # 4 GPUs
python3 benchmarks/collectives/all_gather_i64_c7_ranks4/benchmark_gather.py    # 4 GPUs
```

Four spawned ranks: graph replay with changed maps and payloads, strided rows, unequal owners, empty ranks, all rows
on one rank, odd counts. The shared protocol's sequence wrap and 1000 slot reuses are tested by the sampler gather.
