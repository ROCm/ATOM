# Vocabulary embedding by owner reads on four ranks, hidden 6144

## What it does

Reads each token's embedding row straight from the rank that owns its vocabulary row: four TP ranks keep their
bf16 vocabulary shards (four equal contiguous `[V, 6144]` shards in rank order, a plain contiguous partition with no
added-vocabulary gaps) and peers map them once at load time over HIP IPC. One stateless kernel in place of the
mask / gather / fill + all-reduce of `VocabParallelEmbedding`: no weight copy, no payload sum, no staging buffer and
no per-call handshake. Token `t` reads shard `t // V`, row `t % V`; ids are a contiguous 1-D int32 or int64 tensor;
the output is a contiguous bf16 `[rows, 6144]` on each rank. Caller: the MiniMax-M3.1 DSpark draft's embedding at
TP4 (V = 50016, vocabulary 200064), through `decoder/dense_mxfp8_h6144_i3072_ranks4`;
`embedding/vocab_gather_gemma_mxfp8_h6144` reuses the mapping. Contract (`reference.py`): finite rows bit for bit;
bf16 subnormals and signed zeros become +0 (AITER's fp32 flush when it adds the one owner to zeros); invalid ids
(< 0 or >= 4V) write +0; replays and eager calls are bitwise equal on every rank.

## Entry points

- `VocabGather`: the collective load-time mapping of the four shards, built on every rank after selecting its GPU
  and after this rank's shard reaches its final allocation; it checks equal shard sizes, publishes the GPU-written
  weight to peers and exchanges the IPC handles. The weights stay alive and unchanged until `close()`; never move or
  overwrite them while a captured graph may read them.
- `VocabGather.__call__`: the `[rows, 6144]` bf16 rows of the ids, eager or inside a CUDA graph; no inter-rank
  synchronization per call.
- `VocabGather.warm_up`: compiles the int32 and int64 variants (one per id dtype, any row count).
- `VocabGather.close`: collective on every rank, after all graph use.

No dispatch: one kernel per id dtype (`host.py`), rows a runtime argument.

## Performance

MI355X, four ranks; graph replay of 100 distinct batches (7 rows per
request: one random anchor + six mask ids), warm weights, max over ranks, median of 5 rounds x 10 replays, us per
call. The unfused chain (mask / gather / fill + all-reduce) is timed in the whole draft step, not as a unit.

| Requests (rows) | 1 (7) | 10 (70) | 16 (112) | 32 (224) | 64 (448) | 128 (896) | 256 (1792) |
|---|---|---|---|---|---|---|---|
| `lookup` us | 3.10 | 4.02 | 4.43 | 5.40 | 7.59 | 13.51 | 26.70 |

## Tests

```bash
python3 -m pytest tests/embedding/vocab_gather_h6144 -q  # 4 GPUs
python3 benchmarks/embedding/vocab_gather_h6144/benchmark_gather.py  # 4 GPUs
```
