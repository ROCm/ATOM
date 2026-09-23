# Engram host lookup dependency

The Engram libraries originate from ROCm/ATOM PR #2185, revision
`236953870b1d5a2352c647875d43239eebe5e208`. DeepSeek-V4.1 drives them through
`atom/model_engine/engram_runtime.py`, which prepares rows once final GPU token
IDs and restored state are available. See
[DeepSeek-V4.1-Flash](DeepSeek-V4.1-Flash.md), and
[environment variables](../docs/environment_variables.md#engram-deepseek-v41)
for the residency and cache-directory switches.

| Module | Responsibility |
|---|---|
| `atom/model_ops/engram.py` | Configuration, tokenizer compression, hashes, native host tables and existing staging helpers |
| `atom/model_ops/engram_layer.py` | Projection/gate library and the upstream attachment factory |
| `atom/models/deepseek_v41/weights.py` | V4.1 source schema, six-tensor ownership, independent-shard lookup and mmap loading |

The published tables use E4M3 values with per-row/group32 E8M0 scales, about
188.83 GiB combined. The initial provider maps both tables on the host and only
gathers/dequantizes requested rows. It indexes their bytes before converting
the selected rows because PyTorch 2.9 lacks CPU float8 advanced indexing.
HBM sharding remains a later performance comparison, not a capacity prohibition.

Each Engram layer owns `embed.weight`, `embed.scale`, `wkv.weight`, `wkv.scale`,
`k_weight` and `q_weight`. These tensors may reside in different safetensors
shards. The native V4.1 loader accounts for all six and prevents generic GPU
parameter loading from copying the tables. Keep the `CheckpointReader` alive
while consuming its mmap-backed tables.

The upstream projection/attachment factory is retained as a dependency; it is
not the V4.1 model entry point. What the integration has to preserve, and does:
A8 QAT on `wkv`, FP32 gating and residual arithmetic, three prior token IDs with
image DEAD markers, real prefill lookup, and injection **before attention** on
layers 1 and 14. Prefetch can overlap work only after the required token IDs are
known, and does not guarantee that host latency is fully hidden.
