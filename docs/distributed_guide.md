# ATOM Distributed Inference Guide

ATOM (AiTer Optimized Model) supports three parallelism strategies for distributed LLM inference on AMD ROCm/HIP GPUs: Tensor Parallelism (TP), Data Parallelism (DP), and Expert Parallelism (EP). These can be combined to scale across multiple GPUs for large model serving.

## Quick Reference

| Parallelism | CLI Flag | Purpose | Communication |
|-------------|----------|---------|---------------|
| Tensor Parallel (TP) | `-tp N` / `--tensor-parallel-size N` | Shard weights across GPUs | NCCL AllReduce |
| Data Parallel (DP) | `-dp N` / `--data-parallel-size N` | Replicate model, split requests | Gloo AllReduce (CPU) |
| Expert Parallel (EP) | `--enable-expert-parallel` | Distribute MoE experts across GPUs | MORI All-to-All |
| DP Attention | `--enable-dp-attention` | Flatten DP into TP for MoE layers | NCCL AllGather/ReduceScatter |

**Common configurations:**

| Model Type | Configuration | Example |
|-----------|---------------|---------|
| Dense (Llama, Qwen3) | TP only | `-tp 8` |
| MoE (Qwen3-235B) | TP + EP | `-tp 8 --enable-expert-parallel` |
| MoE throughput scaling | TP + DP + EP | `-tp 4 -dp 2 --enable-expert-parallel` |
| Dense throughput scaling | TP + DP | `-tp 4 -dp 2` |

---

## 1. Tensor Parallelism (TP)

Tensor Parallelism shards model weights across GPUs so each GPU holds a slice of every layer. ATOM uses AITER's `init_dist_env()` to initialize NCCL process groups.

### Weight Sharding

ATOM provides parallel linear layer classes in `atom/model_ops/linear.py`:

- **`ColumnParallelLinear`** -- splits the output dimension (dim 0) across TP ranks. Each GPU computes a shard of the output independently.
- **`RowParallelLinear`** -- splits the input dimension (dim 1) across TP ranks. After the local matmul, an AllReduce across the TP group aggregates partial results.
- **`QKVParallelLinear`** -- extends `ColumnParallelLinear` for attention Q/K/V projections. Partitions heads across TP ranks, replicating KV heads when `num_kv_heads < tp_size`.
- **`MergedColumnParallelLinear`** -- merges multiple column-parallel outputs (e.g., gate and up projections) into a single weight tensor, sharded along dim 0.
- **`ReplicatedLinear`** -- no sharding; weight is replicated on every rank.

### Process Group Initialization

In `ModelRunner.__init__()`, the distributed environment is set up via AITER:

```python
from aiter import init_dist_env
from aiter.dist.parallel_state import get_tp_group, get_dp_group, get_pp_group

init_dist_env(
    config.tensor_parallel_size,
    rankID=rank,
    backend="nccl",
    distributed_init_method=distributed_init_method,
    data_parallel_size=config.parallel_config.data_parallel_size,
    data_parallel_rank=config.parallel_config.data_parallel_rank,
)
```

After initialization, `get_tp_group()`, `get_dp_group()`, and `get_pp_group()` provide the respective process groups for collective operations.

### AllReduce

The AllReduce happens inside `LinearBase.forward()` when `tp_dim == 1` (row-parallel):

```python
if self.tp_dim == 1 and self.tp_size > 1 and self.reduce_results:
    y = get_tp_group().all_reduce(y, ca_fp8_quant=False)
```

### Configuration

- `Config.tensor_parallel_size` (int, default `1`): Number of TP ranks. Must satisfy `1 <= tensor_parallel_size <= 8`.
- CLI: `--tensor-parallel-size N` or `-tp N`

---

## 2. Data Parallelism (DP)

Data Parallelism runs multiple independent engine replicas, each handling a subset of incoming requests. DP is coordinated at the scheduling level rather than the model level -- each DP rank has its own `EngineCore`, scheduler, and model runner.

### Architecture

When `data_parallel_size > 1`, `EngineCore.run_engine()` instantiates a `DPEngineCoreProc` instead of a plain `EngineCore`:

```python
# atom/model_engine/engine_core.py
@staticmethod
def run_engine(config, input_address, output_address):
    if config.parallel_config.data_parallel_size > 1:
        engine = DPEngineCoreProc(config, input_address, output_address)
    else:
        engine = EngineCore(config, input_address, output_address)
    engine.busy_loop()
```

### DP Process Group Initialization

`DPEngineCoreProc._init_data_parallel()` creates a Gloo-based process group for CPU-side coordination:

```python
def _init_data_parallel(self, config):
    dp_rank = config.parallel_config.data_parallel_rank
    dp_size = config.parallel_config.data_parallel_size
    local_dp_rank = config.parallel_config.data_parallel_rank_local

    assert dp_size > 1
    assert local_dp_rank is not None

    self.dp_rank = dp_rank
    self.dp_group = config.parallel_config.stateless_init_dp_group()
```

The `stateless_init_dp_group()` method (in `ParallelConfig`) calls `stateless_init_torch_distributed_process_group()` with the `gloo` backend, creating an isolated process group that does not interfere with the NCCL TP group.

### Synchronized Busy Loop

The DP busy loop overrides the base `EngineCore.busy_loop()` to synchronize state across DP ranks before each step. The `_sync_dp_state()` method packs four signals into an int64 tensor and performs a single `AllReduce(MAX)`:

```python
# State synced: [is_prefill, num_tokens, has_unfinished, shutdown]
state_tensor = torch.tensor(
    [
        1 if local_is_prefill else 0,
        local_num_tokens,
        1 if local_has_unfinished else 0,
        1 if local_shutdown else 0,
    ],
    dtype=torch.int64, device="cpu",
)
torch.distributed.all_reduce(
    state_tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
)
```

This ensures:
- **All ranks agree on the batch type** (prefill vs. decode). Since MORI requires all DP ranks to execute the same phase, a rank that has no prefill work must run a dummy prefill when any other rank does prefill.
- **Graceful shutdown**: all ranks must agree before exiting.
- **Token count alignment**: the maximum token count across ranks is used for padding.

### Dummy Batch Execution

When a DP rank has no real work but other ranks do, it executes dummy batches to participate in collective operations:

- **`_execute_dummy_batch()`** -- runs a 1-token decode dummy through the model, triggering AllReduce and MORI collectives so other ranks are not blocked.
- **`_execute_dummy_prefill(num_tokens)`** -- runs a dummy prefill with the same token count as the max across DP ranks, so that MORI dispatch/combine stays synchronized.

### Device Assignment

When DP is enabled on a single node, each DP rank uses a different set of GPUs. The device mapping in `ModelRunner.__init__()` is:

```python
local_device_rank = dp_rank_local * config.tensor_parallel_size + rank
device = torch.device(f"cuda:{local_device_rank}")
```

For example, with DP=2 and TP=4:
- DP rank 0: GPUs 0, 1, 2, 3
- DP rank 1: GPUs 4, 5, 6, 7

### DPMetadata

The `DPMetadata` dataclass (in `atom/utils/forward_context.py`) tracks token distribution across DP ranks for padding and collective operations:

```python
@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor   # Max tokens on any DP rank
    cu_tokens_across_dp_cpu: torch.Tensor    # Cumulative token counts
    max_tokens_across_dp: int                # Pre-computed int for CUDA graph
```

`DPMetadata.num_tokens_across_dp()` gathers token counts via an AllReduce on the DP CPU group:

```python
num_tokens_across_dp = [0] * dp_size
num_tokens_across_dp[dp_rank] = num_tokens
num_tokens_tensor = torch.tensor(num_tokens_across_dp, device="cpu", dtype=torch.int32)
dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
```

### CoreManager (DP Orchestration)

`CoreManager` (in `atom/model_engine/engine_core_mgr.py`) manages multiple DP engine processes:

1. For each DP rank, it creates a `Config` copy with the appropriate `data_parallel_rank` and `data_parallel_rank_local`.
2. Launches each `EngineCore` in a separate `multiprocessing.Process`.
3. Uses ZMQ (ROUTER/DEALER) sockets for input distribution and ZMQ (PUSH/PULL) for output collection.
4. Distributes incoming requests across DP ranks via round-robin load balancing.
5. Waits for READY signals from all ranks before accepting requests.

When `enable_dp_attention` is set, `CoreManager` flattens TP into DP:

```python
if config.enable_dp_attention:
    self.local_engine_count = config.tensor_parallel_size * config.parallel_config.data_parallel_size
    config.parallel_config.data_parallel_size = self.local_engine_count
    config.tensor_parallel_size = 1
```

### Configuration

- `ParallelConfig.data_parallel_size` (int, default `1`): Number of DP replicas.
- `ParallelConfig.data_parallel_rank` (int, default `0`): This rank's DP index.
- `ParallelConfig.data_parallel_rank_local` (int, default `None`): Local DP rank on this node.
- CLI: `--data-parallel-size N` or `-dp N`

---

## 3. Expert Parallelism (EP)

Expert Parallelism distributes MoE experts across GPUs so that each GPU owns a subset of experts. Tokens are routed to the correct GPU via all-to-all communication.

### FusedMoEParallelConfig

The `FusedMoEParallelConfig` dataclass (in `atom/model_ops/moe.py`) determines how MoE layers are parallelized:

```python
@dataclass
class FusedMoEParallelConfig:
    tp_size: int        # Tensor parallel size (1 when EP is active)
    dp_size: int        # Data parallel size
    ep_size: int        # Expert parallel size
    tp_rank: int
    dp_rank: int
    ep_rank: int
    use_ep: bool        # Whether EP is enabled
    local_ep_size: int  # Number of EP ranks on this node
```

Key properties:

- **`use_all2all_kernels`**: returns `True` when `dp_size > 1 and use_ep and mori is available`. This activates the MORI all-to-all dispatch/combine kernels.
- When EP is enabled, `tp_size` is set to 1 and `ep_size = dp_size * tp_size` (the original TP size). Each device fully owns its assigned experts.

The `FusedMoEParallelConfig.make()` static method constructs the config:

```python
use_ep = dp_size_ * tp_size_ > 1 and parallel_config.enable_expert_parallel

if enable_dp_attention:
    # Flatten DP into TP: effective tp_size = dp_size * tp_size
    tp_size, tp_rank = flatten_tp_across_dp(dp_rank)

if use_ep:
    ep_size = tp_size
    ep_rank = tp_rank
    # Each device owns experts fully -- no intra-expert tensor parallelism
    return FusedMoEParallelConfig(tp_size=1, tp_rank=0, ep_size=ep_size, ...)
```

### Expert Distribution

In `FusedMoE.__init__()`, when EP is active, the global experts are partitioned:

```python
if self.use_ep:
    self.local_num_experts, self.expert_map = determine_expert_map(
        ep_size=self.ep_size,
        ep_rank=self.ep_rank,
        global_num_experts=self.global_num_experts,
    )
else:
    self.local_num_experts = self.global_num_experts
    self.expert_map = None
```

Each GPU only loads weights for its assigned experts, reducing per-GPU memory usage proportionally.

### MORI Communication

When `use_all2all_kernels` is `True`, the `MoriPrepareAndFinalize` class (in `atom/model_ops/fused_moe/mori_prepare_finalize.py`) handles token routing:

**Dispatch phase** (`prepare()`):
1. Receives input activations, top-k weights, and top-k expert IDs.
2. Calls `self.mori_op.dispatch()` to send each token to the GPU that owns its selected expert.
3. Returns dispatched activations, scales, expert IDs, weights, and per-expert token counts.

```python
(dispatch_a1, dispatch_weights, dispatch_scale, dispatch_ids, dispatch_recv_token_num
) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids, block_num, warp_per_block)
```

**Combine phase** (`finalize()`):
1. After expert computation, calls `self.mori_op.combine()` to route results back to the originating GPU.
2. Copies the combined result into the output tensor.

```python
result = self.mori_op.combine(fused_expert_output, None, topk_ids, block_num, warp_per_block)[0]
output.copy_(result[:num_token])
```

The block configuration adapts to the batch type: prefill uses `block_num=128, warp_per_block=16`, while decode uses `block_num=64, warp_per_block=4`.

### Configuration

- `Config.enable_expert_parallel` (bool, default `False`): Activates EP for MoE layers.
- `Config.enable_dp_attention` (bool, default `False`): Flattens DP ranks into the TP/EP dimension for MoE, while using per-rank attention for non-MoE layers.
- CLI: `--enable-expert-parallel`, `--enable-dp-attention`

---

## 4. Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ATOM_DP_RANK` | int | `0` | Data parallel rank index |
| `ATOM_DP_RANK_LOCAL` | int | `0` | Local data parallel rank on this node |
| `ATOM_DP_SIZE` | int | `1` | Total number of data parallel replicas |
| `ATOM_DP_MASTER_IP` | str | `127.0.0.1` | IP address for DP Gloo rendezvous |
| `ATOM_DP_MASTER_PORT` | int | `29500` | Port for DP Gloo rendezvous |
| `ATOM_ENFORCE_EAGER` | bool | `False` | Disable CUDA graphs (set automatically) |
| `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION` | bool | `False` | Fuse QK-norm + RoPE + cache quant (for Qwen3-MoE) |

Environment variables in `atom/utils/envs.py` are evaluated lazily via `__getattr__`. If `ATOM_DP_SIZE`, `ATOM_DP_RANK`, or `ATOM_DP_RANK_LOCAL` are set in the environment, they override programmatic `ParallelConfig` defaults in `ParallelConfig.__post_init__()`.

**AITER environment variable (not in envs.py):**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AITER_QUICK_REDUCE_QUANTIZATION` | str | -- | Set to `INT4` to enable quantized AllReduce for prefill (read by AITER's AllReduce kernel) |

---

## 5. Multi-GPU Deployment Examples

### DeepSeek-R1 on 8 GPUs (TP8)

From the project README -- a dense MLA model deployed with pure tensor parallelism:

```bash
python -m atom.entrypoints.openai_server \
    --kv_cache_dtype fp8 \
    -tp 8 \
    --model deepseek-ai/DeepSeek-R1
```

### Qwen3-235B-A22B on 8 GPUs (TP8 + EP)

From `recipes/Qwen3-235b.md` -- a MoE model with 128 experts, deployed with tensor parallelism and expert parallelism:

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

python -m atom.entrypoints.openai_server \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --enable-expert-parallel \
    --max-model-len 16384 \
    --max-num-batched-tokens 20000
```

Tips from the recipe:
- Use FP8 KV cache (`--kv_cache_dtype fp8`) for memory efficiency.
- Quick AllReduce with INT4 quantization reduces prefill TTFT.
- QK-norm + RoPE + cache quant fusion improves Qwen3-MoE kernel performance.

### Kimi-K2-Thinking on 4 GPUs (TP4)

From `recipes/Kimi-K2-Thinking.md` -- an MXFP4 MoE model:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

python -m atom.entrypoints.openai_server \
    --model amd/Kimi-K2-Thinking-MXFP4 \
    --trust-remote-code \
    -tp 4 \
    --kv_cache_dtype fp8
```

---

## 6. Combined Parallelism Strategies

### TP Only (Dense Models)

For dense models like Llama and Qwen3 (non-MoE), use pure tensor parallelism:

```bash
python -m atom.entrypoints.openai_server --model meta-llama/Meta-Llama-3-8B -tp 8
```

All weights are sharded across GPUs. AllReduce collectives synchronize after each `RowParallelLinear`.

### TP + EP (MoE Models)

For MoE models, enable expert parallelism so each GPU holds a subset of experts:

```bash
python -m atom.entrypoints.openai_server --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 -tp 8 --enable-expert-parallel
```

Dense layers (attention, norms) remain tensor-parallel. MoE layers distribute experts across the `ep_size = tp_size` GPUs. MORI all-to-all routes tokens to the correct expert owner.

### TP + DP (Dense Throughput)

For throughput scaling with dense models, run multiple DP replicas:

```bash
# On a node with 8 GPUs: 2 replicas, each using 4 GPUs
python -m atom.entrypoints.openai_server --model meta-llama/Meta-Llama-3-8B -tp 4 -dp 2
```

Each DP replica independently processes a subset of requests. The `CoreManager` distributes requests via round-robin. Device mapping:
- DP rank 0, TP ranks 0-3 --> GPUs 0-3
- DP rank 1, TP ranks 0-3 --> GPUs 4-7

Formula: `local_device_rank = dp_rank_local * tp_size + tp_rank`

### TP + DP + EP (MoE Throughput)

For MoE models with DP + EP, the expert parallel dimension spans all `tp_size * dp_size` devices:

```bash
python -m atom.entrypoints.openai_server \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    -tp 4 -dp 2 \
    --enable-expert-parallel
```

In this configuration:
- Dense layers: each DP replica has TP=4 for sharding.
- MoE layers: EP size = `dp_size * tp_size = 8`, spreading experts across all 8 GPUs.
- MORI all-to-all crosses DP boundaries to route tokens to the correct expert owner.

### DP Attention Mode

When `--enable-dp-attention` is set, `CoreManager` flattens the TP dimension into DP:

```python
local_engine_count = tensor_parallel_size * data_parallel_size
data_parallel_size = local_engine_count
tensor_parallel_size = 1
```

This means each GPU runs an independent attention computation (no TP AllReduce for attention), while MoE layers still use the full EP group across all GPUs. This can reduce communication overhead for attention-heavy workloads.

---

## Source Files

| File | Description |
|------|-------------|
| `atom/config.py` | `ParallelConfig`, `Config.tensor_parallel_size`, `enable_expert_parallel`, `enable_dp_attention` |
| `atom/utils/envs.py` | `ATOM_DP_*` environment variables (lazy evaluation) |
| `atom/model_engine/engine_core.py` | `EngineCore`, `DPEngineCoreProc` (DP busy loop, sync, dummy batches) |
| `atom/model_engine/engine_core_mgr.py` | `CoreManager` (multi-process DP orchestration, ZMQ IPC) |
| `atom/model_engine/model_runner.py` | `ModelRunner` (`init_dist_env`, device assignment, `DPMetadata` usage) |
| `atom/model_engine/arg_utils.py` | `EngineArgs` CLI argument definitions |
| `atom/utils/distributed/utils.py` | `stateless_init_torch_distributed_process_group()` (Gloo PG creation) |
| `atom/utils/forward_context.py` | `DPMetadata`, `ForwardContext` (per-step DP token metadata) |
| `atom/model_ops/linear.py` | `ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear` |
| `atom/model_ops/moe.py` | `FusedMoE`, `FusedMoEParallelConfig` (EP configuration and expert distribution) |
| `atom/model_ops/fused_moe/mori_prepare_finalize.py` | `MoriPrepareAndFinalize` (MORI dispatch/combine for EP) |
