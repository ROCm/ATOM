# Distributed training

AtomRL scales along two axes: **training backend parallelism** (FSDP2 vs Megatron-Core) and **cluster orchestration** (Ray placement, multi-node SLURM). This page summarizes how the pieces fit together.

## FSDP2 backend

`policy.training_backend: fsdp2` selects the PyTorch FSDP2 integration under
`atomrl/engine/training/fsdp_backend.py`. It is the default dense-model backend.

Strengths:

- Native integration with Hugging Face–style module trees
- Works well for single-node and moderate multi-node jobs
- Compatible with Lumen FP8 hooks via `FP8TrainingManager`

## Megatron-Native backend

`policy.training_backend: megatron_native` selects the TransformerEngine-based
Megatron-Core engine. Configure tensor, pipeline, and context parallel sizes via
`policy.training.megatron_cfg` (`MegatronConfig` in {doc}`/api/config`).

## Backend support matrix

| Capability | FSDP2 | Megatron-Native |
| --- | --- | --- |
| Dense BF16 training | Yes | Yes |
| Dense FP8 training | Yes | Not validated |
| FP8 rollout (ATOM) | Yes | Yes |
| FSDP parameter sharding | Yes | No |
| Tensor / pipeline / context parallelism | No | Yes |
| Megatron dist-checkpoint | No | Yes |

```{note}
The repository currently ships validated dense recipes. Add new expert-parallel
recipes only after validating them against `megatron_native`.
```

## Multi-node launch

At minimum, align YAML cluster fields with the physical allocation:

```yaml
cluster:
  num_nodes: 2
  gpus_per_node: 8
  ray_address: null    # or "auto" / explicit head address
```

For SLURM, keep the driver command in a single variable so `sbatch` can reuse it:

```bash
bash scripts/launch_slurm.sh 4 configs/grpo_dense_fp8.yaml \
  policy.model_name=Qwen/Qwen3-8B
```

Adapt `scripts/ray.sub` to export `RAY_ADDRESS`, NCCL/MORI environment variables, and conda/module initialization for your site.

## Ray cluster setup

Ray options:

- **Single-node local** — default `ray.init()` inside controller utilities when `cluster.ray_address` is unset.
- **Head + workers** — set `cluster.ray_address` to `ray://<head-ip>:10001` (port depends on your Ray version and firewall rules).

Ensure object store memory (`--object-store-memory`) and `/dev/shm` size are adequate for `DataProto` payloads containing long responses and MoE router tensors.

### Ray controller dispatch modes

With `controller.ray.enabled: true`, each role's dispatch mode is one of
`dp_compute_proto` (default), `dp_compute`, `dp_compute_proto_with_func`,
`dp_compute_metric`, `one_to_all`, `all_to_all` and `rank_zero`. `broadcast` is
accepted as an alias of `one_to_all`; `direct_rollout_method` is rejected on the
controller dispatch path. `configs/recipes/ray_controller_dispatch_modes_1n8g.yaml`
exercises them on one node.

## SLURM integration

Typical pattern:

1. `sbatch` allocates nodes and GPUs.
2. Prolog starts Ray head on rank 0 and Ray workers on other ranks.
3. Rank 0 executes the `COMMAND` Python entrypoint with `cluster.ray_address` pointing at the head.

Validate networking (`NCCL_SOCKET_IFNAME`, RoCE GIDs) independently of AtomRL—distributed hangs here manifest as “stuck rollout” rather than Python tracebacks.

Related pages: {doc}`/quickstart/quick_start`, {doc}`/architecture`, {doc}`/api/config`.
