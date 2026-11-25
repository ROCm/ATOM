# verl + AtomRL plugin: Qwen3-8B GRPO

verl 0.9.0 trains with AtomRL's `atomrl_fsdp2` engine and generates with either vLLM or
ATOM. AtomRL registers both through verl's plugin entry point, with no monkey-patching
and no registration code; verl needs one assertion removed (see known issues).

## Results

3-step GRPO on Qwen3-8B-Base, 8× MI300X:

| Rollout | Config | Wall clock | Image |
|---|---|---|---|
| vLLM BF16 | `qwen3_8b_grpo_fp8.yaml` | 1 min 13 s | Legacy validation image |
| ATOM BF16 | `qwen3_8b_grpo_atom.yaml` | 2 min 57 s | Legacy validation image |

With ATOM, step 1 takes about 90 s (kernel JIT) and steps 2–3 about 40 s each.

## What the plugin registers

On `import verl`, verl discovers the AtomRL entry point and runs
`atomrl.plugin.verl.register:register()`:

- **`atomrl_fsdp2` training engine** in verl's `EngineRegistry`, wrapping AtomRL's `FSDP2Engine`.
- **`atom` rollout** in verl's `RolloutReplicaRegistry`, backed by `ATOMRayServer`.
- vLLM rollout uses verl's native support (`rollout.name: vllm`).

## Run

Models and data under `$DATA_ROOT`: `models/Qwen3-8B-Base` and the DAPO-Math-17k / AIME-2024
parquet files filtered to 1024-token prompts ([data preparation](../../docs/03-data.md)).

```bash
bash examples/run.sh verl-plugin-qwen3-8b-vllm      # in the published image, judged
```

By hand, from the ATOM repository's `atomrl/` directory:

```bash
docker run --rm --network=host --ipc=host \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 64g --tmpfs /tmp:exec,size=10g \
  -v $DATA_ROOT:/data -v $(pwd):/workspace/ATOM/atomrl \
  -e DATA_ROOT=/data -e CONFIG=examples/plugin/verl/qwen3_8b_grpo_fp8.yaml -e STEPS=3 \
  --entrypoint bash atomrl-verl-plugin:latest \
  -c 'cd /workspace/ATOM/atomrl && bash examples/plugin/verl/run.sh'
```

A successful run ends with `Training Progress: 100% | 3/3` and finite `actor/loss`,
`actor/entropy` and `critic/reward`.

**ATOM rollout.** Use `CONFIG=examples/plugin/verl/qwen3_8b_grpo_atom.yaml` in an image
built on an ATOM base. ATOM needs its AITER kernels compiled once per container (about
5 min), for example by one short `AsyncLLMEngine.generate` on GPU 0 before the first run.

**Build your own image** from the ATOM repository root:

```bash
docker build -f atomrl/examples/plugin/Dockerfile.plugin \
  --target verl-test -t atomrl-plugin-test-verl .
```

## Configuration

Both configs inherit verl's defaults (`defaults: [_generated_ppo_trainer, _self_]`):
`atomrl_fsdp2` engine, batch size 8, learning rate 1e-6, prompt and response length 512,
verl's v0 trainer.

## Known issues

| Issue | Workaround |
|---|---|
| Framework/API drift | the plugin validation image pins `verl==0.9.1` |
| ROCm stack replacement | the dedicated target asserts `torch.version.hip` after installation |
| verl's v1 trainer imports `transfer_queue`, which is not on PyPI | `trainer.use_v1: false` (both configs) |
| Several ATOM replicas can deadlock on shared-memory broadcast | the plugin launches one ATOM server with `max_concurrency=1` |
