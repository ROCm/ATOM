# AtomRL plugin examples

AtomRL's training engines and ATOM rollout plugged into external RL frameworks, on
Qwen3-8B with 8 AMD GPUs.

| Framework | Training | Rollout | Mechanism | Result |
|---|---|---|---|---|
| [verl](verl/README.md) | AtomRL FSDP2 | vLLM or ATOM | `verl.plugins` entry point, discovered on import | 3-step GRPO in 1 min 13 s (vLLM) / 2 min 57 s (ATOM), 8× MI300X |
| [vime](vime/README.md) | Megatron + Lumen FP8 | vLLM, or ATOM | `--custom-*-path` hooks | |
| [miles](miles/README.md) | Megatron + Lumen FP8 | SGLang, or ATOM | `--custom-*-path` hooks | unvalidated |
| [slime](slime/README.md) | Megatron + Lumen FP8 | SGLang, or ATOM | pinned install/import contract; no upstream training example | contract smoke |

Each host decides when to export weights. AtomRL lifecycle hooks then reload
ATOM from the resulting HF checkpoint; verl converts its named-tensor stream
through a scoped temporary checkpoint.

```bash
bash examples/run.sh verl-plugin-qwen3-8b-vllm    # 3-step smokes in each framework's image
bash examples/run.sh vime-plugin-qwen3-8b
bash examples/run.sh miles-plugin-qwen3-8b
bash examples/run.sh slime-plugin-contract
```

Pinned framework validation is isolated from the public image and public unit
suite:

```bash
bash atomrl/examples/plugin/test.sh
```

The dedicated test layers pin verl 0.9.1, vime `8707a5a…`, miles `fe7037f…`,
and slime `79aa41a…`. The vLLM and SGLang public bases are also locked by
digest in `Dockerfile.plugin`; miles additionally pins its Megatron fork.

Design: [docs/plugin-integration.md](../../docs/plugin-integration.md). Plugin code:
[atomrl/plugin/](../../atomrl/plugin/README.md).
