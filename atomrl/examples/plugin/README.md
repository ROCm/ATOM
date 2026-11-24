# AtomRL plugin examples

AtomRL's training engines and ATOM rollout plugged into external RL frameworks, on
Qwen3-8B with 8 AMD GPUs.

| Framework | Training | Rollout | Mechanism | Result |
|---|---|---|---|---|
| [verl](verl/README.md) | AtomRL FSDP2 | vLLM or ATOM | `verl.plugins` entry point, discovered on import | 3-step GRPO in 1 min 13 s (vLLM) / 2 min 57 s (ATOM), 8× MI300X |
| [vime](vime/README.md) | Megatron + Lumen FP8 | vLLM, or ATOM | `--custom-*-path` hooks | |

Weight sync is never injected: each framework keeps its own mechanism, and ATOM manages
its weights through `sleep` / `wake_up`.

```bash
bash examples/run.sh verl-plugin-qwen3-8b-vllm    # 3-step smokes in each framework's image
bash examples/run.sh vime-plugin-qwen3-8b
```

Design: [docs/plugin-integration.md](../../docs/plugin-integration.md). Plugin code:
[atomrl/plugin/](../../atomrl/plugin/README.md).
