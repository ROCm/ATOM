# AtomRL Training Reference

## Temporary Bug Note File

Use the repo-local file `.cursor/tmp-rl-bugs.md` relative to the `AtomRL` repo root to track possible bugs found during testing.

Rules:

- read the whole file at the start of every new debug session
- use it to avoid repeating dead ends and to reuse prior evidence
- do not treat it as proof that overrides fresh reference diffs
- append or update findings after each meaningful test, repro, or validation step

Treat any fresh return to the same debugging problem as a new debug session:

- a new chat or agent session
- a new day or work block
- returning after unrelated work
- starting a new round of debug after prior tests finished

Meaningful test or experiment means a step that changes confidence in a hypothesis, for example:

- a new minimal repro (single-step GRPO, isolated rollout, etc.)
- a BF16 vs FP8 rollout log-prob comparison
- a router distribution diff (recorded vs replayed)
- a weight sync round-trip check
- a reward curve comparison across configs
- a targeted integration test

Do not log every identical rerun. Do log negative results that rule a suspicion out.

Each entry should record:

- date or session marker
- symptom (reward collapse, KL spike, NaN, FP8 drift, etc.)
- possible bug or suspicion
- evidence collected so far
- next check
- status: open, ruled out, or resolved

## If No Trusted Reference Exists

Do not tune first. Freeze a comparison target before bring-up or debugging:

- run the BF16 baseline recipe (`configs/grpo_dense_bf16.yaml`) to convergence
- record the reward curve, KL curve, and final reward as the reference
- define the success signal (e.g., reward at step N within X% of baseline)
- record the exact command, seed, dataset version, and model checkpoint

## Pre-Training Comparison Checklist

Freeze one trusted reference before starting bring-up or debugging:

- config file path and any CLI overrides
- model checkpoint or cold-start state
- exact command line
- seed
- first signal that defines success or failure (reward at step N, KL threshold, etc.)

Compare effective runtime values, not just config files:

- record the compared values in one diff note or table
- do not claim alignment from memory

### Model and parallelism

- model family and architecture (dense vs MoE, number of experts)
- training backend: FSDP2 or Megatron-Core
- tensor, expert, pipeline, and data parallel settings
- microbatch, global batch, num_generations per prompt

### RL algorithm

- algorithm name (GRPO, DAPO, PPO)
- clip ratio (symmetric or asymmetric for DAPO)
- KL coefficient
- num_ppo_epochs, num_mini_batches
- advantage normalization method
- dynamic sampling (DAPO-specific)
- token-level vs sequence-level PG loss

### Quantization

- rollout precision: bf16 or fp8
- FP8 recipe: blockwise or tensorwise
- num_first_layers_in_bf16, num_last_layers_in_bf16
- training FP8: null, hybrid, or full
- fp8_weight_cache: true or false
- rollout correction: enabled, method (tis/mis), clip value

### MoE and R3

- R3 enabled or disabled
- replay mode: distribution or hard_assignment
- record_router_logits: true or false
- expert_parallel_size
- number of experts

### Reward

- reward type: function or model
- reward function name
- dataset name and version
- dataset split

### Inference engine

- ATOM tensor_parallel_size
- kv_cache_dtype
- max_model_len

If any of the above differs from the reference, fix that diff before claiming the runs are aligned.

## Debug Entry Checklist

Before running deeper debug:

1. Capture the exact failing behavior:
   - step number where anomaly first appears
   - reward curve shape (collapse, plateau, divergence)
   - KL divergence value at failure point
   - NaN or Inf location (rollout, training, reward)
   - which worker type (actor, rollout, ref, reward) shows the issue
2. Reduce to the smallest repro that still fails:
   - single-GPU if possible
   - 3-5 RL steps
   - same seed, same dataset, same checkpoint
   - same config with minimal changes
3. Remove noise:
   - disable wandb logging
   - disable unrelated features (e.g., turn off R3 if debugging FP8)
   - avoid changing multiple variables at once

If you are unsure whether the run is "bad enough" to stop, stop and diff against the reference anyway.

## FP8 Debug Checklist

Use this after relevant RL and config parameters are aligned with the BF16 baseline.

1. Compare rollout outputs:
   - same model weights, same prompts, same seed
   - BF16 rollout vs FP8 rollout
   - compare per-token log-prob distributions
   - record max and mean absolute error
2. If rollout outputs differ beyond tolerance:
   - check weight quantizer round-trip error per layer
   - check if first/last layers should be in BF16
   - check KV-cache scale recalibration (must run every RL step)
3. If rollout outputs are acceptable:
   - check if rollout correction (TIS/MIS) is enabled
   - compare corrected vs uncorrected advantage distributions
   - verify training loss tracks BF16 baseline within tolerance
4. If training FP8 is enabled:
   - compare FP8 training loss curve vs BF16 training loss curve
   - check fp8_weight_cache behavior with optimizer step

Tolerances:
- rollout log-prob error: `max_abs < 0.05` per token is typical
- reward at convergence: within 5% of BF16 baseline
- per-step loss: within 10% of BF16 baseline during first 50 steps

## R3 Debug Checklist

Use this after RL algorithm and quantization settings are aligned.

1. Verify recording:
   - router logits captured for every MoE layer
   - correct shape: `[batch_size * seq_len, num_experts]` per layer
   - no NaN or Inf in recorded logits
   - recorded values are in expected range (pre-softmax logits)
2. Verify transfer:
   - all layer indices present in DataProto after `transfer_distributions`
   - tensor dtypes are float32
   - tensors are on CPU (for cross-worker transfer)
3. Verify replay:
   - replayed logits are bit-identical to recorded logits
   - expert assignments match between rollout and training forward pass
   - gradient flows through the replayed routing (not detached)
4. Validate effect:
   - KL divergence with R3 < KL divergence without R3
   - expert load balance stable over training steps
   - no reward collapse over 100+ steps

## Weight Sync Debug Checklist

1. Before sync:
   - record actor model output on a fixed input
   - record state_dict checksum (sum of all param norms)
2. After sync:
   - verify rollout model output changes (weights were actually transferred)
   - verify rollout model output matches actor model output on same input
   - if FP8 on-the-fly conversion: verify quantized weights round-trip correctly
3. For MoE models:
   - verify expert weight ordering matches between training EP layout and inference EP layout
   - check if EP resharding is needed (train_ep_size != infer_ep_size)

## Reward Debug Checklist

1. Verify reward function:
   - run reward function on known inputs with expected outputs
   - check for off-by-one in sequence indexing
2. Verify reward distribution:
   - rewards should have nonzero variance within each generation group
   - check for degenerate rewards (all zeros, all same value)
3. Verify advantage computation:
   - group-relative advantages (GRPO): mean should be ~0 within each group
   - GAE advantages (PPO): verify discount and lambda settings match reference

## What Not To Do

- Do not change already-aligned RL hyperparameters to hide a mismatch.
- Do not declare FP8 parity from a single step. Compare over 50+ steps.
- Do not disable R3 on MoE models without recording routing distributions first.
- Do not claim reward convergence from a reward curve that is still falling.
- Do not debug FP8 precision issues by changing clip ratio or KL coefficient.
- Do not compare runs with different seeds, datasets, or tokenizers.
- Do not skip weight sync verification when debugging output quality.
- Do not run broad hyperparameter sweeps before the first real divergence is localized.
