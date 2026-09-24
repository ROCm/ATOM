# DeepSeek-V4.1 DSpark

Native DSpark uses the checkpoint's three draft stages to propose five tokens.
Target verification processes the anchor plus up to five proposals and commits
only the accepted input prefix. It reuses ATOM's DSparkProposer, VerifyScheduler,
request scheduler and unchanged V4 BF16 attention kernels.

**Quality and performance acceptance remain incomplete.** Fixed-width request
lifecycle support is implemented; numerical equivalence, calibrated scheduling
and throughput require validation for the deployment workload.

## Supported configuration

DSpark requires BF16 KV, the FP8 index plane and text requests. Tensor
parallelism follows the model's dimension-divisibility checks, with no TP4-only
admission gate. Configuration tests cover TP1/2/4/8; the GPU benchmark evidence
covers TP2 and TP4 without EP at level 3 FULL. See the
[AgentX recipe](../recipes/DeepSeek-V4.1-Flash-Agentic.md) for commands and the
fixed acceptance length used in those measurements.

The existing quality/development baseline uses TP4 with whole-expert EP at
compilation level 0. Target execution can be eager or use whole-forward decode
graphs (`FULL`); the draft has its own graph. The runtime also accepts
`PIECEWISE` at level 0, and level 3 with FULL graphs or eager execution.

Fixed acceptance schedules use the common `--spec-decode-acceptance-length`
or `--spec-decode-acceptance-rate` options and the shared rejection sampler.

Not admitted: packed speculative caches, multimodal speculation,
relaxed MTP acceptance, and speculative output token logprobs — the
current output protocol cannot return the last of these correctly. Both target
and draft MoE reuse V4 FusedMoE. Non-speculative vision and packed cache support
are independent of all of this.

Fixed-length verification is the default when DSpark is explicitly selected:

```python
from atom.config import CompilationConfig, Config, CUDAGraphMode, SpeculativeConfig

model = "/mnt/DeepSeek-V4.1-Flash"
config = Config(
    model=model,
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    kv_cache_dtype="bf16",
    index_cache_dtype="fp8",
    max_num_seqs=4,
    max_num_batched_tokens=512,
    max_model_len=4096,
    enforce_eager=True,
    compilation_config=CompilationConfig(level=0),
    speculative_config=SpeculativeConfig(
        method="dspark", model=model, num_speculative_tokens=5,
    ),
)
```

For target graphs, set `enforce_eager=False` and
`compilation_config=CompilationConfig(level=0, cudagraph_mode=CUDAGraphMode.FULL)`.
A decode step is then two replays: the draft's, keyed by request count, and the
target's, keyed by `(batch size, query bucket)`. `PIECEWISE` remains available
and records the dense pieces only. Capture must not bind warmup slots or another
request's window; `validate_runtime --graph` is what asserts it does not.

## State and sampling contracts

- Target layer **inputs** 37, 38 and 39 contribute the mean of the four residual
  streams. The target embedding and output head are shared with the drafter.
- Draft blocks use 128 experts/top-3, Markov rank 256 and noise token 128799;
  target blocks retain 384 experts/top-6. The draft block is bidirectional over
  its own five positions and attends to the valid target context window.
- There are 43 request-owned windows: 40 target and three draft context windows.
  Logical visibility remains 128; physical rings have 133 rows so every possible
  accepted prefix retains its preceding window after six-row verification.
- Compressor tails and compressed Engram histories are staged per input prefix.
  Committing zero draft tokens still commits the anchor input. Checkpointing and
  drafting require the tentative state to have been resolved.
- Preemption replays only finalized host token IDs, excluding deferred output and
  draft placeholders. A replayed prefill discards stale deferred results for the
  same request ID. Output timing likewise counts finalized tokens only.
- Greedy verification matches target argmax. Stochastic verification draws from
  each request's target distribution with independent noise per row, accepts
  matching proposals, and stops at the first mismatch. Draws are broadcast before
  rejection so all TP ranks commit the same prefix. This preserves the target
  sampling distribution; it is not probability-ratio rejection sampling and
  does not promise the latter's acceptance rate.

Sampling and accepted-state contracts are defined relative to the target
logits. They do not promise identical free-running generations across batch
shapes: BF16 reduction and quantization boundaries can change close logits.
Integer state, accepted-prefix and output ownership contracts remain exact.

Model math lives in `models/deepseek_v41/dspark.py` and
`model_ops/deepseek_v41/dspark.py`. Tentative state belongs to the V4.1 cache
backend. Generic speculation code owns proposal execution, confidence scheduling
and sampling, without Engram hash or CSA2 compression formulas.

## Dynamic verification and its calibration profile

Dynamic verification requires `DSparkConfig(confidence_schedule=True,
ragged=True, calibration_profile="/path/to/profile.json")`. No V4 SPS curve or
synthetic cost stub is admitted for V4.1, and an explicit profile takes
precedence over automatic warmup calibration.

A profile carries two fitted objects. **Confidence temperatures** rescale the
drafter's per-position survival probabilities; they are fitted against
cumulative-survival ECE on held-out blocks. **An SPS table** maps total query
rows to target-forward throughput, aggregated from max-rank CUDA-event medians
and converted to a conservative monotone latency envelope before interpolation.
This one-dimensional cost model does not capture context or routing variation.

Loading validates the config and index-file hashes, model type, TP size, cache
types, expert backend, graph setting, GPU name and Torch/HIP versions, plus
proposal width, batch coverage, positive finite temperatures and a non-increasing
SPS table. Those hashes identify checkpoint metadata, not every weight byte.
Recalibrate after a model or runtime change, or for a materially different
workload; a profile is an optional deployment artifact, not a model default.

The offline fitting harness is not shipped with the repository. Supply a profile
calibrated for the intended deployment and keep requests within its batch and
query coverage.

## Validation

Use the environment described in [the runtime guide](deepseek_v41_runtime.md).
The manual lifecycle validator uses production configuration admission and
compilation level 0:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m tests.models.deepseek_v41.validate_dspark_lifecycle \
  --production --graph --sampling-probe \
  --output /tmp/dspark-lifecycle.json
```

Omit `--graph` for eager target execution. The validator aligns the checkpoint
interval to the prefix hash block and uses prompts that cross a stored
checkpoint. It exercises stochastic requests, preemption, reorder, prefix forks,
cancellation, slot reuse, output limits and stop/EOS inside accepted blocks.
Restore and accepted-prefix assertions must remain enabled.

Automated coverage lives in `tests/models/deepseek_v41/` and
`tests/attentions/deepseek_v41/`. Request-state checks are separate from
generation-quality and throughput evaluation.

## Current limitations

Fixed-width DSpark is the default when speculation is explicitly selected.
Dynamic verification requires a deployment-specific calibrated profile.
Quality equivalence to non-speculative generation and a general throughput
improvement have not been established.

The three additional draft windows and ring slack consume more request-state
memory. Draft execution also adds work before the first output token. Measure
TTFT, decode latency, throughput and memory on the intended workload using
[the serving benchmark](serving_benchmarking_guide.md), with matched prompt
formatting, sampling settings and generation limits. Favorable acceptance
rates alone do not establish a speedup.
