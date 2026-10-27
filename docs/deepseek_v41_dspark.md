# DeepSeek-V4.1 DSpark

Native DSpark uses the checkpoint's three draft stages to propose five tokens.
Target verification processes the anchor plus up to five proposals and commits
only the accepted input prefix. It reuses ATOM's DSparkProposer, VerifyScheduler,
request scheduler and unchanged V4 BF16 attention kernels.

**Quality and performance acceptance remain incomplete.** Fixed-width request
lifecycle support is implemented; numerical equivalence, calibrated scheduling
and throughput require validation for the deployment workload.

## Supported configuration

DSpark requires an FP8 (the deployed configuration) or BF16 KV cache and text
requests. The index plane is FP8 by default or FP4. Tensor
parallelism follows the model's dimension-divisibility checks, with no TP4-only
admission gate. Configuration tests cover TP1/2/4/8; the GPU benchmark evidence
covers TP2 and TP4 without EP at level 3 FULL. See the
[AgentX recipe](../recipes/DeepSeek-V4.1-Flash-Agentic.md) for commands and the
fixed acceptance length used in those measurements.

Data-parallel attention is also supported, on the same KV caches and index
planes. Text prefill can use TBO with tensor-sharded experts or the
validated DPA4 + MORI EP4 high-throughput configuration (16 GiB MoRI heap);
see the [DPA and TBO configuration](deepseek_v41_runtime.md#data-parallel-attention).

The existing quality/development baseline uses TP4 with whole-expert EP at
compilation level 0. Target execution can be eager or use whole-forward decode
graphs (`FULL`); the draft has its own graph. The runtime also accepts
`TR_HYBRID` at level 0, and level 3 with FULL graphs or eager execution.

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
    kv_cache_dtype="fp8",
    index_cache_dtype="fp4",
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
target's, keyed by `(batch size, query bucket)`. `TR_HYBRID` remains available
and records token graphs only. Capture must not bind warmup slots or another
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
cumulative-survival ECE on training blocks and evaluated on held-out request
hashes. **An SPS table** maps total query
rows to target-forward throughput, aggregated from max-rank CUDA-event medians
and converted to a conservative monotone latency envelope before interpolation.
This one-dimensional cost model does not capture context or routing variation.

Loading validates the config and index-file hashes, model type, TP size, cache
types, graph setting, GPU name and Torch/HIP versions, plus
proposal width, batch coverage, positive finite temperatures and a non-increasing
SPS table. Those hashes identify checkpoint metadata, not every weight byte.
Recalibrate after a model or runtime change, or for a materially different
workload; a profile is an optional deployment artifact, not a model default.

### Building a profile

`atom.benchmarks.profile_dspark_sps` measures actual target-forward GPU time in
a disposable model process. The initial harness uses BF16 KV, FP4 index,
compilation level 3 FULL, batch coverage 128, and query buckets 1/3/6. Its
128-token startup context is a baseline, not a long-context latency measurement.
The profiler exits after measurement; its synthetic cache is never used to
serve requests. Match these settings to the intended deployment.

```bash
torchrun --standalone --nproc-per-node=2 \
  -m atom.benchmarks.profile_dspark_sps \
  --model /mnt/DeepSeek-V4.1-Flash --output /tmp/sps-measured.json
```

For STS observations, start a **fixed-width** V4.1 server with
`ATOM_DSPARK_CALIBRATION_DIR=/tmp/dspark-confidence`, then replay the deployment
workload. Only TP rank zero records numeric confidence, accepted draft length,
and a hash of the request ID. It pairs an observation with that same request's
next verification, including zero acceptance. Every eighth step is sampled.
Collection copies sampled tensors to CPU and synchronizes: do not use this run
as a performance baseline. Separate directories are required for each process.
The data contains no prompts or token IDs; request hashes do not identify
conversation trajectories, so the holdout is not a trajectory-disjoint split.

```bash
python -m atom.benchmarks.fit_dspark \
  --sps-measurements /tmp/sps-measured.json \
  --confidence /tmp/dspark-confidence/confidence.jsonl \
  --output /tmp/dspark-profile.json
```

The fitter rejects mismatched deployment identities, insufficient holdout
coverage, and incomplete measured row coverage. Inspect training and held-out
ECE before adopting a profile. Keep requests within its batch/query coverage.

### Online SPS correction

Set `online_sps=true` in `--dspark-config` alongside the calibrated profile:

```json
{
  "confidence_schedule": true,
  "ragged": true,
  "ragged_graph_sizes": "1,3,6",
  "calibration_profile": "/tmp/dspark-profile.json",
  "online_sps": true
}
```

The runtime keeps the measured startup curve and corrects its latency using
completed serving steps:

`latency_ms(rows, requests) = baseline_ms(rows) + fixed_ms(requests) + slope_ms(requests) * rows + post_ms(requests)`.

A size-independent residual is a fixed offset, not a marginal per-row cost.
Additional slope is learned only from different executed graph sizes at the
same request count and request-set generation within one publication-sized
decode-step cohort (64 steps by default). Each size needs two observations;
the batch needs ten supporting observations across eligible cohorts. Adjacent
per-size residual medians produce slopes, whose median is clamped nonnegative.
With no shape diversity or support, the additional slope is zero and the
measured startup curve still supplies its original marginal costs. Positive
slopes are never borrowed from another batch.

The estimator retains the latest 5000 eligible decode steps
(`online_sps_window_steps`), approximately 625 samples at the default cadence.
Every publication expires all components, including fixed and post-target
history, even when no new timing is ready. Fixed offsets are residual medians
after removing the supported local slope. Fixed and post-target costs use nine
recent observations per request count, at least five samples, and global fixed
fallbacks. All corrections are nonnegative to preserve the startup floor and
monotonicity. Post-target timing includes state commit, sampling and drafting
on the forward stream; it excludes output-copy and HTTP latency. Target timing
still begins after `prepare_model`; this model does not measure preparation
or CPU scheduling cost. STS temperatures remain fixed.

By default, `online_sps_sample_interval=8` and
`online_sps_update_interval=64`. Events are queried without an added CUDA
synchronize. A synchronous, small CPU Gloo collective publishes the intersection
of completed samples on every TP rank, using maximum rank times for each phase.
Samples must be at least two decode steps old. Pending events are bounded and
never reused before completion. Prefill, dummy runs, capture and eager forwards
do not update the estimator. `DSpark online SPS` log entries report versions,
sample counts and the fitted correction.

This implementation currently requires TP-only execution, DP/PP/PCP/DCP=1,
TR_HYBRID token graphs, no EP and no TBO. Select `--cudagraph-mode TR_HYBRID`
explicitly: the V4.1 default remains FULL, which does not apply ragged lengths.
On this main revision V4.1 admits TR_HYBRID at compilation level 0. Token-graph
record/replay is independent of torch.compile; level 3 still admits FULL only.
The online runtime rejects that combination. Profiles bind both the graph mode
and compilation level; regenerate the SPS measurements when changing them.
The default confidence policy retains early stopping. Experimental
`bucket_global=true` searches all prefix budgets and configured per-request q
caps using actual captured decode graph sizes, including q divisibility. Costs
use those graph rows rather than logical token counts. Uncovered candidates are
excluded. After current bonus floors are applied, a contraction that does not
reduce the executable graph is discarded. Both policies preserve the fixed
two-generation delayed per-request adoption and all boundary/state guards.
This is not DeepInfer's separate N−2 budget / N−1 allocation policy; per-request
allocations remain two generations old. Nearby, same-request shape contrasts
reduce context confounding but are observational, not identical-KV replays or
a complete model of V4.1 attention cost. Snapshots expose batch-local slopes,
support counts, fixed offsets and post-target costs. `slope_ms_per_row` is only
an aggregate diagnostic; the decision table uses batch-local coefficients.

`online_sps_observe_only=true` keeps sampling, TP publication and cost-table
rebuilding active while using the static cost table for decisions. It is an
ablation option and requires `online_sps=true`. The post-target nine-observation
window is unchanged; it has not been converted to a nine-step window. Online
SPS and bucket-global decisions both remain disabled by default pending SA
latency and throughput validation.

An example V4.1 TP2 measurement, including counter definitions and limits, is
recorded in [the online SPS SA comparison](deepseek_v41_dspark_online_sps_results.md).

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
