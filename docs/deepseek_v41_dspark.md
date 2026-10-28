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

## Dynamic verification and its cost curve

Dynamic verification requires `DSparkConfig(confidence_schedule=True,
ragged=True)`. No V4 SPS curve or synthetic cost stub is admitted for V4.1: the
scheduler needs a measured **SPS table**, mapping total query rows to
target-forward throughput. It comes from one of two places.

**Measured at startup (default).** Right after the decode token graphs are
captured, before the server accepts a request, every captured (batch, q) decode
graph is replayed and timed, as DeepInfer's startup `profile_decode_perf` does
(`atom/spec_decode/startup_sps.py`). The replays use the synthetic metadata
capture itself used, so no request state exists for them to disturb. Each
shape takes 3 warmup and 15 timed target forwards (including eager regions and
the LM head); the median is reduced by MAX over TP and DP, since the slowest
rank sets the step. The points become a conservative monotone latency envelope,
interpolated per row; rows past the largest recorded graph (a DP graph cap)
extend along the last measured slope. This requires `TR_HYBRID` token graphs.
The sweep is cached in `ATOM_DSPARK_SPS_CACHE_DIR` (default
`~/.cache/atom/dspark_sps`, empty disables) under a key of the deployment
identity below plus the timed shapes; a restart of the same deployment reuses
it. All ranks agree on hit or miss before the sweep, since it runs collectives.
Without a profile, the confidence temperatures are fitted online
(`online_sts`, default on; `atom/spec_decode/online_sts.py`): the TP leader
async-copies sampled draft confidence, pairs it two forwards later with the
deferred verdict (verified and accepted drafts), and labels only verified
positions -- unverified ones are censored, not rejections. A background thread
refits per-position temperatures by censored cumulative-survival ECE, each
position only past 3000 verified observations and pulled toward T=1 as
1/sqrt(n); every 256 forwards the TP leader broadcasts the result over its TP
group, so every TP worker sizes verification identically. Each DP rank fits on
its own traffic (its temperatures only size its own requests). Until then T=1.

The startup context is the capture context (about 128 tokens), so the curve is
a short-context target-only baseline; `online_sps` adds what serving adds.

**An explicit `calibration_profile`** takes precedence. It is main's existing
interface for externally produced measurements (an SPS table and per-position
**confidence temperatures**); no in-tree tool produces one, and startup SPS
plus online STS are the supported path. A profile is validated against the
deployment identity: model and index-file hashes, model type, TP size, cache
types, graph setting, GPU name and Torch/HIP versions, plus proposal width,
batch coverage, positive finite temperatures and a non-increasing SPS table.
Profiles in main's format, which carry no DP/EP layout or graph mode, load on
plain TP only; a field a profile does carry must match.

### Online SPS correction

Set `online_sps=true` in `--dspark-config` (a `calibration_profile` is optional):

```json
{
  "confidence_schedule": true,
  "ragged": true,
  "ragged_graph_sizes": "1,3,6",
  "online_sps": true
}
```

The runtime keeps the measured startup curve and corrects its latency using
completed serving steps:

`latency_ms(rows, requests) = baseline_ms(rows) + fixed_ms(requests) + slope_ms(requests) * rows + post_ms(requests)`.

A size-independent residual is a fixed offset, not a marginal per-row cost.
The additional slope is a marginal per-row cost, so it is fitted only from
contrasts that differ in nothing but the row count: the same batch size and
the same request set (context generation), where the graph size changed with
the verify cap. Different batches or request sets carry different fixed
overheads (more requests, longer KV), which would otherwise read as per-row
cost. Each (batch, context, graph size) gives a median residual; the slope is
the median of the pairwise slopes inside each (batch, context), clamped
nonnegative, needing at least three distinct sizes and ten observations. A
batch with its own contrasts gets its own slope; others share the slope pooled
from every batch's within-context pairs. Without contrasts the slope is zero
and the measured startup curve supplies the marginal costs. Unlike DeepInfer's
`ForwardRuntimeEstimator`, which divides each residual by its graph size, a
constant offset never becomes a per-row cost.

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
of completed samples on every TP/DP worker, using maximum worker times for each phase.
Samples must be at least two decode steps old. Pending events are bounded and
never reused before completion. Prefill, capture and eager forwards do not
update the estimator. A TP-only dummy is excluded; under DP, an idle worker's
dummy participates in the group's decode timing and publication. Mixed P/D
steps are excluded on every worker. `DSpark online SPS` log entries report versions,
sample counts and the fitted correction.

This implementation supports TP, DP, DP attention and EP, with PP/PCP/DCP=1,
TR_HYBRID token graphs and TBO disabled. Select `--cudagraph-mode TR_HYBRID`
explicitly: the V4.1 default remains FULL, which does not apply ragged lengths.
On this main revision V4.1 admits TR_HYBRID at compilation level 0. Token-graph
record/replay is independent of torch.compile; level 3 still admits FULL only.
The online runtime rejects that combination. Profiles bind both the graph mode
and compilation level, as well as DP size, DP attention, EP and its backend;
regenerate the SPS measurements when changing them. Legacy TP profiles are
accepted only for the original plain-TP topology.
The default TP confidence policy retains early stopping. Experimental
`bucket_global=true` searches all prefix budgets and configured per-request q
caps using actual captured decode graph sizes, including q divisibility. Costs
use those graph rows rather than logical token counts. Uncovered candidates are
excluded. After current bonus floors are applied, a contraction that does not
reduce the executable graph is discarded. Both policies preserve the fixed
two-generation delayed per-request adoption and all boundary/state guards.
This is not DeepInfer's separate N−2 budget / N−1 allocation policy. Under TP,
per-request allocations remain two generations old. Nearby, same-request shape
contrasts reduce context confounding but are observational, not identical-KV replays or
a complete model of V4.1 attention cost. Snapshots expose batch-local slopes,
support counts, fixed offsets and post-target costs. `slope_ms_per_row` is only
an aggregate diagnostic; the decision table uses batch-local coefficients.

### Joint DP verification budget

With DP enabled, either `online_sps` or `bucket_global` opts into the joint
planner. Each TP group's leader asynchronously copies cumulative survival
probabilities; the planner reads exactly forward N−2, never the latest ready
copy. It rebuilds candidate allocations using the current request set and
mandatory bonus-token floors. No current draft tokens or confidence are used.

Each rank first applies its own ragged budget (the same one used without DP);
the group then runs the graph its busiest rank needs. The joint plan only fills
that graph: among allocations whose common executable graph is exactly the
fallback's, it takes the one with the most summed expected output. The step
costs the same, and ranks that shrank on their own get back drafts the busiest
rank is paying for regardless. A joint shrink below the fallback is not taken:
measured on DP2 AgentX it lost more accepted tokens (3.13 -> 2.99 per forward)
than the smaller graph saved. Each rank's frontier rides `ForwardMode`'s one
per-step DP exchange (`sync_dp_metadata`); every rank resolves the same plan
from the gathered rows and rewrites its own batch to its share. The plan adds
no DP collective of its own. Idle dummies retain their reserved shape and
contribute zero expected output tokens.

The planner's payload is built on every forward, including idle dummies,
middle prefill chunks and mixed P/D steps; under TP>1 its delayed confidence is
broadcast from the TP leader. A missing N−2 record or a request-set
boundary keeps the full layout for that group step; it does not reuse stale
confidence. Pending copies retain their source and destination until completion
and are bounded to four entries. This broadcast and CPU planning add
overhead; they are outside the target-event timing interval and must be included
when evaluating end-to-end performance.

The online DP cost curve uses the maximum local real request count as its batch
key, common executed rows, and phase maxima across all TP/DP workers. A change
in any rank's request-set generation or real/idle/prefill state changes the
group generation used for slope fitting. This is an approximate group cost
model, not a full predictor of the per-rank KV/context distribution.

`online_sps_observe_only=true` keeps sampling, TP/DP publication and cost-table
rebuilding active while using the static cost curve for decisions. Under DP,
the same joint planner stays active so that only online cost correction is
ablated. It requires `online_sps=true`. The post-target nine-observation
window is unchanged; it has not been converted to a nine-step window. Online
SPS and bucket-global decisions both remain disabled by default pending SA
latency and throughput validation.

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
Dynamic verification measures its cost curve at startup on TR_HYBRID token
graphs (cached per deployment) unless an external profile is supplied; without
one, confidence temperatures are fitted online.
Quality equivalence to non-speculative generation and a general throughput
improvement have not been established.

The three additional draft windows and ring slack consume more request-state
memory. Draft execution also adds work before the first output token. Measure
TTFT, decode latency, throughput and memory on the intended workload using
[the serving benchmark](serving_benchmarking_guide.md), with matched prompt
formatting, sampling settings and generation limits. Favorable acceptance
rates alone do not establish a speedup.
