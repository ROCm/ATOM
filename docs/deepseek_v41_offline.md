# DeepSeek-V4.1 offline text development baseline

The eager text backbone executes all 40 layers, including Engram and the
checkpoint's FP8/FP4 quantization steps. The paged ModelRunner integration is
described in [the runtime guide](deepseek_v41_runtime.md). This offline entry
point remains the development and numerical comparison baseline.

## Run

Use a ROCm environment with ATOM and AITER installed and the native checkpoint
available locally. The latest complete independent numerical comparison uses
MI355X and TP4. TP8 also has parameter-layout and collective regression coverage.

```bash
torchrun --standalone --nproc_per_node=8 \
  -m atom.examples.deepseek_v41_offline \
  --model /mnt/DeepSeek-V4.1-Flash \
  --prompt "The capital of France is" \
  --max-length 256 --max-new-tokens 16
```

The example accepts one raw text prefix. It owns the parallel group, prepares
Engram rows through the host provider, and maintains a private eager cache.
It selects RCCL collectives through AITER's initialization API because the
current custom collective path fails repeated-input checks for this workload.
This entry point uses a private eager cache. Paged batches and optional graph
execution are available through ModelRunner; chat/tool encoding, vision and
speculative decoding remain separate milestones.

## Equal-score index selection

`text_config.index_topk_tie_break` accepts `small_position` (default) or
`large_position`. The same policy applies to CSA2 top-k, candidate blocks and
Reindex. The newest visible candidate block is always retained; selected token
positions are returned in ascending order under either policy.

Use `--index-topk-tie-break large_position` to override the checkpoint config for
an offline run. The model captures the selected policy at construction time.

## Implementation boundaries

- `atom/models/deepseek_v41/` composes projections, mHC, attention and experts.
- `atom/model_ops/deepseek_v41/` owns rotary, compressor, indexer and MoE math.
- `atom/model_ops/attentions/deepseek_v41_state.py` owns the eager SWA,
  incomplete compression groups and one global main/index allocation per owner.
- `atom/examples/deepseek_v41_offline.py` owns checkpoint loading and request
  preparation. Token hashing and table lookup use the existing Engram runtime.

The four global cache owners are layers 2, 8, 14 and 20. Reuse layers share
their owner's values and top-k; Reindex layers score only the supplied compact
candidate blocks. All attention caches contain BF16 values after the original
FP8/FP4 quantization and dequantization steps. Attention calls the existing V4
BF16 paged prefill/decode entry points with unchanged dispatch and kernels.
The cache owns a fixed BF16 row pool: one SWA ring per layer and one global
region per owner. Prefill reads the prior ring and current chunk separately;
decode writes its current row before reading the pool. Both use V4's existing
ring writer. No forward step concatenates the complete global KV history.
`atom/model_ops/deepseek_v41/paged_indices.py` prepares the caller's CSR indices;
it contains no attention computation.

Weights retain native FP8 32x32 or FP4 1x32 storage. The correctness GEMM converts
register tiles for BF16 MFMA and keeps scaled block sums in FP64 until output
conversion, retaining FP8 activation quantization. V4.1 RMSNorm uses the
published FP32 evaluation order at the intermediate quantization boundaries.
Index-key normalization uses a small V4/AITER leaf after complete task
regression; final normalization also uses V4/AITER. Routed experts use
whole-expert partitioning; the shared expert uses TP with FP32
partials and rounds after reduction. The eager dispatch still synchronizes
expert counts to the CPU. The ModelRunner path adds optional packed caches and
AITER A8W4 expert dispatch with a user-accepted precision change. Dense native
FP8 MFMA remains disabled. See the [P09 report](deepseek_v41_performance.md).

## Validation and open gates

Reference model sources are pinned to revision
`dba1be0a40aa45a94ad051997016db3960a90277`; tests verify their hashes.

```bash
ATOM_DSV41_REFERENCE=/mnt/DeepSeek-V4.1-Flash \
  python -m pytest -q tests/models/deepseek_v41
```

Operator tests compare with independent PyTorch contracts and pinned upstream
methods. Attention integration fixtures retain the real 512-dimensional heads
and eight local heads. Cache tests cover Full/Reuse/Reindex ownership, causal
visibility, ring wraparound, separate requests and batch rows, and ratio-2
boundaries. The actual V4 BF16 prefill/decode tests cover a single sink across
SWA/global representations and 192/640 selected entries.

Quantized graph tests first compare each real backend's output with the upstream
attention output (relative L2 <= 0.003), then supply the upstream output to the
remaining projections. This controls accepted BF16 rounding before discontinuous
FP8 quantization and isolates model composition. These tests do not establish
end-to-end numerical parity. The production model has no such substitution.

An isolated run of the official TileLang HIP attention also checked the
unmodified V4 kernels at 5/192/640 entries. The maximum observed relative L2 was
0.002461 for decode and 0.001203 for prefill across those cases.

The independent resident-model comparison and paired lm-eval results are
recorded in [deepseek_v41_validation.md](deepseek_v41_validation.md), together
with reproducible commands and the remaining quality gate. These runs execute
all 40 layers with real checkpoint weights, Engram and all QAT operations.

The numerical corpus includes a 2,049-token case crossing top-512 selection;
separate indexer tests exercise 32,771 keys and actual candidate-block pruning.
This is not a full-model 32K or 1M-context validation. The serving request
lifecycle is covered by the completed P05 validation.
