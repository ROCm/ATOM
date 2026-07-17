"""Profiler ``record_function`` label taxonomy for forward passes.

Kept in its own dependency-free module so the label contract is unit-testable
without importing the heavy ``model_runner`` chain (aiter kernels, config, …).

Kinds (label prefix — groups in Perfetto UI, greppable):
  - ``prefill``             real prefill (eager)
  - ``decode``             real decode via CUDAGraph
  - ``eager_decode``       real decode forced eager
  - ``dummy_decode``       DP-sync dummy (CUDAGraph path)
  - ``dummy_eager_decode`` DP-sync dummy forced eager
  - ``dummy_prefill``      warmup dummy prefill

Fields (``key=value`` inside brackets):
  - ``bs``   effective batch size
  - ``tok``  total scheduled tokens
  - ``ctx``  per-seq context lengths (prefill/eager paths; truncated if many)
  - ``p``/``d``  prefill / decode seq counts (cudagraph decode path)
  - ``spec`` speculative steps (when > 0)
  - ``tbo=1`` appended when the step ran Two-Batch-Overlap ubatches

NOTE: ``tools/parse_trace.py`` selects real steps via ``startswith("prefill[")``
/ ``startswith("decode[")``. The ``dummy_`` / ``eager_`` prefixes deliberately
fall outside those, so dummies never pollute the prefill/decode statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from atom.model_engine.scheduler import ScheduledBatch


def build_run_label(
    *,
    is_prefill: bool,
    use_cudagraph: bool,
    is_dummy: bool,
    tbo_on: bool,
    bs: int,
    batch: Optional["ScheduledBatch"],
) -> str:
    """Build the ``record_function`` label for one forward pass.

    Pure function (no runtime state) — see module docstring for the taxonomy.
    """
    if use_cudagraph:
        kind = "dummy_decode" if is_dummy else "decode"
        label = f"{kind}[bs={bs}"
        if batch is not None:
            label += f" tok={batch.total_tokens_num}"
            if batch.total_seqs_num_prefill > 0:
                label += f" p={batch.total_seqs_num_prefill}"
            label += f" d={batch.total_seqs_num_decode}"
            if batch.num_spec_step > 0:
                label += f" spec={batch.num_spec_step}"
    else:
        if is_prefill:
            kind = "dummy_prefill" if is_dummy else "prefill"
        else:
            kind = "dummy_eager_decode" if is_dummy else "eager_decode"
        label = f"{kind}[bs={bs}"
        if batch is not None:
            ctx = batch.context_lens
            if len(ctx) == 1:
                ctx_str = str(ctx[0])
            elif len(ctx) <= 5:
                ctx_str = str(ctx.tolist())
            else:
                ctx_str = f"{ctx[:3].tolist()}...+{len(ctx) - 3}"
            label += f" tok={batch.total_tokens_num} ctx={ctx_str}"
    if tbo_on:
        label += " tbo=1"
    label += "]"
    return label
