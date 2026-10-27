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
  - ``bs``   the scheduled (real) batch size; on the CUDAGraph path shown as
             ``<real>/<graph>`` when the replayed graph is padded above the real
             batch (e.g. ``bs=117/128``). ``parse_trace.py`` reads the leading
             ``\\d+`` so the real batch is still what it extracts.
  - ``tok``  this rank's scheduled tokens; shown as ``<scheduled>/<running>``
             when the forward runs padded rows past them (a token-graph
             bucket, a DP peer's height, a decode graph's rectangle).
  - ``ctx``  per-seq context lengths (prefill/eager paths; truncated if many)
  - ``p``/``d``  prefill / decode seq counts (cudagraph decode path)
  - ``spec`` speculative steps (when > 0)
  - ``dp``   under DP: every rank's scheduled tokens and phase (``p``/``d``),
             this rank's marked ``*``
  - ``tbo=1`` appended when the step ran Two-Batch-Overlap ubatches

NOTE: ``tools/parse_trace.py`` selects real steps via ``startswith("prefill[")``
/ ``startswith("decode[")``. The ``dummy_`` / ``eager_`` prefixes deliberately
fall outside those, so dummies never pollute the prefill/decode statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atom.model_engine.scheduler import ScheduledBatch


def build_run_label(
    *,
    is_prefill: bool,
    use_cudagraph: bool,
    is_dummy: bool,
    tbo_on: bool,
    scheduled_bs: int,
    batch: ScheduledBatch | None,
    running_bs: int,
    detailed_suffix: str = "",
    running_tokens: int | None = None,
    dp: tuple[tuple[int, bool], ...] | None = None,
    dp_rank: int = 0,
) -> str:
    """Build the ``record_function`` label for one forward pass.

    Pure function (no runtime state) — see module docstring for the taxonomy.

    ``detailed_suffix`` is an already-formatted ``" key=value ..."`` fragment
    (empty on the normal path) appended inside the brackets; see
    ``ModelRunner._detailed_label_suffix``.

    Both batch sizes are required: every step has both, and a step that ran
    padded is exactly the one worth seeing in a trace. The cudagraph path
    reports ``running_bs`` -- what actually ran -- and prefixes the real
    ``scheduled_bs`` as ``bs=<real>/<graph>`` when the two differ. The eager
    path reports ``scheduled_bs``, because eager attention runs the scheduled
    rows even where MoE pads past them.

    ``running_tokens`` is the height the forward runs (``tok``'s second
    number); ``dp`` every DP rank's ``(scheduled_tokens, is_prefill)`` and
    ``dp_rank`` this rank's index in it -- what a decode beside a prefilling
    peer was padded for.
    """
    if use_cudagraph:
        kind = "dummy_decode" if is_dummy else "decode"
        if running_bs > scheduled_bs:
            label = f"{kind}[bs={scheduled_bs}/{running_bs}"
        else:
            # `decide` ceils to a captured size, so not-greater means equal.
            label = f"{kind}[bs={running_bs}"
        if batch is not None:
            label += f" tok={_tokens(batch.total_tokens_num, running_tokens)}"
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
        label = f"{kind}[bs={scheduled_bs}"
        if batch is not None:
            ctx = batch.context_lens
            if len(ctx) == 1:
                ctx_str = str(ctx[0])
            elif len(ctx) <= 5:
                ctx_str = str(ctx.tolist())
            else:
                ctx_str = f"{ctx[:3].tolist()}...+{len(ctx) - 3}"
            tok = _tokens(batch.total_tokens_num, running_tokens)
            label += f" tok={tok} ctx={ctx_str}"
            if is_prefill and not is_dummy:
                # Attention work of the chunk: sum over seqs of new tokens x the
                # mean context they attend (prior context + half the chunk).
                n = getattr(batch, "num_scheduled_tokens", None)
                if n is not None and len(n) == len(ctx):
                    n = n.astype("float64")
                    span = (n * (ctx - n / 2)).sum()
                    label += f" nctx={span / 1e6:.1f}M"
    if dp is not None:
        ranks = ",".join(
            f"{tokens}{'p' if prefill else 'd'}{'*' if rank == dp_rank else ''}"
            for rank, (tokens, prefill) in enumerate(dp)
        )
        label += f" dp=[{ranks}]"
    label += detailed_suffix
    if tbo_on:
        label += " tbo=1"
    label += "]"
    return label


def _tokens(scheduled: int, running: int | None) -> str:
    """``<scheduled>/<running>`` where the forward runs padded rows, else the
    one number."""
    if running is None or running <= scheduled:
        return str(scheduled)
    return f"{scheduled}/{running}"
