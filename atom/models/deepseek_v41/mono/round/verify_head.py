# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi path's verify head as ``ModelRunner``'s model-side verify head (``HiDecodeModel.verify_head``).

The hi programs run the LM head inside the target's forward (``hi_runner``: greedy, or one draw a sampled row) and
write each row's pick into a persistent buffer. ``HiVerifyHead`` is the runner's seam onto it:

- ``before_forward``: the step's sampling rows. Only a speculative verify where every request samples (T > 0) with
  top-p alone (no top-k), without logprobs, fan-out noise or relaxed acceptance, is drawn by the head: ATOM's verify of
  such a step is sample-and-match (one independent sample a row, then the greedy match kernel), which the head's
  per-row draws stand in for. Every other step's rows are greedy. Under forced acceptance
  (``--spec-decode-acceptance-length``) the head serves as it does without it (a sampled row's draw), the schedule
  deciding only acceptance.
- ``ids``: this verify's ids from the head, or None (the logits path): a speculative verify whose requests are all
  greedy or all drawn by the head, exact acceptance or forced acceptance; no logprobs, structured output, required
  tool channel, relaxed acceptance or context parallelism (each reads the logits). A replayed graph's ids are the
  buffer its capture recorded (``captured``); an eager step's the head's last.
- ``verdict``: the verdict of a head-served step (``post_verify.verdict``: the in-graph results, or one launch).
- ``ids_identical_on_tp_ranks``: every rank holds the same ids (the head's candidates are exchanged, then every rank
  picks alike), so the runner skips their broadcast.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger("atom")


@functools.cache
def _limits() -> tuple[int, float]:
    """ATOM's relaxed-acceptance top-n and its sampler's greedy threshold (imported on the first sampled step: both
    modules import AITER)."""
    from atom.model_ops.rejection_sampler import RELAXED_TOP_N
    from atom.model_ops.sampler import SAMPLER_EPS

    return RELAXED_TOP_N, SAMPLER_EPS


def _spec():
    """This step's speculative-decode metadata (None: not a verify)."""
    from atom.utils.forward_context import get_forward_context

    return get_forward_context().spec_decode_metadata


def _pcp_world_size() -> int:
    from atom.distributed.pcp_utils import get_pcp_world_size

    return get_pcp_world_size()


class HiVerifyHead:
    """The hook's verify head behind ``ModelRunner``'s seam (module docstring)."""

    ids_identical_on_tp_ranks = True

    def __init__(self, hook) -> None:
        self.hook = hook
        self.sampled = (
            False  # this step's head ids are its samples (``before_forward``)
        )
        self.rows_sampling = False  # the head's rows hold a sampled step's (T, top_p)
        self.graph_ids: dict = (
            {}
        )  # request bucket -> the ids buffer its captured forward wrote

    def _greedy_rows(self) -> None:
        if self.rows_sampling:
            self.hook.set_head_sampling(None, None, None)
            self.rows_sampling = False

    def before_forward(self, runner, batch) -> None:
        """The step's sampling rows, before the forward (module docstring); host work on every greedy step is on the
        critical path: kept to the first test."""
        self.sampled = False
        if batch is None:
            return
        if not batch.temperatures.any():
            self._greedy_rows()
            return
        relaxed_top_n, eps = _limits()
        noise = getattr(batch, "needs_independent_noise", None)
        sampled = (
            _spec() is not None
            and bool((batch.temperatures > eps).all())
            and bool((batch.top_ks == -1).all())
            and not batch.any_logprobs
            and not (noise is not None and bool(noise.any()))
            and relaxed_top_n <= 1
            and getattr(runner, "rejection_sampler", None) is not None
        )
        if sampled:
            self.sampled = bool(self.hook.set_head_sampling(batch.temperatures, batch.top_ps,
                                                            batch.num_scheduled_tokens))  # fmt: skip
            self.rows_sampling = self.sampled
        else:
            self._greedy_rows()

    def ids(self, runner, batch, hidden_states, graph_key=None):
        """This verify's ids from the head, or None (module docstring)."""
        if batch is None:
            return None
        served = bool((batch.temperatures == 0).all()) or self.sampled
        if (
            _spec() is None
            or not served
            or batch.any_logprobs
            or getattr(batch, "structured_requests", None)
            or getattr(batch, "required_channel", None) is not None
            or _limits()[0] > 1
            or getattr(runner, "rejection_sampler", None) is None
            or _pcp_world_size() > 1
        ):
            return None
        ids = (
            self.graph_ids.get(graph_key)
            if graph_key is not None
            else self.hook.head_out()
        )
        if ids is None or ids.shape[0] < hidden_states.shape[0]:
            return None
        return ids[: hidden_states.shape[0]]

    def captured(self, graph_key) -> None:
        """After a graph capture of request bucket ``graph_key``: the ids buffer its recorded forward wrote (None
        when the head did not run in it)."""
        ids = self.hook.head_out()
        self.graph_ids[graph_key] = ids
        if ids is not None:
            logger.info(
                "V4.1 mono hi: verify ids from the head in the graph of bucket %s",
                graph_key,
            )

    def verdict(self, spec, head_ids, sampler):
        """A head-served step's (output ids, verdict, next ids) (``post_verify.verdict``)."""
        from atom.models.deepseek_v41.mono.round import post_verify

        return post_verify.verdict(self.hook, spec, head_ids, sampler)
