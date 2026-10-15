# SPDX-License-Identifier: MIT
"""Adapt V4.1 attention metadata to the shared piecewise AF runner."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from atom.config import CUDAGraphMode
from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep
from atom.utils.cuda_graph import StableOutputs
from atom.utils.piecewise_af import PiecewiseAFRunner, SharedGraphInputs

if TYPE_CHECKING:
    from .attention import Attention
    from .bounded_replay import QueryRows

_STEP_TENSORS = (
    "positions",
    "cu_seqlens_q",
    "slots",
    "batch_ids",
    "block_tables",
    "swa_replay_start",
    "swa_cached_until",
)
_PLAN_TENSORS = (
    "compress_plan_gpu",
    "write_plan_gpu",
    "key_rope_positions_gpu",
)


@dataclass
class AttentionCapture:
    """Stable inputs and Python-visible results of one attention recording.

    Warmup executes the live step; recording creates the tensors replay writes.
    Keep both results so the first call publishes warmup state, and later calls
    publish the recorded state without repeating the attention's KV writes.
    """

    step: BatchStep
    queries: QueryRows | None
    warmup: BatchStep | None = None
    recorded: BatchStep | None = None

    @contextmanager
    def bind(self) -> Iterator[dict]:
        # Rebuild derived indices in both warmup and recording. Selections from
        # upstream layers remain inputs, while this layer's writes stay private.
        step = replace(
            self.step,
            selected=self.step.selected.copy(),
            candidates=self.step.candidates.copy(),
            tiles={},
            candidate_blocks={},
            group_indices={},
            window_rows=None,
            af_inputs={},
        )
        queries = self.queries
        if queries is not None:
            queries = replace(
                queries,
                index=replace(
                    step,
                    batch_ids=queries.index.batch_ids,
                    indptrs=queries.index.indptrs,
                ),
            )
        yield {"step": step, "queries": queries}
        if self.warmup is None:
            self.warmup = step
        self.recorded = step

    def publish(self, step: BatchStep, layer_id: int, *, first_call: bool) -> None:
        source = self.warmup if first_call else self.recorded
        assert source is not None
        if layer_id in source.selected:
            step.selected[layer_id] = source.selected[layer_id]
        if layer_id in source.candidates:
            step.candidates[layer_id] = source.candidates[layer_id]
        step.group_indices.update(source.group_indices)
        if source.window_rows is not None:
            step.window_rows = source.window_rows
        # The live step now owns these results; retain only the recording.
        self.warmup = None


class AttentionGraphs(PiecewiseAFRunner):
    """V4.1 host branches, staged BatchStep construction and result publication."""

    def __init__(self, outputs=None, inputs: SharedGraphInputs | None = None):
        super().__init__(
            outputs if outputs is not None else StableOutputs(token_dim=1), inputs
        )

    @staticmethod
    def eligible(metadata, context):
        return (
            bool(metadata.step.requests)
            and not getattr(context, "in_hipgraph", False)
            and not getattr(context.context, "is_dummy_run", False)
            and not getattr(metadata, "dummy", False)
        )

    @staticmethod
    def _inputs(
        attention: Attention,
        hidden: torch.Tensor,
        step: BatchStep,
        queries: QueryRows | None,
    ) -> dict:
        spec, ratio = attention.spec, attention.spec.ratio
        inputs = {
            "hidden": hidden,
            **{name: getattr(step, name) for name in _STEP_TENSORS},
        }
        groups = {
            "visible": {ratio: step.visible[ratio]} if ratio else {},
            "selected": (
                {spec.topk_owner: step.selected[spec.topk_owner]}
                if spec.topk_owner is not None and spec.topk_owner != spec.layer_id
                else {}
            ),
            "candidates": (
                {spec.candidate_source: step.candidates[spec.candidate_source]}
                if spec.candidate_source is not None
                else {}
            ),
            "planned": {
                k: step.planned[k]
                for k in sorted(step.planned)
                if k.endswith(f"_{ratio}")
            },
            "planned_host": {
                k: step.planned_host[k]
                for k in sorted(step.planned_host)
                if k.endswith(f"_{ratio}")
            },
        }
        for group, values in groups.items():
            inputs.update({(group, name): value for name, value in values.items()})
        inputs.update(
            {
                ("indptrs", ratio, i): value
                for i, value in enumerate(step.indptrs[ratio])
            }
        )
        plan = step.plans.get(ratio)
        if plan is not None:
            inputs.update(
                {("plan", ratio, name): getattr(plan, name) for name in _PLAN_TENSORS}
            )
        if queries is not None:
            inputs.update(
                {
                    ("queries", name): getattr(queries, name)
                    for name in ("rows", "positions", "ptr_rows")
                }
            )
            inputs["query_batch_ids"] = queries.index.batch_ids
            inputs.update(
                {
                    ("query_indptrs", ratio, i): value
                    for i, value in enumerate(queries.index.indptrs[ratio])
                }
            )
        return inputs

    @contextmanager
    def bind(
        self,
        attention: Attention,
        hidden: torch.Tensor,
        metadata,
        forward_context,
        *,
        capture: bool,
        zero_copy_hidden: bool = False,
    ) -> Iterator[dict]:
        """Bind the model method's arguments and publish results after its call."""
        cache, step, queries = metadata.cache, metadata.step, metadata.queries
        arguments = {
            "normed": hidden,
            "cache": cache,
            "step": step,
            "queries": queries,
            "piecewise": forward_context.cudagraph_runtime_mode
            == CUDAGraphMode.PIECEWISE,
            "capture": False,
            "runner": self,
            "outputs": self.outputs,
            "forward_context": forward_context,
        }
        if not (
            arguments["piecewise"]
            and capture
            and self.eligible(metadata, forward_context)
        ):
            yield arguments
            return

        sources = self._inputs(attention, hidden, step, queries)
        key = (
            step.decode,
            step.width,
            step.max_q_len,
            bool(step.request_cached_until.any()),
            cache.pool.data_ptr(),
        )
        with self.bind_capture(
            key,
            sources,
            lambda inputs: self._capture_state(attention, step, queries, inputs),
            memo=step.af_inputs,
            copy_each_call=("hidden",),
            zero_copy=("hidden",) if zero_copy_hidden else (),
        ) as binding:
            if binding is None:
                yield arguments
                return
            arguments.update(normed=binding.inputs["hidden"], **binding.arguments)
            yield arguments
            binding.state.publish(
                step, attention.spec.layer_id, first_call=binding.first_call
            )

    @staticmethod
    def _capture_state(attention, step, queries, inputs):
        ratio = attention.spec.ratio
        groups = {
            name: {}
            for name in (
                "visible",
                "selected",
                "candidates",
                "planned",
                "planned_host",
            )
        }
        for name, value in inputs.items():
            if isinstance(name, tuple) and name[0] in groups:
                groups[name[0]][name[1]] = value
        staged_step = replace(
            step,
            **{name: inputs[name] for name in _STEP_TENSORS},
            **groups,
            indptrs={
                ratio: tuple(
                    inputs["indptrs", ratio, i] for i in range(len(step.indptrs[ratio]))
                )
            },
            plans=(
                {}
                if step.plans.get(ratio) is None
                else {
                    ratio: replace(
                        step.plans[ratio],
                        **{name: inputs["plan", ratio, name] for name in _PLAN_TENSORS},
                    )
                }
            ),
            tiles={},
            candidate_blocks={},
            group_indices={},
            window_rows=None,
            af_inputs={},
        )
        query = None
        if queries is not None:
            query = replace(
                queries,
                **{
                    name: inputs["queries", name]
                    for name in ("rows", "positions", "ptr_rows")
                },
                index=replace(
                    staged_step,
                    batch_ids=inputs["query_batch_ids"],
                    indptrs={
                        ratio: tuple(
                            inputs["query_indptrs", ratio, i]
                            for i in range(len(queries.index.indptrs[ratio]))
                        )
                    },
                ),
            )
        return AttentionCapture(staged_step, query)
