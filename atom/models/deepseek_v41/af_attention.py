# SPDX-License-Identifier: MIT
"""Stage V4.1 attention metadata and publish the state produced by graph replay."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from atom.config import CUDAGraphMode
from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep
from atom.utils.cuda_graph import CudagraphCaptureRunner, StableOutputs
from atom.utils.graph_holders import register_graph_holder

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


def _signature(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


class SharedAttentionInputs(StableOutputs):
    """Fixed metadata addresses shared by layers, refreshed once per live step.

    Graph holders release this storage with the attention graphs. Memoization
    belongs to BatchStep and is reset by begin_forward, including reused steps.
    Hidden states are deliberately never memoized: each layer has new values.
    """

    def __init__(self):
        super().__init__()
        register_graph_holder(self)

    def release_graphs(self):
        self._slots.clear()
        return 0

    def prepare(self, inputs: dict, step: BatchStep, *, zero_copy_hidden: bool) -> dict:
        staged = {}
        for name, source in inputs.items():
            if not isinstance(source, torch.Tensor):
                staged[name] = source
                continue
            hidden = name == "hidden"
            if hidden and zero_copy_hidden:
                staged[name] = source
                continue
            key = (name, _signature(source), source.device)
            memo = (key, source.data_ptr())
            if not hidden and memo in step.af_inputs:
                staged[name] = step.af_inputs[memo]
                continue
            buffer = self.slot(key, source).copy_(source)
            if not hidden:
                step.af_inputs[memo] = buffer
            staged[name] = buffer
        return staged


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


class AttentionGraphs(CudagraphCaptureRunner):
    # Bound request-shaped live captures per layer; additional shapes stay eager.
    max_entries = 32

    def __init__(
        self,
        outputs: StableOutputs | None = None,
        inputs: SharedAttentionInputs | None = None,
    ):
        super().__init__()
        self.outputs = outputs if outputs is not None else StableOutputs(token_dim=1)
        self.inputs = inputs if inputs is not None else SharedAttentionInputs()
        self._captures: dict[tuple, AttentionCapture] = {}

    def release_graphs(self):
        self._captures.clear()
        return super().release_graphs()

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
        signature = tuple(
            (name, _signature(value) if isinstance(value, torch.Tensor) else value)
            for name, value in sources.items()
        )
        key = (
            step.decode,
            step.width,
            step.max_q_len,
            bool(step.request_cached_until.any()),
            cache.pool.data_ptr(),
            signature,
            # Directly captured producer addresses MUST match, even for an
            # eager/compiled producer unexpectedly reached at the same shape.
            hidden.data_ptr() if zero_copy_hidden else None,
        )
        if not self.has_graph(key) and len(self._graphs) >= self.max_entries:
            yield arguments
            return
        inputs = self.inputs.prepare(sources, step, zero_copy_hidden=zero_copy_hidden)
        first_call = not self.has_graph(key)
        if first_call:
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
                        inputs["indptrs", ratio, i]
                        for i in range(len(step.indptrs[ratio]))
                    )
                },
                plans=(
                    {}
                    if step.plans.get(ratio) is None
                    else {
                        ratio: replace(
                            step.plans[ratio],
                            **{
                                name: inputs["plan", ratio, name]
                                for name in _PLAN_TENSORS
                            },
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
            state = AttentionCapture(staged_step, query)
        else:
            state = self._captures[key]

        arguments.update(
            normed=inputs["hidden"],
            capture=True,
            capture_key=key,
            prepare=state.bind,
        )
        yield arguments
        if first_call:
            self._captures[key] = state
        state.publish(step, attention.spec.layer_id, first_call=first_call)
