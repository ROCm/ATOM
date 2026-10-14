# SPDX-License-Identifier: MIT
"""Attention graphs between V4.1's dense pieces.

Like V4's piecewise_core, read stable producer outputs directly and write the
consumer's output slot. Dynamic request metadata is published once per step,
shared across layers. CSR construction belongs to the graph that consumes it.
"""

from dataclasses import replace

import torch

from atom.utils.cuda_graph import CudagraphCaptureRunner, StableOutputs
from atom.utils.graph_holders import register_graph_holder


def _signature(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def _tensor_tree(tree):
    """Flatten only declared inputs, keeping host branch constants in the key."""
    tensors = {}

    def encode(value, path):
        if isinstance(value, torch.Tensor):
            tensors[path] = value
            return ("tensor", path, _signature(value))
        if isinstance(value, dict):
            return (
                "dict",
                tuple((k, encode(v, f"{path}.{k}")) for k, v in sorted(value.items())),
            )
        if isinstance(value, tuple):
            return (
                "tuple",
                tuple(encode(v, f"{path}.{i}") for i, v in enumerate(value)),
            )
        if value is None or isinstance(value, (int, bool, str)):
            return ("constant", value)
        raise TypeError(f"Unstaged attention graph input: {type(value)}")

    signature = encode(tree, "")

    def bind(inputs):
        def decode(node):
            kind, value, *_ = node
            if kind == "tensor":
                return inputs[value]
            if kind == "dict":
                return {k: decode(v) for k, v in value}
            if kind == "tuple":
                return tuple(decode(v) for v in value)
            return value

        return decode(signature)

    return tensors, signature, bind


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

    def prepare(self, tensors, step, *, zero_copy_hidden):
        staged = {}
        for name, source in tensors.items():
            hidden = name == ".hidden"
            if hidden and zero_copy_hidden:
                staged[name] = source
                continue
            signature = _signature(source)
            key = (name, signature, source.device)
            memo = (key, source.data_ptr())
            if not hidden and memo in step.af_inputs:
                staged[name] = step.af_inputs[memo]
                continue
            buffer = self.slot(key, source).copy_(source)
            staged[name] = buffer
            if not hidden:
                step.af_inputs[memo] = buffer
        return staged


class AttentionGraphs(CudagraphCaptureRunner):
    max_entries = 32

    def __init__(self):
        super().__init__()
        self.outputs = StableOutputs()
        self.capture_stream = torch.cuda.Stream()

    def release_graphs(self):
        self.outputs = StableOutputs()
        return super().release_graphs()

    @staticmethod
    def eligible(attention, metadata, context):
        return (
            bool(metadata.step.requests)
            and not getattr(context, "in_hipgraph", False)
            and not getattr(context.context, "is_dummy_run", False)
            and not getattr(metadata, "dummy", False)
        )

    @staticmethod
    def _inputs(attention, hidden, step, queries):
        spec, ratio = attention.spec, attention.spec.ratio
        plan = step.plans.get(ratio)
        fields = (
            "positions",
            "cu_seqlens_q",
            "slots",
            "batch_ids",
            "block_tables",
            "swa_replay_start",
            "swa_cached_until",
        )
        metadata = {
            **{name: getattr(step, name) for name in fields},
            "visible": {ratio: step.visible[ratio]} if ratio else {},
            "indptrs": {ratio: step.indptrs[ratio]},
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
                k: v for k, v in step.planned.items() if k.endswith(f"_{ratio}")
            },
            "planned_host": {
                k: v for k, v in step.planned_host.items() if k.endswith(f"_{ratio}")
            },
        }
        return {
            "hidden": hidden,
            "step": metadata,
            "plan": (
                None
                if plan is None
                else {
                    name: getattr(plan, name)
                    for name in (
                        "compress_plan_gpu",
                        "write_plan_gpu",
                        "key_rope_positions_gpu",
                    )
                }
            ),
            "queries": (
                None
                if queries is None
                else {
                    "rows": queries.rows,
                    "positions": queries.positions,
                    "ptr_rows": queries.ptr_rows,
                    "index": {
                        "batch_ids": queries.index.batch_ids,
                        "indptrs": {ratio: queries.index.indptrs[ratio]},
                    },
                }
            ),
        }

    @staticmethod
    def _publish(step, produced):
        selected, candidates, groups, window = produced
        step.selected.update(selected)
        step.candidates.update(candidates)
        step.group_indices.update(groups)
        if window is not None:
            step.window_rows = window

    def forward(
        self,
        attention,
        hidden,
        cache,
        step,
        rope,
        queries=None,
        *,
        output_slot=None,
        zero_copy_hidden=False,
    ):
        def slot_for(output):
            return (
                self.outputs.slot(_signature(output), output)
                if output_slot is None
                else output_slot(output)
            )

        tree = self._inputs(attention, hidden, step, queries)
        tensors, signature, bind = _tensor_tree(tree)
        shared = getattr(cache, "_af_inputs", None)
        if shared is None:
            shared = cache._af_inputs = SharedAttentionInputs()
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
            output = attention(hidden, None, cache, step, rope, queries=queries)
            slot = slot_for(output)
            slot.copy_(output)
            return slot
        inputs = shared.prepare(tensors, step, zero_copy_hidden=zero_copy_hidden)
        if self.has_graph(key):
            # No per-layer input copies: the step publisher/dense producer has
            # already written every address the graph captured.
            output = self.replay(key, {})
            self._publish(step, self._graphs[key]["published"])
            return output

        produced = None

        def compute(**read_from):
            nonlocal produced
            values = bind(read_from)
            ratio = attention.spec.ratio
            frozen = replace(
                step,
                **values["step"],
                plans=(
                    {}
                    if values["plan"] is None
                    else {ratio: replace(step.plans[ratio], **values["plan"])}
                ),
                tiles={},
                candidate_blocks={},
                group_indices={},
                window_rows=None,
                af_inputs={},
            )
            query = values["queries"]
            if query is not None:
                query["index"] = replace(frozen, **query["index"])
                query = replace(queries, **query)
            output = attention(
                values["hidden"], None, cache, frozen, rope, queries=query
            )
            layer = attention.spec.layer_id
            produced = (
                {layer: frozen.selected[layer]} if layer in frozen.selected else {},
                {layer: frozen.candidates[layer]} if layer in frozen.candidates else {},
                frozen.group_indices,
                frozen.window_rows,
            )
            return output

        # Match V4's zero-copy policy: these are producer-owned fixed inputs.
        current = torch.cuda.current_stream()
        self.capture_stream.wait_stream(current)
        with torch.cuda.stream(self.capture_stream):
            warm = compute(**inputs)
            warm_produced = produced
            slot = slot_for(warm)
            self.capture(key, inputs, {}, compute, slot)
            self._graphs[key]["published"] = produced
            # Keep producer buffers alive for all graph replays. The graph runner
            # otherwise only retains COPIED inputs in its refresh dictionary.
            self._graphs[key]["read_from"] = inputs
            slot.copy_(warm)
        current.wait_stream(self.capture_stream)
        self._publish(step, warm_produced)
        return slot
