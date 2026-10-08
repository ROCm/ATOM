# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-independent MiniMax-M3 mono library.

Construct collectively on the caller's TP CPU group before graph capture. Call
prepare_step once before the first sparse layer, then forward_layer in layer
order on the same device stream. Every TP rank must use the same step shape
and layer order; TPContext.device must be the current CUDA device. Returned
buffers are reused after two layers; clone auxiliary outputs to retain them.
Eager execution and CUDA graphs are supported; torch.compile functionalization
is not supported. Only uniform decode and verify batches are accepted. The
caller owns dispatch and fallback decisions.

Keep this object and native cache storage alive until all captured graphs have
been destroyed. close drains device work and agrees across ranks before freeing
IPC storage; call it before TP group teardown. No collectives run from GC.
"""

import itertools
import os
import weakref
from dataclasses import fields

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from atom.models.minimax_m3.mono.execution import SparseExecution
from atom.models.minimax_m3.mono.kernels.post_attn import K4_ABI, build_post_attn_kernel
from atom.models.minimax_m3.mono.library_types import (
    CacheSpec,
    LayerSpec,
    StepMetadata,
    TPContext,
)
from atom.models.minimax_m3.mono.library_weights import (
    PreparedLayer,
    require,
    validate_cache,
)
from atom.mono.runtime.compile import compile_only
from atom.mono.runtime.consensus import bind_agreed
from atom.mono.runtime.widths import WidthBuilds

__all__ = ["AtomM3Mono", "CacheSpec", "LayerSpec", "StepMetadata", "TPContext"]
SUPPORTED_TOKENS = (1, 4, 8, 16)
_RUNTIMES = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


@triton.jit(
    do_not_specialize=[
        "main_stride",
        "index_stride",
        "main_width",
        "index_width",
        "rows",
    ]
)
def _token_rows(
    main_source,
    index_source,
    lengths,
    main_slots,
    index_slots,
    main_out,
    index_out,
    seq_out,
    slot_out,
    index_slot_out,
    batch_ids,
    main_stride,
    index_stride,
    main_width,
    index_width,
    rows,
    query_len: tl.constexpr,
    width: tl.constexpr,
):
    token = tl.program_id(0)
    request = token // query_len
    seq = tl.load(lengths + request, request < rows, other=0)
    slot = tl.load(main_slots + token)
    index_slot = tl.load(index_slots + token)
    live = (seq >= query_len) & (slot >= 0) & (index_slot >= 0)
    causal_seq = tl.where(live, seq - query_len + token % query_len + 1, 0)
    tl.store(seq_out + token, causal_seq)
    tl.store(batch_ids + token, tl.where(live, request, -1))
    tl.store(slot_out + token, tl.where(live, slot, -1))
    tl.store(index_slot_out + token, tl.where(live, index_slot, -1))
    col = tl.arange(0, width)
    main = tl.load(
        main_source + request * main_stride + col,
        (request < rows) & (col < main_width) & live,
        other=0,
    )
    index = tl.load(
        index_source + request * index_stride + col,
        (request < rows) & (col < index_width) & live,
        other=0,
    )
    tl.store(main_out + token * width + col, main)
    tl.store(index_out + token * width + col, index)


@torch.library.custom_op("atom::m3_mono_prepare", mutates_args=("resources",))
def _prepare(
    main_table: torch.Tensor,
    index_table: torch.Tensor,
    lengths: torch.Tensor,
    main_slots: torch.Tensor,
    index_slots: torch.Tensor,
    resources: list[torch.Tensor],
    handle: int,
    tokens: int,
    query_len: int,
) -> None:
    _RUNTIMES[handle]._prepare_step(
        StepMetadata(
            main_table, index_table, lengths, main_slots, index_slots, tokens, query_len
        )
    )


@_prepare.register_fake
def _prepare_fake(
    main_table,
    index_table,
    lengths,
    main_slots,
    index_slots,
    resources,
    handle,
    tokens,
    query_len,
):
    return None


@torch.library.custom_op(
    "atom::m3_mono_layer",
    mutates_args=("output", "residual_out", "resources", "caches"),
)
def _layer(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    residual_out: torch.Tensor,
    weights: list[torch.Tensor],
    resources: list[torch.Tensor],
    caches: list[torch.Tensor],
    handle: int,
    index: int,
    query_len: int,
) -> None:
    _RUNTIMES[handle]._forward_layer(index, hidden, residual, positions, query_len)


@_layer.register_fake
def _layer_fake(
    hidden,
    residual,
    positions,
    output,
    residual_out,
    weights,
    resources,
    caches,
    handle,
    index,
    query_len,
):
    return None


class AtomM3Mono:
    def __init__(
        self,
        layer_specs: list[LayerSpec],
        cache_specs: list[CacheSpec],
        tp_context: TPContext,
    ):
        self.tp = tp_context
        self.closed = False
        self._step = None
        self._handle = next(_HANDLES)
        bind_agreed(lambda: self._bind(layer_specs, cache_specs), self.tp.cpu_group)
        self.execution = SparseExecution(
            self.weights,
            [c.index for c in self.caches],
            self.tp.cpu_group,
            self.tp.rank,
            self.tp.size,
            self.tp.device,
        )
        try:
            bind_agreed(self._allocate_metadata, self.tp.cpu_group)
            self.builds = WidthBuilds(
                self._build,
                lambda k: [(k, K4_ABI)],
                self.tp.cpu_group,
                "MiniMax-M3 tensor library",
            )
            for n in SUPPORTED_TOKENS:
                require(self.builds.prepare(n), f"compilation failed for {n} rows")
            bind_agreed(self._warm_metadata, self.tp.cpu_group)
        except Exception:
            self.execution.close()
            raise
        _RUNTIMES[self._handle] = self

    def _bind(self, specs, caches):
        require(
            os.environ.get("COMPILE_ONLY", "0") != "1",
            "COMPILE_ONLY disables execution",
        )
        require(self.tp.size == 4 and 0 <= self.tp.rank < 4, "requires TP4")
        require(
            self.tp.cpu_group is not None
            and dist.get_world_size(self.tp.cpu_group) == self.tp.size
            and dist.get_rank(self.tp.cpu_group) == self.tp.rank,
            "TP CPU group does not match context",
        )
        require(
            self.tp.device.type == "cuda"
            and self.tp.device.index == torch.cuda.current_device(),
            "TPContext.device must be the caller's current CUDA device",
        )
        props = torch.cuda.get_device_properties(self.tp.device)
        require(
            props.gcnArchName.split(":")[0] == "gfx950"
            and props.multi_processor_count == 256,
            "requires gfx950 with 256 CUs",
        )
        require(
            bool(specs) and len(specs) == len(caches), "layers and caches must pair"
        )
        ids = [s.layer_id for s in specs]
        require(
            all(type(i) is int and 0 <= i < 2**31 - 1 for i in ids),
            "layer IDs must produce nonzero Int32 mailbox tags",
        )
        require(
            ids == sorted(set(ids)) and ids == [c.layer_id for c in caches],
            "layer IDs must be unique, ordered and match caches",
        )
        constants = lambda s: (
            s.eps,
            s.route_scale,
            s.swiglu_limit,
            s.shared_weight,
            s.sm_scale,
            s.init_blocks,
            s.local_blocks,
        )
        require(len({constants(s) for s in specs}) == 1, "layer constants must agree")
        self.constants = constants(specs[0])
        require(
            specs[0].sm_scale == 128**-0.5
            and specs[0].init_blocks == 0
            and specs[0].local_blocks == 1,
            "unsupported sparse attention constants",
        )
        require(
            len({c.max_context for c in caches}) == 1, "cache capacities must agree"
        )
        for cache in caches:
            validate_cache(cache, self.tp.device)
        self.caches = list(caches)
        self.width = triton.next_power_of_2((caches[0].max_context + 127) // 128)
        self.weights = [PreparedLayer.from_spec(s, self.tp.device) for s in specs]
        self._indices = {s.layer_id: i for i, s in enumerate(specs)}
        self._weights = [
            [
                getattr(w, f.name)
                for f in fields(w)
                if isinstance(getattr(w, f.name), torch.Tensor)
            ]
            for w in self.weights
        ]

    def _build(self, tokens):
        eps, route, limit, shared, scale, init, local = self.constants
        return build_post_attn_kernel(
            self.tp.size,
            scale,
            eps,
            route,
            shared,
            limit,
            init,
            local,
            tokens,
            fuse_k1=True,
            cache_mode="vllm",
        )

    def _allocate_metadata(self):
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=self.tp.device)

        self.main_table = zeros((16, self.width), torch.int32)
        self.index_table = zeros((16, self.width), torch.int32)
        self.seq_lens = zeros((16,), torch.int32)
        self.main_slots = zeros((16,), torch.int64)
        self.index_slots = zeros((16,), torch.int64)
        self.batch_ids = zeros((16,), torch.int32)
        self.cache_args = torch.tensor(
            [self.index_slots.data_ptr(), self.index_table.data_ptr(), self.width],
            dtype=torch.int64,
            device=self.tp.device,
        )
        self.resources = [
            *self.execution.resources,
            self.main_table,
            self.index_table,
            self.seq_lens,
            self.main_slots,
            self.index_slots,
            self.batch_ids,
            self.cache_args,
        ]

    def _expand(self, md):
        _token_rows[(md.token_count,)](
            md.main_table,
            md.index_table,
            md.seq_lens,
            md.main_slots,
            md.index_slots,
            self.main_table,
            self.index_table,
            self.seq_lens,
            self.main_slots,
            self.index_slots,
            self.batch_ids,
            md.main_table.stride(0),
            md.index_table.stride(0),
            md.main_table.shape[1],
            md.index_table.shape[1],
            md.seq_lens.numel(),
            md.query_len,
            self.width,
        )

    def _warm_metadata(self):
        # Use independent dummy inputs; never touch native cache storage.
        lengths = torch.ones(4, dtype=torch.int32, device=self.tp.device)
        slots = torch.full((16,), -1, dtype=torch.int64, device=self.tp.device)
        table = torch.zeros((4, self.width), dtype=torch.int32, device=self.tp.device)
        for rows in (1, 2, 3, 4):
            for query_len in (1, 4):
                self._expand(
                    StepMetadata(
                        table, table, lengths[:rows], slots, slots, 16, query_len
                    )
                )
        with compile_only():
            self.execution.mailboxes.begin_step()
        torch.cuda.synchronize(self.tp.device)

    def prepare_step(self, metadata: StepMetadata):
        require(
            not torch.compiler.is_compiling(),
            "torch.compile is unsupported; use eager or CUDA graphs",
        )
        require(not self.closed, "runtime is closed")
        require(
            metadata.token_count in SUPPORTED_TOKENS
            and metadata.query_len in (1, 4)
            and metadata.token_count % metadata.query_len == 0,
            "unsupported step shape",
        )
        require(
            metadata.seq_lens.ndim == 1 and metadata.seq_lens.numel() <= 4,
            "requires at most four request rows",
        )
        for table in (metadata.main_table, metadata.index_table):
            require(
                table.ndim == 2
                and table.dtype == torch.int32
                and table.device == self.tp.device
                and table.stride(1) == 1
                and table.shape[0] >= metadata.seq_lens.numel()
                and table.shape[1] <= self.width,
                "invalid block table",
            )
        for tensor, dtype in (
            (metadata.seq_lens, torch.int32),
            (metadata.main_slots, torch.int64),
            (metadata.index_slots, torch.int64),
        ):
            require(
                tensor.ndim == 1
                and tensor.dtype == dtype
                and tensor.device == self.tp.device
                and tensor.is_contiguous(),
                "invalid step tensor",
            )
        require(
            min(metadata.main_slots.numel(), metadata.index_slots.numel())
            >= metadata.token_count,
            "slot mappings must cover padded tokens",
        )
        self._step = (metadata.token_count, metadata.query_len)
        _prepare(
            metadata.main_table,
            metadata.index_table,
            metadata.seq_lens,
            metadata.main_slots,
            metadata.index_slots,
            self.resources,
            self._handle,
            *self._step,
        )

    def _prepare_step(self, metadata):
        self._expand(metadata)
        self.execution.mailboxes.begin_step()

    def forward_layer(self, layer_id, hidden, residual, positions):
        require(
            not torch.compiler.is_compiling(),
            "torch.compile is unsupported; use eager or CUDA graphs",
        )
        require(
            not self.closed and self._step is not None,
            "prepare_step must precede layers",
        )
        n, query_len = self._step
        require(layer_id in self._indices, "unknown sparse layer")
        require(
            hidden.shape == residual.shape == (n, 6144)
            and hidden.dtype == residual.dtype == torch.bfloat16
            and hidden.device == residual.device == self.tp.device
            and hidden.is_contiguous()
            and residual.is_contiguous(),
            "invalid activations",
        )
        require(
            positions.ndim == 1
            and positions.numel() >= n
            and positions.dtype == torch.int64
            and positions.device == self.tp.device
            and positions.is_contiguous(),
            "invalid positions",
        )
        i = self._indices[layer_id]
        output, res_out = (
            self.execution.ars[(i + 1) % 2][:n],
            self.execution.h_mids[(i + 1) % 2][:n],
        )
        cache = self.caches[i]
        _layer(
            hidden,
            residual,
            positions,
            output,
            res_out,
            [*self._weights[i], cache.k_scale, cache.v_scale],
            self.resources,
            [cache.main, cache.index],
            self._handle,
            i,
            query_len,
        )
        return output, res_out

    def _forward_layer(self, i, hidden, residual, positions, query_len):
        cache = self.caches[i]
        return self.execution.forward_layer(
            i,
            hidden,
            residual,
            positions,
            (cache.k, cache.v, cache.k_scale, cache.v_scale),
            (self.main_table, self.seq_lens),
            self.main_slots,
            self.batch_ids,
            query_len,
            self.builds[hidden.shape[0]],
            cache_args=self.cache_args.data_ptr(),
        )

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.tp.device)
        dist.barrier(group=self.tp.cpu_group)
        self.closed = True
        _RUNTIMES.pop(self._handle, None)
        self.execution.close()
