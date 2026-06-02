from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import aiter
except (ImportError, ModuleNotFoundError):  # pragma: no cover - runtime fallback
    aiter = None

try:
    from rtp_llm.models_py.modules.factory.attention.common import (
        reshape_paged_kv_cache,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - runtime fallback
    reshape_paged_kv_cache = None

try:
    from rtp_llm.ops import AttentionConfigs, KvCacheDataType
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCacheDecodeOpNonAsm,
        FusedRopeKVCachePrefillOpNonAsm,
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - runtime fallback
    AttentionConfigs = None
    KvCacheDataType = None
    FusedRopeKVCacheDecodeOpNonAsm = None
    FusedRopeKVCachePrefillOpNonAsm = None

from atom.model_ops.base_attention import BaseAttention
from atom.plugin.prepare import is_plugin_mode, is_rtpllm
from atom.utils.forward_context import get_forward_context


def _align_kv_heads_for_cache(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    target_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_kv_heads = int(key.shape[1])
    if current_kv_heads == int(target_kv_heads):
        return key, value
    if current_kv_heads <= 0 or int(target_kv_heads) <= 0:
        raise ValueError(
            f"invalid kv head count: current={current_kv_heads}, target={target_kv_heads}"
        )
    if int(target_kv_heads) % current_kv_heads != 0:
        raise ValueError(
            f"cannot align kv heads from {current_kv_heads} to {target_kv_heads}"
        )
    dup_factor = int(target_kv_heads) // current_kv_heads
    key_aligned = key.repeat_interleave(dup_factor, dim=1)
    value_aligned = value.repeat_interleave(dup_factor, dim=1)
    return key_aligned, value_aligned


def _write_kv_cache_with_rtp_fused_kernel(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_cache: object,
    attn_inputs: object,
    tokens_per_block: int,
    qkv_buffer: torch.Tensor | None = None,
    fused_op: object | None = None,
    fused_params_cache: dict[int, object] | None = None,
) -> bool:
    if fused_op is None:
        return False

    q_flat = query.reshape(query.shape[0], -1)
    k_flat = key.reshape(key.shape[0], -1)
    v_flat = value.reshape(value.shape[0], -1)
    total_dim = int(q_flat.shape[1] + k_flat.shape[1] + v_flat.shape[1])
    # Caller (_get_fused_qkv_buffer) is responsible for providing a stable buffer;
    # under cuda-graph capture it errors if no prewarm. Here we only allocate as
    # an eager-mode safety net.
    if (
        qkv_buffer is None
        or qkv_buffer.device != query.device
        or qkv_buffer.dtype != query.dtype
        or int(qkv_buffer.shape[0]) < int(query.shape[0])
        or int(qkv_buffer.shape[1]) < total_dim
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "AttentionForRTPLLM fused-write requires a prewarmed qkv_buffer in "
                "cuda-graph capture mode."
            )
        qkv = torch.empty(
            (int(query.shape[0]), total_dim),
            dtype=query.dtype,
            device=query.device,
        )
    else:
        qkv = qkv_buffer[: int(query.shape[0]), :total_dim]
    q_end = int(q_flat.shape[1])
    k_end = q_end + int(k_flat.shape[1])
    qkv[:, :q_end].copy_(q_flat)
    qkv[:, q_end:k_end].copy_(k_flat)
    qkv[:, k_end:].copy_(v_flat)
    op = fused_op
    use_cached_params = bool(
        fused_params_cache is not None
        and (
            torch.cuda.is_current_stream_capturing()
            or bool(getattr(attn_inputs, "is_cuda_graph", False))
        )
    )
    params = None
    if use_cached_params:
        params = fused_params_cache.get(id(op))
    if params is None:
        params = op.prepare(attn_inputs)
        if use_cached_params:
            fused_params_cache[id(op)] = params
    else:
        update_kv_cache_offset = getattr(params, "update_kv_cache_offset", None)
        if callable(update_kv_cache_offset):
            update_kv_cache_offset(attn_inputs.kv_cache_kernel_block_id_device)
    _ = op.forward(qkv, layer_cache, params)
    return True


def _resolve_block_tables_for_layer(
    attn_inputs: object,
    layer_num: int,
    *,
    layer_group_map: dict[int, int] | None = None,
) -> torch.Tensor | None:
    # Mirror RTP select_block_map_for_layer semantics:
    # 1) compute gid from kv_cache_layer_to_group[layer]
    # 2) if by-group block map exists, select by gid
    # 3) otherwise fallback to current kv_cache_kernel_block_id_device
    current = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
    by_group = getattr(attn_inputs, "kv_cache_kernel_block_id_device_by_group", None)

    gid = (
        int(layer_group_map[layer_num])
        if (layer_group_map is not None and layer_num in layer_group_map)
        else 0
    )

    if isinstance(by_group, (list, tuple)) and len(by_group) > gid:
        t = by_group[gid]
        if t is not None and t.numel() > 0:
            return t
    return current


def _run_nonasm_paged_attention(
    *,
    query: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    kv_scale_base: torch.Tensor | None,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
    static_bufs: dict | None = None,
) -> torch.Tensor:
    """RTP plugin paged attention.

    When ``static_bufs`` is provided (cuda-graph capture path), all temporary
    tensors are sliced from prewarmed buffers so capture records stable
    addresses. When None, fall back to fresh allocations (eager path).
    """
    if aiter is None:
        raise ValueError(
            "AttentionForRTPLLM requires aiter for nonasm paged attention."
        )

    key_cache = paged_kv_cache.select(1, 0)
    value_cache = paged_kv_cache.select(1, 1)
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[2]
    num_seqs, num_heads, _ = query.shape
    block_size = value_cache.shape[2]
    max_seq_len = int(max_seq_len)
    scale = 1.0 / math.sqrt(head_size)

    partition_size = 256
    max_num_partitions = (max_seq_len + partition_size - 1) // partition_size

    if static_bufs is not None:
        # cuda-graph capture path: every buffer must be a stable-address slice.
        prewarmed_partitions = int(static_bufs["max_num_partitions"])
        if prewarmed_partitions < max_num_partitions:
            raise RuntimeError(
                "AttentionForRTPLLM prewarmed max_num_partitions "
                f"({prewarmed_partitions}) is smaller than required "
                f"({max_num_partitions}); recapture with larger max_seq_len."
            )
        # Use the prewarmed maximum so kernel launch arg is the same Python int
        # in capture and replay (kernel must read the same partition count).
        max_num_partitions = prewarmed_partitions
        # aiter's pa.py recomputes npar_loops = ceil(max_num_partitions / warp_size)
        # from `max_context_len` and bakes it into the JIT-compiled kernel
        # template. RTP's capture warmup feeds plugin_md.max_seq_len=0, which
        # would yield npar_loops=0 → __shared__ float shared_exp_sums[0] →
        # HIP compile error ("zero-length arrays not permitted"). Clamp to the
        # prewarm bucket so the compiled kernel matches replay.
        max_seq_len = prewarmed_partitions * partition_size
        output = static_bufs["output"][:num_seqs, :num_heads, :head_size]
        tmp_output = static_bufs["tmp_output"][
            :num_seqs, :num_heads, :max_num_partitions, :head_size
        ]
        exp_sums = static_bufs["exp_sums"][:num_seqs, :num_heads, :max_num_partitions]
        max_logits = static_bufs["max_logits"][
            :num_seqs, :num_heads, :max_num_partitions
        ]
        unit_scale = static_bufs["unit_scale"]
    else:
        # Defensive clamp: aiter requires max_context_len >= partition_size to
        # avoid npar_loops=0 → zero-length __shared__ array compile failure.
        if max_seq_len < partition_size:
            max_seq_len = partition_size
            max_num_partitions = 1
        output = torch.empty_like(query).view((num_seqs, num_heads, head_size))
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.ones_like(exp_sums)
        unit_scale = torch.ones(1, dtype=torch.float32, device=query.device)

    k_scale = None
    v_scale = None
    if (
        key_cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
        and value_cache.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
        and kv_scale_base is not None
    ):
        k_scale = kv_scale_base.select(1, 0)
        v_scale = kv_scale_base.select(1, 1)
    else:
        # Keep fallback semantics aligned with RTP non-ASM decode path.
        k_scale = unit_scale
        v_scale = unit_scale

    aiter.paged_attention_rocm(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        float(scale),
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        None,  # alibi_slopes
        "auto",  # kv_cache_dtype
        k_scale,
        v_scale,
        None,  # fp8_out_scale
        partition_size,
    )
    return output


class RTPFullAttention(BaseAttention):
    """RTP-style full attention adapter for rtpllm plugin mode."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        kv_cache_dtype: str = "bf16",
        layer_num: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            kv_cache_dtype=kv_cache_dtype,
            layer_num=layer_num,
            **kwargs,
        )
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_kv_heads = int(num_kv_heads)
        self.scale = float(scale)
        self.layer_num = int(layer_num)
        self._fused_qkv_buf: torch.Tensor | None = None
        self._paged_kv_cache: torch.Tensor | None = None
        self._paged_kv_cache_sig: tuple[int, int, int, int, int] | None = None
        self._fused_kv_op_cache: dict[
            tuple[torch.dtype, str, int, int, int, int, bool], object
        ] = {}
        # key: id(fused_op) -> params object from fused_op.prepare(attn_inputs)
        self._fused_kv_params_cache: dict[int, object] = {}
        self._backend_ready = aiter is not None and reshape_paged_kv_cache is not None
        # cuda-graph static buffers: allocated by prewarm_for_cuda_graph(),
        # reused across all capture/replay calls so addresses stay stable.
        self._cg_static_bufs: dict | None = None
        # Effective num_kv_heads after RTP-side duplicate-KV alignment (kv_head_num<tp_size).
        # Lazy-resolved at first eager forward; capture path requires it to be set.
        self._effective_num_kv_heads: int | None = None

    def _get_fused_qkv_buffer(
        self,
        *,
        num_tokens: int,
        total_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get a fused [num_tokens, total_dim] buffer for QKV concatenation.

        cuda-graph path (prewarmed buffer exists): always slice into the
        prewarmed max-sized buffer so addresses stay stable. Re-allocating
        inside a captured stream would yield unstable pointers on replay.
        """
        buf = self._fused_qkv_buf
        if (
            buf is not None
            and buf.device == device
            and buf.dtype == dtype
            and int(buf.shape[0]) >= int(num_tokens)
            and int(buf.shape[1]) >= int(total_dim)
        ):
            return buf[: int(num_tokens), : int(total_dim)]

        # Buffer missing or too small: in capture mode this is fatal.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "AttentionForRTPLLM requires prewarm_for_cuda_graph(...) to allocate "
                "_fused_qkv_buf with sufficient capacity before cuda-graph capture; "
                f"need=[{num_tokens},{total_dim}], have="
                f"{None if buf is None else tuple(buf.shape)}."
            )
        buf = torch.empty((int(num_tokens), int(total_dim)), dtype=dtype, device=device)
        self._fused_qkv_buf = buf
        return buf

    def _get_paged_kv_cache(
        self,
        *,
        raw: torch.Tensor,
        tokens_per_block: int,
    ) -> torch.Tensor:
        signature = (
            int(raw.data_ptr()),
            int(raw.numel()),
            int(self.num_kv_heads),
            int(self.head_dim),
            int(tokens_per_block),
        )
        cached = self._paged_kv_cache
        if cached is None or self._paged_kv_cache_sig != signature:
            cached = reshape_paged_kv_cache(
                raw,
                num_kv_heads=self.num_kv_heads,
                tokens_per_block=tokens_per_block,
                head_dim=self.head_dim,
            )
            self._paged_kv_cache = cached
            self._paged_kv_cache_sig = signature
        return cached

    def _get_fused_kv_op(
        self,
        *,
        query_dtype: torch.dtype,
        kv_cache_dtype: torch.dtype | None,
        tokens_per_block: int,
        num_kv_heads: int,
        is_prefill: bool,
    ) -> object | None:
        if (
            AttentionConfigs is None
            or KvCacheDataType is None
            or FusedRopeKVCacheDecodeOpNonAsm is None
            or FusedRopeKVCachePrefillOpNonAsm is None
        ):
            return None
        kv_dtype_key = (
            "fp8"
            if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            else "base"
        )
        cache_key = (
            query_dtype,
            kv_dtype_key,
            int(tokens_per_block),
            int(num_kv_heads),
            bool(is_prefill),
        )
        op = self._fused_kv_op_cache.get(cache_key)
        if op is not None:
            return op
        attn_configs = AttentionConfigs()
        attn_configs.head_num = int(self.num_heads)
        attn_configs.kv_head_num = int(num_kv_heads)
        attn_configs.size_per_head = int(self.head_dim)
        attn_configs.tokens_per_block = int(tokens_per_block)
        attn_configs.kernel_tokens_per_block = int(tokens_per_block)
        attn_configs.is_causal = True
        attn_configs.use_mla = False
        attn_configs.q_scaling = 1.0
        attn_configs.dtype = query_dtype
        if kv_dtype_key == "fp8":
            attn_configs.kv_cache_dtype = KvCacheDataType.FP8
        else:
            attn_configs.kv_cache_dtype = KvCacheDataType.BASE
        # Keep RoPE mocked on ATOM side for this experiment;
        # we only reuse RTP fused address/write semantics here.
        attn_configs.need_rope_kv_cache = False
        if is_prefill:
            op = FusedRopeKVCachePrefillOpNonAsm(attn_configs)
        else:
            op = FusedRopeKVCacheDecodeOpNonAsm(attn_configs)
        self._fused_kv_op_cache[cache_key] = op
        return op

    # ----------------------------- cuda-graph hooks -----------------------------
    # See rtp+atom_graph.md §4.1 for design rationale.

    def prepare_cuda_graph(self, attn_inputs) -> None:
        """RTP CudaGraphRunner.cc:122 calls this on attn_pyobj before each replay.

        Keep ATOM fused-KV params lifecycle aligned with RTP native decode path:
        params object is persistent, and replay updates block-offset mapping
        in-place via CKAttn.update_kv_cache_offset(...). The prewarmed
        _cg_static_bufs are deliberately not refreshed here: replay slices and
        writes them in-place, so their underlying captured addresses remain
        stable across requests.
        """
        for params in self._fused_kv_params_cache.values():
            update_kv_cache_offset = getattr(params, "update_kv_cache_offset", None)
            if callable(update_kv_cache_offset):
                update_kv_cache_offset(attn_inputs.kv_cache_kernel_block_id_device)
        return

    def prewarm_for_cuda_graph(
        self,
        *,
        max_num_tokens: int,
        max_seq_len: int,
        query_dtype: torch.dtype,
        device: torch.device,
        effective_num_kv_heads: int | None = None,
    ) -> None:
        """Pre-allocate every tensor that _forward_impl_plugin_mode would otherwise
        create with torch.empty/torch.ones inside a captured graph.

        Must be called once per layer BEFORE PyWrappedModel.initCapture() runs the
        capture warmup. The buffers are sized at the maximum bucket; per-step
        replay slices into [:num_seqs, ...] views which keep the underlying
        data_ptr() stable.
        """
        eff_kv_heads = int(
            effective_num_kv_heads
            if effective_num_kv_heads is not None
            else self.num_kv_heads
        )
        self._effective_num_kv_heads = eff_kv_heads

        fused_dim = int(
            self.num_heads * self.head_dim + 2 * eff_kv_heads * self.head_dim
        )
        self._fused_qkv_buf = torch.empty(
            (int(max_num_tokens), fused_dim), dtype=query_dtype, device=device
        )

        partition_size = 256
        max_num_partitions = (int(max_seq_len) + partition_size - 1) // partition_size
        self._cg_static_bufs = {
            "max_num_partitions": int(max_num_partitions),
            "output": torch.empty(
                (int(max_num_tokens), int(self.num_heads), int(self.head_dim)),
                dtype=query_dtype,
                device=device,
            ),
            "tmp_output": torch.empty(
                (
                    int(max_num_tokens),
                    int(self.num_heads),
                    int(max_num_partitions),
                    int(self.head_dim),
                ),
                dtype=query_dtype,
                device=device,
            ),
            "exp_sums": torch.empty(
                (int(max_num_tokens), int(self.num_heads), int(max_num_partitions)),
                dtype=torch.float32,
                device=device,
            ),
            "max_logits": torch.empty(
                (int(max_num_tokens), int(self.num_heads), int(max_num_partitions)),
                dtype=torch.float32,
                device=device,
            ),
            "unit_scale": torch.ones(1, dtype=torch.float32, device=device),
            # Prewarm aligned k/v buffers so capture can write in-place instead
            # of recording fresh repeat_interleave allocations whose addresses
            # may be reused by PyTorch's caching allocator after capture.
            "k_aligned": torch.empty(
                (int(max_num_tokens), int(eff_kv_heads), int(self.head_dim)),
                dtype=query_dtype,
                device=device,
            ),
            "v_aligned": torch.empty(
                (int(max_num_tokens), int(eff_kv_heads), int(self.head_dim)),
                dtype=query_dtype,
                device=device,
            ),
            # Stabilize q as well: ATOM's QKV linear can hand capture a transient
            # caching-pool address that later gets reused before graph replay.
            "q_aligned": torch.empty(
                (int(max_num_tokens), int(self.num_heads), int(self.head_dim)),
                dtype=query_dtype,
                device=device,
            ),
        }

    def _forward_impl_plugin_mode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del positions, kwargs
        if not self._backend_ready:
            raise ValueError(
                "AttentionForRTPLLM requires aiter and reshape_paged_kv_cache in plugin mode."
            )
        fwd_ctx = get_forward_context()
        if fwd_ctx is None:
            raise ValueError(
                "AttentionForRTPLLM requires forward context in plugin mode."
            )

        attn_metadata = fwd_ctx.attn_metadata
        if attn_metadata is None:
            raise ValueError(
                "AttentionForRTPLLM requires attn_metadata in forward context."
            )

        # Short-circuit RTP's `initCapture forward for output datatype` probe.
        # When RTP feeds dummy seq_lens=[0,...] / block_tables=[0,...] purely to
        # discover the output dtype, running real attention against zero
        # metadata is meaningless and aiter.paged_attention_rocm page-faults
        # (it pre-fetches block_tables / KV slots before bounds-checking
        # context_len). Return correctly-shaped zero output with q.dtype so the
        # probe's only purpose — discovering output dtype/shape — still works.
        plugin_md_probe = getattr(attn_metadata, "plugin_metadata", None)
        if plugin_md_probe is not None and bool(
            getattr(plugin_md_probe, "is_dummy_warmup", False)
        ):
            num_tokens = int(query.shape[0])
            return torch.zeros(
                (num_tokens, self.num_heads * self.head_dim),
                dtype=query.dtype,
                device=query.device,
            )

        attn_inputs = attn_metadata.rtp_attn_inputs
        if attn_inputs is None:
            raise ValueError(
                "AttentionForRTPLLM requires rtp_attn_inputs in attn_metadata."
            )

        kv_cache_data = fwd_ctx.kv_cache_data
        if kv_cache_data is None:
            raise ValueError(
                "AttentionForRTPLLM requires kv_cache_data in forward context."
            )
        layer_cache_entry = kv_cache_data.get(f"layer_{self.layer_num}")
        if layer_cache_entry is None or layer_cache_entry.k_cache is None:
            raise ValueError(
                f"AttentionForRTPLLM requires layer cache for layer_{self.layer_num}."
            )
        layer_cache = layer_cache_entry.k_cache

        q = query.view(-1, self.num_heads, self.head_dim)
        k = key.view(-1, self.num_kv_heads, self.head_dim)
        v = value.view(-1, self.num_kv_heads, self.head_dim)
        # In capture mode, copy q into a per-layer prewarm buffer so the captured
        # kernel reads from a stable address instead of a transient allocator slot.
        if (
            torch.cuda.is_current_stream_capturing()
            and self._cg_static_bufs is not None
            and "q_aligned" in self._cg_static_bufs
        ):
            n_q = int(q.shape[0])
            q_buf = self._cg_static_bufs["q_aligned"][:n_q]
            q_buf.copy_(q)
            q = q_buf
        is_prefill = bool(getattr(attn_inputs, "is_prefill", False))
        raw = getattr(layer_cache, "kv_cache_base", None)
        if raw is None:
            raise ValueError(
                f"AttentionForRTPLLM layer_{self.layer_num} missing kv_cache_base."
            )
        kernel_seq_size_per_block = int(
            getattr(attn_metadata, "rtp_kernel_seq_size_per_block", 0) or 16
        )
        paged_kv = self._get_paged_kv_cache(
            raw=raw,
            tokens_per_block=kernel_seq_size_per_block,
        )
        if paged_kv.dim() != 5 or int(paged_kv.shape[1]) != 2:
            raise ValueError(
                "AttentionForRTPLLM expects paged kv cache "
                f"[num_blocks,2,H,T,D], got {tuple(paged_kv.shape)}"
            )

        key_cache = paged_kv.select(1, 0)
        value_cache = paged_kv.select(1, 1)
        target_kv_heads = int(key_cache.shape[1])
        # Latch effective num_kv_heads on first forward — RTP may duplicate KV
        # heads when kv_head_num<tp_size, and capture path must use the same
        # value as eager (see rtp+atom_graph.md §4.4 / Round-1 cache-key bug).
        if self._effective_num_kv_heads is None:
            self._effective_num_kv_heads = int(target_kv_heads)
        elif int(self._effective_num_kv_heads) != int(target_kv_heads):
            raise RuntimeError(
                f"AttentionForRTPLLM layer_{self.layer_num} effective_num_kv_heads "
                f"changed across forwards: cached={self._effective_num_kv_heads}, "
                f"current={target_kv_heads}; cuda-graph cannot capture this layer."
            )
        if target_kv_heads != self.num_kv_heads:
            # Capture writes into prewarmed k/v buffers so kernels reference
            # stable addresses instead of fresh repeat_interleave allocations.
            in_capture_align = (
                torch.cuda.is_current_stream_capturing()
                and self._cg_static_bufs is not None
                and "k_aligned" in self._cg_static_bufs
            )
            if in_capture_align:
                n = int(k.shape[0])
                k_buf = self._cg_static_bufs["k_aligned"][:n]
                v_buf = self._cg_static_bufs["v_aligned"][:n]
                # Fast path when input has 1 head: broadcast-copy via expand
                # (no extra alloc; expand is a stride-only view).
                if int(self.num_kv_heads) == 1:
                    k_buf.copy_(k.expand(n, int(target_kv_heads), int(self.head_dim)))
                    v_buf.copy_(v.expand(n, int(target_kv_heads), int(self.head_dim)))
                else:
                    # General per-head copy mirroring repeat_interleave semantics.
                    dup_factor = int(target_kv_heads) // int(self.num_kv_heads)
                    for src_h in range(int(self.num_kv_heads)):
                        for rep in range(dup_factor):
                            dst_h = src_h * dup_factor + rep
                            k_buf[:, dst_h, :].copy_(k[:, src_h, :])
                            v_buf[:, dst_h, :].copy_(v[:, src_h, :])
                k, v = k_buf, v_buf
            else:
                # Eager path: keep original repeat_interleave (allocs new tensor;
                # safe in eager, unsafe in capture — guarded above).
                k, v = _align_kv_heads_for_cache(
                    key=k,
                    value=v,
                    target_kv_heads=target_kv_heads,
                )
        layer_group_map = getattr(attn_metadata, "rtp_layer_group_map", None)
        block_tables = _resolve_block_tables_for_layer(
            attn_inputs,
            int(self.layer_num),
            layer_group_map=layer_group_map,
        )
        if block_tables is None or block_tables.numel() == 0:
            raise ValueError(
                f"AttentionForRTPLLM requires block table for layer_{self.layer_num}."
            )
        plugin_md = attn_metadata.plugin_metadata
        seq_lens = plugin_md.seq_lens
        if seq_lens is None:
            raise ValueError(
                "AttentionForRTPLLM requires block tables and sequence lengths."
            )
        fused_qkv_dim = int(
            self.num_heads * self.head_dim + 2 * int(k.shape[1]) * self.head_dim
        )
        fused_qkv_buf = self._get_fused_qkv_buffer(
            num_tokens=int(q.shape[0]),
            total_dim=fused_qkv_dim,
            device=q.device,
            dtype=q.dtype,
        )
        used_fused_write = _write_kv_cache_with_rtp_fused_kernel(
            query=q,
            key=k,
            value=v,
            layer_cache=layer_cache,
            attn_inputs=attn_inputs,
            tokens_per_block=kernel_seq_size_per_block,
            qkv_buffer=fused_qkv_buf,
            fused_op=self._get_fused_kv_op(
                query_dtype=q.dtype,
                kv_cache_dtype=getattr(raw, "dtype", None),
                tokens_per_block=kernel_seq_size_per_block,
                num_kv_heads=int(k.shape[1]),
                is_prefill=is_prefill,
            ),
            fused_params_cache=self._fused_kv_params_cache,
        )
        if not used_fused_write:
            raise RuntimeError(
                "RTP fused KV write is required but unavailable; "
                "fallback slot_mapping path is removed."
            )
        cu_seqlens_q = getattr(plugin_md, "rtp_cu_seqlens_q", None)
        if is_prefill and cu_seqlens_q is not None and cu_seqlens_q.numel() > 1:
            q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)
            num_seqs = int(q_lens.numel())
            max_q_len = int(plugin_md.max_query_len)
            max_seq_len = int(plugin_md.max_seq_len)
            has_prefix = bool(getattr(plugin_md, "rtp_has_prefix", False))
            if has_prefix:
                key_cache_aiter = key_cache
                value_cache_aiter = value_cache
                x = 16 // key_cache_aiter.element_size()
                kv_sizes = key_cache_aiter.shape
                key_cache_aiter = key_cache_aiter.view(
                    kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x
                )
                value_cache_aiter = value_cache_aiter.view(
                    kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x
                )
                kv_indptr = torch.zeros(
                    num_seqs + 1, dtype=torch.int32, device=q.device
                )
                kv_page_indices = torch.zeros(1, dtype=torch.int32, device=q.device)
                q_descale = None
                k_descale = None
                v_descale = None
                if key_cache_aiter.dtype in (
                    torch.float8_e4m3fnuz,
                    torch.float8_e4m3fn,
                ):
                    q_descale = torch.ones(1, dtype=torch.float32, device=q.device)
                    k_descale = torch.ones(1, dtype=torch.float32, device=q.device)
                    v_descale = torch.ones(1, dtype=torch.float32, device=q.device)
                output = aiter.mha_batch_prefill_func(
                    q,
                    key_cache_aiter,
                    value_cache_aiter,
                    cu_seqlens_q,
                    kv_indptr,
                    kv_page_indices,
                    max_q_len,
                    max_seq_len,
                    causal=True,
                    block_table=block_tables[:num_seqs],
                    seqlen_k=seq_lens[:num_seqs],
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            else:
                output = aiter.flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_q,
                    max_q_len,
                    max_q_len,
                    dropout_p=0.0,
                    causal=True,
                )
            return output.reshape(int(q.shape[0]), self.num_heads * self.head_dim)

        num_seqs = int(q.shape[0])
        # In capture mode, hand the prewarmed static buffers down so kernel
        # tensors keep stable addresses across replays.
        static_bufs = (
            self._cg_static_bufs
            if (
                self._cg_static_bufs is not None
                and torch.cuda.is_current_stream_capturing()
            )
            else None
        )
        output = _run_nonasm_paged_attention(
            query=q,
            paged_kv_cache=paged_kv,
            kv_scale_base=getattr(layer_cache, "kv_scale_base", None),
            seq_lens=seq_lens[:num_seqs],
            block_tables=block_tables[:num_seqs],
            max_seq_len=int(plugin_md.max_seq_len),
            static_bufs=static_bufs,
        )
        output = output.view(num_seqs, -1)
        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        q_scale: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del q_scale
        if not is_plugin_mode() or not is_rtpllm():
            raise NotImplementedError(
                "RTPFullAttention is only supported in rtpllm plugin mode."
            )
        return self._forward_impl_plugin_mode(
            query=query,
            key=key,
            value=value,
            positions=positions,
            **kwargs,
        )


AttentionForRTPLLM = RTPFullAttention
