"""ROCm ATOM PD split SSM/conv storage with one live decode state."""
import os
import torch
from atom.config import KVCacheTensor


def enabled():
    # The request-owned cache topology replaces token-page SSM aliases.
    return False


def configure_request_capacity(model_config, max_batch):
    role = os.getenv("ROLE_TYPE", "")
    mtp = (getattr(model_config, "atom_qwen4_exp_mtp_target", False)
           or getattr(model_config, "is_mtp", False))
    if torch.version.hip and role in ("PREFILL", "DECODE") and mtp:
        # Resolve the model default before C++ creates its cache pools.
        # Explicit overrides remain authoritative; other devices/modes are unchanged.
        os.environ.setdefault("ATOM_RTP_SPLIT_SSM_STORAGE", "1")
        if role == "DECODE":
            os.environ.setdefault("ATOM_RTP_SSM_STATE_SLOTS", "1")
    if not enabled():
        return
    from rtp_llm.ops import KVCacheSpecType, CacheCapacityPolicyDesc
    descs = model_config.kv_cache_spec_descs
    if (not any(d.cache_type == KVCacheSpecType.LINEAR for layer in descs for d in layer)
            or any(d.cache_type == KVCacheSpecType.LINEAR and d.linear_state_component == 0
                   for layer in descs for d in layer)):
        model_config.kv_cache_spec_descs = split_descs(descs)
    if role != "DECODE" or os.getenv("ATOM_RTP_SSM_STATE_SLOTS") != "1":
        return
    if max_batch is None or int(max_batch) <= 0:
        raise ValueError("Single-state SSM requires the configured request capacity")
    requests = int(os.getenv("ATOM_RTP_SSM_MAX_REQUESTS", str(int(max_batch))))
    if requests <= 0:
        raise ValueError("SSM request capacity must be positive")
    descs = model_config.kv_cache_spec_descs
    for layer in descs:
        for state in layer:
            if state.tag == "linear" and state.linear_state_component == 1:
                capacity = CacheCapacityPolicyDesc()
                capacity.explicit_block_num = requests + 1
                capacity.charge_to_paged_budget = True
                state.capacity = capacity
    model_config.kv_cache_spec_descs = descs


def split_descs(descs):
    if not enabled():
        return descs
    import pickle
    from rtp_llm.ops import KVCacheSpecType, CacheCapacityPolicyDesc
    for layer in descs:
        for index, original in enumerate(list(layer)):
            if original.cache_type != KVCacheSpecType.LINEAR:
                continue
            # build_hybrid_kv_cache_spec_descs shares descriptors across layers.
            # Never mutate that template while emitting a second cache group.
            state = pickle.loads(pickle.dumps(original))
            layer[index] = state
            conv = pickle.loads(pickle.dumps(state))
            state.linear_state_component = 1
            conv.linear_state_component = 2
            conv.tag = "gdn_conv"
            if os.getenv("ROLE_TYPE") == "DECODE":
                requests = int(os.getenv("ATOM_RTP_SSM_MAX_REQUESTS", "16"))
                if requests <= 0:
                    raise ValueError("SSM request capacity must be positive")
                slots = int(os.getenv("ATOM_RTP_SSM_STATE_SLOTS", "0"))
                if slots not in (0, 1):
                    raise ValueError("Only legacy and single-state service modes implemented so far")
                if slots == 1:
                    capacity = CacheCapacityPolicyDesc()
                    capacity.explicit_block_num = requests + 1
                    capacity.charge_to_paged_budget = True
                    state.capacity = capacity
            layer.append(conv)
    # Fixed-capacity A/B mode: preserve the baseline non-SSM pools rather than
    # silently reinvesting the released SSM memory in every other cache group.
    reference_blocks = int(os.getenv("ATOM_RTP_SPLIT_REFERENCE_BLOCKS", "0"))
    if reference_blocks:
        if reference_blocks < 2:
            raise ValueError("Reference pool capacity must be at least two blocks")
        paged_blocks = int(os.getenv("ATOM_RTP_SPLIT_PAGED_BLOCKS", str(reference_blocks)))
        if paged_blocks < 2:
            raise ValueError("Paged pool capacity must be at least two blocks")
        for layer in descs:
            for index, original in enumerate(layer):
                state = pickle.loads(pickle.dumps(original))
                layer[index] = state
                if (state.tag == "linear" and os.getenv("ROLE_TYPE") == "DECODE"
                        and os.getenv("ATOM_RTP_SSM_STATE_SLOTS") == "1"):
                    continue
                capacity = CacheCapacityPolicyDesc()
                capacity.explicit_block_num = (reference_blocks if state.cache_type == KVCacheSpecType.LINEAR
                                               else paged_blocks)
                capacity.charge_to_paged_budget = True
                state.capacity = capacity
    return descs


def build_caches(runtime, layer_maps, parent):
    caches = parent(runtime, ({}, layer_maps[1], layer_maps[2]))
    for number, layer in layer_maps[0].items():
        raw = runtime.kv_cache.get_layer_cache(number, "linear").kv_cache_base
        conv_raw = runtime.kv_cache.get_layer_cache(number, "gdn_conv").kv_cache_base
        heads = int(layer.dt_bias.numel())
        kd, vd = int(layer.head_k_dim), int(layer.head_v_dim)
        dim, length = int(layer.conv1d.weight.size(0)), int(layer.conv1d.weight.size(2))-1
        if raw.dtype != torch.bfloat16 or conv_raw.dtype != torch.bfloat16:
            raise ValueError("Split state pools require BF16 byte storage")
        state_base = raw.reshape(raw.shape[0], -1).view(torch.float32)
        conv_base = conv_raw.reshape(conv_raw.shape[0], -1)
        if state_base.shape[1] < heads*kd*vd or conv_base.shape[1] < dim*length:
            raise ValueError("Split state pool has insufficient row bytes")
        state = torch.as_strided(state_base, (raw.shape[0],heads,kd,vd),
                                (state_base.stride(0),kd*vd,vd,1))
        conv = torch.as_strided(conv_base, (conv_raw.shape[0],dim,length),
                               (conv_base.stride(0),1,dim))
        caches[f"layer_{number}"] = KVCacheTensor(layer_num=number,k_cache=conv,
            v_cache=state,k_scale=None,v_scale=None,per_request_state=True)
    return caches


def warm_buffers(runtime):
    if not enabled():
        return
    if not hasattr(runtime, "_split_conv_graph_buffers"):
        runtime._split_conv_graph_buffers = {}
    for buffers in (getattr(runtime,"_cg_meta_bufs",None),getattr(runtime,"_cg_mtp_meta_bufs",None)):
        if buffers is not None and id(buffers) not in runtime._split_conv_graph_buffers:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Split conv buffers were not prewarmed")
            runtime._split_conv_graph_buffers[id(buffers)] = {
                k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in buffers.items()}


def bind_conv(cls, runtime, inputs, positions, metadata, buffers):
    if not enabled() or metadata is None:
        return
    if os.getenv("ROLE_TYPE") == "DECODE" and os.getenv("ATOM_RTP_SSM_STATE_SLOTS") == "1":
        if not getattr(runtime, "_hybrid_enabled", False):
            raise RuntimeError("Single-state SSM requires active hybrid replay commit hooks")
    if getattr(metadata, "rtp_conv_metadata", None) is not None:
        return
    tagged = getattr(inputs,"attention_inputs_by_tag",{})
    conv_inputs = tagged.get("gdn_conv")
    if conv_inputs is None:
        raise ValueError("Split storage requires gdn_conv attention inputs")
    scratch = None
    if torch.cuda.is_current_stream_capturing():
        scratch = runtime._split_conv_graph_buffers[id(buffers)]
    conv = cls._build_gdn_metadata(conv_inputs,
        seq_size_per_block=int(runtime.kv_cache.get_seq_size_per_block("gdn_conv")),
        num_tokens=int(positions.numel()),cg_bufs=scratch)
    if conv.num_prefills and getattr(runtime._rtp_model_config,"atom_qwen4_exp_prefix_cache_enabled",False):
        conv.ssm_checkpoints = cls._prefill_state_checkpoints(conv_inputs,conv,
            int(runtime.kv_cache.get_seq_size_per_block("gdn_conv")),
            int(runtime.kv_cache.get_kernel_seq_size_per_block("gdn_conv")))
        conv.ssm_conv_state_len = metadata.ssm_conv_state_len
    metadata.rtp_conv_metadata = conv


# The target-verify graph shares sequence geometry between SSM and convolution.
# Generate both sets of indices and validate both tables in one kernel.
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
import triton
import triton.language as tl

_split_target_scope = ContextVar("rtp_split_target_scope", default=None)


@triton.jit
def _split_target_state_slots_kernel(
    state_table, conv_table, lengths, prefixes,
    state_reads, state_writes, conv_reads, conv_writes, invalid_rows,
    rows, state_sr, state_sc, conv_sr, conv_sc,
    state_cols: tl.constexpr, conv_cols: tl.constexpr,
    block_size: tl.constexpr, query_len: tl.constexpr, tile: tl.constexpr,
):
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    live = row < rows
    length = tl.load(lengths + row, live, 0)
    read_col = tl.maximum(length - 2, 0) // block_size
    write_col = tl.maximum(length - 1, 0) // block_size
    real = live & (tl.load(prefixes + row, live, 0) > 0)
    last = write_col + query_len - 1
    invalid = real & ((read_col >= state_cols) | (last >= state_cols)
                      | (read_col >= conv_cols) | (last >= conv_cols))
    for token in tl.static_range(query_len):
        col = write_col + token
        state = tl.load(state_table + row * state_sr + tl.minimum(col, state_cols - 1) * state_sc,
                        live & (col < state_cols) & (length >= 2), -1)
        conv = tl.load(conv_table + row * conv_sr + tl.minimum(col, conv_cols - 1) * conv_sc,
                       live & (col < conv_cols) & (length >= 2), -1)
        invalid |= real & ((state < 0) | (conv < 0))
        tl.store(state_writes + row * query_len + token, tl.maximum(state, 0), live)
        tl.store(conv_writes + row * query_len + token, tl.maximum(conv, 0), live)
    state = tl.load(state_table + row * state_sr + tl.minimum(read_col, state_cols - 1) * state_sc,
                    live & (read_col < state_cols) & (length >= 2), -1)
    conv = tl.load(conv_table + row * conv_sr + tl.minimum(read_col, conv_cols - 1) * conv_sc,
                   live & (read_col < conv_cols) & (length >= 2), -1)
    invalid |= real & ((state < 0) | (conv < 0))
    for token in tl.static_range(query_len):
        tl.store(state_reads + row * query_len + token, tl.maximum(state, 0), live)
        tl.store(conv_reads + row * query_len + token, tl.maximum(conv, 0), live)
    tl.store(invalid_rows + tl.program_id(0), tl.sum(invalid.to(tl.int32), 0))


@contextmanager
def target_metadata_scope(runtime, inputs, buffers):
    token = None
    primary = inputs.attention_inputs
    if (enabled() and os.getenv("ROLE_TYPE") == "DECODE"
            and os.getenv("ATOM_RTP_SSM_STATE_SLOTS") == "1"
            and getattr(primary, "is_target_verify", False)
            and torch.cuda.is_current_stream_capturing()):
        conv = inputs.attention_inputs_by_tag.get("gdn_conv")
        if conv is None or buffers is None:
            raise RuntimeError("Split target metadata requires convolution inputs and graph buffers")
        if runtime.kv_cache.get_seq_size_per_block("linear") != runtime.kv_cache.get_seq_size_per_block("gdn_conv"):
            raise ValueError("Fused split target metadata requires matching block spans")
        scratch = runtime._split_conv_graph_buffers[id(buffers)]
        token = _split_target_scope.set((primary.kv_cache_kernel_block_id_device.data_ptr(), conv, scratch))
    try:
        yield
    finally:
        if token is not None:
            _split_target_scope.reset(token)


def target_slots(attn_inputs, state_table, lengths, prefixes, reads, writes,
                 invalid, batch, block_size, query_len):
    scope = _split_target_scope.get()
    if scope is None or state_table.data_ptr() != scope[0]:
        return None
    conv_inputs, scratch = scope[1:]
    conv_table = conv_inputs.kv_cache_kernel_block_id_device
    if conv_table.ndim != 2 or conv_table.shape[0] != state_table.shape[0] or conv_table.shape[1] <= 0:
        raise ValueError("Invalid convolution graph state table")
    cr = scratch["target_read_slots_i32"][:batch]
    cw = scratch["target_write_slots_i32"][:batch]
    if cr.shape != reads.shape or cw.shape != writes.shape:
        raise ValueError("Split convolution graph buffers differ from SSM buffers")
    _split_target_state_slots_kernel[(triton.cdiv(batch, 128),)](
        state_table, conv_table, lengths, prefixes, reads, writes, cr, cw, invalid,
        batch, state_table.stride(0), state_table.stride(1), conv_table.stride(0), conv_table.stride(1),
        state_table.shape[1], conv_table.shape[1], block_size, query_len, 128,
    )
    return cr, cw, conv_table, conv_inputs


def attach_target_conv_metadata(metadata, slots):
    if slots is None:
        return
    conv = copy(metadata)
    conv.spec_state_indices_in_tensor, conv.spec_state_indices_tensor = slots[:2]
    conv.rtp_target_verify_block_map = slots[2]
    conv.rtp_attn_inputs = slots[3]
    metadata.rtp_conv_metadata = conv
