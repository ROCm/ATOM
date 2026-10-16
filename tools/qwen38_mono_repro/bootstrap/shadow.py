"""Restore request-owned state around an eager native decode reference."""
import torch
import os
from atom.utils.forward_context import get_forward_context

active_trace = None


def clone_layer(output):
    streams, pending = output
    return (streams.clone(), None if pending is None else tuple(t.clone() for t in pending))


def trace_metrics(actual, reference):
    rows = []
    for index in sorted(actual.keys() & reference.keys()):
        a, b = actual[index], reference[index]
        row = dict(layer=index, streams=metrics(a[0], b[0]))
        if a[1] is not None and b[1] is not None:
            row['pending'] = metrics(a[1][0], b[1][0])
            row['raw'] = metrics(a[1][1], b[1][1])
        rows.append(row)
    return rows


def install_mono_trace():
    from atom.models.qwen4_exp_mono.runner import QwenMonoRunner
    original = QwenMonoRunner._layer
    def layer(self, task, *args, **kwargs):
        result = original(self, task, *args, **kwargs)
        if active_trace is not None:
            active_trace[task.layer_id] = clone_layer(result)
        return result
    QwenMonoRunner._layer = layer
    original_pull = QwenMonoRunner._tp8_pull
    def pull(self, layer_id, output, view):
        result = original_pull(self, layer_id, output, view)
        if active_trace is not None and layer_id in active_trace:
            streams, pending = active_trace[layer_id]
            active_trace[layer_id] = (streams, (output.clone(), pending[1]))
        return result
    QwenMonoRunner._tp8_pull = pull


def native_reference(runner, input_ids):
    from atom.models.qwen4_exp_mono.dispatch import MonoDecodeModel
    forward = get_forward_context()
    if forward.context.is_prefill or forward.context.forward_mode.use_cudagraph:
        raise RuntimeError('Same-state shadow reference requires eager decode')
    metadata = forward.attn_metadata
    model = runner.model.model if isinstance(runner.model, MonoDecodeModel) else runner.model
    saved = []
    seen = set()
    trace = {}
    handles = []
    if os.environ.get('GEN_SHADOW_LAYERS') == '1':
        for index, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(
                lambda module, args, output, index=index: trace.__setitem__(index, clone_layer(output))))

    def preserve(tensor, indices):
        if tensor is None:
            return
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
        if key in seen:
            return
        indices = indices[indices >= 0].long().unique()
        if bool((indices >= tensor.shape[0]).any()):
            raise RuntimeError('Shadow state index exceeds cache capacity')
        seen.add(key)
        saved.append((tensor, indices, tensor.index_select(0, indices).clone()))

    gdn = metadata.gdn_metadata
    write = gdn.non_spec_state_indices_tensor
    read = getattr(gdn, 'non_spec_state_indices_in_tensor', None)
    slots = torch.cat([write.flatten(), (write if read is None else read).flatten()])
    for cache in forward.kv_cache_data.values():
        # GDN state pool is [slots, history/channels] or [slots, heads, K, V].
        # QSA pages are handled using the attention's public cache views below.
        if cache.k_cache.ndim == 3:
            preserve(cache.k_cache, slots)
            preserve(cache.v_cache, slots)
    qsa = metadata.qsa_metadata.for_tokens(input_ids.numel())
    for layer in model.model.layers:
        if layer.layer_type != 'linear_attention':
            attention = layer.self_attn
            page_size = attention.k_cache.shape[1]
            # Include explicit destinations even when a page table is padded.
            destinations = qsa.slot_mapping[qsa.slot_mapping >= 0] // page_size
            compressed_slots = getattr(qsa, 'compressed_slot_mapping', None)
            if compressed_slots is not None:
                compressed_size = attention.compressed_key_cache.shape[1]
                compressed_pages = compressed_slots[compressed_slots >= 0] // compressed_size
                destinations = torch.cat([destinations, compressed_pages])
            pages = torch.cat([qsa.block_tables.flatten(), destinations])
            for name in ('k_cache', 'v_cache', 'raw_key_cache',
                         'compressed_key_cache', 'rope_position_cache'):
                preserve(getattr(attention, name, None), pages)
    ple = metadata.ple_metadata
    if ple is not None:
        ple_slots = torch.cat([ple.state_indices_in.flatten(), ple.state_indices_out.flatten()])
        preserve(ple.ngram_state, ple_slots)
        preserve(ple.conv_state, ple_slots)
    try:
        hidden = model(input_ids, forward.context.positions)
        logits = model.compute_logits(hidden)
        return hidden.clone(), logits.clone(), trace
    finally:
        for handle in handles:
            handle.remove()
        for tensor, indices, values in saved:
            tensor.index_copy_(0, indices, values)


def metrics(actual, reference):
    actual, reference = actual.float(), reference.float()
    delta = actual - reference
    return dict(rel_l2=float(delta.norm() / reference.norm().clamp_min(1e-12)),
                max_abs=float(delta.abs().max()), finite=bool(torch.isfinite(actual).all()))
