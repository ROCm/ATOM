# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Register Native ATOM Flash side caches with SGLang's PD transport.

Native QSA uses page-indexed raw and compressed keys, unlike SGLang's
request-indexed pending ring. Both therefore use QSA_COMPRESSED's full-page
index payload. PLE is indexed by physical Mamba slots and follows their
lifecycle through SGLang's SlotIndexedState interface.
"""

from __future__ import annotations

import functools
import logging
import os
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class NativePLESlotState:
    def __init__(self, conv, ngram, eos):
        self.conv, self.ngram, self.eos = conv, ngram, eos

    def reset_slots(self, indices):
        self.conv[indices] = 0
        self.ngram[indices] = self.eos

    def copy_slots(self, src_index, dst_index):
        self.conv[dst_index] = self.conv[src_index]
        self.ngram[dst_index] = self.ngram[src_index]

    def get_cpu_slots(self, indices):
        return tuple(
            t[indices].to("cpu", non_blocking=True) for t in (self.conv, self.ngram)
        )

    def load_cpu_slots(self, data, indices):
        for tensor, saved in zip((self.conv, self.ngram), data):
            tensor[indices] = saved.to(tensor.device, non_blocking=True)

    def iter_transfer_state_entries(self):
        # Unique uint32 IDs, separate from SGLang's own PLE sibling pools.
        yield "atom_ple_conv", self.conv, None, (1 << 32) - 3
        yield "atom_ple_ngram", self.ngram, None, (1 << 32) - 4


def prepare_native_pd_buffers(runner):
    from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

    if get_disagg().disaggregation_mode == "null":
        return
    wrapper = runner.model
    if type(wrapper).__module__ not in (
        "atom.plugin.sglang.models.qwen4_exp",
        "atom.plugin.sglang.models.qwen4_exp_nextn_wrapper",
    ):
        return
    if get_parallel().pp_size != 1 or not get_memory().disable_radix_cache:
        raise ValueError(
            "Native ATOM Qwen4Exp PD currently requires PP=1 and --disable-radix-cache"
        )
    if get_disagg().disaggregation_transfer_backend != "mooncake":
        raise ValueError("Native ATOM Qwen4Exp PD currently requires Mooncake")
    if getattr(get_disagg(), "disaggregation_enable_kv_checksum", False):
        raise ValueError(
            "SGLang's single-index KV checksum cannot cover Native QSA pages "
            "and Mamba slots together. For single-node verification, use "
            "ATOM_FLASH_PD_VERIFY_DIR on both P and D instead."
        )

    from atom.plugin.sglang.qwen4_exp_bridge import (
        _ensure_ple_states,
        _hf_text_config,
        _mamba_physical_slots,
        bind_qsa_caches,
    )

    native = getattr(wrapper, "_atom_lm", None)
    is_draft = native is None
    if is_draft:
        native = wrapper.model
    pool = runner.token_to_kv_pool
    req_pool = runner.req_to_token_pool
    batch = SimpleNamespace(token_to_kv_pool=pool, req_to_token_pool=req_pool)
    bind_qsa_caches(native, batch, wrapper.atom_config)
    raw = native._atom_qwen4_exp_qsa_raw
    compressed = native._atom_qwen4_exp_qsa_compressed
    layers = [m for m in native.modules() if getattr(m, "is_qsa_attention", False)]
    layer_ids = [0 if is_draft else int(m.layer_num) for m in layers]
    pool._atom_pd_qsa = (
        list(raw.unbind(0)) + list(compressed.unbind(0)),
        layer_ids * 2,
    )
    if os.environ.get("ATOM_FLASH_PD_VERIFY_DIR"):
        from atom.plugin.sglang.qwen4_exp_pd_verify import register_verify_views

        register_verify_views(pool, layers, is_draft)
    if not is_draft and not hasattr(req_pool, "_atom_pd_ple"):
        hf = _hf_text_config(wrapper.atom_config)
        if getattr(hf, "ple_layer_ids", None):
            conv, ngram = _ensure_ple_states(
                native, wrapper.atom_config, _mamba_physical_slots(req_pool)
            )
            eos = getattr(hf, "eos_token_id", 0)
            eos = eos[0] if isinstance(eos, (tuple, list)) else eos
            sibling = NativePLESlotState(conv, ngram, int(eos))
            req_pool.mamba_pool.register_slot_state(sibling)
            req_pool._atom_pd_ple = sibling
    logger.info(
        "ATOM Flash PD buffers ready: draft=%s, QSA page planes=%s, PLE=%s",
        is_draft,
        len(pool._atom_pd_qsa[0]),
        hasattr(req_pool, "_atom_pd_ple"),
    )


def append_native_qsa_state(kv_args, pool, draft_pool, total_kv_layers):
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.disaggregation.utils import append_state_component

    if not hasattr(pool, "_atom_pd_qsa"):
        # Non-Native callers may use this helper. The Native Flash setup
        # wrapper below rejects an unprepared target before reaching here.
        return
    tensors, layer_ids = pool._atom_pd_qsa
    tensors, layer_ids = list(tensors), list(layer_ids)
    if draft_pool is not None:
        if not hasattr(draft_pool, "_atom_pd_qsa") or total_kv_layers is None:
            raise RuntimeError("Native ATOM PD draft QSA buffers were not prepared")
        draft_tensors, draft_ids = draft_pool._atom_pd_qsa
        tensors.extend(draft_tensors)
        layer_ids.extend(total_kv_layers + i for i in draft_ids)
    # SGLang still allocates a QSATokenToKVPool for this config even though
    # the plugin keeps ATOM's attention backend. Those index buffers are not
    # written by Native compute. Remove both upstream QSA components before
    # advertising the actual buffers, preserving Mamba and other components.
    for index in reversed(range(len(kv_args.state_types))):
        if kv_args.state_types[index] in (
            StateType.QSA_PENDING,
            StateType.QSA_COMPRESSED,
        ):
            for field in (
                "state_types",
                "state_data_ptrs",
                "state_data_lens",
                "state_item_lens",
                "state_dim_per_tensor",
                "state_conv_shard_groups",
                "state_slice_outer_counts",
                "state_layer_ids",
            ):
                getattr(kv_args, field).pop(index)
    append_state_component(
        kv_args,
        StateType.QSA_COMPRESSED,
        [t.data_ptr() for t in tensors],
        [t.nbytes for t in tensors],
        [t[0].nbytes for t in tensors],
        layer_ids=layer_ids,
    )
    logger.info(
        "ATOM Flash PD registered %s QSA page planes (including draft)", len(tensors)
    )


def install_qwen4_exp_pd_patch():
    from sglang.srt.disaggregation import utils
    from sglang.srt.model_executor.model_runner import ModelRunner

    original_graphs = ModelRunner.init_cuda_graphs
    if getattr(original_graphs, "_atom_flash_pd", False):
        return

    @functools.wraps(original_graphs)
    def init_cuda_graphs(self, *args, **kwargs):
        # SGLang calls this initializer even when graph capture is disabled.
        prepare_native_pd_buffers(self)
        return original_graphs(self, *args, **kwargs)

    init_cuda_graphs._atom_flash_pd = True
    ModelRunner.init_cuda_graphs = init_cuda_graphs
    original_setup = utils.setup_state_kv_args

    @functools.wraps(original_setup)
    def setup(
        kv_args,
        token_to_kv_pool,
        draft_token_to_kv_pool=None,
        total_kv_layers=None,
        req_to_token_pool=None,
    ):
        # Installed only while constructing a Native Flash model. Do not let
        # a skipped initializer register SGLang's unused QSA buffers instead.
        if not hasattr(token_to_kv_pool, "_atom_pd_qsa"):
            raise RuntimeError(
                "Native ATOM Qwen4Exp PD target QSA buffers were not prepared. "
                "ModelRunner.init_cuda_graphs() must prepare Native state before "
                "PD registration, including when CUDA graph capture is disabled."
            )
        original_setup(
            kv_args,
            token_to_kv_pool,
            draft_token_to_kv_pool,
            total_kv_layers,
            req_to_token_pool,
        )
        append_native_qsa_state(
            kv_args, token_to_kv_pool, draft_token_to_kv_pool, total_kv_layers
        )

    utils.setup_state_kv_args = setup
    # Scheduler imports these modules before loading a model.
    from sglang.srt.disaggregation import decode, prefill

    prefill.setup_state_kv_args = setup
    decode.setup_state_kv_args = setup

    from atom.plugin.sglang.qwen4_exp_pd_verify import install_verify_hooks

    install_verify_hooks()
