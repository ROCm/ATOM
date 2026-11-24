# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The mixed decode view's row checks, and the one mixed carrier both builders use."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from test_mixed_decode_view import make_batch

from atom.utils.forward_context import AttnState


@pytest.fixture(scope="module", autouse=True)
def kernel_imports():
    with atom_without_aiter():
        yield


def _mixed_carrier(*args):
    from atom.model_ops.attentions.deepseek_v4_attn import _mixed_carrier as build

    return build(*args)


def _batch(n_p=2, n_d=3, num_spec_step=1):
    batch = make_batch(num_spec_step=num_spec_step)
    batch.num_scheduled_tokens[2:] = 1
    batch.total_tokens_num_decode = 3
    batch.total_tokens_num = 19
    batch.scheduled_tokens = batch.scheduled_tokens[:19]
    return batch


def _meta(cu, slots, max_q):
    return SimpleNamespace(
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        state_slot_out=torch.tensor(slots, dtype=torch.int32),
        max_seqlen_q=max_q,
        max_seqlen_k=max_q,
        has_cached=False,
        total_kv=0,
        state=AttnState.PREFILL_PREFIX if max_q > 1 else AttnState.DECODE,
        swa_dest_rows=None,
    )


def test_carrier_has_whole_batch_spans_and_state_slots():
    prefill = _meta([0, 5, 9], [7, 8], 5)
    decode = _meta([0, 1, 2, 3], [4, 5, 6], 1)
    merged = _mixed_carrier(prefill, decode, 2, 3, 9, 3, 6)
    assert merged.is_mixed is True
    assert merged.cu_seqlens_q.tolist() == [0, 5, 9, 10, 11, 12, 12]
    # Padded to running_bs with 0, as prepare_decode pads.
    assert merged.state_slot_out.tolist() == [7, 8, 4, 5, 6, 0]
    assert merged.num_prefill_seqs == 2
    assert merged.sampled_token_indices.tolist() == [4, 8, 9, 10, 11]
    assert (merged.num_prefill_tokens, merged.num_decode_tokens) == (9, 3)
    assert merged.prefill_attn_metadata is prefill
    assert merged.decode_attn_metadata is decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="writes draft KV on GPU")
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    "prefill_lengths,decode_lengths,running_bs",
    [([5, 4], [8, 8], 8), ([5] * 24, [8], 32), ([5, 4], [1, 3], 4)],
)
def test_dspark_context_writer_consumes_padded_mixed_carrier(
    monkeypatch, fp8, prefill_lengths, decode_lengths, running_bs
):
    from aiter import dtypes

    from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
        DENSE_RATIO,
        UnifiedPoolGeometry,
    )
    from atom.model_ops.v4_kernels.state_writes import swa_write_reference
    from atom.models.deepseek_v4_dspark import DSparkLayer
    from atom.utils import forward_context

    lengths = prefill_lengths + decode_lengths
    n_p, n_d = len(prefill_lengths), len(decode_lengths)
    slots = list(range(len(lengths), 0, -1))

    def meta(counts, request_slots, state):
        cu = torch.tensor([0, *counts], device="cuda", dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        return SimpleNamespace(
            cu_seqlens_q=cu,
            state_slot_out=torch.tensor(
                request_slots, device="cuda", dtype=torch.int32
            ),
            max_seqlen_q=max(counts),
            max_seqlen_k=128,
            has_cached=True,
            total_kv=0,
            state=state,
            swa_dest_rows=None,
        )

    carrier = _mixed_carrier(
        meta(prefill_lengths, slots[:n_p], AttnState.PREFILL_PREFIX),
        meta(decode_lengths, slots[n_p:], AttnState.DECODE),
        n_p,
        n_d,
        sum(prefill_lengths),
        sum(decode_lengths),
        running_bs,
    )
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(running_bs=running_bs), attn_metadata=carrier
        ),
    )
    geometry = UnifiedPoolGeometry(
        [DENSE_RATIO],
        num_blocks=2,
        num_slots=len(lengths) + 1,
        ring_slots=11,
        block_size=256,
    )
    window = geometry.window_params(DENSE_RATIO)
    positions = torch.cat(
        [torch.arange(20 * i, 20 * i + count) for i, count in enumerate(lengths)]
    ).to(device="cuda", dtype=torch.int64)
    dims = [512, 64] if fp8 else [576]
    types = [dtypes.fp8, torch.bfloat16] if fp8 else [torch.bfloat16]
    rows = tuple(
        (torch.arange(sum(lengths) * dim, device="cuda").reshape(-1, dim) % 17).to(
            dtype
        )
        for dim, dtype in zip(dims, types)
    )
    planes = [
        torch.full((geometry.plane_rows, dim), -1, device="cuda", dtype=dtype)
        for dim, dtype in zip(dims, types)
    ]
    expected = [plane.clone() for plane in planes]
    real_cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    real_slots = torch.tensor(slots, device="cuda", dtype=torch.int32)
    for row, reference in zip(rows, expected):
        swa_write_reference(row, positions, real_cu, real_slots, reference, window, 11)
    layer = SimpleNamespace(
        attn=SimpleNamespace(
            swa_plane=planes[0],
            swa_plane_rope=planes[1] if fp8 else None,
            swa_window=window,
            layer_name="draft",
        ),
        write_per_batch=11,
    )
    DSparkLayer.write_context_kv_rows(layer, rows, positions)
    torch.cuda.synchronize()
    for actual, reference in zip(planes, expected):
        # Check the whole plane, including slot 0 shared by all padded rows.
        torch.testing.assert_close(actual.float(), reference.float(), rtol=0, atol=0)


@pytest.mark.parametrize("decode_width", [0, 1, 3])
def test_lm_head_selects_prefill_ends_and_all_decode_tokens(monkeypatch, decode_width):
    from atom.distributed import ulysses_sp
    from atom.model_ops import embed_head
    from atom.utils.forward_context import AttentionMetaData

    prefill = _meta([0, 5, 9], [7, 8], 5)
    decode = _meta([0, decode_width, 2 * decode_width], [4, 5], decode_width)
    decode.state = AttnState.DECODE
    metadata = (
        _mixed_carrier(prefill, decode, 2, 2, 9, 2 * decode_width, 4)
        if decode_width
        else AttentionMetaData(cu_seqlens_q=prefill.cu_seqlens_q)
    )
    context = SimpleNamespace(is_prefill=True, is_draft=False, scheduled_bs=2)
    monkeypatch.setattr(embed_head, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(embed_head, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(ulysses_sp, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(
        embed_head,
        "get_forward_context",
        lambda: SimpleNamespace(context=context, attn_metadata=metadata),
    )
    monkeypatch.setattr(
        embed_head,
        "tgemm",
        SimpleNamespace(mm=lambda x, w, b, *, out=None: torch.mm(x, w.T, out=out)),
    )
    head = SimpleNamespace(
        weight=torch.eye(2),
        bias=None,
        tp_size=1,
        _can_use_dp_sharded_head=lambda _: False,
    )
    head.local_logits = embed_head.ParallelLMHead.local_logits.__get__(head)
    head.gather_logits = embed_head.ParallelLMHead.gather_logits.__get__(head)
    hidden = torch.arange((9 + 2 * decode_width) * 2).float().reshape(-1, 2)
    actual = embed_head.ParallelLMHead.forward(head, hidden)
    expected = torch.cat((hidden[[4, 8]], hidden[9:]))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("num_spec_step", [0, 1])
def test_prepare_mixed_calls_both_halves_with_their_real_signatures(
    monkeypatch, num_spec_step
):
    """Drives `prepare_mixed` with the two halves stubbed, each binding its
    arguments against the real method. A builder signature change that the
    mixed call does not follow then fails here instead of on the first mixed
    GPU step (`prepare_prefill` gained `running_tokens` and the call kept two)."""
    import contextlib
    import inspect

    import numpy as np

    from atom.model_ops.attentions import deepseek_v4_attn as v4

    builder_cls = v4.DeepseekV4AttentionMetadataBuilder
    n_p, n_d, p_tokens = 2, 3, 16
    batch = _batch(n_p=n_p, n_d=n_d, num_spec_step=num_spec_step)
    batch.total_seqs_num_prefill = n_p
    batch.num_scheduled_tokens = np.array([8, 8, 1, 1, 1], dtype=np.int32)
    calls = {}

    def _stub(name, positions, metadata):
        sig = inspect.signature(getattr(builder_cls, name))

        def run(self, *args, **kwargs):
            calls[name] = sig.bind(self, *args, **kwargs).arguments
            return metadata, positions

        return run

    cu = np.arange(n_d + 1, dtype=np.int32)  # one decode token per row
    builder = builder_cls.__new__(builder_cls)
    builder.model_runner = SimpleNamespace(
        forward_vars={"cu_seqlens_q": SimpleNamespace(np=cu)},
        tokenID_processor=SimpleNamespace(
            num_rejected=(
                np.array([99, 98, 1, 0, 2], dtype=np.int32) if num_spec_step else None
            )
        ),
        h2d_groups={"decode_metadata": SimpleNamespace(publish=lambda: None)},
    )
    monkeypatch.setattr(
        builder_cls,
        "mixed_prefill_bank_active",
        lambda self, b: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        builder_cls,
        "prepare_prefill",
        _stub("prepare_prefill", torch.arange(16), _meta([0, 8, 16], [7, 8], 8)),
    )
    monkeypatch.setattr(
        builder_cls,
        "stage_decode",
        _stub("stage_decode", torch.arange(3), _meta([0, 1, 2, 3], [4, 5, 6], 1)),
    )
    monkeypatch.setattr(builder_cls, "finish_decode", lambda *a: None)

    merged, positions = builder.prepare_mixed(batch, n_p + n_d, p_tokens + n_d)

    pre = calls["prepare_prefill"]
    assert (pre["running_bs"], pre["running_tokens"]) == (n_p, p_tokens)
    dec = calls["stage_decode"]
    if num_spec_step:
        assert dec["num_rejected"].tolist() == [1, 0, 2]
    else:
        assert dec["num_rejected"] is None
    if num_spec_step == 0:
        assert dec["batch"].scheduled_spec_decode_tokens is None
    assert (dec["running_bs"], dec["running_tokens"], dec["max_seqlen_q"]) == (
        n_d,
        n_d,
        1,
    )
    assert positions.numel() == p_tokens + n_d
    assert merged.is_mixed
    assert (merged.num_prefill_tokens, merged.num_decode_tokens) == (p_tokens, n_d)
    assert merged.cu_seqlens_q.tolist() == [0, 8, 16, 17, 18, 19]
    assert merged.sampled_token_indices.tolist() == [7, 15, 16, 17, 18]
    assert merged.state_slot_out.tolist() == [7, 8, 4, 5, 6]


def test_mixed_prefill_banks_exist_before_the_first_mixed_step(monkeypatch):
    """The banks are built up front, one per `forward_vars` ring slot, and a
    lookup never builds one: allocating on the first mixed step would take
    memory the KV cache was already sized against."""
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    get_bank = DeepseekV4AttentionMetadataBuilder._get_mixed_prefill_bank
    build_banks = DeepseekV4AttentionMetadataBuilder.build_mixed_prefill_banks
    ring = [{"mtp_k": 0}, {"mtp_k": 1}]
    publication_owner = object()
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    builder = SimpleNamespace(
        mixed_prefill_groups=DeepseekV4AttentionMetadataBuilder.mixed_prefill_groups,
        model_runner=SimpleNamespace(
            _fv_ring=ring,
            _fv_idx=1,
            h2d_groups={},
            build_h2d_slot=lambda *args, **kw: (publication_owner, {}),
        ),
    )
    build_banks(builder)
    variables, owner, groups = get_bank(builder)
    assert variables == ring[1] and variables is not ring[1]
    assert owner is publication_owner and groups == {}
    builder.model_runner._fv_idx = 0
    assert get_bank(builder)[0] == ring[0]


@pytest.mark.parametrize("mixed", [False, True])
def test_runner_builds_the_banks_before_warmup(monkeypatch, mixed):
    from unittest.mock import Mock

    from atom.model_engine import model_runner as mr
    from atom.model_engine import model_setup
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder as Builder,
    )
    from atom.model_ops.fused_moe import sp_mega_moe
    from atom.utils import debug_helper

    class WarmupReached(Exception):
        pass

    def builder(runner=None, *, model_runner=None):
        obj = Builder.__new__(Builder)
        obj.model_runner = model_runner
        return obj

    cfg = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v4", architectures=["test"]),
        kv_cache_block_size=4,
        kv_cache_dtype="bf16",
        enforce_eager=True,
        tensor_parallel_size=1,
        tp_world_size=1,
        torch_dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(
            data_parallel_rank_local=0, data_parallel_size=1
        ),
        torch_profiler_dir=None,
        speculative_config=None,
        max_num_batched_tokens=8,
        max_num_seqs=4,
        max_model_len=16,
        quant_config=Mock(),
        enable_tbo=False,
        enable_mixed_prefill_decode=mixed,
    )
    monkeypatch.setattr(mr, "set_current_atom_config", lambda _: None)
    monkeypatch.setattr(mr, "DynamicChunkingWorker", lambda _: None)
    monkeypatch.setattr(mr, "attn_family", lambda _: None)
    monkeypatch.setattr(mr, "has_mla_indexer", lambda _: False)
    monkeypatch.setattr(
        mr,
        "get_attn_backend",
        lambda _: SimpleNamespace(get_builder_cls=lambda: builder),
    )
    monkeypatch.setattr(mr, "init_exit_handler", lambda _: None)
    monkeypatch.setattr(mr, "tokenIDProcessor", lambda *a: None)
    monkeypatch.setattr(mr, "sampler_from_config", lambda _: None)
    monkeypatch.setattr(model_setup, "support_model_arch_dict", {"test": "test"})
    monkeypatch.setattr(model_setup, "resolve_obj_by_qualname", lambda _: object)
    monkeypatch.setattr(mr, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(mr, "mono_decode_installers", {})
    monkeypatch.setattr(mr.PPAuxRelay, "build", lambda *a: None)
    monkeypatch.setattr(mr, "initialize_eplb_runtime", lambda _: None)
    monkeypatch.setattr(sp_mega_moe, "initialize_sp_mega_moe", lambda *a: None)
    monkeypatch.setattr(debug_helper, "install_block_forward_hooks", lambda _: 0)
    monkeypatch.setattr(debug_helper, "maybe_dump_weights_and_exit", lambda _: None)
    monkeypatch.setattr(torch, "set_default_dtype", lambda _: None)
    monkeypatch.setattr(torch, "set_default_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "Event", lambda *a: None)

    class Runner(mr.ModelRunner):
        def _setup_device_and_distributed(self, *args):
            self.device = torch.device("cpu")

        def _build_and_load_model(self, config, model_class, device):
            return torch.nn.Identity()

        def allocate_forward_vars(self):
            self.forward_vars = {"mtp_k": 0}

        def _arm_lossless(self, _):
            pass

        def _state_pool_names(self):
            return []

        def _refuse_unrelayed_pp_taps(self, _):
            pass

        def _init_forward_vars_ring(self):
            self._fv_ring = [self.forward_vars]
            self._fv_idx = 0

        def _init_h2d_publication(self):
            self.h2d_groups = {}
            self.h2d_owner = object()

        def build_h2d_slot(self, variables, event, *, partial=False):
            assert partial
            return object(), {}

        def _maybe_warmup(self):
            assert self.h2d_groups == {}
            if mixed:
                bank, owner, groups = (
                    self.attn_metadata_builder._get_mixed_prefill_bank()
                )
                assert bank == self.forward_vars and bank is not self.forward_vars
                assert owner is not None and owner is not self.h2d_owner
                assert groups == {}
            else:
                assert not hasattr(self.attn_metadata_builder, "_mixed_prefill_banks")
            raise WarmupReached

    with pytest.raises(WarmupReached):
        Runner(0, cfg)


@pytest.mark.parametrize("slots", [1, 2])
def test_prefill_bank_clones_only_its_writers_and_preserves_live_bindings(
    monkeypatch, slots
):
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )
    from atom.utils import CpuGpuBuffer

    groups = {
        "positions": "positions",
        "cu_seqlens_q": "early",
        "block_tables": "seq_metadata",
        "v4_state": "v4_state",
        "plans": "v4_plans",
        "indexer": "v4_indexer",
        "batch_id_per_q_token": None,
        "input_ids": "input_ids",
        "sampling": "sampling",
        "spec": "spec_decode",
        "decode_only": "v4_qo",
        "decode_scratch": None,
    }
    ring = [
        {
            name: CpuGpuBuffer(
                4,
                dtype=torch.int32,
                device="cpu",
                pin_memory=False,
                publication_group=group,
            )
            for name, group in groups.items()
        }
        for _ in range(slots)
    ]
    bound = []

    def bind(variables, event, *, partial):
        assert partial
        bound.append(set(variables))
        return object(), {}

    builder = DeepseekV4AttentionMetadataBuilder.__new__(
        DeepseekV4AttentionMetadataBuilder
    )
    builder.model_runner = SimpleNamespace(
        _fv_ring=ring, _fv_idx=0, h2d_groups={}, build_h2d_slot=bind
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    builder.build_mixed_prefill_banks()
    shared = {"input_ids", "sampling", "spec", "decode_only", "decode_scratch"}
    for i, original in enumerate(ring):
        bank = builder._mixed_prefill_banks[i][0]
        assert bound[i] == set(groups) - shared
        for name, buffer in original.items():
            assert (bank[name] is buffer) == (name in shared)
            if name not in shared:
                assert bank[name].cpu.data_ptr() != buffer.cpu.data_ptr()
                assert bank[name].gpu.data_ptr() != buffer.gpu.data_ptr()
        bank["positions"].cpu.fill_(i + 1)
    if slots == 2:
        assert builder._mixed_prefill_banks[0][0]["positions"].cpu.tolist() == [1] * 4


def test_mixed_fused_swa_mask_preserves_prefill_history():
    prefill = _meta([0, 5], [7], 5)
    decode = _meta([0, 2, 3], [4, 5], 1)
    decode.batch_id_per_q_token = torch.tensor([0, 0, 1], dtype=torch.int32)
    decode.swa_dest_rows = {
        1: torch.tensor([31, 32, 77]),
        4: torch.tensor([15, 16, 22]),
    }
    merged = _mixed_carrier(prefill, decode, 1, 2, 5, 3, 3)
    assert merged.batch_id_per_q_token.tolist() == [-1] * 5 + [0, 0, 1]
    assert merged.swa_dest_rows[1].tolist() == [0] * 5 + [31, 32, 77]
    assert merged.swa_dest_rows[4].tolist() == [0] * 5 + [15, 16, 22]


def test_qkn_token_slice_keeps_every_output_plane():
    from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import QKNormRopeOut

    fields = {
        name: torch.arange(8).view(4, 2) for name in QKNormRopeOut.__dataclass_fields__
    }
    result = QKNormRopeOut(**fields).slice_tokens(slice(2, 4))
    for name, value in fields.items():
        assert getattr(result, name).tolist() == [[4, 5], [6, 7]]
        assert (
            getattr(result, name).untyped_storage().data_ptr()
            == value.untyped_storage().data_ptr()
        )
