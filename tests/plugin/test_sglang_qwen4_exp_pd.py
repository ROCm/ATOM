# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Native Flash PD: preserve page and slot addressing across distinct pools."""

import ctypes
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.disaggregation.base.conn import KVArgs, StateType

from atom.plugin.sglang.qwen4_exp_pd import NativePLESlotState, append_native_qsa_state


def empty_args():
    args = KVArgs()
    for name in (
        "state_types",
        "state_data_ptrs",
        "state_data_lens",
        "state_item_lens",
        "state_dim_per_tensor",
        "state_conv_shard_groups",
        "state_slice_outer_counts",
        "state_layer_ids",
    ):
        setattr(args, name, [])
    return args


def page_pool(pages, layer_ids, fill):
    tensors = [
        torch.full((pages, width, 1, 8), fill, dtype=torch.bfloat16)
        for width in (64, 16)
        for _ in layer_ids
    ]
    return SimpleNamespace(_atom_pd_qsa=(tensors, layer_ids * 2))


def test_qsa_transfer_uses_page_bytes_and_includes_draft():
    src, dst = page_pool(5, [3, 7], 11), page_pool(7, [3, 7], 0)
    src_d, dst_d = page_pool(5, [0], 19), page_pool(7, [0], 0)
    a, b = empty_args(), empty_args()
    append_native_qsa_state(a, src, src_d, 48)
    append_native_qsa_state(b, dst, dst_d, 48)
    assert a.state_types == [StateType.QSA_COMPRESSED]
    assert a.state_layer_ids == [[3, 7, 3, 7, 48, 48]]
    assert a.state_item_lens == b.state_item_lens
    # Different physical page numbers, including a prompt's partially filled
    # last page. Copy using the advertised wire descriptors, not tensor slices.
    for sp, dp, size in zip(
        a.state_data_ptrs[0], b.state_data_ptrs[0], a.state_item_lens[0]
    ):
        for si, di in ((1, 4), (3, 2)):
            ctypes.memmove(dp + di * size, sp + si * size, size)
    for s, d in zip(
        src._atom_pd_qsa[0] + src_d._atom_pd_qsa[0],
        dst._atom_pd_qsa[0] + dst_d._atom_pd_qsa[0],
    ):
        torch.testing.assert_close(d[4], s[1])
        torch.testing.assert_close(d[2], s[3])
        assert torch.count_nonzero(d[0]) == 0


def test_ple_slot_handoff_and_reset_preserve_other_requests():
    state = NativePLESlotState(
        torch.arange(60).reshape(5, 4, 3).clone(),
        torch.arange(10).reshape(5, 2).clone(),
        99,
    )
    expected = state.get_cpu_slots(torch.tensor([1]))
    state.copy_slots(torch.tensor([1]), torch.tensor([3]))
    state.reset_slots(torch.tensor([1]))
    assert state.conv[1].count_nonzero() == 0
    assert state.ngram[1].tolist() == [99, 99]
    torch.testing.assert_close(state.conv[3], expected[0][0])
    torch.testing.assert_close(state.ngram[3], expected[1][0])
    state.load_cpu_slots(expected, torch.tensor([4]))
    torch.testing.assert_close(state.conv[4], expected[0][0])
    entries = list(state.iter_transfer_state_entries())
    assert all(entry[2] is None for entry in entries)  # TP replicated
    assert len({entry[3] for entry in entries}) == 2


def test_missing_draft_refused_and_unused_sglang_qsa_replaced():
    from sglang.srt.disaggregation.utils import append_state_component

    pool = page_pool(2, [3], 0)
    with pytest.raises(RuntimeError, match="draft QSA"):
        append_native_qsa_state(empty_args(), pool, SimpleNamespace(), 48)
    args = empty_args()
    append_state_component(args, StateType.MAMBA, [100], [200], [20])
    append_state_component(args, StateType.QSA_PENDING, [300], [400], [40])
    append_state_component(args, StateType.QSA_COMPRESSED, [500], [600], [60])
    append_native_qsa_state(args, pool, None, 48)
    assert args.state_types == [StateType.MAMBA, StateType.QSA_COMPRESSED]
    assert args.state_data_ptrs[0] == [100]
    assert args.state_data_ptrs[1] == [t.data_ptr() for t in pool._atom_pd_qsa[0]]
    assert all(len(getattr(args, name)) == 2 for name in vars(args))


def test_non_native_pool_unchanged():
    args = empty_args()
    append_native_qsa_state(args, SimpleNamespace(), None, 48)
    assert not args.state_types


def test_byte_verifier_detects_corruption_and_matches_remapped_slots():
    from atom.plugin.sglang.qwen4_exp_pd_verify import hash_rows

    source = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    destination = torch.zeros(6, 16)
    destination[5] = source[2]
    expected = hash_rows(source, torch.tensor([2]))
    assert hash_rows(destination, torch.tensor([5])) == expected
    destination[5, 15] += 1
    assert hash_rows(destination, torch.tensor([5])) != expected
    assert hash_rows(destination, torch.tensor([2])) != expected


def test_native_ple_is_included_in_sglang_mamba_wire_descriptors():
    from sglang.srt.mem_cache.memory_pool import MambaPool

    # Exercise the real descriptor builder without allocating a GPU model.
    pool = object.__new__(MambaPool)
    pool.mamba_cache = SimpleNamespace(
        conv=[torch.zeros(1, 5, 2, 3)], temporal=torch.zeros(1, 5, 2, 4, 4)
    )
    pool.mamba_layer_ids = [0]
    pool.conv_slice_axis = 0
    pool.conv_shard_groups = None
    ple = NativePLESlotState(
        torch.zeros(5, 4, 9), torch.zeros(5, 2, dtype=torch.int64), 0
    )
    pool.register_slot_state(ple)
    pointers, lengths, item_lengths = pool.get_contiguous_buf_infos()
    assert pointers[-2:] == [ple.conv.data_ptr(), ple.ngram.data_ptr()]
    assert lengths[-2:] == [ple.conv.nbytes, ple.ngram.nbytes]
    assert item_lengths[-2:] == [ple.conv[0].nbytes, ple.ngram[0].nbytes]
    assert pool.get_state_dim_per_tensor()[-2:] == [0, 0]  # TP replicated
    assert pool.get_state_slice_outer_counts()[-2:] == [1, 1]
    assert pool.get_state_conv_shard_groups()[-2:] == [None, None]
    assert len(set(pool.get_state_layer_ids()[-2:])) == 2


@pytest.mark.parametrize(
    "pp_size,radix_disabled,backend,checksum,error",
    [
        (2, True, "mooncake", False, "PP=1"),
        (1, False, "mooncake", False, "disable-radix-cache"),
        (1, True, "fake", False, "requires Mooncake"),
        (1, True, "mooncake", True, "single-index KV checksum"),
    ],
)
def test_unsupported_pd_modes_fail_before_allocating_buffers(
    monkeypatch, pp_size, radix_disabled, backend, checksum, error
):
    from sglang.srt import runtime_context

    from atom.plugin.sglang.qwen4_exp_pd import prepare_native_pd_buffers

    monkeypatch.setattr(
        runtime_context,
        "get_disagg",
        lambda: SimpleNamespace(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend=backend,
            disaggregation_enable_kv_checksum=checksum,
        ),
    )
    monkeypatch.setattr(
        runtime_context,
        "get_memory",
        lambda: SimpleNamespace(disable_radix_cache=radix_disabled),
    )
    monkeypatch.setattr(
        runtime_context, "get_parallel", lambda: SimpleNamespace(pp_size=pp_size)
    )
    wrapper = type(
        "Wrapper", (), {"__module__": "atom.plugin.sglang.models.qwen4_exp"}
    )()
    with pytest.raises(ValueError, match=error):
        prepare_native_pd_buffers(SimpleNamespace(model=wrapper))


def test_integrated_mode_does_not_touch_model_or_pools(monkeypatch):
    from sglang.srt import runtime_context

    from atom.plugin.sglang.qwen4_exp_pd import prepare_native_pd_buffers

    monkeypatch.setattr(
        runtime_context,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_mode="null"),
    )
    prepare_native_pd_buffers(SimpleNamespace())


@pytest.fixture
def native_pd_hooks(monkeypatch):
    from sglang.srt.disaggregation import decode, prefill, utils
    from sglang.srt.model_executor.model_runner import ModelRunner

    from atom.plugin.sglang.qwen4_exp_pd import install_qwen4_exp_pd_patch

    # Exercise the installed initializer without allocating/capturing GPU graphs.
    def eager_init(self, capture_decode_cuda_graph=True):
        assert self.server_args.disable_cuda_graph
        self.eager_initialized = True

    monkeypatch.setattr(ModelRunner, "init_cuda_graphs", eager_init)
    # The installer updates aliases imported by the two PD queue modules.
    # Track all of them so the fixture restores them after each test.
    for module in (utils, prefill, decode):
        monkeypatch.setattr(module, "setup_state_kv_args", module.setup_state_kv_args)
    monkeypatch.delenv("ATOM_FLASH_PD_VERIFY_DIR", raising=False)
    install_qwen4_exp_pd_patch()
    return SimpleNamespace(utils=utils, prefill=prefill, decode=decode)


@pytest.mark.parametrize("entry", ["utils", "prefill", "decode"])
def test_native_pd_registration_refuses_skipped_initialization(native_pd_hooks, entry):
    args = empty_args()
    args.state_types = [StateType.MAMBA]
    setup = getattr(native_pd_hooks, entry).setup_state_kv_args
    with pytest.raises(RuntimeError, match="target QSA buffers were not prepared"):
        setup(args, SimpleNamespace(), total_kv_layers=48)
    # Refuse before the original setup clears/registers any state descriptors.
    assert args.state_types == [StateType.MAMBA]


@pytest.mark.parametrize("entry", ["prefill", "decode"])
def test_native_pd_registration_accepts_prepared_target(native_pd_hooks, entry):
    args = empty_args()
    pool = page_pool(2, [3], 0)
    getattr(native_pd_hooks, entry).setup_state_kv_args(args, pool, total_kv_layers=48)
    assert args.state_types == [StateType.QSA_COMPRESSED]
    assert args.state_data_ptrs == [[t.data_ptr() for t in pool._atom_pd_qsa[0]]]


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_disabled_graph_startup_prepares_qsa_and_ple(
    native_pd_hooks, monkeypatch, role
):
    from sglang.srt import runtime_context
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MambaPool
    from sglang.srt.model_executor.model_runner import ModelRunner

    from atom.plugin.sglang import qwen4_exp_bridge as bridge

    monkeypatch.setattr(
        runtime_context,
        "get_disagg",
        lambda: SimpleNamespace(
            disaggregation_mode=role, disaggregation_transfer_backend="mooncake"
        ),
    )
    monkeypatch.setattr(
        runtime_context, "get_memory", lambda: SimpleNamespace(disable_radix_cache=True)
    )
    monkeypatch.setattr(
        runtime_context, "get_parallel", lambda: SimpleNamespace(pp_size=1)
    )
    monkeypatch.setattr(bridge, "_server_args", lambda: SimpleNamespace(page_size=64))
    monkeypatch.setattr(bridge, "_is_capturing", lambda: False)

    # Tiny CPU buffers go through the real Native QSA binding, PLE allocation,
    # Mamba sibling registration, and SGLang state descriptor builder.
    mamba = object.__new__(MambaPool)
    mamba.mamba_cache = SimpleNamespace(
        conv=[torch.zeros(1, 5, 2, 3)], temporal=torch.zeros(1, 5, 2, 4, 4)
    )
    mamba.mamba_layer_ids = [0]
    mamba.conv_slice_axis = 0
    mamba.conv_shard_groups = None
    req_pool = object.__new__(HybridReqToTokenPool)
    req_pool.mamba_pool = mamba
    layer = SimpleNamespace(is_qsa_attention=True, layer_num=3)
    layer.bind_caches = lambda *buffers: setattr(layer, "bound", buffers)
    native = SimpleNamespace(
        modules=lambda: [layer], parameters=lambda: iter([torch.zeros(1)])
    )
    wrapper = type(
        "Wrapper", (), {"__module__": "atom.plugin.sglang.models.qwen4_exp"}
    )()
    wrapper._atom_lm = native
    wrapper.atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen4_exp_text",
            hidden_size=4,
            hc_count=2,
            ngram_size=3,
            ple_conv_kernel_size=4,
            ple_layer_ids=[2],
            eos_token_id=9,
            indexer_head_dim=8,
            indexer_compress_ratio=4,
        ),
        torch_dtype=torch.bfloat16,
    )
    kv = (torch.zeros(128, 1, 4), torch.zeros(128, 1, 4))
    pool = SimpleNamespace(size=64, page_size=64, get_kv_buffer=lambda _: kv)
    runner = object.__new__(ModelRunner)
    runner.model = wrapper
    runner.token_to_kv_pool = pool
    runner.req_to_token_pool = req_pool
    runner.server_args = SimpleNamespace(disable_cuda_graph=True)
    worker = object.__new__(TpModelWorker)
    worker._model_runner = runner
    worker.model_runner_list = [runner]

    # Use the real scheduler -> TP worker -> patched ModelRunner call chain.
    Scheduler.init_all_cuda_graphs(SimpleNamespace(tp_worker=worker, draft_worker=None))
    assert runner.eager_initialized
    ple = req_pool._atom_pd_ple
    assert ple in mamba._slot_siblings
    assert ple.conv is native._atom_ple_conv_state
    assert ple.ngram is native._atom_ple_ngram_state
    assert layer.bound[2].data_ptr() == pool._atom_pd_qsa[0][0].data_ptr()

    args = empty_args()
    getattr(native_pd_hooks, role).setup_state_kv_args(
        args, pool, total_kv_layers=48, req_to_token_pool=req_pool
    )
    assert args.state_types == [StateType.MAMBA, StateType.QSA_COMPRESSED]
    assert args.state_data_ptrs[0][-2:] == [ple.conv.data_ptr(), ple.ngram.data_ptr()]
    assert args.state_data_ptrs[1] == [t.data_ptr() for t in pool._atom_pd_qsa[0]]
