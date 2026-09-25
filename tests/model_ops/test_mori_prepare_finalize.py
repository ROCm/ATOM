# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoRI prepare/finalize: AITER's EP sentinel column and the launch geometry.

The sentinel: AITER fused_moe drops the last top-k column from its tuned-kernel
key whenever an expert_mask is given, so the six real columns MoRI delivers
look up top-5 rows and miss every tuned kernel. The column must be appended
after the receive-buffer trim, and combine must keep the caller's six columns.

The geometry: MoRI's dispatch/combine take (block_num, rdma_block_num,
warp_per_block), so the warp count has to be passed by keyword, and the grid
follows the group's largest per-rank count, looked up in MoRI's own tables.
"""

import bisect
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("mori")

import atom.model_ops.fused_moe.modular_kernel as mk
import atom.model_ops.fused_moe.mori_prepare_finalize as mpf

NUM_EXPERTS = 384
TOPK = 6
HIDDEN = 7168


def _expert_map():
    # FusedMoE's EP layout: one slot per expert plus the trailing -1 sentinel.
    return torch.cat(
        (torch.arange(NUM_EXPERTS, dtype=torch.int32), torch.tensor([-1]))
    ).to(torch.int32)


def _prepare_finalize():
    pf = mpf.MoriPrepareAndFinalize.__new__(mpf.MoriPrepareAndFinalize)
    pf._launch_tables = {}
    pf._mask_pad_rows = False
    return pf


class _FakeMoriOp:
    def __init__(self, *, kernel_type="IntraNode", rank=1, rows=64):
        self.config = SimpleNamespace(
            kernel_type=SimpleNamespace(name=kernel_type),
            world_size=8,
            num_experts_per_token=TOPK,
            use_external_inp_buf=True,
            quant_type="none",
            rank=rank,
        )
        self.rows = rows
        self.calls = []

    def dispatch(self, input, weights, scales, indices, **kwargs):
        self.calls.append(("dispatch", kwargs))
        rows = self.rows
        return (
            input.new_zeros((rows, input.shape[1])),
            weights.new_ones((rows, weights.shape[1])),
            None,
            indices.new_zeros((rows, indices.shape[1])),
            torch.tensor([rows], dtype=torch.int32),
        )

    def combine(self, input, weights, indices, **kwargs):
        self.calls.append(("combine", kwargs))
        self.combine_indices = indices
        return (input.new_zeros((indices.shape[0], input.shape[1])),)


@pytest.fixture
def mi355x(monkeypatch):
    """Resolve MoRI's shipped gfx950 MI355X tables without a GPU."""
    from aiter.jit.utils import chip_info
    from mori.ops import tuning_config

    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(tuning_config, "detect_gpu_model", lambda: "mi355x")
    monkeypatch.setattr(mpf, "get_cu_num", lambda: 256)
    monkeypatch.setattr(mpf, "_LAUNCH_POLICY", "tuned")
    mpf.mori_tuned_launch_table.cache_clear()
    yield
    mpf.mori_tuned_launch_table.cache_clear()


def _context(monkeypatch, across_dp, *, is_prefill=False):
    context = SimpleNamespace(
        running_tokens_across_dp=None if across_dp is None else tuple(across_dp),
        is_prefill=is_prefill,
    )
    monkeypatch.setattr(
        mpf, "get_forward_context", lambda: SimpleNamespace(context=context)
    )


# --- sentinel -----------------------------------------------------------------


def test_sentinel_is_appended_for_aiter():
    ids = torch.randint(0, NUM_EXPERTS, (5, TOPK), dtype=torch.int32)
    weights = torch.rand(5, TOPK)
    out_ids, out_weights = _prepare_finalize().adapt_routing_for_fused_moe(
        ids, weights, NUM_EXPERTS, _expert_map()
    )
    assert out_ids.shape == (5, TOPK + 1)
    assert out_weights.shape == (5, TOPK + 1)
    assert torch.equal(out_ids[:, :TOPK], ids)
    assert torch.equal(out_weights[:, :TOPK], weights)
    assert (out_ids[:, -1] == NUM_EXPERTS).all()
    assert (out_weights[:, -1] == 0).all()
    # The sentinel indexes the masked tail slot of expert_mask.
    expert_mask = (_expert_map() > -1).to(torch.int32)
    assert expert_mask.numel() == NUM_EXPERTS + 1
    assert expert_mask[out_ids[:, -1].long()].sum() == 0


def test_no_sentinel_without_the_extended_expert_map():
    ids = torch.zeros(5, TOPK, dtype=torch.int32)
    weights = torch.ones(5, TOPK)
    pf = _prepare_finalize()
    for expert_map in (None, torch.arange(NUM_EXPERTS, dtype=torch.int32)):
        out_ids, out_weights = pf.adapt_routing_for_fused_moe(
            ids, weights, NUM_EXPERTS, expert_map
        )
        assert out_ids is ids and out_weights is weights


def test_sentinel_after_trim_and_combine_keeps_real_columns(monkeypatch):
    """fused_moe sees the trimmed rows plus one column; combine the caller's."""
    across_dp = (2, 3)
    arena_rows = 64
    _context(monkeypatch, across_dp)
    monkeypatch.setattr(
        mk,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(running_tokens_across_dp=across_dp)
        ),
    )
    seen = {}

    def fake_fused_moe(a1, w1, w2, weights, ids, expert_mask, *args, **kwargs):
        seen["a1_rows"] = a1.shape[0]
        seen["ids"] = ids
        seen["weights"] = weights
        return a1.new_zeros(a1.shape)

    monkeypatch.setattr(mk, "fused_moe", fake_fused_moe)
    monkeypatch.setattr(mpf, "_LAUNCH_POLICY", "legacy")
    monkeypatch.setattr(mpf, "get_cu_num", lambda: 256)

    pf = _prepare_finalize()
    op = _FakeMoriOp(rows=arena_rows)
    pf._mori_ops = (op,)
    pf.dispatch_format = mpf.MoriDispatchFormat(
        dtype=torch.bfloat16, quant_type=None, scale_dim=0, scale_type_size=4
    )
    kernel = mk.FusedMoEModularKernel(pf)

    hidden_states = torch.zeros(3, 16, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, NUM_EXPERTS, (3, TOPK), dtype=torch.int32)
    kernel(
        hidden_states,
        torch.zeros(48, 1),
        torch.zeros(48, 1),
        torch.rand(3, TOPK),
        topk_ids,
        global_num_experts=NUM_EXPERTS,
        expert_map=_expert_map(),
        expert_mask=(_expert_map() > -1).to(torch.int32),
    )

    assert seen["a1_rows"] == sum(across_dp)
    assert seen["ids"].shape == (sum(across_dp), TOPK + 1)
    assert seen["weights"].shape == (sum(across_dp), TOPK + 1)
    assert (seen["ids"][:, -1] == NUM_EXPERTS).all()
    # finalize hands mori's combine this rank's own six-column routing.
    assert op.calls[-1][0] == "combine"
    assert op.combine_indices is topk_ids


# --- launch geometry ------------------------------------------------------------


# Per-rank tokens -> ((dispatch blocks, warps), (combine blocks, warps)) from
# MoRI's gfx950_mi355x_IntraNode_ep8 tables: bf16, hidden 7168, top-6, push
# (use_external_inp_buf) combine without quantization.
_MI355X_EP8_DSV4 = {
    7: ((192, 4), (128, 8)),
    56: ((192, 4), (128, 8)),
    112: ((220, 4), (256, 8)),
    224: ((192, 4), (256, 8)),
    448: ((256, 4), (256, 16)),
    1792: ((225, 8), (256, 16)),
    4096: ((208, 12), (256, 16)),
    16384: ((208, 12), (256, 16)),
}


@pytest.mark.parametrize("tokens", sorted(_MI355X_EP8_DSV4))
def test_tuned_geometry_follows_mori_tables(mi355x, monkeypatch, tokens):
    _context(monkeypatch, (tokens,) * 8)
    pf, op = _prepare_finalize(), _FakeMoriOp()
    dispatch = pf._get_launch_config("dispatch", op, tokens, torch.bfloat16, HIDDEN)
    combine = pf._get_launch_config("combine", op, tokens, torch.bfloat16, HIDDEN)
    assert (dispatch, combine) == _MI355X_EP8_DSV4[tokens]


def test_table_matches_mori_lookup_everywhere(mi355x):
    from mori.ops.tuning_config import TuningConfigManager

    manager = TuningConfigManager.get_instance("gfx950", "IntraNode", 8, "mi355x")
    for phase, rules, kwargs in (
        ("dispatch", manager.dispatch_rules, {}),
        ("combine", manager.combine_rules, {"zero_copy": False, "quant_type": "none"}),
    ):
        ceilings, geometries = mpf.mori_tuned_launch_table(
            phase, 8, torch.bfloat16, HIDDEN, TOPK, **kwargs
        )
        for tokens in range(0, 20000, 7):
            params = TuningConfigManager.lookup(
                rules, torch.bfloat16, tokens, HIDDEN, topk=TOPK, **kwargs
            )
            got = geometries[bisect.bisect_left(ceilings, tokens)]
            assert got == (params.block_num, params.warp_per_block), (phase, tokens)


def test_geometry_keys_on_the_group_not_this_rank(mi355x, monkeypatch):
    """A decoding rank still carries the prefilling peer's traffic."""
    _context(monkeypatch, (7, 7, 4096, 7, 7, 7, 7, 7))
    pf, op = _prepare_finalize(), _FakeMoriOp()
    assert pf._get_launch_config("combine", op, 7, torch.bfloat16, HIDDEN) == (
        256,
        16,
    )


def test_geometry_without_group_counts_uses_local_count(mi355x, monkeypatch):
    _context(monkeypatch, None)
    pf, op = _prepare_finalize(), _FakeMoriOp()
    assert pf._get_launch_config("dispatch", op, 112, torch.bfloat16, HIDDEN) == (
        220,
        4,
    )


def test_block_num_is_capped_at_cu_count(mi355x, monkeypatch):
    monkeypatch.setattr(mpf, "get_cu_num", lambda: 80)
    _context(monkeypatch, (4096,) * 8)
    pf, op = _prepare_finalize(), _FakeMoriOp()
    for phase in ("dispatch", "combine"):
        block_num, warp_per_block = pf._get_launch_config(
            phase, op, 4096, torch.bfloat16, HIDDEN
        )
        assert block_num <= 80 and warp_per_block <= 16


@pytest.mark.parametrize("is_prefill, expected", [(True, (128, 16)), (False, (64, 4))])
def test_legacy_policy(mi355x, monkeypatch, is_prefill, expected):
    monkeypatch.setattr(mpf, "_LAUNCH_POLICY", "legacy")
    _context(monkeypatch, (4096,) * 8, is_prefill=is_prefill)
    pf, op = _prepare_finalize(), _FakeMoriOp()
    for phase in ("dispatch", "combine"):
        assert (
            pf._get_launch_config(phase, op, 4096, torch.bfloat16, HIDDEN) == expected
        )


def test_non_intranode_kernels_keep_the_legacy_grid(mi355x, monkeypatch):
    _context(monkeypatch, (112,) * 8)
    pf, op = _prepare_finalize(), _FakeMoriOp(kernel_type="InterNodeV1")
    assert pf._get_launch_config("dispatch", op, 112, torch.bfloat16, HIDDEN) == (
        64,
        4,
    )


def test_warp_per_block_is_passed_by_keyword(mi355x, monkeypatch):
    """Positionally it lands in rdma_block_num, which IntraNode ignores."""
    _context(monkeypatch, (112,) * 8)
    pf = _prepare_finalize()
    op = _FakeMoriOp()
    pf._mori_ops = (op,)
    pf.dispatch_format = mpf.MoriDispatchFormat(
        dtype=torch.bfloat16, quant_type=None, scale_dim=0, scale_type_size=4
    )
    a1 = torch.zeros(112, HIDDEN, dtype=torch.bfloat16)
    topk_ids = torch.zeros(112, TOPK, dtype=torch.int32)
    pf.prepare(a1, torch.ones(112, TOPK), topk_ids, NUM_EXPERTS, None, False, None)
    pf.finalize(
        None, torch.zeros(64, HIDDEN, dtype=torch.bfloat16), None, topk_ids, False
    )
    assert op.calls == [
        ("dispatch", {"block_num": 220, "warp_per_block": 4}),
        ("combine", {"block_num": 256, "warp_per_block": 8}),
    ]


# --- zero-copy (pull) combine ----------------------------------------------------


class _FakeZeroCopyOp(_FakeMoriOp):
    """Adds MoRI's registered combine input buffer: [max_recv, hidden] rows."""

    def __init__(self, *, max_recv=64, hidden=HIDDEN, **kwargs):
        super().__init__(**kwargs)
        self.config.max_token_type_size = 2
        self.config.hidden_dim = hidden
        self.max_recv = max_recv
        self.symm = torch.zeros(max_recv * hidden * 2, dtype=torch.uint8)

    def get_registered_combine_input_buffer(self, dtype, hidden_dim=-1):
        rows = self.symm.numel() // (hidden_dim * dtype.itemsize)
        return self.symm.view(dtype)[: rows * hidden_dim].view(rows, hidden_dim)

    def combine(self, input, weights, indices, **kwargs):
        self.combine_input = input
        return super().combine(input, weights, indices, **kwargs)


def _zero_copy_pf(mode, *, op=None):
    pf = _prepare_finalize()
    pf._mori_ops = (op or _FakeZeroCopyOp(),)
    pf._zero_copy_combine = mode
    pf._combine_inputs = {}
    return pf


def _intranode_config(**overrides):
    import mori

    fields = {
        "kernel_type": mori.ops.EpDispatchCombineKernelType.IntraNode,
        "quant_type": "none",
        **overrides,
    }
    return SimpleNamespace(config=SimpleNamespace(**fields))


@pytest.mark.parametrize(
    "mode, expected", [("0", None), ("1", "direct"), ("copy", "copy")]
)
def test_zero_copy_modes(mode, expected):
    assert mpf.resolve_zero_copy_combine(mode, _intranode_config()) == expected


def test_zero_copy_rejects_unknown_values():
    with pytest.raises(ValueError, match="ATOM_MORI_ZERO_COPY_COMBINE"):
        mpf.resolve_zero_copy_combine("true", _intranode_config())


def test_zero_copy_needs_intranode_without_a_combine_codec():
    import mori

    for config in (
        _intranode_config(kernel_type=mori.ops.EpDispatchCombineKernelType.AsyncLL),
        _intranode_config(quant_type="fp8_direct_cast"),
        _intranode_config(quant_type="fp8_blockwise"),
    ):
        assert mpf.resolve_zero_copy_combine("1", config) is None


def test_push_combine_is_unchanged_by_default(mi355x, monkeypatch):
    _context(monkeypatch, (112,) * 8)
    pf = _zero_copy_pf(None)
    out = torch.ones(64, HIDDEN, dtype=torch.bfloat16)
    pf.finalize(None, out, None, torch.zeros(112, TOPK, dtype=torch.int32), False)
    op = pf._mori_ops[0]
    assert op.combine_input is out
    assert "use_external_inp_buf" not in op.calls[-1][1]
    assert not op.symm.any()
    assert pf.expert_output_buffer(64, HIDDEN, torch.bfloat16) is None


def test_direct_output_is_pulled_in_place(mi355x, monkeypatch):
    _context(monkeypatch, (112,) * 8)
    pf = _zero_copy_pf("direct")
    op = pf._mori_ops[0]
    out = pf.expert_output_buffer(40, HIDDEN, torch.bfloat16)
    registered = op.get_registered_combine_input_buffer(torch.bfloat16, HIDDEN)
    assert out.shape == (40, HIDDEN) and out.is_contiguous()
    assert out.data_ptr() == registered.data_ptr()
    out.fill_(3)
    pf.finalize(None, out, None, torch.zeros(112, TOPK, dtype=torch.int32), False)
    assert op.calls[-1][1]["use_external_inp_buf"] == 0
    assert op.combine_input.data_ptr() == registered.data_ptr()
    assert (registered[:40] == 3).all() and not registered[40:].any()


def test_copy_mode_stages_the_rows_peers_read(mi355x, monkeypatch):
    _context(monkeypatch, (112,) * 8)
    pf = _zero_copy_pf("copy")
    op = pf._mori_ops[0]
    assert pf.expert_output_buffer(40, HIDDEN, torch.bfloat16) is None
    out = torch.randn(40, HIDDEN).to(torch.bfloat16)
    pf.finalize(None, out, None, torch.zeros(112, TOPK, dtype=torch.int32), False)
    registered = op.get_registered_combine_input_buffer(torch.bfloat16, HIDDEN)
    assert op.calls[-1][1]["use_external_inp_buf"] == 0
    assert op.combine_input.data_ptr() == registered.data_ptr()
    assert torch.equal(registered[:40], out)


def test_no_direct_output_while_a_ubatch_runs(monkeypatch):
    pf = _zero_copy_pf("direct")
    monkeypatch.setattr(mpf.MoriPrepareAndFinalize, "supports_async", lambda s: True)
    assert pf.expert_output_buffer(40, HIDDEN, torch.bfloat16) is None


def test_expert_output_wider_than_the_rows_is_refused():
    pf = _zero_copy_pf("direct")
    with pytest.raises(RuntimeError, match="does not fit"):
        pf.expert_output_buffer(4, HIDDEN, torch.float32)
    with pytest.raises(RuntimeError, match="does not fit"):
        pf.expert_output_buffer(4, HIDDEN * 2, torch.bfloat16)


def test_zero_copy_combine_uses_the_pull_tables(mi355x, monkeypatch):
    from mori.ops.tuning_config import TuningConfigManager

    manager = TuningConfigManager.get_instance("gfx950", "IntraNode", 8, "mi355x")
    push = _prepare_finalize()
    pull = _zero_copy_pf("direct")
    for tokens in (7, 56, 112, 224, 448, 1792, 4096):
        _context(monkeypatch, (tokens,) * 8)
        expected = TuningConfigManager.lookup(
            manager.combine_rules,
            torch.bfloat16,
            tokens,
            HIDDEN,
            zero_copy=True,
            quant_type="none",
            topk=TOPK,
        )
        op = pull._mori_ops[0]
        got = pull._get_launch_config(
            "combine", op, tokens, torch.bfloat16, HIDDEN, pull_combine=True
        )
        assert got == (expected.block_num, expected.warp_per_block), tokens
        # The same slot-0 op pushes as TBO ubatch 0: the push rules again.
        assert (
            pull._get_launch_config("combine", op, tokens, torch.bfloat16, HIDDEN)
            == _MI355X_EP8_DSV4[tokens][1]
        )
        # Dispatch has one table, and the push op keeps the push rules.
        assert pull._get_launch_config(
            "dispatch", op, tokens, torch.bfloat16, HIDDEN
        ) == push._get_launch_config(
            "dispatch", _FakeMoriOp(), tokens, torch.bfloat16, HIDDEN
        )
        assert (
            push._get_launch_config(
                "combine", _FakeMoriOp(), tokens, torch.bfloat16, HIDDEN
            )
            == _MI355X_EP8_DSV4[tokens][1]
        )


def test_fused_moe_writes_into_the_pulled_buffer(monkeypatch):
    """End to end through the modular kernel: trimmed rows, sentinel column,
    output= the registered rows, and combine pulls exactly that buffer."""
    across_dp = (2, 3)
    _context(monkeypatch, across_dp)
    monkeypatch.setattr(
        mk,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(running_tokens_across_dp=across_dp)
        ),
    )
    seen = {}

    def fake_fused_moe(a1, w1, w2, weights, ids, expert_mask, *args, **kwargs):
        output = kwargs["output"]
        seen["ids"] = ids
        assert output.shape == (a1.shape[0], w2.shape[1])
        output.fill_(5)
        return output

    monkeypatch.setattr(mk, "fused_moe", fake_fused_moe)
    monkeypatch.setattr(mpf, "_LAUNCH_POLICY", "legacy")
    monkeypatch.setattr(mpf, "get_cu_num", lambda: 256)

    op = _FakeZeroCopyOp(rows=64, hidden=16)
    pf = _zero_copy_pf("direct", op=op)
    pf.dispatch_format = mpf.MoriDispatchFormat(
        dtype=torch.bfloat16, quant_type=None, scale_dim=0, scale_type_size=4
    )
    kernel = mk.FusedMoEModularKernel(pf)
    topk_ids = torch.randint(0, NUM_EXPERTS, (3, TOPK), dtype=torch.int32)
    kernel(
        torch.zeros(3, 16, dtype=torch.bfloat16),
        torch.zeros(48, 1),
        torch.zeros(48, 16, 1),
        torch.rand(3, TOPK),
        topk_ids,
        global_num_experts=NUM_EXPERTS,
        expert_map=_expert_map(),
        expert_mask=(_expert_map() > -1).to(torch.int32),
    )
    registered = op.get_registered_combine_input_buffer(torch.bfloat16, 16)
    rows = sum(across_dp)
    assert seen["ids"].shape == (rows, TOPK + 1)
    assert op.calls[-1][1]["use_external_inp_buf"] == 0
    assert op.combine_input.data_ptr() == registered.data_ptr()
    assert op.combine_input.shape == (rows, 16)
    assert (registered[:rows] == 5).all() and not registered[rows:].any()
    assert op.combine_indices is topk_ids
