# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Kimi-K3.1's midstep checkpoints: each interior one re-run as a head.

K3's chunk kernel keeps no state inside a forward, so its prefill appends each
interior checkpoint's head -- the row's tokens before the checkpoint, from the
row's initial state -- to the same varlen call as one more sequence. With that,
the prompt-end checkpoint no longer cuts a prefill in two. These tests pin the
index lists that build the call from the scheduler's reservations and what
the forward writes from them.
"""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from test_state_checkpoint import (
    BLOCK,
    PAGED_COPY_SPEC,
    PROMPT,
    ckpt_config,
    make_block_manager,
    stateful_seq,
)

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import (
    StateRuntime,
    StateTransfer,
    midstep_checkpoint_targets,
    midstep_rerun_rows,
)

#: K3's `conv_kernel - 1`.
CONV = 3


def runtime(readable):
    """K3's transfer with and without the re-run (`test_kda_layout_id` pins
    that the builder declares grid 1 and offset `CONV`)."""
    kw = {"readable_midstep": True, "midstep_grid": 1, "midstep_min_offset": CONV}
    return StateRuntime(
        transfer=StateTransfer.copy(
            PAGED_COPY_SPEC.layout_id, **(kw if readable else {})
        ),
        checkpoint_spec=PAGED_COPY_SPEC,
    )


def anchor_only_bm(readable=True):
    """Recipe F8's checkpoints: no ladder, so the prompt-end anchor is all."""
    config = ckpt_config(state_checkpoint_interval_tokens=-1, pool_entries={"state": 8})
    return make_block_manager(config, state_runtime=runtime(readable))


def admitted(bm, tokens):
    seq = stateful_seq(list(tokens))
    bm.allocate(seq, bm.can_allocate(seq))
    return seq


def rerun_rows(seqs, scheduled):
    batch = ScheduledBatch(
        seqs=dict(enumerate(seqs)),
        num_scheduled_tokens=scheduled,
        total_tokens_num=sum(scheduled),
        num_cached_tokens=[0] * len(seqs),
    )
    targets = midstep_checkpoint_targets(
        batch.state_rows,
        batch.num_cached_tokens,
        batch.num_scheduled_tokens,
        batch.state_slots_committed,
        batch.state_save_all,
        num_slots=64,
        grid=1,
        min_offset=CONV,
    )
    return midstep_rerun_rows(targets, scheduled)


class TestAnchorInsideAPrefill:
    def test_without_the_re_run_the_anchor_cuts_the_prompt(self):
        """PROMPT's anchor is its last hashable block: 40 of 44 tokens."""
        bm = anchor_only_bm(readable=False)
        assert bm.checkpoint_cut(admitted(bm, PROMPT), 0, 44) == 10 * BLOCK

    def test_with_it_the_prompt_prefills_in_one_step(self):
        bm = anchor_only_bm()
        seq = admitted(bm, PROMPT)
        assert bm.checkpoint_cut(seq, 0, 44) == 0
        bm.plan_midstep(seq, 0, 44)
        assert [p for _s, p, _h in seq.midstep_reservations] == [10 * BLOCK]

    def test_scarce_page_scratch_keeps_the_anchor_before_the_ladder(self):
        config = ckpt_config(
            state_checkpoint_interval_tokens=32, pool_entries={"state": 2}
        )
        bm = make_block_manager(config, state_runtime=runtime(True))
        seq = admitted(bm, PROMPT)
        assert [p for p, _h in bm.midstep_positions(seq, 0, 44)] == [32, 40]
        bm.plan_midstep(seq, 0, 44)
        assert [p for _s, p, _h in seq.midstep_reservations] == [40]
        bm.cancel_midstep(seq)
        assert bm.state.num_free() == 1

    def test_the_head_rides_behind_every_row(self):
        bm = anchor_only_bm()
        other = admitted(bm, range(100, 112))
        seq = admitted(bm, PROMPT)
        bm.plan_midstep(seq, 0, 44)
        [(scratch, _p, _h)] = seq.midstep_reservations

        rows = rerun_rows([other, seq], [12, 44])

        assert rows["head_ranges"] == [(12, 40)]
        assert rows["kda_rows"] == [0, 1, 1]
        assert rows["kda_cu"] == [0, 12, 56, 96]
        assert rows["rows"] == [1] and rows["slots"] == [scratch]
        assert rows["offs"] == [40] and rows["h_rows"] == [0]
        assert rows["is_end"] == [0]

    def test_an_anchor_at_the_forward_s_end_copies_the_row(self):
        bm = anchor_only_bm()
        seq = admitted(bm, PROMPT)
        bm.plan_midstep(seq, 0, 40)
        [(scratch, _p, _h)] = seq.midstep_reservations

        rows = rerun_rows([seq], [40])

        assert rows == {
            "rows": [0],
            "slots": [scratch],
            "offs": [40],
            "is_end": [1],
            "runtime": [seq.state_slot],
            "h_rows": [0],
        }


@pytest.fixture(scope="module")
def k3_modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            builder=load("atom.model_ops.attentions.kimi_mla_gdn_attn"),
            model=load("atom.models.kimi_k3.model"),
            writer=load("atom.model_ops.fla_ops.state_checkpoint"),
            initial=load("atom.model_ops.kimi_k3"),
        )


def test_the_state_pool_lends_scratch_only_where_it_is_borrowed(k3_modules):
    """One slot per request; the scratch on top is for the checkpoints that
    two steps hold at once, and only where K3's forward writes them."""
    K3 = k3_modules.builder
    from test_kda_layout_id import builder

    def extra(**kw):
        b = object.__new__(K3.KimiAiterMLAGDNMetadataBuilder)
        inputs = builder(K3._KimiMLAGDNCommon, **kw)
        b.model_runner = inputs.model_runner
        b.num_spec, b.replayssm = inputs.num_spec, inputs.replayssm
        b._kpool_size = inputs._kpool_size
        b.num_state_layers = lambda: 1
        b._state_shape_for_runner = lambda: ((3, 8), (1, 2, 2))
        b._state_dtypes = lambda: (torch.bfloat16, torch.float32)
        b._replayssm_bytes_per_slot = lambda: 0
        b.slots_per_req = lambda: 1
        b._kpool_tail_bytes = lambda: 0
        return b.state_spec().extra_entries

    assert extra() == K3.MIDSTEP_SCRATCH_SLOTS > 0
    assert extra(pp=2) == extra(model_type="glm5_next_text") == 0

    from atom.model_ops.attentions.gdn_attn import DECODE_SNAPSHOT_SCRATCH_SLOTS

    # Midstep and decode images borrow from the same pool; keep both budgets.
    assert extra(replayssm=True, interval=32) == (
        K3.MIDSTEP_SCRATCH_SLOTS + DECODE_SNAPSHOT_SCRATCH_SLOTS
    )
    assert extra(replayssm=True, interval=32, pp=2) == DECODE_SNAPSHOT_SCRATCH_SLOTS
    assert extra(replayssm=True, interval=-1) == K3.MIDSTEP_SCRATCH_SLOTS
    assert extra(replayssm=True, interval=32, prefix_caching=False) == 0
    assert extra(interval=-1, prefix_caching=False) == 0
    assert extra(replayssm=True, interval=0) == 0
    assert extra(kpool=4) == 0
    assert extra(interval=3) == extra(interval=4, dcp=2) == 0
    assert extra(replayssm=True, interval=3) == 0
    assert extra(interval=4) == extra(interval=5) == K3.MIDSTEP_SCRATCH_SLOTS
    assert extra(interval=-1, dcp=2) == K3.MIDSTEP_SCRATCH_SLOTS

    transfer = {"kv_connector": "moriio", "state_transfer_slots": 8}
    assert extra(replayssm=True, interval=32, kv_transfer_config=transfer) == (
        8 + K3.MIDSTEP_SCRATCH_SLOTS + DECODE_SNAPSHOT_SCRATCH_SLOTS
    )
    assert extra(replayssm=True, interval=3, kv_transfer_config=transfer) == 8
    assert extra(replayssm=True, interval=-1, kv_transfer_config=transfer) == (
        8 + K3.MIDSTEP_SCRATCH_SLOTS
    )


@pytest.mark.parametrize(
    "interval, expected", [(-1, -1), (0, 0), (511, 0), (512, 512), (513, 512)]
)
def test_builder_interval_matches_the_block_manager(interval, expected, k3_modules):
    from test_kda_layout_id import builder

    cls = k3_modules.builder._KimiMLAGDNCommon
    b = builder(cls, interval=interval, block_size=64, dcp=8)
    bm = make_block_manager(
        ckpt_config(
            state_checkpoint_interval_tokens=interval,
            kv_cache_block_size=64,
            decode_context_parallel_size=8,
        ),
        state_runtime=runtime(True),
    )
    assert (
        cls._checkpoint_interval(b) == bm.state_checkpoint_interval_tokens == expected
    )


def test_the_builder_hands_the_forward_its_rows(k3_modules):
    """`kda_cu` joins the int32 query starts the KDA call is cut by."""
    K3 = k3_modules.builder
    from test_kda_layout_id import builder

    bm = anchor_only_bm()
    seq = admitted(bm, PROMPT)
    bm.plan_midstep(seq, 0, 44)
    batch = ScheduledBatch(
        seqs={0: seq},
        num_scheduled_tokens=[44],
        total_tokens_num=44,
        num_cached_tokens=[0],
    )
    b = object.__new__(K3.KimiAiterMLAGDNMetadataBuilder)
    b.model_runner = builder(K3._KimiMLAGDNCommon).model_runner
    b.model_runner.mamba_k_cache = torch.empty(1, 64)
    b.device = "cpu"

    got = b._checkpoint_targets(batch)

    expected = rerun_rows([seq], [44])
    expected.pop("head_ranges")
    expected["kda_tokens"] = list(range(44)) + list(range(40))
    assert {k: v.tolist() for k, v in got.items()} == expected
    assert got["kda_cu"].dtype == torch.int32
    assert got["kda_tokens"].dtype == torch.int64
    assert (
        len(
            {
                v.untyped_storage().data_ptr()
                for k, v in got.items()
                if k != "kda_tokens"
            }
        )
        == 1
    )


@pytest.mark.parametrize("tokens", [8192, 32768])
def test_checkpoint_metadata_transfers_only_small_descriptors(
    tokens, monkeypatch, k3_modules
):
    builder = SimpleNamespace(
        device="cpu",
        _midstep_targets=lambda batch, grid: [(0, 2, tokens - 512, 0, 0)],
    )
    batch = SimpleNamespace(state_save_all=[[True]], num_scheduled_tokens=[tokens])
    transfers, make_tensor = [], torch.tensor

    def record(values, **kwargs):
        transfers.append(len(values))
        return make_tensor(values, **kwargs)

    monkeypatch.setattr(torch, "tensor", record)
    got = k3_modules.builder._KimiMLAGDNCommon._checkpoint_targets(builder, batch)
    assert transfers == [11]
    assert got["kda_cu"].tolist() == [0, tokens, 2 * tokens - 512]
    assert torch.equal(got["kda_tokens"][:tokens], torch.arange(tokens))
    assert torch.equal(got["kda_tokens"][tokens:], torch.arange(tokens - 512))


def test_finish_prefill_keeps_kda_targets_out_of_gdn_chunk_snapshots(
    monkeypatch, k3_modules
):
    K3 = k3_modules.builder
    from test_kda_layout_id import builder

    bm = anchor_only_bm()
    seq = admitted(bm, PROMPT)
    bm.plan_midstep(seq, 0, 44)
    batch = ScheduledBatch(
        seqs={0: seq},
        num_scheduled_tokens=[44],
        total_tokens_num=44,
        num_cached_tokens=[0],
    )
    b = object.__new__(K3.KimiAiterMLAGDNMetadataBuilder)
    b.model_runner = builder(K3._KimiMLAGDNCommon).model_runner
    b.model_runner.mamba_k_cache = torch.empty(1, 64)
    b.device = "cpu"
    meta = SimpleNamespace(non_spec_query_start_loc=torch.tensor([0, 44]))
    b.prepare_gdn_metadata = lambda *a, **kw: meta
    monkeypatch.setattr(K3.AiterMLAMetadataBuilder, "finish_prefill", lambda *a: None)
    attn = SimpleNamespace()

    b.finish_prefill(batch, attn, 1, 44)

    assert attn.gdn_metadata is meta
    expected = rerun_rows([seq], [44])
    expected.pop("head_ranges")
    expected["kda_tokens"] = list(range(44)) + list(range(40))
    assert {k: v.tolist() for k, v in meta.ssm_checkpoints.items()} == expected
    assert not hasattr(meta, "ssm_chunk_offsets")


@pytest.fixture
def cpu_checkpoint_writer(monkeypatch, k3_modules):
    class Kernel:
        def __getitem__(self, grid):
            def copy(
                h,
                pool,
                x,
                conv,
                rows,
                slots,
                offs,
                ends,
                runtime,
                chunks,
                h_rows,
                cu,
                *strides,
                **kw,
            ):
                assert grid[0] == len(rows)
                assert strides == (x.stride(0), *conv.stride())
                assert kw["STATE_VK"] == (pool.stride(-2) == 1)
                assert kw["CHUNK"] == 1 and kw["HAS_H_ROWS"]
                states = h.reshape(-1, *pool.shape[1:])
                for i in range(len(rows)):
                    src = pool[int(runtime[i])] if ends[i] else states[int(h_rows[i])]
                    pool[int(slots[i])].copy_(src)
                    end = int(cu[int(rows[i])] + offs[i])
                    conv[int(slots[i]), :, : kw["STATE_LEN"]].copy_(
                        x[end - kw["STATE_LEN"] : end].T
                    )

            return copy

    monkeypatch.setattr(k3_modules.writer, "_copy_checkpoints_kernel", Kernel())
    monkeypatch.setattr(k3_modules.writer.triton, "cdiv", lambda n, d: (n + d - 1) // d)
    return k3_modules.writer.write_state_checkpoints


@pytest.mark.parametrize("n", [0, 2])
@pytest.mark.parametrize(
    "hkv,d,ssm_blocks,conv_blocks",
    [(1, 1, 1, 1), (1023, 1025, 1, 2), (1024, 1024, 1, 1), (1025, 2049, 2, 3)],
)
def test_checkpoint_writer_grid_boundaries(
    n, hkv, d, ssm_blocks, conv_blocks, monkeypatch, k3_modules
):
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs["NBLK_SSM"]))

            return launch

    monkeypatch.setattr(k3_modules.writer, "_copy_checkpoints_kernel", Kernel())
    monkeypatch.setattr(
        k3_modules.writer.triton,
        "cdiv",
        lambda *args: pytest.fail("checkpoint launch must use host integer arithmetic"),
    )
    indices = torch.zeros(n, dtype=torch.int32)
    state = torch.empty(4, 1, 1, hkv)
    k3_modules.writer.write_state_checkpoints(
        state,
        state,
        torch.empty(3, d),
        torch.empty(4, d, 3),
        indices,
        indices,
        indices,
        indices,
        indices,
        indices,
        indices,
        chunk_size=1,
        conv_state_len=3,
    )
    assert launches == ([((n, ssm_blocks + conv_blocks), ssm_blocks)] if n else [])


def test_the_forward_writes_heads_ends_and_conv_windows(cpu_checkpoint_writer):
    mixed_qkv = torch.arange(8.0)[:, None].repeat(1, 3)[:, :2]  # padded rows
    conv_state = torch.zeros(6, 2, 4)  # [slots, D, kernel - 1 + num_spec]
    ssm_state = torch.zeros(6, 1, 1, 1)
    ssm_state[0] = 9.0

    index = lambda values: torch.tensor(values, dtype=torch.int32)
    cpu_checkpoint_writer(
        torch.tensor([[[[7.0]]]]),
        ssm_state,
        mixed_qkv,
        conv_state,
        index([0, 1]),
        index([4, 5]),
        index([4, 4]),
        index([0, 1]),
        index([0, 0]),
        index([0, 4, 8]),
        index([0, 4, 8]),
        1,
        3,
        h_rows=index([0, 0]),
    )

    assert ssm_state[4].item() == 7.0 and ssm_state[5].item() == 9.0
    assert conv_state[4, :, :3].tolist() == [[1.0, 2.0, 3.0]] * 2
    assert conv_state[5, :, :3].tolist() == [[5.0, 6.0, 7.0]] * 2
    assert not conv_state[:, :, 3].any()


@pytest.mark.parametrize("kind", ["none", "end", "mixed"])
def test_public_forward_reaches_the_prefill_and_checkpoint_writer(
    kind, monkeypatch, k3_modules, cpu_checkpoint_writer
):
    model = k3_modules.model
    layer = object.__new__(model.KimiKDAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name, layer.layer_num = "layer_0", 0
    layer.local_proj_size, layer.num_local_heads, layer.head_dim = 2, 1, 2
    layer.fuse_f_a_quant, layer.activation, layer.conv_weight = False, "silu", None
    fused = torch.arange(9 * 13, dtype=torch.float32).reshape(9, 13).bfloat16()[:, :11]
    layer.in_proj = lambda x, **kw: fused
    layer.f_b_proj = lambda x: x.new_zeros((len(x), 2))
    layer.o_norm = lambda x, gate, **kw: x
    layer.o_proj = lambda x: x
    pool = torch.full((8, 1, 2, 2), -9.0)
    pool[2], pool[3] = 20, 30
    conv_storage = torch.full((8, 5, 6), -5.0, dtype=torch.bfloat16)
    conv = conv_storage.transpose(-1, -2)
    expected_pool, expected_conv = pool.clone(), conv.clone()
    expected_pool[0], expected_pool[1] = 120, 131
    cu = torch.tensor([0, 4, 9], dtype=torch.int32)
    ckpt = None
    if kind != "none":
        values = (
            ([1, 0], [6, 7], [3, 4], [0, 1], [1, 0], [0, 0])
            if kind == "mixed"
            else ([0], [7], [4], [1], [0], [0])
        )
        ckpt = {
            key: torch.tensor(value, dtype=torch.int32)
            for key, value in zip(
                ("rows", "slots", "offs", "is_end", "runtime", "h_rows"), values
            )
        }
        expected_pool[7] = 120
        expected_conv[7, :, :3] = fused[1:4, :6].T
        if kind == "mixed":
            ckpt.update(
                kda_tokens=torch.tensor([*range(9), 4, 5, 6]),
                kda_rows=torch.tensor([0, 1, 1], dtype=torch.int32),
                kda_cu=torch.tensor([0, 4, 9, 12], dtype=torch.int32),
            )
            expected_pool[6] = 132
            expected_conv[6, :, :3] = fused[4:7, :6].T
    metadata = SimpleNamespace(
        num_actual_tokens=9,
        num_prefills=2,
        num_decodes=0,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([0, 1]),
        non_spec_state_indices_in_tensor=torch.tensor([2, 3]),
        non_spec_query_start_loc=cu,
        has_initial_state=torch.tensor([True, True]),
        ssm_checkpoints=ckpt,
        ssm_conv_state_len=3,
    )
    context = SimpleNamespace(
        tr_graph=None,
        in_hipgraph=False,
        attn_metadata=SimpleNamespace(kda_metadata=metadata),
        kv_cache_data={"layer_0": SimpleNamespace(k_cache=conv_storage, v_cache=pool)},
    )
    monkeypatch.setattr(model, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        model,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={"layer_0": layer}
            )
        ),
    )
    # Only the CUDA dispatcher is replaced; its production Python target still runs.
    monkeypatch.setattr(
        torch.ops.aiter, "kda_attention_with_output", model.kda_attention_with_output
    )
    monkeypatch.setattr(
        model, "causal_conv1d_fn", lambda x, *a, **kw: x.T.split(2, dim=-1)
    )
    monkeypatch.setattr(
        k3_modules.initial,
        "gather_kda_initial_state",
        lambda state, slots, has: state.index_select(0, slots),
    )
    calls = []

    def run(q, k, v, gate, beta, initial, offsets, final):
        calls.append((q.clone(), initial.clone(), offsets.clone()))
        assert final
        assert len(initial) + 1 == offsets.numel()
        assert q.shape[1] == int(offsets[-1])
        states = initial + torch.arange(len(initial)).reshape(-1, 1, 1, 1) + 100
        return q.contiguous(), states

    layer._run_kda = run
    if kind != "mixed":
        monkeypatch.setattr(
            layer, "_prefill_kda", lambda *a: pytest.fail("unneeded prefill wrapper")
        )
    output = layer.forward(torch.zeros(11, 2, dtype=torch.bfloat16))
    assert len(calls) == 1
    q, initial, offsets = calls[0]
    selected = [*range(9), 4, 5, 6] if kind == "mixed" else list(range(9))
    assert torch.equal(q.reshape(-1, 2), fused[selected, :2])
    assert initial[:, 0, 0, 0].tolist() == (
        [20, 30, 30] if kind == "mixed" else [20, 30]
    )
    assert offsets.tolist() == ([0, 4, 9, 12] if kind == "mixed" else [0, 4, 9])
    torch.testing.assert_close(output, fused[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashKDA needs a GPU")
def test_a_head_ends_on_the_state_a_forward_ending_there_would(k3_modules):
    """Row 1's checkpoint off FlashKDA's 32-token chunk grid, behind a row
    resuming from a state, the way a mixed step lays them out."""
    attn = k3_modules.model.KimiKDAAttention
    torch.manual_seed(0)
    heads, dim, lens, at = 4, 128, [300, 1100], 1001
    t = sum(lens)
    dev = "cuda"
    stub = SimpleNamespace(
        A_log=torch.randn(heads, device=dev),
        dt_bias=torch.randn(heads * dim, device=dev),
        _kda_gate_lower_bound=-5.0,
    )
    stub._run_kda = attn._run_kda.__get__(stub)

    def bf16(*shape):
        return torch.randn(1, t, *shape, device=dev, dtype=torch.bfloat16)

    q, k, v, g = (bf16(heads, dim) for _ in range(4))
    beta = bf16(heads)
    initial = torch.randn(2, heads, dim, dim, device=dev) * 0.1
    initial[1] = 0
    cu = torch.tensor([0, 300, t], device=dev, dtype=torch.int32)
    builder = SimpleNamespace(
        device=dev,
        _midstep_targets=lambda batch, grid: [(1, 7, at, 0, 3), (1, 8, 1100, 1, 3)],
    )
    batch = SimpleNamespace(state_save_all=[[True]], num_scheduled_tokens=lens)
    ckpt = k3_modules.builder._KimiMLAGDNCommon._checkpoint_targets(builder, batch)

    out, last, head = attn._prefill_kda(stub, q, k, v, g, beta, initial, ckpt)

    want_out, want_last = stub._run_kda(q, k, v, g, beta, initial, cu, True)
    span = slice(300, 300 + at)
    _, want_head = stub._run_kda(
        *(x[:, span] for x in (q, k, v, g, beta)),
        initial[1:],
        torch.tensor([0, at], device=dev, dtype=torch.int32),
        True,
    )
    torch.testing.assert_close(out, want_out, rtol=0, atol=0)
    torch.testing.assert_close(last, want_last, rtol=0, atol=0)
    torch.testing.assert_close(head, want_head, rtol=1e-5, atol=1e-5)
