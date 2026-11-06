# SPDX-License-Identifier: MIT
"""ATOM_MTP_FUSED_DRAFT_GLUE: `EagleProposer.propose` with its eager glue
between the draft passes fused (`atom/spec_decode/eagle_fused_glue.py`).

The CPU half checks who takes the fused path. The GPU half drives the real
`propose` -- both paths -- on a GDN builder and real `DraftGraph` recordings,
with a cheap stand-in draft model whose output depends on every input and
metadata buffer a pass reads, and requires the two to agree bit for bit: the
draft ids, `context_lens`, `kv_indptr`, `kv_indices`, the write slots,
`cu_seqlens_q`, the staged inputs of every pass, the anchor scatter into the
target's ids, and the host fields of `attn_metadata`.
"""

import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

ep = pytest.importorskip(
    "atom.spec_decode.eagle_proposer",
    reason="the drafter imports aiter at module load",
    exc_type=ImportError,
)

from atom.spec_decode.draft_graph import DraftGraph, StagedInput


def _gate_proposer(**over):
    builder = NS(
        supports_fused_mtp_glue=True,
        fuse_mtp_decode_position_update=False,
        dcp_world_size=1,
    )
    runner = NS(
        attn_metadata_builder=builder,
        attn_family=NS(is_mla=False),
        forward_vars={"kv_indices": None},
    )
    p = NS(
        speculative_config=NS(method="mtp"),
        step=object(),
        mtp_k=3,
        _captures_aux=False,
        _share_mtp_indices=False,
        runner=runner,
        model=None,
    )
    for k, v in over.items():
        setattr(p, k, v)
    p._fused_glue_deployment_applies = (
        lambda: ep.EagleProposer._fused_glue_deployment_applies(p)
    )
    p._fused_glue_supported = lambda: ep.EagleProposer._fused_glue_supported(p)
    return p


@pytest.fixture
def no_pcp(monkeypatch):
    monkeypatch.setattr(ep, "_pcp_active_for_draft_model", lambda model: False)


def test_the_flag_off_keeps_the_default_path(monkeypatch, no_pcp):
    monkeypatch.delenv("ATOM_MTP_FUSED_DRAFT_GLUE", raising=False)
    p = _gate_proposer()
    assert not ep.EagleProposer._fused_glue_applies(p, NS(scheduled_bs=4))


@pytest.mark.parametrize(
    "why,over",
    [
        ("eagle3", {"speculative_config": NS(method="eagle3")}),
        ("no mid-step pass", {"step": None}),
        ("one draft token", {"mtp_k": 1}),
        ("aux capture", {"_captures_aux": True}),
        ("shared MTP indices", {"_share_mtp_indices": True}),
    ],
)
def test_only_the_gdn_mtp_draft_takes_the_fused_path(monkeypatch, no_pcp, why, over):
    monkeypatch.setenv("ATOM_MTP_FUSED_DRAFT_GLUE", "1")
    assert ep.EagleProposer._fused_glue_applies(_gate_proposer(), NS(scheduled_bs=4))
    assert not ep.EagleProposer._fused_glue_applies(
        _gate_proposer(**over), NS(scheduled_bs=4)
    ), why


def test_other_builders_keep_the_default_path(monkeypatch, no_pcp):
    monkeypatch.setenv("ATOM_MTP_FUSED_DRAFT_GLUE", "1")
    for builder in (
        NS(fuse_mtp_decode_position_update=False),  # no capability
        NS(supports_fused_mtp_glue=True, fuse_mtp_decode_position_update=True),
        NS(supports_fused_mtp_glue=True, dcp_world_size=2),
    ):
        p = _gate_proposer()
        p.runner.attn_metadata_builder = builder
        assert not ep.EagleProposer._fused_glue_applies(p, NS(scheduled_bs=4))
    p = _gate_proposer()
    p.runner.attn_family = NS(is_mla=True)
    assert not ep.EagleProposer._fused_glue_applies(p, NS(scheduled_bs=4))
    p = _gate_proposer()
    p.runner.draft_kv_builder = object()
    assert not ep.EagleProposer._fused_glue_applies(p, NS(scheduled_bs=4))


def test_a_step_with_no_real_sequence_keeps_the_default_path(monkeypatch, no_pcp):
    monkeypatch.setenv("ATOM_MTP_FUSED_DRAFT_GLUE", "1")
    assert not ep.EagleProposer._fused_glue_applies(
        _gate_proposer(), NS(scheduled_bs=0)
    )


# ---------------------------------------------------------------------------
# GPU: the real propose, both paths.
# ---------------------------------------------------------------------------

H = 2048
BLOCK = 64
MAX_BS = 160
MAX_TOKENS = 4096 + 256
MAX_LEN = 65536
COLS = MAX_LEN // BLOCK + 8
MTP_K = 3
Q = MTP_K + 1
VOCAB = 248320


class _Buf:
    def __init__(self, t):
        self.gpu = t


class _FakeDraft(nn.Module):
    """Every input a real pass reads moves the output: ids, positions, hidden
    rows, and the slot / context-length / query-prefix buffers."""

    draft_step0_graph = True

    def __init__(self, var):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(7)
        self.emb = (torch.randn(4096, H, device="cuda", generator=g) * 0.05).to(
            torch.bfloat16
        )
        self.var = var

    def forward(self, input_ids, positions, hidden_states):
        from atom.utils.forward_context import get_forward_context

        n = input_ids.shape[0]
        pos = positions[0] if positions.ndim == 2 else positions
        x = hidden_states * 0.5 + F.embedding(input_ids.long() % 4096, self.emb)
        meta = (pos[:n] % 17) + (self.var["slot_mapping"].gpu[:n] % 11)
        ctx = self.var["context_lens"].gpu
        fc = get_forward_context()
        if fc.context.is_prefill:
            meta = meta + 1
        elif fc.attn_metadata.max_seqlen_q == 1:
            meta = meta + (ctx[:n] % 13) + (self.var["cu_seqlens_q"].gpu[1 : n + 1] % 5)
            split = getattr(fc.attn_metadata, "gfx12_token_split_tiles", None)
            if split is not None:  # ATOM_MTP_ADVANCE_TOKEN_SPLIT
                meta = meta + split[0] % 7
        else:
            meta = meta + (ctx[torch.arange(n, device="cuda") // Q] % 7)
        return x + meta.to(torch.bfloat16)[:, None]

    def compute_draft_ids(self, hidden_states, spec_step_idx=0, *, out):
        v = (hidden_states.float().sum(-1) * 64).to(torch.int64).abs() % VOCAB
        out.copy_(v.to(out.dtype))
        return out


class _Env:
    """A GDN builder over `forward_vars`, an `EagleProposer` with both passes
    declared and recorded, and a forward context the way `propose` finds it."""

    def __init__(self, sizes, token_split=False):
        import atom.utils.forward_context as fctx
        from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder

        self.fctx = fctx
        self.stream = torch.cuda.Stream()
        i32 = {"dtype": torch.int32, "device": "cuda"}
        i64 = {"dtype": torch.int64, "device": "cuda"}
        self.var = var = {
            "slot_mapping": _Buf(torch.zeros(MAX_TOKENS, **i64)),
            "context_lens": _Buf(torch.zeros(MAX_BS, **i32)),
            "block_tables": _Buf(torch.zeros(MAX_BS, COLS, **i32)),
            "cu_seqlens_q": _Buf(torch.zeros(MAX_BS + 1, **i32)),
            "kv_indptr": _Buf(torch.zeros(MAX_BS + 1, **i32)),
            "kv_indices": _Buf(torch.zeros(MAX_BS * (MAX_LEN // BLOCK), **i32)),
            "batch_info": _Buf(torch.zeros(3, **i32)),
        }
        runner = NS(
            forward_vars=var,
            attn_family=NS(is_mla=False),
            use_mrope=True,
            running_tokens_per_dp_rank=np.zeros(4, dtype=np.int32),
            block_size=BLOCK,
        )
        b = object.__new__(GDNAttentionMetadataBuilder)
        b.model_runner = runner
        b.block_size = BLOCK
        b.block_ratio = 1
        b.num_attention_heads = 16
        b.row_ids = torch.arange(MAX_BS + 1, **i32)
        runner.attn_metadata_builder = b
        p = object.__new__(ep.EagleProposer)
        p.config = NS(
            parallel_config=NS(data_parallel_size=4, data_parallel_rank=0),
            max_num_seqs=MAX_BS,
        )
        p.speculative_config = NS(method="mtp")
        p.runner = runner
        p.device = torch.device("cuda")
        p.dtype = torch.bfloat16
        p.mtp_k = MTP_K
        p.model = _FakeDraft(var)
        p._captures_aux = False
        p._aux_buffers = []
        p._share_mtp_indices = False
        p._reuse_step_buffers = False
        p.step = DraftGraph(
            forward=p._step_forward,
            epilogue=p._step_head,
            capture_epilogue=True,
            inputs={
                "input_ids": StagedInput(dtype=torch.int32),
                "positions": StagedInput(dtype=torch.int64),
                "hidden_states": StagedInput(shape=(H,), dtype=torch.bfloat16),
            },
        ).bind(p.config, "cuda")
        p.step0 = DraftGraph(
            forward=p._step0_forward,
            epilogue=p._step0_head,
            capture_epilogue=True,
            inputs={
                "input_ids": StagedInput(shape=(Q,), dtype=torch.int32),
                "positions": StagedInput(shape=(Q,), dtype=torch.int64),
                "hidden_states": StagedInput(shape=(Q, H), dtype=torch.bfloat16),
                "last_token_indices": StagedInput(dtype=torch.int64),
            },
        ).bind(p.config, "cuda")
        if token_split:
            from atom.spec_decode.eagle_fused_glue import DraftTokenSplit

            # aiter's defaults (AITER_GFX12_TOKEN_SPLIT_PROGRAMS / MIN / MAX).
            p._token_split = DraftTokenSplit(
                torch.zeros(1, **i32), BLOCK, 1, 2048, 32, 256
            )
        self.p, self.builder = p, b
        with torch.cuda.stream(self.stream):
            for bs in sizes:
                var["context_lens"].gpu[:bs] = 1000
                self._context(bs, bs, False, 1)[1].is_draft = True
                p.step.warmup(bs)
                self._context(bs, bs * Q, False, Q)[1].is_draft = True
                p.step0.warmup(bs)
        torch.cuda.synchronize()

    def _context(self, bs, tokens, is_prefill, max_q, mode=None, dummy=False, sbs=None):
        from atom.utils.forward_context import AttentionMetaData, Context

        var = self.var
        md = AttentionMetaData(
            cu_seqlens_q=var["cu_seqlens_q"].gpu[: bs + 1],
            max_seqlen_q=max_q,
            max_seqlen_k=MAX_LEN,
            slot_mapping=var["slot_mapping"].gpu[:tokens],
            context_lens=var["context_lens"].gpu[:bs],
            block_tables=var["block_tables"].gpu[:bs],
            kv_indptr=var["kv_indptr"].gpu[: bs + 1],
            kv_indices=var["kv_indices"].gpu,
        )
        ctx = Context(
            positions=None,
            is_prefill=is_prefill,
            is_dummy_run=dummy,
            scheduled_bs=bs if sbs is None else sbs,
            scheduled_tokens=tokens,
            running_bs=bs,
            running_tokens=tokens,
            forward_mode=mode,
        )
        self.fctx.set_forward_context(
            attn_metadata=md,
            atom_config=NS(
                parallel_config=self.p.config.parallel_config,
                compilation_config=NS(static_forward_context={}),
            ),
            context=ctx,
            running_tokens_per_dp_rank=np.full(4, tokens, dtype=np.int32),
        )
        return md, ctx

    def batch(self, seed, sbs, rbs, prefill_tokens=None):
        g = torch.Generator(device="cuda").manual_seed(seed)

        def rnd(lo, hi, n):
            return torch.randint(
                lo, hi, (n,), generator=g, device="cuda", dtype=torch.int32
            )

        if prefill_tokens is None:
            n, running_tokens = sbs * Q, rbs * Q
            ctx = rnd(2000, 30000, sbs)
            reject = rnd(0, MTP_K + 1, sbs)
            anchors = (
                torch.arange(sbs, device="cuda", dtype=torch.int32) * Q + MTP_K - reject
            ).to(torch.int32)
            pos = (ctx[:, None] - Q + torch.arange(Q, device="cuda")).reshape(-1)
        else:
            n = running_tokens = prefill_tokens
            cuts = torch.sort(
                torch.randperm(n - 1, generator=g, device="cuda")[: sbs - 1] + 1
            ).values
            starts = torch.cat([cuts.new_zeros(1), cuts])
            ends = torch.cat([cuts, cuts.new_full((1,), n)])
            lens = (ends - starts).to(torch.int32)
            prefix = rnd(0, 20000, sbs)
            ctx = prefix + lens
            reject = torch.zeros(sbs, dtype=torch.int32, device="cuda")
            anchors = (ends - 1).to(torch.int32)
            seg = torch.repeat_interleave(torch.arange(sbs, device="cuda"), lens.long())
            pos = prefix.long()[seg] + torch.arange(n, device="cuda") - starts[seg]
        positions = torch.zeros(3, running_tokens, dtype=torch.int64, device="cuda")
        positions[:, :n] = pos
        context_lens = torch.zeros(MAX_BS, dtype=torch.int32, device="cuda")
        context_lens[:sbs] = ctx
        context_lens[sbs:rbs] = rnd(0, 3, rbs - sbs)
        tables = torch.zeros(MAX_BS, COLS, dtype=torch.int32, device="cuda")
        tables[:rbs, : MAX_LEN // BLOCK] = torch.randint(
            1,
            2_000_000,
            (rbs, MAX_LEN // BLOCK),
            generator=g,
            device="cuda",
            dtype=torch.int32,
        )
        return {
            "n": n,
            "sbs": sbs,
            "rbs": rbs,
            "prefill": prefill_tokens is not None,
            "input_ids": rnd(0, VOCAB, MAX_TOKENS),
            "positions": positions,
            "hidden": (
                torch.randn(running_tokens, H, generator=g, device="cuda") * 2
            ).to(torch.bfloat16),
            "next_ids": rnd(0, VOCAB, sbs),
            "reject": reject,
            "anchors": anchors,
            "context_lens": context_lens,
            "block_tables": tables,
            "slot_mapping": torch.randint(
                0, 1 << 40, (MAX_TOKENS,), generator=g, device="cuda"
            ),
            "kv_indptr": rnd(0, 1000, MAX_BS + 1),
            "cu_seqlens_q": rnd(0, 1000, MAX_BS + 1),
            "max_seqlen_k": int(ctx.max()),
        }

    def propose(self, st, fused, dummy=False):
        os.environ["ATOM_MTP_FUSED_DRAFT_GLUE"] = "1" if fused else "0"
        self.p._fused_glue_deployment = None
        var = self.var
        with torch.cuda.stream(self.stream):
            for name in (
                "context_lens",
                "block_tables",
                "slot_mapping",
                "kv_indptr",
                "cu_seqlens_q",
            ):
                var[name].gpu.copy_(st[name])
            var["kv_indices"].gpu.zero_()
            var["batch_info"].gpu.copy_(torch.tensor([st["sbs"], st["n"], 0]))
            ids = st["input_ids"].clone()
            sbs, rbs, n = st["sbs"], st["rbs"], st["n"]
            if st["prefill"]:
                md, ctx = self._context(rbs, n, True, 4096, dummy=dummy, sbs=sbs)
            else:
                mode = NS(
                    use_cudagraph=True,
                    is_prefill=False,
                    max_seqlen_q=Q,
                    running_tokens_are_unified=True,
                    running_tokens=rbs * Q,
                    token_graph=False,
                )
                md, ctx = self._context(
                    rbs, rbs * Q, False, Q, mode=mode, dummy=dummy, sbs=sbs
                )
            md.max_seqlen_k = st["max_seqlen_k"]
            out = self.p.propose(
                target_token_ids=ids[1 : n + 1],
                target_positions=st["positions"],
                target_hidden_states=st["hidden"],
                num_reject_tokens=st["reject"],
                next_token_ids=st["next_ids"],
                last_token_indices=st["anchors"],
            )
            indptr = var["kv_indptr"].gpu[: rbs + 1].clone()
            split = getattr(self.p, "_token_split", None)
            snap = {
                "draft_ids": out.clone(),
                "context_lens": var["context_lens"].gpu.clone(),
                "kv_indptr": indptr,
                "kv_indices": var["kv_indices"].gpu.clone(),
                "slot_mapping": var["slot_mapping"].gpu.clone(),
                "cu_seqlens_q": var["cu_seqlens_q"].gpu.clone(),
                "target_input_ids": ids,
                "step_inputs": [
                    self.p.step.buffer(r, rbs).clone()
                    for r in ("input_ids", "positions", "hidden_states")
                ],
                "host": (
                    md.max_seqlen_q,
                    md.max_seqlen_k,
                    md.slot_mapping.data_ptr(),
                    md.slot_mapping.shape[0],
                    md.context_lens.shape[0],
                    md.cu_seqlens_q.shape[0],
                    ctx.is_prefill,
                ),
            }
            if split:
                snap["token_split"] = split.tiles.clone()
            if not st["prefill"]:
                snap["step0_inputs"] = [
                    self.p.step0.buffer(r, rbs).clone()
                    for r in (
                        "input_ids",
                        "positions",
                        "hidden_states",
                        "last_token_indices",
                    )
                ]
        torch.cuda.synchronize()
        return snap


def _assert_same(a, b):
    for key in a:
        x, y = a[key], b[key]
        xs = x if isinstance(x, list) else [x]
        ys = y if isinstance(y, list) else [y]
        for u, v in zip(xs, ys):
            if isinstance(u, torch.Tensor):
                assert u.dtype == v.dtype and u.shape == v.shape, key
                assert torch.equal(
                    u.view(torch.int16) if u.dtype == torch.bfloat16 else u,
                    v.view(torch.int16) if v.dtype == torch.bfloat16 else v,
                ), key
            else:
                assert u == v, key


@pytest.fixture(scope="module")
def env():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    import atom.utils.forward_context as fctx

    old_pcp = ep._pcp_active_for_draft_model
    old_gather = fctx.DPMetadata.gather_running_tokens
    ep._pcp_active_for_draft_model = lambda model: False
    # An eager step 0 gathers the DP row counts over a CPU all_reduce.
    fctx.DPMetadata.gather_running_tokens = staticmethod(
        lambda parallel_config, running_tokens: np.full(
            4, running_tokens, dtype=np.int32
        )
    )
    old_flag = os.environ.get("ATOM_MTP_FUSED_DRAFT_GLUE")
    try:
        yield _Env([1, 8, 16, 128, 160])
    finally:
        ep._pcp_active_for_draft_model = old_pcp
        fctx.DPMetadata.gather_running_tokens = old_gather
        if old_flag is None:
            os.environ.pop("ATOM_MTP_FUSED_DRAFT_GLUE", None)
        else:
            os.environ["ATOM_MTP_FUSED_DRAFT_GLUE"] = old_flag


@pytest.mark.parametrize("dummy", [False, True])
@pytest.mark.parametrize(
    "sbs,rbs", [(1, 1), (1, 8), (6, 8), (16, 16), (103, 128), (150, 160)]
)
def test_a_decode_step_drafts_and_leaves_the_same_state(env, sbs, rbs, dummy):
    st = env.batch(sbs * 7 + rbs, sbs, rbs)
    _assert_same(env.propose(st, False, dummy), env.propose(st, True, dummy))


@pytest.mark.parametrize(
    "tokens,sbs,rbs", [(700, 3, 8), (4100, 9, 16), (4096, 120, 128)]
)
def test_a_prefill_step_drafts_and_leaves_the_same_state(env, tokens, sbs, rbs):
    st = env.batch(tokens + sbs, sbs, rbs, prefill_tokens=tokens)
    _assert_same(env.propose(st, False), env.propose(st, True))


def test_prefill_then_decode_on_the_same_buffers(env):
    """The prefill -> decode switch: the steps share every buffer, so a fused
    step that left anything different would show in the next one."""
    steps = [
        (16, 16, 4000),
        (6, 8, None),
        (103, 128, None),
        (9, 16, 2500),
        (1, 1, None),
    ]
    for fused_first in (False, True):
        a, b = [], []
        for k, (sbs, rbs, n) in enumerate(steps):
            st = env.batch(1000 + k, sbs, rbs, prefill_tokens=n)
            a.append(env.propose(st, fused_first))
            b.append(env.propose(st, not fused_first))
        for x, y in zip(a, b):
            _assert_same(x, y)


def test_a_ragged_verify_batch_stages_step_0_as_before(env):
    """A sequence short of mtp_k + 1 rows: step 0 stages through `_stage_step0`
    and the later passes still take the fused advance."""
    st = env.batch(4242, 6, 8)
    st["n"] = 6 * Q - 1  # the last sequence verifies one draft fewer
    st["anchors"][-1] = min(int(st["anchors"][-1]), st["n"] - 1)
    _assert_same(env.propose(st, False), env.propose(st, True))


def test_warmup_compiles_every_variant_serving_launches(env):
    from atom.spec_decode import eagle_fused_glue as glue

    def variants(fn):
        return sum(len(entry[0]) for entry in fn.device_caches.values())

    with torch.cuda.stream(env.stream):
        glue.warm_fused_glue_kernels(
            env.builder,
            hidden_size=H,
            hidden_dtype=torch.bfloat16,
            ids_dtype=torch.int32,
            query_width=Q,
            max_bs=MAX_BS,
        )
    torch.cuda.synchronize()
    before = variants(glue._stage_verify_step0_kernel), variants(
        glue._advance_draft_pass_kernel
    )
    for sbs, rbs, n in [(5, 8, None), (13, 16, None), (127, 128, None), (3, 8, 701)]:
        env.propose(env.batch(sbs + rbs, sbs, rbs, prefill_tokens=n), True)
    after = variants(glue._stage_verify_step0_kernel), variants(
        glue._advance_draft_pass_kernel
    )
    assert after == before


@pytest.fixture(scope="module")
def split_env(env):
    """`env`'s setup, with ATOM_MTP_ADVANCE_TOKEN_SPLIT on and its buffer in place
    before the recordings, as `warmup_draft_graphs` makes it."""
    old = os.environ.get("ATOM_MTP_ADVANCE_TOKEN_SPLIT")
    os.environ["ATOM_MTP_ADVANCE_TOKEN_SPLIT"] = "1"
    try:
        yield _Env([1, 8, 16, 128], token_split=True)
    finally:
        if old is None:
            os.environ.pop("ATOM_MTP_ADVANCE_TOKEN_SPLIT", None)
        else:
            os.environ["ATOM_MTP_ADVANCE_TOKEN_SPLIT"] = old


@pytest.mark.parametrize("sbs,rbs", [(1, 1), (6, 8), (16, 16), (103, 128)])
def test_the_advance_writes_the_token_split_unified_attention_would(
    split_env, sbs, rbs
):
    """ATOM_MTP_ADVANCE_TOKEN_SPLIT: each pass reads the tiles per segment (the
    stand-in draft folds them into its output) the fused advance wrote, and they
    are what the eager advance writes and what aiter computes from seqused_k."""
    from aiter.ops.triton.attention.unified_attention import (
        _gfx12_token_split_tiles_torch,
    )

    st = split_env.batch(sbs * 11 + rbs, sbs, rbs)
    eager, fused = split_env.propose(st, False), split_env.propose(st, True)
    _assert_same(eager, fused)
    ref = _gfx12_token_split_tiles_torch(fused["context_lens"][:rbs], 1, BLOCK)
    assert torch.equal(fused["token_split"], ref)


def test_the_token_split_matches_aiter_on_any_lengths():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from aiter.ops.triton.attention.unified_attention import (
        _gfx12_token_split_tiles_torch,
    )

    from atom.spec_decode.eagle_fused_glue import DraftTokenSplit

    g = torch.Generator(device="cuda").manual_seed(3)
    for rows, hi, tile, heads in [
        (1, 10, 64, 1),
        (16, 300000, 128, 1),
        (160, 70000, 128, 2),
        (64, 5, 16, 4),
    ]:
        lens = torch.randint(
            0, hi, (rows,), generator=g, device="cuda", dtype=torch.int32
        )
        split = DraftTokenSplit(
            torch.zeros(1, dtype=torch.int32, device="cuda"), tile, heads, 2048, 32, 256
        )
        split.write(lens)
        assert torch.equal(
            split.tiles, _gfx12_token_split_tiles_torch(lens, heads, tile)
        )


@pytest.mark.parametrize("sbs,rbs", [(6, 8), (103, 128)])
def test_skipping_kv_indices_changes_nothing_else(env, sbs, rbs, monkeypatch):
    """ATOM_MTP_SKIP_DRAFT_KV_INDICES: the fused advance leaves kv_indices
    unwritten and everything else as the eager path does."""
    st = env.batch(sbs * 13 + rbs, sbs, rbs)
    eager = env.propose(st, False)
    monkeypatch.setenv("ATOM_MTP_SKIP_DRAFT_KV_INDICES", "1")
    env.p._gen_kv_indices = False  # draft_reads_kv_indices: page 64, unified
    try:
        fused = env.propose(st, True)
    finally:
        del env.p._gen_kv_indices
    assert not fused["kv_indices"].any()
    eager.pop("kv_indices"), fused.pop("kv_indices")
    _assert_same(eager, fused)
