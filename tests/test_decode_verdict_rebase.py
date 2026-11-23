# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""An MTP verify laid out at the scheduler's lengths and moved on the device
to the previous verdict's lands where the host would have laid it out after
waiting for that verdict.

The GDN builder (Qwen3-Next, Qwen3.5) consumes the verdict on the device
(`consumes_device_verdict`), so the runner builds the step from zero rejected
counts for the requests carried over and takes the status after the launch.
These tests run the real `stage_decode` / `finish_decode` /
`prepare_num_accepted_tokens` twice on CPU buffers -- once from the host's
counts, as before, once from zeros plus the device rebase -- and hold every
published buffer the verdict moves to agree. The rebase runs as its torch
reference here; `test_kernels_match_their_references` holds the Triton kernels
to it on a GPU.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.model_ops.attentions.token_layout.batch_ids import build_cu_seqlens
from atom.utils import CpuGpuBuffer

# The builder modules, loaded by `modules` for this file's tests.
M = SimpleNamespace()


@pytest.fixture(scope="module", autouse=True)
def modules():
    """The builders against AITER / Triton stand-ins where they cannot run
    (a CPU host), the real ones on a GPU host."""
    with atom_without_aiter() as load:
        for name, path in (
            ("aiter_attention", "atom.model_ops.attentions.aiter_attention"),
            ("gdn_attn", "atom.model_ops.attentions.gdn_attn"),
            ("qwen4_exp_attn", "atom.model_ops.attentions.qwen4_exp_attn"),
            ("decode_verdict", "atom.model_ops.decode_verdict"),
        ):
            setattr(M, name, load(path))
        yield M


MAX_BS, MAX_TOKENS, COLS = 16, 128, 128


def _scenario(lengths, *, kv_block, seed=0):
    """A decode batch whose requests interleave ones carried over from the
    previous verify (in another order, beside one it no longer holds) with
    ones admitted since, whose counts the host already has."""
    rng = np.random.default_rng(seed)
    bs = len(lengths)
    src = np.array([2, -1, 0, 4, -1, 1][:bs], np.int32)
    num_prev = 5
    # [bonus, rejected] per previous request; k = 3 drafts a verify.
    rejected_prev = np.array([2, 0, 3, 1, 3], np.int32)
    bonus_prev = 3 - rejected_prev
    # What the scheduler had for the admitted ones.
    host_rejected = np.where(src < 0, rng.integers(0, 4, bs), 0).astype(np.int32)
    host_bonus = np.where(src < 0, 3 - host_rejected, 0).astype(np.int32)
    true_rejected = np.where(src >= 0, rejected_prev[src], host_rejected)
    true_bonus = np.where(src >= 0, bonus_prev[src], host_bonus)
    # Scheduler lengths, the upper bound; several sit just past a block edge
    # so a rejection moves the request's last token into the previous block.
    base = max(2 * kv_block, 16)
    ctx = np.array([base + 2, 40, base + 1, 29, 75, base + 3][:bs], np.int32)
    blocks = rng.permutation(1000)[: bs * COLS].reshape(bs, COLS).astype(np.int32)
    block_tables = [blocks[i, : -(-int(ctx[i]) // kv_block)].copy() for i in range(bs)]
    return SimpleNamespace(
        src=src,
        verdict=torch.tensor(np.stack([bonus_prev, rejected_prev]), dtype=torch.int32),
        num_prev=num_prev,
        host_rejected=host_rejected,
        host_bonus=host_bonus,
        true_rejected=true_rejected.astype(np.int32),
        true_bonus=true_bonus.astype(np.int32),
        batch=SimpleNamespace(
            total_seqs_num=bs,
            total_seqs_num_decode=bs,
            total_seqs_num_prefill=0,
            total_tokens_num_decode=int(sum(lengths)),
            is_dummy_run=False,
            context_lens=ctx,
            num_scheduled_tokens=np.array(lengths, np.int32),
            block_tables=block_tables,
            req_ids=[f"r{i}" for i in range(bs)],
            mrope_position_deltas={"r0": -7, "r3": 5},
        ),
    )


def _builder(cls, sc, *, rebase, kv_block, page, use_mrope, running_bs):
    """A builder over CPU buffers, at the counts the host path (``rebase``
    False: the verdict's) or the deferred path (zero for carried) reads."""

    def buf(*size, dtype):
        return CpuGpuBuffer(*size, dtype=dtype, device="cpu", pin_memory=False)

    var = {
        "cu_seqlens_q": buf(MAX_BS + 1, dtype=torch.int32),
        "slot_mapping": buf(MAX_TOKENS, dtype=torch.int64),
        "positions": buf(MAX_TOKENS, dtype=torch.int64),
        "context_lens": buf(MAX_BS, dtype=torch.int32),
        "block_tables": buf(MAX_BS, COLS, dtype=torch.int32),
        "kv_indptr": buf(MAX_BS + 1, dtype=torch.int32),
        "kv_indices": buf(MAX_BS * COLS, dtype=torch.int32),
        "mrope_positions": buf(3, MAX_TOKENS, dtype=torch.int64),
        "decode_src": buf(MAX_TOKENS, dtype=torch.int32),
        "max_qlen": 4,
    }
    batch = sc.batch
    build_cu_seqlens(batch.num_scheduled_tokens, running_bs, out=var["cu_seqlens_q"].np)
    var["decode_src"].np[: batch.total_seqs_num] = sc.src
    for name in ("cu_seqlens_q", "decode_src"):  # published with the token inputs
        var[name].copy_to_gpu()

    carried = sc.src >= 0
    rejected = np.where(carried, 0 if rebase else sc.true_rejected, sc.host_rejected)
    bonus = np.where(carried, 0 if rebase else sc.true_bonus, sc.host_bonus)
    processor = SimpleNamespace(
        is_deferred_out=True,
        prev_batch=SimpleNamespace(total_seqs_num=sc.num_prev),
        device_spec_counts=sc.verdict,
        carries_over=True,
        num_rejected=rejected.astype(np.int32),
        num_bonus=bonus.astype(np.int32),
        # `tokenIDProcessor.verify_ctx_lens`, deferred output.
        verify_ctx_lens=lambda c: c - rejected.astype(c.dtype),
    )
    runner = SimpleNamespace(
        forward_vars=var,
        block_size=kv_block,
        use_mrope=use_mrope,
        tokenID_processor=processor,
        h2d_groups=None,
        config=SimpleNamespace(
            enable_tbo_decode=False,
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
        _mrope_positions_view=lambda n: var["mrope_positions"].gpu.as_strided(
            (3, n), (n, 1)
        ),
    )
    builder = object.__new__(cls)
    builder.model_runner = runner
    builder.block_size = page
    builder.block_ratio = kv_block // page
    builder.token_axis_scratch = np.empty(MAX_TOKENS, np.int64)
    builder._has_sparse_attention = False
    builder.num_accepted_tokens = torch.ones(MAX_BS, dtype=torch.int32)
    builder.refresh_flydsl_plan = lambda *a, **k: None
    return builder


class _Group:
    """The decode metadata's publication: whatever was counted goes up."""

    def __init__(self, var):
        self.var, self.counts, self.hooks = var, {}, []

    def set_counts(self, counts):
        for name, count in counts.items():
            if count is not None:
                self.counts[name] = count

    def set_count(self, buffer, count):
        name = next(k for k, v in self.var.items() if v is buffer)
        if count is not None:
            self.counts[name] = count

    def after_publish(self, hook):
        self.hooks.append(hook)

    def publish(self):
        for name, count in self.counts.items():
            if name == "mrope_positions":  # counted in elements
                buffer = self.var[name]
                flat = buffer.gpu.view(-1)
                flat[:count] = torch.from_numpy(buffer.np.reshape(-1)[:count])
            else:
                self.var[name].copy_to_gpu(count)
        for hook in self.hooks:
            hook()


@pytest.fixture
def references(monkeypatch):
    """The device rebase as its torch twin; no kv_indices kernel on CPU."""
    monkeypatch.setattr(
        M.aiter_attention,
        "rebase_decode_on_verdict",
        M.decode_verdict.rebase_decode_on_verdict_reference,
        raising=False,
    )
    monkeypatch.setattr(
        M.gdn_attn,
        "accepted_from_verdict",
        M.decode_verdict.accepted_from_verdict_reference,
        raising=False,
    )
    monkeypatch.setattr(
        M.aiter_attention, "kv_indices_generate_triton", lambda *a, **k: None
    )


def _step(sc, *, rebase, kv_block, page, use_mrope, running_bs, running_tokens):
    """One verify's metadata on the GDN builder, as `prepare_model` builds it:
    the builder asked, laid out, published, finished, then GDN's counts."""
    builder = _builder(
        M.gdn_attn.GDNAttentionMetadataBuilder,
        sc,
        rebase=rebase,
        kv_block=kv_block,
        page=page,
        use_mrope=use_mrope,
        running_bs=running_bs,
    )
    if rebase:
        # The runner asks before the step's input ids; True defers the wait.
        builder.consumes_device_verdict(sc.batch, 4)
    var = builder.model_runner.forward_vars
    group = _Group(var)
    cls = M.aiter_attention.AiterAttentionMetadataBuilder
    attn_metadata, positions = cls.stage_decode(
        builder, sc.batch, running_bs, running_tokens, 4, group
    )
    group.publish()
    cls.finish_decode(builder, sc.batch, attn_metadata, running_bs, running_tokens, 4)
    builder.prepare_num_accepted_tokens(sc.batch)
    return builder, group.counts, positions


@pytest.mark.parametrize("use_mrope", [False, True])
@pytest.mark.parametrize(
    "lengths, kv_block, page",
    [
        ([4, 4, 4, 4, 4, 4], 16, 16),  # uniform MTP-3
        ([4, 1, 3, 4, 2], 16, 16),  # ragged, as DSpark's buckets
        ([4, 4, 4, 4], 64, 16),  # kv_indptr pages finer than a block entry
        ([4, 4, 4], 1, 1),
    ],
)
def test_rebased_step_lands_where_the_host_lays_it_out(
    references, lengths, kv_block, page, use_mrope
):
    sc = _scenario(lengths, kv_block=kv_block)
    bs = len(lengths)
    running_bs, running_tokens = bs + 3, sum(lengths) + 5  # a graph bucket's pad
    host, host_counts, host_pos = _step(
        sc,
        rebase=False,
        kv_block=kv_block,
        page=page,
        use_mrope=use_mrope,
        running_bs=running_bs,
        running_tokens=running_tokens,
    )
    dev, dev_counts, dev_pos = _step(
        sc,
        rebase=True,
        kv_block=kv_block,
        page=page,
        use_mrope=use_mrope,
        running_bs=running_bs,
        running_tokens=running_tokens,
    )
    # The same buffers go up, at the same counts.
    assert dev_counts == host_counts
    hv, dv = host.model_runner.forward_vars, dev.model_runner.forward_vars
    # The deferred path did lay the carried requests out longer.
    carried = sc.src >= 0
    moved = carried & (sc.true_rejected > 0)
    assert moved.any()
    assert (dv["context_lens"].np[:bs][moved] > hv["context_lens"].np[:bs][moved]).all()
    for name, count in host_counts.items():
        if name == "mrope_positions":
            got = dv[name].gpu.view(-1)[:count]
            want = hv[name].gpu.view(-1)[:count]
        else:
            got, want = dv[name].gpu[:count], hv[name].gpu[:count]
        assert torch.equal(got, want), name
    # What the model reads its positions from, and GDN its accepted counts.
    assert torch.equal(dev_pos, host_pos)
    assert dev.num_accepted_tokens.tolist() == host.num_accepted_tokens.tolist()
    assert dev.num_accepted_tokens[:bs].tolist() == (sc.true_bonus + 1).tolist()


def _decision(cls=None, **overrides):
    sc = _scenario([4, 4, 4], kv_block=16)
    builder = _builder(
        cls or M.gdn_attn.GDNAttentionMetadataBuilder,
        sc,
        rebase=True,
        kv_block=16,
        page=16,
        use_mrope=False,
        running_bs=3,
    )
    batch, processor = sc.batch, builder.model_runner.tokenID_processor
    q = overrides.pop("max_seqlen_q", 4)
    for name, value in overrides.items():
        target = processor if hasattr(processor, name) else batch
        if name in ("_has_sparse_attention",):
            target = builder
        elif name == "enable_tbo_decode":
            target = builder.model_runner.config
        setattr(target, name, value)
    return builder.consumes_device_verdict(batch, q), builder


def test_the_gdn_builder_consumes_the_verdict_on_an_mtp_verify():
    consumes, builder = _decision()
    assert consumes and getattr(builder, "_rebases_this_step", False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_seqlen_q": 1},  # nothing drafted, nothing to move
        {"total_seqs_num_prefill": 2},
        {"is_dummy_run": True},
        {"is_deferred_out": False},  # PP: the host has its counts already
        {"prev_batch": None},
        {"device_spec_counts": None},  # e.g. an overridden step's
        {"_has_sparse_attention": True},
        {"enable_tbo_decode": True},
    ],
)
def test_other_steps_wait_for_the_verdict_on_the_host(overrides):
    consumes, builder = _decision(**overrides)
    assert not consumes and not getattr(builder, "_rebases_this_step", False)


@pytest.mark.parametrize(
    "module, cls",
    [
        ("aiter_attention", "AiterAttentionMetadataBuilder"),
        ("qwen4_exp_attn", "Qwen4ExpMetadataBuilder"),
    ],
)
def test_builders_that_did_not_opt_in_keep_the_host_path(module, cls):
    consumes, builder = _decision(getattr(getattr(M, module), cls))
    assert not consumes and not getattr(builder, "_rebases_this_step", False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("use_mrope", [False, True])
@pytest.mark.parametrize("kv_block, page", [(16, 16), (64, 16), (1, 1)])
def test_kernels_match_their_references(use_mrope, kv_block, page):
    rng = np.random.default_rng(1)
    for trial in range(20):
        # Past the kernel's row block too, so its prefix sum carries over.
        bs = int(rng.integers(1, 40 if trial < 14 else 300))
        running_bs = bs + int(rng.integers(0, 9))
        lengths = rng.integers(1, 5, bs) if trial % 2 else np.full(bs, 4)
        cu = np.zeros(running_bs + 1, np.int32)
        build_cu_seqlens(lengths.astype(np.int32), running_bs, out=cu)
        width = int(cu[bs]) + 7  # a graph bucket's pad tokens past them
        num_prev = bs + 3
        rejected = rng.integers(0, 4, num_prev).astype(np.int32)
        ctx = np.zeros(running_bs, np.int32)
        ctx[:bs] = rng.integers(8, 900, bs)
        case = {
            "rejected": torch.tensor(rejected),
            "bonus": torch.tensor(3 - rejected),
            "src": torch.tensor(
                np.where(rng.random(bs) < 0.7, rng.permutation(num_prev)[:bs], -1),
                dtype=torch.int32,
            ),
            "cu": torch.tensor(cu),
            "ctx": torch.tensor(ctx),
            "indptr": torch.full((running_bs + 4,), -5, dtype=torch.int32),
            "slots": torch.full((width,), -9, dtype=torch.int64),
            "table": torch.tensor(
                rng.integers(0, 1 << 20, (running_bs, 1024 // kv_block + 2)),
                dtype=torch.int32,
            ),
            "pos": torch.full((width,), -3, dtype=torch.int64),
            "mrope": torch.arange(3 * width, dtype=torch.int64).view(3, width),
            "accepted": torch.full((bs + 2,), 1, dtype=torch.int32),
        }
        out = []
        for device, rebase, fill in (
            (
                "cuda",
                M.decode_verdict.rebase_decode_on_verdict,
                M.decode_verdict.accepted_from_verdict,
            ),
            (
                "cpu",
                M.decode_verdict.rebase_decode_on_verdict_reference,
                M.decode_verdict.accepted_from_verdict_reference,
            ),
        ):
            t = {k: v.to(device) for k, v in case.items()}
            rebase(
                t["rejected"],
                t["src"],
                t["cu"],
                t["ctx"],
                t["indptr"],
                t["slots"],
                t["table"],
                running_bs=running_bs,
                max_seqlen_q=4,
                kv_block_size=kv_block,
                page_size=page,
                positions=None if use_mrope else t["pos"],
                mrope_positions=t["mrope"] if use_mrope else None,
            )
            fill(t["accepted"], t["src"], t["bonus"])
            out.append({k: v.cpu() for k, v in t.items()})
        gpu, ref = out
        for name in gpu:
            assert torch.equal(gpu[name], ref[name]), (trial, name)
