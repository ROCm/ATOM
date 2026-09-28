# SPDX-License-Identifier: MIT
"""DCP support code: merge-path ops, the QREP config gate, and the QREP row view.

Ops from ``atom/model_ops/dcp_ops.py`` that the sparse work leans on:

  * ``correct_attn_out``        -- the LSE merge kernel behind ``cp_lse_ag_out_rs``
  * ``get_dcp_local_seq_lens``  -- how many tokens of a sequence this rank holds
  * ``reorg_kvcache``           -- AllGathered chunk blocks -> per-seq contiguous

Plus the two pieces DCP query replication (QREP) rests on:

  * ``DCPConfig`` / ``qrep_unsupported_reason`` -- parsing and feasibility gating
  * ``ColumnParallelLinear.make_row_view``      -- the narrow q_proj for prefill

And the gathered query-head width tables from ``atom/model_ops/attention_mla.py``
-- same category as the QREP gate above: pure config-time functions whose wrong
answer is a silently miscomputing kernel, not a crash. See the section comment
further down for which widths are unsafe where.

The merge is the load-bearing one. ``cp_lse_ag_out_rs`` reconstructs a global
softmax from per-rank partial attentions, so the test here is not "the kernel
runs" but "summing the corrected per-rank outputs reproduces plain dense
attention over the union" -- the premise the whole DCP design rests on.

The empty-rank case gets its own test because it is the one this branch changed:
under sparse prefill a rank routinely owns no candidate for a row, and aiter then
returns ``o=NaN`` with ``lse=-inf``. Without the ``factor == 0 -> 0`` scrub in
the kernel, ``NaN * 0 = NaN`` survives the ReduceScatter and poisons EVERY
rank's output for that row -- silently, with no fault.

``get_dcp_local_seq_lens`` and ``reorg_kvcache`` run on CPU tensors; the merge
and row-view tests need a GPU. The config tests need neither, which is why the
`dcp_ops` import below is guarded per-test rather than module-wide: a
module-level skip would take the config tests down with it on the CPU CI
runner, and those are the only ones that gate actually runs.
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# atom.config and atom.distributed.dcp_utils both import cleanly without
# triton/aiter, so the tests below run on the CPU gate.
from atom.config import (
    DCPConfig,
    q_proj_has_row_sliceable_scale,
    q_proj_is_qrep_widened,
    qrep_enabled_for_layer,
    qrep_unsupported_reason,
)
from atom.distributed.dcp_utils import (
    mla_dcp_decode_is_persistent,
    mla_dcp_sparse_prefill_is_persistent,
)

try:
    import triton
    from aiter import dtypes

    from atom.config import QuantizationConfig
    from atom.model_ops.attention_mla import (
        _MLA_DCP_KERNEL_WIDTHS,
        _MLA_DCP_KERNEL_WIDTHS_NON_PERSISTENT,
        _MLA_DCP_KERNEL_WIDTHS_NON_PERSISTENT_FP8,
        _MLA_DCP_SPARSE_PREFILL_WIDTHS,
        _MLA_DCP_SPARSE_PREFILL_WIDTHS_PERSISTENT,
        mla_dcp_kernel_num_heads,
        mla_dcp_sparse_prefill_num_heads,
    )
    from atom.model_ops.dcp_ops import (
        _dcp_a2a_pack_kernel,
        _dcp_a2a_unpack_combine_kernel,
        _lse_pack_slots,
        correct_attn_out,
        dcp_global_pos,
        dcp_local_index,
        dcp_owner_rank,
        get_dcp_local_seq_lens,
        reorg_kvcache,
    )
    from atom.model_ops.linear import ColumnParallelLinear
    from atom.quant_spec import LayerQuantConfig, QuantType

    _DCP_OPS_ERR = None
except ImportError as _e:  # triton absent on a CPU-only runner
    _DCP_OPS_ERR = str(_e)

needs_dcp_ops = pytest.mark.skipif(
    _DCP_OPS_ERR is not None, reason=f"requires full atom import env: {_DCP_OPS_ERR}"
)
needs_gpu = pytest.mark.skipif(
    _DCP_OPS_ERR is not None or not torch.cuda.is_available(),
    reason="Triton kernel needs a GPU",
)

DEV = "cuda"
NEG_INF = -float("inf")


# ─────────────────────────────────────────────────────────── correct_attn_out ──


def _dense_and_shards(B, H, L, D, N, dtype, seed=0):
    """One attention problem, plus its N disjoint round-robin shards.

    Mirrors DCP: global position p lives on rank p % N. Returns the dense
    reference (o, lse) and the per-rank partial (o_r, lse_r), all in fp32.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, H, D, generator=g, device=DEV, dtype=torch.float32)
    k = torch.randn(B, H, L, D, generator=g, device=DEV, dtype=torch.float32)
    v = torch.randn(B, H, L, D, generator=g, device=DEV, dtype=torch.float32)
    scale = D**-0.5

    logits = torch.einsum("bhd,bhld->bhl", q, k) * scale
    dense_o = torch.einsum("bhl,bhld->bhd", torch.softmax(logits, dim=-1), v)
    dense_lse = torch.logsumexp(logits, dim=-1)

    outs, lses = [], []
    for r in range(N):
        part = logits[:, :, r::N]
        outs.append(
            torch.einsum(
                "bhl,bhld->bhd", torch.softmax(part, dim=-1), v[:, :, r::N]
            ).to(dtype)
        )
        lses.append(torch.logsumexp(part, dim=-1))
    return dense_o, dense_lse, outs, torch.stack(lses)


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N", [2, 4, 8])
def test_merge_reproduces_dense_attention(dtype, N):
    """Sum of the corrected per-rank outputs == dense attention over the union."""
    B, H, L, D = 6, 4, 128, 64
    dense_o, dense_lse, outs, lses = _dense_and_shards(B, H, L, D, N, dtype)

    merged = torch.zeros(B, H, D, device=DEV, dtype=torch.float32)
    for r in range(N):
        # correct_attn_out writes in place; hand it a private copy per rank, the
        # way each rank owns its own buffer.
        corrected, glse = correct_attn_out(outs[r].clone(), lses, r)
        merged += corrected.float()  # the ReduceScatter(sum) in cp_lse_ag_out_rs
        # every rank must agree on the global LSE, that is what makes the
        # per-rank correction factors sum to one
        torch.testing.assert_close(glse, dense_lse, rtol=1e-5, atol=1e-5)

    tol = 1e-5 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(merged, dense_o, rtol=tol, atol=tol)


@needs_gpu
def test_empty_rank_contributes_zero_not_nan():
    """lse=-inf + o=NaN (a rank owning no candidate) must correct to 0.

    This is the ``tl.where(factor == 0.0, 0.0, output)`` line. Drop it and the
    assertion below fails with NaN everywhere, on every rank, for that row.
    """
    B, H, D, N = 3, 4, 32, 4
    # Only ranks 0 and 1 own anything; 2 and 3 are empty for every row.
    lses = torch.stack(
        [
            torch.randn(B, H, device=DEV),
            torch.randn(B, H, device=DEV),
            torch.full((B, H), NEG_INF, device=DEV),
            torch.full((B, H), NEG_INF, device=DEV),
        ]
    )
    outs = [
        torch.randn(B, H, D, device=DEV),
        torch.randn(B, H, D, device=DEV),
        torch.full((B, H, D), float("nan"), device=DEV),
        torch.full((B, H, D), float("nan"), device=DEV),
    ]

    merged = torch.zeros(B, H, D, device=DEV)
    for r in range(N):
        corrected, _ = correct_attn_out(outs[r].clone(), lses, r)
        if r >= 2:
            assert torch.all(corrected == 0), "empty rank must contribute exactly 0"
        merged += corrected

    assert torch.isfinite(merged).all(), "NaN from an empty rank reached the sum"

    # And the surviving two ranks still merge to the right answer: dropping the
    # empty ranks entirely must give the same result.
    ref = torch.zeros(B, H, D, device=DEV)
    for r in range(2):
        corrected, _ = correct_attn_out(outs[r].clone(), lses[:2], r)
        ref += corrected
    torch.testing.assert_close(merged, ref, rtol=1e-5, atol=1e-5)


@needs_gpu
def test_all_ranks_empty_stays_finite():
    """Every rank empty for a row: global lse is -inf and the output is 0."""
    B, H, D, N = 2, 2, 16, 4
    lses = torch.full((N, B, H), NEG_INF, device=DEV)
    out = torch.full((B, H, D), float("nan"), device=DEV)
    corrected, glse = correct_attn_out(out, lses, 0)
    assert torch.all(corrected == 0)
    assert torch.all(torch.isneginf(glse))


@needs_gpu
def test_nan_and_posinf_in_gathered_lse_are_sanitized():
    """A rank reporting NaN/+inf must be treated as if it reported -inf.

    aiter allocates its lse buffer with torch.empty and has been caught leaving
    it unwritten on some kernel paths, so a garbage value from a peer is a real
    possibility; the kernel folds it to -inf rather than letting it swallow the
    whole softmax.
    """
    B, H, D, N = 4, 4, 32, 4
    g = torch.Generator(device=DEV).manual_seed(3)
    lses = torch.randn(N, B, H, generator=g, device=DEV)
    out = torch.randn(B, H, D, generator=g, device=DEV)

    clean = lses.clone()
    clean[2] = NEG_INF
    dirty = lses.clone()
    dirty[2, :, ::2] = float("nan")
    dirty[2, :, 1::2] = float("inf")

    got_o, got_lse = correct_attn_out(out.clone(), dirty, 0)
    exp_o, exp_lse = correct_attn_out(out.clone(), clean, 0)
    torch.testing.assert_close(got_o, exp_o, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_lse, exp_lse, rtol=1e-6, atol=1e-6)


@needs_gpu
def test_non_contiguous_lses_view():
    """The kernel writes the global LSE with `lses`' own B/H strides.

    Hence the empty_strided allocation: a contiguous output tensor would be
    written with the wrong offsets as soon as `lses` is a view.
    """
    B, H, D, N = 5, 4, 32, 4
    g = torch.Generator(device=DEV).manual_seed(4)
    big = torch.randn(N, B, 2 * H, generator=g, device=DEV)
    view = big[:, :, :H]
    assert not view.is_contiguous()

    out = torch.randn(B, H, D, generator=g, device=DEV)
    got_o, got_lse = correct_attn_out(out.clone(), view, 1)
    exp_o, exp_lse = correct_attn_out(out.clone(), view.contiguous(), 1)
    torch.testing.assert_close(got_o, exp_o, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_lse, exp_lse, rtol=1e-6, atol=1e-6)


@needs_gpu
def test_non_power_of_two_world_size_is_rejected():
    """N is baked in as N_ROUNDED for tl.arange; fail loudly, not cryptically."""
    out = torch.randn(2, 2, 16, device=DEV)
    lses = torch.randn(3, 2, 2, device=DEV)
    with pytest.raises(AssertionError, match="power of two"):
        correct_attn_out(out, lses, 0)


# ─────────────────────────────────────────────────────── get_dcp_local_seq_lens ──


def _brute_local_len(seq_len, dcp_size, dcp_rank, interleave):
    """Definition, straight from the storage rule: token i lives on rank
    (i // cp_kv_cache_interleave_size) % dcp_size."""
    return sum(1 for i in range(seq_len) if (i // interleave) % dcp_size == dcp_rank)


@needs_dcp_ops
@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 4])
def test_local_seq_lens_match_the_storage_rule(dcp_size, interleave):
    lens = np.arange(0, 201, dtype=np.int64)
    per_rank = [
        get_dcp_local_seq_lens(lens, dcp_size, r, interleave) for r in range(dcp_size)
    ]
    for r in range(dcp_size):
        expect = np.array(
            [_brute_local_len(int(L), dcp_size, r, interleave) for L in lens]
        )
        np.testing.assert_array_equal(
            per_rank[r], expect, err_msg=f"rank {r}, interleave {interleave}"
        )
    # No token is dropped or double-counted -- a shard-length bug here desyncs
    # the KV writes from the reads with no error anywhere.
    np.testing.assert_array_equal(sum(per_rank), lens)


# ──────────────────────────────────── dcp_owner_rank / dcp_local_index (Part 1) ──
# Block-level interleave (cp_kv_cache_interleave_size > 1) enabler: these two
# helpers centralize the owner + local-index math that was inlined as `% W` /
# `// W` all over the DCP paths. Every write/read site will call them, so a bug
# here silently desyncs KV writes from reads. The tests pin them to the storage
# rule (token i -> rank (i//S)%W, local index (i//(S*W))*S + i%S) and cross-check
# against get_dcp_local_seq_lens and the vLLM slot formula.


def _brute_local_index(i, dcp_size, dcp_rank, interleave):
    """Local index of global token i on its owning rank, by counting: how many
    earlier tokens (j < i) also land on the same rank."""
    assert (i // interleave) % dcp_size == dcp_rank
    return sum(1 for j in range(i) if (j // interleave) % dcp_size == dcp_rank)


@needs_dcp_ops
@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 4, 8])
def test_dcp_owner_and_local_index_match_storage_rule(dcp_size, interleave):
    pos = np.arange(0, 500, dtype=np.int64)
    owners = dcp_owner_rank(pos, dcp_size, interleave)
    local = dcp_local_index(pos, dcp_size, interleave)
    for i in range(len(pos)):
        r = (i // interleave) % dcp_size
        assert int(owners[i]) == r, f"owner i={i} S={interleave} W={dcp_size}"
        assert int(local[i]) == _brute_local_index(
            i, dcp_size, r, interleave
        ), f"local_index i={i} S={interleave} W={dcp_size}"


@needs_dcp_ops
@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 4, 8])
def test_dcp_global_pos_inverts_local_index(dcp_size, interleave):
    # dcp_global_pos(local_index(g), owner(g)) must round-trip to g. The sparse
    # candidate exchange rebuilds global ids this way, and the tie-break needs
    # them to be a correct total order over global positions.
    pos = np.arange(0, 500, dtype=np.int64)
    for g in pos:
        r = int(dcp_owner_rank(g, dcp_size, interleave))
        j = int(dcp_local_index(g, dcp_size, interleave))
        assert int(dcp_global_pos(j, r, dcp_size, interleave)) == int(
            g
        ), f"g={g} S={interleave} W={dcp_size} r={r} j={j}"
    # And S=1 reduces to the round-robin j*W + r.
    j = pos
    for r in range(dcp_size):
        np.testing.assert_array_equal(
            dcp_global_pos(j, r, dcp_size, 1), j * dcp_size + r
        )


@needs_dcp_ops
@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
def test_dcp_helpers_reduce_to_round_robin_when_interleave_1(dcp_size):
    # S == 1 must be bit-identical to the old inline round-robin (owner = i%W,
    # local index = i//W) -- this is the S=1 regression guarantee.
    pos = np.arange(0, 300, dtype=np.int64)
    np.testing.assert_array_equal(dcp_owner_rank(pos, dcp_size, 1), pos % dcp_size)
    np.testing.assert_array_equal(dcp_local_index(pos, dcp_size, 1), pos // dcp_size)


@needs_dcp_ops
@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 4, 8])
def test_dcp_local_index_max_equals_local_seq_len(dcp_size, interleave):
    # The largest local index a rank produces for a seq of length L, plus 1, must
    # equal that rank's get_dcp_local_seq_lens(L) -- the two must agree or writes
    # overflow / underflow the reserved per-rank KV.
    for L in [0, 1, 7, 63, 64, 65, 130, 257, 500]:
        pos = np.arange(0, L, dtype=np.int64)
        owners = dcp_owner_rank(pos, dcp_size, interleave)
        local = dcp_local_index(pos, dcp_size, interleave)
        for r in range(dcp_size):
            owned = local[owners == r]
            expect = int(
                get_dcp_local_seq_lens(np.array([L]), dcp_size, r, interleave)[0]
            )
            got = int(owned.max()) + 1 if owned.size else 0
            assert got == expect, f"L={L} r={r} S={interleave} W={dcp_size}"


@needs_dcp_ops
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 4, 8])
@pytest.mark.parametrize("block_size", [8, 16, 64])
def test_dcp_slot_matches_vllm_reference(dcp_size, interleave, block_size):
    # Cross-check the (block_table_index, slot_offset) our helpers imply against
    # vLLM's merged slot kernel (block_table.py:413-439), the authoritative
    # block-level layout. Requires block_size % S == 0 (the config constraint).
    if block_size % interleave != 0:
        pytest.skip("block_size must be a multiple of cp_kv_cache_interleave_size")
    vbs = block_size * dcp_size
    for i in range(4 * vbs + 3):
        r = (i // interleave) % dcp_size
        # ours
        loc = dcp_local_index(i, dcp_size, interleave)
        our_blk = i // vbs
        our_off = loc % block_size
        assert loc // block_size == our_blk  # block_size % S == 0 keeps these aligned
        # vLLM reference on the virtual-block offset
        vb_off = i % vbs
        assert (vb_off // interleave) % dcp_size == r
        ref_loc = (vb_off // (dcp_size * interleave)) * interleave + (
            vb_off % interleave
        )
        ref_blk = i // vbs + ref_loc // block_size
        ref_off = ref_loc % block_size
        assert (our_blk, our_off) == (
            ref_blk,
            ref_off,
        ), f"i={i} S={interleave} W={dcp_size} bs={block_size}"


# ─────────────────────────────────────────────────────────────── reorg_kvcache ──

POISON = -777.0  # padding slot content: must never reach the output


def _build_chunk(cached_lens, dcp, block_size, chunk_size, chunk_idx, dim=4, pe=2):
    """Recreate one AllGathered chunk exactly as _build_mla_chunk_meta_dcp does.

    Each row carries its GLOBAL token position as its value, so the check can be
    written against the DCP storage rule instead of against reorg's own index
    arithmetic. Padding slots carry POISON.
    """
    vbs = block_size * dcp
    bs = len(cached_lens)
    local_lens = np.stack(
        [get_dcp_local_seq_lens(np.asarray(cached_lens), dcp, r) for r in range(dcp)],
        axis=1,
    )  # [bs, dcp]
    padded_local = -(-np.asarray(cached_lens) // vbs) * block_size  # ceil * block

    c_lo = chunk_idx * chunk_size
    c_hi = c_lo + chunk_size
    plc = np.clip(np.minimum(padded_local, c_hi) - c_lo, 0, None)  # [bs]
    real = np.clip(np.minimum(local_lens, c_hi) - c_lo, 0, None)  # [bs, dcp]
    toks = int(plc.sum())

    kv_c = torch.full((dcp * toks, 1, dim), POISON)
    k_pe = torch.full((dcp * toks, 1, pe), POISON)
    for r in range(dcp):
        off = 0
        for s in range(bs):
            for j in range(int(plc[s])):
                if j < int(real[s, r]):  # real token, else a padded slot
                    tag = float((c_lo + j) * dcp + r)  # global position
                    kv_c[r * toks + off + j] = tag
                    k_pe[r * toks + off + j] = -tag
            off += int(plc[s])

    return {
        "kv_c": kv_c,
        "k_pe": k_pe,
        "plc": plc.astype(int).tolist(),
        "local_lens": local_lens.astype(int).tolist(),
        "real": real,
        "toks": toks,
        "sum_seq_len": int(real.sum()),
        "max_seq_len": int(real.sum(axis=1).max(initial=0)),
        "c_lo": c_lo,
        "c_hi": c_hi,
    }


@needs_dcp_ops
@pytest.mark.parametrize("chunk_idx", [0, 1, 2, 3])
def test_reorg_kvcache_rebuilds_each_sequence(chunk_idx):
    dcp, block_size, chunk_size = 4, 4, 4
    # 37: not block-aligned, ranks end up with unequal local lengths (10/9/9/9)
    #  3: shorter than dcp, so rank 3 owns nothing at all
    # 64: exactly block-aligned, spans every chunk
    #  0: no cached context
    cached_lens = [37, 3, 64, 0]
    c = _build_chunk(cached_lens, dcp, block_size, chunk_size, chunk_idx)

    kv_c, k_pe = reorg_kvcache(
        c["kv_c"],
        c["k_pe"],
        padded_local_chunk_seq_lens_lst=c["plc"],
        local_context_lens_allranks=c["local_lens"],
        sum_seq_len=c["sum_seq_len"],
        max_seq_len=c["max_seq_len"],
        chunk_size=chunk_size,
        chunk_idx=chunk_idx,
        toks=c["toks"],
    )

    assert kv_c.shape[0] == c["sum_seq_len"]
    assert k_pe.shape[0] == c["sum_seq_len"]
    got = kv_c[:, 0, 0]
    assert torch.all(got != POISON), "a padding slot survived into the output"
    torch.testing.assert_close(k_pe[:, 0, 0], -got)
    assert torch.all(kv_c == got.view(-1, 1, 1)), "rows are not internally uniform"

    # Per-seq contents, stated in terms of the DCP storage rule rather than of
    # reorg's slicing: sequence s must receive exactly the global positions it
    # has cached whose LOCAL index falls in this chunk's window.
    pos = 0
    for s, glen in enumerate(cached_lens):
        n = int(c["real"][s].sum())
        seg = got[pos : pos + n].tolist()
        pos += n
        want = {float(p) for p in range(glen) if c["c_lo"] <= p // dcp < c["c_hi"]}
        assert set(seg) == want, f"seq {s}: wrong token set in chunk {chunk_idx}"
        assert len(seg) == len(want), f"seq {s}: duplicated tokens"

        # Layout contract: rank-major, ascending within a rank. The context
        # attention is unmasked so the order does not change the result, but a
        # scrambled order would mean the segment walk is off.
        by_rank = [[t for t in seg if int(t) % dcp == r] for r in range(dcp)]
        assert seg == [t for grp in by_rank for t in grp], "not rank-major"
        for grp in by_rank:
            assert grp == sorted(grp), "not ascending within a rank"
    assert pos == c["sum_seq_len"]


@needs_dcp_ops
def test_reorg_kvcache_rejects_a_wrong_total():
    """The internal asserts are the only guard the caller has; keep them live."""
    dcp, block_size, chunk_size = 4, 4, 4
    c = _build_chunk([37, 3, 64, 0], dcp, block_size, chunk_size, 0)
    with pytest.raises(AssertionError):
        reorg_kvcache(
            c["kv_c"],
            c["k_pe"],
            padded_local_chunk_seq_lens_lst=c["plc"],
            local_context_lens_allranks=c["local_lens"],
            sum_seq_len=c["sum_seq_len"] + 1,  # wrong on purpose
            max_seq_len=c["max_seq_len"],
            chunk_size=chunk_size,
            chunk_idx=0,
            toks=c["toks"],
        )


# ══════════════════════════════════════════ DCPConfig + the QREP gate (CPU) ══
#
# DCPConfig parsing and the DCP query-replication (QREP) gate.
# QREP now defaults to ON, which changes what a missing test costs: a refactor
# that quietly disables it produces no error and no visible symptom -- just a
# slower server. That already happened: a merge re-parented the gate under
# `if dcp_config.interleave_size > 1`, so with the default interleave_size=1
# QREP was force-disabled in every ordinary configuration while the log blamed
# `dcp <= 1`. It went unnoticed through a full day of benchmarking. Hence the
# two guards below: the gate must key off the DCP size and NOT interleave_size,
# and the shipped default must stay on.
#
# These need neither a GPU nor dcp_ops, so they are the DCP tests the CPU gate
# actually runs.


# ──────────────────────────────────────────────────── project-before-merge ──


def _per_head_project(o, w):
    """The V up-projection: ``[B, H, L] x [H, L, V] -> [B, H, V]``, per head."""
    return torch.einsum("bhl,hlv->bhv", o, w)


def _merge_partials(o_per_rank, lses):
    """``cp_lse_ag_out_rs`` without the collectives: correct each rank, then sum.

    ``correct_attn_out`` applies rank r's weight in place, and the ReduceScatter
    that follows it in the real path is a plain sum across ranks -- the head-dim
    scatter only decides who keeps which slice, it does not change values. So
    summing here reproduces the merged result.
    """
    n = lses.shape[0]
    acc = None
    for r in range(n):
        out_r, _ = correct_attn_out(o_per_rank[r].clone(), lses, r, ctx=None)
        acc = out_r.clone() if acc is None else acc + out_r
    return acc


@needs_gpu
@pytest.mark.parametrize("n_ranks", [2, 8])
def test_projection_commutes_with_the_merge(n_ranks):
    """The premise project-before-merge rests on.

    The merge is a per-(token, head) SCALAR weighting followed by a sum across
    ranks; the V up-projection is per-head LINEAR. A scalar commutes with a
    linear map, and a linear map distributes over the sum, so projecting each
    rank's partial and then merging must equal merging first and projecting the
    result. That identity is what lets the merge exchange ``v_head_dim`` per
    head instead of ``kv_lora_rank``.
    """
    b, h, latent, v_dim = 3, 8, 32, 16
    g = torch.Generator(device=DEV).manual_seed(n_ranks)
    o = torch.randn(n_ranks, b, h, latent, generator=g, device=DEV)
    lses = torch.randn(n_ranks, b, h, generator=g, device=DEV)
    w_v = torch.randn(h, latent, v_dim, generator=g, device=DEV)

    merge_then_project = _per_head_project(_merge_partials(o, lses), w_v)
    project_then_merge = _merge_partials(
        torch.stack([_per_head_project(o[r], w_v) for r in range(n_ranks)]), lses
    )

    # Not bitwise: the two orders sum a different number of terms in a different
    # sequence. fp32 tolerances, since that is what the kernel accumulates in.
    torch.testing.assert_close(
        project_then_merge, merge_then_project, rtol=1e-5, atol=1e-5
    )


@needs_gpu
def test_projection_does_not_defeat_the_empty_rank_scrub():
    """A rank owning no KV for a row returns ``o=NaN`` with ``lse=-inf``.

    The merge kernel forces that contribution to zero (``factor == 0 -> 0``).
    Projecting first puts a matmul in front of the scrub, and ``NaN`` through a
    matmul is still ``NaN`` -- so the question is whether the scrub still catches
    it. If it does not, one empty rank poisons EVERY rank's output for that row,
    silently and with no fault raised.
    """
    b, h, latent, v_dim, n_ranks = 2, 4, 32, 16, 2
    g = torch.Generator(device=DEV).manual_seed(7)
    o = torch.randn(n_ranks, b, h, latent, generator=g, device=DEV)
    lses = torch.randn(n_ranks, b, h, generator=g, device=DEV)
    w_v = torch.randn(h, latent, v_dim, generator=g, device=DEV)

    # Rank 0 owns nothing for row (0, 0): aiter's signature for that case.
    o[0, 0, 0] = float("nan")
    lses[0, 0, 0] = NEG_INF

    projected = torch.stack([_per_head_project(o[r], w_v) for r in range(n_ranks)])
    got = _merge_partials(projected, lses)
    assert torch.isfinite(got).all(), "empty-rank NaN survived the projection"

    # With rank 0 contributing nothing, the row must equal rank 1 alone -- whose
    # weight is exp(lse_1 - logsumexp(lse_1)) == 1.
    torch.testing.assert_close(got[0, 0], projected[1, 0, 0], rtol=1e-5, atol=1e-5)


# ──────────────────────────────────────────────────────── A2A merge backend ──


def _a2a_roundtrip(o_per_rank, lse_per_rank, n_ranks, owned_counts_per_rank=None):
    """pack -> all-to-all -> combine, with the collective done as a local permute.

    ``all_to_all_single`` needs a real process group, which a single-process test
    does not have. But the collective is a pure permutation -- rank r's chunk n
    becomes rank n's chunk r -- so stacking the send buffers reproduces exactly
    what every rank would receive. That leaves the two Triton kernels and the LSE
    bit-packing as the only things under test, which is where the bugs would be.

    Returns ``[B, H_total, D]``: each rank owns heads ``[j*H_local, ...)``, so
    concatenating the per-rank outputs in rank order rebuilds the original head
    layout.
    """
    b, h_total, d = o_per_rank[0].shape
    h_local = h_total // n_ranks
    dtype = o_per_rank[0].dtype
    pack = _lse_pack_slots(dtype)

    sends = []
    for r in range(n_ranks):
        send = torch.empty((n_ranks, b, h_local, d + pack), dtype=dtype, device=DEV)
        o_r = o_per_rank[r].contiguous()
        l_r = lse_per_rank[r].contiguous().to(torch.float32)
        has_owned_counts = owned_counts_per_rank is not None
        counts_r = owned_counts_per_rank[r].contiguous() if has_owned_counts else l_r
        _dcp_a2a_pack_kernel[(b, h_total)](
            o_r,
            l_r,
            counts_r,
            send,
            o_r.stride(0),
            o_r.stride(1),
            l_r.stride(0),
            l_r.stride(1),
            send.stride(0),
            send.stride(1),
            send.stride(2),
            H_LOCAL=h_local,
            HEAD_DIM=d,
            LSE_PACK=pack,
            HAS_OWNED_COUNTS=has_owned_counts,
        )
        sends.append(send)

    outs = []
    for j in range(n_ranks):
        recv = torch.stack([sends[r][j] for r in range(n_ranks)]).contiguous()
        out = torch.empty((b, h_local, d), dtype=dtype, device=DEV)
        out_lse = torch.empty((b, h_local), dtype=torch.float32, device=DEV)
        _dcp_a2a_unpack_combine_kernel[(b, h_local)](
            recv,
            out,
            out_lse,
            recv.stride(0),
            recv.stride(1),
            recv.stride(2),
            out.stride(0),
            out.stride(1),
            out_lse.stride(0),
            out_lse.stride(1),
            n_ranks,
            HEAD_DIM=d,
            LSE_PACK=pack,
            N_ROUNDED=triton.next_power_of_2(n_ranks),
            WRITE_LSE=True,
        )
        outs.append(out)
    return torch.cat(outs, dim=1)


@needs_gpu
@pytest.mark.parametrize("n_ranks", [2, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a2a_reproduces_the_ag_rs_merge(n_ranks, dtype):
    """The two backends must agree: same weighted sum, same head ownership.

    They arrive there differently -- ag_rs pre-multiplies each rank's partial and
    lets ReduceScatter sum, a2a moves everything to the owning rank and does both
    locally -- so this is a comparison, not a bit-for-bit check.

    bf16 is the case that matters most: it is what decode actually runs, and it
    is the only one where the fp32 LSE has to survive being split across two
    16-bit slots of the transfer buffer.
    """
    b, h, d = 3, 8, 32
    g = torch.Generator(device=DEV).manual_seed(n_ranks)
    o = [
        torch.randn(b, h, d, generator=g, device=DEV).to(dtype) for _ in range(n_ranks)
    ]
    lse = [torch.randn(b, h, generator=g, device=DEV) for _ in range(n_ranks)]

    got = _a2a_roundtrip(o, lse, n_ranks)
    ref = _merge_partials(torch.stack(o), torch.stack(lse))

    tol = 1e-5 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(got.float(), ref.float(), rtol=tol, atol=tol)


@needs_gpu
def test_a2a_lse_survives_the_16_bit_split():
    """A bf16 buffer stores the fp32 LSE as two raw 16-bit halves.

    Nothing does arithmetic on those slots -- the collective copies bits -- so the
    value must come back EXACTLY, not approximately. If it did not, every rank's
    softmax weights would be subtly wrong in a way the tolerance above could hide.

    Checked by giving each rank a distinct LSE and one-hot outputs, so the merged
    result is a pure function of the weights.
    """
    b, h, d, n_ranks = 2, 4, 32, 4
    g = torch.Generator(device=DEV).manual_seed(11)
    lse = [torch.randn(b, h, generator=g, device=DEV) * 5.0 for _ in range(n_ranks)]
    # Rank r contributes the constant r, so out == sum_r weight_r * r exactly.
    o = [
        torch.full((b, h, d), float(r), device=DEV, dtype=torch.bfloat16)
        for r in range(n_ranks)
    ]

    got = _a2a_roundtrip(o, lse, n_ranks).float()

    stacked = torch.stack(lse)  # [N, B, H]
    w = torch.softmax(stacked, dim=0)  # exp(lse_r - global_lse)
    want = sum(w[r].unsqueeze(-1) * float(r) for r in range(n_ranks))
    torch.testing.assert_close(
        got, want.expand_as(got).contiguous(), rtol=8e-3, atol=8e-3
    )


@needs_gpu
def test_a2a_scrubs_the_empty_rank():
    """A dummy-backed empty rank has finite LSE and a meaningless output.

    The existing pack kernel must use the true count to send LSE=-inf, after
    which the combine kernel forces its contribution to a hard zero. This is
    the launch-free sparse-DCP zero-row path used by decode and prefill.
    """
    b, h, d, n_ranks = 2, 4, 32, 2
    g = torch.Generator(device=DEV).manual_seed(5)
    o = [
        torch.randn(b, h, d, generator=g, device=DEV).bfloat16() for _ in range(n_ranks)
    ]
    lse = [torch.randn(b, h, generator=g, device=DEV) for _ in range(n_ranks)]
    o[0][0] = float("nan")
    owned_counts = [
        torch.tensor([0, 1], dtype=torch.int32, device=DEV),
        torch.tensor([1, 1], dtype=torch.int32, device=DEV),
    ]

    got = _a2a_roundtrip(o, lse, n_ranks, owned_counts)
    assert torch.isfinite(got).all(), "empty-rank NaN reached the a2a output"
    # Rank 0 contributed nothing, so the row is rank 1 alone (weight 1).
    torch.testing.assert_close(got[0].float(), o[1][0].float(), rtol=8e-3, atol=8e-3)


@needs_dcp_ops
def test_lse_pack_slots_covers_the_transfer_dtypes():
    """fp32 needs one slot, 16-bit dtypes need two; anything else must raise
    rather than silently truncate the LSE."""
    assert _lse_pack_slots(torch.float32) == 1
    assert _lse_pack_slots(torch.bfloat16) == 2
    assert _lse_pack_slots(torch.float16) == 2
    with pytest.raises(NotImplementedError, match="a2a merge buffer dtype"):
        _lse_pack_slots(torch.float8_e4m3fn)


@needs_dcp_ops
def test_cp_lse_a2a_is_a_no_op_without_a_dcp_group():
    """dcp==1 has nothing to merge; the input must come back untouched."""
    from atom.model_ops.dcp_ops import cp_lse_a2a

    o = torch.randn(2, 4, 8)
    lse = torch.randn(2, 4)
    group = SimpleNamespace(world_size=1)
    assert cp_lse_a2a(o, lse, group) is o
    out, out_lse = cp_lse_a2a(o, lse, group, return_lse=True)
    assert out is o and out_lse is lse


# ─────────────────────────────────────────────────────────── DCPConfig parsing ──


def test_defaults():
    cfg = DCPConfig()
    assert cfg.interleave_size == 1
    # Deliberately pinned: both ship enabled. Flipping either default is a
    # product decision, so it should require editing a test that says so.
    # It also decides what "pass nothing" means for an A/B control arm, which
    # is the way a default flip silently turns a comparison into a no-op.
    assert cfg.enable_query_replication is True
    assert cfg.enable_project_before_merge is True
    # a2a: measured faster than ag_rs on the decode shapes here. Pinned because
    # it also decides what an A/B control arm gets when it passes nothing.
    assert cfg.comm_backend == "a2a"


def test_from_dict_parses_both_keys():
    cfg = DCPConfig.from_dict(
        {
            "interleave_size": 16,
            "enable_query_replication": False,
            "enable_project_before_merge": False,
            "comm_backend": "a2a",
        }
    )
    assert cfg.interleave_size == 16
    assert cfg.enable_query_replication is False
    assert cfg.enable_project_before_merge is False
    assert cfg.comm_backend == "a2a"


def test_from_dict_empty_and_none_give_defaults():
    for arg in (None, {}):
        cfg = DCPConfig.from_dict(arg)
        assert (
            cfg.interleave_size,
            cfg.enable_query_replication,
            cfg.enable_project_before_merge,
            cfg.comm_backend,
        ) == (1, True, True, "a2a")


def test_from_dict_coerces_types():
    """JSON is permissive; the dataclass should not be."""
    cfg = DCPConfig.from_dict(
        {
            "interleave_size": "8",
            "enable_query_replication": 0,
            "enable_project_before_merge": 0,
        }
    )
    assert cfg.interleave_size == 8 and isinstance(cfg.interleave_size, int)
    assert cfg.enable_query_replication is False
    assert cfg.enable_project_before_merge is False


def test_from_dict_rejects_unknown_key():
    """Typos must fail loudly -- a silently ignored key reads as 'it did not work'."""
    with pytest.raises(ValueError, match="Unknown --dcp-config key"):
        DCPConfig.from_dict({"interleve_size": 4})


def test_from_dict_rejects_the_old_flag_name():
    """`enable_dcp_query_replication` was the pre-DCPConfig top-level flag.

    Anyone carrying an old command line should get an error naming the supported
    keys, not a server that silently ignores the setting.
    """
    with pytest.raises(ValueError, match="enable_dcp_query_replication"):
        DCPConfig.from_dict({"enable_dcp_query_replication": True})


def test_comm_backend_rejects_unknown_value():
    """A typo here would silently keep the default backend, so it must raise."""
    with pytest.raises(AssertionError, match="comm_backend"):
        DCPConfig.from_dict({"comm_backend": "all2all"})


def test_interleave_size_must_be_positive():
    with pytest.raises(AssertionError, match="interleave_size"):
        DCPConfig.from_dict({"interleave_size": 0})


# ────────────────────────────────────────────────────────────────── QREP gate ──


@pytest.mark.parametrize(
    "dcp, mxfp4, expected",
    [
        (8, False, None),  # the ordinary case: supported
        (2, False, None),
        (1, False, "decode_context_parallel_size <= 1 (no DCP group)"),
        (8, True, "fp4 (mxfp4) BMM weights"),
        # dcp is checked first: with no DCP group the other reasons are moot.
        (1, True, "decode_context_parallel_size <= 1 (no DCP group)"),
    ],
)
def test_gate_truth_table(dcp, mxfp4, expected):
    assert qrep_unsupported_reason(dcp, mxfp4) == expected


def test_gate_takes_no_interleave_input():
    """The gate must not be able to see the KV interleave granularity.

    ⚠️ Scope, stated plainly: this checks the FUNCTION, not the call site. The
    merge bug lived at the call site (the gate was nested inside an
    ``if interleave_size > 1`` branch in ``Config.__post_init__``), and **this
    test would not catch that recurring** -- covering it would mean building a
    real Config, which needs a model directory and an HF config. What this does
    buy: interleave can no longer reach the decision *through the signature*, so
    re-coupling the two requires deliberately passing it in, and the docstring
    on `qrep_unsupported_reason` says not to.
    """
    params = inspect.signature(qrep_unsupported_reason).parameters
    assert "interleave" not in " ".join(params), (
        "the QREP gate must not depend on the KV interleave granularity; "
        f"got parameters {list(params)}"
    )


def test_gate_reason_is_human_readable():
    """The reason string is logged verbatim; it should name the actual cause."""
    reason = qrep_unsupported_reason(1, False)
    assert "decode_context_parallel_size" in reason


def test_gate_does_not_look_at_speculative_config():
    """QREP's provenance of q_out is invisible to both verify-path consumers
    (`cprr` and the sparse indexer's candidate exchange), so speculative decode
    never needed this gate. Pin it at the signature, not a value assertion --
    a re-added `speculative_config` keyword-with-default would pass a value
    check again but not this one, same as `test_gate_takes_no_interleave_input`.
    """
    params = inspect.signature(qrep_unsupported_reason).parameters
    assert not any(
        "spec" in p for p in params
    ), f"the QREP gate must not key off the speculative config; got {list(params)}"


# ─────────────────────────────────────── per-layer QREP eligibility (CPU) ──
#
# `q_proj_is_qrep_widened` keeps layers that never opted into QREP (eagle3 /
# DSpark drafts, GLM-5.3's `_ZeroRopePad`) on the AllGather path instead of
# misreading a narrow q as the wide QREP layout. It and `qrep_enabled_for_layer`
# live in atom.config, not atom.model_ops.attention_mla, so these tests run on
# the CPU-only CI gate instead of being silently skipped there.


class _FakeLinear:
    def __init__(self, out_features, in_features=8, effective_tp_overridden=False):
        self.weight = torch.empty(out_features, in_features)
        self.effective_tp_overridden = effective_tp_overridden


class _NoWeight:
    """Stand-in for `_ZeroRopePad`: wraps a linear but exposes no `.weight`."""

    def __init__(self, inner):
        self.inner = inner


def test_q_proj_is_qrep_widened_true_when_shard_matches():
    q_proj = _FakeLinear(
        out_features=64 * 128, effective_tp_overridden=True
    )  # qrep_num_heads * qk_head_dim
    assert q_proj_is_qrep_widened(q_proj, qrep_num_heads=64, qk_head_dim=128)


def test_q_proj_is_qrep_widened_false_for_plain_per_rank_shard():
    """The un-widened case: e.g. an eagle3 / DSpark draft's own q_proj, built
    without `qrep_tp_override`, still at the plain `num_heads/tp` width."""
    q_proj = _FakeLinear(out_features=8 * 128)  # num_local_heads * qk_head_dim
    assert not q_proj_is_qrep_widened(q_proj, qrep_num_heads=64, qk_head_dim=128)


def test_q_proj_is_qrep_widened_false_when_no_weight_attr():
    """GLM-5.3's `_ZeroRopePad` wraps its inner Linear without exposing
    `.weight` on itself -- must fall back cleanly, not raise."""
    q_proj = _NoWeight(_FakeLinear(out_features=8 * 128, effective_tp_overridden=True))
    assert not q_proj_is_qrep_widened(q_proj, qrep_num_heads=64, qk_head_dim=128)


def test_q_proj_is_qrep_widened_false_when_shape_matches_but_not_overridden():
    """The tp == dcp degeneracy: qrep_tp_override's override_tp_size becomes
    tp // dcp == 1 there, a no-op, so a QREP-widened q_proj and a plain
    ReplicatedLinear that was never sharded at all end up the same width --
    shape alone can't tell them apart. Provenance can: only a layer that
    actually went through LinearBase's override path sets
    `effective_tp_overridden`.
    """
    q_proj = _FakeLinear(out_features=64 * 128)  # right shape, wrong provenance
    assert not q_proj_is_qrep_widened(q_proj, qrep_num_heads=64, qk_head_dim=128)


def test_qrep_enabled_for_layer_is_the_and_not_just_the_helper():
    """Pins the `and`, not just `q_proj_is_qrep_widened` -- a weakened or
    dropped `and` at the call site would leave every helper test green while
    re-enabling QREP for unwired layers.
    """
    widened = _FakeLinear(out_features=64 * 128, effective_tp_overridden=True)
    narrow = _FakeLinear(out_features=8 * 128)
    assert qrep_enabled_for_layer(True, widened, qrep_num_heads=64, qk_head_dim=128)
    assert not qrep_enabled_for_layer(
        False, widened, qrep_num_heads=64, qk_head_dim=128
    ), "must not enable QREP just because q_proj happens to be wide enough"
    assert not qrep_enabled_for_layer(
        True, narrow, qrep_num_heads=64, qk_head_dim=128
    ), "must not enable QREP for a layer whose q_proj was never widened"


class _FakeQuantType:
    """Stand-in for a pybind11-bound aiter QuantType member: has a `.name`."""

    def __init__(self, name):
        self.name = name


def test_q_proj_has_row_sliceable_scale_true_when_no_quant_type():
    """Unquantized (e.g. bf16), no `quant_type` attribute at all: safe."""
    q_proj = _FakeLinear(out_features=64 * 128, effective_tp_overridden=True)
    assert q_proj_has_row_sliceable_scale(q_proj)


def test_q_proj_has_row_sliceable_scale_true_for_safe_quant_types():
    for name in ("No", "per_Tensor", "per_1x128", "per_Token"):
        q_proj = _FakeLinear(out_features=64 * 128, effective_tp_overridden=True)
        q_proj.quant_type = _FakeQuantType(name)
        assert q_proj_has_row_sliceable_scale(
            q_proj
        ), f"{name} should be safe to row-slice"


def test_q_proj_has_row_sliceable_scale_false_for_mxfp4_per_1x32():
    """The is_quark_static_mxfp4 gap: the env-var mxfp4 gate says nothing
    about a specific layer's actual quant type. A genuinely mxfp4-quantized
    q_proj (per_1x32) isn't one make_row_view can row-slice.
    """
    q_proj = _FakeLinear(out_features=64 * 128, effective_tp_overridden=True)
    q_proj.quant_type = _FakeQuantType("per_1x32")
    assert not q_proj_has_row_sliceable_scale(q_proj)


def test_q_proj_has_row_sliceable_scale_false_before_weight_scale_exists():
    """The online-quant-streaming timing gap this check used to miss: a
    layer built with `source_quant_dtype` set (LinearBase.__init__) defers
    creating `weight_scale` to `process_weights_after_loading`, well after
    MLAAttention.__init__ runs -- so `q_proj` here has no `weight_scale`
    attribute yet, even though `quant_type` (set unconditionally at
    construction) already says `per_1x32`. Reading the absent `weight_scale`
    as "safe" would enable QREP on a layer that raises the first time
    prefill actually row-slices its real, later-built scale.
    """
    q_proj = _FakeLinear(out_features=64 * 128, effective_tp_overridden=True)
    q_proj.quant_type = _FakeQuantType("per_1x32")
    assert not hasattr(q_proj, "weight_scale")
    assert not q_proj_has_row_sliceable_scale(q_proj)


def test_qrep_enabled_for_layer_falls_back_when_scale_not_row_sliceable():
    q_proj = _FakeLinear(out_features=64 * 128, effective_tp_overridden=True)
    q_proj.quant_type = _FakeQuantType("per_1x32")
    assert not qrep_enabled_for_layer(
        True, q_proj, qrep_num_heads=64, qk_head_dim=128
    ), "a widened q_proj whose scale can't be row-sliced must still fall back"


_ATTENTION_MLA_PATH = (
    Path(__file__).resolve().parent.parent / "atom" / "model_ops" / "attention_mla.py"
)


def test_mlaattention_init_actually_calls_qrep_enabled_for_layer():
    """Pins the real call site, not just the functions it calls.

    Every test above exercises `qrep_enabled_for_layer` / `q_proj_is_qrep_widened`
    / `q_proj_has_row_sliceable_scale` directly -- none of them would notice if
    `MLAAttention.__init__`'s own `self.qrep_enabled = qrep_enabled_for_layer(...)`
    were changed to pass the wrong arguments, hardcode a value, or call
    something else entirely. Read by `ast` on the source text, not by
    importing: `attention_mla.py` needs triton/aiter at module scope, so this
    stays on the CPU gate like the rest of this file's QREP coverage.
    """
    tree = ast.parse(_ATTENTION_MLA_PATH.read_text(encoding="utf-8"))
    calls = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "qrep_enabled_for_layer"
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "qrep_enabled"
    ]
    assert len(calls) == 1, (
        "expected exactly one `self.qrep_enabled = qrep_enabled_for_layer(...)` "
        f"assignment in attention_mla.py; found {len(calls)}"
    )
    (call,) = calls
    assert (
        not call.keywords
    ), "expected qrep_enabled_for_layer's call site to use positional args"

    def _name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return None

    arg_names = [_name(arg) for arg in call.args]
    assert arg_names == [
        "wants_qrep",
        "self.q_proj",
        "self.qrep_num_heads",
        "self.qk_head_dim",
    ], f"qrep_enabled_for_layer's call site changed shape: got {arg_names}"


def test_rebuild_sparse_dcp_persistent_metadata_call_sites_use_their_own_width():
    """Pins gathered_num_heads at both call sites, not just the function.

    Decode and sparse prefill round the gathered query width through
    different tables (mla_dcp_kernel_num_heads vs mla_dcp_sparse_prefill_
    num_heads) and can disagree (e.g. num_heads=12, dcp=8 -> 96 vs 128).
    Passing the wrong one plans persistent work descriptors for the wrong
    nhead; nothing importable on the CPU gate would notice a regression
    here, so read the two call sites from source instead.
    """
    tree = ast.parse(_ATTENTION_MLA_PATH.read_text(encoding="utf-8"))

    def _name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return None

    by_enclosing_func = {}
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_rebuild_sparse_dcp_persistent_metadata"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                by_enclosing_func[func.name] = _name(node.args[6])

    assert by_enclosing_func == {
        "_forward_decode": "self.dcp_kernel_num_heads",
        "_forward_prefill_mla": "self.dcp_sparse_prefill_num_heads",
    }, (
        "a _rebuild_sparse_dcp_persistent_metadata call site is passing the "
        f"wrong gathered_num_heads (7th positional arg): got {by_enclosing_func}"
    )


@needs_dcp_ops
def test_dcp_kernel_and_sparse_prefill_num_heads_can_diverge():
    """Demonstrates the gap the test above guards against is real, not just
    hypothetical: at this head count the two width tables actually disagree.
    """
    decode_width = mla_dcp_kernel_num_heads(
        12, 8, kv_cache_dtype="bf16", persistent=True
    )
    prefill_width = mla_dcp_sparse_prefill_num_heads(12, 8, persistent=True)
    assert (decode_width, prefill_width) == (96, 128), (
        "expected decode's and sparse prefill's width tables to disagree "
        f"here; got decode={decode_width}, prefill={prefill_width} -- if "
        "they now agree, the case above no longer proves the two call sites "
        "must use their own width, pick another num_heads/dcp pair that "
        "still diverges"
    )


# ──────────────── QREP wiring at the q_proj producers (CPU, source-level) ──
#
# Pins the other half of the safety net above: models that are SUPPOSED to
# get QREP actually pass `qrep_tp_override` into q_proj. Nothing else covers
# this -- dropping the override from a draft model leaves every other test
# green, with only a silently-lost AllGather saving as the symptom. Read
# from source (ast), not by constructing a layer, which would need a GPU
# and put this behind @needs_gpu, off the CPU gate.

_ATOM_MODELS = Path(__file__).resolve().parent.parent / "atom" / "models"

# Query projections here must carry qrep_tp_override.
_QREP_WIRED_MODELS = (
    "deepseek_v2.py",  # DeepSeek V2/V3/V3.2 and GLM-5.2 (GlmMoeDsaForCausalLM)
    "kimi_k3.py",  # Kimi-K3 target
    "kimi_k3_dspark.py",  # DSpark draft for Kimi-K3
    "eagle3_deepseek_mla.py",  # eagle3 MLA draft
)

# MLA models that deliberately do NOT wire QREP, with the reason. A newly added
# MLA model has to land in one list or the other -- that is what the census
# below forces, so "nobody remembered the override" cannot pass silently.
_QREP_UNWIRED_MODELS = {
    "glm5_next.py": (
        "GLM-5.3-Flash wraps q_proj in _ZeroRopePad, which hardcodes the "
        "per-rank head count and exposes no .weight for the width check to see "
        "through; wiring it takes three coordinated changes, not the one-line "
        "override."
    ),
}


def _q_proj_linear_calls(path):
    """Every ``self.q_proj``/``self.q_b_proj = <Linear>(...)`` in the module,
    each paired with its enclosing function (typically ``__init__``).

    The enclosing function is returned alongside the call so the override-name
    lookup below can be scoped to it, instead of the whole module: two
    unrelated classes in the same file binding the same variable name to two
    different things must not let one satisfy the other.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in ("q_proj", "q_b_proj")
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    found.append((target.attr, node.value, node.lineno, func))
    return found


def _qrep_override_names(scope):
    """Local names bound to ``qrep_tp_override(...)`` within `scope`.

    `scope` is a function node, not the whole module -- two unrelated classes
    binding the same variable name must not let one satisfy the other.
    Matches both spellings (`**qrep_tp_override(tp)` inline, or bound to a
    name first) since pinning only the inline form would pass just one of
    the four wired models.
    """
    names = set()
    for node in ast.walk(scope):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "qrep_tp_override"
        ):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _passes_qrep_override(call, override_names):
    """Whether the call gets `qrep_tp_override`'s kwargs, inline or via a name."""
    for kw in call.keywords:
        if kw.arg is not None:
            continue
        v = kw.value
        if (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Name)
            and v.func.id == "qrep_tp_override"
        ):
            return True
        if isinstance(v, ast.Name) and v.id in override_names:
            return True
    return False


def test_qrep_override_names_is_scoped_to_the_enclosing_function():
    """A name bound in one function must not satisfy a call in another.

    This used to be checked module-globally (any name ever bound to
    `qrep_tp_override(...)` anywhere in the file passed at any q_proj call
    on any branch), which a coincidental variable-name collision across two
    unrelated classes/functions in the same file could fool.
    """
    source = """
def build_wired(tp_size):
    q_qrep_override = qrep_tp_override(tp_size)
    self.q_proj = ColumnParallelLinear(**q_qrep_override)

def build_unwired(tp_size):
    q_qrep_override = some_other_thing(tp_size)
    self.q_proj = ColumnParallelLinear(**q_qrep_override)
"""
    tree = ast.parse(source)
    functions = {f.name: f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)}

    def _q_proj_call(func):
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "q_proj"
            ):
                return node.value
        raise AssertionError("no q_proj assignment found")

    wired_call = _q_proj_call(functions["build_wired"])
    unwired_call = _q_proj_call(functions["build_unwired"])

    assert _passes_qrep_override(
        wired_call, _qrep_override_names(functions["build_wired"])
    )
    assert not _passes_qrep_override(
        unwired_call, _qrep_override_names(functions["build_unwired"])
    ), "a same-named variable bound in a DIFFERENT function must not pass"

    # Confirms the scoping above is actually doing the work: checked against
    # the whole module's bound names instead, the same call incorrectly
    # passes, because build_wired's binding is visible module-wide.
    assert _passes_qrep_override(unwired_call, _qrep_override_names(tree)), (
        "sanity check failed -- if this now also fails, the synthetic source "
        "above no longer reproduces the module-wide false pass this test "
        "guards against"
    )


@pytest.mark.parametrize("filename", _QREP_WIRED_MODELS)
def test_q_proj_producers_pass_qrep_tp_override(filename):
    """Both target models and both speculative drafts must widen q_proj.

    The drafts are the ones with no other coverage at all: a draft builds its
    own q_proj independently of the target, so forgetting the override there
    costs the AllGather saving for those layers while every functional test
    still passes.
    """
    path = _ATOM_MODELS / filename
    calls = _q_proj_linear_calls(path)
    assert calls, f"{filename}: no self.q_proj/self.q_b_proj assignment found"
    missing = [
        f"{filename}:{lineno} (self.{attr})"
        for attr, call, lineno, func in calls
        if not _passes_qrep_override(call, _qrep_override_names(func))
    ]
    assert not missing, (
        "query projections built without qrep_tp_override -- MLAAttention will "
        "fall back to AllGather Q for their layers: " + ", ".join(missing)
    )


def _builds_mla_modules(path):
    """Whether this file constructs an ``MLAModules(...)``, robust to aliasing.

    A real check (import-tracked name, actually called), not a literal
    substring match on ``"MLAModules("`` -- which an aliased import
    (``import ... as MM``) would evade, and a comment merely mentioning the
    name would falsely trip.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    local_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
            "attention_mla"
        ):
            local_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "MLAModules"
            )
    if not local_names:
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in local_names
        for node in ast.walk(tree)
    )


def test_every_mla_model_has_an_explicit_qrep_decision():
    """A new MLA model must be classified, not silently left unwired.

    `rglob`, not `glob`: a non-recursive scan would miss an MLA model added
    under a subpackage (`atom/models/<family>/...`), the direction some of
    `atom/model_ops/` already organizes new code in.
    """
    producers = {
        path.name for path in _ATOM_MODELS.rglob("*.py") if _builds_mla_modules(path)
    }
    undecided = producers - set(_QREP_WIRED_MODELS) - set(_QREP_UNWIRED_MODELS)
    assert not undecided, (
        "these models build an MLA q_proj but are in neither the QREP-wired "
        f"list nor the documented-unwired list: {sorted(undecided)}. Add "
        "qrep_tp_override to the query projection, or record why it cannot be "
        "wired in _QREP_UNWIRED_MODELS."
    )


# ═══════════════════════════════ ColumnParallelLinear.make_row_view (GPU) ══
#
# DCP query replication widens q_proj so decode holds the whole DCP group's
# heads and can skip its AllGather Q. Prefill needs only this rank's heads, and
# slicing the *output* of the wide projection still paid for the whole group
# (measured +14.8 ms/step of prefill on GLM-5.2 tp8/dcp8). make_row_view instead
# hands prefill a zero-copy row VIEW of the weight.
#
# Why a plain row slice is legal on an already-shuffled weight: shuffle_weight
# reshapes to (N//16, 16, K//BK, BK//K, K) and permutes with N//16 still
# leading, so elements move only WITHIN a 16-row block and blocks never mix.
# Get the boundary wrong and there is no error -- the layer quietly serves
# another rank's heads. Hence bitwise equality below, not a tolerance: this is
# pure work elimination, not an approximation.


DCP = 8
HEADS_PER_RANK = 8
QK_HEAD_DIM = 256  # 192 nope + 64 rope, as on GLM-5.2
ROWS = HEADS_PER_RANK * QK_HEAD_DIM  # 2048 rows per rank
N_WIDE = ROWS * DCP  # 16384, the whole DCP group
K = 2048  # q_lora_rank
TOKENS = 512


@pytest.fixture(scope="module")
def tp_group():
    """world=1 TP group so ColumnParallelLinear can be constructed.

    Deliberately 1: the layer must not re-shard, so `output_size` stays the
    group-wide N and the row view is the only slicing under test.
    """
    import os

    import torch.distributed as dist

    try:
        from aiter.dist.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
    except ImportError as e:
        # A fixture that raises reports as a setup ERROR, not a skip -- which is
        # exactly how a missing @needs_gpu on a consumer surfaced on a CPU-only
        # runner. Skipping here keeps that mistake from turning CI red.
        pytest.skip(f"requires aiter: {e}", allow_module_level=False)

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29578")
        torch.cuda.set_device(0)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29578",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
    yield


def _build_layer(quant_type=None, n_wide=N_WIDE):
    """A group-wide q_proj taken through the real post-loading path.

    `quant_type` defaults inside the body, not in the signature: a default
    argument is evaluated at import time, and `QuantType` only exists when the
    guarded aiter import above succeeded. Evaluating it in the signature would
    turn a missing aiter into a collection-time NameError that takes the
    CPU-only config tests in this file down with it.
    """
    quant_type = QuantType.per_1x128 if quant_type is None else quant_type
    qc = QuantizationConfig()
    qc.global_spec = LayerQuantConfig(
        quant_type=quant_type, quant_dtype=dtypes.fp8, is_dynamic=True
    )
    layer = ColumnParallelLinear(
        input_size=K, output_size=n_wide, bias=False, quant_config=qc, prefix="q_b_proj"
    ).to(DEV)

    g = torch.Generator(device=DEV).manual_seed(0)
    w = torch.randn(n_wide, K, generator=g, device=DEV, dtype=torch.bfloat16) * 0.05
    layer.weight.data = w.to(layer.weight.dtype)
    if quant_type == QuantType.per_1x128:
        s = torch.rand((n_wide + 127) // 128, (K + 127) // 128, generator=g, device=DEV)
        layer.weight_scale.data = (s * 0.01 + 0.01).to(layer.weight_scale.dtype)
    layer.process_weights_after_loading()
    return layer


@needs_gpu
def test_row_view_matches_output_slicing_bitwise(tp_group):
    """Every rank's view must equal that rank's slice of the wide projection."""
    layer = _build_layer()
    g = torch.Generator(device=DEV).manual_seed(1)
    x = torch.randn(TOKENS, K, generator=g, device=DEV, dtype=torch.bfloat16)

    wide = layer(x).view(-1, DCP * HEADS_PER_RANK, QK_HEAD_DIM)
    for rank in range(DCP):
        narrow = layer.make_row_view(rank * ROWS, ROWS)(x).view(
            -1, HEADS_PER_RANK, QK_HEAD_DIM
        )
        want = wide[:, rank * HEADS_PER_RANK : (rank + 1) * HEADS_PER_RANK, :]
        assert torch.equal(narrow, want), (
            f"rank {rank}: row view differs from the wide projection's slice "
            "-- the slice is landing on the wrong rows"
        )


@needs_gpu
def test_row_view_is_zero_copy(tp_group):
    """The point is to avoid a second weight, so it must share storage."""
    layer = _build_layer()
    view = layer.make_row_view(3 * ROWS, ROWS)
    assert (
        view.weight.untyped_storage().data_ptr()
        == layer.weight.untyped_storage().data_ptr()
    )
    assert view.weight.shape == (ROWS, K)
    assert view.output_size == ROWS


@needs_gpu
def test_row_view_slices_the_blockscale(tp_group):
    """per_1x128 scale is [N/128, K/128] and unshuffled: it slices by 128."""
    layer = _build_layer()
    rank = 5
    view = layer.make_row_view(rank * ROWS, ROWS)
    assert view.weight_scale.shape == (ROWS // 128, K // 128)
    assert torch.equal(
        view.weight_scale.data,
        layer.weight_scale.data[rank * ROWS // 128 : (rank + 1) * ROWS // 128],
    )


@needs_gpu
def test_row_view_does_not_disturb_the_parent(tp_group):
    """Taking a view must leave the full projection usable and unchanged."""
    layer = _build_layer()
    g = torch.Generator(device=DEV).manual_seed(2)
    x = torch.randn(TOKENS, K, generator=g, device=DEV, dtype=torch.bfloat16)

    before = layer(x).clone()
    layer.make_row_view(0, ROWS)
    assert torch.equal(layer(x), before)
    assert layer.output_size == N_WIDE


@needs_gpu
@pytest.mark.parametrize("start, length", [(8, ROWS), (0, 24), (ROWS + 16, ROWS)])
def test_row_view_rejects_misaligned_slices(tp_group, start, length):
    """A shuffled weight may only be cut on 16-row blocks; per_1x128 wants 128.

    Without this the slice silently yields a scrambled weight, so the assert is
    the only thing standing between a typo and wrong attention output.
    """
    layer = _build_layer()
    with pytest.raises(AssertionError):
        layer.make_row_view(start, length)


@needs_gpu
def test_row_view_rejects_out_of_range(tp_group):
    layer = _build_layer()
    with pytest.raises(AssertionError, match="out of range"):
        layer.make_row_view(N_WIDE - ROWS + 128, ROWS)


@needs_gpu
def test_row_view_refuses_unhandled_scale_layout(tp_group):
    """Only the scale layouts that were verified are allowed through.

    per_Tensor has a scalar scale that the shallow copy shares correctly, but a
    layout with a per-output-channel scale this code has not been taught to
    slice must raise rather than hand back a mismatched scale.
    """
    layer = _build_layer(quant_type=QuantType.per_1x32)
    if getattr(layer, "quant_type", None) != QuantType.per_1x32:
        pytest.skip("per_1x32 not constructible in this build")
    with pytest.raises(NotImplementedError, match="per-output-channel scale"):
        layer.make_row_view(0, ROWS)


# ────────────────────────────────────────────── gathered query-head widths ──
#
# Both DCP call sites all-gather Q on the head dim, so the width reaching the
# kernel is ``num_heads * dcp_world_size``, not the per-rank count. Not every
# width computes correctly there: gqa=64 is served only by the PERSISTENT decode
# kernel (fp8/fp8 aborts on it, bf16 silently miscomputes), and fp8 has no gqa=32
# kernel outside persistent mode either. So the width table has to be picked by
# the mode the kernel actually runs in.
#
# The invariant under test is the PAIRING: whichever persistence predicate a call
# site uses, the width it pads to must come from the matching table. Asserting a
# predicate's current value instead would just have to be edited by whoever
# legitimately enables persistent sparse prefill; the pairing keeps holding
# across that change, which is the point.

# Per-rank counts that put the GATHERED width on each interesting value: 8 heads
# x dcp8 and 16 x dcp4 both land on 64, the width GLM-5.2 uses and the one that
# is wrong outside persistent mode.
HEAD_WIDTH_SHAPES = [(8, 2), (8, 4), (8, 8), (16, 2), (16, 4), (32, 4), (4, 8)]
HEAD_WIDTH_MIN = 16


@needs_dcp_ops
@pytest.mark.parametrize("num_heads, dcp", HEAD_WIDTH_SHAPES)
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_non_persistent_decode_never_dispatches_gqa64(num_heads, dcp, dtype):
    """gqa=64 is wrong for BOTH dtypes without persistent mode."""
    w = mla_dcp_kernel_num_heads(
        num_heads, dcp, HEAD_WIDTH_MIN, kv_cache_dtype=dtype, persistent=False
    )
    assert w != 64, f"non-persistent {dtype} decode selected gqa=64 ({num_heads}x{dcp})"


@needs_dcp_ops
@pytest.mark.parametrize("num_heads, dcp", HEAD_WIDTH_SHAPES)
def test_non_persistent_fp8_decode_also_skips_gqa32(num_heads, dcp):
    """fp8 has no gqa=32 kernel outside persistent mode; bf16 does."""
    w = mla_dcp_kernel_num_heads(
        num_heads, dcp, HEAD_WIDTH_MIN, kv_cache_dtype="fp8", persistent=False
    )
    assert w != 32, f"non-persistent fp8 decode selected gqa=32 ({num_heads}x{dcp})"


@needs_dcp_ops
@pytest.mark.parametrize("num_heads, dcp", HEAD_WIDTH_SHAPES)
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_decode_width_comes_from_the_matching_table(num_heads, dcp, dtype):
    gathered = max(num_heads * dcp, HEAD_WIDTH_MIN)
    for persistent in (False, True):
        w = mla_dcp_kernel_num_heads(
            num_heads, dcp, HEAD_WIDTH_MIN, kv_cache_dtype=dtype, persistent=persistent
        )
        if persistent:
            allowed = _MLA_DCP_KERNEL_WIDTHS
        else:
            allowed = (
                _MLA_DCP_KERNEL_WIDTHS_NON_PERSISTENT_FP8
                if dtype == "fp8"
                else _MLA_DCP_KERNEL_WIDTHS_NON_PERSISTENT
            )
        if gathered <= allowed[-1]:  # past the table it falls back and warns
            assert (
                w in allowed
            ), f"{dtype} persistent={persistent}: {w} not in {allowed}"
            assert w >= gathered, "width must cover the gathered heads"


@needs_dcp_ops
@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("num_heads, dcp", HEAD_WIDTH_SHAPES)
def test_sparse_prefill_width_matches_its_own_persistence_predicate(
    num_heads, dcp, rebuild
):
    """The pairing this whole file exists for.

    The width must come from this call site's own table
    (``_MLA_DCP_SPARSE_PREFILL_WIDTHS*``), not decode's -- reading decode's
    answer here is exactly how a sparse prefill would get its width chosen
    from the wrong table.
    """
    persistent = mla_dcp_sparse_prefill_is_persistent(
        dcp, True, sparse_metadata_rebuild=rebuild
    )
    w = mla_dcp_sparse_prefill_num_heads(
        num_heads, dcp, HEAD_WIDTH_MIN, persistent=persistent
    )
    allowed = (
        _MLA_DCP_SPARSE_PREFILL_WIDTHS_PERSISTENT
        if persistent
        else _MLA_DCP_SPARSE_PREFILL_WIDTHS
    )
    gathered = max(num_heads * dcp, HEAD_WIDTH_MIN)
    if gathered <= allowed[-1]:
        assert w in allowed, f"persistent={persistent}: {w} not in {allowed}"
        assert w >= gathered
    if not persistent:
        assert w != 64, "non-persistent sparse prefill must not dispatch gqa=64"


def test_sparse_prefill_persistence_does_not_look_at_dtype():
    """Was gated on KV cache dtype (fp8-only); see
    mla_dcp_sparse_prefill_is_persistent's docstring for why that gate was
    wrong. A value assertion would not catch a dtype parameter re-added as an
    optional keyword with a default; pin the decoupling at the signature
    instead.
    """
    params = inspect.signature(mla_dcp_sparse_prefill_is_persistent).parameters
    assert not any("dtype" in p or "kv_cache" in p for p in params), (
        "the DCP sparse-prefill persistence predicate must not key off KV "
        f"cache dtype; got parameters {list(params)}"
    )


@pytest.mark.parametrize("dcp_world_size", [1, 2, 8])
@pytest.mark.parametrize("dcp_persistent_supported", [False, True])
@pytest.mark.parametrize("sparse_metadata_rebuild", [False, True])
def test_sparse_prefill_persistence_is_decodes_with_is_sparse_true(
    dcp_world_size, dcp_persistent_supported, sparse_metadata_rebuild
):
    """`mla_dcp_sparse_prefill_is_persistent` is a thin `is_sparse=True` call
    into `mla_dcp_decode_is_persistent`, not a second copy of its body --
    pins that the delegation actually agrees with the general function
    across the inputs that matter, not just that it happens to today by
    construction.
    """
    assert mla_dcp_sparse_prefill_is_persistent(
        dcp_world_size,
        dcp_persistent_supported,
        sparse_metadata_rebuild=sparse_metadata_rebuild,
    ) == mla_dcp_decode_is_persistent(
        True,
        dcp_world_size,
        dcp_persistent_supported,
        sparse_metadata_rebuild=sparse_metadata_rebuild,
    )


def test_decode_is_persistent_false_below_dcp_2():
    assert not mla_dcp_decode_is_persistent(
        False, 1, True, sparse_metadata_rebuild=True
    )


def test_decode_is_persistent_sparse_requires_metadata_rebuild():
    """Dense decode (`is_sparse=False`) never looks at `sparse_metadata_rebuild`
    at all; sparse decode is unconditionally non-persistent without it."""
    assert mla_dcp_decode_is_persistent(False, 8, True, sparse_metadata_rebuild=False)
    assert not mla_dcp_decode_is_persistent(
        True, 8, True, sparse_metadata_rebuild=False
    )
    assert mla_dcp_decode_is_persistent(True, 8, True, sparse_metadata_rebuild=True)


def test_decode_is_persistent_requires_platform_support():
    assert not mla_dcp_decode_is_persistent(
        False, 8, False, sparse_metadata_rebuild=True
    )


@needs_dcp_ops
def test_sparse_prefill_persistent_table_would_allow_gqa64():
    """The persistent row is wired up, not dead.

    Guards against the table being present but unreachable -- if enabling
    persistent sparse prefill still could not pick 64, the plumbing would be
    pointless and the 64->128 padding would stay.
    """
    assert (
        mla_dcp_sparse_prefill_num_heads(8, 8, HEAD_WIDTH_MIN, persistent=True) == 64
    ), "persistent sparse prefill should reach gqa=64, not pad past it"
    assert (
        mla_dcp_sparse_prefill_num_heads(8, 8, HEAD_WIDTH_MIN, persistent=False) == 128
    )
