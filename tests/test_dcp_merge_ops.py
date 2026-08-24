# SPDX-License-Identifier: MIT
"""DCP merge-path ops in ``atom/model_ops/dcp_ops.py``.

Three ops that the sparse work leans on but that had no coverage:

  * ``correct_attn_out``        -- the LSE merge kernel behind ``cp_lse_ag_out_rs``
  * ``get_dcp_local_seq_lens``  -- how many tokens of a sequence this rank holds
  * ``reorg_kvcache``           -- AllGathered chunk blocks -> per-seq contiguous

The merge is the load-bearing one. ``cp_lse_ag_out_rs`` reconstructs a global
softmax from per-rank partial attentions, so the test here is not "the kernel
runs" but "summing the corrected per-rank outputs reproduces plain dense
attention over the union" -- the premise the whole DCP design rests on.

The empty-rank case gets its own test because it is the one this branch changed:
under sparse prefill a rank routinely owns no candidate for a row, and aiter then
returns ``o=NaN`` with ``lse=-inf``. Without the ``factor == 0 -> 0`` scrub in
the kernel, ``NaN * 0 = NaN`` survives the ReduceScatter and poisons EVERY
rank's output for that row -- silently, with no fault.

``get_dcp_local_seq_lens`` and ``reorg_kvcache`` run on CPU tensors; only the
merge tests need a GPU.
"""

import numpy as np
import pytest
import torch

try:
    from atom.model_ops.dcp_ops import (
        correct_attn_out,
        correct_attn_out_rs_layout,
        dcp_global_pos,
        dcp_local_index,
        dcp_owner_rank,
        get_dcp_local_seq_lens,
        reorg_kvcache,
    )
except ImportError as _e:  # triton absent on a CPU-only runner
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel needs a GPU"
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("N", [2, 4, 8])
def test_rs_layout_merge_reproduces_dense_attention(dtype, N):
    """Rank-major correction plus RS chunking reproduces the dense result."""
    B, H, L, D = 5, 8, 128, 64
    dense_o, _, outs, lses = _dense_and_shards(B, H, L, D, N, dtype, seed=11)
    local_heads = H // N
    packed_per_source = []

    for source_rank in range(N):
        packed = correct_attn_out_rs_layout(
            outs[source_rank].clone(), lses, source_rank
        )
        assert packed.shape == (N * B, local_heads, D)

        # The optimized kernel must be exactly the general correction arranged
        # as N contiguous reduce-scatter destination chunks.
        corrected, _ = correct_attn_out(outs[source_rank].clone(), lses, source_rank)
        reference_layout = (
            corrected.view(B, N, local_heads, D)
            .permute(1, 0, 2, 3)
            .reshape(N * B, local_heads, D)
        )
        tol = 1e-6 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(packed, reference_layout, rtol=tol, atol=tol)
        packed_per_source.append(packed.float())

    # Simulate ReduceScatter(sum): sum every source rank's packed contribution,
    # then give destination rank r its contiguous B-row chunk.
    reduced = torch.stack(packed_per_source).sum(0)
    tol = 1e-5 if dtype == torch.float32 else 3e-2
    for destination_rank in range(N):
        got = reduced[destination_rank * B : (destination_rank + 1) * B]
        expected = dense_o[
            :, destination_rank * local_heads : (destination_rank + 1) * local_heads
        ]
        torch.testing.assert_close(got, expected, rtol=tol, atol=tol)


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


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
def test_dcp_helpers_reduce_to_round_robin_when_interleave_1(dcp_size):
    # S == 1 must be bit-identical to the old inline round-robin (owner = i%W,
    # local index = i//W) -- this is the S=1 regression guarantee.
    pos = np.arange(0, 300, dtype=np.int64)
    np.testing.assert_array_equal(dcp_owner_rank(pos, dcp_size, 1), pos % dcp_size)
    np.testing.assert_array_equal(dcp_local_index(pos, dcp_size, 1), pos // dcp_size)


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
