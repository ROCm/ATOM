# SPDX-License-Identifier: MIT
"""Gate real target/draft cache writers through the ATOM Gluon adapter on gfx950."""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def use_checkout():
    # A script puts its own directory first; cwd does not override an editable
    # /app/ATOM install as it does for python -c or python -m. Pin this checkout
    # before importing any ATOM module, even with a conflicting PYTHONPATH.
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    spec = importlib.util.find_spec("atom")
    expected = root / "atom/__init__.py"
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise RuntimeError(f"Gluon adapter gate must import {expected}; got {spec}")
    print(f"[gluon-adapter-gate] atom_source={spec.origin}", flush=True)


def load_attention_dispatch():
    # AITER's legacy-path loader can replace the canonical module during ATOM
    # import. Resolve AFTER importing ATOM, from the exact callable it uses.
    from atom.model_ops import attention_mha

    module = importlib.import_module(attention_mha.unified_attention.__module__)
    if module.unified_attention is not attention_mha.unified_attention:
        raise RuntimeError("Gluon gate is observing a different attention dispatcher")
    return module


def check_case(
    draft, qlen, klen, scalar=1.0, decode_graph=False, heads=16, draft_kv_heads=16
):
    import torch

    from atom.model_ops.attention_mha import PagedAttentionImpl
    from atom.model_ops.layernorm import GemmaRMSNorm

    device = torch.device("cuda")
    kv_heads, dim, page = draft_kv_heads if draft else 1, 128, 128
    torch.manual_seed(42)
    pages = (klen + page - 1) // page
    blocks = pages + 3
    table = (torch.randperm(blocks - 1, device=device)[:pages] + 1).to(torch.int32)[
        None, :
    ]
    positions = torch.arange(klen, device=device, dtype=torch.int64)
    slots = table[0, positions // page].long() * page + positions % page
    kc = torch.full(
        (blocks, kv_heads, 8, page, 16),
        float("nan"),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    vc = torch.full_like(kc, float("nan"))
    # The draft writer must leave these unused planes poisoned.
    ks = torch.full((blocks, kv_heads, page), float("nan"), device=device)
    vs = torch.full_like(ks, float("nan"))
    angles = positions.float()[:, None] * torch.linspace(
        0.001, 0.1, dim // 2, device=device
    )
    rope = SimpleNamespace(
        cos_cache=angles.cos(),
        sin_cache=angles.sin(),
        is_neox_style=True,
        rotary_dim=dim,
    )
    norm_args = (
        {}
        if draft
        else {
            "q_norm": GemmaRMSNorm(dim).to(device),
            "k_norm": GemmaRMSNorm(dim).to(device),
        }
    )
    layer = PagedAttentionImpl(
        heads,
        dim,
        dim**-0.5,
        kv_heads,
        None,
        kv_cache_dtype="fp8",
        layer_num=60 if draft else 0,
        rotary_emb=rope,
        **norm_args,
    )
    layer.kv_scale.fill_(scalar)
    meta = SimpleNamespace(
        has_cached=True,
        block_tables=table,
        cu_seqlens_q=torch.tensor([0, qlen, 999999], dtype=torch.int32, device=device),
        context_lens=torch.tensor([klen, 999999], dtype=torch.int32, device=device),
        slot_mapping=slots,
        max_seqlen_q=qlen,
        max_seqlen_k=klen,
    )
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, is_draft=draft, scheduled_bs=1),
        attn_metadata=meta,
        kv_cache_data={
            f"layer_{layer.layer_num}": SimpleNamespace(
                k_cache=kc, v_cache=vc, k_scale=ks, v_scale=vs
            )
        },
    )
    qkv = torch.randn(
        klen, (heads + 2 * kv_heads) * dim, device=device, dtype=torch.bfloat16
    )
    # Populate a prefix, then append only the scheduled query tokens. Writer
    # dispatch uses the number of NEW tokens, not the total context length.
    prefix_len = klen - qlen
    assert prefix_len > 0
    written = []
    for begin, end in ((0, prefix_len), (prefix_len, klen)):
        chunk = qkv[begin:end]
        q, k, v = [
            t.view(end - begin, h, dim)
            for t, h in zip(
                chunk.split([heads * dim, kv_heads * dim, kv_heads * dim], dim=-1),
                (heads, kv_heads, kv_heads),
            )
        ]
        meta.slot_mapping = slots[begin:end]
        meta.has_cached = begin > 0
        q, k, v, kc_out, vc_out, kscale, vscale = layer.rope_cache(
            q, k, v, chunk, positions[begin:end], ctx
        )
        assert kc_out.data_ptr() == kc.data_ptr() and vc_out.data_ptr() == vc.data_ptr()
        written.append((k, v))
    all_keys = torch.cat([pair[0] for pair in written])
    all_values = torch.cat([pair[1] for pair in written])
    if draft:
        assert not layer.per_token_quant
        assert kscale is layer.kv_scale and vscale is layer.kv_scale
        assert ks.isnan().all() and vs.isnan().all()
    else:
        assert kscale is ks and vscale is vs
    # Read the actual writer's native pages using independent Torch indexing.
    # Round the dequantized values to the BF16 operands used by attention.
    dense_k = kc.permute(0, 3, 1, 2, 4).reshape(blocks, page, kv_heads, dim).float()
    dense_v = vc.permute(0, 2, 4, 1, 3).reshape(blocks, page, kv_heads, dim).float()
    page_ids, offsets = slots // page, slots % page
    keys, values = dense_k[page_ids, offsets], dense_v[page_ids, offsets]
    keys *= kscale if draft else kscale[page_ids, :, offsets].unsqueeze(-1)
    values *= vscale if draft else vscale[page_ids, :, offsets].unsqueeze(-1)
    assert (
        keys.isfinite().all() and values.isfinite().all()
    ), "Writer left valid KV uninitialized"
    # A wrong SHUFFLE layout must fail before evaluating attention.
    torch.testing.assert_close(keys, all_keys.float(), atol=0.25, rtol=0.15)
    torch.testing.assert_close(values, all_values.float(), atol=0.25, rtol=0.15)
    keys = keys.bfloat16().float().repeat_interleave(heads // kv_heads, dim=1)
    values = values.bfloat16().float().repeat_interleave(heads // kv_heads, dim=1)
    query = q
    assert query.shape[0] == qlen
    scores = torch.einsum("qhd,khd->hqk", query.float(), keys) * dim**-0.5
    mask = positions[None, :] > positions[-qlen:, None]
    scores.masked_fill_(mask, -float("inf"))
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).bfloat16()
    assert expected.isfinite().all()
    if decode_graph:
        # Three graph rows, with a changing number of live requests. Page 0
        # stays poisoned; only padding may read it and its output must be zero.
        ctx.context.is_prefill = False
        ctx.context.scheduled_bs = 99  # must never be baked into the graph
        qpad = torch.empty(3 * qlen, heads, dim, device=device, dtype=query.dtype)
        tables = table.expand(3, -1).clone()
        lengths = torch.empty(3, device=device, dtype=torch.int32)
        slots_pad = torch.empty(3 * qlen, device=device, dtype=torch.int64)
        cu = torch.empty(4, device=device, dtype=torch.int32)
        meta.block_tables, meta.context_lens = tables, lengths
        meta.slot_mapping, meta.cu_seqlens_q = slots_pad, cu
        attention = layer.dispatch_backend(ctx, qpad, k, v)
        assert attention.__func__ is PagedAttentionImpl.paged_attention_gluon

        def setup(live, variant):
            qpad.copy_(query.repeat(3, 1, 1) * (1 + variant * 0.125))
            tables.copy_(table.expand(3, -1))
            if variant:
                # Permute complete pages, keeping the partial final page last.
                tables[:, :2] = tables[:, :2].flip(1)
            tables[live:] = 0
            lengths.fill_(klen - variant * 17)
            lengths[live:] = variant  # padded draft lengths may be positive
            slots_pad.fill_(-1)
            slots_pad[: live * qlen] = 128
            cu.copy_(
                torch.tensor(
                    [min(i, live) * qlen for i in range(4)],
                    device=device,
                    dtype=torch.int32,
                )
            )

        def reference(live):
            result = torch.zeros_like(qpad)
            for seq in range(live):
                length = int(lengths[seq])
                pos = torch.arange(length, device=device)
                ids, offs = tables[seq, pos // page].long(), pos % page
                key = dense_k[ids, offs] * (
                    kscale if draft else kscale[ids, :, offs].unsqueeze(-1)
                )
                val = dense_v[ids, offs] * (
                    vscale if draft else vscale[ids, :, offs].unsqueeze(-1)
                )
                key = key.bfloat16().float().repeat_interleave(heads // kv_heads, 1)
                val = val.bfloat16().float().repeat_interleave(heads // kv_heads, 1)
                query_seq = qpad[seq * qlen : (seq + 1) * qlen].float()
                score = torch.einsum("qhd,khd->hqk", query_seq, key) * dim**-0.5
                score.masked_fill_(
                    pos[None, :]
                    > (length - qlen + torch.arange(qlen, device=device))[:, None],
                    -float("inf"),
                )
                result[seq * qlen : (seq + 1) * qlen] = torch.einsum(
                    "hqk,khd->qhd", score.softmax(-1), val
                ).bfloat16()
            return result

        def launch():
            return attention(qpad, k, v, kc, vc, kscale, vscale, ctx)

        setup(2, 0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = launch()
        for live, variant in ((2, 0), (1, 1), (3, 2), (0, 0)):
            setup(live, variant)
            before = (lengths.clone(), cu.clone(), tables.clone())
            graph.replay()
            expected = reference(live)
            torch.testing.assert_close(graph_output, expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(launch(), expected, atol=0.02, rtol=0.02)
            for source, saved in zip((lengths, cu, tables), before):
                torch.testing.assert_close(source, saved)
        print(
            f"[gluon-adapter-gate] decode graph draft={draft} q={qlen} kv={klen} "
            "live batches=2/1/3/0; changed contexts, pages and queries passed",
            flush=True,
        )
        return
    attention = layer.dispatch_backend(ctx, query, k, v)
    assert attention.__func__ is PagedAttentionImpl.prefill_attention
    out = attention(query, k, v, kc, vc, kscale, vscale, ctx)
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
    print(
        f"[gluon-adapter-gate] draft={draft} q={qlen} kv={klen} scalar={scalar} passed",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-source",
        action="store_true",
        help="Verify checkout selection without GPU dependencies",
    )
    args = parser.parse_args()
    use_checkout()
    if args.check_source:
        return

    from atom.utils import envs

    ua = load_attention_dispatch()
    assert (
        ua.DEVICE_ARCH == "gfx950"
    ), "Writer/attention gate requires the target gfx950 GPU"
    launches = []
    configs = []
    original = ua._unified_attention_gfx950
    kernel = ua._unified_attention_kernel_gfx950

    def record(*args, **kwargs):
        launches.append(kwargs.get("NUM_SEGMENTS", 1))
        configs.append(
            {key: kwargs[key] for key in ("BLOCK_M", "TILE_SIZE", "num_warps")}
        )
        return original(*args, **kwargs)

    def forbid(*args, **kwargs):
        raise AssertionError("Gluon adapter fell back to Triton attention")

    class GlobalLoads:
        def __getitem__(self, grid):
            def launch(**kwargs):
                kwargs["USE_LOAD_BUFFER_OP"] = False
                return kernel[grid](**kwargs)

            return launch

    with (
        patch.object(envs, "ATOM_FORCE_ATTN_TRITON", True),
        patch.object(envs, "ATOM_USE_UNIFIED_ATTN", False),
        patch.object(ua, "_unified_attention_gfx950", record),
        patch.object(ua, "_unified_attention_2d_triton", forbid),
        patch.object(ua, "_unified_attention_3d_triton", forbid),
    ):
        cases = [(False, 4, 513, 1.0)]
        cases += [
            (True, q, 16385 if q < 5 else 2049, scale)
            for q in (1, 4, 1024)
            for scale in (1.0, 0.375)
        ]
        for case in cases:
            before = len(launches)
            check_case(*case)
            assert len(launches) == before + 1
        with patch.object(ua, "_unified_attention_kernel_gfx950", GlobalLoads()):
            for q in (1, 1024):
                before = len(launches)
                check_case(True, q, 16385 if q == 1 else 2049, 0.375)
                assert len(launches) == before + 1
        for draft in (False, True):
            for q in (1, 4):
                before = len(launches)
                check_case(draft, q, 16385, 0.375 if draft else 1.0, decode_graph=True)
                assert any(n > 1 for n in launches[before:]), launches[before:]
        # EAGLE3-GQA shares target's global four KV heads. TP4 gives
        # Q16/KV1, TP8 gives Q8/KV1 with pairs of replicated KV heads.
        for heads in (16, 8):
            for draft in (False, True):
                for q in (1, 4, 1024):
                    check_case(
                        draft,
                        q,
                        16385 if q < 5 else 2049,
                        0.375 if draft else 1.0,
                        decode_graph=q < 5,
                        heads=heads,
                        draft_kv_heads=1,
                    )
    assert 1 in launches and any(n > 1 for n in launches), launches
    print(
        f"[gluon-adapter-gate] old draft and GQA TP4/TP8 writer/decode graph cases passed; "
        f"observed launches={len(launches)}; splits={launches}; official configs={configs}",
        flush=True,
    )


if __name__ == "__main__":
    main()
