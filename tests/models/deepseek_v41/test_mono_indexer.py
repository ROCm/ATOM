# SPDX-License-Identifier: MIT
"""Standalone Mono against the native MXFP8/FP4 indexer and graph replay."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""),
    reason="gfx950 required",
)


def make_case(
    rows, context, ratio, layout, *, candidate=False, reindex=False, qlen=1, pad=True
):
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
        pack_kv_cache,
    )
    from aiter.tuned_gemm import tgemm

    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.model_ops.blockscale import (
        native_quant_linear,
        quantize_fp4,
        quantize_fp8,
    )
    from atom.model_ops.deepseek_v41.candidate_table import bind_candidates
    from atom.model_ops.deepseek_v41.index_plane import IndexUnits
    from atom.model_ops.deepseek_v41.paged_scoring import (
        quantize_query_fp4,
        score_topk_quantized,
    )
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
    from atom.model_ops.utils import shuffle_weights
    from atom.models.deepseek_v41.config import AttentionMode, LayerAttentionSpec
    from atom.models.deepseek_v41.mono.indexer import MonoIndexer

    torch.manual_seed(304241 + rows + context)
    dev = "cuda"
    layer = 24 if reindex else 20 if candidate else 2
    spec = LayerAttentionSpec(
        layer,
        ratio,
        AttentionMode.REINDEX if reindex else AttentionMode.FULL,
        kv_owner=20 if reindex or candidate else 2,
        topk_owner=layer,
        candidate_owner=20 if candidate or reindex else None,
    )
    raw = torch.randn(128, 32, 40, 32, device=dev).bfloat16() * 0.03
    amax = raw.float().abs().amax((1, 3))
    ws = (2 ** torch.ceil(torch.log2(amax / 448))).to(torch.float8_e8m0fnu)
    w = (
        (raw.float() / ws.float()[:, None, :, None])
        .to(torch.float8_e4m3fn)
        .reshape(4096, 1280)
    )
    iw = torch.randn(32, 5120, device=dev).bfloat16() * 0.03
    w = torch.nn.Parameter(w, requires_grad=False)
    shuffle_weights(w)
    indexer = SimpleNamespace(
        spec=spec,
        heads=32,
        head_dim=128,
        topk=512,
        block_size=8,
        topk_blocks=2048,
        weights_scale=1 / 64,
        wq_b=SimpleNamespace(weight=w, weight_scale=ws),
        weights_proj=SimpleNamespace(weight=iw),
    )

    def publish(selected, chosen, step):
        step.selected[layer] = selected.unsqueeze(0)
        if chosen is not None:
            step.candidates[layer] = chosen

    indexer._publish = publish
    rope = RotaryEmbedding(
        64,
        context + 128,
        base=160000.0,
        original_length=65536,
        factor=16.0,
        beta_fast=32,
        beta_slow=1,
    ).cuda()
    model = SimpleNamespace(
        layers=[SimpleNamespace(attn=SimpleNamespace(indexer=indexer, spec=spec))],
        global_rope=rope,
        window_rope=rope,
    )
    from atom.models.deepseek_v41.mono.indexer_workspace import IndexerWorkspace

    runner = MonoIndexer(
        model,
        context + 128,
        True,
        layout,
        workspace=IndexerWorkspace(model, context + 128, 1),
    )
    geo = V41PoolGeometry(
        40,
        ((2, 2), (8, 2), (14, 2), (20, 1)),
        128,
        128,
        512,
        128,
        index_topk=512,
        index_block_rows=8,
        index_fp4=True,
        index_layout=layout,
    )
    width = ((context + 128 + 127) // 128) * 128 // ratio
    pages = width // geo.rows_per_page(ratio)
    table = torch.randperm(pages, device=dev, dtype=torch.int32).unsqueeze(0)
    batch = torch.arange(rows, device=dev, dtype=torch.int32) // qlen
    table = table.repeat(int(batch[-1]) + 1, 1)
    positions = torch.full((rows,), context - 1, device=dev, dtype=torch.int64)
    positions += torch.arange(rows, device=dev) % qlen
    visible = ((positions + 1) // ratio).int()
    # A zero-history padding row exercises -1 output and no scoring tasks.
    if pad and rows > 1:
        batch[-1] = -1
        positions[-1] = 0
        visible[-1] = 0
    step = SimpleNamespace(
        batch_ids=batch,
        slots=torch.arange(table.shape[0], device=dev, dtype=torch.int32),
        positions=positions,
        block_tables=table,
        visible={ratio: visible},
        selected={},
        candidates={},
        candidate_blocks={},
        decode=True,
        running_tokens=rows,
        requests=(1,),
    )
    values, scales = quantize_fp4(
        torch.randn(width, 128, device=dev).bfloat16(),
        group_size=32,
        scale_dtype=torch.float8_e8m0fnu,
    )
    values, scales = values.view(torch.uint8), scales.view(torch.uint8)
    if layout == "mfma":
        values = values.view(-1, 64, 4, 16).permute(0, 2, 1, 3).contiguous()
        scales = scales.view(-1, 2, 32, 2, 2).permute(0, 4, 2, 3, 1).contiguous()
    else:
        values, scales = pack_kv_cache(values, scales, 8)
    units = IndexUnits(values.view(-1, 8, 64), scales.view(-1, 8, 4), layout)
    cache = SimpleNamespace(
        geometry=geo,
        num_pages=pages,
        kv_fp8=False,
        packed=False,
        index_units={spec.kv_owner: units},
        rope_positions=lambda step: step.positions,
    )

    def candidates(step, source, ratio):
        if source not in step.candidate_blocks:
            step.candidate_blocks[source] = bind_candidates(
                step.candidates[source],
                table,
                batch,
                geo.index_blocks_per_page(ratio),
                visible,
                rows_per_block=8,
            )
        return step.candidate_blocks[source]

    cache.candidate_blocks = candidates
    if reindex:
        # Spread the candidates across the whole context, including newest.
        count = min(2048, (int(visible.max()) + 7) // 8)
        ids = torch.linspace(
            0, max((int(visible.max()) - 1) // 8, 0), count, device=dev
        ).int()
        chosen = torch.full((rows, 2048), -1, device=dev, dtype=torch.int32)
        chosen[:, :count] = ids
        if pad and rows > 1:
            chosen[-1] = -1
        step.candidates[20] = chosen
    hidden = torch.randn(1, rows, 5120, device=dev).bfloat16()
    qr, qrs = quantize_fp8(torch.randn(1, rows, 1280, device=dev).bfloat16())

    def native(query_override=None, weights_override=None):
        query = native_quant_linear(
            qr, w, ws, x_scale=qrs, weight_preshuffled=True
        ).view(rows, 32, 128)
        weights = (
            tgemm.mm(hidden, iw, None, otype=torch.bfloat16)[0]
            if weights_override is None
            else weights_override
        )
        q = (
            quantize_query_fp4(query, rope, positions, layout=layout)
            if query_override is None
            else query_override
        )
        # The upstream row-group scorer's ragged fast path mis-scores a
        # partial or padded final six-row group. Use its one-row path as
        # the reference for those tails (same pages, query and arithmetic).
        scoring_qlen = qlen if rows % qlen == 0 and not pad else 1
        scoring_table = (
            table if scoring_qlen == qlen else table[batch.clamp_min(0).long()]
        )
        return (
            q,
            weights,
            score_topk_quantized(
                q,
                weights,
                units,
                None,
                visible,
                topk=512,
                weight_scale=1 / 64,
                ragged=(
                    None
                    if reindex
                    else __import__(
                        "atom.model_ops.fp4_mqa_ragged_metadata",
                        fromlist=["Fp4MqaRaggedMetadata"],
                    ).Fp4MqaRaggedMetadata(
                        (
                            torch.arange(
                                scoring_table.shape[0] + 1,
                                device=dev,
                                dtype=torch.int32,
                            )
                            * scoring_qlen
                        ).clamp_max(rows),
                        scoring_qlen,
                        scoring_table,
                        geo.index_blocks_per_page(ratio),
                    )
                ),
                candidates=candidates(step, 20, ratio) if reindex else None,
                block_size=8,
                candidate_count=2048 if candidate else 0,
            ),
        )

    return SimpleNamespace(
        runner=runner,
        indexer=indexer,
        hidden=hidden,
        qr=qr,
        qrs=qrs,
        cache=cache,
        step=step,
        rope=rope,
        native=native,
    )


@pytest.mark.parametrize(
    "rows,context,ratio,layout,candidate,reindex,qlen",
    [
        (1, 8192, 2, "mfma", False, False, 1),
        (8, 8192, 1, "preshuffle", True, False, 1),
        (24, 8192, 1, "mfma", False, True, 1),
        (40, 163840, 2, "mfma", False, False, 1),
        (16, 163840, 1, "preshuffle", False, True, 1),
        (13, 8192, 2, "mfma", False, False, 6),
        (40, 163840, 1, "preshuffle", True, False, 6),
        (64, 8192, 2, "mfma", False, False, 1),
        (6, 8192, 1, "mfma", False, True, 6),
        (48, 163840, 2, "mfma", False, False, 6),
    ],
)
def test_standalone_mono_matches_native_and_graph(
    rows, context, ratio, layout, candidate, reindex, qlen, mixed=False
):
    from atom.models.deepseek_v41.mono.runner import V41MonoDecodeRunner

    case = make_case(
        rows, context, ratio, layout, candidate=candidate, reindex=reindex, qlen=qlen
    )
    if mixed:
        lengths = torch.tensor(
            [
                1,
                511,
                512,
                513,
                8191,
                8192,
                8193,
                16383,
                16384,
                16385,
                32767,
                65536,
                81920,
                4096,
                32768,
                1024,
                0,
            ],
            dtype=torch.int32,
            device="cuda",
        )
        case.step.visible[ratio].copy_(lengths)
        case.step.positions.copy_((lengths.long() * ratio - 1).clamp_min(0))
    r = case.runner
    assert r.prepare(rows)
    r.active = True

    def run():
        r.begin_step(rows, case.cache, case.step)
        assert r.run(
            case.indexer,
            case.hidden,
            case.qr,
            case.qrs,
            case.cache,
            case.step,
            case.rope,
        )

    def check():
        q, _weights, _selection = case.native()
        proxy = SimpleNamespace(
            index_fp4=True, index_layout=layout, iq=r.iq[:rows], iq_scale=r.iqs[:rows]
        )
        mono_q = V41MonoDecodeRunner.index_query(proxy)
        # Projection FP32 sum order may move a BF16 midpoint. Scoring itself
        # is compared on the same quantized query as well as end-to-end.
        assert torch.equal(mono_q[1].view(torch.uint8), q[1].view(torch.uint8))
        mismatch = (mono_q[0] != q[0]).sum().item()
        assert mismatch / q[0].numel() < 0.002
        # The native multi-row BF16 GEMM rounds partial sums, whereas Mono's
        # GEMV accumulates FP32. Qualify IW against FP32 accumulation directly,
        # then isolate the scorer with identical query and head weights.
        reference_iw = (
            case.hidden[0].float() @ case.indexer.weights_proj.weight.float().T
        ).bfloat16().float() / 64
        torch.testing.assert_close(r.iw[:rows], reference_iw, rtol=0.008, atol=1e-6)
        _, _, (same_q, same_chosen) = case.native(mono_q, (r.iw[:rows] * 64).bfloat16())
        got = case.step.selected[case.indexer.spec.layer_id][0]
        assert torch.equal(got, same_q), (got != same_q).sum().item()
        if same_chosen is not None:
            assert torch.equal(
                case.step.candidates[case.indexer.spec.layer_id], same_chosen
            )

    run()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    case.hidden.mul_(0.8)
    graph.replay()
    check()


@pytest.mark.parametrize(
    "ratio,layout,candidate", [(2, "mfma", False), (1, "preshuffle", True)]
)
def test_mono_indexer_mixed_selection_depths(ratio, layout, candidate):
    """Small and large requests share dispatch while retaining separate trees."""
    test_standalone_mono_matches_native_and_graph(
        17, 163840, ratio, layout, candidate, False, 1, mixed=True
    )


@pytest.mark.parametrize("rows,context,ratio", [(1, 1024, 1), (8, 2048, 2)])
def test_short_full_selection_keeps_fp4_score_staging_in_bounds(rows, context, ratio):
    test_standalone_mono_matches_native_and_graph(
        rows, context, ratio, "mfma", False, False, 1
    )
