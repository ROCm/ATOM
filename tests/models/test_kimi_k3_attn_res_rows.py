# SPDX-License-Identifier: MIT
"""A Kimi-K3 MoE decoder layer on simulated TP8 ranks. On a row-sharded step
o_proj's partial sum is reduce-scattered, the attention residual before the
MoE runs on each rank's rows and the MoE input is all-gathered; the layer's
summed output still matches the all-reduced base."""

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")

from atom.model_ops import kimi_k3 as kimi_k3_ops
from atom.model_ops import module_dispatch_ops
from atom.models import qwen4_exp
from atom.models.kimi_k3 import model as kimi
from atom.utils.tbo import ubatching
from tests.models.test_kimi_k3_moe_row_tail import (  # noqa: F401
    HIDDEN,
    WORLD,
    _block,
    _weights,
    ranks,
    split_op_on_cpu,
    stub_block_parts,
)

BLOCK = 4  # layer 4 closes an attention-residual block, layer 5 does not
_events = threading.local()


def _logs(name, fn):
    """`fn`, recording `(name, rows)` in the calling rank's log."""

    def run(x, *args, **kwargs):
        _events.log.append((name, x.shape[0]))
        return fn(x, *args, **kwargs)

    return run


def _torch_attn_res(
    prefix,
    block_residual,
    score_weight,
    eps,
    add_hidden=None,
    out_norm_weight=None,
    out_eps=1e-6,
    add_hidden2=None,
    quant_dtype=None,
):
    """`apply_attn_res` in FP32 torch: the block rows and the prefix, mixed by
    the softmax of their RMS-normed scores, then RMS-normed."""
    assert quant_dtype is None
    ps = prefix.float()
    for add in (add_hidden, add_hidden2):
        if add is not None:
            ps = ps + add.float()
    cands = torch.cat([block_residual.float(), ps[:, None]], 1)
    scores = (cands * score_weight).sum(-1) * torch.rsqrt(cands.pow(2).mean(-1) + eps)
    y = (scores.softmax(-1)[..., None] * cands).sum(1)
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + out_eps)
    y = y * out_norm_weight.float()
    return y.to(prefix.dtype), prefix if add_hidden is None else ps.to(prefix.dtype)


class _Attention(nn.Module):
    """This rank's share of a KDA mixer's output, all-reduced unless the layer
    turns `o_proj.reduce_results` off, as RowParallelLinear does."""

    fuse_input_norm_quant = False
    input_quant_prefix = "self_attn.in_proj"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.o_proj = SimpleNamespace(reduce_results=True)

    def forward(self, x):
        _events.log.append(("attention", x.shape[0]))
        out = self.share * torch.tanh(x @ self.w)
        if self.o_proj.reduce_results:
            out = kimi.tensor_model_parallel_all_reduce(out)
        return out


@pytest.fixture
def logged(ranks, monkeypatch, split_op_on_cpu):  # noqa: F811
    """`logged(fn)` runs `fn(rank)` on the simulated ranks and returns, per
    rank, its result and the log of its collectives, attn_res calls and
    sub-module calls, each with its row count."""
    group = qwen4_exp.get_tp_group()
    logged_group = SimpleNamespace(
        reduce_scatter_tensor=_logs("reduce_scatter", group.reduce_scatter_tensor),
        all_gather=_logs("all_gather", group.all_gather),
    )
    monkeypatch.setattr(qwen4_exp, "get_tp_group", lambda: logged_group)
    all_reduce = _logs("all_reduce", kimi.tensor_model_parallel_all_reduce)
    monkeypatch.setattr(kimi, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(
        kimi_k3_ops, "apply_attn_res", _logs("attn_res", _torch_attn_res)
    )
    monkeypatch.setattr(
        torch.ops.aiter,
        "kimi_k3_post_attention_forward",
        module_dispatch_ops.kimi_k3_post_attention_forward,
    )

    def run(fn):
        def on_rank(rank):
            _events.log = []
            return fn(rank), _events.log

        return ranks(on_rank)

    return run


def _layers(layer_num):
    """One MoE decoder layer per rank, built by `KimiDecoderLayer.__init__`
    around `_block`'s block and a stub attention, every stub logging, with
    dual-stream off as the harness runs."""
    w = _weights()
    gen = torch.Generator().manual_seed(layer_num)
    w_attn = (torch.randn(HIDDEN, HIDDEN, generator=gen) / HIDDEN**0.5).bfloat16()
    score_weights = torch.randn(2, HIDDEN, generator=gen)
    norm_weights = (1 + 0.1 * torch.randn(2, HIDDEN, generator=gen)).bfloat16()
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
        num_experts=4,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        kimi_kda_layers=[layer_num],
        attn_res_block_size=BLOCK,
    )
    layers = []
    for rank in range(WORLD):
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
            block = _block(w, rank, [])
            for name in (
                "gate",
                "routed_expert_down_proj",
                "experts",
                "routed_expert_norm",
                "routed_expert_up_proj",
                "shared_experts",
            ):
                stub = getattr(block, name)
                stub.fn = _logs(name, stub.fn)
            patch.setattr(kimi, "KimiSparseMoeBlock", lambda *a, b=block, **k: b)
            patch.setattr(kimi, "KimiKDAAttention", _Attention)
            layer = kimi.KimiDecoderLayer(
                SimpleNamespace(hf_config=config, quant_config=None),
                prefix=f"rank{rank}.layers.{layer_num}",
                layer_num=layer_num,
            )
        layer.self_attn.w = w_attn
        layer.self_attn.share = (rank + 1) / (WORLD * (WORLD + 1) / 2)
        layer.self_attention_attn_res.score_weight = score_weights[0]
        layer.mlp_attn_res.score_weight = score_weights[1]
        layer.input_layernorm.weight = norm_weights[0]
        layer.post_attention_layernorm.weight = norm_weights[1]
        layers.append(layer)
    return layers


def _inputs(tokens):
    """Prefix, a two-row block bank and the previous layer's two addends."""
    gen = torch.Generator().manual_seed(tokens)
    x, add, add2 = torch.randn(3, tokens, HIDDEN, generator=gen).bfloat16()
    bank = torch.randn(tokens, 2, HIDDEN, generator=gen).bfloat16()
    return x, bank, add, add2


def _base_forward(layer, x, bank, add, add2):
    """`KimiDecoderLayer.forward` as it was before the op: the attention
    reduces o_proj and the layer runs mlp_attn_res and the FFN itself."""
    o_proj = layer.self_attn.o_proj
    reduce_results, o_proj.reduce_results = o_proj.reduce_results, True
    try:
        h, prefix = layer.self_attention_attn_res(x, bank, add, add2)
        bank, prefix = layer.self_attention_attn_res.maybe_close_block(prefix, bank)
        h, prefix = layer.mlp_attn_res(prefix, bank, layer.self_attn(h))
        h, shared = layer._ffn(h)
    finally:
        o_proj.reduce_results = reduce_results
    return prefix, h, shared, bank


@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_row_sharded_layer_sums_to_the_base(logged, monkeypatch, tokens, layer_num):
    layers = _layers(layer_num)
    inputs = _inputs(tokens)
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "100000")
    base = logged(lambda rank: _base_forward(layers[rank], *inputs))
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    rows = logged(lambda rank: layers[rank](None, *inputs))

    chunk = -(-tokens // WORLD)
    padded = chunk * WORLD
    for (want, _), (got, log) in zip(base, rows):
        # mlp_attn_res and the routed tail on this rank's rows, no all-reduce
        assert log == [
            ("attn_res", tokens),
            ("attention", tokens),
            ("reduce_scatter", padded),
            ("attn_res", chunk),
            ("all_gather", chunk),
            ("gate", tokens),
            ("routed_expert_down_proj", tokens),
            ("experts", tokens),
            ("reduce_scatter", padded),
            ("routed_expert_norm", chunk),
            ("routed_expert_up_proj", chunk),
            ("shared_experts", tokens),
            ("reduce_scatter", padded),
            ("all_gather", chunk),
        ]
        residual, routed, shared, bank = got
        assert not routed.any() and not shared.any()
        assert torch.equal(bank, want[3])
        # The row-sharded sum rounds to BF16 twice, the reference once.
        exact = sum(t.float() for t in want[:3])
        bound = 3 * 2**-8 * sum(t.float().abs() for t in want[:3])
        assert ((residual.float() - exact).abs() <= bound).all()
        assert torch.equal(kimi.KimiDecoderLayer.aux_hidden_state(got), residual)


@pytest.mark.parametrize(
    "threshold,capturing,tbo_step",
    [("13", False, False), ("4", True, False), ("4", False, True)],
    ids=["at-threshold", "graph-capture", "tbo-step"],
)
@pytest.mark.parametrize("layer_num", [4, 5], ids=["block-closes", "mid-block"])
def test_base_kernels_launch_in_order(
    logged, monkeypatch, threshold, capturing, tbo_step, layer_num
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", threshold)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(ubatching, "tbo_active", lambda: tbo_step)
    layers = _layers(layer_num)
    inputs = _inputs(13)
    base = logged(lambda rank: _base_forward(layers[rank], *inputs))
    got = logged(lambda rank: layers[rank](None, *inputs))
    for (want, want_log), (out, log) in zip(base, got):
        assert log == want_log
        assert log.count(("all_reduce", 13)) == 3
        assert all(torch.equal(a, b) for a, b in zip(out, want))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@torch.inference_mode()
def test_folded_zeros_are_added_without_loading(monkeypatch):
    """`apply_attn_res` takes a row-sharded layer's zero addends as zero
    without loading them, still returns a fresh prefix, and releases them."""
    from atom.model_ops.kimi_k3 import attention_residual as attn_res

    gen = torch.Generator().manual_seed(7)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).bfloat16().cuda()

    prefix, bank, norm = randn(13, 7168), randn(13, 3, 7168), randn(7168)
    score_weight = randn(7168).float() * 0.01
    kernel, flags = attn_res._attn_res_fused_kernel, []

    class _Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                flags.append(
                    (kwargs["DO_ADD"], kwargs["DO_ADD2"], kwargs["WRITE_PREF"])
                )
                return kernel[grid](*args, **kwargs)

            return launch

    monkeypatch.setattr(attn_res, "_attn_res_fused_kernel", _Kernel())

    def call(add, add2):
        return attn_res._apply_attn_res_impl(
            prefix, bank, score_weight, 1e-6, add, norm, 1e-6, add2
        )

    folded = call(*attn_res.folded_zeros_like(prefix))
    loaded = call(torch.zeros_like(prefix), torch.zeros_like(prefix))
    assert flags == [(False, False, True), (True, True, True)]
    assert not attn_res._folded_zeros
    assert folded[1].data_ptr() != prefix.data_ptr()
    assert all(torch.equal(a, b) for a, b in zip(folded, loaded))
