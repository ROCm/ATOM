# SPDX-License-Identifier: MIT
"""Kimi row-sharded residual and MoE tail on CPU-simulated TP ranks."""

import ast
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from torch import nn

kimi = module_dispatch_ops = kimi_k3_ops = sequence_parallel = ubatching = None


@pytest.fixture(scope="module", autouse=True)
def imported_modules():
    global kimi, module_dispatch_ops, kimi_k3_ops, sequence_parallel, ubatching
    with atom_without_aiter() as load:
        kimi = load("atom.models.kimi_k3.model")
        module_dispatch_ops = load("atom.model_ops.module_dispatch_ops")
        kimi_k3_ops = load("atom.model_ops.kimi_k3")
        sequence_parallel = load("atom.distributed.sequence_parallel")
        ubatching = load("atom.utils.tbo.ubatching")
        yield
    kimi = module_dispatch_ops = kimi_k3_ops = sequence_parallel = ubatching = None


WORLD, HIDDEN, LATENT, EXPERTS = 8, 16, 8, 4


class _Collectives:
    """In-process TP group: one thread per rank, collectives meet at a barrier."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(WORLD, timeout=30)
        self.slots = [None] * WORLD

    def _exchange(self, rank, x):
        self.slots[rank] = x
        self.barrier.wait()
        inputs = list(self.slots)
        self.barrier.wait()
        return inputs

    def total(self, rank, x):
        # Round the exact sum once for both all-reduce and reduce-scatter.
        return torch.stack(self._exchange(rank, x)).double().sum(0).to(x.dtype)

    def gather(self, rank, x):
        return torch.cat(self._exchange(rank, x))


@pytest.fixture
def ranks(monkeypatch):
    local = threading.local()

    def reduce_scatter_tensor(x, use_custom=False, dim=0):
        assert dim == 0 and x.shape[0] % WORLD == 0
        return local.comm.total(local.rank, x).chunk(WORLD)[local.rank].clone()

    def all_gather(x, use_custom=False, dim=0):
        assert dim == 0
        return local.comm.gather(local.rank, x)

    group = SimpleNamespace(
        reduce_scatter_tensor=reduce_scatter_tensor, all_gather=all_gather
    )
    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: group)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_rank", lambda: local.rank)
    monkeypatch.setattr(
        kimi,
        "tensor_model_parallel_all_reduce",
        lambda x: local.comm.total(local.rank, x),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def run(fn):
        comm = _Collectives()

        def on_rank(rank):
            local.rank, local.comm = rank, comm
            return fn(rank)

        with ThreadPoolExecutor(WORLD) as pool:
            return list(pool.map(on_rank, range(WORLD)))

    return run


class _Stub(nn.Module):
    """Stands in for a sub-module class; the test sets `fn` after `__init__`."""

    allows_concurrent_shared_stream = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args):
        return self.fn(*args)


@pytest.fixture(autouse=True)
def stub_block_parts(monkeypatch):
    monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", "0")
    empty_like = torch.empty_like

    def poisoned_empty_like(*args, **kwargs):
        out = empty_like(*args, **kwargs)
        return out.fill_(float("nan")) if out.is_floating_point() else out

    monkeypatch.setattr(torch, "empty_like", poisoned_empty_like)
    for name in ("ReplicatedLinear", "FusedMoE", "KimiMLP", "RMSNorm"):
        monkeypatch.setattr(kimi, name, _Stub)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: WORLD)
    config = SimpleNamespace(
        enable_tbo=False,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(kimi, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(module_dispatch_ops, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(
        torch.ops.aiter,
        "maybe_dual_stream_split_forward",
        module_dispatch_ops.maybe_dual_stream_split_forward,
    )
    return config


def _weights():
    gen = torch.Generator().manual_seed(3)
    shapes = {
        "gate": (HIDDEN, EXPERTS),
        "down": (HIDDEN, LATENT),
        "up": (LATENT, HIDDEN),
        "shared": (HIDDEN, HIDDEN),
    }
    return {
        name: torch.randn(*shape, generator=gen).to(torch.bfloat16)
        for name, shape in shapes.items()
    }


def _block(w, rank, up_rows):
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_token=2,
        routed_expert_hidden_size=LATENT,
        moe_intermediate_size=HIDDEN,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        num_shared_experts=1,
        latent_moe_use_norm=True,
        rms_norm_eps=1e-6,
    )
    block = kimi.KimiSparseMoeBlock(
        config, prefix=f"rank{rank}.mlp", alt_stream=object()
    )
    share = (rank + 1) / (WORLD * (WORLD + 1) / 2)
    block.gate.fn = lambda x: x @ w["gate"]
    block.routed_expert_down_proj.fn = lambda x: x @ w["down"]
    block.experts.fn = lambda x, logits: share * torch.tanh(
        x * logits.sum(-1, keepdim=True)
    )
    block.shared_experts.fn = lambda x: share * torch.tanh(x @ w["shared"])
    # Normalize in FP32 so CPU BF16 rounding is independent of row count.
    block.routed_expert_norm.fn = lambda x: (
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    ).to(x.dtype)

    def up_proj(x):
        up_rows.append(x.shape[0])
        return x @ w["up"]

    block.routed_expert_up_proj.fn = up_proj
    return block


def _base_and_op_target(ranks, tokens, target=None):
    w = _weights()
    gen = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)
    target = target or (lambda block, x: block.single_stream_split_moe_forward(x))

    def on_rank(rank):
        up_rows = []
        block = _block(w, rank, up_rows)
        base = block.split_moe_forward(x)
        return base, target(block, x), up_rows

    return ranks(on_rank)


@pytest.mark.parametrize("tokens", [8, 13, 1027])
def test_row_sharded_pair_sums_to_the_base_pair(ranks, monkeypatch, tokens):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    results = _base_and_op_target(ranks, tokens)
    for (routed, shared), (summed, zeros), up_rows in results:
        assert up_rows == [tokens, -(-tokens // WORLD)]
        assert summed.shape == zeros.shape == (tokens, HIDDEN)
        assert not zeros.any()
        assert zeros.data_ptr() != summed.data_ptr()
        expected = (routed.float() + shared.float()).to(summed.dtype)
        torch.testing.assert_close(summed, expected)


def test_forward_with_dual_stream_off_takes_the_row_sharded_tail(ranks, monkeypatch):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    tokens = 13
    for (routed, shared), (summed, zeros), up_rows in _base_and_op_target(
        ranks, tokens, lambda block, x: block.forward(x)
    ):
        assert up_rows == [tokens, -(-tokens // WORLD)]
        assert not zeros.any()
        assert zeros.data_ptr() != summed.data_ptr()
        expected = (routed.float() + shared.float()).to(summed.dtype)
        torch.testing.assert_close(summed, expected)


def test_forward_with_tbo_enabled_keeps_the_base_pair(
    ranks, monkeypatch, stub_block_parts
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    stub_block_parts.enable_tbo = True
    tokens = 13
    for (routed, shared), (routed_out, shared_out), up_rows in _base_and_op_target(
        ranks, tokens, lambda block, x: block.forward(x)
    ):
        assert up_rows == [tokens, tokens]
        assert torch.equal(routed_out, routed) and torch.equal(shared_out, shared)


@pytest.mark.parametrize(
    "floor,sharded",
    [(None, False), ("14", False), ("13", True)],
    ids=["default-floor", "below-floor", "at-floor"],
)
def test_row_floor(ranks, monkeypatch, floor, sharded):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    if floor is None:
        monkeypatch.delenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS")
    else:
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", floor)
    tokens = 13
    for (routed, shared), (first, second), up_rows in _base_and_op_target(
        ranks, tokens, lambda block, x: block.forward(x)
    ):
        assert up_rows == [tokens, -(-tokens // WORLD) if sharded else tokens]
        if not sharded:
            assert torch.equal(first, routed) and torch.equal(second, shared)


@pytest.mark.parametrize(
    "threshold,capturing,tbo_step",
    [
        ("13", False, False),
        ("14", False, False),
        ("4", True, False),
        ("4", False, True),
    ],
    ids=["at-threshold", "below-threshold", "graph-capture", "tbo-step"],
)
def test_base_pair_kept_bit_for_bit(ranks, monkeypatch, threshold, capturing, tbo_step):
    tokens = 13
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", threshold)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(ubatching, "tbo_active", lambda: tbo_step)
    results = _base_and_op_target(ranks, tokens)
    for (routed, shared), (routed_out, shared_out), up_rows in results:
        assert up_rows == [tokens, tokens]
        assert torch.equal(routed_out, routed) and torch.equal(shared_out, shared)


BLOCK = 4  # layer 4 closes an attention-residual block, layer 5 does not
_events = threading.local()


def _logs(name, fn):

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
    num_blocks=None,
):
    """Attention residual and output RMS norm in FP32 torch."""
    assert quant_dtype is None
    if num_blocks is not None:
        block_residual = block_residual[:, :num_blocks]
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
def logged(ranks, monkeypatch):
    group = sequence_parallel.get_tp_group()
    logged_group = SimpleNamespace(
        reduce_scatter_tensor=_logs("reduce_scatter", group.reduce_scatter_tensor),
        all_gather=_logs("all_gather", group.all_gather),
    )
    monkeypatch.setattr(sequence_parallel, "get_tp_group", lambda: logged_group)
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
    gen = torch.Generator().manual_seed(tokens)
    x, add, add2 = torch.randn(3, tokens, HIDDEN, generator=gen).bfloat16()
    bank = torch.randn(tokens, 2, HIDDEN, generator=gen).bfloat16()
    return x, bank, add, add2


def _base_forward(layer, x, bank, add, add2):
    """Replicated attention reduction, residual and FFN before the new op."""
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
    base = logged(
        lambda rank: _base_forward(layers[rank], *(x.clone() for x in inputs))
    )
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "4")
    rows = logged(lambda rank: layers[rank](None, *(x.clone() for x in inputs)))

    chunk = -(-tokens // WORLD)
    padded = chunk * WORLD
    for (want, _), (got, log) in zip(base, rows):
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
        assert routed is None and shared is None
        assert torch.equal(bank, want[3])
        # The row-sharded sum rounds to BF16 twice, the reference once.
        exact = sum(t.float() for t in want[:3])
        bound = 3 * 2**-8 * sum(t.float().abs() for t in want[:3])
        assert ((residual.float() - exact).abs() <= bound).all()
        assert torch.equal(kimi.KimiDecoderLayer.aux_hidden_state(got), residual)
        cloned = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in got)
        assert torch.equal(kimi.KimiDecoderLayer.aux_hidden_state(cloned), residual)


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
    base = logged(
        lambda rank: _base_forward(layers[rank], *(x.clone() for x in inputs))
    )
    got = logged(lambda rank: layers[rank](None, *(x.clone() for x in inputs)))
    for (want, want_log), (out, log) in zip(base, got):
        assert log == want_log
        assert log.count(("all_reduce", 13)) == 3
        assert all(torch.equal(a, b) for a, b in zip(out, want))


@pytest.mark.parametrize("floor", [None, "14"], ids=["default-floor", "below-floor"])
def test_below_row_floor_all_reduces(logged, monkeypatch, floor):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    if floor is None:
        monkeypatch.delenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS")
    else:
        monkeypatch.setenv("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", floor)
    layers = _layers(BLOCK + 1)
    inputs = _inputs(13)
    base = logged(
        lambda rank: _base_forward(layers[rank], *(x.clone() for x in inputs))
    )
    got = logged(lambda rank: layers[rank](None, *(x.clone() for x in inputs)))
    for (want, want_log), (out, log) in zip(base, got):
        assert log == want_log
        assert all(torch.equal(a, b) for a, b in zip(out, want))


@pytest.mark.parametrize(
    "tokens,threshold,floor,expected",
    [
        (8191, None, None, False),
        (8192, None, None, True),
        (8192, "8192", None, False),
        (8193, "8192", None, True),
    ],
)
def test_row_gate_intersects_floor_and_strict_stream_threshold(
    ranks, monkeypatch, tokens, threshold, floor, expected
):
    for name, value in [
        ("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", threshold),
        ("ATOM_KIMI_K3_MOE_ROW_SHARD_MIN_TOKENS", floor),
    ]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert (
        kimi.KimiSparseMoeBlock.takes_row_tail(SimpleNamespace(tp_size=WORLD), tokens)
        is expected
    )


@pytest.mark.parametrize(
    "world,latent,shared,tbo,eligible",
    [
        (WORLD, LATENT, 1, False, True),
        (1, LATENT, 1, False, False),
        (WORLD, None, 1, False, False),
        (WORLD, LATENT, 0, False, False),
        (WORLD, LATENT, 1, True, False),
    ],
)
def test_constructor_limits_row_dispatch(
    monkeypatch, stub_block_parts, world, latent, shared, tbo, eligible
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "0")
    config = _block(_weights(), 0, []).config
    config.routed_expert_hidden_size, config.num_shared_experts = latent, shared
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: world)
    stub_block_parts.enable_tbo = tbo
    block = kimi.KimiSparseMoeBlock(config, prefix="scope", alt_stream=object())
    assert block.row_shardable is eligible and block._use_dispatch_op is eligible


@pytest.mark.parametrize("shared_stream_allowed", [True, False])
@pytest.mark.parametrize("world", [1, WORLD])
def test_shared_stream_refusal_preserves_row_dispatch(
    monkeypatch, stub_block_parts, shared_stream_allowed, world
):
    monkeypatch.setenv("ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD", "32")
    monkeypatch.setattr(_Stub, "allows_concurrent_shared_stream", shared_stream_allowed)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: world)
    block = _block(_weights(), 0, [])

    assert block._use_dual_stream is shared_stream_allowed
    assert block.row_shardable is (world > 1)
    dispatch = shared_stream_allowed or world > 1
    assert block._use_dispatch_op is dispatch
    registered = stub_block_parts.compilation_config.static_forward_context
    assert (block.prefix in registered) is dispatch
    if dispatch:
        assert registered[block.prefix] is block


def test_mono_fallback_preserves_typed_forward_outputs():
    path = Path(kimi.__file__).parent / "mono/runner.py"
    fn = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_fallback"
    )
    namespace = {}
    exec(  # noqa: S102 - execute only the checked-in fallback method
        compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace
    )
    prefix, columns, bank = (
        torch.ones(2, 4),
        torch.full((2, 2), 2.0),
        torch.zeros(2, 1, 4),
    )
    pending = kimi.ColumnTail(columns, "gather", 2)
    calls = []

    def forward(*args):
        calls.append(args)
        return prefix, pending, None, bank

    runner = SimpleNamespace(
        inner=SimpleNamespace(layers=[SimpleNamespace(forward=forward)])
    )
    residual, add, add2 = namespace["_fallback"](
        runner, SimpleNamespace(layer_id=0), None, prefix, bank, None, None
    )
    assert len(calls) == 1 and calls[0][1] is prefix and calls[0][2] is bank
    assert residual is prefix and add is pending and add2 is None
