# SPDX-License-Identifier: MIT
"""Checkpoint-time Key/V sharding and context-independent Engram dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("aiter")
from atom.model_ops import linear
from atom.model_ops.engram.device import layer
from atom.quant_spec import LayerQuantConfig


@pytest.fixture
def tp_group(monkeypatch):
    group = SimpleNamespace(world_size=2, rank_in_group=0)
    monkeypatch.setattr(linear, "get_tp_group", lambda: group)
    monkeypatch.setattr(layer, "get_tp_group", lambda: group)
    return group


def config(group_rows=32):
    from aiter import QuantType

    spec = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=torch.float8_e4m3fn,
        weight_block_size=(group_rows, 32),
    )
    return SimpleNamespace(get_layer_quant_config=lambda _: spec, online_quant=False)


def make_op(hc=4, dim=64, width=96, group_rows=32):
    return layer.EngramOp(
        1,
        hidden_size=dim,
        engram_hidden_size=width,
        hc_mult=hc,
        quant_config=config(group_rows) if group_rows else None,
    )


def checkpoint(op):
    rows, cols = (op.hc_mult + 1) * op.hidden_size, op.engram_hidden_size
    # V and every Key branch have distinct row patterns; scales vary on both
    # axes so selecting the right weight with the wrong scale cannot pass.
    values = torch.arange(rows * cols).reshape(rows, cols)
    weight = (values.remainder(113).float() - 56).to(torch.float8_e4m3fn)
    group_rows = op.wkv.native_a8_group_rows
    scales = torch.arange(rows // group_rows * (cols // 32)).reshape(
        rows // group_rows, cols // 32
    )
    scales = (scales.remainder(16) + 112).to(torch.uint8).view(torch.float8_e8m0fnu)
    return weight, scales


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("tp", [2, 4])
def test_full_checkpoint_loads_complete_keys_and_replicated_value(
    monkeypatch, tp_group, tp
):
    tp_group.world_size = tp
    original_init = linear.ReplicatedLinear.__init__
    created = {}

    def capture_parameters(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.update(
            (name, (param, param.weight_loader, param.weight_loader_process))
            for name, param in self.named_parameters()
        )

    monkeypatch.setattr(linear.ReplicatedLinear, "__init__", capture_parameters)
    for rank in range(tp):
        tp_group.rank_in_group = rank
        op = make_op()
        wkv = op.wkv
        assert wkv.local_hc == 4 // tp
        rows = (4 // tp + 1) * 64
        assert wkv.output_size == rows
        assert wkv.output_partition_sizes == [rows]
        assert set(dict(op.named_parameters())) == {
            "k_weight",
            "q_weight",
            "wkv.weight",
            "wkv.weight_scale",
        }
        weight, scales = checkpoint(op)
        for name, source, block in (
            ("weight", weight, 1),
            ("weight_scale", scales, 32),
        ):
            param = getattr(wkv, name)
            original, loader, processor = created[name]
            assert param is original
            assert param.weight_loader is loader
            assert param.weight_loader_process is processor
            assert param.shape == (rows // block, 96 // (32 if block == 32 else 1))
            assert (
                param.untyped_storage().nbytes() == param.numel() * param.element_size()
            )
            spy = Mock(wraps=processor)
            monkeypatch.setattr(param, "weight_loader_process", spy)
            loader(param, source)
            spy.assert_called_once()
            assert spy.call_args.args[1].shape == param.shape
            key_rows = 4 // tp * 64 // block
            start = rank * key_rows
            assert_bytes(param[:key_rows], source[start : start + key_rows])
            value = source[4 * 64 // block :]
            assert_bytes(
                param[key_rows:], torch.roll(value, -wkv.value_start // block, 0)
            )


@pytest.mark.parametrize("direct", [False, True], ids=["production", "direct-helper"])
@pytest.mark.parametrize("tp", [2, 4])
def test_loading_shuffles_local_weight_once(monkeypatch, tp_group, tp, direct):
    from aiter.ops.shuffle import shuffle_weight

    tp_group.world_size, tp_group.rank_in_group = tp, tp - 1
    monkeypatch.setattr(linear, "weight_is_stored_preshuffled", lambda *a, **k: True)
    shuffle = Mock(wraps=linear.shuffle_weights)
    monkeypatch.setattr(linear, "shuffle_weights", shuffle)
    op = make_op()
    weight, scales = checkpoint(op)
    k = torch.arange(256).reshape(4, 64).float() / 256
    q = k + 1
    if direct:
        op.load_checkpoint_weights(weight, k, q, scales)
    else:
        for name, source in (("weight", weight), ("weight_scale", scales)):
            param = getattr(op.wkv, name)
            param.weight_loader(param, source)
        op.k_weight.data.copy_(k)
        op.q_weight.data.copy_(q)
        # The production loader traverses parent first, then linear child.
        op.process_weights_after_loading()
        shuffle.assert_not_called()
        op.wkv.process_weights_after_loading()
    shuffle.assert_called_once_with(op.wkv.weight)
    torch.testing.assert_close(op.gate_weight, q * k)
    key_rows = 4 // tp * 64
    start = (tp - 1) * key_rows
    local = torch.cat(
        (
            weight[start : start + key_rows],
            torch.roll(weight[256:], -op.wkv.value_start, 0),
        )
    )
    assert_bytes(op.wkv.weight, shuffle_weight(local, layout=(16, 16)))
    assert getattr(op.wkv.weight, "is_shuffled", False)
    assert_bytes(op.wkv.weight_scale[: key_rows // 32], scales[start // 32 : 8])
    assert_bytes(
        op.wkv.weight_scale[key_rows // 32 :],
        torch.roll(scales[8:], -op.wkv.value_start // 32, 0),
    )


@pytest.mark.parametrize("tp", [2, 4])
def test_small_projection_uses_prefix_of_the_only_weight(monkeypatch, tp_group, tp):
    from atom.model_ops import blockscale

    tp_group.world_size = tp
    monkeypatch.setattr(linear, "weight_is_stored_preshuffled", lambda *a, **k: True)
    for rank in range(tp):
        tp_group.rank_in_group = rank
        op = make_op(dim=128)
        weight, scales = checkpoint(op)
        op.load_checkpoint_weights(
            weight, torch.ones(4, 128), torch.ones(4, 128), scales
        )
        stored = op.wkv.weight
        assert stored.untyped_storage().nbytes() == stored.numel()
        assert set(dict(op.wkv.named_parameters())) == {"weight", "weight_scale"}
        x = torch.randn(12, 96)
        projection = Mock(return_value=torch.empty(12, 5 * 128 // tp))
        monkeypatch.setattr(blockscale, "native_quant_linear", projection)
        op.wkv(x)
        _, prefix, scale_prefix = projection.call_args.args
        assert prefix.shape == (5 * 128 // tp, 96)
        assert prefix.is_contiguous() and scale_prefix.is_contiguous()
        assert (
            prefix.untyped_storage().data_ptr() == stored.untyped_storage().data_ptr()
        )
        assert (
            scale_prefix.untyped_storage().data_ptr()
            == op.wkv.weight_scale.untyped_storage().data_ptr()
        )
        assert projection.call_args.kwargs["weight_preshuffled"]
        from aiter.ops.shuffle import shuffle_weight

        keys = 4 * 128 // tp
        value_start = 4 * 128 + rank * (128 // tp)
        expected = torch.cat(
            (
                weight[rank * keys : (rank + 1) * keys],
                weight[value_start : value_start + 128 // tp],
            )
        )
        assert_bytes(prefix, shuffle_weight(expected, (16, 16)))


@pytest.mark.parametrize("name", ["weight", "weight_scale"])
@pytest.mark.parametrize("bad_shape", ["local", "transpose", "flatten", "short"])
def test_loader_rejects_non_checkpoint_shapes(tp_group, name, bad_shape):
    op = make_op()
    sources = dict(zip(("weight", "weight_scale"), checkpoint(op)))
    param, source = getattr(op.wkv, name), sources[name]
    bad = {
        "local": torch.empty_like(param),
        "transpose": source.T,
        "flatten": source.flatten(),
        "short": source[:-1],
    }[bad_shape]
    with pytest.raises(ValueError, match="full checkpoint shape"):
        param.weight_loader(param, bad)


@pytest.mark.parametrize(
    "bad", ["local_scale", "missing_scale", "scale_dtype", "weight_dtype"]
)
def test_direct_helper_validates_full_native_layout(monkeypatch, tp_group, bad):
    op = make_op()
    weight, scales = checkpoint(op)
    if bad == "local_scale":
        scales = torch.empty_like(op.wkv.weight_scale)
    elif bad == "missing_scale":
        scales = None
    elif bad == "scale_dtype":
        scales = scales.view(torch.uint8)
    else:
        weight = weight.float()
    loader = Mock(side_effect=AssertionError("invalid source must not be copied"))
    monkeypatch.setattr(op.wkv, "weight_loader", loader)
    with pytest.raises(ValueError, match="matching weight and scale layout"):
        op.load_checkpoint_weights(weight, torch.ones(4, 64), torch.ones(4, 64), scales)
    loader.assert_not_called()


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize(
    "shape", [(0,), (1,), (11,), (12,), (13,), (16,), (128,), (2, 3), (2, 6), (2, 8)]
)
def test_dispatches_local_projection_without_forward_context(
    monkeypatch, tp_group, tp, shape
):
    tp_group.world_size, tp_group.rank_in_group = tp, tp - 1
    op = make_op(dim=128)
    op.process_weights_after_loading()
    hidden = torch.randn(*shape, 4, 128)
    embeddings = torch.randn(*shape, 96)
    mask = torch.rand(shape) > 0.25
    small = 0 < embeddings.numel() // 96 <= 12
    kv = torch.randn(*shape, 5 * 128 // tp if small else (4 // tp + 1) * 128)
    projection = Mock(return_value=kv)
    monkeypatch.setattr(op.wkv, "forward", projection)
    output = torch.randn_like(hidden)
    gate = Mock(return_value=output)
    monkeypatch.setattr(
        layer, "engram_post_wkv" if small else "engram_post_wkv_tp", gate
    )
    monkeypatch.setattr(
        layer,
        "engram_post_wkv_tp" if small else "engram_post_wkv",
        Mock(side_effect=AssertionError),
    )
    # Dispatch depends on total tensor rows, including graph padding, with no
    # forward context lookup or runtime weight packing.
    monkeypatch.setattr(
        layer, "get_forward_context", Mock(side_effect=AssertionError), raising=False
    )
    monkeypatch.setattr(torch, "cat", Mock(side_effect=AssertionError))
    gathered = torch.randn(embeddings.numel() // 96, 5 * 128)
    tp_group.all_gather = (
        Mock(return_value=gathered) if small else Mock(side_effect=AssertionError)
    )
    assert op(hidden, embeddings, mask) is output
    projection.assert_called_once_with(embeddings)
    if small:
        tp_group.all_gather.assert_called_once()
        assert tp_group.all_gather.call_args.kwargs == {"dim": 1, "use_custom": True}
        assert torch.equal(
            tp_group.all_gather.call_args.args[0], kv.reshape(-1, 5 * 128 // tp)
        )
        assert torch.equal(gate.call_args.args[1], gathered.reshape(*shape, 5 * 128))
        assert gate.call_args.kwargs == {"kv_tp_size": tp}
    else:
        gate.assert_called_once_with(
            hidden,
            kv,
            op.gate_weight,
            tp_group,
            mask,
            op.norm_eps,
            value_start=op.wkv.value_start,
        )
        tp_group.all_gather.assert_not_called()


@pytest.mark.parametrize(
    "tp,hc,dim,group_rows",
    [
        (1, 4, 64, 32),
        (3, 6, 64, 32),
        (4, 6, 64, 32),
        (2, 4, 64, 1),
        (4, 7, 16, 32),
        (4, 4, 64, None),
    ],
    ids=["tp1", "tp3", "uneven-hc", "1x32", "unaligned-dim", "unquantized"],
)
def test_unsupported_configs_keep_full_loading_and_gate(
    monkeypatch, tp_group, tp, hc, dim, group_rows
):
    tp_group.world_size = tp
    op = make_op(hc=hc, dim=dim, group_rows=group_rows)
    assert op.wkv.local_hc == hc
    assert op.wkv.weight.shape == ((hc + 1) * dim, 96)
    assert op.wkv.output_size == (hc + 1) * dim
    if group_rows:
        weight, scales = checkpoint(op)
        for name, source in (("weight", weight), ("weight_scale", scales)):
            param = getattr(op.wkv, name)
            param.weight_loader(param, source)
            assert_bytes(param, source)
    else:
        weight = torch.randn_like(op.wkv.weight)
        op.load_checkpoint_weights(weight, torch.ones(hc, dim), torch.ones(hc, dim))
        torch.testing.assert_close(op.wkv.weight, weight)
    op.process_weights_after_loading()
    hidden, embeddings = torch.randn(2, 3, hc, dim), torch.randn(2, 3, 96)
    kv = torch.randn(2, 3, (hc + 1) * dim)
    monkeypatch.setattr(op.wkv, "forward", Mock(return_value=kv))
    gate = Mock(return_value=hidden)
    monkeypatch.setattr(layer, "engram_post_wkv", gate)
    monkeypatch.setattr(layer, "engram_post_wkv_tp", Mock(side_effect=AssertionError))
    monkeypatch.setattr(layer, "get_tp_group", Mock(side_effect=AssertionError))
    assert op(hidden, embeddings) is hidden
    gate.assert_called_once_with(hidden, kv, op.gate_weight, None, op.norm_eps)


@pytest.fixture(scope="module")
def gpu_checkpoint():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from aiter.jit.core import get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("Engram native FP8 projection targets gfx950")
    torch.manual_seed(123)
    weight = torch.randn(25600, 6144, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.randint(119, 128, (800, 192), device="cuda", dtype=torch.uint8)
    return weight, scales.view(torch.float8_e8m0fnu)


@pytest.mark.parametrize("m", [1, 12, 13, 16, 32, 256])
@pytest.mark.parametrize("tp", [2, 4])
def test_loaded_native_projection_matches_gathered_keys_and_value(
    monkeypatch, tp_group, gpu_checkpoint, m, tp
):
    monkeypatch.setenv("ATOM_GROUP32_WEIGHT_PRESHUFFLE", "1")
    weight, scales = gpu_checkpoint
    with torch.device("cuda"):
        x = torch.randn(m, 6144, dtype=torch.bfloat16)
        gate_weight = torch.ones(4, 5120)
        tp_group.world_size = 1
        full = make_op(dim=5120, width=6144)
        full.load_checkpoint_weights(weight, gate_weight, gate_weight, scales)
        expected = full.wkv(x).float()
        del full
        tp_group.world_size = tp
        keys, values = [], []
        for rank in range(tp):
            tp_group.rank_in_group = rank
            op = make_op(dim=5120, width=6144)
            op.load_checkpoint_weights(weight, gate_weight, gate_weight, scales)
            assert op.wkv.weight.is_shuffled
            assert op.wkv.weight.shape == ((4 // tp + 1) * 5120, 6144)
            actual = op.wkv(x).float()
            keys.append(actual[:, : (4 // tp) * 5120])
            value = actual[:, (4 // tp) * 5120 :]
            values.append(
                value if m <= 12 else torch.roll(value, op.wkv.value_start, -1)
            )
        gathered = torch.cat(keys + (values if m <= 12 else [values[0]]), dim=-1)
        # Independently tuned projection sizes can change accumulation order.
        # Bound aggregate error and the fraction of BF16 rounding outliers.
        for actual, reference in [(gathered, expected)] + [
            (
                value,
                (
                    expected[
                        :,
                        4 * 5120
                        + rank * (5120 // tp) : 4 * 5120
                        + (rank + 1) * (5120 // tp),
                    ]
                    if m <= 12
                    else expected[:, 4 * 5120 :]
                ),
            )
            for rank, value in enumerate(values)
        ]:
            assert torch.isfinite(actual).all()
            nrmse = (
                (actual - reference).square().mean() / reference.square().mean()
            ).sqrt()
            assert nrmse < 1e-3
            mismatch = ~torch.isclose(actual, reference, rtol=1e-2, atol=1e-2)
            assert mismatch.float().mean() < 1e-3
