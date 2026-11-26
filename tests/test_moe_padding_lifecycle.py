# SPDX-License-Identifier: MIT
"""Padding survives reload, rollout sleep/wake, and dummy initialization on CPU."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from conftest import atom_config_double


@pytest.fixture(scope="module")
def impl():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            moe=load("atom.model_ops.moe"),
            layout=load("atom.model_ops.fused_moe.expert_layout"),
            loader=load("atom.model_loader.loader"),
            memory=load("atom.rollout.memory_manager"),
            updater=load("atom.rollout.weight_updater"),
            utils=load("atom.model_ops.utils"),
            shuffle=load("aiter.ops.shuffle"),
            selector=load("atom.utils.selector"),
        )


def _pack(weight, *args, **kwargs):
    # An invertible, nontrivial CPU layout stand-in. It moves the padding too,
    # so clearing row-major tails after the shuffle cannot pass these tests.
    if weight.element_size() == 1:
        return weight.view(torch.uint8).flip((-2, -1)).view(weight.dtype)
    return weight.flip((-2, -1))


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch, impl):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(impl.utils, "shuffle_weight", _pack)
    monkeypatch.setattr(impl.shuffle, "shuffle_weight", _pack)


def _layer(impl, mode="bf16", tp=4):
    m = impl.moe
    layer = object.__new__(m.FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.prefix = layer.layer_name = "layers.0.mlp.experts"
    layer.quant_config = None
    layer.moe_parallel_config = SimpleNamespace(
        tp_size=tp, tp_rank=tp - 1, use_ep=False
    )
    layer.expert_map = None
    layer.num_redundant_experts = 0
    layer.global_num_experts = 2
    layer.local_num_experts = 3
    layer.expert_layout = SimpleNamespace(num_routed=2, num_fused_shared_experts=1)
    layer.intermediate_size_per_partition = 640 // tp
    spec = m.LayerQuantConfig(
        quant_dtype=m.dtypes.fp8 if mode == "fp8" else torch.bfloat16,
        quant_type=m.QuantType.per_Token if mode == "fp8" else m.QuantType.No,
    )
    layer.layer_quant_config = spec
    config = SimpleNamespace(in_dtype=torch.bfloat16, hidden_dim=8)
    method = (
        m.CompressedTensorsFp8MoEMethod(spec, config)
        if mode == "fp8"
        else m.UnquantizedFusedMoEMethod(config)
    )
    method.pad_align = 64
    if mode == "fp8":
        # Dummy weights below are already in the native FP8 storage dtype.
        method.need_normalize_e4m3fn_to_e4m3fnuz = False
    layer.quant_method = method
    method.create_weights(
        layer, 3, 8, 640 // tp, spec.quant_dtype, weight_loader=layer.weight_loader
    )
    return layer


def _runner(impl, layer):
    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    block.mlp.experts = layer
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([block])
    model.get_expert_mapping = lambda: impl.moe.FusedMoE.make_expert_params_mapping(
        "gate_proj", "down_proj", "up_proj", 3
    )

    class Runner(impl.memory.MemoryManagerMixin, impl.updater.WeightUpdaterMixin):
        def __init__(self):
            self._init_weight_updater()
            self.model = model
            self.device = torch.device("cpu")
            self.rank = 0
            self.world_size = layer.moe_parallel_config.tp_size
            self.label = "padding-test"
            self.enforce_eager = True
            self.config = atom_config_double(
                parallel_config=SimpleNamespace(data_parallel_rank_local=0)
            )
            self.attn_family = impl.selector.Family.MHA

        def clear_kv_cache(self):
            pass

    return Runner()


@pytest.mark.parametrize("mode", ["bf16", "fp8"])
@pytest.mark.parametrize("load_full", [False, True])
@pytest.mark.parametrize("shard", ["w1", "w3", "w2"])
def test_expert_write_clears_only_its_padding(impl, mode, load_full, shard):
    layer = _layer(impl, mode)
    param = layer.w2_weight if shard == "w2" else layer.w13_weight
    param.data.fill_(7)
    dest = param.data if load_full else param.data[0]
    dim = (1 if shard == "w2" else 0) + int(load_full)
    half = impl.layout.expert_shard_view(dest, shard, dim)
    half.fill_(float("nan"))
    shape = list(half.shape)
    logical = layer.intermediate_size_per_partition
    shape[dim] = logical if load_full else 640
    source = (torch.arange(prod(shape)).reshape(shape) % 11 + 1).to(param.dtype)
    expected = param.data.clone()
    written = impl.layout.expert_shard_view(
        expected if load_full else expected[0], shard, dim
    )
    written.zero_()
    shard_source = (
        source if load_full else source.narrow(dim, layer.tp_rank * logical, logical)
    )
    written.narrow(dim, 0, logical).copy_(shard_source)

    if shard == "w2":
        layer._load_w2(dest, dim, source, layer.tp_rank, load_full=load_full)
    else:
        layer._load_w13(dest, dim, shard, source, layer.tp_rank, load_full=load_full)
    # Whole-buffer equality also protects untouched experts (including shared)
    # and the other W13 half from an over-broad zero operation.
    assert torch.equal(param.data.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("mode", ["bf16", "fp8"])
def test_dummy_xavier_padding_is_zero_before_pack(impl, mode, tp):
    layer = _layer(impl, mode, tp)
    impl.loader.initialize_dummy_weights(layer, "xavier")
    before13, before2 = layer.w13_weight.data.clone(), layer.w2_weight.data.clone()
    logical, physical = 640 // tp, before2.shape[-1]
    before_scale = layer.w13_weight_scale.data.clone() if mode == "fp8" else None
    layer.quant_method.process_weights_after_loading(layer)
    raw13, raw2 = _pack(layer.w13_weight.data), _pack(layer.w2_weight.data)
    for half in range(2):
        start = half * physical
        assert torch.equal(
            raw13[:, start : start + logical].view(torch.uint8),
            before13[:, start : start + logical].view(torch.uint8),
        )
        assert (
            not raw13[:, start + logical : start + physical]
            .view(torch.uint8)
            .count_nonzero()
        )
        if before_scale is not None:
            scale = layer.w13_weight_scale.data[:, start : start + physical]
            torch.testing.assert_close(
                scale[:, :logical], before_scale[:, start : start + logical]
            )
            assert not scale[:, logical:].count_nonzero()
    assert torch.equal(
        raw2[:, :, :logical].view(torch.uint8),
        before2[:, :, :logical].view(torch.uint8),
    )
    assert not raw2[:, :, logical:].view(torch.uint8).count_nonzero()


@pytest.mark.parametrize("tp", [4, 8])
def test_release_resume_sync_restores_zero_padding(impl, tp):
    layer = _layer(impl, tp=tp)
    runner = _runner(impl, layer)
    model = runner.model
    for update in (1, 2):
        runner._release_weights()
        assert all(p.numel() == 0 for p in model.parameters())
        runner._resume_weights()
        # torch.empty is allowed to return NaNs: poison every new placeholder
        # deterministically instead of depending on the CPU allocator's history.
        for param in model.parameters():
            param.data.fill_(float("nan"))
        pointers = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
        weights = []
        for expert in range(3):
            for proj, shape in (
                ("gate_proj", (640, 8)),
                ("up_proj", (640, 8)),
                ("down_proj", (8, 640)),
            ):
                source = (
                    (torch.arange(prod(shape)).reshape(shape) % 7 + 1 + expert + update)
                    * 0.01
                ).to(torch.bfloat16)
                weights.append((f"layers.0.mlp.experts.{expert}.{proj}.weight", source))
        assert runner.update_weights(weights) == 9
        assert pointers == (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
        raw13, raw2 = _pack(layer.w13_weight.data), _pack(layer.w2_weight.data)
        logical, physical = 640 // tp, raw2.shape[-1]
        source_slice = slice((tp - 1) * logical, tp * logical)
        for expert in range(3):
            gate, up, down = [t for _, t in weights[expert * 3 : expert * 3 + 3]]
            torch.testing.assert_close(raw13[expert, :logical], gate[source_slice])
            torch.testing.assert_close(
                raw13[expert, physical : physical + logical], up[source_slice]
            )
            torch.testing.assert_close(raw2[expert, :, :logical], down[:, source_slice])
        assert not raw13[:, logical:physical].count_nonzero()
        assert not raw13[:, physical + logical :].count_nonzero()
        assert not raw2[:, :, logical:].count_nonzero()
        x = torch.ones((1, 8))
        gate, up = (x @ part.float().T for part in raw13[0].chunk(2))
        actual = (torch.nn.functional.silu(gate) * up) @ raw2[0].float().T
        expected = (
            torch.nn.functional.silu(gate[:, :logical]) * up[:, :logical]
        ) @ raw2[0, :, :logical].float().T
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize(
    "buffer_names",
    [
        ("w13_weight",),
        ("w2_weight",),
        ("w13_weight", "w2_weight"),
        ("w2_weight", "w13_weight"),
    ],
    ids=["w13-only", "w2-only", "w13-then-w2", "w2-then-w13"],
)
def test_named_buffer_sync_clears_only_rewritten_padding(impl, tp, buffer_names):
    layer = _layer(impl, tp=tp)
    runner = _runner(impl, layer)
    logical = layer.intermediate_size_per_partition
    physical = layer.w2_weight.shape[-1]
    names = ("w13_weight", "w2_weight")

    def fill_padding(tensor, name, value):
        if name == "w13_weight":
            tensor[:, logical:physical].fill_(value)
            tensor[:, physical + logical :].fill_(value)
        else:
            tensor[:, :, logical:].fill_(value)

    def row_major(name, version):
        param = getattr(layer, name)
        values = torch.arange(param.numel()).reshape(param.shape) % 17 + 1 + version
        values = (values * 0.01).to(param.dtype)
        fill_padding(values, name, 0)
        return values

    expected = {name: row_major(name, 0) for name in names}
    for name in names:
        getattr(layer, name).data.copy_(expected[name])
    layer.quant_method.process_weights_after_loading(layer)
    pointers = {name: getattr(layer, name).data_ptr() for name in names}
    assert runner._reload_refusal() is None

    for version, poison in enumerate((float("nan"), 7.0), start=1):
        before = {name: getattr(layer, name).data.clone() for name in names}
        incoming = []
        for name in buffer_names:
            expected[name] = row_major(name, version)
            source = expected[name].clone()
            fill_padding(source, name, poison)
            incoming.append((f"layers.0.mlp.experts.{name}", source))
        source_bytes = [source.view(torch.uint8).clone() for _, source in incoming]

        assert runner.update_weights(incoming) == len(buffer_names)
        assert not runner._expert_relayout_pending
        for name in names:
            param = getattr(layer, name)
            assert param.data_ptr() == pointers[name]
            assert param.is_shuffled
            # Includes every routed and shared slot, both halves and the tail.
            raw = _pack(param.data)
            assert torch.equal(raw.view(torch.uint8), expected[name].view(torch.uint8))
            if name not in buffer_names:
                # The other buffer is still shuffled: row-major tail clearing
                # on it would corrupt live values even if its own pad was zero.
                assert torch.equal(
                    param.data.view(torch.uint8), before[name].view(torch.uint8)
                )
        for (_, source), original in zip(incoming, source_bytes):
            assert torch.equal(source.view(torch.uint8), original)
