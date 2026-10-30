# SPDX-License-Identifier: MIT
"""CPU checks of SP MegaMoE tiling, native fallback and fused-block ownership."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_comm_fused_moe import atom_modules  # noqa: F401
from test_sp_moe_dispatch import _layer
from test_sp_moe_dispatch import sp_modules as _sp_modules

from atom.config import use_custom_atom_config
from atom.model_ops import sp_mega_moe as fusion
from atom.model_ops.sp_mega_moe import SPMegaMoE, local_token_tile


@pytest.fixture(autouse=True)
def mock_mega_buffer_ops(monkeypatch):
    from atom.model_ops import sp_mega_moe

    # Replace only the GPU copy boundary for CPU orchestration tests. The real
    # copies and launch counts are covered on GPU.
    def copy(source, source_scale, out, out_scale):
        out.view(torch.uint8).copy_(source.view(torch.uint8))
        if source_scale is not None:
            out_scale.view(-1).copy_(source_scale)

    monkeypatch.setattr(sp_mega_moe, "copy_tail", copy)


@pytest.fixture
def sp_modules(request):
    return _sp_modules.__wrapped__(request)


@pytest.mark.parametrize("rows,use_mega", [(16, False), (63, False), (64, True)])
def test_small_batches_keep_the_native_moe(monkeypatch, sp_modules, rows, use_mega):
    from importlib import import_module

    moe = sp_modules.moe
    layer = _layer(moe)
    layer.w13_weight = torch.empty(1, dtype=moe.dtypes.fp4x2)
    resolver = Mock(return_value=moe.dtypes.fp4x2 if use_mega else torch.bfloat16)
    monkeypatch.setattr(
        import_module("aiter.fused_moe"),
        "resolve_activation_dtype",
        resolver,
        raising=False,
    )
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(world_size=4, tail_eps=1e-6), forward=Mock()
    )
    backend = SPMegaMoE(runtime, 1)
    assert backend.supports(layer, rows) is use_mega
    assert resolver.call_args.kwargs["M"] == rows * 4
    if use_mega:
        return

    eps_used = []

    def norm(x, residual, weight, eps, fp8):
        eps_used.append(eps)
        return x + residual, torch.ones(x.shape[0], 1), x + residual

    monkeypatch.setattr(fusion, "_norm_input", norm)
    layer.forward_impl = Mock(side_effect=lambda x, logits: x * 2)
    attention = torch.ones(rows, 8, dtype=torch.bfloat16)
    gate = torch.zeros(128, 8, dtype=torch.bfloat16)
    weight = torch.ones(8, dtype=torch.bfloat16)
    normalized, _, residual = backend.forward_block(
        layer, attention, attention * 3, weight, gate, weight, 1e-5, 1e-6, False
    )
    # Each norm uses its own eps, never the runtime's tail eps.
    assert eps_used == [1e-5, 1e-6]
    x, logits = layer.forward_impl.call_args.args
    torch.testing.assert_close(x, attention * 4)
    assert logits.shape == (rows, 128)
    torch.testing.assert_close(normalized, attention * 12)
    torch.testing.assert_close(residual, attention * 12)
    runtime.forward.assert_not_called()


@pytest.mark.parametrize(
    ("budget", "world_size", "tile"),
    [(2048, 4, 512), (4098, 4, 1025), (65536, 4, 8192), (16385, 8, 2049)],
)
def test_tile_covers_the_largest_local_shard(budget, world_size, tile):
    assert local_token_tile(budget, world_size) == tile


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("rows", [0, 16, 65])
def test_full_op_preserves_rank_rows_residual_and_output_ownership(
    monkeypatch, sp_modules, fp8, rows
):
    from aiter import dtypes

    # Real custom-op implementation and backend orchestration; only the device
    # boundaries are replaced. Shared scratch is overwritten on every launch.
    x = (torch.arange(rows) % 8).bfloat16()[:, None].expand(rows, 8).contiguous()
    residual = torch.ones_like(x)
    expected_residual = (x + residual) * 2 + (x + residual)
    scratch = torch.empty((4 * 32, 8), dtype=torch.bfloat16)
    tail_calls, plain_calls = [], []

    def norm(x, residual, weight, eps, fp8):
        r = x + residual
        return r.to(dtypes.fp8 if fp8 else r.dtype), torch.ones(r.shape[0], 1), r

    monkeypatch.setattr(fusion, "_norm_input", norm)

    def route(layer, hidden, router):
        return (
            torch.ones(hidden.shape[0], 5),
            hidden[:, :1].int().expand(-1, 5).contiguous(),
        )

    def launch(hidden, weights, ids, *, tail=None, bf16=False):
        torch.testing.assert_close(ids[:, :1].bfloat16(), hidden[:, :1])
        y = hidden * 2
        if tail is None:
            plain_calls.append(hidden.shape[0])
            return y
        tail_calls.append(hidden.shape[0])
        res_in, res_out, _nw = tail
        res_out.copy_(y + res_in)
        m = hidden.shape[0]
        # Non-owning rows contain poison; selecting global row 0 would fail.
        scratch.fill_(-42)
        scratch[2 * m : 3 * m].copy_(res_out)
        q = scratch.to(dtypes.fp8).view(torch.uint8)
        result = (y, q, torch.ones(m * 4))
        return (*result, scratch) if bf16 else result

    backend = fusion.SPMegaMoE(
        SimpleNamespace(
            cfg=SimpleNamespace(rank=2, world_size=4, tail_eps=1e-6),
            max_local_tokens=32,
            set_weights=Mock(),
            forward=launch,
            tail_ok=lambda m: m >= 16,
        ),
        1,
    )
    backend.supports = lambda layer, rows: True
    backend._route = route
    layer = SimpleNamespace(
        sp_mega_moe=backend,
        w13_weight=object(),
        w13_weight_scale=object(),
        w2_weight=object(),
        w2_weight_scale=object(),
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"e": layer})
    )
    weight = torch.ones(8, dtype=torch.bfloat16)
    gate = torch.zeros(128, 8, dtype=torch.bfloat16)
    before_x, before_res = x.clone(), residual.clone()
    with use_custom_atom_config(config):
        first = fusion.sp_mega_moe_block(
            x, residual, weight, gate, weight, "e", 1e-6, 1e-6, fp8
        )
        second = fusion.sp_mega_moe_block(
            x, residual, weight, gate, weight, "e", 1e-6, 1e-6, fp8
        )
    second[0].zero_()
    second[2].zero_()
    expected_normalized = expected_residual.to(dtypes.fp8) if fp8 else expected_residual
    torch.testing.assert_close(first[0].float(), expected_normalized.float())
    torch.testing.assert_close(first[2], expected_residual)
    torch.testing.assert_close(x, before_x)
    torch.testing.assert_close(residual, before_res)
    assert first[1].shape == (rows, 1)
    assert first[0].dtype == (dtypes.fp8 if fp8 else torch.bfloat16)
    if rows == 65:
        assert plain_calls == [1, 1]
        assert tail_calls == [32, 32, 32, 32]


def _servable_layer(moe):
    layer = _layer(moe)
    method = layer.quant_method
    vars(method).update(
        quant_dtype=moe.dtypes.fp4x2,
        is_guinterleave=False,
        use_triton=False,
        use_triton_decode=False,
        use_triton_ep=False,
        fused_experts=None,
        hidden_pad=0,
        intermediate_pad=0,
    )
    hidden, inter, experts = 256, 128, 5
    vars(layer).update(
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        local_num_experts=experts,
        w13_bias=None,
        w2_bias=None,
        w13_input_scale=None,
        w2_input_scale=None,
    )
    for name, shape in (
        ("w13_weight", (experts, 2 * inter, hidden // 2)),
        ("w2_weight", (experts, hidden, inter // 2)),
        ("w13_weight_scale", (experts * 2 * inter, hidden // 32)),
        ("w2_weight_scale", (experts * hidden, inter // 32)),
    ):
        value = torch.empty(shape, dtype=torch.uint8)
        value.is_shuffled = True
        setattr(layer, name, value)
    return layer


def test_layer_shapes_follow_the_model_and_ineligible_layers_stay_native(
    monkeypatch, sp_modules
):
    from atom.distributed import ulysses_sp

    layer = _servable_layer(sp_modules.moe)
    assert fusion._unsupported_layer(layer) is None
    layer.w2_weight = layer.w2_weight.view(5, 128, 128)
    assert "w2_weight" in fusion._unsupported_layer(layer)

    # Enabled by default: an ineligible model must load on the native MoE.
    monkeypatch.setattr(ulysses_sp, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(fusion, "_unsupported_config", lambda config: None)
    monkeypatch.setattr(fusion, "_model_layers", lambda model, config: ([layer], 1e-6))
    config = SimpleNamespace(sp_mega_moe_enabled=True)
    assert fusion.initialize_sp_mega_moe(object(), config) is None
    # The compile hash reads the flag; native graphs must not reuse mega ones.
    assert config.sp_mega_moe_enabled is False
    assert getattr(layer, "sp_mega_moe", None) is None


def _initialize(monkeypatch, sp_modules, runtime, mnbt=1024, **stubs):
    from atom.distributed import ulysses_sp

    layer = _servable_layer(sp_modules.moe)
    mega = ModuleType("aiter.ops.flydsl.mega_moe_tp")
    mega.MegaMoeTPConfig = dict
    mega.MegaMoeTP = runtime
    comm = ModuleType("aiter.ops.communication")
    comm._mori = stubs.get("mori", Mock())
    norm = ModuleType("atom.model_ops.gemma_norm_fp8")
    norm.gemma_rms_norm_fp8_supported = stubs.get("norm_supported", lambda h: True)
    for module in (mega, comm, norm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(ulysses_sp, "get_sp_world_size", lambda: 4)
    monkeypatch.setattr(
        ulysses_sp,
        "get_sp_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=0, cpu_group=None),
    )
    monkeypatch.setattr(fusion, "_unsupported_config", lambda config: None)
    monkeypatch.setattr(fusion, "_model_layers", lambda model, config: ([layer], 1e-6))
    config = SimpleNamespace(sp_mega_moe_enabled=True, max_num_batched_tokens=mnbt)
    return config, layer, lambda: fusion.initialize_sp_mega_moe(object(), config)


@pytest.mark.parametrize(
    "error,falls_back",
    [
        (RuntimeError("SdmaComm: MORI SDMA setup failed on ranks [0, 1, 2, 3]"), True),
        # One rank's compile error, lazy import or OOM leaves its peers inside
        # a collective.
        (RuntimeError("compile failed"), False),
        (ImportError("lazy import failed"), False),
        (torch.OutOfMemoryError("HIP out of memory"), False),
    ],
)
def test_runtime_setup_failure_keeps_the_native_moe(
    monkeypatch, sp_modules, error, falls_back
):
    runtime = Mock(side_effect=error)
    config, layer, initialize = _initialize(monkeypatch, sp_modules, runtime)
    if not falls_back:
        with pytest.raises(type(error)):
            initialize()
        assert config.sp_mega_moe_enabled is True
        return
    assert initialize() is None
    assert config.sp_mega_moe_enabled is False
    assert runtime.call_args.args[0]["max_local_tokens"] == 256
    assert getattr(layer, "sp_mega_moe", None) is None


@pytest.mark.parametrize(
    "stubs",
    [
        {"mori": Mock(side_effect=ImportError("no MORI"))},
        {"norm_supported": lambda hidden: False},
    ],
)
def test_missing_dependency_falls_back_before_any_collective(
    monkeypatch, sp_modules, stubs
):
    runtime = Mock()
    config, _, initialize = _initialize(monkeypatch, sp_modules, runtime, **stubs)
    assert initialize() is None
    assert config.sp_mega_moe_enabled is False
    runtime.assert_not_called()


def test_epoch_count_covers_every_tile_of_the_largest_shard(monkeypatch, sp_modules):
    # 65536 tokens over SP4 are 16384 local rows: eight 2048-row launches.
    runtime = Mock(return_value=Mock(max_local_tokens=2048))
    _, layer, initialize = _initialize(monkeypatch, sp_modules, runtime, mnbt=65536)
    backend = initialize()
    assert runtime.call_args.args[0]["max_local_tokens"] == 8192
    assert backend.launches_per_step == 8
    assert layer.sp_mega_moe is backend


@pytest.fixture
def host_copies(monkeypatch):
    # The timeout bits are copied to pinned memory behind an event.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=Mock(), query=lambda: True)
    )


def test_epoch_reset_follows_counted_steps(monkeypatch, host_copies):
    monkeypatch.setattr(fusion, "RESET_LAUNCHES", 8)
    runtime = SimpleNamespace(
        reset=Mock(), error_flag=lambda: torch.zeros(1, dtype=torch.int32)
    )
    backend = SPMegaMoE(runtime, 3)
    for _ in range(3):
        backend.count_step()
    runtime.reset.assert_called_once_with()
    assert backend.launches == 0
    backend.count_step()
    assert backend.launches == 3


def test_timed_out_launch_fails_a_later_step(monkeypatch, host_copies):
    # The step's output is already invalid; the epoch reset must not hide it.
    monkeypatch.setattr(fusion, "RESET_LAUNCHES", 2)
    flag = torch.zeros(1, dtype=torch.int32)
    runtime = SimpleNamespace(
        error_flag=lambda: flag, reset=Mock(side_effect=flag.zero_), cfg=Mock(rank=2)
    )
    backend = SPMegaMoE(runtime, 1)
    backend.count_step()
    flag[0] = 4  # a wait in this step timed out
    backend.count_step()
    runtime.reset.assert_called_once_with()
    with pytest.raises(RuntimeError, match="rank 2: a MegaMoeTP wait timed out"):
        backend.count_step()
