# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The start-up options of a recipe's ``engine`` block: no MoRI EP op handles
at model build (``mori_defer_handles``), dummy prefill forwards at listed row
counts (``prefill_warmup_ladder``) and the draft's context projection at the
decode widths (``decode_warmup_widths`` / ``decode_warmup_tails``). CPU only:
the runner and the draft model are stood in for; what is checked is which
forwards and which projections the warm-ups ask for, and that nothing runs
with the options unset."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

runner_module = None
dspark_module = None
recipe_module = None
mori_module = None


@pytest.fixture(scope="module", autouse=True)
def _modules():
    global runner_module, dspark_module, recipe_module, mori_module
    with atom_without_aiter() as load:
        runner_module = load("atom.model_engine.model_runner")
        dspark_module = load("atom.models.deepseek_v41_plugin.dspark")
        recipe_module = load("atom.plugin_recipe")
        mori_module = load("atom.model_ops.fused_moe.deferred_mori_op")
        yield


@pytest.fixture
def engine(monkeypatch):
    """Set the recipe's engine options as the runner reads them."""

    def set_options(**options):
        plan = SimpleNamespace(engine=recipe_module.EngineOptions.from_mapping(options))
        monkeypatch.setattr(runner_module.plugin_host, "_plan", plan)

    monkeypatch.setattr(runner_module.plugin_host, "_plan", None)
    return set_options


# -- engine.mori_defer_handles ------------------------------------------------


def test_mori_handles_are_built_unless_the_recipe_defers_them(engine):
    assert not mori_module.defer_mori_handles()
    engine(mori_defer_handles=True)
    assert mori_module.defer_mori_handles()


def test_a_deferred_handle_raises_at_its_first_use():
    handle = mori_module.DeferredMoriOp(1)
    with pytest.raises(RuntimeError, match=r"slot 1.*engine\.mori_defer_handles"):
        _ = handle.dispatch


# -- engine.prefill_warmup_ladder ----------------------------------------------


def _ladder_runner(cap=64):
    batches = []
    cleaned = []
    runner = SimpleNamespace(
        label="rank0",
        block_size=16,
        warmup_max_tokens=lambda: cap,
        forward=batches.append,
        tokenID_processor=SimpleNamespace(clean=lambda: cleaned.append(1)),
    )
    return runner, batches, cleaned


def _run_ladder(runner):
    runner_module.ModelRunner._warmup_prefill_ladder(runner)


def test_no_ladder_no_forward(engine):
    runner, batches, _ = _ladder_runner()
    _run_ladder(runner)
    engine(prefill_warmup_ladder=[])
    _run_ladder(runner)
    assert batches == []


def test_the_ladder_runs_each_row_count_once_largest_first(engine):
    engine(prefill_warmup_ladder=[7, 2, 33, 7, 1, 64, 65, 4096])
    runner, batches, cleaned = _ladder_runner(cap=64)
    _run_ladder(runner)
    # 1 row cannot be split over two prompts; 65 and 4096 are past the cap
    assert [b.total_tokens_num for b in batches] == [64, 33, 7, 2]
    assert len(cleaned) == len(batches)
    for batch in batches:
        lens = [int(n) for n in batch.num_scheduled_tokens]
        total = batch.total_tokens_num
        assert lens == [total - total // 2, total // 2]
        assert batch.is_dummy_run and batch.total_seqs_num == 2
        assert batch.total_tokens_num_prefill == total
        assert len(batch.req_ids) == 2


def test_the_ladder_may_be_a_file_of_row_counts(engine, tmp_path):
    path = tmp_path / "ladder.txt"
    path.write_text("8\n2\n16 4\n")
    engine(prefill_warmup_ladder=str(path))
    runner, batches, _ = _ladder_runner()
    _run_ladder(runner)
    assert [b.total_tokens_num for b in batches] == [16, 8, 4, 2]


def test_a_row_count_that_cannot_run_is_skipped(engine):
    engine(prefill_warmup_ladder=[8, 4, 2])
    runner, batches, cleaned = _ladder_runner()

    def forward(batch):
        if batch.total_tokens_num == 4:
            raise RuntimeError("no dummy at this size")
        batches.append(batch)

    runner.forward = forward
    _run_ladder(runner)
    assert [b.total_tokens_num for b in batches] == [8, 2]
    assert len(cleaned) == 3


# -- engine.decode_warmup_widths -------------------------------------------------


def _decode_runner(monkeypatch, *, model="default", window=8, q_buckets=(6,)):
    monkeypatch.setattr(runner_module.torch.cuda, "synchronize", lambda: None)
    warmed = []
    if model == "default":
        model = SimpleNamespace(
            warm_context_projection=lambda rows, packed: warmed.append((rows, packed))
        )
    runner = SimpleNamespace(
        label="rank0",
        drafter=SimpleNamespace(model=model, mtp_k=5),
        config=SimpleNamespace(capture_sizes=[4, 0, 1, 2], max_num_batched_tokens=64),
        _dspark_capture_q_buckets=lambda full_q: [q for q in q_buckets if q <= full_q],
        attn_metadata_builder=SimpleNamespace(
            geometry=SimpleNamespace(window_size=window, window_fp8=True)
        ),
    )
    return runner, warmed


def _run_decode(runner):
    runner_module.ModelRunner._warmup_decode_draft_context(runner)


def test_no_widths_no_projection(engine, monkeypatch):
    runner, warmed = _decode_runner(monkeypatch)
    _run_decode(runner)
    assert warmed == []


def test_capture_warms_the_capture_ladder_and_its_prefill_tails(engine, monkeypatch):
    engine(decode_warmup_widths="capture")
    runner, warmed = _decode_runner(monkeypatch)
    _run_decode(runner)
    # widths 1, 2, 4: six rows a request on a decode, eight on a prefill tail
    assert warmed == [(rows, True) for rows in (6, 8, 12, 16, 24, 32)]


def test_a_width_list_and_tails_of_their_own(engine, monkeypatch):
    engine(decode_warmup_widths="1-3, 10", decode_warmup_tails="2 8-9")
    runner, warmed = _decode_runner(monkeypatch, q_buckets=(3, 6))
    _run_decode(runner)
    decode = {w * q for w in (1, 2, 3, 10) for q in (3, 6)}
    tails = {16, 64}  # tails of 8 and 9 requests are both cut to the batch
    assert [rows for rows, _ in warmed] == sorted(decode | tails)


def test_tails_zero_warms_the_decode_widths_only(engine, monkeypatch):
    engine(decode_warmup_widths="2", decode_warmup_tails="0")
    runner, warmed = _decode_runner(monkeypatch)
    _run_decode(runner)
    assert warmed == [(12, True)]


def test_row_counts_past_the_batch_are_left_out(engine, monkeypatch):
    engine(decode_warmup_widths="4,16", decode_warmup_tails="0")
    runner, warmed = _decode_runner(monkeypatch)
    _run_decode(runner)
    assert warmed == [(24, True)]  # width 16 is past the batch


def test_a_draft_without_the_projection_or_no_draft_is_skipped(engine, monkeypatch):
    engine(decode_warmup_widths="capture")
    runner, _ = _decode_runner(monkeypatch, model=SimpleNamespace())
    _run_decode(runner)
    runner, _ = _decode_runner(monkeypatch, model=None)
    _run_decode(runner)
    del runner.drafter
    _run_decode(runner)


def test_a_width_that_cannot_run_is_skipped(engine, monkeypatch):
    engine(decode_warmup_widths="1,2", decode_warmup_tails="0")
    seen = []

    def warm(rows, packed):
        seen.append(rows)
        if rows == 6:
            raise RuntimeError("no kernel for this width")

    runner, _ = _decode_runner(
        monkeypatch, model=SimpleNamespace(warm_context_projection=warm)
    )
    _run_decode(runner)
    assert seen == [6, 12]


# -- the plugin model's projection ----------------------------------------------


class _Draft:
    """The parts of the plugin draft model `warm_context_projection` and
    `_fused_context_keys` read, recording what they are handed."""

    def __init__(self, fused, tail=None):
        self.calls = []
        proj = SimpleNamespace(input_size=12, output_size=4)
        norm = SimpleNamespace(weight=torch.ones(4))
        self.mtp = [SimpleNamespace(main_proj=proj, main_norm=norm)]
        self.rope = SimpleNamespace(cos_cache="cos", sin_cache="sin")
        self._fused, self._tail = fused, tail
        self.context_layers = (self._layer(0), self._layer(1))

    def _layer(self, index):
        def project_context(hidden, positions, rope, *, packed):
            self.calls.append(("stage", index, tuple(positions.shape), packed))

        return SimpleNamespace(
            project_context=project_context,
            kv_norm=SimpleNamespace(weight=f"w{index}", eps=1e-6),
        )

    def _context_kv_fusion(self):
        return self._fused

    def _draft_kv_tail_fusion(self):
        return self._tail

    def project_context(self, taps):
        self.calls.append(("norm", [tuple(t.shape) for t in taps], taps[0].dtype))
        return "hidden"

    def project_context_quant(self, taps):
        self.calls.append(("norm_quant", [tuple(t.shape) for t in taps]))
        return "hidden", "scale"

    def _fused_context_keys(self, hidden, hidden_scale, positions, *, packed):
        self.calls.append(
            ("keys", hidden, hidden_scale, tuple(positions.shape), packed)
        )


@pytest.fixture
def draft_env(monkeypatch):
    monkeypatch.setattr(
        dspark_module,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    monkeypatch.delenv("ATOM_USE_MODEL_SENSITIVE_RMSNORM", raising=False)
    gemms = []
    monkeypatch.setattr(
        dspark_module,
        "native_quant_linear",
        lambda hidden, weight, weight_scale, **kw: gemms.append(
            (hidden, weight, weight_scale, kw)
        ),
    )
    return gemms


def _warm(draft, rows, packed):
    dspark_module.DeepseekV41DSpark.warm_context_projection(draft, rows, packed)


def test_the_projection_takes_constant_taps_at_the_row_count(draft_env):
    draft = _Draft(fused=None)
    _warm(draft, 6, True)
    assert draft.calls == [
        ("norm", [(6, 4)] * 3, torch.bfloat16),
        ("stage", 0, (1, 6), True),
        ("stage", 1, (1, 6), True),
    ]
    assert draft_env == []


def test_an_unknown_window_layout_stops_after_the_projection(draft_env):
    draft = _Draft(fused=None)
    _warm(draft, 6, None)
    assert [call[0] for call in draft.calls] == ["norm"]


def test_the_fused_stages_run_through_the_keys_of_the_real_write(draft_env):
    draft = _Draft(fused=("weight", "weight_scale", 32, 8, False))
    _warm(draft, 10, False)
    assert draft.calls == [
        ("norm_quant", [(10, 4)] * 3),
        ("keys", "hidden", "scale", (1, 10), False),
    ]
    assert draft_env == []


def test_the_fused_gemm_alone_when_the_window_layout_is_unknown(draft_env):
    draft = _Draft(fused=("weight", "weight_scale", 32, 8, True))
    _warm(draft, 10, None)
    assert [call[0] for call in draft.calls] == ["norm_quant"]
    assert draft_env == [
        (
            "hidden",
            "weight",
            "weight_scale",
            {"weight_group_rows": 32, "weight_preshuffled": True, "x_scale": "scale"},
        )
    ]


@pytest.mark.parametrize("packed", [False, True])
def test_the_keys_are_stage_major_with_or_without_the_fused_tail(
    draft_env, monkeypatch, packed
):
    """`_fused_context_keys` is what `write_context_kv` writes to the windows:
    one GEMM for every stage's wkv, then per stage the norm, and RoPE and the
    quant over the stage-major rows -- or the one kernel that does all three."""
    rows, width = 5, 8
    kv = torch.arange(rows * 2 * width, dtype=torch.float32).view(1, rows, 2 * width)
    monkeypatch.setattr(
        dspark_module, "native_quant_linear", lambda hidden, *a, **kw: kv
    )
    normed = []

    def rmsnorm(x, weight, eps, dim):
        normed.append((weight, dim))
        return x

    monkeypatch.setattr(dspark_module, "rmsnorm2d_fwd_", rmsnorm)
    monkeypatch.setattr(
        dspark_module,
        "quantize_fp8",
        lambda x, dequantize: ("quant", x, dequantize),
    )
    positions = torch.arange(rows)[None]
    keys = dspark_module.DeepseekV41DSpark._fused_context_keys

    draft = _Draft(fused=("weight", "weight_scale", 32, width, False))
    draft.rope = lambda x, pos: x + 0
    tag, out, dequantize = keys(draft, "hidden", "scale", positions, packed=packed)
    assert tag == "quant" and dequantize is (not packed)
    assert normed == [("w0", width), ("w1", width)]
    stages = kv.unflatten(-1, (2, width))
    assert torch.equal(out, torch.cat([stages[..., 0, :], stages[..., 1, :]]))

    tails = []
    monkeypatch.setattr(
        dspark_module,
        "fused_draft_kv_tail",
        lambda kv, weight, pos, cos, sin, eps, packed: tails.append(
            (tuple(kv.shape), weight, tuple(pos.shape), cos, sin, eps, packed)
        )
        or "tail-keys",
    )
    draft = _Draft(
        fused=("weight", "weight_scale", 32, width, False), tail=("nw", 1e-5)
    )
    assert keys(draft, "hidden", "scale", positions, packed=packed) == "tail-keys"
    assert tails == [((1, rows, 2, width), "nw", (rows,), "cos", "sin", 1e-5, packed)]


# -- dspark_options.context_taps_concat -------------------------------------------


def test_the_taps_are_quantized_where_they_lie_unless_the_recipe_concatenates(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        dspark_module,
        "quantize_fp8_taps",
        lambda taps: calls.append([tuple(t.shape) for t in taps]) or ("fp8", "scale"),
    )
    taps = [torch.full((5, 4), float(i)) for i in range(3)]
    concat = dspark_module.DeepseekV41DSpark._concat_taps
    monkeypatch.setattr(dspark_module.plugin_host, "_plan", None)
    assert concat(taps, None, 3) == ("fp8", "scale")
    assert calls == [[(3, 4)] * 3]
    plan = SimpleNamespace(
        dspark_options=recipe_module.DSparkOptions(context_taps_concat=True)
    )
    monkeypatch.setattr(dspark_module.plugin_host, "_plan", plan)
    joined, scale = concat(taps, None, 3)
    assert scale is None and len(calls) == 1
    assert joined.shape == (3, 12) and torch.equal(joined, torch.cat(taps, -1)[:3])
    rows = torch.tensor([4, 0])
    joined, _ = concat(taps, rows, 2)
    assert torch.equal(joined, torch.cat(taps, -1)[rows])
    single, scale = concat(taps[1], None, 2)
    assert scale is None and torch.equal(single, taps[1][:2])
