# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark on the SGLang plugin: the sampled verify's top-p
target probabilities with PyTorch's exact pivot and no vocabulary sort
(`dspark_topp_verify_probs_patch`, `dspark_topp_pivot`).

CPU tests: the scan-chunk formula, the branch predicate, delegation, the
adaptive guard, the registry switches, install only with the sampled draft,
idempotent install and the version / source guards. GPU tests (skipped
without a GPU and SGLang 0.5.22's renorm kernels):
the patched build is bitwise SGLang's on adversarial tables -- ties at the
cut, flat and uniform rows, top_p = 1, tiny top_p, mixed per-request top_p
and temperature, thresholds placed on PyTorch's own cdf values and one ulp
either side, rows the fallback must sort -- at both of torch's scan chunk
sizes, and it issues no host synchronisation."""

import os
from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang import deepseek_v41_envs
from atom.plugin.sglang.patches import dspark_topp_pivot as pivot
from atom.plugin.sglang.patches import dspark_topp_verify_probs_patch as patch
from atom.plugin.sglang.patches.sglang_version_guard import SUPPORTED_SGLANG_VERSIONS

V = 129280


@pytest.fixture(autouse=True)
def _sampled_draft(monkeypatch):
    # The patch installs only with the sampled draft (the flag unset).
    monkeypatch.delenv("SGLANG_DSPARK_USE_GREEDY_DRAFT", raising=False)


# --------------------------------------------------------------------------- CPU


def _aten_chunk(num_rows, row_size):
    # ATen get_log_num_threads_x_inner_scan<uint32_t>, written out independently
    lx = max(0, (row_size - 1).bit_length())
    ly = max(0, (num_rows - 1).bit_length())
    diff = (lx - ly) % (1 << 32)
    v = ((9 + diff) % (1 << 32)) // 2
    return 2 * (1 << min(max(4, v), 9))


@pytest.mark.parametrize("rows", [2, 6, 48, 192, 256, 257, 300, 384, 512, 513, 2048])
@pytest.mark.parametrize("vocab", [4096, 129280, 131072])
def test_scan_chunk_matches_aten(rows, vocab):
    assert pivot.torch_scan_chunk(rows, vocab) == _aten_chunk(rows, vocab)


def test_scan_chunk_values_for_v41():
    assert pivot.torch_scan_chunk(6, V) == 1024
    assert pivot.torch_scan_chunk(256, V) == 1024
    assert pivot.torch_scan_chunk(257, V) == 512
    assert pivot.torch_scan_chunk(512, V) == 512


def test_supports_v41_rows_only():
    assert pivot.supports(6, V) and pivot.supports(384, V) and pivot.supports(512, V)
    assert not pivot.supports(1, V)  # a 1-D cumsum takes another (device-scan) path
    # above MAX_ROWS (torch's chunk is still 512 up to 1024 rows, 256 from 1025):
    # a conservative cap, SGLang's path
    assert pivot.torch_scan_chunk(1024, V) == 512
    assert pivot.torch_scan_chunk(1025, V) == 256
    assert not pivot.supports(513, V)
    assert not pivot.supports(6, 1000)  # vocabulary smaller than the bitonic tile
    assert not pivot.supports(6, V + 7)  # head chunks no longer aligned


def test_rounding_coef_is_conservative():
    # 2 u ((L+1)(n_c+1)/2 + 3L + 2), L = 10, n_c = (129280 - 1280) / 1024 = 125
    coef = pivot.rounding_coef(1024, V, 1280)
    assert coef == pytest.approx(2 * 2.0**-24 * (11 * 126 / 2 + 32))
    assert 5e-5 < coef < 1e-4


class _Logits:
    """A CUDA-looking [rows, vocab] tensor for the predicate (CPU-only tests)."""

    def __init__(self, rows=12, vocab=V, device="cuda"):
        self.device = torch.device(device)
        self.shape = (rows, vocab)

    def dim(self):
        return len(self.shape)


def _info(bs=2, **kw):
    base = {
        "need_top_k_sampling": False,
        "need_top_p_sampling": True,
        "top_ps": SimpleNamespace(numel=lambda: bs, device=torch.device("cuda")),
        "temperatures": object(),
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_takes_only_the_dense_top_p_branch():
    x = _Logits(rows=12)
    assert patch.takes_exact_topp(x, _info(), 6, 2)
    assert not patch.takes_exact_topp(x, _info(need_top_k_sampling=True), 6, 2)
    assert not patch.takes_exact_topp(x, _info(need_top_p_sampling=False), 6, 2)


@pytest.mark.parametrize(
    "logits,bs",
    [
        (_Logits(rows=12, device="cpu"), 2),
        (_Logits(rows=12), 3),  # rows != bs * draft_token_num
        (_Logits(rows=1026), 171),  # more than 512 rows
        (_Logits(rows=12, vocab=1000), 2),
    ],
)
def test_other_tables_go_to_sglang(logits, bs):
    assert not patch.takes_exact_topp(logits, _info(bs=bs), 6, bs)


def test_one_top_p_per_request_required():
    x = _Logits(rows=12)
    assert not patch.takes_exact_topp(
        x,
        _info(top_ps=SimpleNamespace(numel=lambda: 12, device=torch.device("cuda"))),
        6,
        2,
    )
    other_device = SimpleNamespace(numel=lambda: 2, device=torch.device("cpu"))
    assert not patch.takes_exact_topp(x, _info(top_ps=other_device), 6, 2)


def test_other_calls_reach_sglang_with_their_arguments(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    seen = []

    def original(**kw):
        seen.append(kw)
        return "sglang"

    counters = SimpleNamespace(calls=0, last=None, budget_ms=0.0)
    guard = patch._Guard(counters, enabled=False)
    build = patch.make_exact_topp_build(original, None, counters, guard)
    kw = {
        "next_token_logits": _Logits(rows=12),
        "sampling_info": _info(need_top_k_sampling=True),
        "draft_token_num": 6,
        "bs": 2,
        "max_top_k": 7,
        "uniform_top_k_value": 7,
        "use_sparse_topk": False,
    }
    assert build(**kw) == "sglang"
    assert seen == [kw]
    assert getattr(build, patch._MARK)
    assert guard.calls == 0  # the guard only counts calls of the patched branch


def test_a_graph_capture_goes_to_sglang(monkeypatch):
    seen = []

    def original(**kw):
        seen.append(kw)
        return "sglang"

    counters = SimpleNamespace(calls=0, last=None, budget_ms=0.0)
    guard = patch._Guard(counters, enabled=False)
    build = patch.make_exact_topp_build(original, None, counters, guard)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    kw = {
        "next_token_logits": _Logits(rows=12),
        "sampling_info": _info(),
        "draft_token_num": 6,
        "bs": 2,
    }
    assert patch.takes_exact_topp(kw["next_token_logits"], kw["sampling_info"], 6, 2)
    assert build(**kw) == "sglang"
    assert len(seen) == 1 and guard.calls == 0


class _FakeCounters:
    def __init__(self):
        self.calls = 0
        self.fallback = 0
        self.budget_ms = 0.0
        self.last = None

    def poll(self):
        # the newest "completed copy" lags one window behind, like the real one
        out = self.last
        self.last = (self.calls, self.budget_ms, [0, 0, self.fallback, 0])
        return out


def _drive(guard, counters, n, fallback_per_call, rows=192):
    picks = []
    for _ in range(n):
        use = guard.use_exact()
        picks.append(use)
        if use:
            counters.calls += 1
            counters.fallback += fallback_per_call
            counters.budget_ms += patch.saving_ms(rows)
    return picks


def test_saving_table():
    assert patch.saving_ms(192) == pytest.approx(5.59)
    assert patch.saving_ms(6) == pytest.approx(0.96)
    assert patch.saving_ms(120) == pytest.approx((1.36 + 5.59) / 2)
    assert patch.saving_ms(512) > patch.saving_ms(384) > patch.saving_ms(192)
    assert patch.saving_ms(3) == pytest.approx(0.48)


def test_guard_switches_when_fallbacks_cost_more_than_they_save():
    c = _FakeCounters()
    g = patch._Guard(c, enabled=True)
    # 2 fallback rows a call at 192 rows: <= 10.6 ms of fallback against 5.59 ms saved
    picks = _drive(g, c, 4 * patch.GUARD_WINDOW, fallback_per_call=2)
    assert not all(picks) and g.epoch == 1  # one switch to SGLang's sort
    first_off = picks.index(False)
    assert first_off <= 3 * patch.GUARD_WINDOW + 1
    # the cool-down lasts GUARD_COOLDOWN calls of the patched branch, this call included
    off = _drive(g, c, patch.GUARD_COOLDOWN, fallback_per_call=0)
    assert sum(1 for p in picks[first_off:] + off if not p) == patch.GUARD_COOLDOWN
    assert all(_drive(g, c, 3 * patch.GUARD_WINDOW, fallback_per_call=0))


def _drive_rate(guard, counters, n, every, rows):
    picks = []
    for i in range(n):
        picks += _drive(guard, counters, 1, 1 if i % every == 0 else 0, rows=rows)
    return picks


def test_guard_keeps_the_exact_path_while_it_is_cheaper():
    # one fallback row every 2nd call at 192 rows: <= 2.65 ms against 5.59 ms saved
    c = _FakeCounters()
    g = patch._Guard(c, enabled=True)
    assert all(_drive_rate(g, c, 20 * patch.GUARD_WINDOW, 2, rows=192)) and g.epoch == 0


def test_guard_weighs_the_row_count():
    # the same fallback rate at one request (6 rows) saves only 0.96 ms a call
    c = _FakeCounters()
    g = patch._Guard(c, enabled=True)
    assert not all(_drive_rate(g, c, 20 * patch.GUARD_WINDOW, 2, rows=6))
    assert g.epoch >= 1


def test_guard_keeps_the_exact_path_on_rare_fallbacks():
    c = _FakeCounters()
    g = patch._Guard(c, enabled=True)
    assert all(_drive_rate(g, c, 20 * patch.GUARD_WINDOW, 50, rows=6)) and g.epoch == 0


def test_guard_off_always_exact():
    c = _FakeCounters()
    g = patch._Guard(c, enabled=False)
    assert all(_drive(g, c, 5 * patch.GUARD_WINDOW, fallback_per_call=5))


_SWITCHES = (
    "ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY",
    "ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY_GUARD",
)


@pytest.mark.parametrize("name", _SWITCHES)
def test_switches_parse_as_every_plugin_knob(monkeypatch, name):
    assert name in deepseek_v41_envs.environment_variables
    for raw, on in (("0", False), ("false", False), ("off", False), ("", False)):
        monkeypatch.setenv(name, raw)
        assert getattr(deepseek_v41_envs, name) is on
        monkeypatch.setenv(name, "1")
        assert getattr(deepseek_v41_envs, name) is True
    monkeypatch.delenv(name)
    assert getattr(deepseek_v41_envs, name) is True


def test_switches_are_in_the_recipe():
    from pathlib import Path

    recipe = Path(__file__).parents[2] / "recipes/atom_sglang/DeepSeek-V4.1-Flash.md"
    text = recipe.read_text()
    assert all(f"**{name}**" in text for name in _SWITCHES)


def test_switch_off_keeps_sglang(monkeypatch):
    monkeypatch.setenv("ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY", "0")
    assert patch.install_dspark_topp_verify_probs_patch() is False


def test_a_greedy_draft_does_not_install(monkeypatch, caplog):
    # Its top-p-only verify takes the draw rule (`dspark_draw_verify_patch`):
    # no fingerprints, no GPU self-check, SGLang's function stays.
    pytest.importorskip("sglang")
    monkeypatch.setenv("SGLANG_DSPARK_USE_GREEDY_DRAFT", "1")
    monkeypatch.setattr(
        patch, "_upstream_shape_error", lambda *a: pytest.fail("install checked")
    )
    with caplog.at_level("INFO", logger="atom"):
        assert patch.install_dspark_topp_verify_probs_patch() is False
    assert "SGLANG_DSPARK_USE_GREEDY_DRAFT=1" in caplog.text


def _sglang_modules():
    try:
        from sglang.kernels.ops.sampling import renorm_triton
        from sglang.kernels.ops.speculative.dspark import dspark_accept
        from sglang.srt.speculative import dflash_utils
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"SGLang DSpark modules unavailable: {e}")
    return dspark_accept, dflash_utils, renorm_triton


def test_guards_accept_the_supported_runtime():
    mods = _sglang_modules()
    import sglang

    if sglang.__version__ not in SUPPORTED_SGLANG_VERSIONS:
        pytest.skip("not SGLang 0.5.22")
    if (torch.__version__, torch.version.git_version) not in patch.PINNED_TORCH_BUILDS:
        pytest.skip("not the pinned PyTorch build")
    if not getattr(mods[0].build_speculative_verify_target_probs, patch._MARK, False):
        assert patch._upstream_shape_error(*mods) is None


def test_guards_refuse_other_builds_and_sources(monkeypatch):
    dspark_accept, dflash_utils, renorm_triton = _sglang_modules()
    pinned_version, pinned_git = patch.PINNED_TORCH_BUILDS[0]
    monkeypatch.setattr(torch, "__version__", "2.11.0")
    monkeypatch.setattr(torch.version, "git_version", pinned_git)
    assert "torch 2.11.0" in patch._upstream_shape_error(
        dspark_accept, dflash_utils, renorm_triton
    )
    # same version string, another build: refused on the git hash
    monkeypatch.setattr(torch, "__version__", pinned_version)
    monkeypatch.setattr(torch.version, "git_version", "0" * 40)
    reason = patch._upstream_shape_error(dspark_accept, dflash_utils, renorm_triton)
    assert reason is not None and "git 0000" in reason
    monkeypatch.setattr(torch.version, "git_version", pinned_git)
    fake_renorm = SimpleNamespace(
        __name__="renorm_triton",
        top_p_renorm_probs_triton=renorm_triton.top_p_renorm_probs_triton,
        _renorm_from_pivots=renorm_triton._renorm_from_pivots,
        _prepare_probs=renorm_triton._prepare_probs,
    )
    monkeypatch.setattr(dflash_utils, "top_p_renorm_prob", lambda p, t: p)
    assert "not renorm_triton" in patch._upstream_shape_error(
        dspark_accept, dflash_utils, fake_renorm
    )
    monkeypatch.setattr(
        dflash_utils, "top_p_renorm_prob", renorm_triton.top_p_renorm_probs_triton
    )
    if getattr(dspark_accept.build_speculative_verify_target_probs, patch._MARK, False):
        return  # installed in this process: the wrapper identity check fires first
    # a pinned function whose source changed (here: the pin itself differs)
    pins = dict(patch.PINNED_SOURCE_SHA256)
    pins["renorm_triton.top_p_renorm_probs_triton"] = "f" * 64
    monkeypatch.setattr(patch, "PINNED_SOURCE_SHA256", pins)
    reason = patch._upstream_shape_error(dspark_accept, dflash_utils, renorm_triton)
    assert reason is not None and "fingerprint mismatch" in reason
    assert "renorm_triton.top_p_renorm_probs_triton" in reason
    assert "_prepare_probs" not in reason


def _prepare_probs_with_an_extra_line(probs):
    probs = probs.clamp_min(0.0)
    return probs.float().contiguous()


def test_an_inserted_upstream_line_refuses_the_install(monkeypatch):
    dspark_accept, dflash_utils, renorm_triton = _sglang_modules()
    if getattr(dspark_accept.build_speculative_verify_target_probs, patch._MARK, False):
        pytest.skip("installed in this process")
    if patch._upstream_shape_error(dspark_accept, dflash_utils, renorm_triton):
        pytest.skip("not the pinned runtime")
    monkeypatch.setattr(
        renorm_triton, "_prepare_probs", _prepare_probs_with_an_extra_line
    )
    reason = patch._upstream_shape_error(dspark_accept, dflash_utils, renorm_triton)
    assert reason is not None and "renorm_triton._prepare_probs" in reason


def test_pins_cover_every_function_the_patch_relies_on():
    assert set(patch.PINNED_SOURCE_SHA256) == {
        "dspark_accept._accept_sampling_core",
        "dflash_utils.build_speculative_verify_target_probs",
        "dflash_utils._dflash_top_p_renorm_prob",
        "renorm_triton.top_p_renorm_probs_triton",
        "renorm_triton._prepare_probs",
    }
    assert all(len(v) == 64 for v in patch.PINNED_SOURCE_SHA256.values())


def test_source_fingerprint_sees_insertions_not_indentation():
    src = (
        "def f(probs, top_p):\n"
        "    probs_fp32 = _prepare_probs(probs)\n"
        "    return _renorm_from_pivots(probs_fp32, pivots)\n"
    )
    reindented = src.replace("    ", "        ")
    inserted = src.replace(
        "    return", "    top_p = top_p.clamp(0.0, 1.0)\n    return"
    )
    assert patch.source_fingerprint(src) == patch.source_fingerprint(reindented)
    assert patch.source_fingerprint(src) != patch.source_fingerprint(inserted)
    assert patch.source_fingerprint(src) != patch.source_fingerprint(
        src.replace("probs_fp32, pivots", "probs, pivots")
    )


def test_install_is_idempotent(monkeypatch):
    dspark_accept, _, _ = _sglang_modules()

    def marked(**kw):
        return None

    setattr(marked, patch._MARK, True)
    monkeypatch.setattr(dspark_accept, "build_speculative_verify_target_probs", marked)
    monkeypatch.setenv("ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY", "1")
    assert patch.install_dspark_topp_verify_probs_patch() is True
    assert dspark_accept.build_speculative_verify_target_probs is marked


# --------------------------------------------------------------------------- GPU


def _gpu_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    mods = _sglang_modules()
    return mods


def _rows(kind, rows, g):
    dev = "cuda"
    if kind == "lm":  # a few strong tokens over a noisy tail: the head settles it
        z = torch.randn(rows, V, device=dev, generator=g) * 2.0
        z[:, :50] += torch.linspace(12, 4, 50, device=dev)
        return z
    if kind == "zipf":  # heavy tail: the cut often lies below the head
        ranks = torch.arange(1, V + 1, device=dev, dtype=torch.float32)
        s = 0.6 + 0.8 * torch.rand(rows, 1, device=dev, generator=g)
        lp = -s * torch.log(ranks)[None, :]
        perm = torch.argsort(torch.rand(rows, V, device=dev, generator=g), -1)
        return torch.gather(lp, 1, perm)
    if kind == "flat":
        return torch.randn(rows, V, device=dev, generator=g) * 0.3
    if kind == "uniform":  # every entry tied
        return torch.zeros(rows, V, device=dev)
    if kind == "ties_at_cut":  # tied blocks across the 0.95 cut
        z = torch.full((rows, V), -30.0, device=dev)
        z[:, :10] = 0.0
        z[:, 10:30] = -3.1135  # e^-3.1135 * 20 ~ the mass that straddles 0.95
        return z
    if (
        kind == "tied_head_past_k"
    ):  # 2000 equal tops: the cut sits in a tie wider than the head
        z = torch.full((rows, V), -20.0, device=dev)
        z[:, :2000] = 0.0
        return z
    if kind == "onehot":
        z = torch.full((rows, V), -1e4, device=dev)
        z[:, 7] = 0.0
        return z
    raise ValueError(kind)


def _info_for(bs, top_ps, temps):
    return SimpleNamespace(
        need_top_k_sampling=False,
        need_top_p_sampling=True,
        temperatures=temps.view(bs, 1).float(),
        top_ps=top_ps.float(),
        top_ks=torch.full((bs,), V, device="cuda", dtype=torch.int32),
    )


def _boundary_top_ps(z, temps, bs):
    """1 - (one of PyTorch's own cdf values near the top of each request's first
    row, or the next float either side): the threshold sits on a cdf value."""
    rows = 6 * bs
    probs = torch.softmax(z / torch.repeat_interleave(temps.view(bs, 1), 6, dim=0), -1)
    cdf = torch.cumsum(torch.sort(probs, -1).values, -1)[
        ::6
    ]  # same row count, same scan chunk
    out = []
    for b in range(bs):
        c = cdf[b, V - 1 - (2 + 5 * b) % 37]
        step = (b % 3) - 1
        if step:
            c = torch.nextafter(c, torch.tensor(2.0 * step, device="cuda"))
        out.append(1.0 - c)
    assert rows == probs.shape[0]
    return torch.stack(out)


def _top_ps(mode, bs, z, temps):
    dev = "cuda"
    if mode == "0.95":
        return torch.full((bs,), 0.95, device=dev)
    if mode == "1.0":
        return torch.ones(bs, device=dev)
    if mode == "tiny":
        return torch.full((bs,), 1e-7, device=dev)
    if mode == "mixed":
        pat = [0.95, 1.0, 0.5, 0.999, 1e-6, 0.8, 0.9999999, 0.3]
        return torch.tensor([pat[i % len(pat)] for i in range(bs)], device=dev)
    if mode == "boundary":
        return _boundary_top_ps(z, temps, bs)
    raise ValueError(mode)


def _check(dspark_accept_original, bs, kind, mode, mixed_temp, g, expect_fallback=None):
    from sglang.kernels.ops.sampling.renorm_triton import _renorm_from_pivots

    rows = 6 * bs
    z = _rows(kind, rows, g)
    if mixed_temp:
        temps = torch.tensor(
            [[0.3, 0.6, 1.0, 1.7][i % 4] for i in range(bs)], device="cuda"
        )
    else:
        temps = torch.full((bs,), 0.6, device="cuda")
    info = _info_for(bs, _top_ps(mode, bs, z, temps), temps)
    counters = pivot.ToppPivotCounters("cuda")
    guard = patch._Guard(counters, enabled=False)
    build = patch.make_exact_topp_build(
        dspark_accept_original, _renorm_from_pivots, counters, guard
    )
    kw = {"next_token_logits": z, "sampling_info": info, "draft_token_num": 6, "bs": bs}
    ref = dspark_accept_original(**kw)
    out = build(**kw)
    assert counters.calls == 1, "the exact branch was not taken"
    assert out.shape == ref.shape == (bs, 6, V)
    diff = (out.view(torch.int32) != ref.view(torch.int32)).reshape(rows, V).any(-1)
    assert not bool(
        diff.any()
    ), f"{int(diff.sum())} of {rows} rows differ ({kind}, top_p {mode})"
    fb = int(counters.dev[2])
    if expect_fallback is True:
        assert fb > 0, "this table was meant to exercise the fallback"
    if expect_fallback is False:
        assert fb == 0
    return fb


@pytest.fixture(scope="module")
def original_build():
    _dspark_accept, dflash_utils, _ = _gpu_or_skip()
    import inspect

    return inspect.unwrap(dflash_utils.build_speculative_verify_target_probs)


_KINDS = ["lm", "zipf", "flat", "uniform", "ties_at_cut", "tied_head_past_k", "onehot"]
_MODES = ["0.95", "1.0", "tiny", "mixed", "boundary"]


@pytest.mark.parametrize("bs", [1, 8, 32, 50, 64])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("mode", _MODES)
def test_bitwise_sglang_top_p(original_build, bs, kind, mode):
    if mode == "boundary" and kind in ("uniform", "onehot"):
        pytest.skip("no distinct cdf values near the top")
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 * bs + 17 * _KINDS.index(kind) + _MODES.index(mode))
    _check(original_build, bs, kind, mode, mixed_temp=(bs % 2 == 0), g=g)


def test_the_head_settles_lm_rows(original_build):
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    _check(original_build, 32, "lm", "0.95", False, g, expect_fallback=False)


@pytest.mark.parametrize("kind", ["flat", "tied_head_past_k"])
def test_rows_needing_the_fallback(original_build, kind):
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    _check(original_build, 8, kind, "0.95", False, g, expect_fallback=True)


def test_threshold_inside_the_rounding_bracket_takes_the_fallback(original_build):
    g = torch.Generator(device="cuda")
    g.manual_seed(11)
    # zipf rows at a cdf value near the very top: the bracket cannot decide
    fb = _check(original_build, 8, "zipf", "boundary", False, g)
    assert fb > 0


def test_no_host_synchronisation(original_build):
    from sglang.kernels.ops.sampling.renorm_triton import _renorm_from_pivots

    bs = 32
    g = torch.Generator(device="cuda")
    g.manual_seed(3)
    z = _rows("lm", 6 * bs, g)
    z[5] = _rows("flat", 1, g)  # one row through the fallback too
    temps = torch.full((bs,), 0.6, device="cuda")
    info = _info_for(bs, torch.full((bs,), 0.95, device="cuda"), temps)
    counters = pivot.ToppPivotCounters("cuda")
    build = patch.make_exact_topp_build(
        original_build,
        _renorm_from_pivots,
        counters,
        patch._Guard(counters, enabled=True),
    )
    kw = {"next_token_logits": z, "sampling_info": info, "draft_token_num": 6, "bs": bs}
    build(**kw)  # compile / warm outside the check
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        try:  # positive control: the mode does catch a synchronising read here
            torch.ones(1, device="cuda").item()
        except RuntimeError:
            pass
        else:
            pytest.skip("sync debug mode does not catch .item() on this build")
        for _ in range(3):
            build(**kw)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    assert int(counters.dev[2]) >= 1


def _self_check_pair():
    _dspark_accept, dflash_utils, renorm_triton = _gpu_or_skip()
    import inspect

    original = inspect.unwrap(dflash_utils.build_speculative_verify_target_probs)
    counters = pivot.ToppPivotCounters("cuda")
    build = patch.make_exact_topp_build(
        original,
        renorm_triton._renorm_from_pivots,
        counters,
        patch._Guard(counters, enabled=False),
    )
    return original, build, counters


def test_self_check_covers_the_serving_shapes():
    # 6 and 192 rows at torch's chunk 1024, 258 at 512; the fallback replays rows at each
    original, build, counters = _self_check_pair()
    assert [pivot.torch_scan_chunk(6 * b, V) for b in patch.SELF_CHECK_BATCHES] == [
        1024,
        1024,
        512,
    ]
    assert 32 in patch.SELF_CHECK_BATCHES
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    for bs in patch.SELF_CHECK_BATCHES:
        z, temps, top_ps = patch._self_check_table(bs, V, g)
        assert top_ps.shape == (bs,) and top_ps.dtype == torch.float32
        before = int(counters.dev[2])
        info = SimpleNamespace(
            need_top_k_sampling=False,
            need_top_p_sampling=True,
            temperatures=temps,
            top_ps=top_ps,
        )
        kw = {"next_token_logits": z, "sampling_info": info, "draft_token_num": 6}
        a = original(bs=bs, **kw)
        b_ = build(bs=bs, **kw)
        assert torch.equal(a.view(torch.int32), b_.view(torch.int32))
        torch.cuda.synchronize()
        assert int(counters.dev[2]) - before >= (3 if bs == 1 else bs // 4)


@pytest.mark.parametrize("forced_chunk", [512, 1024])
def test_self_check_refuses_a_replay_at_the_wrong_scan_chunk(monkeypatch, forced_chunk):
    # A replay right at one scan chunk size and wrong at the other.
    original, build, _ = _self_check_pair()
    monkeypatch.setattr(pivot, "torch_scan_chunk", lambda rows, size: forced_chunk)
    mismatch = patch._self_check(build, original)
    assert mismatch is not None, mismatch
    # forced 512 is wrong at 6 and 192 rows, forced 1024 at 258 rows
    assert ("bs=32" in mismatch) == (forced_chunk == 512)
    assert ("bs=43" in mismatch) == (forced_chunk == 1024)


def test_install_self_check_passes_and_marks(monkeypatch, caplog):
    dspark_accept, dflash_utils, renorm_triton = _gpu_or_skip()
    import inspect

    original = inspect.unwrap(dflash_utils.build_speculative_verify_target_probs)
    monkeypatch.setattr(
        dspark_accept, "build_speculative_verify_target_probs", original
    )
    monkeypatch.setenv("ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY", "1")
    if (
        patch._upstream_shape_error(dspark_accept, dflash_utils, renorm_triton)
        is not None
    ):
        pytest.skip("not the supported runtime")
    monkeypatch.setenv("ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY_GUARD", "off")
    with caplog.at_level("INFO", logger="atom"):
        assert patch.install_dspark_topp_verify_probs_patch() is True
    assert "self-check bitwise equal; guard off" in caplog.text
    assert getattr(dspark_accept.build_speculative_verify_target_probs, patch._MARK)
    assert patch.install_dspark_topp_verify_probs_patch() is True  # idempotent


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *os.sys.argv[1:]]))
