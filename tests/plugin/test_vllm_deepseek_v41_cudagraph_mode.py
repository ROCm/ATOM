"""What V4.1 does to ``cudagraph_mode``, and why it is PIECEWISE and not FULL.

The interesting case is not "graphs on": it is that the mode must land on
PIECEWISE *exactly*. ``eager_break_during_capture`` skips the break when the
forward context reports a FULL runtime mode, so a mode carrying FULL would
record V4.1's per-step Engram and state work into the graph and never run it
again -- a frozen cursor, every replayed step re-answering the first, with
nothing raised. A test that only asserted "not NONE" would pass on the one
setting that is silently wrong.
"""

import pytest

from atom.plugin.vllm import platform as atom_platform

CUDAGraphMode = pytest.importorskip("vllm.config").CUDAGraphMode


class _Compilation:
    def __init__(self, mode):
        self.cudagraph_mode = mode


class _Model:
    architectures = ("DeepseekV41ForCausalLM",)


class _Config:
    def __init__(self, mode):
        self.model_config = _Model()
        self.compilation_config = _Compilation(mode)
        self.cache_config = None
        self.speculative_config = None
        self.kv_transfer_config = None


@pytest.fixture
def breakable(monkeypatch):
    def _set(available):
        monkeypatch.setattr(
            atom_platform, "_breakable_cudagraph_available", lambda: available
        )

    return _set


@pytest.mark.parametrize(
    "requested",
    [
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.PIECEWISE,
    ],
)
def test_breakable_capture_lands_on_piecewise_from_every_requested_mode(
    breakable, requested
):
    breakable(True)
    cfg = _Config(requested)
    atom_platform.enforce_deepseek_v41_constraints(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_without_breakable_capture_the_mode_is_none(breakable):
    breakable(False)
    cfg = _Config(CUDAGraphMode.FULL_AND_PIECEWISE)
    atom_platform.enforce_deepseek_v41_constraints(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_an_explicit_none_is_left_alone_even_with_breakable_capture(breakable):
    # Asking for eager is a legitimate request -- it is how the two arms of an
    # on/off comparison are held comparable -- and must not be overridden into
    # graphs just because the env var happens to be set.
    breakable(True)
    cfg = _Config(CUDAGraphMode.NONE)
    atom_platform.enforce_deepseek_v41_constraints(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_the_constraint_is_idempotent(breakable):
    breakable(True)
    cfg = _Config(CUDAGraphMode.FULL_AND_PIECEWISE)
    atom_platform.enforce_deepseek_v41_constraints(cfg)
    atom_platform.enforce_deepseek_v41_constraints(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_a_non_v41_model_is_untouched(breakable):
    breakable(True)
    cfg = _Config(CUDAGraphMode.FULL_AND_PIECEWISE)
    cfg.model_config.architectures = ("DeepseekV4ForCausalLM",)
    atom_platform.enforce_deepseek_v41_constraints(cfg)
    assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE


class TestTheExperimentalGateIsClosedByDefault:
    """vLLM auto-enables VLLM_USE_BREAKABLE_CUDAGRAPH for some architectures.

    V4.1 under capture serves without raising and answers noise, so "capture is
    available" must not be the whole condition -- otherwise a vLLM upgrade that
    adds V4.1 to that auto-enable list turns correct answers into wrong ones
    with nothing in the log. The opt-in env var is what stands in the way.
    """

    def test_breakable_capture_alone_does_not_open_it(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
        monkeypatch.delenv(atom_platform._V41_EXPERIMENTAL_CUDAGRAPH_ENV, raising=False)
        assert atom_platform._breakable_cudagraph_available() is False

    def test_and_so_the_mode_is_still_none(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
        monkeypatch.delenv(atom_platform._V41_EXPERIMENTAL_CUDAGRAPH_ENV, raising=False)
        cfg = _Config(CUDAGraphMode.FULL_AND_PIECEWISE)
        atom_platform.enforce_deepseek_v41_constraints(cfg)
        assert cfg.compilation_config.cudagraph_mode == CUDAGraphMode.NONE

    def test_the_opt_in_alone_does_not_open_it_either(self, monkeypatch):
        # Both halves are required: without breakable capture there is no
        # break around the step work at all, which is a different and worse
        # failure than the one the opt-in is for.
        monkeypatch.setenv(atom_platform._V41_EXPERIMENTAL_CUDAGRAPH_ENV, "1")
        monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
        assert atom_platform._breakable_cudagraph_available() is False


def test_the_forward_context_does_not_open_atoms_side_stream_fork():
    """`in_hipgraph` gates a fork that only ATOM's own capture loop may take.

    `side_stream` forks to a side stream when `in_hipgraph` is true. Its
    contract is ATOM's capture loop -- the window where ATOM owns the thread
    and the capture -- which natively coincides with "a graph is recording"
    and under the plugin does not: vLLM captures on its own thread, and a fork
    opened there ends with

        HIP error: attempt to terminate a thread-local capture sequence
        from another thread

    at `capture_end()`. So the plugin's forward context reports False, and
    this test is what keeps a future edit from "fixing" it back to
    `_v41_capture_active()` because the name reads like a status.
    """
    import inspect

    from atom.plugin.vllm import deepseek_v41_bridge as bridge

    source = inspect.getsource(bridge)
    assert "in_hipgraph=False," in source, (
        "the V4.1 plugin forward context must not report in_hipgraph true: it "
        "gates ATOM's side-stream fork, which vLLM's capture cannot end"
    )
    assert "in_hipgraph=_v41_capture_active()" not in source
