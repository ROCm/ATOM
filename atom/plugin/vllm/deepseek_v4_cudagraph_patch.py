"""Keep ATOM's DeepSeek-V4 on the CUDA-graph path it was built for.

For ``DeepseekV4ForCausalLM`` / ``DeepSeekV4MTPModel`` vLLM's
``VllmConfig.__post_init__`` sets ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` when the
variable is unset, and the breakable cudagraph then switches the compilation
mode to NONE. That default is meant for vLLM's own V4. The ATOM V4 plugin
model compiles with ATOM's torch.compile and splits at the V4 attention op
(``v4_attention_with_output``), which runs eagerly between the PIECEWISE
graphs and reads the metadata of the current batch. Under the breakable
cudagraph the whole forward is captured instead: the V4 attention (which has
no eager break) is recorded together with the capture batch's metadata, and
every replay of a prefill / mixed batch reuses it. The output is wrong.

So, before vLLM decides, the plugin opts ATOM's V4 out of the breakable
cudagraph. Only when the architecture resolves to the ATOM class in vLLM's
model registry; every other model (Kimi-K3, MiniMax-M3 rely on the breakable
cudagraph) is left alone, and an explicit user setting is kept.

Second, vLLM's V1 runner decides "uniform decode" (and so the FULL decode
graph) from the batch shape alone. A step where every row has
``uniform_decode_query_len`` tokens but some row is still in its prompt -- the
1-token tail after a prefix hit, a 1-token last chunk, a fresh 1-token prompt,
2 tokens with MTP k=1 -- replays the FULL graph, while the V4 bridge builds
it as PREFILL and leaves the persistent decode buffers the graph reads
untouched. For ATOM's V4 the runner's dispatch is told such a step is not a
uniform decode (what vLLM's Model Runner V2 does since vllm#51865), and the
V4 builder refuses a FULL step it did not build as DECODE.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import logging
import os
import threading

logger = logging.getLogger("atom")

BREAKABLE_ENV = "VLLM_USE_BREAKABLE_CUDAGRAPH"

# The V4 architectures vLLM auto-enables the breakable cudagraph for and the
# plugin overrides with ATOM's model.
_V4_BREAKABLE_ARCHES = ("DeepseekV4ForCausalLM", "DeepSeekV4MTPModel")


def _plugin_disabled() -> bool:
    from atom.utils import envs

    return bool(envs.ATOM_DISABLE_VLLM_PLUGIN)


def _resolves_to_atom(arch: str) -> bool:
    """True iff vLLM's model registry maps ``arch`` to an ATOM class."""
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception:  # noqa: BLE001
        return False
    entry = ModelRegistry.models.get(arch)
    if entry is None:
        return False
    module_name = getattr(entry, "module_name", None)
    if module_name is None:
        module_name = getattr(getattr(entry, "model_cls", None), "__module__", "")
    return module_name == "atom" or module_name.startswith("atom.")


def atom_v4_arch(architectures) -> str | None:
    """The ATOM-served V4 architecture among ``architectures``, or None."""
    for arch in architectures or ():
        if arch in _V4_BREAKABLE_ARCHES and _resolves_to_atom(arch):
            return arch
    return None


def _opt_out_of_breakable_cudagraph(vllm_config) -> None:
    if _plugin_disabled():
        return
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return
    arch = atom_v4_arch(getattr(model_config, "architectures", None))
    if arch is None:
        return
    if BREAKABLE_ENV not in os.environ:
        os.environ[BREAKABLE_ENV] = "0"
        logger.info(
            "[atom-v4-cg] %s is served by ATOM: setting %s=0 (pid=%d). ATOM's V4 "
            "runs prefill / mixed batches on ATOM's PIECEWISE graphs with the V4 "
            "attention eager; vLLM's breakable cudagraph would capture that "
            "attention with the capture batch's metadata.",
            arch,
            BREAKABLE_ENV,
            os.getpid(),
        )
    elif os.environ[BREAKABLE_ENV] == "1":
        logger.warning(
            "[atom-v4-cg] %s=1 is set explicitly for %s served by ATOM. The "
            "breakable cudagraph captures the V4 attention with the capture "
            "batch's metadata: prefill and mixed batches up to the largest "
            "capture size will produce wrong output. Unset it (or set 0).",
            BREAKABLE_ENV,
            arch,
        )


def apply_vllm_v4_breakable_cudagraph_opt_out() -> bool:
    """Wrap ``VllmConfig.__post_init__`` so ATOM's V4 opts out before vLLM's
    auto-enable runs. ``VllmConfig`` is a pydantic dataclass; pydantic calls
    ``__post_init__`` through the class, so the class-level wrap is honoured.
    Idempotent."""
    try:
        from vllm.config import VllmConfig
    except Exception as e:  # noqa: BLE001
        logger.debug("[atom-v4-cg] VllmConfig unavailable (%s), skip", e)
        return False
    original = VllmConfig.__post_init__
    if getattr(original, "_atom_v4_breakable_opt_out", False):
        return False

    @functools.wraps(original)
    def __post_init__(self, *args, **kwargs):
        _opt_out_of_breakable_cudagraph(self)
        return original(self, *args, **kwargs)

    __post_init__._atom_v4_breakable_opt_out = True  # type: ignore[attr-defined]
    VllmConfig.__post_init__ = __post_init__
    return True


# Test-only dispatch counters, one JSON file per process; off unless set.
_INSTR_DIR = os.environ.get("ATOM_V4_CG_INSTR_DIR") or None
_INSTR_WRITE_EVERY = 64


def _new_counters() -> dict:
    return {
        "pid": os.getpid(),
        "steps": 0,
        "modes": {},
        "prefilling_steps": 0,
        "flipped": 0,
        "flipped_modes": {},
        "guard_trips": 0,
    }


_INSTR = _new_counters()


def _instr_write() -> None:
    try:
        path = os.path.join(_INSTR_DIR, f"cg_{os.getpid()}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(_INSTR, f)
        os.replace(path + ".tmp", path)
    except OSError as e:
        logger.warning("[atom-v4-cg] cannot write counters to %s: %s", _INSTR_DIR, e)


def _instr_step(mode, prefilling: bool, flipped: bool) -> None:
    name = getattr(mode, "name", str(mode))
    _INSTR["steps"] += 1
    _INSTR["modes"][name] = _INSTR["modes"].get(name, 0) + 1
    _INSTR["prefilling_steps"] += int(prefilling)
    if flipped:
        _INSTR["flipped"] += 1
        _INSTR["flipped_modes"][name] = _INSTR["flipped_modes"].get(name, 0) + 1
    if flipped or _INSTR["steps"] % _INSTR_WRITE_EVERY == 0:
        _instr_write()


_dispatch_local = threading.local()


def current_v4_dispatch():
    """``(cudagraph_mode, flipped)`` of the step whose attention metadata is
    being built, or None outside the target build of a real step."""
    return getattr(_dispatch_local, "dispatch", None)


@contextlib.contextmanager
def _armed_dispatch(mode, flipped: bool):
    prev = current_v4_dispatch()
    _dispatch_local.dispatch = None if mode is None else (mode, flipped)
    try:
        yield
    finally:
        _dispatch_local.dispatch = prev


def check_v4_dispatch_consistency(state, common_attn_metadata) -> None:
    """Raise if vLLM replays the FULL decode graph for a step the V4 bridge
    built as anything but DECODE (the graph would read stale decode buffers).
    CPU scalars only."""
    armed = current_v4_dispatch()
    if armed is None:
        return
    mode, flipped = armed
    if getattr(mode, "name", None) != "FULL":
        return
    from atom.utils.forward_context import AttnState

    if state == AttnState.DECODE:
        return
    num_reqs = int(getattr(common_attn_metadata, "num_reqs", 0) or 0)
    is_pref = getattr(common_attn_metadata, "is_prefilling", None)
    any_pref = None if is_pref is None else bool(is_pref[:num_reqs].any())
    if _INSTR_DIR:
        _INSTR["guard_trips"] += 1
        _instr_write()
    raise RuntimeError(
        "ATOM DeepSeek-V4: vLLM dispatched the FULL decode cudagraph for a step "
        f"the V4 metadata builder classified as {getattr(state, 'name', state)} "
        f"(num_reqs={num_reqs}, "
        f"max_query_len={getattr(common_attn_metadata, 'max_query_len', None)}, "
        f"num_actual_tokens={getattr(common_attn_metadata, 'num_actual_tokens', None)}, "
        f"any_prefilling={any_pref}, uniform_decode_overridden={flipped}). "
        "Replaying the decode graph would read decode buffers this step did not "
        "write."
    )


def _runner_is_atom_v4(runner) -> bool:
    cached = runner.__dict__.get("_atom_v4_runner")
    if cached is None:
        model_config = getattr(runner, "model_config", None)
        cached = not _plugin_disabled() and (
            atom_v4_arch(getattr(model_config, "architectures", None)) is not None
        )
        runner.__dict__["_atom_v4_runner"] = cached
    return cached


def apply_vllm_v4_prefill_not_uniform_patch() -> bool:
    """Wrap V1 ``GPUModelRunner._determine_batch_execution_and_padding`` and
    ``_build_attention_metadata`` for ATOM's V4. Idempotent."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        logger.debug("[atom-v4-cg] GPUModelRunner unavailable (%s), skip", e)
        return False

    dispatch = GPUModelRunner._determine_batch_execution_and_padding
    build = GPUModelRunner._build_attention_metadata
    if getattr(dispatch, "_atom_v4_prefill_not_uniform", False):
        return False

    param_names = [p for p in inspect.signature(dispatch).parameters if p != "self"]

    @functools.wraps(dispatch)
    def _determine_batch_execution_and_padding(self, *args, **kwargs):
        if not _runner_is_atom_v4(self):
            return dispatch(self, *args, **kwargs)
        if args:
            kwargs = {**dict(zip(param_names, args)), **kwargs}
        # execute_model leaves force_uniform_decode unset; _dummy_run and the
        # capture pass a bool and are not touched.
        if kwargs.get("force_uniform_decode") is not None:
            self.__dict__["_atom_v4_step_dispatch"] = None
            return dispatch(self, **kwargs)
        num_reqs = int(kwargs["num_reqs"])
        batch = self.input_batch
        # Same arrays and predicate as is_prefilling in _build_attention_metadata,
        # which the V4 bridge's _is_pure_uniform_decode rejects DECODE on.
        prefilling = bool(
            (
                batch.num_computed_tokens_cpu[:num_reqs]
                < batch.num_prompt_tokens[:num_reqs]
            ).any()
        )
        flipped = False
        if prefilling:
            flipped = self._is_uniform_decode(
                max_num_scheduled_tokens=kwargs["max_num_scheduled_tokens"],
                uniform_decode_query_len=self.uniform_decode_query_len,
                num_tokens=kwargs["num_tokens"],
                num_reqs=num_reqs,
            )
            kwargs["force_uniform_decode"] = False
        result = dispatch(self, **kwargs)
        self.__dict__["_atom_v4_step_dispatch"] = (result[0], flipped)
        if _INSTR_DIR:
            _instr_step(result[0], prefilling, flipped)
        return result

    @functools.wraps(build)
    def _build_attention_metadata(self, *args, **kwargs):
        pending = self.__dict__.pop("_atom_v4_step_dispatch", None)
        mode, flipped = pending if pending is not None else (None, False)
        with _armed_dispatch(mode, flipped):
            return build(self, *args, **kwargs)

    _determine_batch_execution_and_padding._atom_v4_prefill_not_uniform = True  # type: ignore[attr-defined]
    GPUModelRunner._determine_batch_execution_and_padding = (
        _determine_batch_execution_and_padding
    )
    GPUModelRunner._build_attention_metadata = _build_attention_metadata
    return True
