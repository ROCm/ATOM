"""Run SGLang prefill as Inductor compile-only.

SGLang 0.5.22 rejects ``--cuda-graph-backend-prefill tc_piecewise``. The Qwen
accuracy case launches with the supported ``breakable`` prefill backend, which
still carries the prefill capture sizes. When
``ATOM_SGLANG_PREFILL_COMPILE_ONLY=1`` is set, this patch rewrites that
published config to the internal piecewise backend, forces Inductor on the
prefill config that 0.5.22 actually reads, and executes the compiled callable
against the live ForwardBatch instead of capturing per-bucket CUDA graphs.
Capturing the 8192/16384 prefill graphs during server startup makes the ROCm
worker exit before it is ready.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from atom.plugin.sglang.patches.triton_kernel_retention_patch import (
    retain_triton_kernels,
)
from atom.utils import envs

logger = logging.getLogger("atom.plugin.sglang.prefill_compile_only")
_COMPILE_ONLY_PREFILL_ACTIVE = False


def resolve_prefill_cuda_graph_config(server_args):
    """Prefill CUDA graph config for this process.

    SGLang 0.5.20 publishes the resolved config on ``get_exec().graph`` and
    leaves ``server_args.cuda_graph_config`` as None. Older call sites still
    pass the config on ``server_args``.
    """

    cfg = getattr(server_args, "cuda_graph_config", None)
    prefill = getattr(cfg, "prefill", None) if cfg is not None else None
    if prefill is not None:
        return prefill
    try:
        from sglang.srt.runtime_context import get_exec

        graph = getattr(get_exec(), "graph", None)
        cfg = getattr(graph, "cuda_graph_config", None)
    except Exception:  # noqa: BLE001 - published config is optional
        return None
    if cfg is None:
        return None
    return getattr(cfg, "prefill", None)


def is_compile_only_prefill_active() -> bool:
    """Return whether the current tc_piecewise execution skips CUDA Graphs.

    Compile-only execution still enters SGLang's tc_piecewise replay session so
    its trampoline dispatches to the compiled callable. Consequently,
    ``is_in_tc_piecewise_cuda_graph()`` is true even though no CUDA Graph exists.
    Consumers use this second flag to distinguish compile-only execution from a
    real piecewise CUDA Graph with fixed cross-segment tensor addresses.
    """

    return _COMPILE_ONLY_PREFILL_ACTIVE


@contextmanager
def _compile_only_prefill_scope() -> Iterator[None]:
    """Mark compile-only tracing, warmup, or execution for model-side policy."""

    global _COMPILE_ONLY_PREFILL_ACTIVE
    previous = _COMPILE_ONLY_PREFILL_ACTIVE
    _COMPILE_ONLY_PREFILL_ACTIVE = True
    try:
        yield
    finally:
        _COMPILE_ONLY_PREFILL_ACTIVE = previous


def _published_prefill_config():
    try:
        from sglang.srt.runtime_context import get_exec

        graph = getattr(get_exec(), "graph", None)
        cfg = getattr(graph, "cuda_graph_config", None)
    except Exception:  # noqa: BLE001 - published config is optional
        return None
    if cfg is None:
        return None
    return getattr(cfg, "prefill", None)


def _prefill_config_read_by_trace(server_args):
    """Prefill config that 0.5.22 reads while compiling and tracing.

    ``TcPiecewiseCudaGraphBackend.build_compilation_config`` and the later
    ``fp8_utils`` forward both read
    ``get_exec().graph.cuda_graph_config.prefill``. That object is not
    ``server_args.cuda_graph_config.prefill`` once the runtime has published
    its own copy. Older call sites still only have the server_args object.
    """

    published = _published_prefill_config()
    if published is not None:
        return published
    return resolve_prefill_cuda_graph_config(server_args)


def _pin_compile_only_inductor(server_args) -> None:
    """Leave ``tc_compiler=inductor`` on the config the trace reads.

    0.5.22's default is eager, and the CLI rejects ``tc_compiler``. The
    piecewise compile reads the field while building the compilation config,
    then the forward that traces activation quantization reads it again after
    that function returns. Restoring eager in between skips the PyTorch quant
    fusion. Compile-only mode keeps inductor for the whole trace.
    """

    prefill = _prefill_config_read_by_trace(server_args)
    if prefill is not None:
        prefill.tc_compiler = "inductor"


def _promote_compile_only_prefill(server_args, backend_cls):
    """Point a supported prefill backend at the internal piecewise compiler.

    The runner reads ``get_exec().graph.cuda_graph_config.prefill``. Older
    call sites still pass a config on ``server_args``, so both objects are
    rewritten when they still say ``breakable``.
    """

    configs = []
    for prefill in (
        resolve_prefill_cuda_graph_config(server_args),
        _published_prefill_config(),
    ):
        if prefill is not None and all(prefill is not item for item in configs):
            configs.append(prefill)
    breakable = getattr(backend_cls, "BREAKABLE", "breakable")
    for prefill in configs:
        if getattr(prefill, "backend", None) == breakable:
            prefill.backend = backend_cls.TC_PIECEWISE
    return configs[-1] if configs else None


def _install_piecewise_prefill_resolver(backend_cls, piecewise_backend_cls) -> None:
    """Let the 0.5.22 runner construct the piecewise backend by name.

    ``resolve_prefill_backend`` only builds ``breakable`` and ``full``. The
    prefill runner imports that function by name, so the runner module has to
    see the piecewise constructor as well.
    """

    try:
        from sglang.srt.model_executor.runner_backend import utils as backend_utils
    except ImportError as exc:
        logger.info("SGLang piecewise prefill resolver patch skipped: %s", exc)
        return
    original_resolve = getattr(backend_utils, "resolve_prefill_backend", None)
    if original_resolve is None or getattr(
        original_resolve, "_atom_compile_only", False
    ):
        return

    def resolve_prefill_backend(cuda_graph_runner):
        if (
            getattr(cuda_graph_runner, "prefill_backend_name", None)
            == backend_cls.TC_PIECEWISE
        ):
            return piecewise_backend_cls(cuda_graph_runner)
        return original_resolve(cuda_graph_runner)

    resolve_prefill_backend._atom_compile_only = True
    backend_utils.resolve_prefill_backend = resolve_prefill_backend
    runner_module = sys.modules.get(
        "sglang.srt.model_executor.runner.prefill_cuda_graph_runner"
    )
    if runner_module is not None and hasattr(runner_module, "resolve_prefill_backend"):
        runner_module.resolve_prefill_backend = resolve_prefill_backend


def apply_prefill_compile_only_patch() -> bool:
    """Install the compile-only behavior when explicitly enabled."""

    if not envs.ATOM_SGLANG_PREFILL_COMPILE_ONLY:
        return False

    try:
        from sglang.srt.compilation.backend import SGLangBackend
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )
        from sglang.srt.model_executor.runner.shape_key import ShapeKey
        from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
            TcPiecewiseCudaGraphBackend,
        )
    except ImportError:
        logger.exception(
            "ATOM prefill compile-only mode requires a compatible SGLang "
            "tc_piecewise backend"
        )
        return False

    _install_piecewise_prefill_resolver(Backend, TcPiecewiseCudaGraphBackend)

    if getattr(PrefillCudaGraphRunner, "_atom_compile_only_patched", False):
        return False

    original_runner_init = PrefillCudaGraphRunner.__init__
    original_sglang_backend_init = SGLangBackend.__init__
    original_build_config = TcPiecewiseCudaGraphBackend.build_compilation_config
    original_capture = PrefillCudaGraphRunner.capture
    original_execute = PrefillCudaGraphRunner.execute
    original_can_replay_locally = PrefillCudaGraphRunner.can_replay_locally

    def init_native_functionalization_backend(self, *args, **kwargs):
        original_sglang_backend_init(self, *args, **kwargs)
        if is_compile_only_prefill_active():
            # Native ATOM leaves functionalization v2 enabled. SGLang normally
            # disables it, which prevents the GDN RMSNorm/SILU producer from
            # becoming one persistent kernel before aiter.gemm_a16w16.
            self.inductor_config["enable_auto_functionalized_v2"] = True

    def is_tc_piecewise_runner(runner) -> bool:
        return runner.prefill_backend_name == Backend.TC_PIECEWISE

    def init_compile_only_runner(self, model_runner):
        prefill = _promote_compile_only_prefill(model_runner.server_args, Backend)
        if prefill is None or getattr(prefill, "backend", None) != Backend.TC_PIECEWISE:
            return original_runner_init(self, model_runner)

        # Pin inductor before runner construction. The backend builds its
        # compilation config and then traces; fp8 quantization reads the same
        # published field during that later forward.
        _pin_compile_only_inductor(model_runner.server_args)
        # TcPiecewise compiles during runner construction. Set the branch before
        # the first Dynamo trace so it sees native returning attention ops
        # immediately instead of recompiling on the first live request.
        with _compile_only_prefill_scope():
            original_runner_init(self, model_runner)

    def build_compile_only_config(server_args):
        prefill_config = _prefill_config_read_by_trace(server_args)
        if prefill_config is None:
            return original_build_config(server_args)
        # Do not restore the previous compiler. The trace that decides whether
        # activation quantization fuses into PyTorch ops runs after this
        # function returns, and it reads the same published field.
        prefill_config.tc_compiler = "inductor"
        compile_config = original_build_config(server_args)

        # With no capture sizes, CUDAPiecewiseBackend runs only its general
        # dynamic-shape Inductor artifact and never captures a CUDA Graph.
        if compile_config is not None:
            compile_config.capture_sizes = []
        return compile_config

    def capture_compile_only(self):
        if not is_tc_piecewise_runner(self):
            return original_capture(self)

        # Backend construction has already installed and activated the dynamic
        # Inductor artifact. Retain the runner's one-time kernel warmup, but do
        # not enter graph_capture() or stage static inputs.
        with _compile_only_prefill_scope(), retain_triton_kernels():
            self.warmup()

    def can_replay_locally_compile_only(self, **kwargs):
        if not is_tc_piecewise_runner(self):
            return original_can_replay_locally(self, **kwargs)

        num_tokens = kwargs.get("num_tokens")
        # Preserve all existing feature/metadata eligibility checks, but remove
        # the CUDA-Graph-specific bucket-padding-factor rejection.
        without_shape = dict(kwargs)
        without_shape["num_tokens"] = None
        if not original_can_replay_locally(self, **without_shape):
            return False
        return num_tokens is None or num_tokens <= self.max_num_tokens

    def execute_compile_only(self, forward_batch, **kwargs):
        if not is_tc_piecewise_runner(self):
            return original_execute(self, forward_batch, **kwargs)

        self._validate_capture_hidden_mode(forward_batch)
        self.raw_num_tokens = len(forward_batch.input_ids)
        self.raw_bs = forward_batch.batch_size

        with (
            _compile_only_prefill_scope(),
            retain_triton_kernels(),
            self.backend.replay_session(),
        ):
            # Unlike CUDA Graph replay, compile-only execution consumes the
            # live batch. Rebuild attention/GDN metadata for it and avoid
            # load_batch(), which performs bucket padding and static copies.
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
            with self._prefill_forward_context(
                forward_batch,
                num_tokens=self.raw_num_tokens,
                raw_num_tokens=self.raw_num_tokens,
            ):
                output = self.backend.replay(
                    ShapeKey(size=self.raw_num_tokens),
                    forward_batch,
                    **kwargs,
                )
        return self._finalize_execute_output(output)

    TcPiecewiseCudaGraphBackend.build_compilation_config = staticmethod(
        build_compile_only_config
    )
    SGLangBackend.__init__ = init_native_functionalization_backend
    PrefillCudaGraphRunner.__init__ = init_compile_only_runner
    PrefillCudaGraphRunner.capture = capture_compile_only
    PrefillCudaGraphRunner.can_replay_locally = can_replay_locally_compile_only
    PrefillCudaGraphRunner.execute = execute_compile_only
    PrefillCudaGraphRunner._atom_compile_only_patched = True

    logger.warning(
        "ATOM SGLang prefill compile-only mode enabled: using dynamic Inductor "
        "artifacts without prefill CUDA Graph capture, bucket padding, or "
        "static-buffer staging"
    )
    return True
