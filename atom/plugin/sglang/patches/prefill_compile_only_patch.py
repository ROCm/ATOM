"""Run SGLang tc_piecewise prefill as Inductor compile-only.

SGLang currently couples its prefill ``tc_piecewise`` compiler to per-bucket
CUDA Graph capture and static input staging.  This optional plugin patch keeps
the existing FX split/Inductor path, including ATOM's attention split ops, but
executes the dynamic compiled callable directly against the live ForwardBatch.

Enable with ``ATOM_SGLANG_PREFILL_COMPILE_ONLY=1``.  The patch intentionally
uses the existing ``tc_piecewise`` backend name so no SGLang source or CLI
schema change is required.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from atom.plugin.sglang.patches.triton_kernel_retention_patch import (
    retain_triton_kernels,
)
from atom.utils import envs

logger = logging.getLogger("atom.plugin.sglang.prefill_compile_only")
_COMPILE_ONLY_PREFILL_ACTIVE = False


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
        if (
            model_runner.server_args.cuda_graph_config.prefill.backend
            != Backend.TC_PIECEWISE
        ):
            return original_runner_init(self, model_runner)

        # TcPiecewise compiles during runner construction. Set the branch before
        # the first Dynamo trace so it sees native returning attention ops
        # immediately instead of recompiling on the first live request.
        with _compile_only_prefill_scope():
            original_runner_init(self, model_runner)

    def build_compile_only_config(server_args):
        prefill_config = server_args.cuda_graph_config.prefill
        original_compiler = prefill_config.tc_compiler
        try:
            prefill_config.tc_compiler = "inductor"
            compile_config = original_build_config(server_args)
        finally:
            prefill_config.tc_compiler = original_compiler

        # With no capture sizes, CUDAPiecewiseBackend runs only its general
        # dynamic-shape Inductor artifact and never captures a CUDA Graph.
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
