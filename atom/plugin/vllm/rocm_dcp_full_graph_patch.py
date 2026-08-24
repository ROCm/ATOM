"""Allow ROCm DCP decode batches to keep full CUDA graphs."""

import logging

logger = logging.getLogger("atom")


def apply_vllm_rocm_dcp_full_graph_patch() -> None:
    """Remove vLLM's blanket DCP-to-PIECEWISE downgrade on ROCm.

    The original platform validation still runs, preserving all other defaults
    and constraints. DCP is masked only while that validation executes, so its
    blanket downgrade is skipped while PCP remains PIECEWISE.
    """
    from vllm.platforms.rocm import RocmPlatform

    if getattr(RocmPlatform, "_atom_dcp_full_graph_patch", False):
        return

    original_check = RocmPlatform.check_and_update_config

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        compilation_config = vllm_config.compilation_config
        parallel_config = vllm_config.parallel_config
        requested_mode = compilation_config.cudagraph_mode
        preserve_dcp_full_graph = (
            requested_mode.has_full_cudagraphs()
            and parallel_config.decode_context_parallel_size > 1
            and parallel_config.prefill_context_parallel_size == 1
        )

        if not preserve_dcp_full_graph:
            original_check(vllm_config)
            return

        dcp_size = parallel_config.decode_context_parallel_size
        logger.info(
            "ATOM patch: allowing cudagraph_mode=%s for ROCm DCP%d.",
            requested_mode.name,
            dcp_size,
        )
        parallel_config.decode_context_parallel_size = 1
        try:
            original_check(vllm_config)
        finally:
            parallel_config.decode_context_parallel_size = dcp_size

    RocmPlatform.check_and_update_config = check_and_update_config
    RocmPlatform._atom_dcp_full_graph_patch = True
