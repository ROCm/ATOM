import logging

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

logger = logging.getLogger("atom.plugin.sglang.attention_backend.deepseek_v41")


class ATOMDeepseekV41BackendForSgl(AttentionBackend):
    """SGLang backend shim for ATOM-owned DeepSeek-V4.1 (CSA2) attention.

    SGLang needs a backend object for scheduling and forward-context
    publication: the plugin reads ``req_to_token_pool`` off it during the
    forward. The CSA2 pool, step metadata and kernels belong to ATOM
    (``deepseek_v41_bridge``). Nothing here computes attention.

    Eager steps are staged by the wrapper before the model runs. Decode CUDA
    graphs (``--cuda-graph-backend-decode full``) are staged here instead, in
    ``init_forward_metadata_out_graph``, which SGLang calls outside the
    captured region once per bucket at capture and before every replay.
    """

    needs_cpu_seq_lens = True

    def __init__(self, model_runner, *args, **kwargs):
        del args, kwargs
        logger.info("Initializing ATOMDeepseekV41BackendForSgl")
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        if not getattr(self.token_to_kv_pool, "is_atom_v41_proxy_pool", False):
            raise RuntimeError(
                "ATOMDeepseekV41BackendForSgl needs the ATOM DeepSeek-V4.1 proxy "
                f"KV pool, got {type(self.token_to_kv_pool).__name__}"
            )
        # bs -> (input_ids address, arena address) each bucket was recorded on;
        # None until a decode-graph runner sets the graph state up.
        self._v41_graph_buckets = None
        self._v41_replayed_buckets = set()
        # The last replay's metadata, for debugging: the graph reads fixed
        # addresses and no model Python runs on replay.
        self._v41_last_replay_md = None

    @staticmethod
    def get_name() -> str:
        return "dsv4"

    def init_forward_metadata(self, forward_batch):
        # The bridge derives ATOM's metadata from the batch inside the forward.
        self.forward_metadata = None

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        from sglang.srt.environ import envs

        from atom.plugin.sglang.deepseek_v41_bridge import (
            check_deepseek_v41_graph_capacity,
        )

        if envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get():
            # The glue graph would freeze this backend's host staging (the
            # slot / block-table reads and ATOM's publication) into a replay.
            raise NotImplementedError(
                "DeepSeek-V4.1 decode graphs stage every replay on the host; "
                "unset SGLANG_ENABLE_METADATA_GLUE_GRAPH"
            )
        check_deepseek_v41_graph_capacity(
            self.token_to_kv_pool,
            self.req_to_token_pool.req_to_token,
            max_bs,
            max_num_tokens,
        )
        self._v41_graph_buckets = {}
        self._v41_replayed_buckets = set()
        logger.info(
            "ATOM DeepSeek-V4.1 decode graphs: max_bs=%d max_num_tokens=%d, "
            "capture slots %d..%d",
            max_bs,
            max_num_tokens,
            self.token_to_kv_pool.capture_slot_base,
            self.token_to_kv_pool.num_slots - 1,
        )

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        from atom.plugin.sglang.deepseek_v41_bridge import (
            stage_deepseek_v41_graph_capture,
            stage_deepseek_v41_graph_replay,
        )

        self.forward_metadata = None
        mode = (
            getattr(forward_batch, "actual_forward_mode", None)
            or forward_batch.forward_mode
        )
        if not mode.is_decode():
            raise NotImplementedError(
                "DeepSeek-V4.1 graphs on the SGLang plugin cover plain decode "
                f"only, not {getattr(mode, 'name', mode)}; launch with "
                "--cuda-graph-backend-prefill disabled"
            )
        if self._v41_graph_buckets is None:
            raise RuntimeError(
                "DeepSeek-V4.1 graph metadata staged before init_cuda_graph_state"
            )
        pool = self.token_to_kv_pool
        bs = int(forward_batch.batch_size)
        if in_capture:
            stage_deepseek_v41_graph_capture(forward_batch, pool)
            self._v41_graph_buckets[bs] = (
                forward_batch.input_ids.data_ptr(),
                pool.raw_arena.data_ptr(),
            )
            return
        recorded = self._v41_graph_buckets.get(bs)
        if recorded is None:
            raise RuntimeError(f"DeepSeek-V4.1 decode graph bs={bs} was not captured")
        if recorded != (forward_batch.input_ids.data_ptr(), pool.raw_arena.data_ptr()):
            raise RuntimeError(
                f"DeepSeek-V4.1 decode graph bs={bs} replays on addresses its "
                "capture did not record"
            )
        self._v41_last_replay_md = stage_deepseek_v41_graph_replay(
            forward_batch, self.req_to_token_pool.req_to_token
        )
        if bs not in self._v41_replayed_buckets:
            self._v41_replayed_buckets.add(bs)
            logger.info(
                "V4.1 decode graph replay bs=%d raw=%d",
                bs,
                bs - int(getattr(forward_batch, "num_padding", 0) or 0),
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, *args, **kwargs):
        raise RuntimeError(
            "ATOMDeepseekV41BackendForSgl.forward_decode must not run: "
            "DeepSeek-V4.1 attention executes inside the ATOM model"
        )

    def forward_extend(self, *args, **kwargs):
        raise RuntimeError(
            "ATOMDeepseekV41BackendForSgl.forward_extend must not run: "
            "DeepSeek-V4.1 attention executes inside the ATOM model"
        )
