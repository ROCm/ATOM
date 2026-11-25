import logging

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

logger = logging.getLogger("atom.plugin.sglang.attention_backend.deepseek_v41")


class ATOMDeepseekV41BackendForSgl(AttentionBackend):
    """SGLang backend shim for ATOM-owned DeepSeek-V4.1 (CSA2) attention.

    It publishes ``req_to_token_pool`` to the forward; the pool, the step
    metadata and the kernels are ATOM's (``deepseek_v41_bridge``).

    Eager steps are staged by the wrapper before the model runs. Decode CUDA
    graphs (``--cuda-graph-backend-decode full``) are staged here instead, in
    ``init_forward_metadata_out_graph``, which SGLang calls outside the
    captured region once per bucket at capture and before every replay.

    The bridge reads the host ``seq_lens`` mirror that SGLang's default
    ``needs_cpu_seq_lens`` (True) publishes.
    """

    def __init__(self, model_runner, *args, **kwargs):
        del args, kwargs
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        # Buckets replayed so far (the first replay of each is logged).
        self._v41_replayed_buckets = set()
        # The bridge's staging, bound by `init_cuda_graph_state`.
        self._stage_capture = None
        self._stage_replay = None

    @staticmethod
    def get_name() -> str:
        return "dsv4"

    def init_forward_metadata(self, forward_batch):
        # The bridge derives ATOM's metadata from the batch inside the forward.
        self.forward_metadata = None

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # The metadata glue graph, and Engram staged on the host, are refused
        # at launch (`refuse_unsupported_sglang_features`).
        from atom.plugin.sglang import deepseek_v41_bridge as bridge

        bridge.check_deepseek_v41_graph_capacity(max_bs, max_num_tokens)
        # Bound once: the staging runs before every replay.
        self._stage_capture = bridge.stage_deepseek_v41_graph_capture
        self._stage_replay = bridge.stage_deepseek_v41_graph_replay
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
        self.forward_metadata = None
        # Capture hands the ForwardBatch itself; replay a view whose
        # forward_mode is the capture's and actual_forward_mode the batch's.
        mode = (
            forward_batch.forward_mode
            if in_capture
            else forward_batch.actual_forward_mode
        )
        # The one mode check of the graph path (capture and replay staging
        # assume it).
        if not mode.is_decode():
            raise NotImplementedError(
                "DeepSeek-V4.1 graphs on the SGLang plugin cover plain decode "
                f"only, not {mode.name}; launch with "
                "--cuda-graph-backend-prefill disabled"
            )
        if in_capture:
            self._stage_capture(forward_batch, self.token_to_kv_pool)
            return
        # SGLang replays only a captured bucket, over the static buffers it
        # recorded (`build_replay_fb_view`); the arena never moves.
        self._stage_replay(forward_batch, self.req_to_token_pool.req_to_token)
        bs = int(forward_batch.batch_size)
        if bs not in self._v41_replayed_buckets:
            self._v41_replayed_buckets.add(bs)
            logger.info(
                "V4.1 decode graph replay bs=%d raw=%d",
                bs,
                bs - int(forward_batch.num_padding),
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1
