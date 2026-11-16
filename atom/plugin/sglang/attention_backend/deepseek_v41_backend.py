import logging

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

logger = logging.getLogger("atom.plugin.sglang.attention_backend.deepseek_v41")

_NO_GRAPHS = (
    "DeepSeek-V4.1 CUDA graphs are not enabled on the SGLang plugin yet; launch "
    "with --disable-decode-cuda-graph --disable-prefill-cuda-graph"
)


class ATOMDeepseekV41BackendForSgl(AttentionBackend):
    """SGLang backend shim for ATOM-owned DeepSeek-V4.1 (CSA2) attention.

    SGLang needs a backend object for scheduling and forward-context
    publication: the plugin reads ``req_to_token_pool`` off it during the
    forward. The CSA2 pool, step metadata and kernels belong to ATOM
    (``deepseek_v41_bridge``); the step is staged by the wrapper before the
    model runs. Nothing here computes attention or keeps per-step state.
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

    @staticmethod
    def get_name() -> str:
        return "dsv4"

    def init_forward_metadata(self, forward_batch):
        # The bridge derives ATOM's metadata from the batch inside the forward.
        self.forward_metadata = None

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        if in_capture:
            raise NotImplementedError(_NO_GRAPHS)
        self.forward_metadata = None

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        raise NotImplementedError(_NO_GRAPHS)

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
