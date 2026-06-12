import logging

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


logger = logging.getLogger("atom.plugin.sglang.attention_backend.deepseek_v4")



class ATOMDeepseekV4BackendForSgl(AttentionBackend):
    """SGLang backend shim for ATOM-owned DeepSeek-V4 attention.

    SGLang still needs an attention backend object for scheduling and forward
    context publication.  The actual DeepSeek-V4 cache layout, metadata, and
    kernels are owned by ATOM through ``deepseek_v4_bridge``.
    """
    needs_cpu_seq_lens = True

    def __init__(self, model_runner, *args, **kwargs):
        del args, kwargs
        logger.info("Initializing ATOMDeepseekV4BackendForSgl")
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.forward_metadata = None
        self.atom_v4_graph_metadata = None

    @staticmethod
    def get_name() -> str:
        return "dsv4"

    def init_forward_metadata(self, forward_batch):
        self.atom_v4_graph_metadata = None
        self.forward_metadata = forward_batch

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        self.forward_metadata = forward_batch
        if not (in_capture or hasattr(forward_batch, "actual_forward_mode")):
            self.atom_v4_graph_metadata = None
            return
        if not forward_batch.forward_mode.is_decode_or_idle():
            self.atom_v4_graph_metadata = None
            return

        from atom.plugin.sglang.deepseek_v4_bridge import (
            build_atom_v4_decode_graph_metadata_from_sglang,
        )

        positions = getattr(forward_batch, "positions", None)
        if positions is None:
            graph_runner = getattr(self.model_runner, "graph_runner", None)
            buffers = getattr(graph_runner, "buffers", None)
            positions = getattr(buffers, "positions", None)
        if positions is None:
            self.atom_v4_graph_metadata = None
            return

        atom_model = getattr(getattr(self.model_runner, "model", None), "model", None)
        self.atom_v4_graph_metadata = build_atom_v4_decode_graph_metadata_from_sglang(
            forward_batch,
            positions,
            proxy_pool=self.token_to_kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            model=atom_model,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        del max_bs, max_num_tokens
        return None

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, *args, **kwargs):
        raise RuntimeError("ATOM DeepSeek-V4 SGLang bridge should use ATOM attention")

    def forward_extend(self, *args, **kwargs):
        raise RuntimeError("ATOM DeepSeek-V4 SGLang bridge should use ATOM attention")