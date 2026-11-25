import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


class ATOMDeepseekV41BackendForSgl(AttentionBackend):
    """SGLang backend shim for ATOM-owned DeepSeek-V4.1 (CSA2) attention.

    It publishes ``req_to_token_pool`` to the forward; the pool, the step
    (staged by the wrapper before the model runs) and the kernels are ATOM's.
    The bridge reads the host ``seq_lens`` mirror that SGLang's default
    ``needs_cpu_seq_lens`` (True) publishes.
    """

    def __init__(self, model_runner, *args, **kwargs):
        del args, kwargs
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool

    @staticmethod
    def get_name() -> str:
        return "dsv4"

    def init_forward_metadata(self, forward_batch):
        # The bridge derives ATOM's metadata from the batch inside the forward.
        self.forward_metadata = None

    def get_cuda_graph_seq_len_fill_value(self):
        return 1
