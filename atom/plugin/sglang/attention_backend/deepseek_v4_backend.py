import logging
from types import SimpleNamespace

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

    def _init_decode_cuda_graph_metadata(
        self,
        *,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode,
        seq_lens_cpu=None,
        out_cache_loc=None,
        positions=None,
        actual_forward_mode=None,
    ) -> None:
        if not forward_mode.is_decode_or_idle():
            self.atom_v4_graph_metadata = None
            return

        if positions is None:
            positions = (seq_lens[:bs].to(torch.int64) - 1).clamp_min_(0)
        elif positions.shape[0] < bs:
            padded_positions = (seq_lens[:bs].to(torch.int64) - 1).clamp_min_(0)
            padded_positions[: positions.shape[0]].copy_(positions)
            positions = padded_positions
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.detach().cpu()

        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            actual_forward_mode=actual_forward_mode or forward_mode,
            batch_size=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
        )

        from atom.plugin.sglang.deepseek_v4_bridge import (
            build_atom_v4_decode_graph_metadata_from_sglang,
        )

        atom_model = getattr(getattr(self.model_runner, "model", None), "model", None)
        self.forward_metadata = forward_batch
        self.atom_v4_graph_metadata = build_atom_v4_decode_graph_metadata_from_sglang(
            forward_batch,
            positions,
            proxy_pool=self.token_to_kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            model=atom_model,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        del num_tokens, encoder_lens, spec_info
        self._init_decode_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        del seq_lens_sum, encoder_lens, spec_info
        replay_batch = getattr(self, "_replay_forward_batch", None)
        self._init_decode_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            forward_mode=forward_mode,
            out_cache_loc=getattr(replay_batch, "out_cache_loc", None),
            positions=getattr(replay_batch, "positions", None),
            actual_forward_mode=getattr(replay_batch, "forward_mode", forward_mode),
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
