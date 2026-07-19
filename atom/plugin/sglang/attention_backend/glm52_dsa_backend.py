"""SGLang backend shim for ATOM-owned GLM-5.2 native MLA attention."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

logger = logging.getLogger("atom.plugin.sglang.attention_backend.glm52_dsa")


class ATOMGLM52DSABackendForSgl(AttentionBackend):
    """Publish fixed-address ATOM GLM-5.2 metadata for SGLang CUDA graphs."""

    needs_cpu_seq_lens = True
    _last_atom_glm52_graph_metadata = None

    def __init__(self, model_runner, *args, **kwargs):
        del args, kwargs
        logger.info("Initializing ATOMGLM52DSABackendForSgl")
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.forward_metadata = None
        self.atom_glm52_graph_metadata = None
        self._cuda_graph_seq_len_fill_value = 1
        self.attn_backends = [self]

    @staticmethod
    def get_name() -> str:
        return "atom_glm52_dsa"

    def init_forward_metadata(self, forward_batch):
        self.forward_metadata = forward_batch
        self.atom_glm52_graph_metadata = None

    def _build_decode_graph_metadata(self, forward_batch, positions=None, max_bs=None):
        if not forward_batch.forward_mode.is_decode_or_idle():
            self.atom_glm52_graph_metadata = None
            return None
        if positions is None:
            positions = getattr(forward_batch, "positions", None)
        if positions is None:
            positions = (
                forward_batch.seq_lens[: int(forward_batch.batch_size)].to(torch.int64)
                - 1
            ).clamp_min_(0)

        from atom.config import get_current_atom_config
        from atom.plugin.sglang.glm52_dsa_bridge import (
            build_atom_glm52_decode_graph_metadata_from_sglang,
        )

        atom_config = get_current_atom_config()
        max_context_len = int(self.req_to_token_pool.req_to_token.shape[1])
        self.atom_glm52_graph_metadata = (
            build_atom_glm52_decode_graph_metadata_from_sglang(
                forward_batch,
                positions,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token_pool=self.req_to_token_pool,
                atom_config=atom_config,
                max_bs=max_bs,
                max_context_len=max_context_len,
            )
        )
        forward_batch.atom_glm52_graph_metadata = self.atom_glm52_graph_metadata
        ATOMGLM52DSABackendForSgl._last_atom_glm52_graph_metadata = (
            self.atom_glm52_graph_metadata
        )
        self.forward_metadata = forward_batch
        return self.atom_glm52_graph_metadata

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        if not (in_capture or hasattr(forward_batch, "actual_forward_mode")):
            self.forward_metadata = forward_batch
            self.atom_glm52_graph_metadata = None
            return
        positions = getattr(forward_batch, "positions", None)
        if positions is None:
            graph_runner = getattr(self.model_runner, "graph_runner", None)
            buffers = getattr(graph_runner, "buffers", None)
            positions = getattr(buffers, "positions", None)
        self._build_decode_graph_metadata(forward_batch, positions=positions)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        bs = int(max_bs)
        seq_lens = torch.ones(bs, dtype=torch.int32, device=self.device)
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=self.device)
        positions = torch.zeros(bs, dtype=torch.int64, device=self.device)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            actual_forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.detach().cpu(),
            out_cache_loc=None,
        )
        self._cuda_graph_seq_len_fill_value = 1
        self._build_decode_graph_metadata(
            forward_batch,
            positions=positions,
            max_bs=bs,
        )
        del max_num_tokens
        return None

    def get_cuda_graph_seq_len_fill_value(self):
        return int(self._cuda_graph_seq_len_fill_value)

    def forward_decode(self, *args, **kwargs):
        raise RuntimeError("ATOM GLM-5.2 SGLang bridge should use ATOM attention")

    def forward_extend(self, *args, **kwargs):
        raise RuntimeError("ATOM GLM-5.2 SGLang bridge should use ATOM attention")
