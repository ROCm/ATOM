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
    _last_atom_v4_graph_metadata = None

    def __init__(self, model_runner, *args, **kwargs):
        del args
        logger.info("Initializing ATOMDeepseekV4BackendForSgl")
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.forward_metadata = None
        self.atom_v4_graph_metadata = None
        speculative_num_steps = int(kwargs.pop("speculative_num_steps", 0) or 0)
        # SGLang EAGLE multi-step draft code expects decode backends to expose
        # one attention backend per draft step.  ATOM DSV4 owns the real
        # per-layer state in the model/bridge, so all draft steps can share this
        # shim instance.
        self.attn_backends = [self] * max(1, speculative_num_steps)

    @staticmethod
    def get_name() -> str:
        return "dsv4"

    def init_forward_metadata(self, forward_batch):
        self.atom_v4_graph_metadata = None
        self.forward_metadata = forward_batch

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        self.forward_metadata = forward_batch
        logger.info(
            "ATOM DSV4 init_forward_metadata_out_graph: in_capture=%s mode=%s bs=%s",
            in_capture,
            getattr(getattr(forward_batch, "forward_mode", None), "name", None),
            getattr(forward_batch, "batch_size", None),
        )
        if not (in_capture or hasattr(forward_batch, "actual_forward_mode")):
            self.atom_v4_graph_metadata = None
            return
        from atom.plugin.sglang.deepseek_v4_bridge import (
            build_atom_v4_attention_metadata_from_sglang,
            build_atom_v4_decode_graph_metadata_from_sglang,
        )

        positions = getattr(forward_batch, "positions", None)
        if positions is None:
            graph_runner = getattr(self.model_runner, "graph_runner", None)
            buffers = getattr(graph_runner, "buffers", None)
            positions = getattr(buffers, "positions", None)
        if positions is None:
            self.atom_v4_graph_metadata = None
            logger.info(
                "Skip ATOM DeepSeek-V4 graph metadata init: positions unavailable"
            )
            return

        atom_model = getattr(getattr(self.model_runner, "model", None), "model", None)
        if forward_batch.forward_mode.is_decode_or_idle():
            self.atom_v4_graph_metadata = build_atom_v4_decode_graph_metadata_from_sglang(
                forward_batch,
                positions,
                proxy_pool=self.token_to_kv_pool,
                req_to_token_pool=self.req_to_token_pool,
                model=atom_model,
            )
        else:
            self.atom_v4_graph_metadata = build_atom_v4_attention_metadata_from_sglang(
                forward_batch,
                positions,
                proxy_pool=self.token_to_kv_pool,
                req_to_token_pool=self.req_to_token_pool,
            )
        forward_batch.atom_v4_graph_metadata = self.atom_v4_graph_metadata
        ATOMDeepseekV4BackendForSgl._last_atom_v4_graph_metadata = (
            self.atom_v4_graph_metadata
        )
        logger.info(
            "ATOM DSV4 graph metadata initialized: mode=%s bs=%s metadata=%s",
            getattr(getattr(forward_batch, "forward_mode", None), "name", None),
            getattr(forward_batch, "batch_size", None),
            type(self.atom_v4_graph_metadata).__name__,
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
        forward_batch.atom_v4_graph_metadata = self.atom_v4_graph_metadata
        ATOMDeepseekV4BackendForSgl._last_atom_v4_graph_metadata = (
            self.atom_v4_graph_metadata
        )

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        # New SGLang graph API passes a ForwardBatch.  Older call sites pass
        # unpacked fields.  Support both because speculative draft graph code
        # still calls this legacy-named hook directly.
        if len(args) == 1 and not kwargs and hasattr(args[0], "forward_mode"):
            return self.init_forward_metadata_out_graph(args[0], in_capture=True)

        bs = kwargs.get("bs", args[0] if len(args) > 0 else None)
        req_pool_indices = kwargs.get(
            "req_pool_indices", args[2] if len(args) > 2 else None
        )
        seq_lens = kwargs.get("seq_lens", args[3] if len(args) > 3 else None)
        forward_mode = kwargs.get("forward_mode", args[5] if len(args) > 5 else None)
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
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        from atom.plugin.sglang.deepseek_v4_bridge import (
            build_atom_v4_decode_graph_metadata_from_sglang,
        )

        bs = int(max_bs)
        tokens_per_req = max(1, int(max_num_tokens) // max(1, bs))
        seq_lens = torch.full(
            (bs,), tokens_per_req, dtype=torch.int32, device=self.device
        )
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=self.device)
        positions = torch.arange(tokens_per_req, dtype=torch.int64, device=self.device)
        positions = positions.repeat(bs)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            actual_forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.detach().cpu(),
            out_cache_loc=None,
        )
        atom_model = getattr(getattr(self.model_runner, "model", None), "model", None)
        self.atom_v4_graph_metadata = build_atom_v4_decode_graph_metadata_from_sglang(
            forward_batch,
            positions,
            proxy_pool=self.token_to_kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            model=atom_model,
        )
        ATOMDeepseekV4BackendForSgl._last_atom_v4_graph_metadata = (
            self.atom_v4_graph_metadata
        )
        return None

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(self, *args, **kwargs):
        raise RuntimeError("ATOM DeepSeek-V4 SGLang bridge should use ATOM attention")

    def forward_extend(self, *args, **kwargs):
        raise RuntimeError("ATOM DeepSeek-V4 SGLang bridge should use ATOM attention")
