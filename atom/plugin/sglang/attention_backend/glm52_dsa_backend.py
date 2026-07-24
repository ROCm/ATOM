"""SGLang backend shim for ATOM-owned GLM-5.2 native MLA attention."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

from atom.plugin.sglang.glm52_mtp.common import GLM52_GRAPH_SEQ_LEN_CAPACITY

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
        self._target_verify_graph_metadata = {}
        self._cuda_graph_seq_len_fill_value = 1
        self.attn_backends = [self]

    @staticmethod
    def _is_target_verify(forward_mode) -> bool:
        return bool(getattr(forward_mode, "is_target_verify", lambda: False)())

    @staticmethod
    def _copy_metadata_in_place(source, target) -> None:
        tensor_fields = (
            "cu_seqlens_q",
            "cu_seqlens_k",
            "slot_mapping",
            "context_lens",
            "block_tables",
            "kv_indptr",
            "kv_indices",
            "kv_last_page_lens",
            "sparse_kv_indptr",
            "sparse_kv_last_page_lens",
            "sparse_cu_seqlens_q",
            "token_to_seq_idxs",
            "work_meta_data",
            "work_indptr",
            "work_info_set",
            "reduce_indptr",
            "reduce_final_map",
            "reduce_partial_map",
            "sparse_mtp_work_meta_data",
            "sparse_mtp_work_indptr",
            "sparse_mtp_work_info_set",
            "sparse_mtp_reduce_indptr",
            "sparse_mtp_reduce_final_map",
            "sparse_mtp_reduce_partial_map",
        )
        for name in tensor_fields:
            source_tensor = getattr(source, name, None)
            target_tensor = getattr(target, name, None)
            if not torch.is_tensor(source_tensor) or not torch.is_tensor(target_tensor):
                continue
            if source_tensor.numel() > target_tensor.numel():
                raise RuntimeError(
                    f"GLM target-verify graph field {name} exceeds capture capacity: "
                    f"runtime={tuple(source_tensor.shape)}, "
                    f"capture={tuple(target_tensor.shape)}"
                )
            target_flat = target_tensor.reshape(-1)
            source_flat = source_tensor.reshape(-1)
            target_flat.zero_()
            target_flat[: source_flat.numel()].copy_(source_flat)
        for name in ("max_seqlen_q", "max_seqlen_k", "state", "dtype_q"):
            if hasattr(source, name):
                setattr(target, name, getattr(source, name))

    def _build_target_verify_graph_metadata(
        self,
        forward_batch,
        positions,
    ):
        from atom.config import get_current_atom_config
        from atom.plugin.sglang.glm52_mtp.target_verify import (
            build_mtp_verify_decode_metadata,
        )

        return build_mtp_verify_decode_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            atom_config=get_current_atom_config(),
        )

    def _publish_target_verify_graph_metadata(self, forward_batch, metadata):
        self.atom_glm52_graph_metadata = metadata
        self.forward_metadata = forward_batch
        setattr(forward_batch, "atom_glm52_graph_metadata", metadata)
        ATOMGLM52DSABackendForSgl._last_atom_glm52_graph_metadata = metadata
        return metadata

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

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and hasattr(args[0], "forward_mode"):
            return self.init_forward_metadata_out_graph(args[0], in_capture=True)
        bs = kwargs.get("bs", args[0] if len(args) > 0 else None)
        req_pool_indices = kwargs.get(
            "req_pool_indices", args[2] if len(args) > 2 else None
        )
        seq_lens = kwargs.get("seq_lens", args[3] if len(args) > 3 else None)
        forward_mode = kwargs.get("forward_mode", args[5] if len(args) > 5 else None)
        spec_info = kwargs.get("spec_info", args[6] if len(args) > 6 else None)
        if bs is None or req_pool_indices is None or seq_lens is None:
            self.atom_glm52_graph_metadata = None
            return
        if self._is_target_verify(forward_mode):
            draft_token_num = int(
                getattr(spec_info, "draft_token_num", 0)
                or getattr(spec_info, "num_tokens_per_req", 0)
                or 0
            )
            if draft_token_num <= 0:
                raise RuntimeError(
                    "GLM target-verify graph capture requires draft_token_num"
                )
            seq_lens_cpu = seq_lens.detach().cpu()
            positions = torch.repeat_interleave(
                seq_lens.to(torch.int64) - draft_token_num,
                draft_token_num,
            ) + torch.arange(
                draft_token_num, dtype=torch.int64, device=self.device
            ).repeat(int(bs))
            forward_batch = SimpleNamespace(
                forward_mode=forward_mode,
                actual_forward_mode=forward_mode,
                batch_size=int(bs),
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=torch.zeros(
                    int(bs) * draft_token_num,
                    dtype=torch.int64,
                    device=self.device,
                ),
                positions=positions,
                spec_info=SimpleNamespace(draft_token_num=draft_token_num),
            )
            metadata = self._build_target_verify_graph_metadata(
                forward_batch, positions
            )
            self._target_verify_graph_metadata[int(bs)] = metadata
            return self._publish_target_verify_graph_metadata(
                forward_batch, metadata
            )
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            actual_forward_mode=forward_mode,
            batch_size=int(bs),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.detach().cpu(),
            out_cache_loc=None,
        )
        self._build_decode_graph_metadata(forward_batch, max_bs=int(bs))

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        if len(args) == 2 and hasattr(args[0], "forward_mode"):
            forward_batch, bs = args
            values = dict(getattr(forward_batch, "__dict__", {}))
            values["batch_size"] = int(bs)
            replay_batch = SimpleNamespace(**values)
            return self._build_decode_graph_metadata(
                replay_batch,
                positions=getattr(forward_batch, "positions", None),
                max_bs=int(bs),
            )

        bs = kwargs.get("bs", args[0] if len(args) > 0 else None)
        req_pool_indices = kwargs.get(
            "req_pool_indices", args[1] if len(args) > 1 else None
        )
        seq_lens = kwargs.get("seq_lens", args[2] if len(args) > 2 else None)
        forward_mode = kwargs.get("forward_mode", args[5] if len(args) > 5 else None)
        seq_lens_cpu = kwargs.get("seq_lens_cpu", args[7] if len(args) > 7 else None)
        out_cache_loc = kwargs.get("out_cache_loc", args[8] if len(args) > 8 else None)
        replay_batch = getattr(self, "_replay_forward_batch", None)
        if out_cache_loc is None:
            out_cache_loc = getattr(replay_batch, "out_cache_loc", None)
        if bs is None or req_pool_indices is None or seq_lens is None:
            self.atom_glm52_graph_metadata = None
            return
        if self._is_target_verify(forward_mode):
            fixed = self._target_verify_graph_metadata.get(int(bs))
            if fixed is None:
                raise RuntimeError(
                    f"Missing GLM target-verify graph metadata for batch size {bs}"
                )
            spec_info = kwargs.get("spec_info", args[6] if len(args) > 6 else None)
            draft_token_num = int(
                getattr(spec_info, "draft_token_num", 0)
                or getattr(spec_info, "num_tokens_per_req", 0)
                or 0
            )
            graph_runner = getattr(self.model_runner, "graph_runner", None)
            graph_buffers = getattr(graph_runner, "buffers", None)
            if graph_buffers is None or draft_token_num <= 0:
                raise RuntimeError("GLM target-verify replay buffers are unavailable")
            total_tokens = int(bs) * draft_token_num
            positions = graph_buffers.positions[:total_tokens]
            staged_batch = SimpleNamespace(
                forward_mode=forward_mode,
                actual_forward_mode=getattr(
                    replay_batch, "forward_mode", forward_mode
                ),
                batch_size=int(bs),
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=graph_buffers.out_cache_loc[:total_tokens],
                positions=positions,
                spec_info=SimpleNamespace(draft_token_num=draft_token_num),
            )
            staged = self._build_target_verify_graph_metadata(
                staged_batch, positions
            )
            self._copy_metadata_in_place(staged, fixed)
            return self._publish_target_verify_graph_metadata(staged_batch, fixed)
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            actual_forward_mode=getattr(replay_batch, "forward_mode", forward_mode),
            batch_size=int(bs),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            positions=getattr(replay_batch, "positions", None),
        )
        self._build_decode_graph_metadata(
            forward_batch,
            positions=getattr(forward_batch, "positions", None),
            max_bs=int(bs),
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        bs = int(max_bs)
        is_target_verify_graph = bool(
            getattr(
                getattr(self.model_runner, "spec_algorithm", None),
                "is_speculative",
                lambda: False,
            )()
            and not getattr(self.model_runner, "is_draft_worker", False)
        )
        tokens_per_req = max(1, int(max_num_tokens) // max(1, bs))
        self._cuda_graph_seq_len_fill_value = (
            max(
                tokens_per_req,
                GLM52_GRAPH_SEQ_LEN_CAPACITY,
            )
            if is_target_verify_graph
            else 1
        )
        seq_lens = torch.full(
            (bs,),
            self._cuda_graph_seq_len_fill_value,
            dtype=torch.int32,
            device=self.device,
        )
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
