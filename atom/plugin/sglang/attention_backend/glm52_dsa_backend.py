"""SGLang backend shim for ATOM-owned GLM-5.2 native MLA attention."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

logger = logging.getLogger("atom.plugin.sglang.attention_backend.glm52_dsa")


class ATOMGLM52DSABackendForSgl(AttentionBackend):
    """Publish fixed-address ATOM GLM-5.2 metadata for SGLang CUDA graphs."""

    needs_cpu_seq_lens = True
    _last_atom_glm52_graph_metadata = None

    def __init__(self, model_runner, *args, **kwargs):
        del args
        logger.info("Initializing ATOMGLM52DSABackendForSgl")
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.forward_metadata = None
        self.atom_glm52_graph_metadata = None
        self._cuda_graph_seq_len_fill_value = 1
        self._spec_graph_metadata_cache = {}
        speculative_num_steps = int(kwargs.pop("speculative_num_steps", 0) or 0)
        self.attn_backends = [self] * max(1, speculative_num_steps)

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

    @staticmethod
    def _is_spec_extend_mode(forward_mode) -> bool:
        return bool(
            forward_mode is not None
            and (
                forward_mode.is_target_verify()
                or getattr(forward_mode, "is_draft_extend", lambda **kwargs: False)(
                    include_v2=True
                )
            )
        )

    @staticmethod
    def _spec_graph_key(forward_batch, positions) -> tuple[str, int, int]:
        cache_bs = int(
            getattr(forward_batch, "_graph_cache_bs", int(forward_batch.batch_size))
        )
        cache_rows = int(
            getattr(
                forward_batch,
                "_graph_cache_rows",
                int(positions.numel()) if positions is not None else 0,
            )
        )
        return (
            str(forward_batch.forward_mode),
            cache_bs,
            cache_rows,
        )

    @staticmethod
    def _copy_graph_metadata_in_place(dst, src):
        def _copy_tensor(dst_tensor, src_tensor):
            if dst_tensor.shape == src_tensor.shape and dst_tensor.dtype == src_tensor.dtype:
                dst_tensor.copy_(src_tensor)
                return True
            if (
                dst_tensor.dim() == src_tensor.dim()
                and dst_tensor.dtype == src_tensor.dtype
                and all(o >= v for o, v in zip(dst_tensor.shape, src_tensor.shape))
            ):
                dst_tensor.zero_()
                slices = tuple(slice(0, int(v)) for v in src_tensor.shape)
                dst_tensor[slices].copy_(src_tensor)
                return True
            if dst_tensor.numel() >= src_tensor.numel() and dst_tensor.dtype == src_tensor.dtype:
                dst_tensor.reshape(-1)[: src_tensor.numel()].copy_(src_tensor.reshape(-1))
                if dst_tensor.numel() > src_tensor.numel():
                    dst_tensor.reshape(-1)[src_tensor.numel() :].zero_()
                return True
            return False

        for name, value in vars(src).items():
            old = getattr(dst, name, None)
            if torch.is_tensor(value) and torch.is_tensor(old):
                if _copy_tensor(old, value):
                    continue
            if name == "mla_chunk_meta" and old is not None and value is not None:
                for field_name, field_value in vars(value).items():
                    field_old = getattr(old, field_name, None)
                    if isinstance(field_value, list) and isinstance(field_old, list):
                        for old_item, new_item in zip(field_old, field_value):
                            if torch.is_tensor(old_item) and torch.is_tensor(new_item):
                                _copy_tensor(old_item, new_item)
                        if len(field_old) == len(field_value):
                            continue
                    if torch.is_tensor(field_value) and torch.is_tensor(field_old):
                        if _copy_tensor(field_old, field_value):
                            continue
                    setattr(old, field_name, field_value)
                continue
            setattr(dst, name, value)
        return dst

    def _build_spec_graph_metadata(self, forward_batch, positions=None):
        if not self._is_spec_extend_mode(forward_batch.forward_mode):
            self.atom_glm52_graph_metadata = None
            return None
        tokens_per_req = int(
            getattr(getattr(forward_batch, "spec_info", None), "num_tokens_per_req", 0)
            or getattr(getattr(forward_batch, "spec_info", None), "draft_token_num", 0)
            or 1
        )
        bs = int(forward_batch.batch_size)
        total_rows = bs * max(1, tokens_per_req)
        if positions is None:
            positions = getattr(forward_batch, "positions", None)
        if positions is None:
            if forward_batch.forward_mode.is_target_verify():
                base = forward_batch.seq_lens[:bs].to(torch.int64)
            else:
                base = (
                    forward_batch.seq_lens[:bs].to(torch.int64) - tokens_per_req
                ).clamp_min_(0)
            offsets = torch.arange(tokens_per_req, dtype=torch.int64, device=self.device)
            positions = (base[:, None] + offsets[None, :]).reshape(-1)
        elif int(positions.numel()) < total_rows:
            if forward_batch.forward_mode.is_target_verify():
                base = forward_batch.seq_lens[:bs].to(torch.int64)
            else:
                base = (
                    forward_batch.seq_lens[:bs].to(torch.int64) - tokens_per_req
                ).clamp_min_(0)
            offsets = torch.arange(tokens_per_req, dtype=torch.int64, device=self.device)
            padded_positions = (base[:, None] + offsets[None, :]).reshape(-1)
            padded_positions[: int(positions.numel())].copy_(positions)
            positions = padded_positions
        elif int(positions.numel()) > total_rows:
            positions = positions[:total_rows]
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None or int(out_cache_loc.numel()) < total_rows:
            scratch_slot = max(0, int(getattr(self.token_to_kv_pool, "size", 1)) - 1)
            values = dict(getattr(forward_batch, "__dict__", {}))
            padded_out_cache_loc = torch.full(
                (total_rows,),
                scratch_slot,
                dtype=torch.int64,
                device=self.device,
            )
            if torch.is_tensor(out_cache_loc) and int(out_cache_loc.numel()) > 0:
                padded_out_cache_loc[: int(out_cache_loc.numel())].copy_(out_cache_loc)
            values["out_cache_loc"] = padded_out_cache_loc
            forward_batch = SimpleNamespace(**values)
            out_cache_loc = padded_out_cache_loc
        elif int(out_cache_loc.numel()) > total_rows:
            values = dict(getattr(forward_batch, "__dict__", {}))
            values["out_cache_loc"] = out_cache_loc[:total_rows]
            forward_batch = SimpleNamespace(**values)
            out_cache_loc = values["out_cache_loc"]

        if os.environ.get("ATOM_GLM52_VERIFY_DEBUG", "0") in ("1", "true", "True"):
            try:
                seq_lens = getattr(forward_batch, "seq_lens", None)
                input_ids = getattr(forward_batch, "input_ids", None)
                logger.info(
                    "GLM52 verify graph metadata: mode=%s bs=%s tokens_per_req=%s "
                    "positions_shape=%s positions_head=%s seq_lens_head=%s "
                    "out_cache_loc_shape=%s out_cache_loc_head=%s input_ids_shape=%s "
                    "input_ids_head=%s raw_replay_bs=%s raw_replay_tokens=%s",
                    forward_batch.forward_mode,
                    bs,
                    tokens_per_req,
                    tuple(positions.shape) if torch.is_tensor(positions) else None,
                    positions[: min(12, int(positions.numel()))].detach().cpu().tolist()
                    if torch.is_tensor(positions)
                    else None,
                    seq_lens[: min(8, int(seq_lens.numel()))].detach().cpu().tolist()
                    if torch.is_tensor(seq_lens)
                    else None,
                    tuple(out_cache_loc.shape)
                    if torch.is_tensor(out_cache_loc)
                    else None,
                    out_cache_loc[: min(12, int(out_cache_loc.numel()))]
                    .detach()
                    .cpu()
                    .tolist()
                    if torch.is_tensor(out_cache_loc)
                    else None,
                    tuple(input_ids.shape) if torch.is_tensor(input_ids) else None,
                    input_ids[: min(12, int(input_ids.numel()))].detach().cpu().tolist()
                    if torch.is_tensor(input_ids)
                    else None,
                    int(getattr(getattr(self, "_replay_forward_batch", None), "batch_size", 0) or 0),
                    int(getattr(getattr(getattr(self, "_replay_forward_batch", None), "input_ids", None), "numel", lambda: 0)()),
                )
            except Exception:
                logger.exception("Failed to log GLM52 verify graph metadata debug")

        from atom.config import get_current_atom_config
        from atom.plugin.sglang.glm52_dsa_bridge import (
            build_atom_glm52_attention_metadata_from_sglang,
        )

        atom_config = get_current_atom_config()
        new_metadata = build_atom_glm52_attention_metadata_from_sglang(
            forward_batch,
            positions,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token_pool=self.req_to_token_pool,
            atom_config=atom_config,
        )
        key = self._spec_graph_key(forward_batch, positions)
        cached_metadata = self._spec_graph_metadata_cache.get(key)
        if cached_metadata is None:
            self._spec_graph_metadata_cache[key] = new_metadata
            self.atom_glm52_graph_metadata = new_metadata
        else:
            self.atom_glm52_graph_metadata = self._copy_graph_metadata_in_place(
                cached_metadata, new_metadata
            )
        if os.environ.get("ATOM_GLM52_VERIFY_DEBUG", "0") in ("1", "true", "True"):
            try:
                md = self.atom_glm52_graph_metadata
                row_probe = None
                try:
                    probe_bs = min(2, int(bs))
                    positions_rows = (
                        positions.reshape(bs, tokens_per_req)[:probe_bs]
                        .detach()
                        .cpu()
                        .tolist()
                        if torch.is_tensor(positions)
                        and int(positions.numel()) >= bs * tokens_per_req
                        else None
                    )
                    out_rows = (
                        out_cache_loc.reshape(bs, tokens_per_req)[:probe_bs]
                        .detach()
                        .cpu()
                        .tolist()
                        if torch.is_tensor(out_cache_loc)
                        and int(out_cache_loc.numel()) >= bs * tokens_per_req
                        else None
                    )
                    req_rows = (
                        forward_batch.req_pool_indices[:probe_bs]
                        .detach()
                        .cpu()
                        .tolist()
                        if torch.is_tensor(getattr(forward_batch, "req_pool_indices", None))
                        else None
                    )
                    kv_indptr_cpu = (
                        md.kv_indptr.detach().cpu().tolist()
                        if torch.is_tensor(getattr(md, "kv_indptr", None))
                        else []
                    )
                    kv_indices = getattr(md, "kv_indices", None)
                    kv_ranges = []
                    for row in range(min(probe_bs, max(0, len(kv_indptr_cpu) - 1))):
                        start = int(kv_indptr_cpu[row])
                        end = int(kv_indptr_cpu[row + 1])
                        if torch.is_tensor(kv_indices):
                            head = (
                                kv_indices[start : min(end, start + 8)]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            tail = (
                                kv_indices[max(start, end - 8) : end]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                        else:
                            head = tail = None
                        kv_ranges.append(
                            {
                                "row": row,
                                "start": start,
                                "end": end,
                                "len": end - start,
                                "head": head,
                                "tail": tail,
                            }
                        )
                    sparse_indptr = getattr(md, "sparse_kv_indptr", None)
                    sparse_probe = (
                        sparse_indptr[
                            : min(
                                int(sparse_indptr.numel()),
                                probe_bs * tokens_per_req + 1,
                            )
                        ]
                        .detach()
                        .cpu()
                        .tolist()
                        if torch.is_tensor(sparse_indptr)
                        else None
                    )
                    token_to_seq = getattr(md, "token_to_seq_idxs", None)
                    token_to_seq_probe = (
                        token_to_seq[: min(int(token_to_seq.numel()), probe_bs * tokens_per_req)]
                        .detach()
                        .cpu()
                        .tolist()
                        if torch.is_tensor(token_to_seq)
                        else None
                    )
                    row_probe = {
                        "positions": positions_rows,
                        "out_cache_loc": out_rows,
                        "req_pool_indices": req_rows,
                        "kv_ranges": kv_ranges,
                        "sparse_kv_indptr": sparse_probe,
                        "token_to_seq": token_to_seq_probe,
                    }
                    spec_info = getattr(forward_batch, "spec_info", None)
                    if spec_info is not None:
                        counter = int(
                            getattr(spec_info, "_atom_glm52_verify_counter", 0) or 0
                        ) + 1
                        setattr(spec_info, "_atom_glm52_verify_counter", counter)
                        setattr(spec_info, "_atom_glm52_row_probe", row_probe)
                except Exception:
                    logger.exception("Failed to build GLM52 verify graph row probe")
                logger.info(
                    "GLM52 verify graph md: key=%s max_q=%s max_k=%s total_kv=%s "
                    "has_cached=%s cu_q=%s kv_indptr=%s sparse_cu=%s "
                    "sparse_kv_indptr=%s token_to_seq=%s row_probe=%s",
                    key,
                    getattr(md, "max_seqlen_q", None),
                    getattr(md, "max_seqlen_k", None),
                    getattr(md, "total_kv", None),
                    getattr(md, "has_cached", None),
                    md.cu_seqlens_q[: min(8, int(md.cu_seqlens_q.numel()))]
                    .detach()
                    .cpu()
                    .tolist()
                    if torch.is_tensor(getattr(md, "cu_seqlens_q", None))
                    else None,
                    md.kv_indptr[: min(8, int(md.kv_indptr.numel()))]
                    .detach()
                    .cpu()
                    .tolist()
                    if torch.is_tensor(getattr(md, "kv_indptr", None))
                    else None,
                    md.sparse_cu_seqlens_q[
                        : min(8, int(md.sparse_cu_seqlens_q.numel()))
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                    if torch.is_tensor(getattr(md, "sparse_cu_seqlens_q", None))
                    else None,
                    md.sparse_kv_indptr[
                        : min(8, int(md.sparse_kv_indptr.numel()))
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                    if torch.is_tensor(getattr(md, "sparse_kv_indptr", None))
                    else None,
                    md.token_to_seq_idxs[
                        : min(12, int(md.token_to_seq_idxs.numel()))
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                    if torch.is_tensor(getattr(md, "token_to_seq_idxs", None))
                    else None,
                    row_probe,
                )
            except Exception:
                logger.exception("Failed to log GLM52 verify graph md debug")
        forward_batch.atom_glm52_graph_metadata = self.atom_glm52_graph_metadata
        ATOMGLM52DSABackendForSgl._last_atom_glm52_graph_metadata = (
            self.atom_glm52_graph_metadata
        )
        self.forward_metadata = forward_batch
        return self.atom_glm52_graph_metadata

    def _build_graph_metadata(self, forward_batch, positions=None, max_bs=None):
        if self._is_spec_extend_mode(forward_batch.forward_mode):
            return self._build_spec_graph_metadata(forward_batch, positions=positions)
        return self._build_decode_graph_metadata(
            forward_batch, positions=positions, max_bs=max_bs
        )

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
        self._build_graph_metadata(forward_batch, positions=positions)

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
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            actual_forward_mode=forward_mode,
            batch_size=int(bs),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.detach().cpu(),
            out_cache_loc=None,
            spec_info=spec_info,
        )
        self._build_graph_metadata(forward_batch, max_bs=int(bs))

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        if len(args) == 2 and hasattr(args[0], "forward_mode"):
            forward_batch, bs = args
            values = dict(getattr(forward_batch, "__dict__", {}))
            if self._is_spec_extend_mode(getattr(forward_batch, "forward_mode", None)):
                active_bs = int(getattr(forward_batch, "batch_size", int(bs)) or int(bs))
                tokens_per_req = int(
                    getattr(getattr(forward_batch, "spec_info", None), "num_tokens_per_req", 0)
                    or getattr(getattr(forward_batch, "spec_info", None), "draft_token_num", 0)
                    or 1
                )
                values["batch_size"] = active_bs
                values["_graph_cache_bs"] = int(bs)
                values["_graph_cache_rows"] = int(bs) * max(1, tokens_per_req)
            else:
                values["batch_size"] = int(bs)
            replay_batch = SimpleNamespace(**values)
            return self._build_graph_metadata(
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
        spec_info = kwargs.get("spec_info", args[6] if len(args) > 6 else None)
        seq_lens_cpu = kwargs.get("seq_lens_cpu", args[7] if len(args) > 7 else None)
        out_cache_loc = kwargs.get("out_cache_loc", args[8] if len(args) > 8 else None)
        replay_batch = getattr(self, "_replay_forward_batch", None)
        if out_cache_loc is None:
            out_cache_loc = getattr(replay_batch, "out_cache_loc", None)
        raw_seq_lens = getattr(replay_batch, "seq_lens", None)
        raw_seq_lens_cpu = getattr(replay_batch, "seq_lens_cpu", None)
        is_spec_replay = self._is_spec_extend_mode(forward_mode)
        active_bs = int(bs)
        if is_spec_replay:
            active_bs = int(getattr(replay_batch, "batch_size", 0) or int(bs))
            raw_req_pool_indices = getattr(replay_batch, "req_pool_indices", None)
            if raw_req_pool_indices is not None:
                req_pool_indices = raw_req_pool_indices
        if bs is None or req_pool_indices is None or seq_lens is None:
            self.atom_glm52_graph_metadata = None
            return
        tokens_per_req = int(
            getattr(spec_info, "num_tokens_per_req", 0)
            or getattr(spec_info, "draft_token_num", 0)
            or getattr(getattr(replay_batch, "spec_info", None), "num_tokens_per_req", 0)
            or getattr(getattr(replay_batch, "spec_info", None), "draft_token_num", 0)
            or 1
        )
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            actual_forward_mode=getattr(replay_batch, "forward_mode", forward_mode),
            batch_size=active_bs,
            _graph_cache_bs=int(bs),
            _graph_cache_rows=int(bs) * max(1, tokens_per_req),
            req_pool_indices=req_pool_indices,
            seq_lens=(
                raw_seq_lens
                if is_spec_replay and torch.is_tensor(raw_seq_lens)
                else seq_lens
            ),
            seq_lens_cpu=(
                raw_seq_lens_cpu
                if is_spec_replay and raw_seq_lens_cpu is not None
                else seq_lens_cpu
            ),
            out_cache_loc=out_cache_loc,
            positions=getattr(replay_batch, "positions", None),
            spec_info=spec_info or getattr(replay_batch, "spec_info", None),
        )
        self._build_graph_metadata(
            forward_batch,
            positions=getattr(forward_batch, "positions", None),
            max_bs=int(bs),
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        bs = int(max_bs)
        tokens_per_req = max(1, int(max_num_tokens) // max(1, bs))
        is_target_verify_graph = bool(
            getattr(
                getattr(self.model_runner, "spec_algorithm", None),
                "is_speculative",
                lambda: False,
            )()
            and not getattr(self.model_runner, "is_draft_worker", False)
        )
        is_draft_extend_graph = bool(
            getattr(
                getattr(self.model_runner, "spec_algorithm", None),
                "is_speculative",
                lambda: False,
            )()
            and getattr(self.model_runner, "is_draft_worker", False)
            and tokens_per_req > 1
        )
        is_graph_extend = is_target_verify_graph or is_draft_extend_graph
        seq_lens = torch.ones(bs, dtype=torch.int32, device=self.device)
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=self.device)
        forward_mode = (
            ForwardMode.TARGET_VERIFY
            if is_target_verify_graph
            else (
                ForwardMode.DRAFT_EXTEND_V2
                if is_draft_extend_graph
                else ForwardMode.DECODE
            )
        )
        if is_graph_extend:
            fill_override = int(os.environ.get("ATOM_GLM52_TV_CG_SEQ_LEN_FILL", "0") or 0)
            self._cuda_graph_seq_len_fill_value = (
                max(tokens_per_req, fill_override)
                if fill_override > 0
                else max(tokens_per_req, 1024)
            )
            seq_lens.fill_(self._cuda_graph_seq_len_fill_value)
            offsets = torch.arange(tokens_per_req, dtype=torch.int64, device=self.device)
            if is_target_verify_graph:
                base = seq_lens[:bs].to(torch.int64)
            else:
                base = (seq_lens[:bs].to(torch.int64) - tokens_per_req).clamp_min_(0)
            positions = (base[:, None] + offsets[None, :]).reshape(-1)
            if os.environ.get("ATOM_GLM52_VERIFY_DEBUG", "0") in ("1", "true", "True"):
                logger.info(
                    "GLM52 graph init seq_len fill: mode=%s bs=%s tokens_per_req=%s "
                    "fill=%s override=%s",
                    forward_mode,
                    bs,
                    tokens_per_req,
                    self._cuda_graph_seq_len_fill_value,
                    fill_override,
                )
        else:
            self._cuda_graph_seq_len_fill_value = 1
            positions = torch.zeros(bs, dtype=torch.int64, device=self.device)
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            actual_forward_mode=forward_mode,
            batch_size=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.detach().cpu(),
            out_cache_loc=None,
            spec_info=SimpleNamespace(
                num_tokens_per_req=tokens_per_req,
                draft_token_num=tokens_per_req,
            ),
        )
        self._build_graph_metadata(
            forward_batch,
            positions=positions,
            max_bs=bs,
        )
        del max_num_tokens
        return None

    def get_cuda_graph_seq_len_fill_value(self):
        return int(self._cuda_graph_seq_len_fill_value)

    def get_verify_buffers_to_fill_after_draft(self):
        graph_runner = getattr(self.model_runner, "graph_runner", None)
        buffers = getattr(graph_runner, "buffers", None)
        if buffers is None:
            return [None, None]
        # SGLang fills this captured mask buffer after draft graph replay. The
        # verifier positions are copied in update_verify_buffers_to_fill_after_draft().
        return [getattr(buffers, "custom_mask", None), None]

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        if cuda_graph_bs is None:
            return
        graph_runner = getattr(self.model_runner, "graph_runner", None)
        buffers = getattr(graph_runner, "buffers", None)
        if buffers is None:
            return

        tokens_per_req = int(
            getattr(
                spec_info,
                "num_tokens_per_req",
                getattr(spec_info, "draft_token_num", 1),
            )
            or 1
        )
        total = int(cuda_graph_bs) * tokens_per_req

        positions = getattr(spec_info, "positions", None)
        active_total = int(positions.numel()) if torch.is_tensor(positions) else total
        active_bs = max(1, active_total // max(1, tokens_per_req))
        if torch.is_tensor(positions):
            copy_n = min(int(positions.numel()), total)
            if copy_n:
                buffers.positions[:copy_n].copy_(positions[:copy_n])
            if total > copy_n:
                buffers.positions[copy_n:total].zero_()
            positions = buffers.positions[:active_total]
        else:
            positions = buffers.positions[: active_bs * tokens_per_req]

        custom_mask = getattr(spec_info, "custom_mask", None)
        graph_custom_mask = getattr(buffers, "custom_mask", None)
        if (
            torch.is_tensor(custom_mask)
            and torch.is_tensor(graph_custom_mask)
            and custom_mask.data_ptr() != graph_custom_mask.data_ptr()
        ):
            graph_custom_mask[: custom_mask.numel()].copy_(custom_mask)

        forward_mode = getattr(
            getattr(self, "forward_metadata", None), "forward_mode", None
        )
        if forward_mode is None:
            return
        seq_lens_cpu = getattr(buffers, "seq_lens_cpu", None)
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            actual_forward_mode=forward_mode,
            batch_size=active_bs,
            _graph_cache_bs=int(cuda_graph_bs),
            _graph_cache_rows=total,
            req_pool_indices=buffers.req_pool_indices[:active_bs],
            seq_lens=buffers.seq_lens[:active_bs],
            seq_lens_cpu=(
                seq_lens_cpu[:active_bs] if seq_lens_cpu is not None else None
            ),
            out_cache_loc=buffers.out_cache_loc[:active_total],
            positions=positions,
            spec_info=spec_info,
        )
        self._build_graph_metadata(forward_batch, positions=positions)

    def forward_decode(self, *args, **kwargs):
        raise RuntimeError("ATOM GLM-5.2 SGLang bridge should use ATOM attention")

    def forward_extend(self, *args, **kwargs):
        raise RuntimeError("ATOM GLM-5.2 SGLang bridge should use ATOM attention")
