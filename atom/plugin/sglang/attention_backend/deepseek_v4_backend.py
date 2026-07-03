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
        self._cuda_graph_seq_len_fill_value = 1
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
        is_draft_extend = bool(
            getattr(
                forward_batch.forward_mode, "is_draft_extend", lambda **kwargs: False
            )(include_v2=True)
        )
        draft_extend_runner = getattr(
            self, "_atom_dsv4_draft_extend_graph_runner", None
        )
        if (
            is_draft_extend
            and draft_extend_runner is not None
            and not hasattr(forward_batch, "actual_forward_mode")
        ):
            forward_batch = self._build_draft_extend_replay_metadata_view(
                forward_batch, draft_extend_runner
            )
            self.forward_metadata = forward_batch

        if not (in_capture or hasattr(forward_batch, "actual_forward_mode")):
            self.atom_v4_graph_metadata = None
            return
        from atom.plugin.sglang.deepseek_v4_bridge import (
            build_atom_v4_attention_metadata_from_sglang,
            build_atom_v4_decode_graph_metadata_from_sglang,
            build_atom_v4_verify_graph_metadata_from_sglang,
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
        if forward_batch.forward_mode.is_decode_or_idle():
            self.atom_v4_graph_metadata = (
                build_atom_v4_decode_graph_metadata_from_sglang(
                    forward_batch,
                    positions,
                    proxy_pool=self.token_to_kv_pool,
                    req_to_token_pool=self.req_to_token_pool,
                    model=atom_model,
                )
            )
        elif forward_batch.forward_mode.is_target_verify() or bool(
            getattr(
                forward_batch.forward_mode, "is_draft_extend", lambda **kwargs: False
            )(include_v2=True)
        ):
            self.atom_v4_graph_metadata = (
                build_atom_v4_verify_graph_metadata_from_sglang(
                    forward_batch,
                    positions,
                    proxy_pool=self.token_to_kv_pool,
                    req_to_token_pool=self.req_to_token_pool,
                    model=atom_model,
                )
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

    def _build_draft_extend_replay_metadata_view(self, forward_batch, runner):
        """Fill missing replay fields from EAGLE draft-extend graph buffers.

        SGLang's standalone draft-extend runner builds a lightweight replay
        view that lacks `positions`/`actual_forward_mode` and keeps
        `out_cache_loc` on the raw batch.  ATOM's DSV4 graph metadata must
        reference the padded CUDA graph buffers because those are the tensors
        captured by the graph.
        """
        buffers = getattr(runner, "buffers", None)
        if buffers is None:
            return forward_batch

        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        spec_info = getattr(forward_batch, "spec_info", None)
        tokens_per_req = int(
            getattr(spec_info, "num_tokens_per_req", None)
            or getattr(runner, "num_tokens_per_bs", 1)
            or 1
        )
        total = max(0, bs * max(1, tokens_per_req))

        def _slice(name, stop):
            value = getattr(buffers, name, None)
            return value[:stop] if value is not None else None

        values = dict(getattr(forward_batch, "__dict__", {}))
        values.update(
            actual_forward_mode=getattr(
                forward_batch, "actual_forward_mode", forward_batch.forward_mode
            ),
            input_ids=_slice("input_ids", total),
            positions=_slice("positions", total),
            req_pool_indices=_slice("req_pool_indices", bs),
            seq_lens=_slice("seq_lens", bs),
            seq_lens_cpu=_slice("seq_lens_cpu", bs),
            out_cache_loc=_slice("out_cache_loc", total),
            spec_info=spec_info,
        )
        return SimpleNamespace(**values)

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

    def _init_verify_cuda_graph_metadata(
        self,
        *,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode,
        seq_lens_cpu=None,
        out_cache_loc=None,
        positions=None,
        spec_info=None,
        actual_forward_mode=None,
    ) -> None:
        is_graph_extend = forward_mode.is_target_verify() or bool(
            getattr(forward_mode, "is_draft_extend", lambda **kwargs: False)(
                include_v2=True
            )
        )
        if not is_graph_extend:
            self.atom_v4_graph_metadata = None
            return

        def _positive_int(value):
            try:
                value = int(value)
            except (TypeError, ValueError):
                return None
            return value if value > 0 else None

        tokens_per_req = _positive_int(getattr(spec_info, "num_tokens_per_req", None))
        if tokens_per_req is None:
            tokens_per_req = _positive_int(getattr(spec_info, "draft_token_num", None))
        if tokens_per_req is None:
            tokens_per_req = _positive_int(
                getattr(spec_info, "speculative_num_draft_tokens", None)
            )
        if tokens_per_req is None:
            tokens_per_req = (
                max(1, int(positions.numel()) // max(1, int(bs)))
                if positions is not None
                else 1
            )
        tokens_per_req = int(tokens_per_req)
        if positions is None:
            base = (seq_lens[:bs].to(torch.int64) - tokens_per_req).clamp_min_(0)
            offsets = torch.arange(
                tokens_per_req, dtype=torch.int64, device=self.device
            )
            positions = (base[:, None] + offsets[None, :]).reshape(-1)
        elif positions.shape[0] < bs * tokens_per_req:
            padded_positions = torch.zeros(
                (bs * tokens_per_req,), dtype=torch.int64, device=self.device
            )
            padded_positions[: positions.shape[0]].copy_(positions)
            positions = padded_positions
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.detach().cpu()

        if spec_info is None:
            spec_info = SimpleNamespace(num_tokens_per_req=tokens_per_req)
        elif _positive_int(getattr(spec_info, "num_tokens_per_req", None)) is None:
            spec_info_dict = dict(getattr(spec_info, "__dict__", {}))
            spec_info_dict.pop("num_tokens_per_req", None)
            spec_info = SimpleNamespace(
                **spec_info_dict,
                num_tokens_per_req=tokens_per_req,
            )

        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            actual_forward_mode=actual_forward_mode or forward_mode,
            batch_size=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            spec_info=spec_info,
        )

        from atom.plugin.sglang.deepseek_v4_bridge import (
            build_atom_v4_verify_graph_metadata_from_sglang,
        )

        atom_model = getattr(getattr(self.model_runner, "model", None), "model", None)
        self.forward_metadata = forward_batch
        self.atom_v4_graph_metadata = build_atom_v4_verify_graph_metadata_from_sglang(
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
        spec_info = kwargs.get("spec_info", args[6] if len(args) > 6 else None)
        if forward_mode is not None and (
            forward_mode.is_target_verify()
            or bool(
                getattr(forward_mode, "is_draft_extend", lambda **kwargs: False)(
                    include_v2=True
                )
            )
        ):
            return self._init_verify_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
            )
        self._init_decode_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        # Older SGLang draft graph runners call this hook as
        # ``init_forward_metadata_replay_cuda_graph(forward_batch, bs)``.
        # Newer runners pass unpacked fields. Support both so the ATOM plugin
        # owns DSV4 compatibility without patching SGLang source.
        if len(args) == 2 and hasattr(args[0], "forward_mode"):
            forward_batch, bs = args
            forward_mode = forward_batch.forward_mode
            if forward_mode.is_target_verify() or bool(
                getattr(forward_mode, "is_draft_extend", lambda **kwargs: False)(
                    include_v2=True
                )
            ):
                return self._init_verify_cuda_graph_metadata(
                    bs=bs,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                    seq_lens_cpu=getattr(forward_batch, "seq_lens_cpu", None),
                    forward_mode=forward_mode,
                    out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
                    positions=getattr(forward_batch, "positions", None),
                    spec_info=getattr(forward_batch, "spec_info", None),
                    actual_forward_mode=forward_mode,
                )
            return self._init_decode_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=getattr(forward_batch, "seq_lens_cpu", None),
                forward_mode=forward_mode,
                out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
                positions=getattr(forward_batch, "positions", None),
                actual_forward_mode=forward_mode,
            )

        bs = kwargs.get("bs", args[0] if len(args) > 0 else None)
        req_pool_indices = kwargs.get(
            "req_pool_indices", args[1] if len(args) > 1 else None
        )
        seq_lens = kwargs.get("seq_lens", args[2] if len(args) > 2 else None)
        seq_lens_sum = kwargs.get("seq_lens_sum", args[3] if len(args) > 3 else None)
        encoder_lens = kwargs.get("encoder_lens", args[4] if len(args) > 4 else None)
        forward_mode = kwargs.get("forward_mode", args[5] if len(args) > 5 else None)
        spec_info = kwargs.get("spec_info", args[6] if len(args) > 6 else None)
        seq_lens_cpu = kwargs.get("seq_lens_cpu", args[7] if len(args) > 7 else None)
        del seq_lens_sum, encoder_lens
        replay_batch = getattr(self, "_replay_forward_batch", None)
        if forward_mode.is_target_verify() or bool(
            getattr(forward_mode, "is_draft_extend", lambda **kwargs: False)(
                include_v2=True
            )
        ):
            return self._init_verify_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                forward_mode=forward_mode,
                out_cache_loc=getattr(replay_batch, "out_cache_loc", None),
                positions=getattr(replay_batch, "positions", None),
                spec_info=spec_info,
                actual_forward_mode=getattr(replay_batch, "forward_mode", forward_mode),
            )
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
            build_atom_v4_verify_graph_metadata_from_sglang,
        )

        bs = int(max_bs)
        tokens_per_req = max(1, int(max_num_tokens) // max(1, bs))
        seq_lens = torch.full(
            (bs,), tokens_per_req, dtype=torch.int32, device=self.device
        )
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=self.device)
        positions = torch.arange(tokens_per_req, dtype=torch.int64, device=self.device)
        positions = positions.repeat(bs)
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
        forward_mode = (
            ForwardMode.TARGET_VERIFY
            if is_target_verify_graph
            else (
                ForwardMode.DRAFT_EXTEND_V2
                if is_draft_extend_graph
                else ForwardMode.DECODE
            )
        )
        self._cuda_graph_seq_len_fill_value = (
            max(tokens_per_req, 1024) if is_graph_extend else 1
        )
        if is_graph_extend:
            seq_lens.fill_(self._cuda_graph_seq_len_fill_value)
            positions = (
                torch.arange(tokens_per_req, dtype=torch.int64, device=self.device)
                + (self._cuda_graph_seq_len_fill_value - tokens_per_req)
            ).repeat(bs)
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            actual_forward_mode=forward_mode,
            batch_size=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.detach().cpu(),
            out_cache_loc=None,
            spec_info=SimpleNamespace(num_tokens_per_req=tokens_per_req),
        )
        atom_model = getattr(getattr(self.model_runner, "model", None), "model", None)
        if is_graph_extend:
            self.atom_v4_graph_metadata = (
                build_atom_v4_verify_graph_metadata_from_sglang(
                    forward_batch,
                    positions,
                    proxy_pool=self.token_to_kv_pool,
                    req_to_token_pool=self.req_to_token_pool,
                    model=atom_model,
                )
            )
        else:
            self.atom_v4_graph_metadata = (
                build_atom_v4_decode_graph_metadata_from_sglang(
                    forward_batch,
                    positions,
                    proxy_pool=self.token_to_kv_pool,
                    req_to_token_pool=self.req_to_token_pool,
                    model=atom_model,
                )
            )
        ATOMDeepseekV4BackendForSgl._last_atom_v4_graph_metadata = (
            self.atom_v4_graph_metadata
        )
        return None

    def get_cuda_graph_seq_len_fill_value(self):
        return int(self._cuda_graph_seq_len_fill_value)

    def get_verify_buffers_to_fill_after_draft(self):
        graph_runner = getattr(self.model_runner, "graph_runner", None)
        buffers = getattr(graph_runner, "buffers", None)
        if buffers is None:
            return [None, None]
        # Let SGLang's tree builder fill the captured mask buffer in-place.
        # Keep positions allocated by the builder: it returns the full provided
        # buffer, while replay_prepare expects an exact raw-token-length tensor.
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
        if torch.is_tensor(positions):
            copy_n = min(int(positions.numel()), total)
            if copy_n:
                buffers.positions[:copy_n].copy_(positions[:copy_n])
            if total > copy_n:
                buffers.positions[copy_n:total].zero_()
            positions = buffers.positions[:total]
        else:
            positions = buffers.positions[:total]

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
        self._init_verify_cuda_graph_metadata(
            bs=int(cuda_graph_bs),
            req_pool_indices=buffers.req_pool_indices[: int(cuda_graph_bs)],
            seq_lens=buffers.seq_lens[: int(cuda_graph_bs)],
            seq_lens_cpu=(
                seq_lens_cpu[: int(cuda_graph_bs)] if seq_lens_cpu is not None else None
            ),
            forward_mode=forward_mode,
            out_cache_loc=buffers.out_cache_loc[:total],
            positions=positions,
            spec_info=spec_info,
            actual_forward_mode=forward_mode,
        )

    def forward_decode(self, *args, **kwargs):
        raise RuntimeError("ATOM DeepSeek-V4 SGLang bridge should use ATOM attention")

    def forward_extend(self, *args, **kwargs):
        raise RuntimeError("ATOM DeepSeek-V4 SGLang bridge should use ATOM attention")
