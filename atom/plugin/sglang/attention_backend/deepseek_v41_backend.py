import logging

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

logger = logging.getLogger("atom.plugin.sglang.attention_backend.deepseek_v41")


class ATOMDeepseekV41BackendForSgl(AttentionBackend):
    """SGLang backend shim for ATOM-owned DeepSeek-V4.1 (CSA2) attention.

    SGLang needs a backend object for scheduling and forward-context
    publication: the plugin reads ``req_to_token_pool`` off it during the
    forward. The CSA2 pool, step metadata and kernels belong to ATOM
    (``deepseek_v41_bridge``). Nothing here computes attention.

    Eager steps are staged by the wrapper before the model runs. Decode CUDA
    graphs (``--cuda-graph-backend-decode full``) are staged here instead, in
    ``init_forward_metadata_out_graph``, which SGLang calls outside the
    captured region once per bucket at capture and before every replay. Under
    DSpark the same runner class captures the target's verify (TARGET_VERIFY
    x6, a tentative step staged here) and the draft's block (TARGET_VERIFY
    x5, staged by its own forward from SGLang's fixed buffers).
    """

    needs_cpu_seq_lens = True

    def __init__(self, model_runner, *args, **kwargs):
        del args, kwargs
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        if not getattr(self.token_to_kv_pool, "is_atom_v41_proxy_pool", False):
            raise RuntimeError(
                "ATOMDeepseekV41BackendForSgl needs the ATOM DeepSeek-V4.1 proxy "
                f"KV pool, got {type(self.token_to_kv_pool).__name__}"
            )
        # A DSpark draft runner binds the draft role of the proxy (a view of
        # the target arena); the target binds the arena's owner. A mix is a
        # pool from the wrong runner.
        is_draft = bool(getattr(model_runner, "is_draft_worker", False))
        if is_draft != bool(
            getattr(self.token_to_kv_pool, "is_atom_v41_draft_proxy", False)
        ):
            raise RuntimeError(
                "ATOMDeepseekV41BackendForSgl: the "
                f"{'draft' if is_draft else 'target'} runner holds the "
                f"{'target' if is_draft else 'draft'} role of the V4.1 proxy pool"
            )
        self.role = "draft" if is_draft else "target"
        logger.info("Initializing ATOMDeepseekV41BackendForSgl role=%s", self.role)
        # bs -> (input_ids address, arena address) each bucket was recorded on;
        # None until a decode-graph runner sets the graph state up.
        self._v41_graph_buckets = None
        self._v41_replayed_buckets = set()
        # decode | verify | draft once a decode-graph runner set the state up
        # (`init_cuda_graph_state`); the DSpark stage gate keys on it.
        self._v41_graph_kind = None
        # The last replay's metadata, for debugging: the graph reads fixed
        # addresses and no model Python runs on replay.
        self._v41_last_replay_md = None

    @staticmethod
    def get_name() -> str:
        return "dsv4"

    def init_forward_metadata(self, forward_batch):
        # The bridge derives ATOM's metadata from the batch inside the forward.
        self.forward_metadata = None

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """One decode-graph runner's static state, by what it captures:

        - ``decode``: plain decode, one row a request (no speculation);
        - ``verify``: the DSpark target's TARGET_VERIFY, ``6`` rows a
          request, a tentative ATOM step staged outside the graph;
        - ``draft``: the DSpark draft block, a TARGET_VERIFY of ``5`` rows a
          request on the draft runner; its inputs come off SGLang's fixed
          buffers inside the forward and it writes no KV, so nothing is
          staged outside the graph but checks.
        """
        from sglang.srt.environ import envs

        from atom.plugin.sglang.deepseek_v41_bridge import (
            check_deepseek_v41_graph_capacity,
        )

        if envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get():
            # The glue graph would freeze this backend's host staging (the
            # slot / block-table reads and ATOM's publication) into a replay.
            raise NotImplementedError(
                "DeepSeek-V4.1 decode graphs stage every replay on the host; "
                "unset SGLANG_ENABLE_METADATA_GLUE_GRAPH"
            )
        max_bs, max_num_tokens = int(max_bs), int(max_num_tokens)
        cache = getattr(self.token_to_kv_pool, "atom_cache", None)
        geometry = getattr(cache, "geometry", None)
        if self.role == "draft":
            kind, width = "draft", self._init_draft_graph_state(max_bs, max_num_tokens)
        elif getattr(geometry, "speculative_tokens", 0):
            kind, width = "verify", self._init_verify_graph_state(
                max_bs, max_num_tokens
            )
        else:
            check_deepseek_v41_graph_capacity(
                self.token_to_kv_pool,
                self.req_to_token_pool.req_to_token,
                max_bs,
                max_num_tokens,
            )
            kind, width = "decode", 1
        self._v41_graph_kind = kind
        self._v41_graph_buckets = {}
        self._v41_replayed_buckets = set()
        pool = self.token_to_kv_pool
        logger.info(
            "ATOM V4.1 graph state kind=%s role=%s max_bs=%d width=%d "
            "max_num_tokens=%d capture slots %s",
            kind,
            self.role,
            max_bs,
            width,
            max_num_tokens,
            (
                f"{pool.capture_slot_base}..{pool.num_slots - 1}"
                if kind != "draft"
                else "unused (pad rows read slot 0)"
            ),
        )

    def _dspark_graph_preamble(self, kind, width, max_bs, max_num_tokens):
        from atom.plugin.sglang.deepseek_v41_bridge import (
            get_deepseek_v41_builder,
            install_v41_dspark_graph_stage_gate,
        )

        builder = get_deepseek_v41_builder()
        cache = getattr(self.token_to_kv_pool, "atom_cache", None)
        if builder.cache is None or cache is not builder.cache:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark {kind} graphs: the {self.role} pool is not "
                "bound to the builder's cache"
            )
        if max_num_tokens != max_bs * width:
            raise ValueError(
                f"DeepSeek-V4.1 DSpark {kind} graphs need {width} rows a request: "
                f"max_num_tokens {max_num_tokens} != {max_bs} x {width}"
            )
        if max_bs > builder.max_bs:
            raise ValueError(
                f"DeepSeek-V4.1 DSpark {kind} graphs of {max_bs} requests exceed "
                f"the builder's {builder.max_bs}; align --max-running-requests "
                "with the cuda graph batch sizes"
            )
        # Parsed (and the gate installed) before any capture: a bad stage
        # list fails the launch, not the first replay.
        install_v41_dspark_graph_stage_gate()
        return builder

    def _require_not_capturing(self):
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark graph state must be prepared before capture"
            )

    def _init_draft_graph_state(self, max_bs, max_num_tokens) -> int:
        from atom.plugin.sglang.deepseek_v41_bridge import V41_DSPARK_DRAFT_TOKENS

        pool = self.token_to_kv_pool
        if not getattr(pool, "is_atom_v41_draft_proxy", False):
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark draft graphs need the draft role of the proxy"
            )
        width = V41_DSPARK_DRAFT_TOKENS
        self._dspark_graph_preamble("draft", width, max_bs, max_num_tokens)
        # The draft's lazy caches (the stages' layer ids, the noise row) are
        # built here, outside any capture, rather than by the first warmup.
        self._require_not_capturing()
        prepare = getattr(self.model_runner.model, "prepare_v41_draft_graph", None)
        if prepare is None:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark draft graphs need the ATOM draft wrapper "
                f"(got {type(self.model_runner.model).__name__})"
            )
        prepare()
        self._v41_noise_token_id = int(self.model_runner.model.noise_token_id)
        return width

    def _init_verify_graph_state(self, max_bs, max_num_tokens) -> int:
        from atom.plugin.sglang.deepseek_v41_bridge import (
            V41_DSPARK_DRAFT_TOKENS,
            V41_DSPARK_VERIFY_TOKENS,
            check_deepseek_v41_graph_capacity,
        )

        pool = self.token_to_kv_pool
        speculative = int(pool.atom_cache.geometry.speculative_tokens)
        if speculative != V41_DSPARK_DRAFT_TOKENS:
            raise NotImplementedError(
                f"DeepSeek-V4.1 verify graphs serve DSpark's {V41_DSPARK_DRAFT_TOKENS} "
                f"drafts, not a pool of {speculative}"
            )
        width = V41_DSPARK_VERIFY_TOKENS
        self._dspark_graph_preamble("verify", width, max_bs, max_num_tokens)
        # Arena identity, device Engram, plane score workspace, builder and
        # capture-slot capacity.
        check_deepseek_v41_graph_capacity(
            pool, self.req_to_token_pool.req_to_token, max_bs, max_num_tokens
        )
        # Under decoder SWA bounded replay a verify still runs every row: only
        # a prefill publishes a decoder tail (`_publish_decoder`), and a
        # verify is staged as a tentative non-prefill step, so its in-graph
        # draft context write takes every row it ran
        # (`stage_deepseek_v41_verify_replay` refuses a replaying step).
        # The verify graph records the draft's context-KV write: build the
        # fused weights it reads outside any capture.
        self._require_not_capturing()
        surface = getattr(self.model_runner.model, "_v41_dspark", None)
        draft = getattr(surface, "draft", None)
        if draft is None:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark verify graphs captured before the draft "
                "was attached to the target"
            )
        draft._context_kv_fusion()
        draft._draft_kv_tail_fusion()
        return width

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        self.forward_metadata = None
        mode = (
            getattr(forward_batch, "actual_forward_mode", None)
            or forward_batch.forward_mode
        )
        if self._v41_graph_buckets is None:
            raise RuntimeError(
                "DeepSeek-V4.1 graph metadata staged before init_cuda_graph_state"
            )
        kind = self._v41_graph_kind
        if kind == "decode":
            if not mode.is_decode():
                raise NotImplementedError(
                    "DeepSeek-V4.1 graphs on the SGLang plugin cover plain decode "
                    f"only, not {getattr(mode, 'name', mode)}; launch with "
                    "--cuda-graph-backend-prefill disabled"
                )
        elif not bool(getattr(mode, "is_target_verify", lambda: False)()):
            raise NotImplementedError(
                f"DeepSeek-V4.1 DSpark {kind} graphs replay TARGET_VERIFY only, "
                f"not {getattr(mode, 'name', mode)}"
            )
        if kind == "draft":
            self._stage_draft_graph(forward_batch, in_capture)
        else:
            self._stage_target_graph(forward_batch, in_capture)

    def _stage_target_graph(self, forward_batch, in_capture):
        from atom.plugin.sglang.deepseek_v41_bridge import (
            count_v41_dspark,
            stage_deepseek_v41_graph_capture,
            stage_deepseek_v41_graph_replay,
            stage_deepseek_v41_verify_replay,
        )

        pool = self.token_to_kv_pool
        bs = int(forward_batch.batch_size)
        kind = self._v41_graph_kind
        if in_capture:
            stage_deepseek_v41_graph_capture(forward_batch, pool)
            self._v41_graph_buckets[bs] = (
                forward_batch.input_ids.data_ptr(),
                pool.raw_arena.data_ptr(),
            )
            if kind == "verify":
                logger.info(
                    "V4.1 DSpark verify graph capture bs=%d width=%d",
                    bs,
                    int(forward_batch.input_ids.shape[0]) // max(bs, 1),
                )
            return
        recorded = self._v41_graph_buckets.get(bs)
        if recorded is None:
            raise RuntimeError(f"DeepSeek-V4.1 {kind} graph bs={bs} was not captured")
        if recorded != (forward_batch.input_ids.data_ptr(), pool.raw_arena.data_ptr()):
            raise RuntimeError(
                f"DeepSeek-V4.1 {kind} graph bs={bs} replays on addresses its "
                "capture did not record"
            )
        table = self.req_to_token_pool.req_to_token
        if kind == "verify":
            self._v41_last_replay_md = stage_deepseek_v41_verify_replay(
                forward_batch, table
            )
            count_v41_dspark("verify_graph")
            if int(getattr(forward_batch, "num_padding", 0) or 0):
                count_v41_dspark("verify_padded")
        else:
            self._v41_last_replay_md = stage_deepseek_v41_graph_replay(
                forward_batch, table
            )
        self._log_first_replay(kind, forward_batch, bs)

    def _stage_draft_graph(self, forward_batch, in_capture):
        from atom.plugin.sglang.deepseek_v41_bridge import (
            count_v41_dspark,
            v41_dspark_checks_enabled,
        )

        bs = int(forward_batch.batch_size)
        addresses = (
            forward_batch.input_ids.data_ptr(),
            forward_batch.positions.data_ptr(),
            forward_batch.req_pool_indices.data_ptr(),
        )
        width = int(getattr(forward_batch.spec_info, "draft_token_num", 0) or 0)
        if int(forward_batch.input_ids.shape[0]) != bs * width or width < 1:
            raise ValueError(
                f"DeepSeek-V4.1 DSpark draft graph of {bs} requests x {width} over "
                f"{int(forward_batch.input_ids.shape[0])} input rows"
            )
        if in_capture:
            self._require_not_capturing()
            self._v41_graph_buckets[bs] = addresses
            logger.info("V4.1 DSpark draft graph capture bs=%d width=%d", bs, width)
            return
        recorded = self._v41_graph_buckets.get(bs)
        if recorded is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark draft graph bs={bs} was not captured"
            )
        if recorded != addresses:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark draft graph bs={bs} replays on addresses "
                "its capture did not record"
            )
        # No Python runs on replay: the draft reads committed windows only,
        # so the commit check `read_windows` makes eagerly is made here.
        self.token_to_kv_pool.atom_cache.require_committed()
        if v41_dspark_checks_enabled():
            self._check_draft_rows(forward_batch, bs, width)
        count_v41_dspark("draft_graph")
        if int(getattr(forward_batch, "num_padding", 0) or 0):
            count_v41_dspark("draft_padded")
        self._log_first_replay("draft", forward_batch, bs)

    def _check_draft_rows(self, forward_batch, bs, width):
        """Bring-up checks of the draft block (a sync): the live rows'
        positions are ``seq_lens + 0..width-1``, columns 1.. the noise
        token, one live request row each."""
        raw = bs - int(getattr(forward_batch, "num_padding", 0) or 0)
        slots = forward_batch.req_pool_indices[:raw].cpu()
        if torch.unique(slots).numel() != raw or bool((slots <= 0).any()):
            raise ValueError(
                "DeepSeek-V4.1 DSpark draft graph needs one live req_pool_idx "
                f"per request; got {slots.tolist()}"
            )
        pos = forward_batch.positions[: raw * width].view(raw, width)
        seq = forward_batch.seq_lens[:raw].to(device=pos.device, dtype=pos.dtype)
        expected = seq[:, None] + torch.arange(
            width, device=pos.device, dtype=pos.dtype
        )
        if not torch.equal(pos, expected) or bool((pos[:, 0] < 1).any()):
            raise ValueError(
                "DeepSeek-V4.1 DSpark draft graph positions are not seq_lens + "
                f"0..{width - 1} over a prefix"
            )
        ids = forward_batch.input_ids[: raw * width].view(raw, width)
        if width > 1 and not bool((ids[:, 1:] == self._v41_noise_token_id).all()):
            raise ValueError(
                "DeepSeek-V4.1 DSpark draft graph columns 1.. are not the noise "
                f"token {self._v41_noise_token_id}"
            )

    def _log_first_replay(self, kind, forward_batch, bs):
        if bs in self._v41_replayed_buckets:
            return
        self._v41_replayed_buckets.add(bs)
        raw = bs - int(getattr(forward_batch, "num_padding", 0) or 0)
        if kind == "decode":
            logger.info("V4.1 decode graph replay bs=%d raw=%d", bs, raw)
        else:
            logger.info("V4.1 DSpark %s graph replay bs=%d raw=%d", kind, bs, raw)

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
