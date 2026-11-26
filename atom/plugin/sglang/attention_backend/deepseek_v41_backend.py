import logging

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

logger = logging.getLogger("atom.plugin.sglang.attention_backend.deepseek_v41")


class ATOMDeepseekV41BackendForSgl(AttentionBackend):
    """SGLang backend shim for ATOM-owned DeepSeek-V4.1 (CSA2) attention.

    It publishes ``req_to_token_pool`` to the forward; the pool, the step
    metadata and the kernels are ATOM's (``deepseek_v41_bridge``).

    Eager steps are staged by the wrapper before the model runs. Decode CUDA
    graphs (``--cuda-graph-backend-decode full``) are staged here instead, in
    ``init_forward_metadata_out_graph``, which SGLang calls outside the
    captured region once per bucket at capture and before every replay. Under
    DSpark the same runner class captures the target's verify (TARGET_VERIFY
    x6, a tentative step staged here) and the draft's block (TARGET_VERIFY
    x5, staged by its own forward from SGLang's fixed buffers).

    The bridge reads the host ``seq_lens`` mirror that SGLang's default
    ``needs_cpu_seq_lens`` (True) publishes.
    """

    def __init__(self, model_runner, *args, **kwargs):
        del args, kwargs
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        # A DSpark draft runner binds the draft role of the proxy (a view of
        # the target arena); the target binds the arena's owner. A mix is a
        # pool from the wrong runner.
        is_draft = bool(model_runner.is_draft_worker)
        if is_draft != self.token_to_kv_pool.is_atom_v41_draft_proxy:
            raise RuntimeError(
                "ATOMDeepseekV41BackendForSgl: the "
                f"{'draft' if is_draft else 'target'} runner holds the "
                f"{'target' if is_draft else 'draft'} role of the V4.1 proxy pool"
            )
        self.role = "draft" if is_draft else "target"
        logger.info("Initializing ATOMDeepseekV41BackendForSgl role=%s", self.role)
        # Buckets replayed so far (the first replay of each is logged).
        self._v41_replayed_buckets = set()
        # decode | verify | draft, set by `init_cuda_graph_state`.
        self._v41_graph_kind = None
        # The bridge's staging, bound by `init_cuda_graph_state`.
        self._stage_capture = None
        self._stage_replay = None

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
          staged outside the graph.

        SGLang calls this before any capture. The metadata glue graph, and
        Engram staged on the host, are refused at launch
        (`refuse_unsupported_sglang_features`).
        """
        from atom.plugin.sglang import deepseek_v41_bridge as bridge

        builder = bridge.check_deepseek_v41_graph_capacity(max_bs, max_num_tokens)
        if self.role == "draft":
            kind = "draft"
            # The draft's lazy caches (the stages' layer ids, the noise row),
            # built here rather than by the first warmup.
            self.model_runner.model.prepare_v41_draft_graph()
        elif builder.geometry.speculative_tokens:
            kind = "verify"
            # The verify graph records the draft's context-KV write: build the
            # fused weights it reads now. SGLang attaches the draft
            # (`attach_shared_modules`) before any graph capture.
            draft = self.model_runner.model._v41_dspark.draft
            draft._context_kv_fusion()
            draft._draft_kv_tail_fusion()
        else:
            kind = "decode"
        self._v41_graph_kind = kind
        # Bound once: the staging runs before every replay.
        self._stage_capture = bridge.stage_deepseek_v41_graph_capture
        self._stage_replay = (
            bridge.stage_deepseek_v41_verify_replay
            if kind == "verify"
            else bridge.stage_deepseek_v41_graph_replay
        )
        self._v41_replayed_buckets = set()
        pool = self.token_to_kv_pool
        logger.info(
            "ATOM V4.1 graph state kind=%s role=%s max_bs=%d max_num_tokens=%d "
            "capture slots %s",
            kind,
            self.role,
            max_bs,
            max_num_tokens,
            (
                f"{pool.capture_slot_base}..{pool.num_slots - 1}"
                if kind != "draft"
                else "unused (pad rows read slot 0)"
            ),
        )

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        self.forward_metadata = None
        # Capture hands the ForwardBatch itself; replay a view whose
        # forward_mode is the capture's and actual_forward_mode the batch's.
        mode = (
            forward_batch.forward_mode
            if in_capture
            else forward_batch.actual_forward_mode
        )
        kind = self._v41_graph_kind
        # The one mode check of the graph path (capture and replay staging
        # assume it): plain decode graphs replay DECODE, DSpark's verify and
        # draft graphs TARGET_VERIFY.
        if not (mode.is_decode() if kind == "decode" else mode.is_target_verify()):
            raise NotImplementedError(
                f"DeepSeek-V4.1 {kind} graphs on the SGLang plugin do not replay "
                f"{mode.name}; launch with --cuda-graph-backend-prefill disabled"
            )
        if kind == "draft":
            self._stage_draft_graph(forward_batch, in_capture)
        else:
            self._stage_target_graph(forward_batch, in_capture)

    def _stage_target_graph(self, forward_batch, in_capture):
        if in_capture:
            self._stage_capture(forward_batch, self.token_to_kv_pool)
            return
        # SGLang replays only a captured bucket, over the static buffers it
        # recorded (`build_replay_fb_view`); the arena never moves.
        self._stage_replay(forward_batch, self.req_to_token_pool.req_to_token)
        self._log_first_replay(self._v41_graph_kind, forward_batch)

    def _stage_draft_graph(self, forward_batch, in_capture):
        if in_capture:
            return
        # No Python runs on replay: the draft reads committed windows only,
        # so the commit check `read_windows` makes eagerly is made here.
        self.token_to_kv_pool.atom_cache.require_committed()
        self._log_first_replay("draft", forward_batch)

    def _log_first_replay(self, kind, forward_batch):
        bs = int(forward_batch.batch_size)
        if bs in self._v41_replayed_buckets:
            return
        self._v41_replayed_buckets.add(bs)
        raw = bs - int(forward_batch.num_padding)
        logger.info("V4.1 %s graph replay bs=%d raw=%d", kind, bs, raw)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1
