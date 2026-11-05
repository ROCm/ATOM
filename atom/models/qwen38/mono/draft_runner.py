# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MTP drafter's decoder layer on one mono program (``recipe.mtp_bf16_blocks``: BF16, as ATOM keeps ``mtp.*``):
the first draft step (``SPEC`` rows a request, the MLP half on each request's selected row) and the later one-row
steps, asked first by every draft step of at most 16 rows (``draft_fallback.forward``).
"""

import logging

import torch

from atom.models.qwen38.mono.config import BATCHES, SPEC
from atom.models.qwen38.mono.kernels.prep import draft_meta
from atom.models.qwen38.mono.runner import Kernels, Layer, qsa_module
from atom.models.qwen38.mono.step_meta import QSAPools, draft_rows_error
from atom.models.qwen38.mono.weights import load_layer
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class DraftRunner:
    def __init__(self, inner, atom_config, geo, checkpoint: str):
        self.geo, self.idx = geo, geo.layers
        with torch.device("cpu"):  # (see ``Kernels.serves``)
            self.kernels = Kernels("Qwen mono draft", ("mtp",), geo)
            q = load_layer(self.idx, checkpoint, geo, "mtp", experts=True)
            self.layer = Layer(q, self.kernels)
            logger.info("Qwen mono draft: layer %d (mtp) loaded", self.idx)
        self.qsa = qsa_module(inner.layers[0])
        width = inner.hc_count * inner.hidden_size
        self.io = torch.zeros(max(BATCHES), width, dtype=torch.bfloat16, device="cuda")
        # the program's input: the draft model writes its fc sum here (``MonoDraftModel._fc_sum``), the kernel binds it
        self.inbuf = torch.zeros(
            SPEC * max(BATCHES), width, dtype=torch.bfloat16, device="cuda"
        )
        self.kernels.io.update((self.io.data_ptr(), self.inbuf.data_ptr()))
        self.checked: set[tuple[int, int]] = set()
        table_width = -(
            -int(atom_config.max_model_len) // int(atom_config.kv_cache_block_size)
        )
        for b in BATCHES:
            self.kernels.build(b, 1, table_width)
            self.kernels.build(b, SPEC, table_width)
        # bind (pack) the weights now, once for every shape (the shapes' programs share their weight args), then drop
        # the checkpoint's bf16 expert copies (~5 GB) the pack read
        lk, _ = self.kernels.kernel("mtp", BATCHES[0], 1, table_width)
        with torch.device("cpu"):
            self.layer.bind(lk.blocks)
        experts = [n for n in q.linears if ".experts." in n or ".shared_expert." in n]
        for n in experts:
            del q.linears[n]
        torch.cuda.empty_cache()
        logger.info("Qwen mono draft: layer %d (bf16) hooked, batches %s, served %s",
                    self.idx, BATCHES, sorted(self.kernels.served))  # fmt: skip

    def serve(self, model, hidden, selected_rows):
        """(state, head input) of a draft step on the program, else None (the fallback draft): the first step's
        ``SPEC`` rows a request with ``selected_rows``, or a later step's one row a request.
        """
        n = hidden.shape[0]
        if selected_rows is not None:
            # the first step: the attention on every row, the rest on each selected row
            batch, spec = selected_rows.numel(), SPEC
            if n != batch * SPEC:
                return None
        else:
            batch, spec = n, 1
        if batch not in BATCHES:
            return None
        rows = batch * spec
        fwd = get_forward_context()
        md = getattr(fwd, "attn_metadata", None)
        qsa = getattr(md, "qsa_metadata", None) if md is not None else None
        if qsa is None or fwd.context is None or fwd.context.is_prefill:
            return None
        qsa = qsa.for_tokens(rows)
        width = int(qsa.block_tables.shape[1])
        if not self.kernels.serves(batch, spec, width):
            return None
        pos = qsa.logical_positions[:rows]
        bt = qsa.block_tables[:batch]
        capturing = torch.cuda.is_current_stream_capturing()
        if (
            (batch, spec) not in self.checked
            and not capturing
            and bool((pos >= 0).all())
            and bool((qsa.slot_mapping[:rows] >= 0).all())
        ):  # (a real step: warm-up / capture rows may carry slot -1, nothing written)
            ps = [int(x) for x in pos.tolist()]
            slots = [int(x) for x in qsa.slot_mapping[:rows].tolist()]
            why = draft_rows_error(
                ps, slots, bt.tolist(), self.qsa.k_cache.shape[1], batch, spec
            )
            if why is not None:
                raise RuntimeError(f"Qwen mono draft: B{batch}S{spec} {why}")
            logger.info(
                "Qwen mono draft: B%dS%d addressing checked (positions %s, slots %s)",
                batch,
                spec,
                ps,
                slots,
            )
            self.checked.add((batch, spec))
        q = self.qsa
        pools = QSAPools(
            self.geo, q.k_cache, q.v_cache, q.raw_key_cache, q.compressed_key_cache
        )
        m, counters = self.kernels.begin_step_buffers(batch, spec, width)
        sel = None if spec == 1 else selected_rows.reshape(batch)
        draft_meta(
            m, counters, qsa.logical_positions, qsa.slot_mapping, bt, sel, batch, spec
        )
        out = self.io[:batch]
        self.layer.step(hidden, out, batch, spec, pools)
        head_input, (state, _) = model.hyper_connection_mixer.mix_fused(out, None)
        return state.view(-1, model.hc_count, model.hidden_size), head_input
