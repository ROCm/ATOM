# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Layer-by-layer comparison of the mono path against the original modules.

Debug only (``ATOM_MONO_CHECK=1``, meant for ``--enforce-eager``). The dense
layers run twice where mono has kernels for them, mono then the original
modules, compared where they hand the first sparse layer its (ar, res); the
step goes on from mono's. Every sparse
layer runs twice on the same (ar, res): first the mono layer kernel, then the
original decoder layer, fed ``ar`` on rank 0 and zeros elsewhere so its fused
all-reduce reproduces ``ar`` exactly. Both write the same cache bytes for the
step's tokens (K1 is bit-exact with the original insert), so running both is
side-effect free; mono going first reads only the cache entries it inserted.
Rank 0 logs, per layer and token, the difference of the residual and of the
reduced output.
"""

import json
import logging

import torch
from aiter import topk_select

from atom.model_ops.communication_op import tensor_model_parallel_all_reduce
from atom.models.minimax_m3.mono.config import HIDDEN
from atom.models.minimax_m3.mono.layout import SCRATCH, wide_moe
from atom.models.minimax_m3.mono.runner import N_DENSE, token_rows
from atom.mono.runtime.compare import compare
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def forward_checked(
    runner, input_ids: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    fwd = get_forward_context()
    runner.mailboxes.begin_step()
    ars = runner.ars
    # the dense layers, mono first, then the original ones on the same step,
    # compared where they hand the first sparse layer its (ar, res). A model
    # with no kernels for its prefix (M3.1) has one arm and nothing to compare.
    if not runner.dense_mono:
        res = runner.run_dense(input_ids, positions, (), [])
        n = res.shape[0]
    else:
        res = runner.run_dense_mono(input_ids, positions, fwd, (), []).clone()
        n = res.shape[0]
        ar = ars[0][:n].clone()
        ref_res = runner.run_dense(input_ids, positions, (), [])
        if runner.rank == 0:
            for k in range(n):
                logger.info(
                    "mono check dense layers token %d/%d pos %d: residual %s | "
                    "reduced out %s",
                    k,
                    n,
                    int(positions[k]),
                    compare(res[k], ref_res[k]),
                    compare(ar[k], ars[0][k]),
                )
        ars[0][:n].copy_(ar)
    rows = token_rows(fwd, n)
    zeros = torch.zeros(n, HIDDEN, dtype=ars[0].dtype, device=ars[0].device)
    for i, lw in enumerate(runner.sparse):
        layer = runner.model.layers[lw.layer_id]
        moe = layer.block_sparse_moe
        shared = moe.shared_experts
        h_in = ars[i % 2][:n].clone() if runner.rank == 0 else zeros
        res_in = res.clone()
        # mono first: it reads only the caches it inserts itself, as in a real step
        res = runner.run_sparse_layer(i, lw, fwd, positions, res, rows)
        if envs.ATOM_MONO_CHECK_ROUTED_ONLY:
            # the routed experts alone: the module is put back after the reference ran (a
            # changed module under the compiled original path recompiles and fails)
            moe.shared_experts = None
        if envs.ATOM_MONO_CHECK_MOE:
            # the bf16 xn the kernel's router published (a16w4 builds): the original MoE module on
            # exactly that input, all-reduced, against the kernel's reduced output
            xn = mono_xn(runner, n)
            moe_ar = tensor_model_parallel_all_reduce(layer.block_sparse_moe(xn)).view(
                n, HIDDEN
            )
            # the floor: the same module on the same rows one token at a time (another batch, so another
            # tile / split-K configuration of its kernels: only the accumulation order differs)
            one_ar = tensor_model_parallel_all_reduce(
                torch.cat([layer.block_sparse_moe(xn[k : k + 1]) for k in range(n)])
            ).view(n, HIDDEN)
            if runner.rank == 0:
                for k in range(n):
                    logger.info(
                        "mono check moe layer %d token %d/%d: reduced out %s | floor %s",
                        lw.layer_id,
                        k,
                        n,
                        compare(ars[(i + 1) % 2][k], moe_ar[k]),
                        compare(one_ar[k], moe_ar[k]),
                    )
        ref_partial, ref_res = layer(positions, h_in, res_in)
        if envs.ATOM_MONO_CHECK_ROUTED_ONLY:
            moe.shared_experts = shared
        ref_ar = tensor_model_parallel_all_reduce(ref_partial).view(n, HIDDEN)
        if runner.rank == 0:
            for k in range(n):
                logger.info(
                    "mono check layer %d token %d/%d pos %d seq_len %d: residual %s | "
                    "reduced out %s",
                    lw.layer_id,
                    k,
                    n,
                    int(positions[k]),
                    int(rows[1][k]),
                    compare(res[k], ref_res.view(n, HIDDEN)[k]),
                    compare(ars[(i + 1) % 2][k], ref_ar[k]),
                )
    return runner.finish_step(res)


def _mono_xn4(sc: torch.Tensor, n: int) -> torch.Tensor:
    """a4w4 builds: the MXFP4 xn rows (``xn4p`` / ``xs4p`` plain words, or the narrow build's tagged
    ``xn4`` / ``xs4``) dequantized to bf16: the input the routed experts saw (so the xn
    quantization error is not in a comparison made on it)."""
    if wide_moe(n):
        o, c = SCRATCH["xn4p"], SCRATCH["xs4p"]
        words = sc[o : o + n * HIDDEN // 8 * 4].view(torch.int32)
        codes = sc[c : c + n * HIDDEN // 32 * 4].view(torch.int32)
    else:
        o, c = SCRATCH["xn4"], SCRATCH["xs4"]
        words = sc[o : o + n * HIDDEN // 8 * 8].view(torch.int32)[0::2]
        codes = sc[c : c + n * HIDDEN // 32 * 8].view(torch.int32)[0::2]
    b = words.contiguous().view(torch.uint8).view(n, HIDDEN // 2)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=sc.device,
    )
    v = torch.stack([lut[(b & 15).long()], lut[(b >> 4).long()]], -1).view(
        n, HIDDEN // 32, 32
    )
    scale = torch.ldexp(
        torch.ones_like(codes, dtype=torch.float32), codes.view(n, -1) - 127
    )
    return (v * scale[..., None]).view(n, HIDDEN).to(torch.bfloat16)


def mono_xn(runner, n: int) -> torch.Tensor:
    """The bf16 xn rows of the layer kernel's last launch, from its scratch: the narrow
    build's tagged words (``xnb``) or the wide one's plain words (``xnbp``)."""
    sc = runner.scratch4
    if runner.moe_a4w4:
        return _mono_xn4(sc, n)
    if wide_moe(n):
        off = SCRATCH["xnbp"]
        words = sc[off : off + n * HIDDEN // 2 * 4].view(torch.int32)
    else:
        off = SCRATCH["xnb"]
        words = sc[off : off + n * HIDDEN // 2 * 8].view(torch.int32)[0::2]
    return words.contiguous().view(torch.bfloat16).view(n, HIDDEN).clone()


def _fingerprint(t: torch.Tensor) -> list[int]:
    """Two integer sums of a tensor's 16-bit words (plain and index-weighted):
    equal fingerprints mean equal bits for any difference a run could make."""
    w = t.contiguous().view(torch.int16).flatten().to(torch.int64)
    idx = torch.arange(1, w.numel() + 1, device=w.device)
    return [int(w.sum()), int((w * idx).sum())]


def forward_probed(
    runner, input_ids: torch.Tensor, positions: torch.Tensor, sums: dict
) -> torch.Tensor:
    """``runner.forward`` without aux layers, filling ``sums`` with the fingerprint
    of every stage: each dense layer's (hidden, residual), the first sparse layer's
    ``ar``, each sparse layer's (ar, h_mid) and the final hidden."""
    n = input_ids.numel()
    model = runner.model
    runner.mailboxes.begin_step()
    hidden = model.get_input_embeddings(input_ids)
    sums["embed"] = _fingerprint(hidden)
    residual = None
    for i in range(N_DENSE):
        hidden, residual = model.layers[i](positions, hidden, residual)
        sums[f"dense{i}"] = _fingerprint(hidden) + _fingerprint(residual)
    runner.ars[0][:n].copy_(tensor_model_parallel_all_reduce(hidden).view(n, HIDDEN))
    sums["ar_in"] = _fingerprint(runner.ars[0][:n])
    res = residual.view(n, HIDDEN)
    fwd = get_forward_context()
    rows = token_rows(fwd, n)
    # token 0's full blocks (the tail one also holds stale slots past the token)
    blocks = rows[0][0][: (int(rows[1][0]) - 1) // 128].long()
    for i, lw in enumerate(runner.sparse):
        # the history this layer reads, before this step inserts
        kv = fwd.kv_cache_data[f"layer_{lw.layer_id}"]
        sums[f"cache{lw.layer_id}"] = (
            _fingerprint(kv.k_cache[blocks])
            + _fingerprint(kv.v_cache[blocks])
            + _fingerprint(lw.attn_impl.index_cache[blocks])
        )
        res = runner.run_sparse_layer(i, lw, fwd, positions, res, rows)
        sums[f"L{lw.layer_id}"] = _fingerprint(
            runner.ars[(i + 1) % 2][:n]
        ) + _fingerprint(res)
    out = runner.finish_step(res)
    sums["final"] = _fingerprint(out)
    return out


def trace_logits(
    causal_lm,
    rank: int,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden,
    path,
    sums: dict | None = None,
) -> None:
    """Append this step's rows to ``path`` (rank 0, one JSON line per step): each
    row's input token, position and top-2 logits, to align two runs of one prompt
    token by token. ``hidden``: the mono forward's output."""
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    raw = causal_lm.compute_logits(hidden)
    pick = topk_select(raw, 1, tie="low")[1].view(-1)
    vals, ids = raw.float().topk(2, dim=-1)
    if rank == 0:
        with open(path, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "ids": input_ids.tolist(),
                        "pos": positions.tolist(),
                        "top_ids": ids.tolist(),
                        "top_vals": vals.tolist(),
                        "pick": pick.tolist(),
                        "sums": sums,
                    }
                )
                + "\n"
            )
