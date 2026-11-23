# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Debug check of the whole-row K-up against AITER's NVFP4 epilogue on the
model's own inputs (``ATOM_MLA_NVFP4_KUP_CHECK=1``, eager calls only): each
call also runs the AITER kernel into separate planes and compares the bytes;
the first few calls that differ are logged and saved (inputs and both
outputs) under ``ATOM_MLA_NVFP4_KUP_CHECK_DIR``."""

import logging
import os

import torch

logger = logging.getLogger("atom")
_MAX_DUMPS = int(os.environ.get("ATOM_MLA_NVFP4_KUP_CHECK_DUMPS", "3"))
_state = {"calls": 0, "bad_calls": 0, "dumps": 0}


def check(q_nope, w_k, qn, qa) -> None:
    from atom.model_ops.attentions.nvfp4_q_triton import _aiter_kup_nvfp4

    gluon = _aiter_kup_nvfp4()
    if gluon is None:
        return
    qn_ref = torch.empty_like(qn)
    qa_ref = torch.zeros_like(qa)
    gluon(q_nope.transpose(0, 1), w_k, nvfp4_out=(qn_ref.transpose(0, 1), qa_ref[..., 64:96].transpose(0, 1)))
    bad_n = qn != qn_ref
    bad_s = qa[..., 64:96] != qa_ref[..., 64:96]
    _state["calls"] += 1
    if not (bad_n.any() or bad_s.any()):
        if _state["calls"] % 2000 == 0:
            logger.info("NVFP4 K-up check: %d calls, %d differ", _state["calls"], _state["bad_calls"])
        return
    _state["bad_calls"] += 1
    rows = (bad_n.flatten(1).any(1) | bad_s.flatten(1).any(1)).nonzero().flatten()
    heads = (bad_n.any(2) | bad_s.any(2))[rows].nonzero()
    logger.warning(
        "NVFP4 K-up check: call %d (M=%d, q_nope strides %s, w_k strides %s) differs: %d code bytes, "
        "%d scale bytes, rows %s, (row, head) %s",
        _state["calls"], q_nope.shape[0], tuple(q_nope.stride()), tuple(w_k.stride()),
        int(bad_n.sum()), int(bad_s.sum()), rows[:16].tolist(), heads[:8].tolist(),
    )
    out_dir = os.environ.get("ATOM_MLA_NVFP4_KUP_CHECK_DIR")
    if out_dir and _state["dumps"] < _MAX_DUMPS:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"kup_check_pid{os.getpid()}_call{_state['calls']}.pt")
        torch.save({"q_nope": q_nope.cpu(), "q_nope_stride": tuple(q_nope.stride()), "w_k": w_k.cpu(),
                    "qn": qn.cpu(), "qa": qa.cpu(), "qn_ref": qn_ref.cpu(), "qa_ref": qa_ref.cpu()}, path)
        _state["dumps"] += 1
        logger.warning("NVFP4 K-up check: saved %s", path)
