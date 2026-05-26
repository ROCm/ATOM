from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

logger = logging.getLogger("atom.plugin.sglang.tbo")

_PATCHED = False


def _pad_optional_list_field(batch: Any, field_name: str, batch_size: int) -> None:
    value = getattr(batch, field_name, None)
    if value is None or not isinstance(value, (list, tuple)):
        return
    if len(value) == batch_size:
        return
    if len(value) > batch_size:
        return

    padded = list(value)
    padded.extend([None] * (batch_size - len(value)))
    if isinstance(value, tuple):
        padded = tuple(padded)
    setattr(batch, field_name, padded)


def _normalize_tbo_batch_for_dp_padding(batch: Any) -> None:
    """Make SGLang TBO split tolerate DP max-len padded decode batches.

    Under DP attention SGLang may pad decode tensors from the local real request
    count to the cross-DP max batch size. Some optional host-side list fields
    such as ``rids`` still describe only real requests, but SGLang's native TBO
    splitter asserts that these fields have ``batch_size`` entries before it
    slices child batches. Padding those optional lists with ``None`` keeps the
    dummy padded row aligned without changing token tensors or sampling state.
    """

    batch_size = int(getattr(batch, "batch_size", 0) or 0)
    if batch_size <= 0:
        return

    forward_mode = getattr(batch, "forward_mode", None)
    if forward_mode is None or not getattr(forward_mode, "is_decode", lambda: False)():
        return

    for field_name in ("rids", "lora_ids"):
        _pad_optional_list_field(batch, field_name, batch_size)


def apply_sglang_tbo_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    two_batch_overlap = import_module("sglang.srt.batch_overlap.two_batch_overlap")
    TboForwardBatchPreparer = two_batch_overlap.TboForwardBatchPreparer

    original_prepare_raw = TboForwardBatchPreparer.prepare_raw

    def prepare_raw_with_dp_padding_fix(
        cls,
        batch,
        tbo_children_num_token_non_padded,
    ):
        _normalize_tbo_batch_for_dp_padding(batch)
        return original_prepare_raw(
            batch,
            tbo_children_num_token_non_padded=tbo_children_num_token_non_padded,
        )

    TboForwardBatchPreparer.prepare_raw = classmethod(prepare_raw_with_dp_padding_fix)
    _PATCHED = True
    logger.info("[SGL+ATOM TBO] patched SGLang TBO DP padding list normalization")
