"""Runtime context helpers for ATOM's SGLang plugin path."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import ClassVar, Optional, Union

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

_RUNTIME_SENTINEL = object()
_current_forward_batch: ContextVar[Optional[ForwardBatch]] = ContextVar(
    "atom_sglang_current_forward_batch", default=None
)


def get_current_forward_batch():
    return _current_forward_batch.get()


@contextmanager
def bind_current_forward_batch(forward_batch: Optional[ForwardBatch]):
    token = _current_forward_batch.set(forward_batch)
    try:
        yield
    finally:
        _current_forward_batch.reset(token)


@contextmanager
def plugin_runtime_scope(
    *,
    framework: Optional[str] = None,
    atom_config=_RUNTIME_SENTINEL,
):
    """Temporarily bind process-global ATOM plugin runtime state.

    SGLang target/draft wrappers can coexist during speculative decoding, while
    ATOM core still reads process-global framework/config state in some paths.
    Keep those globals scoped to one wrapper call and restore them afterwards.
    """

    import atom.config as atom_config_module
    import atom.plugin.prepare as plugin_prepare

    prev_framework = plugin_prepare._CURRENT_FRAMEWORK
    prev_atom_config = getattr(atom_config_module, "_current_atom_config", None)

    if framework is not None:
        plugin_prepare._set_framework_backbone(framework)
    if atom_config is not _RUNTIME_SENTINEL:
        atom_config_module._current_atom_config = atom_config

    try:
        yield
    finally:
        plugin_prepare._CURRENT_FRAMEWORK = prev_framework
        atom_config_module._current_atom_config = prev_atom_config


@dataclass(frozen=True)
class SGLangForwardBatchMetadata:
    """Small context object for one SGLang model forward."""

    forward_batch: Optional[ForwardBatch]
    pp_proxy_tensors: Optional[PPProxyTensors] = None
    save_kv_cache: bool = True
    _current: ClassVar[ContextVar[Optional["SGLangForwardBatchMetadata"]]] = ContextVar(
        "atom_sglang_current_forward_batch_metadata",
        default=None,
    )

    @classmethod
    def current(cls) -> Optional["SGLangForwardBatchMetadata"]:
        return cls._current.get()

    @classmethod
    def build(
        cls,
        forward_batch: Optional[
            Union[ForwardBatch, "SGLangForwardBatchMetadata"]
        ] = None,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        save_kv_cache: Optional[bool] = None,
    ) -> Optional["SGLangForwardBatchMetadata"]:
        if isinstance(forward_batch, cls):
            return forward_batch
        if forward_batch is None and pp_proxy_tensors is None and save_kv_cache is None:
            return cls.current()
        return cls(
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            save_kv_cache=True if save_kv_cache is None else save_kv_cache,
        )

    @classmethod
    @contextmanager
    def bind(cls, metadata: Optional["SGLangForwardBatchMetadata"]):
        meta_token = cls._current.set(metadata)
        batch_token = _current_forward_batch.set(
            None if metadata is None else metadata.forward_batch
        )
        try:
            yield metadata
        finally:
            _current_forward_batch.reset(batch_token)
            cls._current.reset(meta_token)

    @staticmethod
    def to_intermediate_tensors(
        intermediate_tensors,
        metadata: Optional["SGLangForwardBatchMetadata"],
    ):
        if intermediate_tensors is not None or metadata is None:
            return intermediate_tensors
        pp_proxy_tensors = metadata.pp_proxy_tensors
        if pp_proxy_tensors is None:
            return intermediate_tensors
        tensors = getattr(pp_proxy_tensors, "tensors", None)
        if tensors is None:
            return intermediate_tensors
        from atom.models.utils import IntermediateTensors

        return IntermediateTensors(dict(tensors))
