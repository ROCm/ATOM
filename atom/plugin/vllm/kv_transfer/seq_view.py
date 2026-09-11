# SPDX-License-Identifier: MIT
"""Present a vLLM ``Request`` as the ``seq`` ATOM's offload scheduler expects.

ATOM's offload scheduler was written against ``atom.model_engine.sequence.
Sequence``. It reads nine attributes off it, four of which are offload's own
mutable bookkeeping, and -- importantly -- it *stores the object* and later
compares identity (``previous is not seq``, ``entry[0] is not seq``) to detect
that a request id was reused by a new request. So a view must be created ONCE
per request and reused; handing out a fresh wrapper per call would look like a
new request every step and keep resetting the load lifecycle.

The read-only half is projected from the vLLM Request; the mutable half lives
here, exactly as it lives on ATOM's Sequence.
"""

from typing import Any


class SeqView:
    """One vLLM request, shaped like an ATOM ``Sequence`` for offload."""

    __slots__ = (
        "_load_operation",
        "_num_cached_tokens",
        "_request",
        "block_table",
        "offload_handoff_boundary_tokens",
        "offload_loaded_tokens",
        "prefix_hashes_published",
    )

    def __init__(self, request: Any) -> None:
        self._request = request
        self._num_cached_tokens = 0
        self.block_table: list[int] = []
        # Offload's own state, mirroring the fields on ATOM's Sequence.
        self.offload_loaded_tokens = 0
        self.offload_handoff_boundary_tokens = 0
        self.prefix_hashes_published = False
        self._load_operation = None

    # -- projected from the vLLM request --------------------------------

    @property
    def id(self) -> str:
        return self._request.request_id

    @property
    def token_ids(self) -> list[int]:
        # LMCache keys are derived from prompt tokens; ``all_token_ids`` grows
        # with decode output, which must not change a prefix's key mid-request.
        return self._request.prompt_token_ids

    @property
    def num_prompt_tokens(self) -> int:
        return len(self._request.prompt_token_ids)

    @property
    def num_cached_tokens(self) -> int:
        """The scheduler's computed frontier for this request.

        ATOM reads this to decide which chunks are safe to save (everything
        below the frontier has been computed) and how much of a lookup hit is
        already resident in HBM. vLLM reports the same quantity as
        ``num_computed_tokens``, pushed in by the connector each step.
        """
        return self._num_cached_tokens

    def set_num_cached_tokens(self, value: int) -> None:
        self._num_cached_tokens = int(value)

    def set_block_table(self, block_ids: list[int]) -> None:
        self.block_table = list(block_ids)

    def __repr__(self) -> str:
        return (
            f"SeqView(id={self.id!r}, prompt={self.num_prompt_tokens}, "
            f"cached={self.num_cached_tokens}, blocks={len(self.block_table)})"
        )


class SeqViewRegistry:
    """Keeps one ``SeqView`` per request id, for as long as offload needs it."""

    def __init__(self) -> None:
        self._views: dict[str, SeqView] = {}

    def get_or_create(self, request: Any) -> SeqView:
        rid = request.request_id
        view = self._views.get(rid)
        # A reused id with a different underlying request must produce a NEW
        # view: that identity change is precisely what tells ATOM's scheduler
        # to drop the previous request's pending load.
        if view is None or view._request is not request:
            view = SeqView(request)
            self._views[rid] = view
        return view

    def get(self, request_id: str) -> SeqView | None:
        return self._views.get(request_id)

    def drop(self, request_id: str) -> None:
        self._views.pop(request_id, None)

    def __len__(self) -> int:
        return len(self._views)
