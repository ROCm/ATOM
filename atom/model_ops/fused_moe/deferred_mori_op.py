# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Leave the MoRI EP dispatch/combine op handles unbuilt (a recipe's
``engine.mori_defer_handles``).

The standard MoE backend builds one MoRI op handle per argument set at model
build, whether or not a forward ever dispatches through it. A server whose
every FFN is served whole by a plugin provider (seam `v41.block.ffn`) never
does, but each handle still brings its own HIP stream and event thread, which
share the device's hardware queues with the streams the decode graphs replay
on.

With the option on, each handle is replaced by a stand-in that raises at first
use. It is not built late instead: MoRI's symmetric heap wants every EP rank
to allocate in the same order, and a rank-local first native call cannot
promise that. So the option is only for servers whose plan never declines to
the native path; a decline then fails the request loudly rather than run on a
handle built out of order.
"""

from atom import plugin_host


def defer_mori_handles() -> bool:
    return bool(plugin_host.engine_options().mori_defer_handles)


class DeferredMoriOp:
    """Stand-in for a MoRI op handle that was not built; any use raises."""

    def __init__(self, slot: int):
        self._slot = slot

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"MoRI EP op handle (slot {self._slot}) was not built "
            f"(engine.mori_defer_handles) but the native MoE path asked for "
            f"`{name}`: a forward was not served by the plugin provider. Drop "
            "engine.mori_defer_handles to build the handles at model build."
        )
