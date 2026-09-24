#!/usr/bin/env python3
"""Check that an installed LMCache wheel carries what ATOM's MP offload path needs.

LMCache's own validator (validate_rocm_torch210_wheel.py) covers the torch ABI
and treats the ATOM adapters as optional. ATOM cannot run without them, so this
check makes them mandatory and adds the native entry points and MP server flags
that atom/kv_transfer/offload/mp relies on.
"""

from __future__ import annotations

import argparse

import torch

# Docker builds and build-only runners have no GPU; LMCache's backend selection
# only needs the predicate to hold, no device is touched below.
torch.cuda.is_available = lambda: True

ATOM_SERVER_ARGS = ["--null-block-id", "-1", "--separate-object-groups"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()

    import lmcache
    import lmcache.cuda_ops
    import lmcache.lmcache_native
    from lmcache.integration.atom import (
        AtomMPSchedulerAdapter,
        AtomMPTransferSpec,
        AtomMPWorkerAdapter,
    )
    from lmcache.utils import EngineType
    from lmcache.v1.multiprocess.config import (
        add_mp_server_args,
        parse_args_to_mp_server_config,
    )
    from lmcache.v1.multiprocess.futures import DeviceMessagingFuture
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    assert lmcache.__version__ == args.expected_version, (
        lmcache.__version__,
        args.expected_version,
    )
    assert "site-packages" in lmcache.__file__, lmcache.__file__
    assert lmcache.cuda_ops.__file__.endswith(".so"), lmcache.cuda_ops.__file__
    assert lmcache.lmcache_native.__file__.endswith(
        ".so"
    ), lmcache.lmcache_native.__file__
    assert hasattr(
        lmcache.cuda_ops, "execute_object_group_transfer"
    ), "cuda_ops extension is incomplete"
    assert EngineType.ATOM.value == "atom"
    for cls in (AtomMPTransferSpec, AtomMPSchedulerAdapter, AtomMPWorkerAdapter):
        assert (
            cls.__module__ == "lmcache.integration.atom.multi_process_adapter"
        ), cls.__module__
    assert DeviceMessagingFuture.__module__ == "lmcache.v1.multiprocess.futures"
    assert EngineGroupInfo.__module__ == "lmcache.v1.multiprocess.group_view"

    # Parse the flags ATOM's README launches the MP server with through the
    # server's own parser; the `lmcache` CLI would also import every other
    # subcommand's optional dependencies.
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    mp_config = parse_args_to_mp_server_config(parser.parse_args(ATOM_SERVER_ARGS))
    assert mp_config.null_block_id == -1, mp_config.null_block_id
    assert mp_config.separate_object_groups, mp_config

    print(
        "OK: lmcache",
        lmcache.__version__,
        "has the ATOM MP adapters and server flags; torch",
        torch.__version__,
    )


if __name__ == "__main__":
    main()
