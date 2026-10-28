"""Preload the native router's finite row domain without running a kernel."""


def preload_router():
    import hashlib

    import torch

    from atom.model_ops.deepseek_v41.router import MAX_TOKENS, _build
    from atom.mono.runtime.compile import compile_only

    # Use the native factory, ABI and stream conversion. These are the native
    # supported widths, not alternative tile choices or a timing sweep.
    rows = []
    with compile_only():
        for width in range(1, MAX_TOKENS + 1):
            launch = _build(width)
            launch(0, 0, 0, stream=torch.cuda.current_stream())
            keys = [
                hashlib.sha256(repr(key).encode()).hexdigest()
                for key in launch._mem_cache
            ]
            if not keys:
                raise RuntimeError("Native router preload did not populate memory")
            rows.append({"width": width, "full_key_sha256": sorted(keys)})
    return rows
