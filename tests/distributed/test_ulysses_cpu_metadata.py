# SPDX-License-Identifier: MIT
"""Forward-context metadata must work without the GPU communication runtime."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_forward_context_without_aiter_or_triton():
    # A fresh interpreter also checks this on GPU developer machines, where a
    # previously imported AITER module would otherwise hide the CI regression.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import sys
                from types import SimpleNamespace

                import torch

                sys.modules["aiter"] = None
                sys.modules["triton"] = None
                torch.cuda.is_available = lambda: False

                from atom.distributed import ulysses_sp as sp
                from atom.utils.forward_context import (
                    get_forward_context,
                    set_forward_context,
                )

                config = SimpleNamespace(
                    parallel_config=SimpleNamespace(data_parallel_size=1),
                    compilation_config=SimpleNamespace(static_forward_context={}),
                )
                context = SimpleNamespace()
                sp.set_sp_world_size(4)
                set_forward_context(None, config, context, num_tokens=17)
                assert get_forward_context().context is context
                assert context.running_tokens_across_dp == (5, 5, 5, 5)

                set_forward_context(None, config, context, num_tokens=8)
                assert context.running_tokens_across_dp == (2, 2, 2, 2)
                set_forward_context(None, config, context)
                assert context.running_tokens_across_dp is None

                sp.set_sp_world_size(1)
                counts = torch.tensor([3, 7], dtype=torch.int32)
                set_forward_context(None, config, context, num_tokens_across_dp=counts)
                assert context.running_tokens_across_dp == (3, 7)
                set_forward_context(None, config, context, num_tokens=17)
                assert context.running_tokens_across_dp is None
                assert sp.get_sp_rank() == 0
                assert sys.modules["aiter"] is None
                assert sys.modules["triton"] is None
                """),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
