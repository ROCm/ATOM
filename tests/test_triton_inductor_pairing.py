# SPDX-License-Identifier: MIT
"""A hand-written Triton kernel has to still compute once Inductor owns it.

`--level` defaults to 3, so every forward goes through Inductor, and the
backends that carry this repo -- DeepSeek-V4/V4.1, GDN, the blockscale GEMMs --
reach hand-written Triton kernels from inside that graph. torch calls them
through `triton_kernel_wrapper_mutation`, which needs a specialization helper
that Triton has renamed twice (`specialize_impl` -> `create_specialize_impl`).
When the installed pair offers neither name, torch logs an ImportError at
WARNING level and the kernel is skipped: the output buffer keeps whatever the
allocator last put there. Nothing raises. The server starts, `/v1/models`
answers, and the model emits garbage -- observed on V4.1 at the default level,
where `The capital of France is` completes to `NamedCS","####</think>`.

The check is a kernel of its own rather than a model, because the property
belongs to the torch/Triton pair and not to any backend: whatever fails here
fails for every model that writes its own kernels.

Both arms are load-bearing. Eager pins the kernel, so a kernel that is simply
wrong cannot make the compiled arm agree with it and pass.
"""

import pytest
import torch

if not torch.cuda.is_available():
    # Above the Triton import, not below it: a raise during collection ends the
    # whole session rather than skipping one file.
    pytest.skip("compiles and runs a Triton kernel", allow_module_level=True)

import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _double_kernel(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask) * 2.0, mask=mask)


def _doubled(x):
    # Allocated inside the traced region, as the real backends do: a dropped
    # kernel then leaves the allocator's previous contents, which is what makes
    # the failure silent instead of an exception or a NaN.
    out = torch.empty_like(x)
    _double_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, out, x.numel(), BLOCK=BLOCK)
    return out


def _specialization_hook() -> str:
    """Which name torch will find on this Triton, for the failure message."""
    jit = triton.runtime.jit
    found = [
        name
        for name in ("create_specialize_impl", "specialize_impl")
        if hasattr(jit, name)
    ]
    return found[0] if found else "NEITHER"


def test_a_triton_kernel_survives_inductor():
    torch.manual_seed(0)
    x = torch.randn(4096, device="cuda")
    expected = x * 2

    eager = _doubled(x)
    assert torch.equal(eager, expected), (
        "the kernel is wrong before Inductor is involved, so this file cannot "
        "say anything about the torch/Triton pair"
    )

    compiled = torch.compile(_doubled, backend="inductor")(x)
    assert torch.equal(compiled, expected), (
        f"Inductor dropped a hand-written Triton kernel: "
        f"max|err|={(compiled - expected).abs().max().item():.3e}, "
        f"torch={torch.__version__}, triton={triton.__version__}, "
        f"torch's specialization hook resolves to {_specialization_hook()}. "
        "Every model with its own kernels is affected; until the pair lines "
        "up they have to run with --level 0."
    )
