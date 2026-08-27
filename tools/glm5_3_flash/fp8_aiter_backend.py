"""aiter-backed replacement for the `finegrained-fp8` hub Triton kernel.

`kernels-community/finegrained-fp8` does not compile on gfx950 (LLVM `iota_range`
assert), which blocks every block-FP8 Linear and the MoE experts. ATOM already ships
exactly the kernel this checkpoint needs -- `aiter.gemm_a8w8_blockscale`, the
DeepSeek-style 128x128 block-FP8 GEMM -- so route through that instead.

Unlike the pure-torch fallback this also quantises the activation to FP8, which is
what the checkpoint was trained for (`activation_scheme: dynamic`), so it is both
much faster and closer to the intended numerics.

Install by calling `install()` before the first forward.
"""

from __future__ import annotations

import aiter
import torch
from aiter import QuantType, dtypes, get_hip_quant

_quant_per_1x128 = get_hip_quant(QuantType.per_1x128)


def _on_device(t: torch.Tensor):
    """Pin the CUDA context to the tensor's device for the duration of a call.

    accelerate's `device_map` hooks move tensors between GPUs but never set the
    current device, so with a multi-GPU device_map `torch.cuda.current_device()`
    stays 0 while the tensors live on cuda:1..3. Ordinary torch ops dispatch on the
    tensor's device, but aiter's kernels launch on the *current* device -- they then
    read and write the wrong GPU's memory and return garbage. (transformers warns
    about exactly this failure mode for DeepGEMM in its FP8 loader.)
    """
    return torch.cuda.device(t.device)


def _quant_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-1x128-block FP8 activation quant, row-major scales."""
    with _on_device(x):
        return _quant_per_1x128(x.contiguous(), quant_dtype=dtypes.fp8)


def _matmul(
    input,
    weight,
    weight_scale_inv,
    block_size=None,
    out_dtype=None,
    activation_scale=None,
):
    """2-D block-FP8 linear: [..., K] @ [N, K]^T -> [..., N]."""
    if activation_scale is not None:
        raise NotImplementedError(
            "static activation scales are not used by GLM-5.3-Flash"
        )
    out_dtype = out_dtype or input.dtype
    lead = input.shape[:-1]
    x = input.reshape(-1, input.shape[-1])
    xq, xs = _quant_act(x)
    with _on_device(x):
        y = aiter.gemm_a8w8_blockscale(
            xq, weight, xs, weight_scale_inv, dtype=torch.bfloat16
        )
    return y.reshape(*lead, y.shape[-1]).to(out_dtype)


def _batched_matmul(x, weight, weight_scale, block_size=None, expert_ids=None):
    """Per-token expert-indexed block-FP8 matmul: x [S, K], weight [E, N, K] -> [S, N].

    `expert_ids` may hold EP sentinels (>= E); the real kernel leaves those rows
    uninitialised and the caller masks them, so leaving them zero is fine.
    """
    num_experts, n, _ = weight.shape
    xq, xs = _quant_act(x)
    out = torch.zeros((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    valid = expert_ids < num_experts
    with _on_device(x):
        for e in torch.unique(expert_ids[valid]).tolist():
            rows = (expert_ids == e).nonzero(as_tuple=True)[0]
            out[rows] = aiter.gemm_a8w8_blockscale(
                xq[rows].contiguous(),
                weight[e],
                xs[rows].contiguous(),
                weight_scale[e],
                dtype=torch.bfloat16,
            )
    return out.to(x.dtype)


def _grouped_matmul(
    x, w, scale_inv, offsets=None, tokens_per_expert=None, block_size=None, **_
):
    """Grouped block-FP8 matmul over expert-sorted rows: x [S, K], w [E, N, K] -> [S, N].

    `offsets` is the int32 cumsum of `tokens_per_expert`; rows past `offsets[-1]` are
    EP sentinels the real kernel skips, so they stay zero.
    """
    n = w.shape[-2]
    xq, xs = _quant_act(x)
    out = torch.zeros((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    start = 0
    with _on_device(x):
        for e, end in enumerate(int(v) for v in offsets.tolist()):
            if end > start:
                out[start:end] = aiter.gemm_a8w8_blockscale(
                    xq[start:end].contiguous(),
                    w[e],
                    xs[start:end].contiguous(),
                    scale_inv[e],
                    dtype=torch.bfloat16,
                )
            start = end
    return out.to(x.dtype)


def _verifying(name, aiter_fn, torch_fn):
    """Wrap an entry point so it cross-checks aiter against the torch reference.

    Reports the first call whose output is non-finite or diverges, with shapes, then
    raises. Only for debugging -- it runs both implementations on every call.
    """
    import fp8_torch_fallback

    state = {"n": 0}

    def wrapped(*args, **kwargs):
        got = aiter_fn(*args, **kwargs)
        state["n"] += 1
        ref = getattr(fp8_torch_fallback, torch_fn)(*args, **kwargs)
        bad = not torch.isfinite(got).all()
        if not bad:
            g, r = got.float().flatten(), ref.float().flatten()
            cos = torch.nn.functional.cosine_similarity(g, r, dim=0).item()
            bad = cos < 0.99
        else:
            cos = float("nan")
        if bad:
            shapes = [tuple(a.shape) if torch.is_tensor(a) else a for a in args]
            print(
                f"[fp8-verify] FAIL in {name} call #{state['n']}: cos={cos:.5f} "
                f"finite={bool(torch.isfinite(got).all())} args={shapes} "
                f"kwargs={ {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in kwargs.items()} }",
                flush=True,
            )
            print(
                f"    devices: {[str(a.device) for a in args if torch.is_tensor(a)]} "
                f"current={torch.cuda.current_device()}",
                flush=True,
            )
            for i, a in enumerate(args):
                if torch.is_tensor(a):
                    af = a.float()
                    print(
                        f"    arg{i} {tuple(a.shape)} {a.dtype}: finite={bool(torch.isfinite(af).all())} "
                        f"absmax={af.abs().max().item():.6g} "
                        f"zero_rows={int((af.reshape(-1, af.shape[-1]).abs().sum(-1) == 0).sum())}",
                        flush=True,
                    )
            xq, xs = _quant_act(args[0].reshape(-1, args[0].shape[-1]))
            print(
                f"    quantised act: xq finite={bool(torch.isfinite(xq.float()).all())} "
                f"xs finite={bool(torch.isfinite(xs).all())} "
                f"xs_min={xs.min().item():.6g} xs_zeros={int((xs == 0).sum())}",
                flush=True,
            )
            nonfinite_rows = (~torch.isfinite(got)).any(-1).nonzero().flatten()[:8]
            print(f"    non-finite output rows: {nonfinite_rows.tolist()}", flush=True)
            raise SystemExit(3)
        return got

    return wrapped


def install(verify: bool = False) -> None:
    from transformers.integrations import finegrained_fp8 as ff

    matmul, batched, grouped = _matmul, _batched_matmul, _grouped_matmul
    if verify:
        matmul = _verifying("matmul", _matmul, "_matmul")
        batched = _verifying("batched_matmul", _batched_matmul, "_batched_matmul")
        grouped = _verifying("grouped_matmul", _grouped_matmul, "_grouped_matmul")

    ff._FINEGRAINED_FP8 = ff.FineGrainedFP8(
        matmul=matmul,
        batched_matmul=batched,
        grouped_matmul=grouped,
    )
    print(
        f"[fp8-aiter] installed aiter gemm_a8w8_blockscale bundle (verify={verify})",
        flush=True,
    )
