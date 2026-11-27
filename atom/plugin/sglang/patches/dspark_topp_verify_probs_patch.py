# SPDX-License-Identifier: MIT
"""TEMPORARY plugin patch -- remove this module, ``dspark_topp_pivot`` and
``tests/plugin/test_sglang_deepseek_v41_topp_verify.py`` once SGLang's ROCm
top-p renorm (``renorm_triton.top_p_renorm_probs_triton``) no longer sorts the
full vocabulary. The kernels are model-agnostic and belong upstream.

DeepSeek-V4.1 DSpark on the ATOM plugin, SGLang 0.5.22: the sampled verify's
top-p target distribution with PyTorch's exact top-p pivot, found without a
full-vocabulary sort and without a host synchronisation.

A DSpark round whose batch holds a sampled request accepts eagerly, outside
the verify CUDA graph (SGLang folds only all-greedy accepts into the graph).
With top-p and no top-k, ``dspark_accept._accept_sampling_core`` builds the
target distribution of every verify row with
``dflash_utils.build_speculative_verify_target_probs``: ``softmax(logits /
T)`` then ``top_p_renorm_probs_triton``, which ``torch.sort``s each of the
``bs * 6`` rows of V = 129280 entries to find the pivot: ~7.0 ms a round at
32 requests and ~12.0 ms at 64 on MI308X. This patch replaces that one branch:
the same softmax, each row's pivot from ``dspark_topp_pivot.exact_top_p_pivots``
(bitwise the pivot SGLang's sort / cumsum / searchsorted finds, ties and flat
rows included), and SGLang's own ``renorm_triton._renorm_from_pivots``, so the
accept sees SGLang's target probabilities bit for bit. ~1.4 ms at 32 requests;
rows the top-1280 head does not settle take a ~5 ms device-side fallback per
wave, decided on the device.

Guard: every ``GUARD_WINDOW`` calls the device counts are copied to the host
without waiting. When, over the newest completed window, the fallback rows --
each costed as a whole ``GUARD_FALLBACK_WAVE_MS`` wave, an upper bound -- cost
more than the exact calls saved over SGLang's sort (``GUARD_SAVING_MS``, an
MI308X table by row count), the next ``GUARD_COOLDOWN`` calls take SGLang's
function. Both paths give the same probabilities; the guard only picks the
cheaper one.

Scope: installed by the DeepSeek-V4.1 DSpark draft wrapper, and only with the
sampled draft (``SGLANG_DSPARK_USE_GREEDY_DRAFT`` unset: a greedy draft's
top-p-only verify takes ``dspark_draw_verify_patch``'s draw rule instead);
only the dense top-p-only branch (``need_top_p_sampling`` and not
``need_top_k_sampling``) with CUDA ``[bs * draft_token_num, V]`` logits,
2..512 rows and one top-p per request, never under graph capture. SGLang
encodes a greedy request as top_k = 1, so a batch holding any greedy or top-k
request takes SGLang's path whole.

Switches (`deepseek_v41_envs`, read at install):
``ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY=0`` keeps SGLang's function, which every
call outside the branch runs anyway;
``ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY_GUARD=0`` keeps the exact path whatever
the fallback costs.

Install guards, each keeping SGLang's function with one WARNING line (the
plugin runs on SGLang 0.5.22 only: `sglang_version_guard` refuses any other at
launch, before this installs): the sha256 of the whitespace-normalised source
of every SGLang function the patch replaces or relies on
(``PINNED_SOURCE_SHA256``), so any upstream edit refuses the install until the
replay is re-validated; the PyTorch build (``PINNED_TORCH_BUILDS``) whose HIP
cumsum / searchsorted arithmetic is replayed; the Triton top-p renorm in use;
then a one-time GPU self-check comparing the patched function bitwise with
SGLang's at 6, 192 and 258 rows (both of torch's scan chunks; see
``_self_check_table``).
"""

from __future__ import annotations

import inspect
import itertools
import logging
from functools import wraps

from atom.plugin.sglang import deepseek_v41_envs
from atom.plugin.sglang.patches.sglang_version_guard import sglang_version

logger = logging.getLogger("atom")

# The PyTorch builds (torch.__version__, torch.version.git_version) whose HIP
# torch.cumsum (ATen ScanUtils.cuh) and searchsorted arithmetic
# dspark_topp_pivot reproduces and the GPU tests verified.
PINNED_TORCH_BUILDS = (
    ("2.10.0+rocm7.2.4.git3d3aa833", "3d3aa833db84eed6b7f5595cb5f162c2f78300a4"),
)
# sha256 of the whitespace-normalised ``inspect.getsource`` of every SGLang
# function this patch replaces or relies on (SGLang 0.5.22, release/v0.5.22
# baa9befa38). A mismatch keeps SGLang's function. Update a pin only after
# re-running tests/plugin/test_sglang_deepseek_v41_topp_verify.py on the new
# source.
PINNED_SOURCE_SHA256 = {
    "dspark_accept._accept_sampling_core": (
        "7a84153e54ac4d08598d4b7a682d77bf73ff923efb94ccbfeb07f9e130df04c4"
    ),
    "dflash_utils.build_speculative_verify_target_probs": (
        "06dcc2c890891e685e1a174190bde51028c833de8134f5b0dcf9e37fe39fdf0a"
    ),
    "dflash_utils._dflash_top_p_renorm_prob": (
        "cbcc1d33384960348133747d8fb230c301902d6cb8fc854d612c4361e1d5a698"
    ),
    "renorm_triton.top_p_renorm_probs_triton": (
        "8151c3b20765f2263f7271d2fe5d7272a51f32d4ce7fc6bd0459b02e6a290311"
    ),
    "renorm_triton._prepare_probs": (
        "6bea7b789f8a7157260acf1c34916d8691ca7b0deade808436713f0001bd075e"
    ),
}
_MARK = "_atom_v41_exact_topp"

GUARD_WINDOW = 64
GUARD_COOLDOWN = 1024
# What a call saves over SGLang's sort (ms, by verify rows) and what one fallback
# wave costs, measured on MI308X only (V = 129280). Only the guard's choice
# between two bitwise-identical paths uses them; on other GPUs the choice may be
# less than optimal, never wrong.
GUARD_SAVING_MS = ((6, 0.96), (48, 1.36), (192, 5.59), (384, 9.49))
GUARD_FALLBACK_WAVE_MS = 5.3


def saving_ms(rows: int) -> float:
    """Estimated ms one exact call saves over SGLang's sort at ``rows`` verify
    rows (piecewise linear through ``GUARD_SAVING_MS``, extended linearly)."""
    pts = GUARD_SAVING_MS
    if rows <= pts[0][0]:
        return pts[0][1] * rows / pts[0][0]
    for (r0, s0), (r1, s1) in itertools.pairwise(pts):
        if rows <= r1:
            return s0 + (s1 - s0) * (rows - r0) / (r1 - r0)
    (r0, s0), (r1, s1) = pts[-2], pts[-1]
    return s1 + (s1 - s0) * (rows - r1) / (r1 - r0)


def source_fingerprint(src: str) -> str:
    """sha256 of ``src`` with every whitespace run collapsed to one space
    (re-indentation keeps it; any token change, insertion or removal does not)."""
    import hashlib

    return hashlib.sha256(" ".join(src.split()).encode()).hexdigest()


def _pinned_functions(dspark_accept, dflash_utils, renorm_triton) -> dict:
    return {
        "dspark_accept._accept_sampling_core": dspark_accept._accept_sampling_core,
        "dflash_utils.build_speculative_verify_target_probs": (
            dflash_utils.build_speculative_verify_target_probs
        ),
        "dflash_utils._dflash_top_p_renorm_prob": (
            dflash_utils._dflash_top_p_renorm_prob
        ),
        "renorm_triton.top_p_renorm_probs_triton": (
            renorm_triton.top_p_renorm_probs_triton
        ),
        "renorm_triton._prepare_probs": renorm_triton._prepare_probs,
    }


def _upstream_shape_error(dspark_accept, dflash_utils, renorm_triton) -> str | None:
    import torch

    build_id = (str(torch.__version__), str(torch.version.git_version))
    if build_id not in PINNED_TORCH_BUILDS:
        return (
            f"torch {build_id[0]} (git {build_id[1]}) is not a pinned build "
            f"{PINNED_TORCH_BUILDS}: its cumsum / searchsorted arithmetic is not verified"
        )
    for mod, attrs in (
        (
            dspark_accept,
            ("_accept_sampling_core", "build_speculative_verify_target_probs"),
        ),
        (
            dflash_utils,
            ("build_speculative_verify_target_probs", "_dflash_top_p_renorm_prob"),
        ),
        (
            renorm_triton,
            ("top_p_renorm_probs_triton", "_renorm_from_pivots", "_prepare_probs"),
        ),
    ):
        lacking = [a for a in attrs if not hasattr(mod, a)]
        if lacking:
            return f"{mod.__name__} lacks {lacking}"
    # On CUDA SGLang binds flashinfer's renorm here, on MUSA its own.
    if (
        getattr(dflash_utils, "top_p_renorm_prob", None)
        is not renorm_triton.top_p_renorm_probs_triton
    ):
        return "the top-p renorm in use is not renorm_triton.top_p_renorm_probs_triton"
    build = inspect.unwrap(dspark_accept.build_speculative_verify_target_probs)
    if build is not inspect.unwrap(dflash_utils.build_speculative_verify_target_probs):
        return (
            "dspark_accept.build_speculative_verify_target_probs is not dflash_utils'"
        )
    changed = []
    for name, fn in _pinned_functions(
        dspark_accept, dflash_utils, renorm_triton
    ).items():
        try:
            got = source_fingerprint(inspect.getsource(inspect.unwrap(fn)))
        except (OSError, TypeError) as e:
            got = f"unreadable ({e})"
        want = PINNED_SOURCE_SHA256[name]
        if got != want:
            changed.append(f"{name} sha256 {got[:16]} != pinned {want[:16]}")
    if changed:
        return (
            "upstream source fingerprint mismatch ("
            + "; ".join(changed)
            + "): re-validate the exact replay on the new source before updating "
            "PINNED_SOURCE_SHA256"
        )
    return None


def takes_exact_topp(next_token_logits, sampling_info, draft_token_num, bs) -> bool:
    """Whether this call is SGLang's dense top-p-only branch on a table the
    exact pivots cover (see the module docstring)."""
    from atom.plugin.sglang.patches.dspark_topp_pivot import supports

    if sampling_info.need_top_k_sampling or not sampling_info.need_top_p_sampling:
        return False
    x = next_token_logits
    if x.device.type != "cuda" or x.dim() != 2:
        return False
    rows, vocab = int(x.shape[0]), int(x.shape[1])
    if rows != int(bs) * int(draft_token_num) or not supports(rows, vocab):
        return False
    top_ps = sampling_info.top_ps
    return int(top_ps.numel()) == int(bs) and top_ps.device == x.device


class _Guard:
    """Picks SGLang's function for ``GUARD_COOLDOWN`` calls when, over a window
    of at least ``GUARD_WINDOW`` exact calls, the fallback rows (each costed as
    one ``GUARD_FALLBACK_WAVE_MS`` wave -- an upper bound, rows of one call share
    a wave) cost more than the exact calls saved over SGLang's sort
    (``saving_ms`` of their row counts). Reads only completed non-blocking
    copies of the device counts. Both paths give the same probabilities."""

    def __init__(self, counters, enabled: bool):
        self.counters = counters
        self.enabled = enabled
        self.calls = 0  # every call of the patched branch
        self.off_until = 0
        self.epoch = 0
        self.prev = (
            None  # (epoch, exact calls, budget ms, fallback rows) at the window start
        )

    def use_exact(self) -> bool:
        self.calls += 1
        if not self.enabled:
            return True
        if self.calls <= self.off_until:
            return False
        if self.counters.calls % GUARD_WINDOW == 0:
            snap = self.counters.poll()
            if snap is not None:
                calls, budget, counts = snap
                prev = self.prev
                if prev is None or prev[0] != self.epoch or calls < prev[1]:
                    self.prev = (self.epoch, calls, budget, counts[2])
                elif calls - prev[1] >= GUARD_WINDOW:
                    rows = counts[2] - prev[3]
                    cost = rows * GUARD_FALLBACK_WAVE_MS
                    saved = budget - prev[2]
                    if cost > saved:
                        self.off_until = (
                            self.calls + GUARD_COOLDOWN - 1
                        )  # this call too
                        self.epoch += 1
                        self.prev = None
                        logger.info(
                            "ATOM DeepSeek-V4.1 DSpark top-p verify: %d fallback "
                            "rows in %d calls (<= %.0f ms) against %.0f ms saved; "
                            "SGLang's sort for the next %d calls (same "
                            "probabilities)",
                            rows,
                            calls - prev[1],
                            cost,
                            saved,
                            GUARD_COOLDOWN,
                        )
                        return False
                    self.prev = (self.epoch, calls, budget, counts[2])
        return True


def make_exact_topp_build(original, renorm, counters, guard):
    """``build_speculative_verify_target_probs`` whose dense top-p-only branch
    takes the exact pivots; every other call goes to ``original``."""
    import torch
    import torch.nn.functional as F

    from atom.plugin.sglang.patches.dspark_topp_pivot import exact_top_p_pivots

    @wraps(original)
    def build_speculative_verify_target_probs(
        *,
        next_token_logits,
        sampling_info,
        draft_token_num,
        bs,
        max_top_k=None,
        uniform_top_k_value=None,
        use_sparse_topk=True,
    ):
        # SGLang accepts sampled batches eagerly; were this ever captured into a
        # graph, the guard's host-side copies must not be.
        in_scope = not torch.cuda.is_current_stream_capturing() and takes_exact_topp(
            next_token_logits, sampling_info, draft_token_num, bs
        )
        if not (in_scope and guard.use_exact()):
            return original(
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=draft_token_num,
                bs=bs,
                max_top_k=max_top_k,
                uniform_top_k_value=uniform_top_k_value,
                use_sparse_topk=use_sparse_topk,
            )
        # SGLang's dense branch, step for step, up to the pivot.
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, draft_token_num, dim=0
        )
        scaled_logits = next_token_logits / expanded_temperature
        target_probs = F.softmax(scaled_logits, dim=-1)
        probs_fp32 = target_probs.float().contiguous()
        top_ps = sampling_info.top_ps.reshape(-1)
        if top_ps.dtype != torch.float32:
            top_ps = top_ps.to(torch.float32)
        pivots = exact_top_p_pivots(
            probs_fp32, top_ps.contiguous(), int(draft_token_num), counters
        )
        counters.budget_ms += saving_ms(int(probs_fp32.shape[0]))
        out = renorm(probs_fp32, pivots)
        return out.view(bs, draft_token_num, -1).contiguous()

    setattr(build_speculative_verify_target_probs, _MARK, True)
    return build_speculative_verify_target_probs


# Install self-check batches: 6 and 192 rows (torch's scan chunk 1024; 192 is
# the c32 operating point) and 258 rows (chunk 512).
SELF_CHECK_BATCHES = (1, 32, 43)
_SELF_CHECK_KINDS = 8


def _self_check_table(bs: int, vocab: int, g):
    """Logits, temperatures and per-request top_p of one self-check batch.

    Requests cycle through 8 kinds (``b % 8``):

    * 0: LM-like rows (a few strong tokens) at top_p 0.95 -- settled by the head;
    * 1: flat rows at 0.95 -- the cut lies far below the head, so the fallback
      sorts the row and replays torch's whole chunked cumsum;
    * 2 / 6 / 7: flat / heavy-tailed (zipf) / LM-like rows whose threshold is
      exactly a torch cdf value among the top 40 positions of the request's
      first row, or one ulp either side -- decided inside the head with a large
      (flat, zipf) or small (LM) running total, the rounding bracket's case;
    * 3: top_p = 1 (the row minimum);
    * 4 / 5: flat rows whose threshold is exactly a torch cdf value below the
      head / one ulp from one -- the fallback's replay must hit it bit for bit.

    Thresholds are taken where the cdf is >= 0.5, where ``1 - top_p`` gives
    them back exactly in fp32. A single request (bs = 1) mixes LM-like and flat
    rows under a below-head cdf threshold."""
    import torch

    rows = 6 * bs
    dev = "cuda"
    lm = torch.randn(rows, vocab, device=dev, generator=g) * 2.0
    lm[:, :40] += torch.linspace(12.0, 4.0, 40, device=dev)
    flat = torch.randn(rows, vocab, device=dev, generator=g) * 0.3
    ranks = torch.arange(1, vocab + 1, device=dev, dtype=torch.float32)
    expo = 0.6 + 0.8 * torch.rand(rows, 1, device=dev, generator=g)
    zipf = -expo * torch.log(ranks)[None, :]
    req = torch.arange(rows, device=dev) // 6
    kind = req % _SELF_CHECK_KINDS
    if bs == 1:
        kind = torch.full_like(kind, 4)
        is_flat = torch.arange(rows, device=dev) % 2 == 1
        is_zipf = torch.zeros_like(is_flat)
    else:
        is_flat = (kind == 1) | (kind == 2) | (kind == 4) | (kind == 5)
        is_zipf = kind == 6
    z = torch.where(is_flat[:, None], flat, torch.where(is_zipf[:, None], zipf, lm))
    del lm, flat, zipf
    temps = torch.full((bs, 1), 0.6, device=dev)
    # torch's own cdf at this row count (its scan chunk depends on it), of the
    # row of each request whose threshold is placed on it
    first = 1 if bs == 1 else 0
    cdf = torch.cumsum(torch.sort(torch.softmax(z / 0.6, -1), -1).values, -1)
    cdf = cdf[first::6].cpu()
    head = vocab - 1280
    kinds = kind[::6].tolist()
    thr = []
    for b in range(bs):
        k = kinds[b]
        if k in (0, 1):
            thr.append(None)
            continue
        if k == 3:
            thr.append(0.0)
            continue
        c = cdf[b]
        p50 = int(torch.searchsorted(c, torch.tensor([0.5])).item())
        if k in (4, 5):
            lo, hi = p50, head - 1
        else:
            lo, hi = max(p50, vocab - 40), vocab - 2
        lo = min(lo, hi)
        t = c[lo + (7919 * (b + 1)) % (hi - lo + 1)].reshape(1)
        # 0: on the cdf value; 1 / 2: one ulp above / below it
        if k == 4:
            step = 0
        elif k == 5:
            step = 1 + (b // _SELF_CHECK_KINDS) % 2
        else:
            step = (b // _SELF_CHECK_KINDS) % 3
        if step == 1:
            t = torch.nextafter(t, torch.tensor([2.0]))
        elif step == 2:
            t = torch.nextafter(t, torch.tensor([0.0]))
        thr.append(float(t.item()))
    one = torch.ones(1, dtype=torch.float32)
    top_ps = torch.cat(
        [torch.tensor([0.95]) if t is None else one - torch.tensor([t]) for t in thr]
    ).float()
    return z, temps, top_ps.to(dev)


def _self_check(patched, original) -> str | None:
    """Bitwise comparison of ``patched`` and ``original`` on the GPU at every
    batch of ``SELF_CHECK_BATCHES`` (see ``_self_check_table``). Returns a
    description of the mismatches or None."""
    from types import SimpleNamespace

    import torch

    vocab = 129280
    g = torch.Generator(device="cuda")
    g.manual_seed(20261119)
    bad = []
    for bs in SELF_CHECK_BATCHES:
        rows = 6 * bs
        z, temps, top_ps = _self_check_table(bs, vocab, g)
        info = SimpleNamespace(
            need_top_k_sampling=False,
            need_top_p_sampling=True,
            temperatures=temps,
            top_ps=top_ps,
            top_ks=torch.full((bs,), vocab, device="cuda", dtype=torch.int32),
        )
        kw = {
            "next_token_logits": z,
            "sampling_info": info,
            "draft_token_num": 6,
            "bs": bs,
        }
        a = original(**kw)
        b_ = patched(**kw)
        if a.shape != b_.shape:
            bad.append(f"shape {tuple(b_.shape)} != {tuple(a.shape)} at bs={bs}")
        elif not torch.equal(a.view(torch.int32), b_.view(torch.int32)):
            n = (a.view(torch.int32) != b_.view(torch.int32)).reshape(rows, -1)
            bad.append(f"{int(n.any(-1).sum())} of {rows} rows differ at bs={bs}")
        del z, a, b_
    if bad:
        return "; ".join(bad) + " (patched vs SGLang's original, bitwise)"
    return None


def install_dspark_topp_verify_probs_patch() -> bool:
    """Install (idempotent) when the DSpark draft samples. Returns whether the
    exact top-p branch is active."""
    from sglang.srt.environ import envs as sglang_envs

    if sglang_envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get():
        # A greedy draft's top-p-only verify takes the draw rule
        # (`dspark_draw_verify_patch`); nothing else reaches the exact branch
        # often enough to carry this patch and its GPU self-check.
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark: top-p verify patch not installed "
            "(SGLANG_DSPARK_USE_GREEDY_DRAFT=1: the draw rule verifies)"
        )
        return False
    if not deepseek_v41_envs.ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY:
        logger.info(
            "ATOM DeepSeek-V4.1 DSpark: sampled verify keeps SGLang's top-p sort "
            "(ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY=0)"
        )
        return False
    from sglang.kernels.ops.sampling import renorm_triton
    from sglang.kernels.ops.speculative.dspark import dspark_accept
    from sglang.srt.speculative import dflash_utils

    if getattr(dspark_accept.build_speculative_verify_target_probs, _MARK, False):
        return True
    reason = _upstream_shape_error(dspark_accept, dflash_utils, renorm_triton)
    if reason is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1 DSpark: top-p verify patch NOT installed; the sampled "
            "verify keeps SGLang's top-p sort (%s)",
            reason,
        )
        return False

    import torch

    from atom.plugin.sglang.patches.dspark_topp_pivot import ToppPivotCounters

    original = dspark_accept.build_speculative_verify_target_probs
    counters = ToppPivotCounters(torch.device("cuda", torch.cuda.current_device()))
    guard = _Guard(counters, enabled=False)
    patched = make_exact_topp_build(
        original, renorm_triton._renorm_from_pivots, counters, guard
    )
    try:
        mismatch = _self_check(patched, original)
    except Exception as e:  # noqa: BLE001 - any failure leaves equality unproven
        mismatch = f"self-check raised {type(e).__name__}: {e}"
    finally:
        torch.cuda.empty_cache()
    if mismatch is not None:
        logger.warning(
            "ATOM DeepSeek-V4.1 DSpark: top-p verify patch NOT installed; the sampled "
            "verify keeps SGLang's top-p sort (install self-check: %s)",
            mismatch,
        )
        return False
    counters.dev.zero_()
    counters.calls = 0
    guard.calls = 0
    guard.enabled = deepseek_v41_envs.ATOM_SGLANG_V41_DSPARK_TOPP_VERIFY_GUARD
    dspark_accept.build_speculative_verify_target_probs = patched
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark: the sampled verify's top-p target probs take "
        "PyTorch's exact pivot without the vocabulary sort (sglang %s; self-check "
        "bitwise equal; guard %s; temporary patch)",
        sglang_version(),
        "on" if guard.enabled else "off",
    )
    return True
