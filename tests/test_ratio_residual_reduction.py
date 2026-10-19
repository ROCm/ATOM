"""The residual draw must never emit a token with zero residual mass.

Reported on #2582: the two-pass residual walk tested for a hit with
`cum_sum + tl.cumsum(val)` but advanced `cum_sum` with `tl.sum(val)`. Those
lower to different reduction trees and disagree by about an ulp, so a
`target_u` landing in the gap makes the tile miss, leaves `cum_sum` already
above `target_u`, and lets the next tile's element 0 satisfy the test before
its own mass is read -- returning a multiple of BLOCK_V picked by position. In
the final tile `found` stays 0 and the kernel returns `VOCAB_SIZE - 1`.

Rate is about fp32 eps (~1.2e-7) per rejection, which at ~8.8M rejections in a
full agentic run is roughly one bad token per run. It is rare, but the token
carries zero probability under the residual -- one the target could never have
emitted -- so it breaks the exactness the whole feature rests on.

The first test needs no GPU: it pins the *arithmetic* premise, which is what
makes the walk unsound, and fails on any machine if the premise stops holding.
The second runs the real kernel.
"""

import numpy as np
import pytest


def _scan_last_vs_tree_sum(vec: np.ndarray) -> tuple[float, float]:
    """`tl.cumsum(...)[-1]` and `tl.sum(...)` as numpy models them.

    `np.cumsum` is a sequential scan; `np.add.reduce` is pairwise. That is the
    same shape of disagreement Triton's scan and tree reduction have.
    """
    scan_last = float(np.cumsum(vec, dtype=np.float32)[-1])
    tree_sum = float(np.add.reduce(vec, dtype=np.float32))
    return scan_last, tree_sum


def test_scan_and_tree_sum_disagree_on_residual_shaped_input():
    """The premise: the two reductions are NOT interchangeable in fp32.

    If this ever stops failing, the bug this file guards is unreachable and the
    kernel's reduction choice stops mattering -- so assert the disagreement
    rather than assuming it.
    """
    rng = np.random.default_rng(0)
    disagree = 0
    scan_below = 0
    trials = 2000
    for _ in range(trials):
        vec = (rng.random(4096) * rng.choice([1e-3, 1e-6, 1.0])).astype(np.float32)
        # relu(p - q) is sparse: most of the vocab has no residual mass.
        vec[rng.random(4096) < 0.7] = 0.0
        scan_last, tree_sum = _scan_last_vs_tree_sum(vec)
        if scan_last != tree_sum:
            disagree += 1
        if scan_last < tree_sum:
            scan_below += 1
    # Measured ~93% disagree, ~46% with the scan below the tree sum (the
    # direction that opens the miss window). Assert far inside that.
    assert disagree > trials * 0.5, f"only {disagree}/{trials} disagreed"
    assert scan_below > trials * 0.2, f"only {scan_below}/{trials} scanned low"


def test_mixed_reduction_walk_can_select_a_zero_mass_index():
    """Model both walks on CPU and show only the mixed one can pick zero mass.

    This is the bug reduced to arithmetic: no Triton, no GPU. The fixed walk
    advances with the same scan it tested with, so a tile that misses leaves
    `cum_sum` exactly at its own last cumulative value and the next tile cannot
    be entered already above `target_u`.
    """
    rng = np.random.default_rng(7)
    block = 4096

    def walk(tiles, target_u, *, mixed):
        cum = np.float32(0.0)
        for start, vec in enumerate(tiles):
            cs = np.cumsum(vec, dtype=np.float32)
            hit = (cum + cs) > target_u
            if hit.any():
                return start * block + int(np.argmax(hit))
            cum = (
                np.float32(cum + np.add.reduce(vec, dtype=np.float32))
                if mixed
                else np.float32(cum + cs[-1])
            )
        return len(tiles) * block - 1  # the VOCAB_SIZE - 1 fallback

    found_zero_mass = False
    for _ in range(4000):
        a = (rng.random(block) * 1e-6).astype(np.float32)
        a[rng.random(block) < 0.5] = 0.0
        b = np.zeros(block, dtype=np.float32)
        b[123] = np.float32(1e-3)  # tile 1 holds mass only at one index
        tiles = [a, b]
        scan_last, tree_sum = _scan_last_vs_tree_sum(a)
        if not scan_last < tree_sum:
            continue
        # Drive target_u into the gap the two reductions leave behind.
        target_u = np.float32((scan_last + tree_sum) / 2.0)
        if not (scan_last < target_u < tree_sum):
            continue
        tok = walk(tiles, target_u, mixed=True)
        if tok == block and b[0] == 0.0:
            found_zero_mass = True
            # The consistent walk must pick a token that actually has mass.
            fixed = walk(tiles, target_u, mixed=False)
            assert fixed != block or b[0] > 0.0, (
                "consistent walk also selected a zero-mass index"
            )
            flat = np.concatenate(tiles)
            assert flat[fixed] > 0.0, f"fixed walk picked zero-mass id {fixed}"
            break

    assert found_zero_mass, (
        "could not construct the gap on this machine; the CPU model did not "
        "reproduce the premise, so the GPU test below is the only coverage"
    )


# The kernel itself needs triton and a real GPU. Guard the import the way the
# rest of the suite does -- a bare import here is a collection ERROR on the
# non-GPU CI job, which fails the whole run rather than skipping.
pytest.importorskip("triton")
torch = pytest.importorskip("torch")


@pytest.mark.skipif(
    not getattr(torch, "cuda", None) or not torch.cuda.is_available(),
    reason="residual kernel needs a GPU",
)
def test_residual_draw_never_returns_zero_mass_token():
    """Volume check on the real kernel: every rejected position has mass.

    Goes through the public `rejection_sample`, not the kernel, so the ragged
    layout and the q plumbing are the production ones. The vocabulary is wide
    on purpose -- the bug is a TILE-boundary effect, so a vocab inside one
    `_RATIO_BLOCK_V` tile cannot express it.

    A rejected position is the one at index `accepted` for each request, and on
    rejection the emitted token is drawn from normalised `relu(p - q)`. Any
    emitted id whose residual mass is zero is a token the target could never
    have produced. Without the reduction fix those ids are a multiple of
    `_RATIO_BLOCK_V`, or `vocab - 1` when the miss lands in the final tile.
    """
    from atom.model_ops import rejection_sampler as rs
    from atom.model_ops.rejection_sampler import rejection_sample

    # The sibling module owns the batch builders; replicating them here would
    # be a second source of truth for the ragged layout.
    helpers = pytest.importorskip("test_speculative_sampling")
    two_token_batch, draw = helpers.two_token_batch, helpers.draw

    torch.manual_seed(20261019)
    block = rs._RATIO_BLOCK_V  # noqa: SLF001
    vocab = block * 40  # 40 tiles, so the walk crosses many boundaries
    n = 500
    width = 2

    # p and q deliberately disagree, so most drafts are rejected and most
    # positions exercise the residual draw rather than the accept path.
    p = torch.rand(vocab, device="cuda", dtype=torch.float32) ** 3
    p /= p.sum()
    q = torch.rand(vocab, device="cuda", dtype=torch.float32) ** 3
    q /= q.sum()

    drafts = draw(q, n * width)
    output, (accepted, _, _) = rejection_sample(
        **two_token_batch(p, drafts, width),
        draft_probs=q.expand(n * width, -1).contiguous(),
        temperatures=torch.ones(n, device="cuda"),
    )

    residual = torch.clamp(p - q, min=0.0)
    rejected = accepted < width
    assert rejected.any(), "no rejections: p and q were too close to test"

    idx = accepted[rejected].long()
    emitted = output[rejected, idx]
    mass = residual[emitted.long()]
    bad = (mass <= 0).nonzero().view(-1)
    assert bad.numel() == 0, (
        f"{bad.numel()}/{int(rejected.sum())} residual draws carried zero mass; "
        f"first bad ids={emitted[bad[:8]].tolist()} "
        f"(a multiple of {block}, or {vocab - 1}, points at the "
        f"mixed-reduction walk)"
    )
