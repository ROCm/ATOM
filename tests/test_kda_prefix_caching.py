# SPDX-License-Identifier: MIT
"""Standalone repro for the Kimi-K3 KDA prefix-caching accuracy bug.

Requires a GPU (ROCm). Run with:

    python -m pytest tests/test_kda_prefix_caching.py -v -s

Background
----------
ATOM's per-request recurrent-state pool supports a *fork*: on a prefix-cache
hit the request RESUMES from the checkpoint group it hit (`state_fork_src`)
but WRITES into a freshly popped group. The two groups differ for exactly one
forward, which is why `GDNAttentionMetadata` carries two index tensors:

    non_spec_state_indices_tensor      -> slot to WRITE
    non_spec_state_indices_in_tensor   -> slot to READ

`atom/model_ops/attention_gdn.py` (Qwen3-Next GDN) threads both.
`atom/models/kimi_k3.py` (KDA) threads only the WRITE tensor, so on a
prefix-cache hit both the conv state and the SSM state are read from the
freshly popped group -- i.e. from whatever the recycled slot still held --
instead of from the checkpoint. This test reproduces that and verifies the fix.

The test drives the same kernels the model uses, so it is a faithful repro
without importing the (compiled) model module.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KDA repro needs a GPU"
)

NUM_HEADS = 4
HEAD_DIM = 128
CONV_KERNEL = 4
GATE_LOWER_BOUND = -5.0
DTYPE = torch.bfloat16

PROJ = NUM_HEADS * HEAD_DIM  # local_proj_size
CONV_DIM = 3 * PROJ  # q | k | v
NUM_SLOTS = 8


def _kda(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens):
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    return chunk_kimi_delta_attn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta.float(),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=GATE_LOWER_BOUND,
        cu_seqlens=cu_seqlens,
        state_v_first=True,
    )


class Weights:
    """Fixed random projections standing in for the KDA layer weights."""

    def __init__(self, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def rnd(*shape, scale=0.05, dtype=DTYPE):
            return (
                torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
                * scale
            ).to(dtype)

        self.conv_weight = rnd(CONV_DIM, CONV_KERNEL, scale=0.3)
        self.A_log = rnd(NUM_HEADS, scale=1.0, dtype=torch.float32)
        self.dt_bias = rnd(PROJ, scale=1.0, dtype=torch.float32)
        self.gen = gen

    def activations(self, num_tokens, seed):
        """Deterministic per-token activations, so a chunk of a sequence has
        exactly the same values as the matching slice of the whole sequence."""
        gen = torch.Generator(device="cuda").manual_seed(seed)
        mixed_qkv = (
            torch.randn(
                num_tokens, CONV_DIM, generator=gen, device="cuda", dtype=torch.float32
            )
            * 0.5
        ).to(DTYPE)
        gate = (
            torch.randn(
                num_tokens,
                NUM_HEADS,
                HEAD_DIM,
                generator=gen,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.5
        ).to(DTYPE)
        beta = (
            torch.randn(
                num_tokens, NUM_HEADS, generator=gen, device="cuda", dtype=torch.float32
            )
            * 0.5
        ).to(DTYPE)
        return mixed_qkv, gate, beta


def run_prefill(
    w,
    conv_state,
    ssm_state,
    mixed_qkv,
    gate,
    beta,
    *,
    write_slot,
    read_slot,
    has_initial,
    apply_fix,
):
    """One KDA prefill forward, mirroring `KimiKDAAttention._forward_impl`.

    `apply_fix=False` reproduces today's ATOM code: the READ slot is ignored
    and the WRITE slot is used for both. `apply_fix=True` threads the READ
    slot into the conv-state gather and the SSM initial-state gather.
    """
    from atom.model_ops.attentions.gdn_attn import compute_causal_conv1d_metadata
    from atom.model_ops.kimi_k3 import gather_kda_initial_state
    from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_fn

    num_tokens = mixed_qkv.shape[0]
    dev = mixed_qkv.device
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=dev)
    write_idx = torch.tensor([write_slot], dtype=torch.int32, device=dev)
    read_idx = torch.tensor([read_slot], dtype=torch.int32, device=dev)
    has_initial_state = torch.tensor([has_initial], dtype=torch.bool, device=dev)

    # ATOM stores conv state as [slot, state_len, conv_dim]; the kernels want
    # [slot, conv_dim, state_len]. Same transpose the model does.
    conv_state_t = conv_state.transpose(-1, -2)

    nums_dict, batch_ptr, tok_off = compute_causal_conv1d_metadata(query_start_loc)

    class _Meta:
        pass

    meta = _Meta()
    meta.nums_dict = nums_dict
    meta.batch_ptr = batch_ptr
    meta.token_chunk_offset_ptr = tok_off

    conv_kwargs = {}
    if apply_fix:
        conv_kwargs["cache_indices_in"] = read_idx

    q, k, v = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        w.conv_weight,
        None,
        activation="silu",
        conv_states=conv_state_t,
        has_initial_state=has_initial_state,
        cache_indices=write_idx,
        query_start_loc=query_start_loc,
        k_dim_size=PROJ,
        v_dim_size=PROJ,
        metadata=meta,
        **conv_kwargs,
    )

    def to_bhd(x):
        return x.view(1, num_tokens, NUM_HEADS, HEAD_DIM)

    gather_idx = read_idx if apply_fix else write_idx
    initial = gather_kda_initial_state(ssm_state, gather_idx, has_initial_state)

    out, last_state = _kda(
        to_bhd(q),
        to_bhd(k),
        to_bhd(v),
        gate.view(1, num_tokens, NUM_HEADS, HEAD_DIM),
        beta.view(1, num_tokens, NUM_HEADS),
        w.A_log,
        w.dt_bias,
        initial,
        query_start_loc,
    )
    ssm_state[write_idx] = last_state.to(ssm_state.dtype)
    return out.squeeze(0)


def fresh_state():
    conv_state = torch.zeros(
        NUM_SLOTS, CONV_KERNEL - 1, CONV_DIM, dtype=DTYPE, device="cuda"
    )
    ssm_state = torch.zeros(
        NUM_SLOTS, NUM_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda"
    )
    return conv_state, ssm_state


def scenario(apply_fix, *, poison=True):
    """Prefix-cache-hit scenario with a state fork.

    Request A prefills 256 tokens into slot 1 and its state is checkpointed
    there. Request B hits that 128-token prefix, so it RESUMES from slot 1 and
    WRITES into freshly popped slot 5. Slot 5 is a recycled group that still
    holds a previous occupant's state (that is what `poison` simulates).

    Returns (chunked_output_tail, reference_output_tail).
    """
    w = Weights()
    total, prefix = 256, 128

    # ---- reference: the whole sequence in one cold prefill -----------------
    conv_state, ssm_state = fresh_state()
    mixed, gate, beta = w.activations(total, seed=1234)
    ref = run_prefill(
        w,
        conv_state,
        ssm_state,
        mixed,
        gate,
        beta,
        write_slot=0,
        read_slot=0,
        has_initial=False,
        apply_fix=apply_fix,
    )

    # ---- chunked: chunk 1 lands in slot 1 (the checkpoint) -----------------
    conv_state, ssm_state = fresh_state()
    run_prefill(
        w,
        conv_state,
        ssm_state,
        mixed[:prefix],
        gate[:prefix],
        beta[:prefix],
        write_slot=1,
        read_slot=1,
        has_initial=False,
        apply_fix=apply_fix,
    )

    if poison:
        # Slot 5 is recycled: it still carries a prior request's state.
        torch.manual_seed(7)
        conv_state[5] = torch.randn_like(conv_state[5]) * 0.5
        ssm_state[5] = torch.randn_like(ssm_state[5]) * 0.5

    # ---- chunk 2: FORK. read the checkpoint (1), write the new group (5) ---
    tail = run_prefill(
        w,
        conv_state,
        ssm_state,
        mixed[prefix:],
        gate[prefix:],
        beta[prefix:],
        write_slot=5,
        read_slot=1,
        has_initial=True,
        apply_fix=apply_fix,
    )
    return tail, ref[prefix:]


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def scenario_no_fork():
    """Control: chunked prefill with NO fork -- chunk 2 reads and writes the
    same slot 1, which is what happens without a prefix-cache hit. Today's
    unfixed code is already correct here, which is why the bug only shows up
    when prefix caching hits."""
    w = Weights()
    total, prefix = 256, 128

    conv_state, ssm_state = fresh_state()
    mixed, gate, beta = w.activations(total, seed=1234)
    ref = run_prefill(
        w,
        conv_state,
        ssm_state,
        mixed,
        gate,
        beta,
        write_slot=0,
        read_slot=0,
        has_initial=False,
        apply_fix=False,
    )

    conv_state, ssm_state = fresh_state()
    run_prefill(
        w,
        conv_state,
        ssm_state,
        mixed[:prefix],
        gate[:prefix],
        beta[:prefix],
        write_slot=1,
        read_slot=1,
        has_initial=False,
        apply_fix=False,
    )
    tail = run_prefill(
        w,
        conv_state,
        ssm_state,
        mixed[prefix:],
        gate[prefix:],
        beta[prefix:],
        write_slot=1,
        read_slot=1,
        has_initial=True,
        apply_fix=False,
    )
    return tail, ref[prefix:]


def test_no_fork_is_correct_control():
    """Proves the harness and the KDA kernels are sound: chunked prefill
    without a fork already matches the reference on UNFIXED code. So the
    failure in the next test is the fork plumbing, not the KDA math."""
    got, want = scenario_no_fork()
    err, c = rel_err(got, want), cos(got, want)
    print(f"\n[control ] rel_err={err:.4f}  cos={c:.4f}")
    assert err < 0.02, f"control must already pass on unfixed code, got {err}"


def test_bug_reproduces_without_fix():
    """Today's ATOM code: the fork's READ slot is ignored, so chunk 2 resumes
    from the recycled slot instead of the checkpoint -> wrong output."""
    got, want = scenario(apply_fix=False)
    err, c = rel_err(got, want), cos(got, want)
    print(f"\n[no fix ] rel_err={err:.4f}  cos={c:.4f}")
    assert err > 0.1, f"expected a large error without the fix, got {err}"


def test_fix_matches_reference():
    """With the READ slot threaded through, chunked prefill with a prefix-cache
    hit matches the single-shot reference."""
    got, want = scenario(apply_fix=True)
    err, c = rel_err(got, want), cos(got, want)
    print(f"\n[with fix] rel_err={err:.4f}  cos={c:.4f}")
    assert err < 0.02, f"fixed path should match the reference, got {err}"
    assert c > 0.999, f"fixed path cosine too low: {c}"


if __name__ == "__main__":
    for fix in (False, True):
        got, want = scenario(apply_fix=fix)
        tag = "with fix" if fix else "no fix  "
        print(f"[{tag}] rel_err={rel_err(got, want):.4f}  cos={cos(got, want):.4f}")
