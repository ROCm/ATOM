# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Arch-agnostic accuracy test for the native RCCL MoE all2all backend.

Defines a full expert-parallel **MoE block** that mirrors a real transformer MoE
layer: it takes the attention output ``hidden_states`` ``[T, H]``, runs the gate
(router) to pick top-k experts, dispatches tokens across ``world_size`` ranks via
the native RCCL prepare()/finalize() (HT variable-length OR LL fixed-capacity),
runs the owning expert MLP on each received token, combines the weighted results,
and returns the post-MoE tensor ``[T, H]`` — the same signature a real block has.

The block's output is checked against a single-process dense MoE reference
(identical gate + expert weights, no all2all). This validates the whole contract
end to end: gate -> global-(rank,expert) routing -> cross-rank all_to_all
dispatch -> correct local expert per row -> all_to_all-back + top-k-weighted
combine.

It runs on CPU over a gloo process group (spawned procs), so it works on ANY
architecture — no GPU / AITER / Triton needed (the prepare/finalize fused kernels
fall back to their pure-torch reference on CPU). Use it to confirm the routing
math and the HT/LL algorithms are correct on a new machine before trusting the
GPU kernels.

Run directly:
    python tests/test_rccl_moe_block_accuracy.py
or under pytest:
    python -m pytest tests/test_rccl_moe_block_accuracy.py -v
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# --------------------------------------------------------------------------- #
# Shared MoE weights (gate + experts). Built identically on every rank / the
# reference from a fixed seed, so all replicas agree bit-for-bit.
# --------------------------------------------------------------------------- #
class MoEWeights:
    def __init__(self, hidden, inter, num_experts, seed=1234):
        g = torch.Generator().manual_seed(seed)
        self.H = hidden
        self.I = inter
        self.E = num_experts
        self.gate = torch.randn(hidden, num_experts, generator=g) * 0.1  # [H, E]
        self.w1 = torch.randn(num_experts, hidden, inter, generator=g) * 0.1  # [E,H,I]
        self.w2 = torch.randn(num_experts, inter, hidden, generator=g) * 0.1  # [E,I,H]

    def route(self, hidden_states, topk):
        """Gate -> softmax -> top-k. Returns (topk_weights [T,K] fp32,
        topk_ids [T,K] int32). Renormalized top-k weights, like a real router."""
        logits = hidden_states.float() @ self.gate  # [T, E]
        probs = torch.softmax(logits, dim=-1)
        w, ids = torch.topk(probs, topk, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)  # renormalize
        return w.to(torch.float32), ids.to(torch.int32)

    def expert(self, x, e):
        """Expert e's MLP on x[.,H] -> [.,H] (SiLU MLP), float32."""
        h = torch.nn.functional.silu(x.float() @ self.w1[e].float())
        return h @ self.w2[e].float()


def dense_moe_block(hidden_states, weights, topk):
    """Single-process reference MoE block: gate -> top-k -> weighted sum of expert
    outputs. Takes hidden_states [T,H], returns [T,H] (float32)."""
    tw, ids = weights.route(hidden_states, topk)
    T, K = ids.shape
    out = torch.zeros(T, weights.H, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            out[t] += tw[t, k].float() * weights.expert(hidden_states[t : t + 1], e)[0]
    return out


# --------------------------------------------------------------------------- #
# The RCCL MoE block under test: hidden_states [T,H] -> [T,H] via prepare()/
# per-expert MLP / finalize() on the given RCCL path.
# --------------------------------------------------------------------------- #
class RcclMoEBlock:
    def __init__(self, pf, weights, rank, world_size, topk):
        self.pf = pf  # RcclHT/LL PrepareAndFinalize instance
        self.weights = weights
        self.rank = rank
        self.world_size = world_size
        self.topk = topk
        self.nle = weights.E // world_size  # local experts on this rank

    def __call__(self, hidden_states):
        """hidden_states [T, H] (this rank's attention output) -> [T, H]."""
        w = self.weights
        tw, ids = w.route(hidden_states, self.topk)

        # 1. dispatch tokens to the ranks that own their routed experts.
        recv_a1, recv_scale, meta, recv_ids, recv_w = self.pf.prepare(
            hidden_states,
            tw,
            ids,
            num_experts=w.E,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=None,
        )
        recv_ids_flat = recv_ids.reshape(-1)
        num_rows = recv_a1.shape[0]

        # 2. run the owning LOCAL expert on each received row (topk==1 layout).
        fused = torch.zeros(num_rows, w.H, dtype=torch.float32)
        for r in range(num_rows):
            e = int(recv_ids_flat[r])
            if e < 0:  # LL padding row
                continue
            le = e - self.rank * self.nle
            assert 0 <= le < self.nle, f"rank {self.rank} got non-local expert {e}"
            fused[r] = w.expert(recv_a1[r : r + 1], e)[0]

        # 3. combine back to per-token outputs (top-k weighted, home order).
        return self.pf.finalize(None, fused.to(recv_a1.dtype), tw, ids, False)


# --------------------------------------------------------------------------- #
# Per-rank worker.
# --------------------------------------------------------------------------- #
def _run_rank(rank, world_size, path, cfg, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", cfg["port"])
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    pg = dist.new_group(list(range(world_size)))

    import atom.model_ops.fused_moe.rccl_prepare_finalize as m

    H, inter, K, E = cfg["hidden"], cfg["inter"], cfg["topk"], cfg["num_experts"]
    T, graph_bs = cfg["tokens_per_rank"], cfg["graph_bs"]
    nle = E // world_size

    class _FakeGroup:
        def __init__(self, pg):
            self.device_group = pg
            self.rank_in_group = rank

    class _Ctx:
        is_dummy_run = False

        def __init__(self, path):
            self.is_prefill = path == "ht"
            self.dp_uniform_decode = path == "ll"
            self.graph_bs = graph_bs

    m.get_forward_context = lambda: type("F", (), {"context": _Ctx(path)})()

    cls = m.RcclHTPrepareAndFinalize if path == "ht" else m.RcclLLPrepareAndFinalize
    pf = cls(
        rank=rank,
        world_size=world_size,
        hidden_dim=H,
        scale_dim=0,
        max_tokens_per_rank=max(T, graph_bs) * 2,
        num_local_experts=nle,
        num_experts_per_token=K,
        in_dtype=torch.float32,
        use_fp8_dispatch=False,
        quant_type=None,
        ep_group=_FakeGroup(pg),
    )

    weights = MoEWeights(H, inter, E)  # identical on every rank + reference
    block = RcclMoEBlock(pf, weights, rank, world_size, K)

    # this rank's attention output.
    gr = torch.Generator().manual_seed(9000 + rank)
    hidden_states = torch.randn(T, H, generator=gr)

    out = block(hidden_states)  # [T, H] via RCCL MoE
    ref = dense_moe_block(hidden_states, weights, K)  # [T, H] dense reference

    rel = (out.float() - ref).norm() / ref.norm().clamp(min=1e-8)
    ret[rank] = float(rel)
    dist.barrier()
    dist.destroy_process_group()


def _run_moe_block(path, world_size=2, port="29610", **overrides):
    cfg = dict(
        hidden=16,
        inter=24,
        topk=4,
        num_experts=8,  # global; must be divisible by world_size
        tokens_per_rank=6,
        graph_bs=6,  # LL per-(src,dst) capacity = graph_bs * topk
        port=port,
    )
    cfg.update(overrides)
    assert cfg["num_experts"] % world_size == 0
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_run_rank, args=(world_size, path, cfg, ret), nprocs=world_size, join=True)
    return [ret[r] for r in range(world_size)]


# --------------------------------------------------------------------------- #
# pytest entry points.
# --------------------------------------------------------------------------- #
def test_ht_moe_block_accuracy():
    rels = _run_moe_block("ht", world_size=2, port="29611")
    for r, rel in enumerate(rels):
        assert rel < 1e-5, f"HT rank {r} rel err {rel:.3e} too high"


def test_ll_moe_block_accuracy():
    # graph_bs*topk must be >= max tokens any single dest receives, else the LL
    # capacity drops tokens (intentionally lossy). Small T + graph_bs=T holds.
    rels = _run_moe_block("ll", world_size=2, port="29612")
    for r, rel in enumerate(rels):
        assert rel < 1e-5, f"LL rank {r} rel err {rel:.3e} too high"


def test_ht_moe_block_world4():
    rels = _run_moe_block("ht", world_size=4, num_experts=8, port="29613")
    for r, rel in enumerate(rels):
        assert rel < 1e-5, f"HT(w4) rank {r} rel err {rel:.3e} too high"


if __name__ == "__main__":
    ok = True
    for name, path, ws, extra in [
        ("HT ws=2", "ht", 2, {"port": "29621"}),
        ("LL ws=2", "ll", 2, {"port": "29622"}),
        ("HT ws=4", "ht", 4, {"port": "29623"}),
        ("LL ws=4", "ll", 4, {"port": "29624"}),
    ]:
        rels = _run_moe_block(path, world_size=ws, **extra)
        worst = max(rels)
        status = "PASS" if worst < 1e-5 else "FAIL"
        ok = ok and worst < 1e-5
        print(f"[{name}] per-rank rel err = {[f'{x:.2e}' for x in rels]}  -> {status}")
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
