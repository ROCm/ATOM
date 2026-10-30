# SPDX-License-Identifier: MIT
"""Qwen3.5 MTP opts into ``reuse_draft_graph_step_buffers``.

With it, the draft mid-step recording writes its hidden states and argmax ids
into its own staged inputs, so a replay hands back the staged buffers (no
clones) and the next stage finds its sources already in place (no copies).
These pin that the opt-in is on (and that the env turns it off), and that the
mid-step loop -- driven through the real ``EagleProposer`` staging, step and
head code with an emulated replay -- produces the same draft ids and the same
KV writes on both paths, padded batch included.
"""

import types

import pytest
import torch

mod_eagle = pytest.importorskip(
    "atom.spec_decode.eagle_proposer",
    reason="the drafter imports aiter at module load",
    exc_type=ImportError,
)

EagleProposer = mod_eagle.EagleProposer

H, VOCAB, NUM_SLOTS = 16, 32, 64


def _qwen35_mtp_class():
    mod = pytest.importorskip(
        "atom.models.qwen3_5_mtp",
        reason="the Qwen3.5 MTP model imports aiter at module load",
        exc_type=ImportError,
    )
    return mod.Qwen3_5MTP


def test_qwen35_mtp_reuses_step_buffers_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_QWEN35_MTP_REUSE_STEP_BUFFERS", raising=False)
    model = object.__new__(_qwen35_mtp_class())
    assert model.reuse_draft_graph_step_buffers is True


def test_env_turns_the_qwen35_reuse_off(monkeypatch):
    monkeypatch.setenv("ATOM_QWEN35_MTP_REUSE_STEP_BUFFERS", "0")
    model = object.__new__(_qwen35_mtp_class())
    assert model.reuse_draft_graph_step_buffers is False


@pytest.mark.parametrize("env, expected", [("1", True), ("0", False)])
def test_declaring_the_qwen35_pass_picks_up_the_flag(monkeypatch, env, expected):
    monkeypatch.setenv("ATOM_QWEN35_MTP_REUSE_STEP_BUFFERS", env)
    proposer = _proposer(object.__new__(_qwen35_mtp_class()))
    (graph,) = EagleProposer._declare_draft_graphs(proposer)
    assert proposer._reuse_step_buffers is expected
    # MRoPE positions, as Qwen3.5 stages them.
    assert graph.inputs["positions"].batch_last


# ---- mid-step equivalence: reuse vs clone-and-restage ----------------------


class _ToyDraft:
    """A deterministic stand-in for the Qwen3.5 MTP forward and LM head.

    Reads ``input_ids`` and ``hidden_states`` (and positions) at the top, as the
    real forward does, normalizes the hidden input OUT of place, writes one KV
    row per sequence through ``slot_mapping`` (skipping -1), and returns a
    freshly allocated ``[running_bs, H]`` output.
    """

    def __init__(self, reuse):
        g = torch.Generator().manual_seed(0)
        self.reuse_draft_graph_step_buffers = reuse
        self.emb = torch.randn(VOCAB, H, generator=g).to(torch.bfloat16)
        self.w = torch.randn(H, H, generator=g).to(torch.bfloat16)
        self.head = torch.randn(H, VOCAB, generator=g).to(torch.bfloat16)
        self.kv = torch.zeros(NUM_SLOTS, H, dtype=torch.bfloat16)
        self.slot_mapping = None

    def __call__(self, *, input_ids, positions, hidden_states):
        e = self.emb[input_ids.long()]
        h = hidden_states / (hidden_states.float().norm(dim=-1, keepdim=True) + 1).to(
            hidden_states.dtype
        )
        x = torch.tanh((e + h) @ self.w + positions[0].unsqueeze(-1).to(e.dtype) / 64)
        slots = self.slot_mapping
        live = slots >= 0
        self.kv[slots[live].long()] = x[live]
        return x

    def compute_draft_ids(self, hidden_states, spec_step_idx=0, *, out):
        out.copy_((hidden_states @ self.head).float().argmax(-1).to(torch.int32))
        return out


def _proposer(model, mtp_k=5):
    proposer = types.SimpleNamespace(
        runner=types.SimpleNamespace(use_mrope=True),
        mtp_k=mtp_k,
        speculative_config=types.SimpleNamespace(
            draft_model_hf_config=types.SimpleNamespace(hidden_size=H)
        ),
        model=model,
        dtype=torch.bfloat16,
        _step_warmup_inputs=lambda *a, **kw: None,
    )
    for name in ("_step_forward", "_step_head", "_stage_step_inputs"):
        setattr(
            proposer, name, types.MethodType(getattr(EagleProposer, name), proposer)
        )
    return proposer


def _capture(graph, running_bs, model, recordings):
    """A recording on CPU: a replay re-runs the captured pass on the captured
    staged views and lands each output in the storage the capture allocated --
    the one property of a replay that the reuse contract leans on."""
    staged = {r: graph._stage_one(r, running_bs, None) for r in graph.inputs}
    model.slot_mapping = torch.full((running_bs,), -1, dtype=torch.int64)
    recorded = graph._forward_and_epilogue(running_bs, **staged)

    def replay():
        fresh = graph._forward_and_epilogue(running_bs, **staged)
        for got, fixed in zip(fresh, recorded):
            if got.data_ptr() != fixed.data_ptr():
                fixed.copy_(got)

    recordings.add(running_bs, recorded, on_replay=replay)


def _run_mid_steps(recordings, reuse, scheduled_bs, running_bs, mtp_k=5):
    """`propose` from step 0's output on: the gather, step 0's ids, positions
    staging, then steps 1..k-1 through stage -> run -> export. Returns the
    draft ids and the KV pool."""
    model = _ToyDraft(reuse)
    proposer = _proposer(model, mtp_k)
    (step,) = EagleProposer._declare_draft_graphs(proposer)
    proposer.step = step
    assert proposer._reuse_step_buffers is reuse
    step.bind(types.SimpleNamespace(max_num_seqs=8), "cpu")
    _capture(step, running_bs, model, recordings)

    g = torch.Generator().manual_seed(1)
    tokens = scheduled_bs * (mtp_k + 1)
    ret_hidden = torch.randn(tokens, H, generator=g).to(torch.bfloat16)
    last_token_indices = torch.arange(scheduled_bs) * (mtp_k + 1) + 2
    draft = torch.empty(mtp_k, scheduled_bs, dtype=torch.int32)

    hidden_out = step.buffer("hidden_states", scheduled_bs) if reuse else None
    sample = torch.index_select(ret_hidden, 0, last_token_indices, out=hidden_out)
    next_ids = model.compute_draft_ids(sample, out=draft[0])
    positions = torch.arange(scheduled_bs, dtype=torch.int64).expand(3, -1) + 40
    positions = step.stage(running_bs, {"positions": positions})["positions"]
    base_slots = torch.arange(scheduled_bs, dtype=torch.int64) * 8

    for i in range(1, mtp_k):
        positions += 1  # `_advance_decode_metadata`, in place on the staged rows
        slots = torch.full((running_bs,), -1, dtype=torch.int64)
        slots[:scheduled_bs] = base_slots + i
        model.slot_mapping = slots
        staged = proposer._stage_step_inputs(running_bs, next_ids, positions, sample)
        if reuse:
            # The previous step's outputs ARE the staged buffers: no copy.
            # Step 0's hidden gather lands there too; its ids do not (they are
            # exported into `draft[0]`), so step 1 still stages the ids.
            assert staged["hidden_states"].data_ptr() == sample.data_ptr()
            if i > 1:
                assert staged["input_ids"].data_ptr() == next_ids.data_ptr()
        ret, ids = step.run(running_bs, **staged)
        if reuse:
            # ...and the replay hands them back without a clone.
            assert ret.data_ptr() == staged["hidden_states"].data_ptr()
            assert ids.data_ptr() == staged["input_ids"].data_ptr()
        sample = ret[:scheduled_bs]
        next_ids = ids[:scheduled_bs]
        draft[i].copy_(next_ids)
    return draft.t().contiguous(), model.kv


@pytest.mark.parametrize("scheduled_bs, running_bs", [(4, 4), (3, 4), (1, 2)])
def test_reuse_matches_the_clone_path_bitwise(
    req_graph_recordings, scheduled_bs, running_bs
):
    ids_reuse, kv_reuse = _run_mid_steps(
        req_graph_recordings, True, scheduled_bs, running_bs
    )
    ids_clone, kv_clone = _run_mid_steps(
        req_graph_recordings, False, scheduled_bs, running_bs
    )
    assert torch.equal(ids_reuse, ids_clone)
    assert torch.equal(kv_reuse, kv_clone)
    # Non-trivial: the steps actually wrote KV and moved the ids around.
    assert kv_reuse.abs().sum() > 0
    assert len(set(ids_reuse.flatten().tolist())) > 1
