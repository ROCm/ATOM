"""The MoRI v2 MegaMoE glue, without a GPU or aiter: MegaMoE, aiter's recipe
types, the cco communicators and triton_mega_moe are stand-ins, so what is
checked is ATOM's side -- the Triton-experts branch, how layers share one
MegaMoE, and the teardown order."""

import enum
import importlib.util
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from aiter_stub import stubbed_aiter

_NO_TRITON = importlib.util.find_spec("triton") is None
_FAKE_TRITON = ("triton", "triton.language") if _NO_TRITON else ()
with stubbed_aiter("aiter.ops.flydsl.moe_common", "aiter.fused_moe", *_FAKE_TRITON):
    from atom.model_ops.fused_moe import mori_v2_prepare_finalize as mv2

TOKENS, TOPK, EPR, WORLD, MAX_TOKENS = 4, 2, 3, 4, 16
ARENA_BYTES = 3 << 20


class DispatchKind(str, enum.Enum):
    COMPACT = "compact"
    MORI = "mori"


class CombineKind(str, enum.Enum):
    FUSED_SCATTER = "fused_scatter"
    MORI = "mori"


@dataclass(frozen=True)
class Compute:
    activation: str
    gate_mode: str
    quant_type: str
    hidden_pad: int
    intermediate_pad: int
    swiglu_limit: float


@dataclass(frozen=True)
class Recipe:
    dispatch: DispatchKind
    combine: CombineKind
    dispatch_wire: str
    combine_wires: tuple
    compute: Compute


@dataclass(frozen=True)
class Geometry:
    rank: int
    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    max_tokens_per_rank: int


API = SimpleNamespace(
    Recipe=Recipe,
    Compute=Compute,
    Geometry=Geometry,
    DispatchKind=DispatchKind,
    CombineKind=CombineKind,
)


class Log(list):
    def call(self, name, error=None):
        def run():
            self.append(name)
            if error is not None:
                raise error

        return run


class DevComm:
    def __init__(self, name, log):
        self.per_rank_size = ARENA_BYTES
        self.close = log.call(f"{name}.dev_comm.close")


class Comm:
    def __init__(self, name, log):
        self.name, self.log = name, log
        self.dev_comms = []
        self.barrier = lambda: None
        self.destroy = log.call(f"{name}.destroy")

    def create_dev_comm(self):
        self.dev_comms.append(DevComm(self.name, self.log))
        return self.dev_comms[-1]


class Mega:
    built: ClassVar[list] = []

    @classmethod
    def required_vmm_bytes(cls, recipe, geometry):
        return ARENA_BYTES

    @classmethod
    def from_recipe(cls, recipe, geometry, communicator, situ_beta, situ_linear_beta):
        mega = cls()
        mega.recipe, mega.geometry, mega.communicator = recipe, geometry, communicator
        mega.close = communicator.log.call(f"mega{len(cls.built)}.close")
        cls.built.append(mega)
        return mega


class Manager:
    def __init__(self, log):
        self.comm = Comm("manager_comm", log)
        self.sizes = []

    def get_or_create_cco_communicator(self, per_rank_vmm):
        self.sizes.append(per_rank_vmm)
        return self.comm


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    log = Log()
    own = []
    gathered = []

    # mori's Communicator; _init_cco_comm itself is the production one.
    def init(ep_size, ep_rank, uid, per_rank_vmm):
        own.append(Comm(f"own_comm{len(own)}", log))
        return own[-1]

    def gather(value):
        gathered.append(value)
        return [value] * WORLD

    monkeypatch.setattr(
        mv2, "Communicator", SimpleNamespace(get_unique_id=lambda: "uid", init=init)
    )
    monkeypatch.setattr(mv2, "_broadcast_from_ep_group", lambda value, src: value)
    monkeypatch.setattr(mv2, "_import_mega_recipe", lambda: API)
    monkeypatch.setattr(mv2, "_import_mega", lambda: Mega)
    monkeypatch.setattr(mv2, "_gather_from_ep_group", gather)
    monkeypatch.setattr(mv2, "_MEGA_TRANSPORTS", {})
    monkeypatch.setattr(mv2, "_OWN_COMMUNICATORS", [])
    monkeypatch.setattr(mv2, "_BORROWED_COMMUNICATOR", None)
    monkeypatch.setattr(mv2, "_SETTINGS_AGREED", False)
    monkeypatch.setattr(Mega, "built", [])
    return SimpleNamespace(log=log, own=own, gathered=gathered, manager=Manager(log))


def _build(
    state, max_tokens=MAX_TOKENS, situ_beta=None, activation="silu", combine_wire="fp4"
):
    return mv2.init_mega_transport(
        ep_rank=0,
        ep_size=WORLD,
        ep_src_global_rank=0,
        hidden_dim=64,
        max_num_inp_token_per_rank=max_tokens,
        num_experts=EPR * WORLD,
        num_experts_per_token=TOPK,
        inter_dim=32,
        activation=activation,
        gate_mode="interleave",
        quant_type="per_1x32",
        hidden_pad=0,
        intermediate_pad=0,
        swiglu_limit=0.0,
        situ_beta=situ_beta,
        combine_wire=combine_wire,
        all2all_manager=state.manager,
    )


def test_layers_share_one_megamoe_on_the_borrowed_communicator(fresh_state):
    beta = torch.tensor([1.5])
    first = _build(fresh_state, situ_beta=beta)
    # Every later layer: an equal recipe and geometry, its own beta tensor.
    assert _build(fresh_state, situ_beta=beta.clone()) is first
    assert Mega.built == [first]
    assert first.communicator is fresh_state.manager.comm
    assert fresh_state.manager.sizes == [ARENA_BYTES + mv2._CCO_DEVCOMM_BYTES]
    assert first.recipe.combine_wires == ("bf16", "fp4")
    assert first._atom_dev_comm is fresh_state.manager.comm.dev_comms[0]
    assert first._atom_per_rank_size == ARENA_BYTES
    assert not fresh_state.own


def test_the_default_wire_builds_bf16_once(fresh_state):
    assert _build(fresh_state, combine_wire="bf16").recipe.combine_wires == ("bf16",)


@pytest.mark.parametrize("other", [torch.tensor([2.5]), None])
def test_a_layer_with_other_situ_betas_is_refused(fresh_state, other):
    _build(fresh_state, situ_beta=torch.tensor([1.5]))
    with pytest.raises(ValueError, match="situ_beta"):
        _build(fresh_state, situ_beta=other)


def test_a_second_shape_gets_its_own_communicator(fresh_state):
    first = _build(fresh_state)
    second = _build(fresh_state, max_tokens=2 * MAX_TOKENS)
    assert first.communicator is fresh_state.manager.comm
    assert second.communicator is fresh_state.own[0]
    assert mv2._OWN_COMMUNICATORS == fresh_state.own
    assert len(fresh_state.manager.sizes) == 1


def test_megamoes_of_equal_size_never_share_a_communicator(fresh_state):
    # Different recipes, the same arena size: each is sized for one arena.
    first = _build(fresh_state)
    second = _build(fresh_state, activation="gelu")
    third = _build(fresh_state, activation="relu")
    assert first.communicator is fresh_state.manager.comm
    assert second.communicator is not third.communicator
    assert mv2._OWN_COMMUNICATORS == fresh_state.own
    assert fresh_state.own == [second.communicator, third.communicator]


def test_ranks_with_different_settings_are_refused(fresh_state, monkeypatch):
    monkeypatch.setattr(
        mv2, "_gather_from_ep_group", lambda value: [value, ("other",) + value[1:]]
    )
    with pytest.raises(RuntimeError, match="same on every EP rank"):
        _build(fresh_state)
    assert not Mega.built and not fresh_state.manager.sizes


def test_settings_are_agreed_before_aiter_checks_the_recipe(fresh_state, monkeypatch):
    # A rank whose gate mode aiter refuses must not leave the others waiting.
    def refuse(**fields):
        raise ValueError("gate_mode")

    monkeypatch.setattr(
        mv2, "_import_mega_recipe", lambda: SimpleNamespace(Compute=refuse)
    )
    with pytest.raises(ValueError, match="gate_mode"):
        _build(fresh_state)
    assert len(fresh_state.gathered) == 1
    assert fresh_state.gathered[0][-1] == "interleave"


def test_teardown_order_survives_a_failing_close(fresh_state):
    _build(fresh_state)
    _build(fresh_state, max_tokens=2 * MAX_TOKENS)
    newest = Mega.built[1]
    newest.close = fresh_state.log.call("mega1.close", RuntimeError("close failed"))

    with pytest.raises(RuntimeError, match="close failed"):
        mv2.close_mega_transports()
    # Newest first, each MegaMoE before its DevComm, all of them before the
    # communicators ATOM created; the borrowed one is the manager's to destroy.
    assert fresh_state.log == [
        "mega1.close",
        "own_comm0.dev_comm.close",
        "mega0.close",
        "manager_comm.dev_comm.close",
        "own_comm0.destroy",
    ]
    assert not mv2._MEGA_TRANSPORTS and not mv2._OWN_COMMUNICATORS
    assert mv2._BORROWED_COMMUNICATOR is None


def _mega(dispatch=DispatchKind.MORI, combine=CombineKind.FUSED_SCATTER):
    recipe = SimpleNamespace(dispatch=dispatch, combine=combine)
    return SimpleNamespace(recipe=recipe, max_tokens_per_rank=MAX_TOKENS)


def _forward(monkeypatch, mega, context):
    calls = []

    def triton_mega_moe(*args, **kwargs):
        calls.append(kwargs)
        return args[1]

    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.fused_moe_triton",
        SimpleNamespace(triton_mega_moe=triton_mega_moe),
    )
    monkeypatch.setattr(
        mv2, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(mv2, "get_dp_group", lambda: SimpleNamespace(world_size=2))
    prepare_finalize = SimpleNamespace(mega=mega, num_dispatchers=lambda: WORLD)
    kernel = mv2.MoriV2ModularKernel(prepare_finalize)
    experts = {
        name: torch.zeros(1)
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_scale",
            "w2_scale",
            "w13_swizzle_layout",
            "w2_swizzle_layout",
        )
    }
    kernel.forward(
        torch.zeros(TOKENS, 8),
        torch.zeros(EPR, 8, 8),
        torch.zeros(EPR, 8, 8),
        torch.zeros(TOKENS, TOPK),
        torch.zeros(TOKENS, TOPK, dtype=torch.int32),
        moe_extra_args={"triton_experts": experts},
    )
    return calls


def test_decode_step_reaches_triton_mega_moe(monkeypatch):
    context = SimpleNamespace(
        running_tokens_across_dp=[3, 5, 2, 4],
        running_tokens=4,
        is_prefill=False,
        running_tokens_are_unified=True,
    )
    (call,) = _forward(monkeypatch, _mega(), context)
    # What the group sent, below the arena's WORLD * MAX_TOKENS rows.
    assert call["recv_token_bound"] == 14
    assert call["m_eff"] == 4 * 2
    assert call["n_local_experts"] == EPR


def test_no_context_keeps_the_full_arena(monkeypatch):
    (call,) = _forward(monkeypatch, _mega(), None)
    assert call["recv_token_bound"] is None and call["m_eff"] is None


@pytest.mark.parametrize(
    "dispatch,combine,message",
    [
        (DispatchKind.COMPACT, CombineKind.FUSED_SCATTER, "ATOM_MEGA_STAGE1_FUSED=0"),
        (DispatchKind.MORI, CombineKind.MORI, "ATOM_MEGA_STAGE2_FUSED=1"),
    ],
)
def test_unsupported_pairings_are_refused(monkeypatch, dispatch, combine, message):
    with pytest.raises(RuntimeError, match=message):
        _forward(monkeypatch, _mega(dispatch, combine), None)
