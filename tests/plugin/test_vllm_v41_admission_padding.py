# SPDX-License-Identifier: MIT
"""CPU regressions for vLLM 0.31's newly admitted speculative padding.

Run the installed, complete Scheduler.schedule and AsyncScheduler update path,
not a copy of their padding predicate. Only KV allocation and encoder storage
are mocks; requests, queues, output construction, and token accounting are real.
Scheduler.__init__ is deliberately bypassed: no model, connector, or GPU pool is
created. This covers admission/accounting, not KV transfer or model execution.

The source contract deliberately fails when upstream adds another lookup use or
changes either of its two logical read sites. There are three Attribute loads:
the padding guard, the next-K guard, and the next-K subscript.
"""

import ast
import copy
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm")

import torch
from vllm.config.speculative import SpeculativeConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from atom.plugin.vllm.v41_speculative import patch_v41_speculative_scheduling

_LIVE_SCHEDULE = inspect.unwrap(Scheduler.schedule)
_V41 = "DeepseekV41ForCausalLM"
_K = 5


@pytest.fixture(autouse=True)
def forbid_gpu_initialization(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("admission-padding tests must not initialize a GPU")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


def _request(req_id, *, computed=256, prompt=256, output=1, drafts=()):
    request = Request(
        request_id=req_id,
        prompt_token_ids=[7] * prompt,
        sampling_params=SamplingParams(max_tokens=64),
        pooling_params=None,
    )
    request.append_output_token_ids([8] * output)
    request.num_computed_tokens = computed
    request.spec_token_ids = list(drafts)
    request.is_prefill_chunk = computed < request.num_tokens - 1
    return request


def _pair(order):
    requests = {
        "P": _request("P", computed=0, prompt=16, output=0),
        "D": _request("D"),
    }
    return [requests[req_id] for req_id in order]


class _DSparkConfig(SimpleNamespace):
    """Use the real drafting budget without constructing/loading model configs."""

    max_num_new_slots_for_drafting = SpeculativeConfig.max_num_new_slots_for_drafting
    use_dflash = SpeculativeConfig.use_dflash
    uses_draft_model = SpeculativeConfig.uses_draft_model


def _scheduler(
    waiting=(),
    *,
    running=(),
    cls=Scheduler,
    arch=_V41,
    lookup=None,
    speculating=True,
):
    scheduler = cls.__new__(cls)
    scheduler.running = list(running)
    for request in scheduler.running:
        request.status = RequestStatus.RUNNING
    scheduler.waiting = FCFSRequestQueue(waiting)
    scheduler.kv_holding_waiting = FCFSRequestQueue()
    scheduler.requests = {
        request.request_id: request
        for request in [*scheduler.running, *scheduler.waiting]
    }
    scheduler.deferred_waiting = set()
    scheduler._inflight_prefills = set()
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.max_num_running_reqs = 4
    scheduler.max_num_active_reqs = 4
    scheduler.max_num_scheduled_tokens = 128
    scheduler.max_model_len = 4096
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.current_step = 0
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.prefill_capacity_bound = False
    scheduler.adaptive_long_prefill_threshold = False
    scheduler.num_spec_tokens = _K if speculating else 0
    scheduler.num_sampled_tokens_per_step = 1
    scheduler.dynamic_sd_lookup = lookup
    scheduler.num_lookahead_tokens = scheduler.num_spec_tokens
    scheduler.num_prefill_lookahead = 1
    scheduler.prefix_replay_tokens = 0
    scheduler.need_mamba_block_aligned_split = False
    scheduler.scheduler_reserve_full_isl = False
    scheduler.is_encoder_decoder = False
    scheduler.lora_config = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.aux_output_connector = None
    scheduler.log_stats = False
    scheduler.defer_block_free = False
    scheduler.needs_kv_cache_zeroing = False
    scheduler.use_v2_model_runner = True
    scheduler.model_uses_mrope = False
    scheduler.use_pp = False
    scheduler.decode_stagger = 1
    scheduler._spec_token_placeholders = [-1] * scheduler.num_spec_tokens
    scheduler.finished_req_ids = set()
    scheduler.reset_preempted_req_ids = set()
    scheduler.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=128,
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
        async_scheduling=cls is AsyncScheduler,
    )
    scheduler.vllm_config = SimpleNamespace(
        speculative_config=(
            _DSparkConfig(
                num_speculative_tokens=_K,
                dynamic_sd_schedule=None,
                method="dspark",
                parallel_drafting=True,
            )
            if speculating
            else None
        ),
        model_config=SimpleNamespace(architectures=[arch]),
        scheduler_config=scheduler.scheduler_config,
    )
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    blocks = KVCacheBlocks(((),))
    manager = scheduler.kv_cache_manager = Mock(spec=KVCacheManager)
    manager.empty_kv_cache_blocks = blocks
    manager.allocate_slots.return_value = blocks
    manager.get_blocks.return_value = blocks
    manager.get_computed_blocks.return_value = (blocks, 0, 0)
    manager.get_num_common_prefix_blocks.return_value = [0]
    manager.take_boundary_state_offloads.return_value = {}
    manager.take_kv_cache_block_copies.return_value = ([], [])
    manager.take_new_block_ids.return_value = []
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=list, get_manager_metadata=lambda: None
    )
    return scheduler


def _install(monkeypatch, original=_LIVE_SCHEDULE):
    monkeypatch.setattr(Scheduler, "schedule", original)
    assert patch_v41_speculative_scheduling()
    assert not patch_v41_speculative_scheduling()


def _controls(scheduler):
    return copy.deepcopy(
        (
            scheduler.num_spec_tokens,
            scheduler.num_sampled_tokens_per_step,
            scheduler.num_lookahead_tokens,
            scheduler.vllm_config,
        )
    )


def _assert_constant_lookup(lookup, scheduler):
    assert type(lookup) is list, "original.schedule must see a plain constant-K list"
    assert lookup == [_K] * (scheduler.max_num_running_reqs + 1)


@pytest.mark.parametrize("batch_size", [0, 1, 4], ids=["empty", "one", "capacity"])
def test_static_wrapper_scopes_lookup_and_keeps_next_k(monkeypatch, batch_size):
    scheduler = _scheduler([_request(f"D{i}") for i in range(batch_size)])
    # Distinguish runner slots from the active-admission limit.
    if not batch_size:
        scheduler.max_num_active_reqs = 2
    controls = _controls(scheduler)
    seen = {}

    def original(self, *args, **kwargs):
        seen["lookup"] = self.dynamic_sd_lookup
        assert _controls(self) == controls
        seen["output"] = _LIVE_SCHEDULE(self, *args, **kwargs)
        seen["metadata"] = copy.deepcopy(vars(seen["output"]))
        assert self.dynamic_sd_lookup is seen["lookup"]
        return seen["output"]

    _install(monkeypatch, original)
    result = scheduler.schedule(throttle_prefills=False)
    assert scheduler.dynamic_sd_lookup is None
    assert _controls(scheduler) == controls
    assert result is seen["output"]
    assert vars(result) == seen["metadata"], "wrapper must not rewrite output metadata"
    assert len(result.num_scheduled_tokens) == batch_size
    assert result.num_spec_tokens_to_schedule == _K
    _assert_constant_lookup(seen["lookup"], scheduler)


def test_static_wrapper_restores_none_and_propagates_original_exception(monkeypatch):
    scheduler = _scheduler()
    controls = _controls(scheduler)
    failure = RuntimeError("original schedule failed")
    seen = []

    def original(self):
        seen.append(self.dynamic_sd_lookup)
        assert _controls(self) == controls
        raise failure

    _install(monkeypatch, original)
    with pytest.raises(RuntimeError, match="original schedule failed") as raised:
        scheduler.schedule()
    assert raised.value is failure
    assert scheduler.dynamic_sd_lookup is None
    assert _controls(scheduler) == controls
    _assert_constant_lookup(seen[0], scheduler)


class _DynamicLookup(list):
    def __init__(self):
        super().__init__([0, 4, 3, 2, 1])
        self.reads = []

    def __getitem__(self, index):
        self.reads.append(index)
        return super().__getitem__(index)


@pytest.mark.parametrize("batch_size,expected_k", [(0, 5), (1, 4), (4, 1)])
def test_real_dynamic_lookup_is_preserved_by_identity(
    monkeypatch, batch_size, expected_k
):
    lookup = _DynamicLookup()
    scheduler = _scheduler(
        [_request(f"D{i}") for i in range(batch_size)], lookup=lookup
    )
    controls = _controls(scheduler)

    def original(self):
        assert self.dynamic_sd_lookup is lookup
        assert _controls(self) == controls
        return _LIVE_SCHEDULE(self)

    _install(monkeypatch, original)
    result = scheduler.schedule()
    assert scheduler.dynamic_sd_lookup is lookup
    assert list(lookup) == [0, 4, 3, 2, 1]
    assert lookup.reads == ([batch_size] if batch_size else [])
    assert result.num_spec_tokens_to_schedule == expected_k
    assert _controls(scheduler) == controls


def test_real_dynamic_lookup_survives_original_exception(monkeypatch):
    lookup = _DynamicLookup()
    scheduler = _scheduler(lookup=lookup)
    failure = RuntimeError("dynamic schedule failed")

    def original(self):
        assert self.dynamic_sd_lookup is lookup
        raise failure

    _install(monkeypatch, original)
    with pytest.raises(RuntimeError, match="dynamic schedule failed") as raised:
        scheduler.schedule()
    assert raised.value is failure
    assert scheduler.dynamic_sd_lookup is lookup
    assert list(lookup) == [0, 4, 3, 2, 1]
    assert lookup.reads == []


def test_existing_empty_dynamic_lookup_is_not_treated_as_none(monkeypatch):
    lookup = []
    scheduler = _scheduler(lookup=lookup)

    def original(self):
        assert self.dynamic_sd_lookup is lookup
        return _LIVE_SCHEDULE(self)

    _install(monkeypatch, original)
    result = scheduler.schedule()
    assert result.num_spec_tokens_to_schedule == _K
    assert scheduler.dynamic_sd_lookup is lookup
    assert lookup == []


@pytest.mark.parametrize(
    "arch,speculating", [("DeepseekV4ForCausalLM", True), (_V41, False)]
)
def test_non_v41_or_no_speculation_does_not_install_lookup(
    monkeypatch, arch, speculating
):
    scheduler = _scheduler(arch=arch, speculating=speculating)
    controls = _controls(scheduler)
    result = object()

    def original(self):
        assert self.dynamic_sd_lookup is None
        assert _controls(self) == controls
        return result

    _install(monkeypatch, original)
    assert scheduler.schedule() is result
    assert scheduler.dynamic_sd_lookup is None
    assert _controls(scheduler) == controls


@pytest.mark.parametrize("cls", [Scheduler, AsyncScheduler], ids=["sync", "async"])
@pytest.mark.parametrize("order", ["PD", "DP"])
def test_upstream_control_new_admissions_really_get_five_padding_tokens(cls, order):
    """Positive control on the same real path: running is empty in both orders."""
    scheduler = _scheduler(_pair(order), cls=cls)
    result = _LIVE_SCHEDULE(scheduler)
    assert result.num_scheduled_tokens == {"P": 16, "D": 6}
    assert result.scheduled_spec_decode_tokens == {"D": [-1] * _K}
    assert result.num_spec_tokens_to_schedule == _K
    assert scheduler.requests["D"].num_computed_tokens == 262
    if cls is AsyncScheduler:
        assert scheduler.requests["D"].num_output_placeholders == _K + 1
        assert scheduler.requests["P"].num_output_placeholders == 1
        assert scheduler._spec_token_placeholders == [-1] * _K


@pytest.mark.parametrize("cls", [Scheduler, AsyncScheduler], ids=["sync", "async"])
@pytest.mark.parametrize("order", ["PD", "DP"])
def test_v41_new_admissions_never_mix_prefill_with_padding(monkeypatch, cls, order):
    scheduler = _scheduler(_pair(order), cls=cls)
    controls = _controls(scheduler)
    _install(monkeypatch)
    result = scheduler.schedule()
    assert scheduler.dynamic_sd_lookup is None
    assert _controls(scheduler) == controls
    # Assert the allocation input as well as the returned output, so a metadata
    # cleanup AFTER original.schedule cannot masquerade as disabling padding.
    allocations = {
        call.args[0].request_id: call.args[1]
        for call in scheduler.kv_cache_manager.allocate_slots.call_args_list
    }
    assert allocations == {"P": 16, "D": 1}
    assert result.num_scheduled_tokens == {"P": 16, "D": 1}
    assert result.total_num_scheduled_tokens == 17
    assert result.scheduled_spec_decode_tokens == {}
    assert result.num_spec_tokens_to_schedule == _K
    assert [request.req_id for request in result.scheduled_new_reqs] == list(order)
    assert scheduler.requests["D"].num_computed_tokens == 257
    assert not scheduler.waiting
    if cls is AsyncScheduler:
        # The current unpadded step still owes its bonus token; the NEXT step
        # still has K draft placeholders. Zeroing either global knob is wrong.
        assert scheduler._spec_token_placeholders == [-1] * _K
        for request in scheduler.running:
            assert request.num_output_placeholders == 1
            assert request.spec_token_ids is scheduler._spec_token_placeholders
            assert request.next_decode_eligible_step == scheduler.current_step + 1


@pytest.mark.parametrize("cls", [Scheduler, AsyncScheduler], ids=["sync", "async"])
@pytest.mark.parametrize("input_budget,expected_tokens", [(5, 1), (4, 0)])
def test_dspark_admission_input_budget_boundary(
    monkeypatch, cls, input_budget, expected_tokens
):
    """One unpadded query plus DSpark's four drafting slots must fit."""
    decode = _request("D")
    scheduler = _scheduler([decode], cls=cls)
    # Keep the target-token budget ample; isolate the input/drafting budget.
    scheduler.scheduler_config.max_num_batched_tokens = input_budget
    _install(monkeypatch)
    result = scheduler.schedule()
    assert result.num_scheduled_tokens == ({"D": 1} if expected_tokens else {})
    assert result.total_num_scheduled_tokens == expected_tokens
    assert result.scheduled_spec_decode_tokens == {}
    assert result.num_spec_tokens_to_schedule == _K
    assert scheduler.dynamic_sd_lookup is None
    assert decode.num_computed_tokens == 256 + expected_tokens
    assert len(scheduler.waiting) == 1 - expected_tokens
    if expected_tokens:
        allocation = scheduler.kv_cache_manager.allocate_slots.call_args
        assert allocation.args[:2] == (decode, 1)
    else:
        scheduler.kv_cache_manager.allocate_slots.assert_not_called()
    if cls is AsyncScheduler:
        assert decode.num_output_placeholders == expected_tokens
        assert scheduler._spec_token_placeholders == [-1] * _K


@pytest.mark.parametrize("cls", [Scheduler, AsyncScheduler], ids=["sync", "async"])
def test_non_v41_live_admission_padding_is_unchanged(monkeypatch, cls):
    scheduler = _scheduler(_pair("PD"), cls=cls, arch="DeepseekV4ForCausalLM")
    _install(monkeypatch)
    result = scheduler.schedule()
    assert result.num_scheduled_tokens == {"P": 16, "D": 6}
    assert result.scheduled_spec_decode_tokens == {"D": [-1] * _K}
    assert result.num_spec_tokens_to_schedule == _K
    assert scheduler.dynamic_sd_lookup is None


@pytest.mark.parametrize("cls", [Scheduler, AsyncScheduler], ids=["sync", "async"])
def test_decode_only_keeps_real_drafts_and_async_bonus(monkeypatch, cls):
    decode = _request("D", drafts=[11, 12, 13, 14, 15])
    scheduler = _scheduler(running=[decode], cls=cls)
    _install(monkeypatch)
    result = scheduler.schedule()
    assert result.num_scheduled_tokens == {"D": 6}
    assert result.scheduled_spec_decode_tokens == {"D": [11, 12, 13, 14, 15]}
    assert result.num_spec_tokens_to_schedule == _K
    assert scheduler.dynamic_sd_lookup is None
    if cls is AsyncScheduler:
        assert decode.num_output_placeholders == _K + 1
        assert decode.spec_token_ids == [-1] * _K


def test_async_empty_step_still_has_next_k_five(monkeypatch):
    scheduler = _scheduler(cls=AsyncScheduler)
    _install(monkeypatch)
    result = scheduler.schedule()
    assert result.num_scheduled_tokens == {}
    assert result.num_spec_tokens_to_schedule == _K
    assert scheduler._spec_token_placeholders == [-1] * _K
    assert scheduler.dynamic_sd_lookup is None


def test_live_lookup_contract_has_only_padding_and_next_k_read_sites():
    """Version coupling, not a replacement implementation of schedule()."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(Scheduler)))
    schedule = next(
        node
        for node in tree.body[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "schedule"
    )
    refs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "dynamic_sd_lookup"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(refs) == 3, "lookup gained another consumer; re-audit the adaptation"
    schedule_nodes = list(ast.walk(schedule))
    assert all(ref in schedule_nodes for ref in refs)
    assert all(
        isinstance(node.ctx, ast.Load)
        for node in schedule_nodes
        if isinstance(node, ast.Attribute) and node.attr == "dynamic_sd_lookup"
    ), "original.schedule now changes lookup state"
    async_tree = ast.parse(textwrap.dedent(inspect.getsource(AsyncScheduler)))
    assert not any(
        (isinstance(node, ast.Attribute) and node.attr == "dynamic_sd_lookup")
        or (isinstance(node, ast.Constant) and node.value == "dynamic_sd_lookup")
        for node in ast.walk(async_tree)
    ), "async gained an independent lookup consumer"
    # An alias via getattr/setattr would evade an Attribute-only read count.
    assert not any(
        isinstance(node, ast.Constant) and node.value == "dynamic_sd_lookup"
        for node in ast.walk(tree)
    )
    sites = [
        node
        for node in ast.walk(schedule)
        if isinstance(node, ast.If)
        and any(ref in list(ast.walk(node.test)) for ref in refs)
    ]
    assert len(sites) == 2
    sites.sort(key=lambda node: node.lineno)
    padding, next_k = sites
    # Freeze the actual upstream guards as a compatibility assertion only.
    expected_padding = ast.parse(
        "(self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None) "
        "and self.num_sampled_tokens_per_step > 0 and num_new_tokens == 1 "
        "and not prefill_scheduled "
        "and (scheduled_running_reqs or num_computed_tokens > 0)",
        mode="eval",
    ).body
    assert ast.dump(padding.test) == ast.dump(expected_padding)
    expected_next_k = ast.parse(
        "if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:\n"
        "    num_spec_tokens_to_schedule = "
        "self.dynamic_sd_lookup[len(num_scheduled_tokens)]\n"
    ).body[0]
    assert ast.dump(next_k) == ast.dump(expected_next_k)
    default_k = schedule.body[schedule.body.index(next_k) - 1]
    assert ast.dump(default_k) == ast.dump(
        ast.parse("num_spec_tokens_to_schedule = self.num_spec_tokens").body[0]
    )
    assert "schedule" not in vars(AsyncScheduler), "async no longer inherits the patch"
