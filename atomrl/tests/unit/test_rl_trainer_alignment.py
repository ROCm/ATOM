# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import logging
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

try:
    import resource  # noqa: F401
except ModuleNotFoundError:
    sys.modules["resource"] = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=0),
    )

from atomrl.trainer import rl_trainer
from atomrl.trainer.rl_trainer import RLTrainer


def _cleanup_trainer(**overrides) -> RLTrainer:
    trainer = object.__new__(RLTrainer)
    defaults = {
        "_profiler": None,
        "_rollout_pipeline": None,
        "_ray_rollout_mgr": None,
        "_actor_wg": None,
        "_ref_wg": None,
        "_ray_cluster": None,
        "_critic_worker": None,
        "_atom_engine": None,
        "_ray_vllm_engine": None,
        "_engine": None,
        "_actor_model": None,
        "_ref_model": None,
        "_optimizer": None,
        "_rank": 0,
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(trainer, name, value)
    return trainer


def test_cleanup_stops_rollout_before_ray_cluster() -> None:
    events = []
    cluster = SimpleNamespace(alive=True)

    class RolloutEngine:
        @staticmethod
        def shutdown():
            if not cluster.alive:
                raise RuntimeError("Ray cluster already stopped")
            events.append("engine")

    def shutdown_cluster():
        events.append("cluster")
        cluster.alive = False

    cluster.shutdown = shutdown_cluster
    trainer = _cleanup_trainer(
        _ray_vllm_engine=RolloutEngine(),
        _ray_cluster=cluster,
    )

    trainer.cleanup()

    assert events == ["engine", "cluster"]


def test_cleanup_is_best_effort_for_third_party_failures() -> None:
    events = []

    class FailingResource:
        def __init__(self, name, method):
            self.name = name
            setattr(self, method, self.fail)

        def fail(self):
            events.append(self.name)
            raise ValueError(f"{self.name} failed")

    trainer = _cleanup_trainer(
        _profiler=FailingResource("profiler", "stop"),
        _ray_vllm_engine=FailingResource("engine", "shutdown"),
        _ray_cluster=FailingResource("cluster", "shutdown"),
    )

    trainer.cleanup()

    assert events == ["profiler", "engine", "cluster"]
    assert trainer._profiler is None
    assert trainer._ray_vllm_engine is None
    assert trainer._ray_cluster is None


@pytest.mark.parametrize(
    "error_type", [AttributeError, pytest.param("ray", id="ray-error")]
)
def test_actor_memory_rpc_failures_remain_best_effort(monkeypatch, error_type) -> None:
    if error_type == "ray":

        class FakeRayError(Exception):
            pass

        ray_module = ModuleType("ray")
        ray_exceptions = ModuleType("ray.exceptions")
        ray_exceptions.RayError = FakeRayError
        ray_module.exceptions = ray_exceptions
        monkeypatch.setitem(sys.modules, "ray", ray_module)
        monkeypatch.setitem(sys.modules, "ray.exceptions", ray_exceptions)
        error = FakeRayError("ray unavailable")
    else:
        error = error_type("worker API unavailable")

    class FailingWorkerGroup:
        @staticmethod
        def execute_all_sync(_method):
            raise error

    trainer = object.__new__(RLTrainer)
    trainer._actor_wg = FailingWorkerGroup()

    trainer._reset_actor_memory_stats()
    assert trainer._collect_actor_memory_metrics() == {}


def test_validation_uses_configured_miles_sampling(monkeypatch) -> None:
    from atomrl.rewards import math_reward

    rollout_calls = []

    class Tokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return r"Answer: \boxed{2}"

    class RolloutEngine:
        @staticmethod
        def sleep():
            return None

    def rollout(prompts, *, num_generations, sampling_params):
        rollout_calls.append((prompts, num_generations, sampling_params))
        batch = len(prompts) * num_generations
        return (
            torch.tensor([[11, 22]] * batch),
            torch.ones(batch, 2, dtype=torch.long),
            [1] * batch,
            None,
            None,
        )

    monkeypatch.setattr(
        math_reward,
        "compute_math_reward",
        lambda responses, ground_truths: (
            torch.ones(len(responses)),
            [{"acc": True}] * len(responses),
        ),
    )

    trainer = object.__new__(RLTrainer)
    trainer._val_dataset = [{"prompt": "1+1", "label": "2"}]
    trainer._rank = 0
    trainer._tokenizer = Tokenizer()
    trainer._ray_vllm_engine = RolloutEngine()
    trainer._actor_wg = SimpleNamespace(num_workers=1)
    trainer._use_vllm = False
    trainer._use_atom = False
    trainer._is_distributed = False
    trainer.global_step = 4
    trainer.config = SimpleNamespace(
        eval=SimpleNamespace(
            num_samples=0,
            num_generations=8,
            temperature=0.8,
            top_p=0.7,
            top_k=-1,
        ),
        val_batch_size=0,
        policy=SimpleNamespace(max_response_length=4, max_total_sequence_length=8),
        logger=SimpleNamespace(num_val_samples_to_print=0),
    )
    trainer._extract_prompt_gt = lambda sample, **_: (sample["prompt"], sample["label"])
    trainer._rollout_with_ray_vllm = rollout

    metrics = trainer.run_validation()

    assert rollout_calls == [
        (
            ["1+1"],
            8,
            {
                "max_tokens": 4,
                "temperature": 0.8,
                "top_p": 0.7,
                "top_k": -1,
            },
        )
    ]
    assert metrics["val-core/acc/mean@8"] == 1.0
    assert metrics["val/num_samples"] == 8.0


def test_dsv4_actor_model_parallel_size_uses_megatron_topology() -> None:
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(
            training_backend="megatron_lumen_dsv4",
            training=SimpleNamespace(
                megatron_cfg=SimpleNamespace(
                    tensor_model_parallel_size=4,
                    pipeline_model_parallel_size=4,
                    context_parallel_size=1,
                )
            ),
        )
    )

    assert trainer._compute_actor_mp() == 16


def test_response_token_ids_respect_left_padding_and_prompt_length() -> None:
    helper = getattr(rl_trainer, "_response_token_ids", None)
    assert helper is not None

    sequence = torch.tensor([0, 0, 11, 12, 21, 22])
    attention_mask = torch.tensor([0, 0, 1, 1, 1, 1])

    response = helper(sequence, attention_mask, prompt_length=2)

    assert response.tolist() == [21, 22]


def test_response_token_ids_exclude_right_padding() -> None:
    helper = getattr(rl_trainer, "_response_token_ids", None)
    assert helper is not None

    sequence = torch.tensor([11, 12, 21, 22, 0, 0])
    attention_mask = torch.tensor([1, 1, 1, 1, 0, 0])

    response = helper(sequence, attention_mask, prompt_length=2)

    assert response.tolist() == [21, 22]


def test_compute_rewards_full_logs_response_diagnostics(caplog) -> None:
    caplog.set_level(logging.INFO, logger="atomrl.trainer.rl_trainer")
    decoded_ids: list[list[int]] = []

    class Tokenizer:
        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            decoded_ids.append(token_ids.tolist())
            return "Answer: " + r"\boxed{437}"

    trainer = object.__new__(RLTrainer)
    trainer._rank = 0
    trainer._tokenizer = Tokenizer()
    trainer._device = torch.device("cpu")
    trainer._is_distributed = False
    trainer.config = SimpleNamespace(policy=SimpleNamespace(max_response_length=2))

    _, responses, _ = trainer._compute_rewards_full(
        sequences=torch.tensor([[0, 0, 11, 12, 21, 22]]),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1]]),
        prompt_lengths=[2],
        gts_expanded=["437"],
    )

    assert decoded_ids == [[21, 22]]
    assert responses == ["Answer: " + r"\boxed{437}"]
    assert "cap_hits=1/1" in caplog.text
    assert "invalid_format=0/1" in caplog.text


def test_compute_rewards_full_can_score_quietly(caplog) -> None:
    caplog.set_level(logging.INFO, logger="atomrl.trainer.rl_trainer")

    class Tokenizer:
        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            return "Answer: " + r"\boxed{437}"

    trainer = object.__new__(RLTrainer)
    trainer._rank = 0
    trainer._tokenizer = Tokenizer()
    trainer._device = torch.device("cpu")
    trainer._is_distributed = False
    trainer.config = SimpleNamespace(policy=SimpleNamespace(max_response_length=2))

    rewards, _, accs = trainer._compute_rewards_full(
        sequences=torch.tensor([[11, 21, 22]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        prompt_lengths=[1],
        gts_expanded=["437"],
        verbose=False,
    )

    assert accs == [1.0]
    assert rewards.shape == (1,)
    assert caplog.text == ""


def test_ray_controller_resume_skips_the_prompts_already_generated() -> None:
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(train_global_batch_size=2048, gen_batch_size=384),
        algorithm=SimpleNamespace(
            name="dapo", dapo=SimpleNamespace(num_generations=16)
        ),
        num_training_steps=60,
        controller=SimpleNamespace(ray=SimpleNamespace(one_step_off_policy=False)),
    )
    trainer._algorithm = object()
    trainer._actor_wg = object()
    trainer._atom_engine = None
    trainer._ray_vllm_engine = SimpleNamespace(enable_sleep=False)
    trainer._resume_step = 60
    trainer._prompt_cursor = 0
    trainer._rollout_pipeline = None
    trainer.callbacks = []

    trainer._train_with_ray_controller()  # resumes at the last step: no step runs

    assert trainer._prompt_cursor == 60 * 384


def test_merge_left_padded_widens_narrow_blocks_in_front() -> None:
    merged = rl_trainer._merge_left_padded(
        [torch.tensor([[1, 2, 3]]), torch.tensor([[4]])],
        value=0,
    )

    assert merged.tolist() == [[1, 2, 3], [0, 0, 4]]


def test_assemble_ray_rollout_left_pads_and_aligns_log_probs() -> None:
    class Tokenizer:
        pad_token_id = 9

    trainer = object.__new__(RLTrainer)
    trainer._tokenizer = Tokenizer()
    trainer._device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(
            generation=SimpleNamespace(
                vllm_cfg=SimpleNamespace(calculate_log_probs=True)
            ),
        ),
        moe=SimpleNamespace(r3=SimpleNamespace(enabled=False, rollout_replay=False)),
    )

    sequences, mask, prompt_lengths, log_probs, ragged = trainer._assemble_ray_rollout(
        [
            {"prompt_token_ids": [1, 2], "token_ids": [3], "logprobs": [-0.1]},
            {
                "prompt_token_ids": [5],
                "token_ids": [6, 7, 8],
                "logprobs": [-0.2, -0.3, -0.4],
            },
        ]
    )

    assert sequences.tolist() == [[9, 1, 2, 3], [5, 6, 7, 8]]
    assert mask.tolist() == [[0, 1, 1, 1], [1, 1, 1, 1]]
    assert prompt_lengths == [2, 1]
    # Column t holds the log-prob of token t + 1.
    torch.testing.assert_close(
        log_probs,
        torch.tensor([[0.0, 0.0, -0.1], [-0.2, -0.3, -0.4]]),
    )
    assert ragged == {}


def test_extract_prompt_gt_uses_dsv4_encoder_and_label(monkeypatch) -> None:
    calls = []

    encoding = ModuleType("vllm.tokenizers.deepseek_v4_encoding")

    def encode_messages(messages, *, thinking_mode):
        calls.append((messages, thinking_mode))
        return (
            "<｜begin▁of▁sentence｜><｜User｜>"
            + messages[0]["content"]
            + "<｜Assistant｜><think>"
        )

    encoding.encode_messages = encode_messages
    tokenizers = ModuleType("vllm.tokenizers")
    tokenizers.deepseek_v4_encoding = encoding
    vllm = ModuleType("vllm")
    vllm.tokenizers = tokenizers
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", tokenizers)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers.deepseek_v4_encoding", encoding)

    class DeepSeekV4Tokenizer:
        chat_template = None

        @staticmethod
        def convert_tokens_to_ids(token):
            return {
                "<｜User｜>": 128803,
                "<｜Assistant｜>": 128804,
            }.get(token)

        @staticmethod
        def apply_chat_template(*args, **kwargs):
            raise ValueError("DeepSeek-V4 does not ship a Jinja chat template")

    trainer = object.__new__(RLTrainer)
    trainer._tokenizer = DeepSeekV4Tokenizer()
    messages = [{"role": "user", "content": "Solve 1+1."}]

    prompt, ground_truth = trainer._extract_prompt_gt(
        {"prompt": messages, "label": "2"}
    )

    assert calls == [(messages, "thinking")]
    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>" "Solve 1+1.<｜Assistant｜><think>"
    )
    assert ground_truth == "2"


def test_extract_prompt_gt_reads_openmathinstruct_fields() -> None:
    # nvidia/OpenMathInstruct-2, the dataset of configs/*.yaml, names its fields
    # problem/expected_answer; reading neither sent empty prompts to the rollout.
    trainer = object.__new__(RLTrainer)
    trainer._tokenizer = None
    row = {
        "problem": "What is 2+3?",
        "generated_solution": "2+3 = \\boxed{5}",
        "expected_answer": "5",
        "problem_source": "augmented_math",
    }

    assert trainer._extract_prompt_gt(row) == ("What is 2+3?", "5")


def test_extract_prompt_gt_adds_validation_answer_format() -> None:
    class Tokenizer:
        chat_template = "configured"

        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return messages[0]["content"]

    trainer = object.__new__(RLTrainer)
    trainer._tokenizer = Tokenizer()

    prompt, ground_truth = trainer._extract_prompt_gt(
        {"prompt": [{"role": "user", "content": "Solve 1+1."}], "label": "2"},
        ensure_answer_format=True,
    )

    assert prompt == (
        "Solve 1+1.\n\n"
        r"Put the final answer on its own line as: Answer: \boxed{answer}."
    )
    assert ground_truth == "2"


@pytest.mark.parametrize("old_seq_len", [6, 7])
def test_response_position_metrics_isolate_late_token_drift(old_seq_len: int) -> None:
    """Rollout/mask are [B, S-1]; Megatron old_log_probs may be width S (7 here)."""
    old_logp = torch.zeros(2, old_seq_len)
    rollout_logp = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.0],
            [0.0, 0.0, 0.5, 0.6, 0.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
        ]
    )

    got = RLTrainer._mismatch_by_response_position(
        old_logp,
        rollout_logp,
        response_mask,
        bucket_edges=(2, 4),
    )

    assert got["rollout_corr/pos_0_1/tokens"] == 4
    assert got["rollout_corr/pos_0_1/kl"] == pytest.approx(0.35)
    assert got["rollout_corr/pos_2_3/tokens"] == 2
    assert got["rollout_corr/pos_2_3/kl"] == pytest.approx(0.35)


def test_response_position_metrics_take_the_common_width() -> None:
    # A Megatron actor pads old_log_probs out to the full sequence width while
    # the mask and the rollout log-probs are built pre-shifted one narrower, so
    # the two arrive with different widths and the same frame. Reading them
    # without lining them up raised
    #   RuntimeError: The size of tensor a (4254) must match the size of tensor
    #                 b (4255) at non-singleton dimension 1
    # on example 7 after the first step's rollout.
    rollout_logp = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4]])
    response_mask = torch.tensor([[0, 1, 1, 1, 1]])
    padded_old_logp = torch.zeros(1, 6)

    got = RLTrainer._mismatch_by_response_position(
        padded_old_logp, rollout_logp, response_mask
    )
    same_width = RLTrainer._mismatch_by_response_position(
        padded_old_logp[:, :5], rollout_logp, response_mask
    )

    assert got == same_width
    assert got["rollout_corr/pos_0_127/tokens"] == 4.0


def test_token_mult_prob_error_matches_nemo_rl_definition() -> None:
    # NeMo-RL: masked mean of exp(|generation_logprobs - train_logprobs|).
    rollout_logp = torch.tensor([[0.0, -0.5, -1.0, -2.0, 0.0]])
    response_mask = torch.tensor([[0, 1, 1, 1, 0]])
    padded_old_logp = torch.tensor([[9.0, -0.4, -1.3, -2.0, 9.0, 9.0]])

    got = RLTrainer._token_mult_prob_error(padded_old_logp, rollout_logp, response_mask)

    want = (math.exp(0.1) + math.exp(0.3) + math.exp(0.0)) / 3
    assert got["rollout_corr/token_mult_prob_error"] == pytest.approx(want)
    assert (
        RLTrainer._token_mult_prob_error(
            padded_old_logp, rollout_logp, torch.zeros_like(response_mask)
        )
        == {}
    )


@pytest.mark.parametrize("use_atom", [True, False], ids=["atom", "vllm"])
def test_ipc_sync_frees_actor_cache_before_waking_the_rollout(use_atom) -> None:
    # A colocated Megatron actor held its training peak in the allocator cache;
    # ATOM's weight wake then found 238 MiB free and the sync failed with
    # "update_weights_ipc failed ... HIP out of memory" on Qwen3-30B-A3B. vLLM
    # on ROCm wakes its KV cache into the same free memory.
    from atomrl.core.config import WeightSyncConfig

    calls: list = []

    class _Stop(Exception):
        pass

    class _Mgr:
        num_replicas = 1

        @property
        def servers(self):
            calls.append("send")
            raise _Stop

    stub = SimpleNamespace(
        _ray_rollout_mgr=_Mgr(),
        _actor_wg=SimpleNamespace(num_workers=1),
        _rollout_wg=None,
        _ray_use_atom=use_atom,
        _ray_use_vllm=not use_atom,
        _ray_vllm_engine=SimpleNamespace(
            enable_sleep=True,
            wake=lambda tags=None: calls.append(("wake", tuple(tags or ()))),
        ),
        _ipc_endpoints_match_actors=lambda mgr: True,
        _release_actor_cached_memory=lambda: calls.append("free"),
        global_step=0,
        config=SimpleNamespace(
            weight_sync=WeightSyncConfig(),
            policy=SimpleNamespace(
                generation=SimpleNamespace(
                    vllm_cfg=SimpleNamespace(
                        update_weights_bucket_megabytes=512, use_shm=False
                    )
                )
            ),
        ),
    )

    with pytest.raises(_Stop):
        RLTrainer._sync_weights_ipc(stub)

    assert calls == ["free", ("wake", ("weights",)), "send"]


def test_ipc_sync_refuses_fp8_quantize_on_the_safetensors_fallback() -> None:
    """The fallback exports BF16 through disk and cannot honour fp8_quantize.

    Degrading silently here would ship the rollout un-quantized weights while the
    config said otherwise -- a train/rollout mismatch that shows up only as drift.
    """
    from atomrl.core.config import WeightSyncConfig

    stub = SimpleNamespace(
        _ray_rollout_mgr=SimpleNamespace(num_replicas=4, servers=[]),
        _actor_wg=SimpleNamespace(num_workers=8),
        _rollout_wg=SimpleNamespace(),  # separated -> fallback
        config=SimpleNamespace(weight_sync=WeightSyncConfig(fp8_quantize=True)),
    )

    with pytest.raises(RuntimeError, match="safetensors weight-sync fallback"):
        RLTrainer._sync_weights_ipc(stub)


def _force_on_policy_trainer(**changes) -> RLTrainer:
    from atomrl.core.config import AtomRLConfig

    cfg = AtomRLConfig()
    cfg.algorithm.force_on_policy_ratio = True
    cfg.policy.training_backend = "megatron_native"
    for path, value in changes.items():
        section, key = path.rsplit(".", 1)
        obj = cfg
        for part in section.split("."):
            obj = getattr(obj, part)
        setattr(obj, key, value)
    trainer = object.__new__(RLTrainer)
    trainer.config = cfg
    return trainer


def test_force_on_policy_ratio_is_off_by_default() -> None:
    trainer = _force_on_policy_trainer(**{"algorithm.force_on_policy_ratio": False})
    assert trainer._check_force_on_policy_ratio() is False


def test_force_on_policy_ratio_accepts_megatron_native_without_correction() -> None:
    trainer = _force_on_policy_trainer()
    assert trainer._check_force_on_policy_ratio() is True
    assert trainer._in_loss_rollout_is is None


def test_force_on_policy_ratio_moves_rollout_is_into_the_loss() -> None:
    """NeMo-RL runs force_on_policy_ratio with IS correction: the weights come
    from the training forward's detached log-probs, so the trainer hands the
    settings to the loss instead of refusing them."""
    trainer = _force_on_policy_trainer(
        **{
            "quantization.rollout_correction.rollout_is": "token",
            "quantization.rollout_correction.rollout_is_threshold": "1e9",
        }
    )
    assert trainer._check_force_on_policy_ratio() is True
    assert trainer._in_loss_rollout_is == {
        "rollout_is": "token",
        "rollout_is_threshold": "1e9",
        "rollout_is_seq_reduction": "sum",
    }


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"policy.training_backend": "fsdp2"}, "training_backend"),
        (
            {
                "quantization.rollout_correction.rollout_is": "token",
                "quantization.rollout_correction.rollout_is_batch_normalize": True,
            },
            "rollout_is_batch_normalize",
        ),
        ({"quantization.rollout_correction.rollout_rs": "seq_mean_k1"}, "rollout_rs"),
        ({"quantization.rollout_correction.bypass_mode": True}, "bypass_mode"),
        (
            {"quantization.rollout_correction.enabled": True},
            "rollout_correction.enabled",
        ),
    ],
)
def test_force_on_policy_ratio_refuses_what_reads_old_log_probs_first(
    changes, named
) -> None:
    """These read the old log-probs before training, need the whole batch's
    weights at once, or (for FSDP2) would turn their absence into an SFT loss,
    so the flag must fail before step 1."""
    trainer = _force_on_policy_trainer(**changes)
    with pytest.raises(ValueError, match=named):
        trainer._check_force_on_policy_ratio()


def test_force_on_policy_ratio_is_refused_off_the_ray_path() -> None:
    """Only the Ray path reads the flag; elsewhere it used to be ignored."""
    trainer = _force_on_policy_trainer()
    trainer._use_ray_controller = False
    with pytest.raises(ValueError, match="Ray controller path only"):
        trainer.setup()


def _trainer_with_policy(**policy) -> RLTrainer:
    from atomrl.core.config import AtomRLConfig

    cfg = AtomRLConfig()
    for key, value in policy.items():
        setattr(cfg.policy, key, value)
    trainer = object.__new__(RLTrainer)
    trainer.config = cfg
    return trainer


def test_prompt_cap_defaults_to_1024_and_follows_max_prompt_length() -> None:
    assert _trainer_with_policy()._max_prompt_length() == 1024
    assert _trainer_with_policy(max_prompt_length=4096)._max_prompt_length() == 4096
    capped = _trainer_with_policy(
        max_prompt_length=9000, max_total_sequence_length=4096
    )
    assert capped._max_prompt_length() == 4096


def test_a_prompt_that_fills_the_sequence_is_left_out_of_the_loss() -> None:
    """NeMo-RL keeps prompts up to max_total_sequence_length and drops one that
    reaches it from the loss; a prompt a token shorter still trains."""
    trainer = _trainer_with_policy(max_prompt_length=6, max_total_sequence_length=6)
    am = torch.ones(2, 7, dtype=torch.long)
    mask = trainer._build_response_mask(am, am, [6, 5])
    assert mask[0].sum() == 0
    assert mask[1].tolist() == [0, 0, 0, 0, 1, 1]

    legacy = _trainer_with_policy(max_total_sequence_length=6)
    assert legacy._build_response_mask(am, am, [6, 5])[0].tolist() == [0] * 5 + [1]
