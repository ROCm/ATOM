# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The STATE tail DeepSeek-V4.1 withholds from vLLM's proxy block pool.

vLLM sizes a KV pool as N uniform blocks and hands every one of them to its
``BlockPool``. V4.1 needs the last few of those blocks for a region vLLM has
no way to describe -- one fixed-size STATE entry per in-flight request -- so
the patch reduces the *block count* while leaving the *tensor size* alone.
Getting that pair backwards either shrinks the allocation the kernels address
or lets the scheduler hand out an id that overlaps the STATE region, and
neither shows up until a request is already running, so the arithmetic is
pinned here.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest

from atom.plugin.vllm.deepseek_v41_bridge import ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME
from atom.plugin.vllm.deepseek_v41_state_reserve_patch import (
    _ATTR,
    _PATCH_TARGETS,
    _reserve_state_tail,
    apply_vllm_v41_state_reserve_patch,
)

OTHER_LAYER = "model.layers.0.self_attn.attn"


def _kv_cache_config(num_blocks, layer_names, tensor_size=1 << 30):
    return SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(layer_names=list(layer_names))],
        kv_cache_tensors=[SimpleNamespace(size=tensor_size)],
    )


def _vllm_config(max_model_len=8192, max_num_seqs=256):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


class TestReserveArithmetic:

    def test_schedulable_blocks_shrink_and_the_allocation_does_not(self):
        # The worker re-derives its own block count from the tensor it got, so
        # the tensor has to keep spanning the tail the STATE region lives in.
        config = _kv_cache_config(10_000, [ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME])
        size_before = config.kv_cache_tensors[0].size

        _reserve_state_tail([config], 900, _vllm_config())

        assert config.num_blocks == 10_000 - 900
        assert config.kv_cache_tensors[0].size == size_before

    def test_configs_without_the_proxy_layer_are_untouched(self):
        # The patch is installed for the whole process; a group that is not
        # ours must come back exactly as it went in. Checked beside one that
        # IS ours, because a reserve with nothing at all to apply it to is a
        # different case and now raises -- see the test below.
        ours = _kv_cache_config(10_000, [ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME])
        theirs = _kv_cache_config(10_000, [OTHER_LAYER])
        _reserve_state_tail([theirs, ours], 900, _vllm_config())
        assert theirs.num_blocks == 10_000
        assert ours.num_blocks == 10_000 - 900

    def test_a_reserve_that_matches_nothing_is_an_error_not_a_no_op(self):
        """Silently skipping leaves the pool full-size and the bind to die.

        The reserve is computed only for V4.1 on the plugin, so a non-zero one
        with no config owning the proxy layer means the layer name moved, not
        that there was nothing to do. Skipping quietly produced a pool built at
        full size and a bind-time failure that named this patch as the cause --
        of a shortfall it had not caused.
        """
        theirs = _kv_cache_config(10_000, [OTHER_LAYER])
        with pytest.raises(RuntimeError) as excinfo:
            _reserve_state_tail([theirs], 900, _vllm_config())
        message = str(excinfo.value)
        assert "900" in message
        # Names what it looked for and what it found, so the next reader does
        # not have to reproduce the miss to learn the layer names.
        assert ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME in message
        assert OTHER_LAYER in message
        assert theirs.num_blocks == 10_000

    def test_a_pool_too_small_for_one_request_fails_with_the_numbers(self):
        # 8192 tokens need 32 PAGEs; 40 blocks minus a 32-block tail leaves 8.
        config = _kv_cache_config(40, [ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME])
        with pytest.raises(ValueError) as excinfo:
            _reserve_state_tail([config], 32, _vllm_config(max_model_len=8192))
        message = str(excinfo.value)
        assert "40 PAGEs" in message
        assert "32 are reserved" in message
        # Actionable: the three knobs that can actually fix it.
        assert "--gpu-memory-utilization" in message
        assert "--max-num-seqs" in message

    def test_exactly_one_max_length_request_still_admits(self):
        """32 PAGEs of history plus the null block vLLM keeps: 33.

        This asserted 32 and passed, which is the off-by-one itself --
        `BlockPool` pops one block as its null placeholder, so 32 block ids
        allocate 31, and an 8191-token prompt needs 32.
        """
        config = _kv_cache_config(65, [ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME])
        _reserve_state_tail([config], 32, _vllm_config(max_model_len=8192))
        assert config.num_blocks == 33

    def test_one_page_short_of_that_is_refused(self):
        """The boundary from the other side: 32 left is one too few, and the
        old check accepted exactly this."""
        config = _kv_cache_config(64, [ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME])
        with pytest.raises(ValueError, match="null block"):
            _reserve_state_tail([config], 32, _vllm_config(max_model_len=8192))


def _recorder():
    """Stands in for vLLM's own ``get_kv_cache_configs``.

    The patch is installed process-wide from the platform hook, so a real
    ``vllm`` may already be wrapped by the time this module runs. Wrapping a
    known-unwrapped callable instead keeps these tests independent of that.
    A function rather than a callable object, because the patch copies
    ``__name__`` and ``__doc__`` off what it wraps.
    """

    def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
        get_kv_cache_configs.calls.append(
            (vllm_config, kv_cache_specs, available_memory)
        )
        return get_kv_cache_configs.result

    get_kv_cache_configs.calls = []
    get_kv_cache_configs.result = []
    return get_kv_cache_configs


@pytest.fixture
def patch_targets(monkeypatch):
    """The vLLM modules the patch rebinds, each holding a fresh recorder."""
    modules = []
    for name in _PATCH_TARGETS:
        try:
            modules.append(importlib.import_module(name))
        except ImportError:  # pragma: no cover - module moved between versions
            continue
    modules = [m for m in modules if hasattr(m, _ATTR)]
    if not modules:
        pytest.skip(f"no vLLM module exposes {_ATTR}")
    recorder = _recorder()
    for module in modules:
        monkeypatch.setattr(module, _ATTR, recorder)
    return SimpleNamespace(modules=modules, recorder=recorder)


def _reserve(monkeypatch, blocks):
    monkeypatch.setattr(
        sys.modules["atom.plugin.vllm.deepseek_v41_state_reserve_patch"],
        "deepseek_v41_state_reserve_blocks",
        lambda _config: blocks,
    )


class TestPatchInstallation:

    def test_every_importer_of_the_name_is_rebound(self, patch_targets):
        # vLLM resolves the function by module attribute at call time, so a
        # module that imported the name keeps calling the original unless it
        # is rebound too.
        assert apply_vllm_v41_state_reserve_patch() is True
        rebound = {id(getattr(m, _ATTR)) for m in patch_targets.modules}
        assert len(rebound) == 1
        assert getattr(
            getattr(patch_targets.modules[0], _ATTR),
            "_atom_v41_state_reserve_patched",
            False,
        )

    def test_applying_twice_wraps_once(self, patch_targets):
        assert apply_vllm_v41_state_reserve_patch() is True
        once = getattr(patch_targets.modules[0], _ATTR)
        assert apply_vllm_v41_state_reserve_patch() is False
        assert getattr(patch_targets.modules[0], _ATTR) is once

    def test_the_wrapper_reserves_the_tail_and_returns_vllms_own_configs(
        self, patch_targets, monkeypatch
    ):
        configs = [_kv_cache_config(10_000, [ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME])]
        patch_targets.recorder.result = configs
        assert apply_vllm_v41_state_reserve_patch() is True
        wrapped = getattr(patch_targets.modules[0], _ATTR)

        vllm_config = _vllm_config()
        _reserve(monkeypatch, 900)
        assert wrapped(vllm_config, {}, 1 << 40) is configs
        assert patch_targets.recorder.calls == [(vllm_config, {}, 1 << 40)]
        assert configs[0].num_blocks == 10_000 - 900

    def test_a_model_with_no_reserve_passes_straight_through(
        self, patch_targets, monkeypatch
    ):
        configs = [_kv_cache_config(10_000, [OTHER_LAYER])]
        patch_targets.recorder.result = configs
        assert apply_vllm_v41_state_reserve_patch() is True
        wrapped = getattr(patch_targets.modules[0], _ATTR)

        _reserve(monkeypatch, 0)
        assert wrapped(_vllm_config(), {}, 1 << 40) is configs
        assert configs[0].num_blocks == 10_000


class TestInstallSite:
    """Where the patch is installed from, which is the whole ballgame.

    The patch has to be in place before ``get_kv_cache_configs`` runs in the
    EngineCore process. ``ATOMPlatform.check_and_update_config`` looks like the
    right home for that and is not: vLLM resolves its platform class from
    inside its own ``import vllm``, so ``register_platform`` can raise on a
    half-built ``vllm`` package, the loader swallows it, and the stock ROCm
    platform stays active with nothing to show for it but a pool that is one
    STATE tail too small at bind time. ``register_model`` -- the
    ``vllm.general_plugins`` hook -- is reached in that process, so that is the
    call site this pins.
    """

    def test_register_model_installs_the_state_reserve_patch(self):
        """Checked on the wiring, because the call happens at most once.

        ``register_model`` is idempotent, so watching for the call answers
        whether some earlier test in the session already made it -- a property
        of the suite, not of the code. This test did exactly that and failed
        only in the full run, which is the tell. The patch's own behaviour is
        covered by ``TestPatchInstallation`` above.
        """
        import inspect

        register = importlib.import_module("atom.plugin.vllm.register")
        assert "apply_vllm_v41_state_reserve_patch()" in inspect.getsource(
            register.register_model
        ), (
            "register_model must install the V4.1 STATE-tail patch; the "
            "platform hook is not guaranteed to run"
        )


class TestLocalHitCapInstallSite:
    """The local-hit cap has to be installed where V4.1's other patch is.

    The platform hook it would naturally live in is not always activated:
    measured on a V4.1 serve, `ATOMPlatform.check_and_update_config` ran zero
    times while the model wrapper's own hook ran four. A scheduler chosen
    there is a scheduler never chosen, and the symptom is prefix caching
    silently going uncapped -- which is a dead engine on the first deep local
    hit, not a slow one.
    """

    def test_register_model_wires_up_both_v41_scheduler_patches(self):
        """Checked on the wiring, not on a call and not on the end state.

        `register_model` is idempotent and this suite's other install-site
        test replaces the installer before calling it, which consumes that one
        shot -- so both "did it call?" and "is it installed?" answer a
        question about the suite rather than about the code. What has to stay
        true is that `register_model` is where these are installed, because
        the platform hook that would otherwise carry them does not reliably
        run. The patches' own behaviour is covered in
        tests/test_vllm_v41_prefill_alignment.py.
        """
        import inspect

        register = importlib.import_module("atom.plugin.vllm.register")
        source = inspect.getsource(register.register_model)
        for installer in (
            "apply_vllm_v41_local_hit_cap_patch",
            "apply_vllm_v41_prefill_alignment_patch",
        ):
            assert f"{installer}()" in source, (
                f"register_model must install {installer}; the platform hook "
                "is not guaranteed to run"
            )


class TestHfOverridesAreRefused:
    """`--hf-overrides` cannot reach ATOM's V4.1 schema, so it is refused.

    Measured through vLLM's own `ModelConfig`, which is what a worker gets:
    the resolved config carries `text_config=None` and no `vision_config`,
    and `normalize_hf_config` is built from exactly those two. So the bridge
    reads the checkpoint, and an override applied to vLLM's object is not in
    that file. Refusing says so; the alternative is a server that starts and
    answers as though the flag had not been passed.
    """

    @staticmethod
    def _config(overrides):
        from types import SimpleNamespace

        return SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["DeepseekV41ForCausalLM"],
                hf_overrides=overrides,
                model="/models/DeepSeek-V4.1-Flash",
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            speculative_config=None,
            kv_transfer_config=None,
        )

    def test_an_override_is_refused_by_name(self):
        from atom.plugin.vllm.platform import _refuse_v41_hf_overrides

        with pytest.raises(ValueError, match="hf-overrides"):
            _refuse_v41_hf_overrides(
                self._config({"text_config": {"rms_norm_eps": 1e-5}})
            )

    def test_no_override_passes(self):
        from atom.plugin.vllm.platform import _refuse_v41_hf_overrides

        for empty in (None, {}, ()):
            _refuse_v41_hf_overrides(self._config(empty))


class TestDsparkDraftKvDtype:
    """DSpark must name its draft's KV dtype, and is told so by name.

    vLLM's V4.1 draft attention takes only a packed `*_ds_mla` layout. Given
    `auto` it resolves to `fp8_ds_mla` and writes that back onto whatever
    `cache_config` it was handed -- the target's, unless the speculative
    config overrode it for the draft alone. ATOM has no fp8 V4.1 pool, so the
    run then died naming a dtype the operator never typed, two layers from
    the flag that fixes it. Reported by a sibling session that hit exactly
    this while trying to measure something else.
    """

    @staticmethod
    def _config(spec_kv):
        from types import SimpleNamespace

        return SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=["DeepseekV41ForCausalLM"], hf_overrides=None
            ),
            speculative_config=SimpleNamespace(method="dspark", kv_cache_dtype=spec_kv),
        )

    def test_a_dspark_config_without_one_is_refused_by_flag_name(self):
        from atom.plugin.vllm.platform import _require_v41_draft_kv_cache_dtype

        with pytest.raises(ValueError, match="kv_cache_dtype") as excinfo:
            _require_v41_draft_kv_cache_dtype(self._config(None))
        message = str(excinfo.value)
        # Actionable: the flag, the value, and that the target's is separate.
        assert "--speculative-config" in message
        assert "fp8_ds_mla" in message
        assert "--kv-cache-dtype" in message

    def test_naming_it_passes(self):
        from atom.plugin.vllm.platform import _require_v41_draft_kv_cache_dtype

        _require_v41_draft_kv_cache_dtype(self._config("fp8_ds_mla"))

    def test_no_speculation_is_unaffected(self):
        from types import SimpleNamespace

        from atom.plugin.vllm.platform import _require_v41_draft_kv_cache_dtype

        _require_v41_draft_kv_cache_dtype(
            SimpleNamespace(speculative_config=None, model_config=None)
        )
