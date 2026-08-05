"""End-to-end equivalence for the batched-SSM DFLASH verify path.

``test_gdn_target_verify_batched_equiv.py`` compares the two *kernel call
patterns* in isolation. This module drives the real
``SGLangGatedDeltaNet.forward()`` TARGET_VERIFY path instead, once with
``ATOM_ENABLE_GDN_SPEC_VERIFY_BATCHED_SSM=0`` and once with ``=1``, and asserts
that the plugin's whole observable contract is unchanged:

* ``core_attn_out``
* ``intermediate_ssm[req, step]`` -- what SGLang's
  ``fused_mamba_state_scatter_with_mask`` commits from
* ``intermediate_conv_window[0][req, step]``
* the live ``conv`` / ``temporal`` state, which must come back untouched

Run on a GPU; the kernels have no CPU fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from atom.plugin.sglang.attention_backend.attention_gdn import SGLangGatedDeltaNet
from atom.plugin.sglang.runtime import bind_current_forward_batch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GDN kernels require a GPU"
)

# Qwen3.5-397B per-rank GDN shape at TP8.
NUM_K_HEADS = 16
NUM_V_HEADS = 64
TP_SIZE = 8
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
STATE_LEN = CONV_KERNEL - 1

K_DIM = (NUM_K_HEADS // TP_SIZE) * HEAD_K_DIM
V_DIM = (NUM_V_HEADS // TP_SIZE) * HEAD_V_DIM
CONV_DIM = 2 * K_DIM + V_DIM

NUM_SLOTS = 24
LAYER_NUM = 7


class _TargetVerifyMode:
    @staticmethod
    def is_target_verify():
        return True


def _make_layer_cache(bs: int, draft: int, gen: torch.Generator):
    return SimpleNamespace(
        conv=[
            torch.randn(
                NUM_SLOTS,
                CONV_DIM,
                STATE_LEN,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )
        ],
        temporal=torch.randn(
            NUM_SLOTS,
            NUM_V_HEADS // TP_SIZE,
            HEAD_K_DIM,
            HEAD_V_DIM,
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        ),
        intermediate_ssm=torch.zeros(
            NUM_SLOTS,
            draft,
            NUM_V_HEADS // TP_SIZE,
            HEAD_K_DIM,
            HEAD_V_DIM,
            device="cuda",
            dtype=torch.float32,
        ),
        intermediate_conv_window=[
            torch.zeros(
                NUM_SLOTS,
                draft,
                CONV_DIM,
                STATE_LEN,
                device="cuda",
                dtype=torch.bfloat16,
            )
        ],
    )


def _make_impl(gen: torch.Generator) -> SGLangGatedDeltaNet:
    impl = SGLangGatedDeltaNet.__new__(SGLangGatedDeltaNet)
    torch.nn.Module.__init__(impl)
    impl.layer_num = LAYER_NUM
    impl.tp_size = TP_SIZE
    impl.num_k_heads = NUM_K_HEADS
    impl.num_v_heads = NUM_V_HEADS
    impl.head_k_dim = HEAD_K_DIM
    impl.head_v_dim = HEAD_V_DIM
    impl.activation = "silu"
    impl.A_log = torch.randn(
        NUM_V_HEADS // TP_SIZE, device="cuda", dtype=torch.float32, generator=gen
    )
    impl.dt_bias = torch.randn(
        NUM_V_HEADS // TP_SIZE, device="cuda", dtype=torch.float32, generator=gen
    )
    impl.conv1d = SimpleNamespace(
        weight=torch.randn(
            CONV_DIM,
            1,
            CONV_KERNEL,
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        ),
        bias=torch.randn(CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=gen),
    )
    return impl


def _run_once(bs: int, draft: int, batched: bool, monkeypatch):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20260804)

    layer_cache = _make_layer_cache(bs, draft, gen)
    impl = _make_impl(gen)
    # Deliberately unsorted, non-contiguous mamba slots.
    cache_indices = torch.tensor(
        [5, 2, 9, 17, 1, 13][:bs], device="cuda", dtype=torch.int32
    )
    linear_backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(mamba_cache_indices=cache_indices),
        req_to_token_pool=SimpleNamespace(
            mamba2_layer_cache=lambda _layer_id: layer_cache
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=_TargetVerifyMode(),
        spec_info=SimpleNamespace(draft_token_num=draft),
        batch_size=bs,
        attn_backend=SimpleNamespace(linear_attn_backend=linear_backend),
    )

    num_tokens = bs * draft
    mixed_qkv = torch.randn(
        num_tokens, CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=gen
    )
    a = torch.randn(
        num_tokens,
        NUM_V_HEADS // TP_SIZE,
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    )
    b = torch.randn(
        num_tokens,
        NUM_V_HEADS // TP_SIZE,
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    )
    core_attn_out = torch.zeros(
        num_tokens,
        NUM_V_HEADS // TP_SIZE,
        HEAD_V_DIM,
        device="cuda",
        dtype=torch.float32,
    )

    conv_before = layer_cache.conv[0].clone()
    ssm_before = layer_cache.temporal.clone()

    monkeypatch.setenv(
        "ATOM_ENABLE_GDN_SPEC_VERIFY_BATCHED_SSM", "1" if batched else "0"
    )
    with bind_current_forward_batch(forward_batch):
        out = impl.forward(mixed_qkv, b, a, core_attn_out, f"layers.{LAYER_NUM}")

    return {
        "out": out.clone(),
        "intermediate_ssm": layer_cache.intermediate_ssm[:bs].clone(),
        "intermediate_conv": layer_cache.intermediate_conv_window[0][:bs].clone(),
        "conv_live": layer_cache.conv[0],
        "ssm_live": layer_cache.temporal,
        "conv_before": conv_before,
        "ssm_before": ssm_before,
    }


@pytest.mark.parametrize("draft", [4, 8, 16])
@pytest.mark.parametrize("bs", [1, 3])
def test_batched_ssm_matches_stepwise_loop(bs: int, draft: int, monkeypatch):
    loop = _run_once(bs, draft, batched=False, monkeypatch=monkeypatch)
    batched = _run_once(bs, draft, batched=True, monkeypatch=monkeypatch)

    torch.testing.assert_close(batched["out"], loop["out"], rtol=0, atol=0)
    torch.testing.assert_close(
        batched["intermediate_ssm"], loop["intermediate_ssm"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        batched["intermediate_conv"], loop["intermediate_conv"], rtol=0, atol=0
    )


@pytest.mark.parametrize("draft", [4, 8, 16])
@pytest.mark.parametrize("batched", [False, True])
def test_live_state_restored_after_verify(draft: int, batched: bool, monkeypatch):
    """Both paths must leave the live conv/SSM state exactly as they found it;
    SGLang commits accepted steps itself from the intermediate buffers."""
    result = _run_once(3, draft, batched=batched, monkeypatch=monkeypatch)
    torch.testing.assert_close(
        result["conv_live"], result["conv_before"], rtol=0, atol=0
    )
    torch.testing.assert_close(result["ssm_live"], result["ssm_before"], rtol=0, atol=0)


def test_batched_path_is_actually_taken(monkeypatch):
    """Guard against the env flag silently not reaching the plugin: with the
    flag on, the stepwise recurrent call must not be used at all."""
    import atom.plugin.sglang.attention_backend.attention_gdn as gdn_mod

    calls = {"n": 0}
    real = gdn_mod.fused_recurrent_gated_delta_rule

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(gdn_mod, "fused_recurrent_gated_delta_rule", counting)

    _run_once(3, 8, batched=False, monkeypatch=monkeypatch)
    loop_calls = calls["n"]
    calls["n"] = 0
    _run_once(3, 8, batched=True, monkeypatch=monkeypatch)
    batched_calls = calls["n"]

    assert loop_calls == 8, f"stepwise path should call the kernel 8x, got {loop_calls}"
    assert batched_calls == 1, f"batched path should call it once, got {batched_calls}"
