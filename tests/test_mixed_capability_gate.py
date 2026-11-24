# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Unsupported mixed configurations are rejected before model loading.

Config._validate_mixed_prefill_decode owns the capability table. Its refusals
must remain effective under python -O, without duplicate per-layer checks.
"""

import ast
import pathlib
from types import MethodType, SimpleNamespace

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _mixed_refusals():
    """Every assert/raise in atom/ whose message is about mixed batching."""
    out = []
    for p in sorted((_ROOT / "atom").rglob("*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        if "mixed" not in src.lower():
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, (ast.Assert, ast.Raise)):
                t = ast.unparse(n).lower()
                if "mixed" in t and (
                    "enable-mixed" in t or "is_mixed" in t or "mixed prefill" in t
                ):
                    rel = p.relative_to(_ROOT)
                    out.append((isinstance(n, ast.Assert), f"{rel}:{n.lineno}"))
    return out


def test_no_mixed_refusal_is_an_assert():
    """An `assert` here is a refusal that disappears under `python -O`."""
    asserts = [where for is_assert, where in _mixed_refusals() if is_assert]
    assert not asserts, (
        "mixed-batch refusals that vanish under `python -O`, leaving the "
        f"unsupported path to run with wrong numerics: {asserts}"
    )


def test_the_config_table_exists_and_is_called():
    cfg = (_ROOT / "atom/config.py").read_text(encoding="utf-8")
    tree = ast.parse(cfg)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_validate_mixed_prefill_decode" in names

    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "__post_init__":
            body = ast.unparse(n)
            if "_validate_mixed_prefill_decode()" in body:
                return
    pytest.fail(
        "_validate_mixed_prefill_decode is defined but never called from a "
        "__post_init__; a table nobody runs refuses nothing"
    )


def test_the_table_returns_early_when_the_flag_is_off():
    """Cost and blast radius are both zero for everyone not using the flag."""
    cfg = (_ROOT / "atom/config.py").read_text(encoding="utf-8")
    tree = ast.parse(cfg)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == (
            "_validate_mixed_prefill_decode"
        ):
            first = ast.unparse(n.body[1] if len(n.body) > 1 else n.body[0])
            assert "enable_mixed_prefill_decode" in first and "return" in first, (
                "the table must short-circuit when the flag is off; it runs in "
                f"every Config.__post_init__. Got: {first[:90]}"
            )
            return
    pytest.fail("no _validate_mixed_prefill_decode")


# ── behavioural: the checks above are source-level and let a real bug through ──
#
# `_validate_mixed_prefill_decode` shipped reading `self.hf_text_config`, which
# is a ModelRunner attribute and not a Config one. Every source-level test
# passed; every launch with the flag on died with AttributeError inside the
# validator. These call it.


class _FakeHF:
    """Minimal HF text config. MLA is 'has a latent rank' (see selector); set
    `model_type` to pick a more specific family."""

    def __init__(self, *, kv_lora_rank=512, index_topk=None):
        self.kv_lora_rank = kv_lora_rank
        self.num_attention_heads = 16
        self.num_hidden_layers = 2
        if index_topk is not None:
            self.index_topk = index_topk


class _FakeConfig:
    """Just enough surface for the validator, bound to the real function."""

    _validate_mixed_prefill_decode = None  # bound below

    def __init__(
        self,
        *,
        enabled=True,
        hf=None,
        spec=None,
        pp=1,
        max_num_seqs=256,
        max_num_batched_tokens=16384,
        cudagraph_mode=None,
        chunked_prefill=True,
        pcp=1,
        dcp=1,
        eplb=False,
        tbo=False,
        tbo_decode=False,
        rapidserve=False,
        kv_transfer_config=None,
    ):
        self.eplb_enable = eplb
        self.enable_tbo = tbo
        self.enable_tbo_decode = tbo_decode
        self.enable_rapidserve = rapidserve
        self.kv_transfer_config = kv_transfer_config
        self.enable_mixed_prefill_decode = enabled
        self.hf_config = hf if hf is not None else _v4_hf()
        self.speculative_config = spec
        self.pipeline_parallel_size = pp
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        self.enable_chunked_prefill = chunked_prefill
        self.prefill_context_parallel_size = pcp
        self.decode_context_parallel_size = dcp


def _v4_hf():
    hf = _FakeHF(kv_lora_rank=512)
    hf.model_type = "deepseek_v4"
    return hf


def _spec(method="dspark", k=7):
    # Not a real SpeculativeConfig: its __post_init__ fetches the draft's HF
    # config. Bind its real capability predicate without loading weights.
    from atom.config import SpeculativeConfig

    spec = SimpleNamespace(
        method=method, num_speculative_tokens=k, draft_model_hf_config=SimpleNamespace()
    )
    spec.use_dspark = MethodType(SpeculativeConfig.use_dspark, spec)
    return spec


def _bind():
    from atom.config import Config

    _FakeConfig._validate_mixed_prefill_decode = Config._validate_mixed_prefill_decode


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("spec", [None, _spec()])
def test_inactive_lossless_flag_does_not_refuse_mixed(monkeypatch, enabled, spec):
    _bind()
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1")
    _FakeConfig(enabled=enabled, spec=spec)._validate_mixed_prefill_decode()


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_mixed_refuses_only_a_lossless_draft_that_would_arm(
    monkeypatch, mixed, supported
):
    from aiter_stub import atom_without_aiter

    with atom_without_aiter() as load:
        module = load("atom.spec_decode.dspark_lossless")
        monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1")
        monkeypatch.setenv("ATOM_ENABLE_RELAXED_MTP", "0")
        monkeypatch.setattr(module, "lossless_unsupported_reason", lambda _: None)
        monkeypatch.setattr(module, "require_ell_from_older_drafts", lambda: None)
        allocations = []
        monkeypatch.setattr(
            module.DSparkLossless,
            "__init__",
            lambda self, drafter: allocations.append(drafter),
        )
        drafter = SimpleNamespace(
            model=SimpleNamespace(supports_lossless_draft=supported),
            config=SimpleNamespace(enable_mixed_prefill_decode=mixed),
            speculative_config=SimpleNamespace(synthetic_acceptance_rates=None),
            runner=SimpleNamespace(rank=1),
        )
        if mixed and supported:
            with pytest.raises(ValueError, match="lossless.*mixed"):
                module.DSparkLossless.arm(drafter)
            assert not allocations
        else:
            result = module.DSparkLossless.arm(drafter)
            assert (result is not None) == supported
            assert allocations == ([drafter] if supported else [])


def test_validator_runs_against_a_real_config_surface():
    """The regression: it read an attribute Config does not have."""
    _bind()
    # V4, no spec -> supported, must not raise (and must not AttributeError
    # on the way to deciding that).
    _FakeConfig()._validate_mixed_prefill_decode()


def test_validator_allows_v4_dspark():
    """The one speculative combination the mixed split is wired for."""
    _bind()
    _FakeConfig(hf=_v4_hf(), spec=_spec())._validate_mixed_prefill_decode()


def _wrap_connectors(entry, depth):
    for _ in range(depth):
        entry = {"kv_connector": "multi", "connectors": [entry]}
    return entry


@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize(
    "entry,pd_producer,disables_spec",
    [
        ({"kv_role": "kv_producer"}, True, True),
        ({"kv_connector": "moriio"}, True, False),
        ({"kv_connector": " MoonCake "}, True, False),
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, True, True),
        ({"kv_connector": "mooncake", "kv_role": "kv_consumer"}, False, False),
        ({"kv_connector": "LMCacheMPConnector", "kv_role": "kv_producer"}, False, True),
        ({"kv_connector": "mooncake_store", "kv_role": "kv_producer"}, False, True),
        ({"kv_connector": "lmcache_mp", "kv_role": "kv_both"}, False, False),
        ({"kv_connector": "lmcache_mp"}, False, False),
        ({"kv_connector": "mooncake_store", "kv_role": "kv_consumer"}, False, False),
    ],
)
def test_mixed_producer_refusal_matches_the_execution_mode(
    depth, speculative, entry, pd_producer, disables_spec
):
    _bind()
    cfg = _FakeConfig(
        kv_transfer_config=_wrap_connectors(entry, depth),
        spec=_spec() if speculative else None,
    )
    if pd_producer or (speculative and disables_spec):
        with pytest.raises(ValueError, match="kv_producer"):
            cfg._validate_mixed_prefill_decode()
    else:
        cfg._validate_mixed_prefill_decode()
    cfg.enable_mixed_prefill_decode = False
    cfg._validate_mixed_prefill_decode()


@pytest.mark.parametrize("name", ["LMCacheMPConnector", "mooncake_store"])
@pytest.mark.parametrize("depth", [0, 1])
def test_non_spec_offload_producer_has_matching_profile_and_mixed_step(
    monkeypatch, name, depth
):
    from conftest import MockConfig

    from atom.model_engine.mixed_profile import _profile_spec_tokens
    from atom.model_engine.scheduler import Scheduler
    from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType

    transfer = _wrap_connectors({"kv_connector": name, "kv_role": "kv_producer"}, depth)
    _bind()
    _FakeConfig(kv_transfer_config=transfer)._validate_mixed_prefill_decode()
    # Test scheduling and profile width, not connector IO or backend compatibility.
    monkeypatch.setattr("atom.utils.forward_context.get_kvconnector", lambda *a: None)
    scheduler = Scheduler(
        MockConfig(kv_transfer_config=transfer, enable_mixed_prefill_decode=True)
    )
    decode = Sequence([10] * 4, block_size=4)
    assert scheduler.block_manager.allocate(decode, 0)
    decode.num_cached_tokens = 4
    decode.append_token(11)
    decode.type = SequenceType.DECODE
    decode.status = SequenceStatus.RUNNING
    scheduler.running.append(decode)
    scheduler.add(Sequence([20] * 4, block_size=4))
    batch, _ = scheduler.schedule()
    assert batch.is_mixed and batch.num_spec_step == 0
    assert scheduler.mtp_k == _profile_spec_tokens(SimpleNamespace()) == 0


@pytest.mark.parametrize("speculative", [False, True])
def test_pd_consumer_with_save_only_offload_uses_speculation_boundary(speculative):
    _bind()
    transfer = {
        "kv_connector": "multi",
        "connectors": [
            {"kv_connector": "mooncake", "kv_role": "kv_consumer"},
            {"kv_connector": "lmcache_mp", "kv_role": "kv_producer"},
        ],
    }
    cfg = _FakeConfig(
        kv_transfer_config=transfer, spec=_spec() if speculative else None
    )
    if speculative:
        with pytest.raises(ValueError, match="speculative decoding with kv_producer"):
            cfg._validate_mixed_prefill_decode()
    else:
        cfg._validate_mixed_prefill_decode()


def test_mixed_allows_embedded_dspark_under_mtp_method():
    _bind()
    spec = _spec("mtp")
    spec.draft_model_hf_config.dspark_block_size = 8
    assert spec.use_dspark()
    _FakeConfig(hf=_v4_hf(), spec=spec)._validate_mixed_prefill_decode()


def test_mixed_refuses_rapidserve_before_model_loading():
    _bind()
    cfg = _FakeConfig(rapidserve=True)
    with pytest.raises(ValueError, match="RapidServe"):
        cfg._validate_mixed_prefill_decode()
    cfg.enable_mixed_prefill_decode = False
    cfg._validate_mixed_prefill_decode()


@pytest.mark.parametrize("tbo,tbo_decode", [(True, False), (False, True), (True, True)])
def test_validator_refuses_mixed_with_either_tbo_mode(tbo, tbo_decode):
    _bind()
    cfg = _FakeConfig(tbo=tbo, tbo_decode=tbo_decode)
    with pytest.raises(ValueError, match="TBO"):
        cfg._validate_mixed_prefill_decode()
    cfg.enable_mixed_prefill_decode = False
    cfg._validate_mixed_prefill_decode()


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_validator_refuses_other_spec_methods(method):
    _bind()
    cfg = _FakeConfig(hf=_v4_hf(), spec=_spec(method))
    with pytest.raises(ValueError, match=f"speculative decoding via '{method}'"):
        cfg._validate_mixed_prefill_decode()


def test_validator_refuses_spec_under_pipeline_parallel():
    _bind()
    cfg = _FakeConfig(hf=_v4_hf(), spec=_spec(), pp=2)
    with pytest.raises(ValueError, match="pipeline parallel"):
        cfg._validate_mixed_prefill_decode()


def test_validator_warns_when_a_full_decode_set_eats_the_step(caplog):
    """256 seqs x (7+1) = 2048 reserved: all of a 2048-token step."""
    _bind()
    cfg = _FakeConfig(
        hf=_v4_hf(), spec=_spec(k=7), max_num_seqs=256, max_num_batched_tokens=2048
    )
    with caplog.at_level("WARNING", logger="atom"):
        cfg._validate_mixed_prefill_decode()
    assert "at least 1/8" in caplog.text
    assert "FCFS tail may wait" in caplog.text
    assert "leaving prefill no budget" not in caplog.text


@pytest.mark.parametrize("rows, warns", [(224, False), (225, True)])
def test_validator_warning_respects_prefill_floor(caplog, rows, warns):
    _bind()
    cfg = _FakeConfig(spec=_spec(k=7), max_num_seqs=rows, max_num_batched_tokens=2048)
    with caplog.at_level("WARNING", logger="atom"):
        cfg._validate_mixed_prefill_decode()
    assert ("FCFS tail may wait" in caplog.text) is warns


def test_validator_is_quiet_when_prefill_keeps_budget(caplog):
    _bind()
    cfg = _FakeConfig(hf=_v4_hf(), spec=_spec(k=7), max_num_batched_tokens=16384)
    with caplog.at_level("WARNING", logger="atom"):
        cfg._validate_mixed_prefill_decode()
    assert "leaving prefill no budget" not in caplog.text


# ---- what the mixed split does not cover (PR #2037 review) ----------------


def test_validator_refuses_unchunked_prefill():
    """Unchunked admission takes a prompt whole, past the decode reserve."""
    _bind()
    cfg = _FakeConfig(chunked_prefill=False)
    with pytest.raises(ValueError, match="chunked prefill off"):
        cfg._validate_mixed_prefill_decode()


def test_validator_refuses_mrope_even_on_v4():
    """Same predicate as the runner's `use_mrope`: the V4 family check alone
    would admit a V4 config that carried an `mrope_section`."""
    _bind()
    hf = _v4_hf()
    hf.rope_parameters = {"mrope_section": [16, 24, 24]}
    with pytest.raises(ValueError, match="M-RoPE"):
        _FakeConfig(hf=hf)._validate_mixed_prefill_decode()


def test_validator_refuses_pcp():
    """PCP's round-robin query split covers the whole batch, decode rows too."""
    _bind()
    with pytest.raises(ValueError, match="prefill context parallel"):
        _FakeConfig(pcp=2)._validate_mixed_prefill_decode()


def test_validator_refuses_dcp():
    """Mixed batching and context parallelism are mutually exclusive."""
    _bind()
    with pytest.raises(ValueError, match="decode context parallel"):
        _FakeConfig(dcp=2)._validate_mixed_prefill_decode()


def test_context_parallel_is_allowed_when_mixed_is_disabled():
    _bind()
    _FakeConfig(enabled=False, pcp=2, dcp=2)._validate_mixed_prefill_decode()


# Every member of the taxonomy, named. The first version of this table tested
# only MHA, passed, and shipped a validator that refused DeepSeek-V4 -- the one
# model the feature exists for. A family list that does not enumerate itself
# lets exactly that through.
_FAMILY_CASES = [
    # (model_type, kv_lora_rank, supported?)
    ("deepseek_v4", 512, True),  # the feature's reason to exist
    ("deepseek_v3", 512, False),  # dense MLA; mixed is V4-only
    ("kimi_linear", 512, False),  # hybrid; no prepare_mixed
    ("qwen3_next", None, False),  # GDN
    ("llama", None, False),  # MHA
]


@pytest.mark.parametrize("model_type,rank,supported", _FAMILY_CASES)
def test_every_attention_family_is_classified_deliberately(model_type, rank, supported):
    _bind()
    hf = _FakeHF(kv_lora_rank=rank)
    hf.model_type = model_type
    cfg = _FakeConfig(hf=hf)
    if supported:
        cfg._validate_mixed_prefill_decode()  # must not raise
    else:
        with pytest.raises(ValueError, match="attention family"):
            cfg._validate_mixed_prefill_decode()


def test_v4_is_not_refused_as_non_mla():
    """The regression, pinned on its own.

    `Family.is_mla` is `self in (MLA, KIMI_MLA)` and V4 is deliberately absent
    from it -- so `not is_mla` reads as "refuse V4". The validator must key off
    the V4 family, not off `is_mla`.
    """
    _bind()
    from atom.utils.selector import Family

    assert not Family.V4.is_mla, (
        "premise changed: if V4 now reports is_mla, re-read the validator -- "
        "this test is guarding against an inference that would become valid"
    )
    hf = _FakeHF(kv_lora_rank=512)
    hf.model_type = "deepseek_v4"
    _FakeConfig(hf=hf)._validate_mixed_prefill_decode()


def test_validator_is_inert_when_the_flag_is_off():
    """Runs in every Config.__post_init__; must cost and refuse nothing."""
    _bind()
    cfg = _FakeConfig(enabled=False, hf=_FakeHF(index_topk=2048), spec=object())
    cfg._validate_mixed_prefill_decode()  # would refuse twice over if it ran


def test_all_reasons_are_reported_not_just_the_first():
    """A user fixing one conflict should not discover the next on relaunch."""
    _bind()
    cfg = _FakeConfig(spec=_spec("mtp"), chunked_prefill=False)
    with pytest.raises(ValueError) as e:
        cfg._validate_mixed_prefill_decode()
    assert "speculative decoding" in str(e.value)
    assert "chunked prefill off" in str(e.value)


@pytest.mark.parametrize(
    "mode,refused",
    [
        ("FULL", False),  # the CLI default
        ("NONE", False),
        ("FULL_DECODE_ONLY", False),
        ("TR_HYBRID", True),
        ("FULL_AND_TR_HYBRID", True),
    ],
)
def test_v4_refuses_every_tr_hybrid_mode(mode, refused):
    """Keyed off `uses_tr_hybrid`, the predicate V4's forward routes on -- not
    off one enum member."""
    _bind()
    from atom.config import CUDAGraphMode

    cfg = _FakeConfig(hf=_v4_hf(), cudagraph_mode=CUDAGraphMode[mode])
    if refused:
        with pytest.raises(ValueError, match=f"--cudagraph-mode {mode} on DeepSeek-V4"):
            cfg._validate_mixed_prefill_decode()
    else:
        cfg._validate_mixed_prefill_decode()


def test_mixed_rejects_eplb_until_loads_can_exclude_decode_rows():
    _bind()
    with pytest.raises(ValueError, match="EPLB"):
        _FakeConfig(eplb=True)._validate_mixed_prefill_decode()
    _FakeConfig(enabled=False, eplb=True)._validate_mixed_prefill_decode()
