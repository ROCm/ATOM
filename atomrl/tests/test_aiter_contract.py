import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LUMEN_ROOT = REPO_ROOT / "atomrl" / "atomrl" / "lumen"


def test_lumen_does_not_probe_aiter_symbols_at_runtime():
    offenders = []
    for path in LUMEN_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if (
            "_probe_aiter_" in text
            or re.search(r"try:\s*\n\s*(?:from|import) aiter", text)
            or "extensions_created = True" in text
        ):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert not offenders, "runtime AITER probing remains in: " + ", ".join(offenders)


def test_dapo_does_not_install_an_aiter_import_hook():
    shim = (
        REPO_ROOT
        / "atomrl"
        / "examples"
        / "DAPO"
        / "atom_aiter_shim"
        / "sitecustomize.py"
    )
    assert not shim.exists()


def test_layernorm_uses_the_supported_aiter_call_contract():
    path = LUMEN_ROOT / "modules" / "layernorm_linear.py"
    text = path.read_text(encoding="utf-8")

    assert "inspect.signature" not in text
    assert '"output_amax"' not in text
    assert '"output_rsigma"' not in text
    assert "out_fp8, out_bf16, _, res_out =" in text
    assert "torch.rsqrt(" in text


def test_mxfp4_shuffle_is_owned_by_aiter():
    path = LUMEN_ROOT / "ops" / "quantize" / "linear.py"
    text = path.read_text(encoding="utf-8")

    assert "from atomrl.lumen.ops.quantize.ops import swizzle_mxfp4_scale" not in text
    assert "_MXFP4_WIDE_SHUFFLE_MIN_BYTES" not in text
    assert "wide.permute" not in text
    assert "shuffle_scale_gemm(" in text
    assert "shuffle_weight(" in text


def test_lumen_skills_require_public_aiter_apis():
    skill_roots = (
        REPO_ROOT / "atomrl" / "lumen" / ".cursor" / "skills" / "lumen-coding",
        REPO_ROOT / "atomrl" / "lumen" / ".claude" / "skills" / "lumen-coding",
    )
    for root in skill_roots:
        skill = (root / "SKILL.md").read_text(encoding="utf-8")
        dispatch = (root / "aiter-dispatch.md").read_text(encoding="utf-8")
        combined = skill + dispatch
        assert "public API directly" in combined
        assert "Do not reimplement AITER kernels" in combined
        assert "MXFP4 weight/scale shuffle" in combined
