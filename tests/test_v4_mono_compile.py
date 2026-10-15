# SPDX-License-Identifier: MIT
from atom.benchmarks.v4_mono_compile import production_cases


def test_production_compile_grid_covers_each_admitted_profile_and_index_phase():
    cases = list(production_cases())
    assert len(cases) == len(set(cases)) == 19
    assert {(case.profile, case.rows, case.tokens_per_seq) for case in cases} == {
        ("ordinary", 1, 1),
        ("ordinary", 4, 1),
        ("ordinary", 8, 1),
        ("mtp3_target", 4, 4),
    }
    for case in cases:
        assert case.compress_ratio in (4, 128)
        assert case.phase in (("one",) if case.rows == 1 else ("attention", "moe"))
        assert case.refresh or (case.profile == "ordinary" and case.compress_ratio == 4)


def test_ci_compiles_production_geometry_not_just_the_build_key_probe():
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/atom-test.yaml"
    ).read_text()
    assert "python -m atom.benchmarks.v4_mono_compile" in workflow
    assert "v4-mono-compile.json" in workflow
