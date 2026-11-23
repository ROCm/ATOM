# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Parse real MLIR assembly with FlyDSL, without compiling or launching a kernel."""

from types import SimpleNamespace

import pytest

from atom.mono.runtime.compile import _code_object, kernel_names, scratch_bytes

pytest.importorskip("flydsl")


@pytest.fixture(params=["bin", "fatbin"])
def object_format(request):
    return request.param


def _object(data, object_format):
    # Binary is printed as `bin = "..."`; default Fatbin omits the label.
    prefix = "bin = " if object_format == "bin" else ""
    escaped = "".join(f"\\{byte:02X}" for byte in data)
    return f'#gpu.object<#rocdl.target<chip = "gfx950">, {prefix}"{escaped}">'


def _module(*objects):
    return "module { gpu.binary @kernels [" + ", ".join(objects) + "] }"


def _launcher(text):
    return SimpleNamespace(_mem_cache={"key": SimpleNamespace(_ir_text=text)})


def test_code_object_preserves_all_bytes(object_format):
    elf = b"\x7fELF" + bytes(range(256))
    assert _code_object(_module(_object(elf, object_format))) == elf


@pytest.mark.parametrize(
    "value, size",
    [(b"\x00", 0), (b"\x24", 36), (b"\xcc\x90", 144), (b"\xcd\x14\x8c", 5260)],
)
def test_scratch_bytes_uses_parsed_code_object(object_format, value, size):
    elf = b"\x7fELF\x9b.private_segment_fixed_size" + value
    assert scratch_bytes(_launcher(_module(_object(elf, object_format)))) == size


def test_kernel_names_uses_parsed_code_object(object_format):
    elf = b"\x7fELF\xa7.symbol\xaaexample.kd"
    assert kernel_names(_launcher(_module(_object(elf, object_format)))) == ["example"]


def test_first_object_matches_flydsl_jit_selection(object_format):
    first, other = b"\x7fELFfirst", b"\x7fELFother"
    text = _module(_object(first, object_format), _object(other, object_format))
    assert _code_object(text) == first


def test_missing_binary_is_rejected():
    with pytest.raises(RuntimeError, match="no gpu.binary"):
        _code_object("module {}")


@pytest.mark.parametrize("data", [b"", b"not an ELF"])
def test_non_elf_object_is_rejected(object_format, data):
    with pytest.raises(RuntimeError, match="does not contain an ELF"):
        _code_object(_module(_object(data, object_format)))


def test_unreadable_first_object_is_not_replaced_by_another(object_format):
    text = _module(
        _object(b"invalid", object_format), _object(b"\x7fELF", object_format)
    )
    with pytest.raises(RuntimeError, match="does not contain an ELF"):
        _code_object(text)


def test_multiple_binary_modules_are_not_silently_mixed(object_format):
    obj = _object(b"\x7fELF", object_format)
    text = f"module {{ gpu.binary @a [{obj}] gpu.binary @b [{obj}] }}"
    with pytest.raises(RuntimeError, match="multiple gpu.binary"):
        _code_object(text)


def test_missing_scratch_metadata_is_not_zero(object_format):
    launcher = _launcher(_module(_object(b"\x7fELF", object_format)))
    with pytest.raises(RuntimeError, match="no .private_segment_fixed_size"):
        scratch_bytes(launcher)


def test_unrelated_bin_text_is_not_a_code_object(object_format):
    elf = b"\x7fELFactual"
    text = (
        r'module attributes {test.note = "bin = \22decoy\22"} {'
        + f" gpu.binary @kernels [{_object(elf, object_format)}] }}"
    )
    assert _code_object(text) == elf
