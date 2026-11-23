# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only: the poll-loop reader (``atom.utils.poll_isa``) on disassembly
written as llvm-objdump prints gfx1250 code -- every instruction with its
``// address: words`` comment, branches as word offsets."""

import importlib.util
import pathlib
import sys

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "atom" / "utils" / "poll_isa.py"
_SPEC = importlib.util.spec_from_file_location("poll_isa_under_test", _PATH)
pi = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pi
_SPEC.loader.exec_module(pi)

PAUSE = "s_sleep 16"
BRANCH = {
    "s_branch": 0xBFA0,
    "s_cbranch_scc1": 0xBFA2,
    "s_cbranch_execz": 0xBFA5,
    "s_cbranch_execnz": 0xBFA6,
}


def sys_load(k):
    return f"buffer_load_b128 v[{4 * k}:{4 * k + 3}], v{40 + k}, s[4:7], null offen scope:SCOPE_SYS"


def dev_load(k):
    return f"buffer_load_b128 v[{4 * k}:{4 * k + 3}], v{40 + k}, s[64:67], null offen scope:SCOPE_DEV"


def listing(items, name="k3_test_kernel_pt", base=0x1700):
    """objdump-like text of ``items``: instruction text, ``"NAME:"`` labels,
    ``(branch mnemonic, label)`` branches. Loads and stores take three words."""
    sizes, at, labels = [], base, {}
    for item in items:
        if isinstance(item, str) and item.endswith(":"):
            labels[item[:-1]] = at
            continue
        words = 3 if isinstance(item, str) and "_b128" in item else 1
        sizes.append((item, at, words))
        at += 4 * words
    out = [f"{base:016x} <{name}>:"]
    for item, addr, words in sizes:
        if isinstance(item, tuple):
            mnemonic, label = item
            simm = (labels[label] - addr - 4) // 4
            word = (BRANCH[mnemonic] << 16) | (simm & 0xFFFF)
            out.append(f"\t{mnemonic} {simm & 0xFFFF}    // {addr:012X}: {word:08X}")
        else:
            code = " ".join(["C405C07C"] + ["00000001"] * (words - 1))
            out.append(f"\t{item:<60}// {addr:012X}: {code}")
    return "\n".join(out) + "\n"


def waits(*counts):
    out = []
    for count in counts:
        out += [f"s_wait_loadcnt {count:#x}", "v_cmp_ne_u32_e32 vcc_lo, v9, v5"]
    return out


def together(n=4, loop_n=None, first_n=None):
    loop_n = n if loop_n is None else loop_n
    first_n = n if first_n is None else first_n
    return [
        "global_store_b128 v[8:9], v[2:5], off scope:SCOPE_SYS",
        *[sys_load(k) for k in range(first_n)],
        *waits(*range(first_n - 1, -1, -1)),
        ("s_cbranch_execz", "done"),
        "poll:",
        "s_sleep 16",
        *[sys_load(k) for k in range(loop_n)],
        *waits(*range(loop_n - 1, -1, -1)),
        "s_and_not1_b32 exec_lo, exec_lo, s9",
        ("s_cbranch_execnz", "poll"),
        "done:",
        "s_or_b32 exec_lo, exec_lo, s9",
        "s_endpgm",
    ]


def one(text, pause=PAUSE):
    (report,) = pi.analyze(text, pause)
    return report


def test_loads_together_inside_the_loop_pass():
    report = one(listing(together(4)))
    assert report.hazards == () and report.notes == ()
    assert (report.sys_loads, report.inside, report.outside) == (8, 4, 4)
    (loop,) = report.loops
    assert (loop.loads, loop.waits_between, loop.drained, loop.simple) == (
        4,
        0,
        True,
        True,
    )
    assert report.together
    assert loop.lines[0].endswith("s_sleep 16")
    assert pi.together_hazards(listing(together(8)), PAUSE) == ()


def test_volatile_loads_are_safe_but_not_together():
    items = [
        sys_load(0), "s_wait_loadcnt 0x0", sys_load(1), "s_wait_loadcnt 0x0",
        ("s_cbranch_execz", "done"),
        "poll:", "s_sleep 16",
        sys_load(0), "s_wait_loadcnt 0x0", sys_load(1), "s_wait_loadcnt 0x0",
        ("s_cbranch_execnz", "poll"),
        "done:", "s_endpgm",
    ]  # fmt: skip
    report = one(listing(items, name="k3_test_kernel"))
    assert report.hazards == ()
    assert not report.together and report.loops[0].waits_between == 1
    assert "load waits between" in report.notes[0]


def test_a_hoisted_load_is_a_hazard():
    items = [
        sys_load(0), sys_load(1), *waits(1, 0),
        ("s_cbranch_execz", "done"),
        "poll:", "s_sleep 16", "v_cmp_ne_u32_e32 vcc_lo, v9, v5",
        ("s_cbranch_execnz", "poll"),
        "done:", "s_endpgm",
    ]  # fmt: skip
    hazards = pi.together_hazards(listing(items), PAUSE)
    assert any("loads nothing" in h for h in hazards)
    assert any("no poll loop" in h for h in hazards)


@pytest.mark.parametrize("loop_n, first_n", [(3, 4), (1, 4)])
def test_a_merged_or_sunk_load_is_a_hazard(loop_n, first_n):
    hazards = pi.together_hazards(listing(together(4, loop_n, first_n)), PAUSE)
    assert any("hoisted, merged or sunk" in h for h in hazards), hazards


def test_a_load_kept_once_for_the_first_pass_and_the_loop():
    """As the eight-rank o_proj builds compile: seven first loads, then the
    loop's seven and one load both share, in the block the loop's branch and
    the first pass enter; the waits follow it. Every line is re-read every
    round, back to back."""
    items = [
        *[sys_load(k) for k in range(7)],
        "merge:", "s_or_b32 exec_lo, exec_lo, s0",
        ("s_cbranch_execz", "done"),
        sys_load(7), *waits(*range(7, -1, -1)),
        "s_and_saveexec_b32 s0, s1",
        ("s_cbranch_execz", "merge"),
        "poll:", "s_sleep 16",
        *[sys_load(k) for k in range(7)],
        "s_and_not1_b32 s8, s8, exec_lo",
        ("s_branch", "merge"),
        "done:", "s_endpgm",
    ]  # fmt: skip
    report = one(listing(items))
    (loop,) = report.loops
    assert report.hazards == () and (report.inside, report.outside) == (8, 7)
    assert (loop.loads, loop.waits_between, loop.drained, loop.simple) == (
        8,
        0,
        True,
        True,
    )
    assert report.together
    assert loop.lines[0].endswith("s_sleep 16") and "SCOPE_SYS" in loop.lines[1]


def test_loads_never_awaited_are_a_hazard():
    items = [
        sys_load(0), "s_wait_loadcnt 0x0", ("s_cbranch_execz", "done"),
        "poll:", "s_sleep 16", sys_load(0), "v_cmp_ne_u32_e32 vcc_lo, v9, v5",
        ("s_cbranch_execnz", "poll"),
        "done:", "s_endpgm",
    ]  # fmt: skip
    assert any(
        "no s_wait_loadcnt 0" in h for h in pi.together_hazards(listing(items), PAUSE)
    )


def test_a_message_wait_with_a_load_out_is_a_hazard():
    """The sampling kernel's first design: the realtime clock asked and
    awaited between a poll load and its wait."""
    clock = [
        "s_sendmsg_rtn_b64 s[46:47], sendmsg(MSG_RTN_GET_REALTIME)",
        "s_wait_kmcnt 0x0",
    ]
    out = [
        sys_load(0), "s_wait_loadcnt 0x0", ("s_cbranch_execz", "done"),
        "poll:", "s_sleep 0", sys_load(0), *clock, "s_wait_loadcnt 0x0",
        "v_cmp_ne_u32_e32 vcc_lo, v9, v5", ("s_cbranch_execnz", "poll"),
        "done:", "s_endpgm",
    ]  # fmt: skip
    report = one(listing(out), "s_sleep 0")
    (loop,) = report.loops
    assert loop.waits_else == ("s_sendmsg_rtn_b64", "s_wait_kmcnt") and loop.messages
    assert any("between a load and its s_wait_loadcnt 0" in h for h in report.hazards)
    # the load waited for at once, then the clock: no hazard, the note stays
    settled = [
        sys_load(0), "s_wait_loadcnt 0x0", ("s_cbranch_execz", "done"),
        "poll:", "s_sleep 0", sys_load(0), "s_wait_loadcnt 0x0", *clock,
        "v_cmp_ne_u32_e32 vcc_lo, v9, v5", ("s_cbranch_execnz", "poll"),
        "done:", "s_endpgm",
    ]  # fmt: skip
    report = one(listing(settled), "s_sleep 0")
    assert report.hazards == () and report.loops[0].messages
    assert any("asks for a message" in n for n in report.notes)
    assert report.loops[0].pause_text == "s_sleep 0"
    # the all-reduces' loops ask for nothing
    assert not one(listing(together(4))).loops[0].messages


def test_a_flag_wait_is_no_line_poll():
    flag = [
        dev_load(0), "s_wait_loadcnt 0x0", ("s_cbranch_execz", "seen"),
        "flag:", "s_sleep 1", dev_load(0), "s_wait_loadcnt 0x0",
        ("s_cbranch_execnz", "flag"),
        "seen:",
    ]  # fmt: skip
    report = one(listing(flag + together(4)))
    assert report.hazards == () and len(report.loops) == 1
    # a spin pause over words of another scope alone: no hazard of its own,
    # but no line poll either
    alone = [
        dev_load(0), "s_wait_loadcnt 0x0",
        "poll:", "s_sleep 16", dev_load(0), "s_wait_loadcnt 0x0",
        ("s_cbranch_execnz", "poll"), "s_endpgm",
    ]  # fmt: skip
    assert one(listing(alone)).hazards == ()
    assert pi.together_hazards(listing(alone), PAUSE) == (
        "no poll loop re-reading system-scope lines",
    )


def test_one_copy_of_the_loads_entered_past_the_pause():
    items = [
        ("s_branch", "load"),
        "poll:", "s_sleep 16",
        "load:", sys_load(0), sys_load(1), *waits(1, 0),
        ("s_cbranch_execnz", "poll"),
        "s_endpgm",
    ]  # fmt: skip
    report = one(listing(items))
    assert report.hazards == () and (report.inside, report.outside) == (2, 0)
    assert report.loops[0].loads == 2 and report.together


def test_a_poll_inside_an_outer_loop():
    items = [
        "outer:",
        sys_load(0), sys_load(1), *waits(1, 0),
        ("s_cbranch_execz", "next"),
        "poll:", "s_sleep 16", sys_load(0), sys_load(1), *waits(1, 0),
        ("s_cbranch_execnz", "poll"),
        "next:", "s_add_co_i32 s12, s12, 1",
        ("s_cbranch_scc1", "outer"),
        "s_endpgm",
    ]  # fmt: skip
    report = one(listing(items))
    assert report.hazards == () and (report.inside, report.outside) == (2, 2)


def test_a_load_some_way_round_the_loop_skips_is_a_hazard():
    items = [
        sys_load(0), "s_wait_loadcnt 0x0",
        "poll:", "s_sleep 16",
        ("s_cbranch_scc1", "skip"),
        sys_load(0), "s_wait_loadcnt 0x0",
        "skip:", "v_cmp_ne_u32_e32 vcc_lo, v9, v5",
        ("s_cbranch_execnz", "poll"),
        "s_endpgm",
    ]  # fmt: skip
    report = one(listing(items))
    assert any("can be skipped" in h for h in report.hazards)
    assert report.loops[0].loads == 0 and not report.loops[0].simple


def test_branches_behind_the_loads_leave_them_on_every_way_round():
    """The compare of many lines may branch inside the loop: every load is
    still issued, and drained, on every way round."""
    items = [
        *[sys_load(k) for k in range(8)], *waits(*range(7, -1, -1)),
        ("s_cbranch_execz", "done"),
        "poll:", "s_sleep 16",
        *[sys_load(k) for k in range(8)], *waits(*range(7, -1, -1)),
        ("s_cbranch_scc1", "latch"),
        "s_add_co_i32 s12, s12, 1",
        "latch:", "s_and_not1_b32 exec_lo, exec_lo, s9",
        ("s_cbranch_execnz", "poll"),
        "done:", "s_endpgm",
    ]  # fmt: skip
    report = one(listing(items))
    (loop,) = report.loops
    assert report.hazards == () and report.together
    assert (loop.loads, loop.loop_loads, loop.simple) == (8, 8, False)
    assert (report.inside, report.outside) == (8, 8)
    # the drain on one way only: the other reaches the pause with loads out
    late = [
        sys_load(0), "s_wait_loadcnt 0x0",
        "poll:", "s_sleep 16", sys_load(0),
        ("s_cbranch_scc1", "latch"),
        "s_wait_loadcnt 0x0",
        "latch:", "s_and_not1_b32 exec_lo, exec_lo, s9",
        ("s_cbranch_execnz", "poll"),
        "s_endpgm",
    ]  # fmt: skip
    assert any("no s_wait_loadcnt 0" in h for h in one(listing(late)).hazards)


def test_branch_targets_come_from_the_instruction_words():
    (insns,) = pi.parse(listing(together(2))).values()
    back = next(i for i in insns if i.mnemonic == "s_cbranch_execnz")
    pause = next(i for i in insns if i.mnemonic == "s_sleep")
    assert back.target == pause.addr and back.operand > 0x8000
    with pytest.raises(ValueError, match="no address"):
        pi.parse("0000000000001700 <k>:\n\ts_sleep 16\n")


def test_only_builds_named_together_are_read():
    assert pi.TOGETHER_NAME.search("k3_k2_gfx1250_oproj_ar_tp8_s8_os_ps_pt")
    assert pi.TOGETHER_NAME.search("k3_k2_gfx1250_tail_tp8_s8_d3_t4_id_pt_rs_b9")
    assert not pi.TOGETHER_NAME.search("k3_k2_gfx1250_tail_tp8_s8_d3_t4_id")
    assert not pi.TOGETHER_NAME.search("k3_k2_gfx1250_oproj_ar_tp8_s8_os_ps_profile")
    assert pi.code_object_hazards("k3_k2_gfx1250_oproj_ar_tp8_s8_os", b"junk") == ()
    (hazard,) = pi.code_object_hazards("k3_k2_gfx1250_oproj_ar_tp8_s8_os_pt", b"junk")
    assert hazard.startswith("poll loops unread")


class _Kernel:
    def __init__(self, name):
        self.name = name


def test_a_build_that_does_not_reread_is_never_handed_on():
    asked = []

    def build(together):
        asked.append(together)
        return _Kernel("k3_x_tp8_s8" + ("" if together is False else "_pt"))

    safe = pi.bound_build(build, "x", hazards_of=lambda kernel: ())
    assert safe.name.endswith("_pt") and asked == [None]

    class Log:
        def __init__(self):
            self.said = []

        def warning(self, *args):
            self.said.append(args)

    log = Log()
    asked.clear()
    kept = pi.bound_build(build, "x", log, hazards_of=lambda kernel: ("hoisted",))
    assert kept.name == "k3_x_tp8_s8" and asked == [None, False]
    assert "hoisted" in log.said[0][-1]
    # a switch that is off builds once and compiles nothing here
    asked.clear()
    off = pi.bound_build(
        lambda together: asked.append(together) or _Kernel("k3_x_tp8_s8"),
        "x",
        hazards_of=lambda kernel: pytest.fail("an off build is not read"),
    )
    assert off.name == "k3_x_tp8_s8" and asked == [None]
    with pytest.raises(RuntimeError, match="still polls together"):
        pi.bound_build(
            lambda together: _Kernel("k3_x_pt"), "x", Log(), hazards_of=lambda k: ("h",)
        )
