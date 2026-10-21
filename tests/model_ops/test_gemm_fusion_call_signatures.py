# SPDX-License-Identifier: MIT
# Call-signature audit for atom.model_ops.gemm_fusion, by ast + inspect.signature.
#
# Why: the runtime cleanup removed the ``device`` parameter of ``before_graph_capture`` while
# ModelRunner still passed ``device=self.device``; every rank died with a TypeError on the GPU and no
# unit test noticed, because the call site lives in a 5000-line file the tests never execute.
#
# What: parse the SOURCE of every atom module (nothing is imported, so it runs on any machine, with or
# without torch), find every call that crosses a module boundary into atom, resolve the callee to its
# ``def`` (following ``from .x import y`` re-exports), build the real ``inspect.signature`` of that
# def, and ``bind`` the call's positional count and keyword names to it.
import ast
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = "atom.model_ops.gemm_fusion"
PKG_DIR = ROOT / "atom" / "model_ops" / "gemm_fusion"

_TREES = {}


def _module_file(mod):
    base = ROOT.joinpath(*mod.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _tree(path):
    if path not in _TREES:
        _TREES[path] = ast.parse(path.read_text(encoding="utf-8"))
    return _TREES[path]


def _file_module(path):
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts), path.name == "__init__.py"


def _absolute(base_mod, is_pkg, level, module):
    """Dotted target of ``from <level dots><module> import ...`` written in ``base_mod``."""
    if level == 0:
        return module
    pkg = base_mod.split(".") if is_pkg else base_mod.split(".")[:-1]
    pkg = pkg[: len(pkg) - (level - 1)]
    return ".".join(pkg + ([module] if module else []))


def _aliases(tree, base_mod, is_pkg):
    """alias -> ("module", dotted) | ("name", dotted_module, original_name), atom-internal only."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = _absolute(base_mod, is_pkg, node.level, node.module)
            if not target.startswith("atom"):
                continue
            for n in node.names:
                alias = n.asname or n.name
                full = f"{target}.{n.name}"
                if _module_file(full) is not None:
                    out[alias] = ("module", full)
                else:
                    out[alias] = ("name", target, n.name)
        elif isinstance(node, ast.Import):
            for n in node.names:
                if n.asname and n.name.startswith("atom"):
                    out[n.asname] = ("module", n.name)
    return out


def _find_def(mod, name, depth=0):
    """The FunctionDef/ClassDef that defines ``mod.name``, following re-exports. -> (node, module) | None"""
    if depth > 6:
        return None
    path = _module_file(mod)
    if path is None:
        return None
    tree = _tree(path)
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            return node, mod
    base_mod, is_pkg = _file_module(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
            (n.asname or n.name) == name for n in node.names
        ):
            # re-exports: the walk also covers imports inside `try:` / `if:` blocks
            target = _absolute(base_mod, is_pkg, node.level, node.module)
            orig = next(n.name for n in node.names if (n.asname or n.name) == name)
            hit = _find_def(target, orig, depth + 1)
            if hit:
                return hit
    return None


def _is_dataclass(node):
    for d in node.decorator_list:
        d = d.func if isinstance(d, ast.Call) else d
        if (isinstance(d, ast.Name) and d.id == "dataclass") or (
            isinstance(d, ast.Attribute) and d.attr == "dataclass"
        ):
            return True
    return False


def _synthesised_class_signature(node):
    """Signature of a class that defines no ``__init__``: a dataclass (its annotated fields, in order)
    or an exception subclass (``*args``). Anything else is reported, not guessed."""
    if _is_dataclass(node):
        params = []
        for st in node.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                ann = ast.unparse(st.annotation)
                if ann.startswith(("ClassVar", "typing.ClassVar")):
                    continue
                v = st.value
                if (
                    isinstance(v, ast.Call)
                    and getattr(v.func, "id", getattr(v.func, "attr", "")) == "field"
                ):
                    if any(
                        k.arg == "init" and getattr(k.value, "value", True) is False
                        for k in v.keywords
                    ):
                        continue
                    has_default = any(
                        k.arg in ("default", "default_factory") for k in v.keywords
                    )
                else:
                    has_default = v is not None
                params.append(
                    inspect.Parameter(
                        st.target.id,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None if has_default else inspect.Parameter.empty,
                    )
                )
        return inspect.Signature(params), ""
    bases = [getattr(b, "id", getattr(b, "attr", "")) for b in node.bases]
    if bases and all(
        b.endswith(("Error", "Exception", "Warning")) or b == "BaseException"
        for b in bases
    ):
        return (
            inspect.Signature(
                [inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL)]
            ),
            "",
        )
    return None, "class without __init__ that is neither a dataclass nor an exception"


def _signature(node):
    """``inspect.signature`` of a parsed def (annotations and default VALUES are irrelevant to binding,
    so they are neutralised and the def is executed in an empty namespace). Classes use ``__init__``
    minus ``self``. -> (signature, note) with signature None when it cannot be built."""
    if isinstance(node, ast.ClassDef):
        init = next(
            (
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
            None,
        )
        if init is None:
            return _synthesised_class_signature(node)
        node, drop_self = init, True
    else:
        drop_self = False
    if node.decorator_list:
        return None, "decorated: the wrapper's signature is not the def's"
    fn = ast.FunctionDef(
        name="f",
        args=node.args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=None,
        type_comment=None,
        **({"type_params": []} if "type_params" in ast.FunctionDef._fields else {}),
    )
    a = fn.args
    for arg in [
        *a.posonlyargs,
        *a.args,
        *a.kwonlyargs,
        *(x for x in (a.vararg, a.kwarg) if x),
    ]:
        arg.annotation = None
    a.defaults = [ast.Constant(None) for _ in a.defaults]
    a.kw_defaults = [None if d is None else ast.Constant(None) for d in a.kw_defaults]
    ns = {}
    exec(  # noqa: S102 - only a signature-only def (annotations and defaults stripped) is run
        compile(ast.fix_missing_locations(ast.Module([fn], [])), "<sig>", "exec"), ns
    )
    sig = inspect.signature(ns["f"])
    if drop_self:
        sig = sig.replace(parameters=list(sig.parameters.values())[1:])
    return sig, ""


def _bind(sig, call):
    pos = [None] * len(call.args)
    kws = {k.arg: None for k in call.keywords if k.arg is not None}
    partial = any(isinstance(x, ast.Starred) for x in call.args) or any(
        k.arg is None for k in call.keywords
    )
    (sig.bind_partial if partial else sig.bind)(*pos, **kws)


def audit_source(src, base_mod, is_pkg, want):
    """Bind every cross-module call in ``src`` whose callee module satisfies ``want(module)``.
    -> (checked: list[str], problems: list[str], unchecked: list[str])."""
    tree = ast.parse(src)
    aliases = _aliases(tree, base_mod, is_pkg)
    checked, problems, unchecked = [], [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        target = None
        if isinstance(f, ast.Name) and aliases.get(f.id, ("",))[0] == "name":
            _, mod, name = aliases[f.id]
            target = (mod, name)
        elif (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and aliases.get(f.value.id, ("",))[0] == "module"
        ):
            target = (aliases[f.value.id][1], f.attr)
        if target is None or not want(target[0]):
            continue
        label = (
            f"{target[0]}.{target[1]} (line {node.lineno}: {ast.unparse(node)[:90]})"
        )
        hit = _find_def(*target)
        if hit is None:
            unchecked.append((label, "callee def not found (variable/alias/dynamic)"))
            continue
        sig, note = _signature(hit[0])
        if sig is None:
            unchecked.append((label, note))
            continue
        try:
            _bind(sig, node)
        except TypeError as e:
            problems.append(f"{label}: {e}; real signature is {target[1]}{sig}")
        else:
            checked.append(f"{target[0]}.{target[1]}")
    return checked, problems, unchecked


def _atom_files(outside_package):
    for p in sorted((ROOT / "atom").rglob("*.py")):
        inside = PKG_DIR in p.parents
        if outside_package != (not inside):
            continue
        if p.name == "test_utils.py" and inside:
            continue  # test code, not runtime
        yield p


def _audit_files(files, want):
    checked, problems, unchecked = [], [], []
    for p in files:
        base_mod, is_pkg = _file_module(p)
        c, pr, un = audit_source(p.read_text(encoding="utf-8"), base_mod, is_pkg, want)
        checked += c
        problems += [f"{p.relative_to(ROOT)}: {x}" for x in pr]
        unchecked += [(f"{p.relative_to(ROOT)}: {lab}", why) for lab, why in un]
    return checked, problems, unchecked


# ------------------------------------------------------------------------------------------------
def test_every_call_from_atom_into_the_gemm_fusion_package_binds_to_its_real_signature():
    """ModelRunner (and anything else outside the package) -> gemm_fusion. This is the audit that
    would have caught ``before_graph_capture(buckets, device=...)`` after the parameter was removed.
    """
    checked, problems, unchecked = _audit_files(
        _atom_files(outside_package=True), lambda m: m == PKG or m.startswith(PKG + ".")
    )
    assert (
        not problems
    ), "calls into gemm_fusion that do not match the real signature:\n" + "\n".join(
        problems
    )
    assert (
        not unchecked
    ), "calls into gemm_fusion whose callee could not be resolved:\n" + "\n".join(
        f"{lab}: {why}" for lab, why in unchecked
    )
    # not vacuous: every entry point the runner uses must have been found and bound
    called = {c.rsplit(".", 1)[1] for c in checked}
    assert called >= {
        "init_gemm_fusion",
        "bind_kv_cache",
        "before_graph_capture",
        "after_graph_capture",
        "check_step",
        "shutdown",
        "install_m3_sites",
    }, sorted(called)


def test_every_call_between_gemm_fusion_modules_and_from_them_into_atom_binds():
    """The package's own cross-module calls (site -> hooks, ops -> hooks, ...) and its calls into the
    rest of ATOM (get_current_atom_config, get_sp_world_size, mark_spliting_op, ...)."""
    checked, problems, unchecked = _audit_files(
        _atom_files(outside_package=False), lambda m: m.startswith("atom")
    )
    assert not problems, "calls that do not match the real signature:\n" + "\n".join(
        problems
    )
    assert (
        len(checked) >= 15
    ), f"audit looks vacuous: only {len(checked)} calls were bound"
    # What cannot be bound is listed so a new unchecked callee is a conscious decision, not silence.
    # (a callee that is a decorated def, an alias or a variable has no signature we can trust here)
    allowed = ("decorated", "callee def not found")
    bad = [f"{lab}: {why}" for lab, why in unchecked if not why.startswith(allowed)]
    assert (
        not bad
    ), "callees the audit cannot bind and has no reason for:\n" + "\n".join(bad)


def test_the_audit_has_power_it_flags_the_exact_call_that_broke_the_gpu_run():
    """The pre-fix ModelRunner call, verbatim in shape: a keyword the function does not take."""
    src = (
        "def capture(self, q_buckets):\n"
        "    from atom.model_ops import gemm_fusion\n"
        "    gemm_fusion.before_graph_capture(sorted(q_buckets), device=self.device)\n"
    )
    checked, problems, _ = audit_source(
        src, "atom.model_engine.model_runner", False, lambda m: m.startswith(PKG)
    )
    assert checked == [] and len(problems) == 1
    assert "before_graph_capture" in problems[0] and "device" in problems[0]
    # the fixed call binds
    ok_src = src.replace(", device=self.device", "")
    checked, problems, _ = audit_source(
        ok_src, "atom.model_engine.model_runner", False, lambda m: m.startswith(PKG)
    )
    assert problems == [] and checked == [f"{PKG}.before_graph_capture"]


@pytest.mark.parametrize(
    "call,ok",
    [
        ("gemm_fusion.init_gemm_fusion(config, get_sp_group())", True),
        ("gemm_fusion.init_gemm_fusion(config)", False),  # missing sp_group
        (
            "gemm_fusion.init_gemm_fusion(config, grp, resources=r)",
            False,
        ),  # removed parameter
        ("gemm_fusion.bind_kv_cache(kv)", True),
        ("gemm_fusion.bind_kv_cache(kv, extra)", False),
        ("gemm_fusion.init_gemm_fusion(config, grp, kv_publisher=True)", True),
        ("gemm_fusion.init_gemm_fusion(config, grp, publisher=True)", False),
        ("gemm_fusion.check_step()", True),
        ("gemm_fusion.check_step(batch_ids)", True),  # the request ids of the step
        ("gemm_fusion.check_step(req_ids=ids)", True),
        ("gemm_fusion.check_step(ids, 1)", False),  # at most the one parameter
        (
            "gemm_fusion.check_step(batch=b)",
            False,
        ),  # the runner passes ids, not a batch
        ("gemm_fusion.shutdown()", True),
        ("gemm_fusion.shutdown(timeout_s=3)", True),
        ("gemm_fusion.after_graph_capture()", True),
        ("gemm_fusion.after_graph_capture(device=d)", False),
        ("gemm_fusion.before_graph_capture(buckets)", True),
        (
            "gemm_fusion.before_graph_capture(*args)",
            True,
        ),  # unverifiable positional count: partial bind
        ("gemm_fusion.before_graph_capture(buckets, out_dtype=t)", False),
    ],
)
def test_binding_verdicts_for_each_runner_entry_point(call, ok):
    src = f"def f():\n    from atom.model_ops import gemm_fusion\n    {call}\n"
    _checked, problems, unchecked = audit_source(
        src, "atom.model_engine.model_runner", False, lambda m: m.startswith(PKG)
    )
    assert not unchecked
    assert (not problems) == ok, (call, problems)
