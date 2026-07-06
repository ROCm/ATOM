"""Split-K prezero pass. Graph surgery on the FULL (pre-split) dynamo graph --
the only stage where the opaque attention op is still one node, so its
qb_prezero/oproj_prezero slots (q_b, o_proj live inside it) can be threaded.

Per input_layernorm donor it allocates one [m, n_total] buffer sliced into
qkv_a (visible gemm) | q_b | o_proj, wires the fused_allreduce_rmsnorm_ ->
ar_rmsnorm_maybe_prezero_ (greedy zeroing up to n_base), rewrites the qkv_a gemm
-> mm_maybe_prezero_, and threads the q_b/o_proj slices into the attn op.
Post_attn gate donors get the single-buffer form (no attention threading).
"""
import operator
import re

import torch


def _opname(t):
    try:
        n = t.name()
        return n.rsplit(".", 1)[0] if "." in n else n
    except Exception:
        return getattr(t, "__name__", str(t))


def _is(target, name):
    return _opname(target).replace("::", ".").split(".default")[0].endswith(name)


def _resolve(qualname):
    ns, name = qualname.split("::")
    pkt = getattr(torch.ops, ns, None)
    op = getattr(pkt, name, None) if pkt else None
    return getattr(op, "default", None) if op else None


def _weight_layer(node):
    """(layer_idx, kind) from a norm-weight placeholder feeding a donor, else None."""
    if not isinstance(node, torch.fx.Node):
        return None
    m = re.search(r"layers_modules_(\d+)_modules_(input_layernorm|post_attention_layernorm)", node.name)
    return (int(m.group(1)), m.group(2)) if m else None


def _attn_layer(node):
    if node.op != "call_function" or not _is(node.target, "unified_attention_with_output_base"):
        return None
    lname = node.args[5] if len(node.args) > 5 else None
    m = re.search(r"layers\.(\d+)\.", lname) if isinstance(lname, str) else None
    return int(m.group(1)) if m else None


def deepseek_v2_prezero_pass(gm, n_total_map=None, n_base_map=None):
    g = gm.graph if hasattr(gm, "graph") else gm
    n_total_map = n_total_map or {}
    stats = {"qkv": 0, "gate": 0, "qb": 0, "oproj": 0}
    ar_pz = _resolve("aiter::ar_rmsnorm_maybe_prezero_")
    mm_pz = _resolve("aiter::mm_maybe_prezero_")
    if ar_pz is None or mm_pz is None:
        print("[prezero] maybe_prezero ops not registered -> no-op", flush=True)
        return gm

    # index attention nodes by layer
    attn_by_layer = {}
    for n in g.nodes:
        li = _attn_layer(n)
        if li is not None:
            attn_by_layer[li] = n

    aten = torch.ops.aten
    for donor in list(g.nodes):
        if donor.op != "call_function" or not _is(donor.target, "fused_allreduce_rmsnorm_"):
            continue
        if len(donor.args) < 4:
            continue
        wl = _weight_layer(donor.args[2])
        if wl is None:
            continue
        layer_idx, kind = wl

        # consumer gemm: donor -> getitem[0] -> gemm_a16w16
        g0 = next((u for u in donor.users if u.target is operator.getitem and u.args[1] == 0), None)
        if g0 is None:
            continue
        gemm = next((u for u in g0.users if u.op == "call_function" and _is(u.target, "gemm_a16w16")), None)
        if gemm is None:
            continue
        val = gemm.meta.get("val", gemm.meta.get("example_value"))
        try:
            n_out = int(val.shape[-1])
        except Exception:
            continue
        xval = donor.args[0].meta.get("val", donor.args[0].meta.get("example_value")) \
            if isinstance(donor.args[0], torch.fx.Node) else None
        if xval is None:
            continue
        dtype, device = xval.dtype, xval.device

        x, res, w, eps = donor.args[0], donor.args[1], donor.args[2], donor.args[3]
        bias = gemm.args[2] if len(gemm.args) > 2 else gemm.kwargs.get("bias")
        normed = gemm.args[0]  # == g0

        # combined input_layernorm donor = qkv_a (visible gemm) + q_b + o_proj
        # (both inside the opaque attn op); post_attn gate donor = single gemm.
        n_total, n_base, attn = n_out, n_out, None
        if kind == "input_layernorm":
            attn = attn_by_layer.get(layer_idx)
            lname = attn.args[5] if attn is not None else None
            nt = n_total_map.get(lname)
            nb = (n_base_map or {}).get(lname)
            if nt and nt > n_out:
                n_total = nt
                n_base = nb if (nb and n_out <= nb <= nt) else nt
        n_qb = n_base - n_out
        n_oproj = n_total - n_base

        with g.inserting_before(donor):
            m = g.call_function(aten.sym_size.int, (x, 0))
            buf = g.call_function(aten.empty.memory_format, ([m, n_total],),
                                  {"dtype": dtype, "device": device})
            qb = oproj = None
            if n_qb > 0 or n_oproj > 0:
                flat = g.call_function(aten.view.default, (buf, [-1]))
                o1 = g.call_function(operator.mul, (m, n_out))
                qkva = g.call_function(aten.view.default,
                    (g.call_function(operator.getitem, (flat, slice(None, o1, None))), [m, n_out]))
                off = o1
                if n_qb > 0:
                    o2 = g.call_function(operator.mul, (m, n_base))
                    qb = g.call_function(aten.view.default,
                        (g.call_function(operator.getitem, (flat, slice(off, o2, None))), [m, n_qb]))
                    off = o2
                if n_oproj > 0:
                    oproj = g.call_function(aten.view.default,
                        (g.call_function(operator.getitem, (flat, slice(off, None, None))), [m, n_oproj]))
            else:
                qkva = buf

        # donor -> ar_rmsnorm_maybe_prezero_(x, res, w, eps, buf, n_total, n_base)
        with g.inserting_before(donor):
            new_donor = g.call_function(ar_pz, (x, res, w, eps, buf, n_total, n_base))
        donor.replace_all_uses_with(new_donor)
        g.erase_node(donor)

        # gemm -> mm_maybe_prezero_(qkva, normed, W, n_base); consumers read qkva
        with g.inserting_before(gemm):
            g.call_function(mm_pz, (qkva, normed, gemm.args[1], n_base),
                            {} if bias is None else {"bias": bias})
        gemm.replace_all_uses_with(qkva)
        g.erase_node(gemm)
        stats["qkv" if kind == "input_layernorm" else "gate"] += 1

        # thread slices into the attn op: qb_prezero=arg[8], oproj_prezero=arg[9]
        if attn is not None and (qb is not None or oproj is not None):
            a = list(attn.args)
            while len(a) < 10:
                a.append(None)
            if qb is not None:
                a[8] = qb
                stats["qb"] += 1
            if oproj is not None:
                a[9] = oproj
                stats["oproj"] += 1
            attn.args = tuple(a)

    g.lint()
    print(f"[prezero] rewrote qkv_a={stats['qkv']} gate={stats['gate']} "
          f"qb_threaded={stats['qb']} oproj_threaded={stats['oproj']}", flush=True)
    return gm
