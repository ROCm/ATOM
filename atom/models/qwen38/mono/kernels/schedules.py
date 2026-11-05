# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Schedule knobs of the served Qwen Mono layer programs, keyed ``(kind, batch, spec)``.

``KNOBS[(kind, batch, spec)]`` is the ``knobs`` dict ``recipe.compose_layer`` builds with: block constructor knobs
by block name, ``moe`` (``router`` / ``line``), ``place`` ({stage: first CTA}; None keeps the derived base),
``hold`` ({stage: s_sleep units}) and ``dual`` (dual-copy mailboxes up to ``max`` bytes a region). Every Linear's
task shape is frozen: ``shape = (row groups per task, K slices, wave rows, prefetch depth)``, the shape the
resource-fitted search chose for the measured build. ``delta`` / ``qsa`` are the target's GDN / QSA layers (MTP3
verify: 4 rows a request), ``mtp`` the BF16 drafter layer (first draft step 4 rows, later steps 1).
"""

KNOBS: dict[tuple[str, int, int], dict] = {
    ("delta", 1, 4): {
        "attn_hc": {"norm_from": "lds", "wait": "direct"},
        "dual": {"max": 98304},
        "gdn": {"splits": 2},
        "in_proj": {"shape": (4, 1, 1, 12)},
        "mlp_hc": {"norm_from": "lds", "wait": "direct"},
        "moe": {
            "line": {"cbase": 165, "dn_issue": "gu", "dn_pre": 10, "w_cpol": 18},
            "router": {"k_split": 2},
        },
        "out_proj": {"shape": (2, 1, 1, 12)},
    },
    ("delta", 2, 4): {
        "attn_hc": {"norm_from": "lds", "wait": "direct"},
        "dual": {"max": 98304},
        "in_proj": {"shape": (4, 1, 1, 12)},
        "mlp_hc": {"norm_from": "lds"},
        "moe": {
            "line": {"cbase": 165, "dn_issue": "epi", "dn_pre": 14, "w_cpol": 18},
            "router": {"k_split": 2},
        },
        "out_proj": {"pre_units": 1, "shape": (1, 1, 1, 6)},
        "place": {"attn_hc.down": 64},
    },
    ("delta", 3, 4): {
        "attn_hc": {"norm_from": "lds"},
        "in_proj": {"shape": (4, 1, 1, 12)},
        "mlp_hc": {"norm_from": "lds"},
        "moe": {"line": {"cbase": 165, "dn_issue": "epi", "dn_pre": 10, "w_cpol": 18}},
        "out_proj": {"shape": (1, 1, 1, 6)},
    },
    ("delta", 4, 4): {
        "attn_hc": {"norm_from": "lds"},
        "in_proj": {"shape": (4, 1, 1, 12)},
        "mlp_hc": {"norm_from": "lds"},
        "moe": {
            "line": {"cbase": 165, "dn_issue": "epi", "dn_pre": 10, "w_cpol": 18},
            "router": {"k_split": 2},
        },
        "out_proj": {"shape": (2, 2, 1, 6)},
    },
    ("qsa", 1, 4): {
        "attn_hc": {"wait": "direct"},
        "dual": {"max": 98304},
        "idx_qk": {"shape": (1, 1, 1, 3)},
        "mlp_hc": {"norm_from": "lds", "wait": "direct"},
        "moe": {
            "line": {"cbase": 165, "dn_issue": "gu", "dn_pre": 10, "w_cpol": 18},
            "router": {"k_split": 2},
        },
        "o_proj": {"shape": (1, 1, 1, 6)},
        "qkv": {"shape": (4, 1, 1, 12)},
    },
    ("qsa", 2, 4): {
        "attn_hc": {"norm_from": "lds"},
        "dual": {"max": 98304},
        "hold": {"attn_hc.up": 64},
        "idx_qk": {"shape": (1, 1, 1, 3)},
        "mlp_hc": {"norm_from": "lds"},
        "moe": {
            "line": {"cbase": 165, "dn_issue": "epi", "dn_pre": 10, "w_cpol": 18},
            "router": {"k_split": 2},
        },
        "o_proj": {"shape": (2, 1, 1, 12)},
        "place": {"attn_hc.down": None},
        "qkv": {"shape": (4, 1, 1, 12)},
    },
    ("qsa", 3, 4): {
        "attn_hc": {"norm_from": "lds"},
        "dual": {"max": 98304},
        "idx_qk": {"shape": (1, 1, 1, 3)},
        "mlp_hc": {"norm_from": "lds"},
        "moe": {
            "line": {
                "cbase": 165,
                "chunk_reg": "v",
                "dn_issue": "epi",
                "dn_pre": 10,
                "w_cpol": 18,
            }
        },
        "o_proj": {"shape": (1, 1, 1, 6)},
        "qkv": {"shape": (4, 1, 1, 12)},
    },
    ("qsa", 4, 4): {
        "attn_hc": {"norm_from": "lds"},
        "dual": {"max": 98304},
        "idx_qk": {"shape": (1, 1, 1, 3)},
        "mlp_hc": {"norm_from": "lds"},
        "moe": {
            "line": {
                "cbase": 165,
                "chunk_reg": "v",
                "dn_issue": "epi",
                "dn_pre": 8,
                "w_cpol": 16,
            }
        },
        "o_proj": {"shape": (4, 2, 1, 12)},
        "place": {"attn_hc.down": 192, "attn_hc.up": 116},
        "qkv": {"shape": (8, 1, 2, 13)},
    },
    ("mtp", 1, 4): {
        "attn_hc": {"wait": "direct"},
        "dual": {"max": 98304},
        "mlp_hc": {"norm_from": "lds", "wait": "direct"},
        "o_proj": {"shape": (1, 1, 1, 6)},
        "qkv_idx": {"shape": (4, 1, 1, 8)},
    },
    ("mtp", 2, 4): {
        "attn_hc": {"norm_from": "lds"},
        "dual": {"max": 98304},
        "hold": {"attn_hc.up": 64},
        "mlp_hc": {"norm_from": "lds"},
        "o_proj": {"shape": (2, 2, 1, 6)},
        "place": {"attn_hc.down": None},
        "qkv_idx": {"shape": (4, 1, 2, 8)},
    },
    ("mtp", 3, 4): {
        "attn_hc": {"norm_from": "lds"},
        "dual": {"max": 98304},
        "mlp_hc": {"norm_from": "lds"},
        "o_proj": {"shape": (2, 3, 1, 4)},
        "qkv_idx": {"shape": (14, 4, 2, 8)},
    },
    ("mtp", 4, 4): {
        "attn_hc": {"norm_from": "lds"},
        "dual": {"max": 98304},
        "mlp_hc": {"norm_from": "lds"},
        "o_proj": {"shape": (4, 4, 2, 6)},
        "place": {"attn_hc.down": 192, "attn_hc.up": 116},
        "qkv_idx": {"shape": (8, 2, 4, 8)},
    },
    ("mtp", 1, 1): {
        "dual": {"max": 65536},
        "o_proj": {"shape": (1, 1, 1, 6)},
        "qkv_idx": {"shape": (4, 1, 1, 9)},
    },
    ("mtp", 2, 1): {
        "o_proj": {"shape": (1, 1, 1, 6)},
        "qkv_idx": {"shape": (4, 1, 1, 9)},
    },
    ("mtp", 3, 1): {
        "o_proj": {"shape": (1, 1, 1, 6)},
        "qkv_idx": {"shape": (4, 1, 1, 9)},
    },
    ("mtp", 4, 1): {
        "o_proj": {"shape": (1, 1, 1, 6)},
        "qkv_idx": {"shape": (4, 1, 1, 8)},
    },
}
