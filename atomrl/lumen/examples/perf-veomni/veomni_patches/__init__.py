# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The optimizations of this example, as patches applied by ``train_dit_lumen.py``.

Nothing here modifies VeOmni's source or the Lumen library; each module rebinds
at runtime what it names, and only when its ``LUMEN_PATCH`` entry is set.

==================  ==============================================================
``LUMEN_PATCH``     module
==================  ==============================================================
``vae_conv*``       :mod:`atomrl.lumen.ops.conv` -- Lumen's AITER-backed
                    convolution, used by ``lumen_vae_conv.py``
``qwen_offline_fix`` :mod:`.qwen_offline` -- VeOmni Qwen-Image offline_training fix
==================  ==============================================================
"""


def _qwen_offline_fix():
    from .qwen_offline import install_qwen_offline_fix

    install_qwen_offline_fix()


INSTALLERS = {
    "qwen_offline_fix": _qwen_offline_fix,
}
