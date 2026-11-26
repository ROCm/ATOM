# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step kernel's weights, from the loaded V4.1 model, in the layouts it reads: ``planes`` the routed and shared
experts (captured before the loader's in-place shuffle), ``layer`` a layer-table row (attention, FFN and attention
fronts), ``draft`` the draft stages' and the step tail's (embedding, LM head tiles, Markov, draft context).
"""
