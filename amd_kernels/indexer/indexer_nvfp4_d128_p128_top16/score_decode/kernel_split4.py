# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Constants of the decode scorer split over the TP4 ranks (``split4.decode_score``): the one-GPU fp8 scorer's
(``kernel``) and the run of blocks a rank scores."""

from .kernel import (
    BLK,
    MT,
    NT,
    PAGE_BYTES,
    PAGE_CODES,
    SENTINEL_INIT,
    SENTINEL_LOCAL,
    D,
    _xlane_max4,
)

CP_CH = 16  # blocks per run of one rank
WORLD = 4
