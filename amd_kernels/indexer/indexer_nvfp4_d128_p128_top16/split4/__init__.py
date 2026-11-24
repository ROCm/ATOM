# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode indexer split over four GPUs (gfx950) on the rank-owned index cache (rank r holds the blocks
b % 4 == r; ``host.IndexSplit4RankOwned``); ``kernel``: its flag words."""
