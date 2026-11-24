# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Switches of the indexer at their served defaults; tests monkeypatch them."""

# the prefill scorer (``score_prefill.kernel``) writes each row's scores 16 blocks (64 B) at a time from LDS; False
# stores every 4-block group as it completes, the same scores byte for byte
INDEX_SCORE_STAGED = True
