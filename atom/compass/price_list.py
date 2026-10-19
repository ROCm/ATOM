# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Measured GEMM prices for one device.

A price is the CUDA-graph replay time of one kernel at one ``(dtype, M, K, N)``.
Lookup is exact. A missing shape is a miss, not a neighbour and not a roofline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class MissingPrice(KeyError):
    """No measured kernel covers this GEMM."""


@dataclass(frozen=True)
class Price:
    dtype: str
    m: int
    k: int
    n: int
    seconds: float
    kernel: str

    def key(self) -> tuple[str, int, int, int]:
        return (self.dtype, self.m, self.k, self.n)


@dataclass(frozen=True)
class PriceList:
    """One measurement campaign. ``entries`` are keyed by dtype and shape."""

    entries: dict[tuple[str, int, int, int], Price]
    device: str
    method: str
    provenance: str = "measured"

    def seconds(self, dtype: str, m: int, k: int, n: int) -> float:
        try:
            return self.entries[(dtype, m, k, n)].seconds
        except KeyError:
            raise MissingPrice(
                f"no measured price for dtype={dtype} M={m} K={k} N={n} "
                f"on {self.device}"
            ) from None

    def save(self, path: str | Path) -> None:
        payload = {
            "provenance": self.provenance,
            "method": self.method,
            "device": self.device,
            "entries": [
                {
                    "dtype": price.dtype,
                    "m": price.m,
                    "k": price.k,
                    "n": price.n,
                    "seconds": price.seconds,
                    "kernel": price.kernel,
                }
                for price in self.entries.values()
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "PriceList":
        payload = json.loads(Path(path).read_text())
        entries = {}
        for raw in payload["entries"]:
            price = Price(
                dtype=raw["dtype"],
                m=int(raw["m"]),
                k=int(raw["k"]),
                n=int(raw["n"]),
                seconds=float(raw["seconds"]),
                kernel=raw["kernel"],
            )
            entries[price.key()] = price
        return cls(
            entries=entries,
            device=payload["device"],
            method=payload["method"],
            provenance=payload.get("provenance", "measured"),
        )
