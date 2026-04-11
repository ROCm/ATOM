"""
Pareto frontier analysis for inference configurations.

Addresses Q10: the two Pareto dimensions are:
- tokens/s/gpu  (efficiency — how well are you using each GPU)
- tokens/s/user (interactivity — how fast does each user get responses)

These represent the fundamental throughput-latency tradeoff in LLM serving:
- High batch size → high tokens/s/gpu but lower tokens/s/user (higher latency)
- Low batch size  → high tokens/s/user but lower tokens/s/gpu (wasted capacity)

The Pareto frontier identifies configurations where you cannot improve one
metric without degrading the other.
"""

from __future__ import annotations

import logging
from typing import Optional

from atom.autotuner.types import BenchmarkResult, InferenceConfig, ParetoPoint

logger = logging.getLogger(__name__)


class ParetoAnalyzer:
    """
    Computes and maintains the Pareto frontier from benchmark results.

    Supports SLA filtering (TTFT ≤ X, TPOT ≤ Y) before frontier computation.
    """

    def __init__(
        self,
        ttft_limit_ms: Optional[float] = None,
        tpot_limit_ms: Optional[float] = None,
        request_latency_limit_ms: Optional[float] = None,
    ):
        self.ttft_limit = ttft_limit_ms
        self.tpot_limit = tpot_limit_ms
        self.req_lat_limit = request_latency_limit_ms
        self._points: list[ParetoPoint] = []

    def add_result(self, result: BenchmarkResult) -> ParetoPoint:
        """Add a benchmark result and return its Pareto point."""
        point = ParetoPoint(
            config=result.config,
            throughput_per_gpu=result.throughput_per_gpu,
            throughput_per_user=result.throughput_per_user,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            request_latency_ms=result.request_latency_ms,
        )
        self._points.append(point)
        return point

    def add_results(self, results: list[BenchmarkResult]) -> None:
        for r in results:
            self.add_result(r)

    def compute_frontier(self) -> list[ParetoPoint]:
        """
        Compute the Pareto frontier after SLA filtering.

        A point is on the frontier if no other point dominates it in both
        throughput_per_gpu AND throughput_per_user (both are "higher is better").
        """
        feasible = self._filter_sla(self._points)
        if not feasible:
            logger.warning("No configurations meet SLA constraints")
            return []

        for p in feasible:
            p.is_frontier = False

        frontier = []
        for i, p in enumerate(feasible):
            dominated = False
            for j, q in enumerate(feasible):
                if i == j:
                    continue
                if (q.throughput_per_gpu >= p.throughput_per_gpu and
                    q.throughput_per_user >= p.throughput_per_user and
                    (q.throughput_per_gpu > p.throughput_per_gpu or
                     q.throughput_per_user > p.throughput_per_user)):
                    dominated = True
                    break
            if not dominated:
                p.is_frontier = True
                frontier.append(p)

        frontier.sort(key=lambda p: p.throughput_per_user)
        logger.info(
            "Pareto frontier: %d points from %d feasible (%d total)",
            len(frontier), len(feasible), len(self._points),
        )
        return frontier

    def best_by_throughput_per_gpu(self) -> Optional[ParetoPoint]:
        frontier = self.compute_frontier()
        if not frontier:
            return None
        return max(frontier, key=lambda p: p.throughput_per_gpu)

    def best_by_throughput_per_user(self) -> Optional[ParetoPoint]:
        frontier = self.compute_frontier()
        if not frontier:
            return None
        return max(frontier, key=lambda p: p.throughput_per_user)

    def best_balanced(self) -> Optional[ParetoPoint]:
        """Pick the frontier point closest to the "ideal" corner."""
        frontier = self.compute_frontier()
        if not frontier:
            return None

        max_gpu = max(p.throughput_per_gpu for p in frontier) or 1
        max_user = max(p.throughput_per_user for p in frontier) or 1

        def score(p: ParetoPoint) -> float:
            norm_gpu = p.throughput_per_gpu / max_gpu
            norm_user = p.throughput_per_user / max_user
            return (norm_gpu ** 2 + norm_user ** 2) ** 0.5

        return max(frontier, key=score)

    def top_n(self, n: int = 5, sort_by: str = "throughput_per_gpu") -> list[ParetoPoint]:
        feasible = self._filter_sla(self._points)
        key_fn = lambda p: getattr(p, sort_by, 0)
        feasible.sort(key=key_fn, reverse=True)
        return feasible[:n]

    def _filter_sla(self, points: list[ParetoPoint]) -> list[ParetoPoint]:
        """Filter points that violate SLA constraints."""
        result = []
        for p in points:
            if self.ttft_limit and p.ttft_ms > self.ttft_limit:
                continue
            if self.tpot_limit and p.tpot_ms > self.tpot_limit:
                continue
            if self.req_lat_limit and p.request_latency_ms > self.req_lat_limit:
                continue
            result.append(p)
        return result

    def format_frontier(self, top_n: int = 10) -> str:
        """Format the Pareto frontier as an ASCII table."""
        frontier = self.compute_frontier()
        if not frontier:
            return "No Pareto frontier points found."

        frontier = frontier[:top_n]
        lines = []
        lines.append(
            f"{'Rank':>4} | {'tokens/s/gpu':>14} | {'tokens/s/user':>14} | "
            f"{'TTFT(ms)':>10} | {'TPOT(ms)':>10} | {'Config':>30}"
        )
        lines.append("-" * 100)

        for i, p in enumerate(sorted(frontier, key=lambda x: -x.throughput_per_gpu)):
            cfg = p.config
            par = f"tp{cfg.tp}pp{cfg.pp}"
            if cfg.disagg:
                par += f" disagg(p{cfg.prefill_workers}d{cfg.decode_workers})"
            par += f" bs{cfg.batch_size} {cfg.quant_format}"
            lines.append(
                f"{i+1:>4} | {p.throughput_per_gpu:>14.2f} | {p.throughput_per_user:>14.2f} | "
                f"{p.ttft_ms:>10.2f} | {p.tpot_ms:>10.2f} | {par:>30}"
            )

        return "\n".join(lines)

    def format_ascii_chart(self, width: int = 72, height: int = 24) -> str:
        """Render a simple ASCII scatter plot of the Pareto frontier."""
        frontier = self.compute_frontier()
        all_feasible = self._filter_sla(self._points)

        if not all_feasible:
            return "No data to plot."

        x_vals = [p.throughput_per_user for p in all_feasible]
        y_vals = [p.throughput_per_gpu for p in all_feasible]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1

        grid = [[" "] * width for _ in range(height)]

        frontier_fps = {id(p) for p in frontier}

        for p in all_feasible:
            x = int((p.throughput_per_user - x_min) / (x_max - x_min) * (width - 1))
            y = int((p.throughput_per_gpu - y_min) / (y_max - y_min) * (height - 1))
            y = height - 1 - y
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))

            if id(p) in frontier_fps:
                grid[y][x] = "*"
            else:
                grid[y][x] = "."

        lines = []
        lines.append(f"  tokens/s/gpu vs tokens/s/user (* = Pareto frontier)")
        lines.append(f"  {y_max:>10.1f} |{''.join(grid[0])}")
        for row in grid[1:-1]:
            lines.append(f"  {'':>10} |{''.join(row)}")
        lines.append(f"  {y_min:>10.1f} |{''.join(grid[-1])}")
        lines.append(f"  {'':>10} +{'-' * width}")
        lines.append(f"  {'':>10}  {x_min:<10.1f}{' ' * (width - 20)}{x_max:>10.1f}")
        lines.append(f"  {'':>10}  {'tokens/s/user':^{width}}")

        return "\n".join(lines)
