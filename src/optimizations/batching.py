"""
Batch Processing Optimizer for LLM Inference

Batch size is one of the most impactful knobs for throughput vs latency:

- Small batches (1-4):   Low latency, low throughput  → use for real-time chat
- Medium batches (8-16): Balanced                     → use for most production
- Large batches (32+):   High throughput, high latency → use for offline jobs

This module experiments with different batch strategies and finds the
optimal configuration for a given latency SLA.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BatchConfig:
    """
    Configuration for a batching experiment.

    Attributes:
        batch_size:       Number of requests processed together
        strategy:         'static' (fixed) or 'dynamic' (fills to max tokens)
        max_wait_ms:      Max time to wait for a batch to fill (dynamic only)
        target_latency_ms: SLA target for p95 latency
    """
    batch_size: int
    strategy: str = "static"      # "static" | "dynamic"
    max_wait_ms: float = 100.0
    target_latency_ms: float = 500.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size":        self.batch_size,
            "strategy":          self.strategy,
            "max_wait_ms":       self.max_wait_ms,
            "target_latency_ms": self.target_latency_ms,
        }


@dataclass
class BatchExperimentResult:
    """Result of a single batch size experiment"""
    batch_size: int
    strategy: str
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_req_per_sec: float
    throughput_tokens_per_sec: float
    meets_sla: bool
    sla_target_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size":               self.batch_size,
            "strategy":                 self.strategy,
            "mean_latency_ms":          round(self.mean_latency_ms,  2),
            "p50_latency_ms":           round(self.p50_latency_ms,   2),
            "p95_latency_ms":           round(self.p95_latency_ms,   2),
            "p99_latency_ms":           round(self.p99_latency_ms,   2),
            "throughput_req_per_sec":   round(self.throughput_req_per_sec,   2),
            "throughput_tokens_per_sec":round(self.throughput_tokens_per_sec,2),
            "meets_sla":                self.meets_sla,
            "sla_target_ms":            self.sla_target_ms,
        }


class BatchOptimizer:
    """
    Experiment with batch sizes and find the optimal configuration.

    Usage:
        optimizer = BatchOptimizer(sla_target_ms=500)
        results   = optimizer.run_experiments(model, prompts, batch_sizes=[1,4,8,16])
        best      = optimizer.find_optimal(results)
    """

    def __init__(self, sla_target_ms: float = 500.0):
        """
        Args:
            sla_target_ms: p95 latency SLA target in milliseconds
        """
        self.sla_target_ms = sla_target_ms
        self.results: List[BatchExperimentResult] = []

    def run_experiment(
        self,
        model,
        prompts: List[str],
        batch_config: BatchConfig,
    ) -> BatchExperimentResult:
        """
        Run a single batch size experiment.

        Args:
            model:        Loaded model instance (BaseModel)
            prompts:      List of prompts to benchmark
            batch_config: Batch configuration to test

        Returns:
            BatchExperimentResult
        """
        batch_size = batch_config.batch_size
        logger.info(f"Testing batch_size={batch_size}, strategy={batch_config.strategy}")

        latencies_ms: List[float] = []
        total_tokens = 0

        start_wall = time.time()

        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            t0 = time.time()
            responses = model.generate_batch(batch, max_tokens=256)
            t1 = time.time()

            batch_latency_ms = (t1 - t0) * 1000

            # Record per-request latency
            for _ in responses:
                latencies_ms.append(batch_latency_ms / len(batch))

            # Count tokens
            for r in responses:
                total_tokens += r.output_tokens

        end_wall = time.time()
        duration_sec = end_wall - start_wall

        # Compute statistics
        latencies = np.array(latencies_ms)
        p95 = float(np.percentile(latencies, 95))

        result = BatchExperimentResult(
            batch_size=batch_size,
            strategy=batch_config.strategy,
            mean_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=p95,
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_req_per_sec=len(prompts) / duration_sec,
            throughput_tokens_per_sec=total_tokens / duration_sec,
            meets_sla=p95 <= self.sla_target_ms,
            sla_target_ms=self.sla_target_ms,
        )

        logger.info(
            f"batch={batch_size}: p95={p95:.0f}ms, "
            f"throughput={result.throughput_tokens_per_sec:.1f} tok/s, "
            f"SLA={'✅' if result.meets_sla else '❌'}"
        )

        return result

    def run_experiments(
        self,
        model,
        prompts: List[str],
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        strategy: str = "static",
    ) -> List[BatchExperimentResult]:
        """
        Run experiments across multiple batch sizes.

        Args:
            model:       Loaded model instance
            prompts:     Benchmark prompts
            batch_sizes: List of batch sizes to test
            strategy:    Batching strategy

        Returns:
            List of BatchExperimentResult
        """
        logger.info(f"Running batch experiments: {batch_sizes}")
        self.results = []

        for bs in batch_sizes:
            config = BatchConfig(
                batch_size=bs,
                strategy=strategy,
                target_latency_ms=self.sla_target_ms,
            )
            result = self.run_experiment(model, prompts, config)
            self.results.append(result)

        return self.results

    def find_optimal(
        self,
        results: Optional[List[BatchExperimentResult]] = None,
    ) -> Optional[BatchExperimentResult]:
        """
        Find optimal batch size: largest batch that still meets SLA.

        Args:
            results: Results to analyse (uses self.results if None)

        Returns:
            Optimal BatchExperimentResult, or None if nothing meets SLA
        """
        results = results or self.results
        sla_passing = [r for r in results if r.meets_sla]

        if not sla_passing:
            logger.warning("No batch size meets the SLA target")
            return None

        # Best = highest throughput among SLA-passing configs
        best = max(sla_passing, key=lambda r: r.throughput_tokens_per_sec)
        logger.info(
            f"Optimal batch_size={best.batch_size}: "
            f"p95={best.p95_latency_ms:.0f}ms, "
            f"{best.throughput_tokens_per_sec:.1f} tok/s"
        )
        return best

    def summary_table(self) -> List[Dict[str, Any]]:
        """Return results as a list of dicts (easy to pass to pandas)"""
        return [r.to_dict() for r in self.results]