"""
Pareto Frontier Analysis for LLM Inference

A Pareto frontier shows the set of "optimal" configurations where you
cannot improve one metric (e.g., cost) without worsening another (e.g., latency).

Example: If Model A is faster AND cheaper than Model B, then B is "dominated"
         and not on the Pareto frontier.

This helps visualize the trade-off space and identify the best options.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ParetoPoint:
    """
    A single point on the Pareto frontier.
    
    Attributes:
        model_name:      Model identifier
        config:          Configuration details (quantization, batch_size, etc.)
        latency_ms:      p95 latency (lower is better)
        cost_per_1m_tok: Cost per 1M tokens in USD (lower is better)
        throughput:      Tokens per second (higher is better)
        quality_score:   Task accuracy or quality metric (higher is better)
        is_pareto_optimal: True if on the Pareto frontier
    """
    model_name: str
    config: Dict[str, Any]
    latency_ms: float
    cost_per_1m_tok: float
    throughput: float
    quality_score: float
    is_pareto_optimal: bool = False
    
    def __repr__(self):
        return (
            f"<ParetoPoint {self.model_name}: "
            f"{self.latency_ms:.0f}ms, ${self.cost_per_1m_tok:.4f}/1M tok, "
            f"{self.throughput:.0f} tok/s>"
        )


class ParetoAnalyzer:
    """
    Identify Pareto-optimal configurations from benchmark results.
    
    Usage:
        analyzer = ParetoAnalyzer()
        pareto_set = analyzer.find_pareto_frontier(
            results_df,
            objectives=["latency_ms", "cost_per_1m_tok"],
            minimize=[True, True]
        )
        analyzer.plot_frontier(pareto_set)
    """
    
    def __init__(self):
        self.pareto_points: List[ParetoPoint] = []
    
    def find_pareto_frontier(
        self,
        df: pd.DataFrame,
        objectives: List[str] = ["latency_ms", "cost_per_1m_tok"],
        minimize: List[bool] = [True, True],
        quality_threshold: float = 0.85,
    ) -> List[ParetoPoint]:
        """
        Find Pareto-optimal points from benchmark results.
        
        Args:
            df:                DataFrame with benchmark results
            objectives:        Metric columns to optimize
            minimize:          For each objective, True = minimize, False = maximize
            quality_threshold: Minimum quality score to consider
        
        Returns:
            List of ParetoPoint objects (pareto-optimal ones marked)
        """
        # Filter by quality threshold
        if "quality_score" in df.columns:
            df = df[df["quality_score"] >= quality_threshold].copy()
        
        if len(df) == 0:
            logger.warning("No data points meet quality threshold")
            return []
        
        # Extract objective values
        points = df[objectives].values
        
        # Flip maximize objectives (so we can treat everything as minimization)
        for i, should_min in enumerate(minimize):
            if not should_min:
                points[:, i] = -points[:, i]
        
        # Find Pareto-optimal points
        is_pareto = np.ones(len(points), dtype=bool)
        
        for i, point in enumerate(points):
            # A point is dominated if another point is better in ALL objectives
            if is_pareto[i]:
                # Compare against all other points
                dominated_by = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
                is_pareto[dominated_by] = False
        
        # Build ParetoPoint objects
        results = []
        for idx, row in df.iterrows():
            pt = ParetoPoint(
                model_name=row.get("model_name", "unknown"),
                config={
                    "quantization": row.get("quantization", "unknown"),
                    "batch_size": row.get("batch_size", 1),
                },
                latency_ms=row.get("latency_ms", 0),
                cost_per_1m_tok=row.get("cost_per_1m_tok", 0),
                throughput=row.get("throughput", 0),
                quality_score=row.get("quality_score", 0),
                is_pareto_optimal=is_pareto[len(results)],
            )
            results.append(pt)
        
        pareto_optimal = [p for p in results if p.is_pareto_optimal]
        logger.info(f"Found {len(pareto_optimal)} Pareto-optimal configurations")
        
        self.pareto_points = results
        return pareto_optimal
    
    def get_pareto_df(self) -> pd.DataFrame:
        """Return Pareto analysis as DataFrame"""
        data = []
        for pt in self.pareto_points:
            data.append({
                "model_name": pt.model_name,
                "quantization": pt.config.get("quantization"),
                "batch_size": pt.config.get("batch_size"),
                "latency_ms": pt.latency_ms,
                "cost_per_1m_tok": pt.cost_per_1m_tok,
                "throughput": pt.throughput,
                "quality_score": pt.quality_score,
                "is_pareto_optimal": pt.is_pareto_optimal,
            })
        return pd.DataFrame(data)
    
    def recommend_by_priority(
        self,
        priority: str = "cost",
    ) -> Optional[ParetoPoint]:
        """
        Recommend the best configuration based on priority.
        
        Args:
            priority: "cost" | "latency" | "throughput" | "balanced"
        
        Returns:
            Recommended ParetoPoint
        """
        pareto = [p for p in self.pareto_points if p.is_pareto_optimal]
        
        if not pareto:
            logger.warning("No Pareto-optimal points to recommend")
            return None
        
        if priority == "cost":
            best = min(pareto, key=lambda p: p.cost_per_1m_tok)
        elif priority == "latency":
            best = min(pareto, key=lambda p: p.latency_ms)
        elif priority == "throughput":
            best = max(pareto, key=lambda p: p.throughput)
        elif priority == "balanced":
            # Normalize and find best average rank
            scores = []
            for p in pareto:
                latency_rank = sorted(pareto, key=lambda x: x.latency_ms).index(p)
                cost_rank = sorted(pareto, key=lambda x: x.cost_per_1m_tok).index(p)
                throughput_rank = sorted(pareto, key=lambda x: -x.throughput).index(p)
                avg_rank = (latency_rank + cost_rank + throughput_rank) / 3
                scores.append((avg_rank, p))
            best = min(scores, key=lambda x: x[0])[1]
        else:
            raise ValueError(f"Unknown priority: {priority}")
        
        logger.info(f"Recommendation ({priority}): {best.model_name}")
        return best
    
    def calculate_savings(
        self,
        baseline_point: ParetoPoint,
        optimized_point: ParetoPoint,
        monthly_tokens_millions: float = 100.0,
    ) -> Dict[str, float]:
        """
        Calculate cost savings from optimization.
        
        Args:
            baseline_point:         Reference configuration (e.g., GPT-3.5)
            optimized_point:        Optimized configuration (e.g., Llama-2 INT8)
            monthly_tokens_millions: Expected monthly token volume
        
        Returns:
            Dict with savings calculations
        """
        baseline_monthly = baseline_point.cost_per_1m_tok * monthly_tokens_millions
        optimized_monthly = optimized_point.cost_per_1m_tok * monthly_tokens_millions
        savings_monthly = baseline_monthly - optimized_monthly
        savings_pct = (savings_monthly / baseline_monthly) * 100 if baseline_monthly > 0 else 0
        
        return {
            "baseline_monthly_usd": round(baseline_monthly, 2),
            "optimized_monthly_usd": round(optimized_monthly, 2),
            "savings_monthly_usd": round(savings_monthly, 2),
            "savings_pct": round(savings_pct, 1),
            "monthly_tokens_millions": monthly_tokens_millions,
            "payback_period_months": 0,  # Assuming no upfront cost
        }