"""
Model Selection Decision Tree

Helps users choose the right model based on their requirements.

Decision criteria:
- Latency SLA (real-time vs batch)
- Budget constraints
- Volume (tokens/month)
- Quality requirements
- Infrastructure preferences (self-hosted vs API)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


class WorkloadType(Enum):
    """Workload characteristics"""
    REAL_TIME_CHAT = "real_time_chat"           # <200ms p95
    INTERACTIVE_APP = "interactive_app"          # <500ms p95
    BATCH_PROCESSING = "batch_processing"        # >1s ok
    HIGH_THROUGHPUT = "high_throughput"          # Volume matters


@dataclass
class Requirements:
    """
    User requirements for model selection.
    
    Attributes:
        workload_type:          Type of workload
        max_latency_p95_ms:     Maximum acceptable p95 latency
        monthly_budget_usd:     Monthly budget constraint
        monthly_tokens_millions:Expected token volume
        min_quality_score:      Minimum acceptable quality (0-1)
        prefer_self_hosted:     Prefer self-hosting over APIs
    """
    workload_type: WorkloadType
    max_latency_p95_ms: float = 500.0
    monthly_budget_usd: float = 10000.0
    monthly_tokens_millions: float = 100.0
    min_quality_score: float = 0.85
    prefer_self_hosted: bool = False


@dataclass
class DeploymentRecommendation:
    """
    Recommended deployment configuration.
    
    Attributes:
        model_name:       Recommended model
        quantization:     Quantization level
        batch_size:       Optimal batch size
        infrastructure:   Instance type recommendation
        estimated_cost:   Monthly cost estimate
        meets_requirements: Whether it meets all requirements
        reasoning:        Explanation of recommendation
    """
    model_name: str
    quantization: str
    batch_size: int
    infrastructure: str
    estimated_cost_monthly: float
    meets_requirements: bool
    reasoning: List[str]
    
    def print_summary(self):
        """Print formatted recommendation"""
        print("=" * 70)
        print("DEPLOYMENT RECOMMENDATION")
        print("=" * 70)
        print(f"\nModel:          {self.model_name}")
        print(f"Quantization:   {self.quantization}")
        print(f"Batch size:     {self.batch_size}")
        print(f"Infrastructure: {self.infrastructure}")
        print(f"Est. cost:      ${self.estimated_cost_monthly:,.2f}/month")
        print(f"\nMeets requirements: {'✅ YES' if self.meets_requirements else '❌ NO'}")
        print("\nReasoning:")
        for reason in self.reasoning:
            print(f"  • {reason}")
        print("=" * 70)


class ModelSelector:
    """
    Select optimal model based on requirements.
    
    Usage:
        selector = ModelSelector(benchmark_results_df)
        
        requirements = Requirements(
            workload_type=WorkloadType.INTERACTIVE_APP,
            max_latency_p95_ms=500,
            monthly_budget_usd=5000,
        )
        
        recommendation = selector.select(requirements)
        recommendation.print_summary()
    """
    
    def __init__(self, benchmark_df: pd.DataFrame):
        """
        Args:
            benchmark_df: DataFrame with all benchmark results
        """
        self.df = benchmark_df
    
    def select(self, req: Requirements) -> DeploymentRecommendation:
        """
        Select best model configuration based on requirements.
        
        Args:
            req: Requirements object
        
        Returns:
            DeploymentRecommendation
        """
        reasoning = []
        
        # Filter by latency SLA
        df = self.df[self.df["p95_latency_ms"] <= req.max_latency_p95_ms].copy()
        reasoning.append(
            f"Filtered to p95 latency ≤ {req.max_latency_p95_ms}ms "
            f"({len(df)}/{len(self.df)} configs)"
        )
        
        if len(df) == 0:
            return self._no_match_recommendation(req, "No configs meet latency SLA")
        
        # Filter by quality
        if "quality_score" in df.columns:
            df = df[df["quality_score"] >= req.min_quality_score]
            reasoning.append(
                f"Filtered to quality ≥ {req.min_quality_score:.0%} "
                f"({len(df)} configs)"
            )
        
        if len(df) == 0:
            return self._no_match_recommendation(req, "No configs meet quality threshold")
        
        # Calculate cost
        df["monthly_cost"] = df["cost_per_1m_tok"] * req.monthly_tokens_millions
        
        # Filter by budget
        df = df[df["monthly_cost"] <= req.monthly_budget_usd]
        reasoning.append(
            f"Filtered to monthly cost ≤ ${req.monthly_budget_usd:,.0f} "
            f"({len(df)} configs)"
        )
        
        if len(df) == 0:
            return self._no_match_recommendation(req, "No configs meet budget constraint")
        
        # Prefer self-hosted if requested
        if req.prefer_self_hosted:
            self_hosted = df[df["deployment_type"] == "self_hosted"]
            if len(self_hosted) > 0:
                df = self_hosted
                reasoning.append("Preferring self-hosted deployments")
        
        # Select best based on workload type
        if req.workload_type == WorkloadType.REAL_TIME_CHAT:
            # Minimize latency
            best = df.loc[df["p95_latency_ms"].idxmin()]
            reasoning.append("Optimizing for lowest latency (real-time chat)")
        
        elif req.workload_type == WorkloadType.INTERACTIVE_APP:
            # Balance latency and cost
            df["score"] = (
                (df["p95_latency_ms"] / df["p95_latency_ms"].max()) * 0.6 +
                (df["monthly_cost"] / df["monthly_cost"].max()) * 0.4
            )
            best = df.loc[df["score"].idxmin()]
            reasoning.append("Balancing latency (60%) and cost (40%)")
        
        elif req.workload_type == WorkloadType.BATCH_PROCESSING:
            # Minimize cost
            best = df.loc[df["monthly_cost"].idxmin()]
            reasoning.append("Optimizing for lowest cost (batch processing)")
        
        elif req.workload_type == WorkloadType.HIGH_THROUGHPUT:
            # Maximize throughput
            best = df.loc[df["throughput"].idxmax()]
            reasoning.append("Optimizing for maximum throughput")
        
        else:
            # Default: minimize cost
            best = df.loc[df["monthly_cost"].idxmin()]
            reasoning.append("Default: optimizing for cost")
        
        # Build recommendation
        return DeploymentRecommendation(
            model_name=best.get("model_name", "unknown"),
            quantization=best.get("quantization", "fp16"),
            batch_size=int(best.get("batch_size", 8)),
            infrastructure=best.get("infrastructure", "g5.xlarge"),
            estimated_cost_monthly=round(best["monthly_cost"], 2),
            meets_requirements=True,
            reasoning=reasoning,
        )
    
    def _no_match_recommendation(
        self,
        req: Requirements,
        reason: str,
    ) -> DeploymentRecommendation:
        """Return a recommendation when no configs meet requirements"""
        return DeploymentRecommendation(
            model_name="No match",
            quantization="N/A",
            batch_size=0,
            infrastructure="N/A",
            estimated_cost_monthly=0.0,
            meets_requirements=False,
            reasoning=[reason, "Consider relaxing constraints"],
        )