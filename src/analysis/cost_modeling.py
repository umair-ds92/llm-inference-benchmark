"""
Cost Modeling for LLM Inference

Calculates Total Cost of Ownership (TCO) including:
- Infrastructure costs (GPU instances)
- API costs (per-token pricing)
- Engineering/ops costs (setup, maintenance)
- Power/cooling costs

Enables apples-to-apples comparison between self-hosted and API models.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CostModel:
    """
    Cost model for a single deployment configuration.
    
    Attributes:
        model_name:               Model identifier
        deployment_type:          "self_hosted" | "api"
        
        # Infrastructure costs (self-hosted only)
        gpu_hourly_rate:          $/hour for GPU instance
        gpu_utilization:          Fraction of time GPU is active
        
        # API costs
        cost_per_1m_input_tokens: $/1M input tokens
        cost_per_1m_output_tokens:$/1M output tokens
        
        # Other costs
        engineering_monthly:      Monthly eng/ops overhead
        power_monthly:            Additional power/cooling costs
    """
    model_name: str
    deployment_type: str  # "self_hosted" | "api"
    
    # Infrastructure
    gpu_hourly_rate: float = 0.0
    gpu_utilization: float = 1.0  # 100% = always on
    
    # API
    cost_per_1m_input_tokens: float = 0.0
    cost_per_1m_output_tokens: float = 0.0
    
    # Ops
    engineering_monthly: float = 0.0
    power_monthly: float = 0.0
    
    def calculate_monthly_cost(
        self,
        monthly_input_tokens_millions: float,
        monthly_output_tokens_millions: float,
    ) -> Dict[str, float]:
        """
        Calculate total monthly cost.
        
        Args:
            monthly_input_tokens_millions:  Input tokens in millions
            monthly_output_tokens_millions: Output tokens in millions
        
        Returns:
            Dict with cost breakdown
        """
        # Infrastructure
        hours_per_month = 730  # ~30.4 days
        infra_cost = self.gpu_hourly_rate * self.gpu_utilization * hours_per_month
        
        # API
        api_cost = (
            monthly_input_tokens_millions * self.cost_per_1m_input_tokens +
            monthly_output_tokens_millions * self.cost_per_1m_output_tokens
        )
        
        # Total
        total_cost = infra_cost + api_cost + self.engineering_monthly + self.power_monthly
        
        return {
            "infrastructure_cost": round(infra_cost, 2),
            "api_cost": round(api_cost, 2),
            "engineering_cost": round(self.engineering_monthly, 2),
            "power_cost": round(self.power_monthly, 2),
            "total_monthly_cost": round(total_cost, 2),
        }


@dataclass
class TCOAnalysis:
    """
    Total Cost of Ownership comparison.
    
    Usage:
        tco = TCOAnalysis(
            baseline_model=gpt35_model,
            optimized_model=llama_model,
            monthly_tokens=100,
        )
        savings = tco.calculate_savings()
        print(f"Save ${savings['monthly_savings']}/mo")
    """
    baseline_model: CostModel
    optimized_model: CostModel
    monthly_input_tokens_millions: float
    monthly_output_tokens_millions: float
    
    def calculate_savings(self) -> Dict[str, Any]:
        """Calculate cost savings and ROI"""
        baseline = self.baseline_model.calculate_monthly_cost(
            self.monthly_input_tokens_millions,
            self.monthly_output_tokens_millions,
        )
        optimized = self.optimized_model.calculate_monthly_cost(
            self.monthly_input_tokens_millions,
            self.monthly_output_tokens_millions,
        )
        
        monthly_savings = baseline["total_monthly_cost"] - optimized["total_monthly_cost"]
        savings_pct = (monthly_savings / baseline["total_monthly_cost"]) * 100
        annual_savings = monthly_savings * 12
        
        return {
            "baseline_monthly": baseline["total_monthly_cost"],
            "optimized_monthly": optimized["total_monthly_cost"],
            "monthly_savings": round(monthly_savings, 2),
            "annual_savings": round(annual_savings, 2),
            "savings_pct": round(savings_pct, 1),
            "baseline_breakdown": baseline,
            "optimized_breakdown": optimized,
        }
    
    def print_summary(self):
        """Print formatted TCO summary"""
        savings = self.calculate_savings()
        
        print("=" * 65)
        print("TOTAL COST OF OWNERSHIP ANALYSIS")
        print("=" * 65)
        print(f"\nBaseline:  {self.baseline_model.model_name}")
        print(f"Optimized: {self.optimized_model.model_name}")
        print(f"\nMonthly volume: {self.monthly_input_tokens_millions + self.monthly_output_tokens_millions:.0f}M tokens")
        print("\nCosts:")
        print(f"  Baseline:  ${savings['baseline_monthly']:>8,.2f}/month")
        print(f"  Optimized: ${savings['optimized_monthly']:>8,.2f}/month")
        print(f"  Savings:   ${savings['monthly_savings']:>8,.2f}/month ({savings['savings_pct']:.1f}%)")
        print(f"\n  Annual:    ${savings['annual_savings']:>8,.2f}/year")
        print("=" * 65)