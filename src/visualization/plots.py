"""
Plotting Utilities for LLM Benchmarking

Creates publication-ready charts for benchmark results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from src.utils import get_logger

logger = get_logger(__name__)


class BenchmarkPlotter:
    """
    Generate standardized plots for benchmark results.
    
    Usage:
        plotter = BenchmarkPlotter(style="seaborn-v0_8")
        plotter.plot_pareto_frontier(df, save_path="pareto.png")
        plotter.plot_cost_comparison(df, save_path="costs.png")
    """
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid", figsize: Tuple[int, int] = (12, 6)):
        """
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 8)
    
    def plot_pareto_frontier(
        self,
        df: pd.DataFrame,
        x_col: str = "cost_per_1m_tok",
        y_col: str = "p95_latency_ms",
        label_col: str = "model_name",
        pareto_col: str = "is_pareto_optimal",
        save_path: Optional[str] = None,
    ):
        """
        Plot cost vs latency with Pareto frontier highlighted.
        
        Args:
            df:         DataFrame with benchmark results
            x_col:      Column for x-axis (cost)
            y_col:      Column for y-axis (latency)
            label_col:  Column for point labels
            pareto_col: Column indicating Pareto-optimal points
            save_path:  Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot all points
        non_pareto = df[~df[pareto_col]] if pareto_col in df.columns else df
        pareto = df[df[pareto_col]] if pareto_col in df.columns else pd.DataFrame()
        
        if len(non_pareto) > 0:
            ax.scatter(
                non_pareto[x_col], non_pareto[y_col],
                alpha=0.4, s=100, c='gray', label='Other configs'
            )
        
        if len(pareto) > 0:
            ax.scatter(
                pareto[x_col], pareto[y_col],
                alpha=0.9, s=200, c='red', marker='*',
                edgecolors='black', linewidths=1.5,
                label='Pareto optimal', zorder=10
            )
            
            # Draw Pareto frontier line
            pareto_sorted = pareto.sort_values(x_col)
            ax.plot(
                pareto_sorted[x_col], pareto_sorted[y_col],
                'r--', alpha=0.5, linewidth=2, zorder=5
            )
            
            # Label Pareto points
            for _, row in pareto.iterrows():
                ax.annotate(
                    row[label_col],
                    (row[x_col], row[y_col]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
                )
        
        ax.set_xlabel(f'{x_col} (USD)', fontsize=12)
        ax.set_ylabel(f'{y_col} (ms)', fontsize=12)
        ax.set_title('Cost-Performance Pareto Frontier', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_cost_comparison(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """Plot cost comparison across models"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Group by model and get mean cost
        cost_summary = df.groupby('model_name')['cost_per_1m_tok'].mean().sort_values()
        
        bars = ax.barh(cost_summary.index, cost_summary.values, color=self.colors[:len(cost_summary)])
        ax.set_xlabel('Cost per 1M Tokens (USD)', fontsize=12)
        ax.set_title('Cost Comparison Across Models', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width, bar.get_y() + bar.get_height()/2,
                f'${width:.4f}',
                ha='left', va='center', fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {save_path}")
        
        plt.show()
        return fig, ax
    
    def plot_throughput_vs_latency(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """Plot throughput vs latency trade-off"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            ax.scatter(
                model_data['p95_latency_ms'],
                model_data['throughput'],
                label=model, s=100, alpha=0.7
            )
        
        ax.set_xlabel('p95 Latency (ms)', fontsize=12)
        ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_title('Throughput vs Latency Trade-off', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {save_path}")
        
        plt.show()
        return fig, ax