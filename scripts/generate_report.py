#!/usr/bin/env python3
"""
Generate Comprehensive Benchmark Report

Automatically generates a markdown report from benchmark results.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

sys.path.insert(0, ".")

from src.analysis import ParetoAnalyzer, CostModel, TCOAnalysis

def load_latest_results():
    """Load most recent benchmark results"""
    baseline_files = glob.glob("results/baseline/benchmark_results_*.csv")
    if not baseline_files:
        print("❌ No baseline results found")
        return None
    
    latest = sorted(baseline_files)[-1]
    print(f"Loading: {latest}")
    return pd.read_csv(latest)

def generate_report(df: pd.DataFrame, output_path: str = "docs/benchmark_report.md"):
    """Generate markdown report"""
    
    report = []
    report.append("# LLM Inference Benchmark Report")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Samples**: {len(df):,}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    models_tested = len(df['model_name'].unique())
    workloads = df['workload'].unique() if 'workload' in df.columns else ['mixed']
    
    report.append(f"This benchmark evaluated **{models_tested} LLM models** across ")
    report.append(f"**{len(workloads)} workload types**, measuring latency, throughput, ")
    report.append(f"memory usage, and cost per 1M tokens.")
    report.append("")
    
    # Key Findings
    report.append("## Key Findings")
    report.append("")
    
    # Fastest model
    fastest = df.loc[df['latency_ms'].idxmin()]
    report.append(f"- **Fastest Model**: {fastest['model_name']} ")
    report.append(f"({fastest['latency_ms']:.0f}ms average latency)")
    
    # Cheapest (if cost available)
    if 'cost_per_1m_tok' in df.columns:
        cheapest = df.loc[df['cost_per_1m_tok'].idxmin()]
        report.append(f"- **Most Cost-Effective**: {cheapest['model_name']} ")
        report.append(f"(${cheapest['cost_per_1m_tok']:.4f} per 1M tokens)")
    
    # Highest throughput
    if 'throughput' in df.columns:
        fastest_throughput = df.loc[df['throughput'].idxmax()]
        report.append(f"- **Highest Throughput**: {fastest_throughput['model_name']} ")
        report.append(f"({fastest_throughput['throughput']:.0f} tokens/sec)")
    
    report.append("")
    
    # Performance Summary Table
    report.append("## Performance Summary")
    report.append("")
    
    summary = df.groupby('model_name').agg({
        'latency_ms': ['mean', lambda x: x.quantile(0.95)],
        'input_tokens': 'sum',
        'output_tokens': 'sum',
    }).round(2)
    
    report.append("| Model | Avg Latency (ms) | p95 Latency (ms) | Total Tokens |")
    report.append("|-------|------------------|------------------|--------------|")
    
    for model in summary.index:
        avg_lat = summary.loc[model, ('latency_ms', 'mean')]
        p95_lat = summary.loc[model, ('latency_ms', '<lambda_0>')]
        total_tok = summary.loc[model, ('input_tokens', 'sum')] + summary.loc[model, ('output_tokens', 'sum')]
        report.append(f"| {model} | {avg_lat:.1f} | {p95_lat:.1f} | {total_tok:,.0f} |")
    
    report.append("")
    
    # Cost Analysis
    if 'cost_per_1m_tok' in df.columns:
        report.append("## Cost Analysis")
        report.append("")
        report.append("Monthly cost estimates for 100M tokens:")
        report.append("")
        
        cost_summary = df.groupby('model_name')['cost_per_1m_tok'].mean()
        
        report.append("| Model | Cost/1M Tokens | Monthly (100M) |")
        report.append("|-------|----------------|----------------|")
        
        for model, cost in cost_summary.items():
            monthly = cost * 100
            report.append(f"| {model} | ${cost:.4f} | ${monthly:,.2f} |")
        
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("### For Real-Time Chat (<200ms latency)")
    real_time = df[df['latency_ms'] < 200]
    if len(real_time) > 0:
        best = real_time.loc[real_time['latency_ms'].idxmin()]
        report.append(f"- Use **{best['model_name']}** with batch_size={best.get('batch_size', 1)}")
    else:
        report.append("- No models meet <200ms latency requirement")
    
    report.append("")
    report.append("### For Cost-Optimized Batch Processing")
    if 'cost_per_1m_tok' in df.columns:
        best_cost = df.loc[df['cost_per_1m_tok'].idxmin()]
        report.append(f"- Use **{best_cost['model_name']}** ")
        report.append(f"(${best_cost['cost_per_1m_tok']:.4f} per 1M tokens)")
    
    report.append("")
    report.append("### For Balanced Production Workloads")
    report.append("- **Recommended**: Llama-2-7B with INT8 quantization, batch_size=8-16")
    report.append("- Provides good balance of latency, cost, and quality")
    
    report.append("")
    
    # Methodology
    report.append("## Methodology")
    report.append("")
    report.append("### Test Configuration")
    report.append("")
    report.append("- **Infrastructure**: AWS g5.xlarge (NVIDIA A10G, 24GB VRAM)")
    report.append("- **Framework**: vLLM for self-hosted models, native APIs for cloud models")
    report.append(f"- **Workloads**: {', '.join(workloads)}")
    report.append(f"- **Total Requests**: {len(df):,}")
    report.append("")
    
    # Write report
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report))
    
    print(f"\n✅ Report generated: {output_path}")
    print(f"   {len(report)} lines")

def main():
    df = load_latest_results()
    if df is None:
        return
    
    generate_report(df)
    print("\n🎉 Report generation complete!")

if __name__ == "__main__":
    main()