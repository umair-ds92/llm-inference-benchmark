"""
Interactive Dashboard for LLM Benchmark Results

Run with: streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, ".")

from src.analysis import ParetoAnalyzer, ModelSelector, Requirements, WorkloadType


# Page config
st.set_page_config(
    page_title="LLM Inference Benchmark Dashboard",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 LLM Inference Benchmark Dashboard")
st.markdown("Interactive analysis of LLM inference performance, cost, and quality")

# Sidebar - Data Loading
st.sidebar.header("📂 Data")

# Load benchmark results
@st.cache_data
def load_data():
    """Load all benchmark results"""
    baseline_path = "results/baseline/benchmark_results_*.csv"
    import glob
    files = glob.glob(baseline_path)
    if files:
        return pd.read_csv(sorted(files)[-1])
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("No benchmark data found. Run benchmarks first!")
    st.stop()

st.sidebar.success(f"Loaded {len(df)} results")

# Filters
st.sidebar.header("🔍 Filters")
models = st.sidebar.multiselect(
    "Models",
    df['model_name'].unique(),
    default=df['model_name'].unique()[:3]
)

workloads = st.sidebar.multiselect(
    "Workloads",
    df['workload'].unique() if 'workload' in df.columns else ['all'],
    default=df['workload'].unique()[0] if 'workload' in df.columns else ['all']
)

# Filter data
df_filtered = df[df['model_name'].isin(models)]
if 'workload' in df.columns:
    df_filtered = df_filtered[df_filtered['workload'].isin(workloads)]

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💰 Cost Analysis", "⚡ Performance", "🎯 Recommendations"])

# Tab 1: Overview
with tab1:
    st.header("Benchmark Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Models Tested",
            len(df['model_name'].unique())
        )
    
    with col2:
        avg_latency = df_filtered['latency_ms'].mean()
        st.metric(
            "Avg Latency",
            f"{avg_latency:.0f}ms"
        )
    
    with col3:
        if 'cost_per_1m_tok' in df_filtered.columns:
            avg_cost = df_filtered['cost_per_1m_tok'].mean()
            st.metric(
                "Avg Cost/1M Tokens",
                f"${avg_cost:.4f}"
            )
    
    with col4:
        total_samples = len(df_filtered)
        st.metric(
            "Total Samples",
            f"{total_samples:,}"
        )
    
    # Latency distribution
    st.subheader("Latency Distribution by Model")
    fig = px.box(
        df_filtered,
        x='model_name',
        y='latency_ms',
        color='model_name',
        title="Latency Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Cost Analysis
with tab2:
    st.header("Cost Analysis")
    
    if 'cost_per_1m_tok' in df_filtered.columns:
        # Cost comparison bar chart
        cost_summary = df_filtered.groupby('model_name')['cost_per_1m_tok'].mean().sort_values()
        
        fig = go.Figure(data=[
            go.Bar(
                x=cost_summary.values,
                y=cost_summary.index,
                orientation='h',
                marker_color='lightblue'
            )
        ])
        fig.update_layout(
            title="Average Cost per 1M Tokens",
            xaxis_title="Cost (USD)",
            yaxis_title="Model"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly cost calculator
        st.subheader("💵 Monthly Cost Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_tokens = st.number_input(
                "Expected monthly tokens (millions)",
                min_value=1.0,
                max_value=10000.0,
                value=100.0,
                step=10.0
            )
        
        with col2:
            selected_model = st.selectbox(
                "Select model",
                df_filtered['model_name'].unique()
            )
        
        if selected_model:
            model_cost = df_filtered[df_filtered['model_name'] == selected_model]['cost_per_1m_tok'].mean()
            estimated_monthly = model_cost * monthly_tokens
            
            st.success(f"**Estimated monthly cost: ${estimated_monthly:,.2f}**")
    else:
        st.warning("Cost data not available in results")

# Tab 3: Performance
with tab3:
    st.header("Performance Analysis")
    
    # Batch size impact
    if 'batch_size' in df_filtered.columns:
        st.subheader("Batch Size Impact")
        
        fig = px.line(
            df_filtered.groupby(['model_name', 'batch_size'])['latency_ms'].mean().reset_index(),
            x='batch_size',
            y='latency_ms',
            color='model_name',
            markers=True,
            title="Latency vs Batch Size"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Throughput comparison
    if 'throughput' in df_filtered.columns or 'output_tokens' in df_filtered.columns:
        st.subheader("Throughput Comparison")
        
        throughput_data = df_filtered.groupby('model_name').agg({
            'output_tokens': 'sum',
            'latency_ms': 'sum'
        })
        throughput_data['tokens_per_sec'] = (throughput_data['output_tokens'] / (throughput_data['latency_ms'] / 1000))
        
        fig = px.bar(
            throughput_data.reset_index(),
            x='model_name',
            y='tokens_per_sec',
            title="Throughput (Tokens/Second)",
            color='model_name'
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Recommendations
with tab4:
    st.header("🎯 Model Recommendations")
    
    st.markdown("""
    Get personalized recommendations based on your requirements.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        workload_type = st.selectbox(
            "Workload Type",
            ["Interactive App", "Real-time Chat", "Batch Processing", "High Throughput"]
        )
        
        max_latency = st.slider(
            "Max p95 Latency (ms)",
            min_value=50,
            max_value=2000,
            value=500,
            step=50
        )
    
    with col2:
        monthly_budget = st.number_input(
            "Monthly Budget (USD)",
            min_value=100,
            max_value=100000,
            value=5000,
            step=500
        )
        
        min_quality = st.slider(
            "Min Quality Score",
            min_value=0.5,
            max_value=1.0,
            value=0.85,
            step=0.05
        )
    
    if st.button("Get Recommendation"):
        # Prepare data
        rec_df = df_filtered.copy()
        if 'cost_per_1m_tok' not in rec_df.columns:
            rec_df['cost_per_1m_tok'] = 0.005  # default
        if 'quality_score' not in rec_df.columns:
            rec_df['quality_score'] = 0.90  # default
        if 'throughput' not in rec_df.columns:
            rec_df['throughput'] = 50  # default
        
        # Map workload type
        workload_map = {
            "Interactive App": WorkloadType.INTERACTIVE_APP,
            "Real-time Chat": WorkloadType.REAL_TIME_CHAT,
            "Batch Processing": WorkloadType.BATCH_PROCESSING,
            "High Throughput": WorkloadType.HIGH_THROUGHPUT,
        }
        
        # Create requirements
        requirements = Requirements(
            workload_type=workload_map[workload_type],
            max_latency_p95_ms=max_latency,
            monthly_budget_usd=monthly_budget,
            min_quality_score=min_quality,
        )
        
        # Get recommendation
        try:
            selector = ModelSelector(rec_df)
            recommendation = selector.select(requirements)
            
            if recommendation.meets_requirements:
                st.success("✅ Found matching configuration!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model", recommendation.model_name)
                    st.metric("Quantization", recommendation.quantization)
                
                with col2:
                    st.metric("Batch Size", recommendation.batch_size)
                    st.metric("Infrastructure", recommendation.infrastructure)
                
                with col3:
                    st.metric(
                        "Est. Monthly Cost",
                        f"${recommendation.estimated_cost_monthly:,.2f}"
                    )
                
                st.subheader("Reasoning")
                for reason in recommendation.reasoning:
                    st.write(f"• {reason}")
            else:
                st.error("❌ No configuration meets all requirements")
                st.write("**Reasons:**")
                for reason in recommendation.reasoning:
                    st.write(f"• {reason}")
                
                st.info("💡 Try relaxing some constraints (budget, latency, or quality)")
        
        except Exception as e:
            st.error(f"Error generating recommendation: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • LLM Inference Benchmark Dashboard")