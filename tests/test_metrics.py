"""
Tests for metrics collection module
"""

import pytest
import time
from src.metrics.collector import (
    MetricsCollector,
    LatencyMetrics,
    ThroughputMetrics,
    ResourceMetrics,
    CostMetrics,
    calculate_cost_metrics,
)


class TestLatencyMetrics:
    """Test LatencyMetrics dataclass"""
    
    def test_latency_metrics_creation(self):
        metrics = LatencyMetrics(
            time_to_first_token=0.05,
            inter_token_latency=[0.01, 0.012, 0.011],
            end_to_end_latency=0.5,
            queue_time=0.02
        )
        
        assert metrics.time_to_first_token == 0.05
        assert len(metrics.inter_token_latency) == 3
        assert metrics.end_to_end_latency == 0.5
        assert metrics.queue_time == 0.02
    
    def test_latency_metrics_to_dict(self):
        metrics = LatencyMetrics(
            time_to_first_token=0.05,
            inter_token_latency=[0.01, 0.012, 0.011],
            end_to_end_latency=0.5,
        )
        
        result = metrics.to_dict()
        assert "ttft_ms" in result
        assert "mean_itl_ms" in result
        assert "end_to_end_ms" in result
        assert result["ttft_ms"] == 50.0  # 0.05 * 1000


class TestThroughputMetrics:
    """Test ThroughputMetrics dataclass"""
    
    def test_throughput_calculation(self):
        metrics = ThroughputMetrics(
            total_tokens=1000,
            total_requests=10,
            duration_seconds=10.0
        )
        
        assert metrics.tokens_per_second == 100.0
        assert metrics.requests_per_second == 1.0
        assert metrics.tokens_per_request == 100.0
    
    def test_throughput_to_dict(self):
        metrics = ThroughputMetrics(
            total_tokens=1000,
            total_requests=10,
            duration_seconds=10.0
        )
        
        result = metrics.to_dict()
        assert result["tokens_per_second"] == 100.0
        assert result["requests_per_second"] == 1.0


class TestResourceMetrics:
    """Test ResourceMetrics dataclass"""
    
    def test_resource_metrics_percentages(self):
        metrics = ResourceMetrics(
            gpu_memory_used_mb=12000,
            gpu_memory_total_mb=24000,
            ram_used_gb=8.0,
            ram_total_gb=16.0
        )
        
        assert metrics.gpu_memory_percent == 50.0
        assert metrics.ram_percent == 50.0
    
    def test_resource_metrics_to_dict(self):
        metrics = ResourceMetrics(
            gpu_memory_used_mb=12000,
            gpu_memory_total_mb=24000,
            cpu_percent=45.0,
        )
        
        result = metrics.to_dict()
        assert "gpu_memory_used_mb" in result
        assert "gpu_memory_percent" in result
        assert "cpu_percent" in result


class TestCostMetrics:
    """Test CostMetrics dataclass"""
    
    def test_api_cost_calculation(self):
        metrics = CostMetrics(
            infrastructure_cost_per_hour=1.0,
            input_tokens=1_000_000,
            output_tokens=500_000,
            cost_per_1m_input_tokens=0.5,
            cost_per_1m_output_tokens=1.5,
            duration_seconds=3600
        )
        
        # API cost: (1M / 1M) * 0.5 + (0.5M / 1M) * 1.5 = 0.5 + 0.75 = 1.25
        assert metrics.api_cost == 1.25
        
        # Infrastructure cost: (3600s / 3600s) * $1.0 = $1.0
        assert metrics.infrastructure_cost == 1.0
        
        # Total cost
        assert metrics.total_cost == 2.25
    
    def test_cost_per_1m_tokens(self):
        metrics = CostMetrics(
            infrastructure_cost_per_hour=1.0,
            input_tokens=500_000,
            output_tokens=500_000,
            cost_per_1m_input_tokens=0.0,
            cost_per_1m_output_tokens=0.0,
            duration_seconds=3600
        )
        
        # Total tokens: 1M
        # Infrastructure cost: $1.0
        # Cost per 1M tokens: $1.0
        assert metrics.cost_per_1m_tokens == 1.0


class TestMetricsCollector:
    """Test MetricsCollector class"""
    
    def test_collector_initialization(self):
        collector = MetricsCollector(gpu_device_id=0)
        assert collector.gpu_device_id == 0
        assert len(collector.latency_samples) == 0
        assert len(collector.resource_samples) == 0
    
    def test_benchmark_lifecycle(self):
        collector = MetricsCollector()
        
        # Start benchmark
        collector.start_benchmark()
        assert collector.start_time is not None
        
        # Simulate some work
        time.sleep(0.1)
        
        # Record a latency sample
        latency = LatencyMetrics(
            time_to_first_token=0.05,
            inter_token_latency=[0.01, 0.01],
            end_to_end_latency=0.1
        )
        collector.record_latency(latency)
        
        # End benchmark
        collector.end_benchmark()
        assert collector.end_time is not None
        assert len(collector.latency_samples) == 1
    
    def test_percentile_calculation(self):
        collector = MetricsCollector()
        collector.start_benchmark()
        
        # Add multiple samples
        for i in range(100):
            latency = LatencyMetrics(
                time_to_first_token=0.05 + (i * 0.001),
                inter_token_latency=[0.01],
                end_to_end_latency=0.1 + (i * 0.001)
            )
            collector.record_latency(latency)
        
        percentiles = collector.compute_latency_percentiles([50, 95, 99])
        
        assert "ttft_ms" in percentiles
        assert "p50" in percentiles["ttft_ms"]
        assert "p95" in percentiles["ttft_ms"]
        assert "p99" in percentiles["ttft_ms"]
    
    def test_throughput_computation(self):
        collector = MetricsCollector()
        collector.start_benchmark()
        
        # Add some samples
        for _ in range(10):
            latency = LatencyMetrics(
                time_to_first_token=0.05,
                inter_token_latency=[0.01],
                end_to_end_latency=0.1
            )
            collector.record_latency(latency)
        
        time.sleep(0.1)
        collector.end_benchmark()
        
        throughput = collector.compute_throughput()
        assert throughput.total_requests == 10
        assert throughput.duration_seconds > 0


def test_calculate_cost_metrics():
    """Test cost calculation helper function"""
    
    model_config = {
        "cost_per_hour": 1.0,
        "cost_per_1m_tokens_input": 0.5,
        "cost_per_1m_tokens_output": 1.5,
    }
    
    cost_metrics = calculate_cost_metrics(
        model_config=model_config,
        input_tokens=1_000_000,
        output_tokens=500_000,
        duration_seconds=3600
    )
    
    assert cost_metrics.infrastructure_cost_per_hour == 1.0
    assert cost_metrics.input_tokens == 1_000_000
    assert cost_metrics.output_tokens == 500_000
    assert cost_metrics.api_cost == 1.25
    assert cost_metrics.infrastructure_cost == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])