"""
Metrics Collection System for LLM Inference Benchmarking

This module provides comprehensive metrics collection for benchmarking LLM inference,
including latency, throughput, resource utilization, and cost tracking.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import psutil
import numpy as np
from datetime import datetime

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False


@dataclass
class LatencyMetrics:
    """Latency metrics for a single inference request"""
    time_to_first_token: float  # seconds
    inter_token_latency: List[float]  # seconds per token
    end_to_end_latency: float  # seconds
    queue_time: Optional[float] = None  # seconds waiting in queue
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttft_ms": self.time_to_first_token * 1000,
            "mean_itl_ms": np.mean(self.inter_token_latency) * 1000 if self.inter_token_latency else 0,
            "end_to_end_ms": self.end_to_end_latency * 1000,
            "queue_time_ms": self.queue_time * 1000 if self.queue_time else 0,
        }


@dataclass
class ThroughputMetrics:
    """Throughput metrics for a benchmark run"""
    total_tokens: int
    total_requests: int
    duration_seconds: float
    
    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.duration_seconds if self.duration_seconds > 0 else 0
    
    @property
    def requests_per_second(self) -> float:
        return self.total_requests / self.duration_seconds if self.duration_seconds > 0 else 0
    
    @property
    def tokens_per_request(self) -> float:
        return self.total_tokens / self.total_requests if self.total_requests > 0 else 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "tokens_per_second": self.tokens_per_second,
            "requests_per_second": self.requests_per_second,
            "tokens_per_request": self.tokens_per_request,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ResourceMetrics:
    """System resource utilization metrics"""
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def gpu_memory_percent(self) -> float:
        if self.gpu_memory_total_mb > 0:
            return (self.gpu_memory_used_mb / self.gpu_memory_total_mb) * 100
        return 0.0
    
    @property
    def ram_percent(self) -> float:
        if self.ram_total_gb > 0:
            return (self.ram_used_gb / self.ram_total_gb) * 100
        return 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "gpu_memory_percent": self.gpu_memory_percent,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "cpu_percent": self.cpu_percent,
            "ram_used_gb": self.ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
            "ram_percent": self.ram_percent,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CostMetrics:
    """Cost tracking for inference"""
    infrastructure_cost_per_hour: float
    input_tokens: int
    output_tokens: int
    cost_per_1m_input_tokens: float = 0.0  # For API models
    cost_per_1m_output_tokens: float = 0.0  # For API models
    duration_seconds: float = 0.0
    
    @property
    def api_cost(self) -> float:
        """Calculate API cost for API-based models"""
        input_cost = (self.input_tokens / 1_000_000) * self.cost_per_1m_input_tokens
        output_cost = (self.output_tokens / 1_000_000) * self.cost_per_1m_output_tokens
        return input_cost + output_cost
    
    @property
    def infrastructure_cost(self) -> float:
        """Calculate infrastructure cost for self-hosted models"""
        return (self.duration_seconds / 3600) * self.infrastructure_cost_per_hour
    
    @property
    def total_cost(self) -> float:
        """Total cost (API + infrastructure)"""
        return self.api_cost + self.infrastructure_cost
    
    @property
    def cost_per_1m_tokens(self) -> float:
        """Normalized cost per 1M tokens"""
        total_tokens = self.input_tokens + self.output_tokens
        if total_tokens > 0:
            return (self.total_cost / total_tokens) * 1_000_000
        return 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "api_cost_usd": self.api_cost,
            "infrastructure_cost_usd": self.infrastructure_cost,
            "total_cost_usd": self.total_cost,
            "cost_per_1m_tokens_usd": self.cost_per_1m_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class MetricsCollector:
    """
    Main metrics collection class for benchmarking.
    
    Tracks latency, throughput, resource utilization, and costs.
    """
    
    def __init__(self, gpu_device_id: int = 0):
        self.gpu_device_id = gpu_device_id
        self.latency_samples: List[LatencyMetrics] = []
        self.resource_samples: List[ResourceMetrics] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def start_benchmark(self) -> None:
        """Mark the start of a benchmark run"""
        self.start_time = time.time()
        self.latency_samples = []
        self.resource_samples = []
        
    def end_benchmark(self) -> None:
        """Mark the end of a benchmark run"""
        self.end_time = time.time()
        
    def record_latency(self, metrics: LatencyMetrics) -> None:
        """Record latency metrics for a single request"""
        self.latency_samples.append(metrics)
        
    def collect_resource_metrics(self) -> ResourceMetrics:
        """
        Collect current system resource utilization.
        
        Returns:
            ResourceMetrics object with current utilization
        """
        metrics = ResourceMetrics()
        
        # CPU and RAM
        metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        metrics.ram_used_gb = ram.used / (1024**3)
        metrics.ram_total_gb = ram.total / (1024**3)
        
        # GPU metrics
        if NVML_AVAILABLE:
            try:
                handle = nvmlDeviceGetHandleByIndex(self.gpu_device_id)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                metrics.gpu_memory_used_mb = mem_info.used / (1024**2)
                metrics.gpu_memory_total_mb = mem_info.total / (1024**2)
            except Exception as e:
                print(f"Warning: Could not collect GPU memory metrics: {e}")
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus and len(gpus) > self.gpu_device_id:
                    gpu = gpus[self.gpu_device_id]
                    metrics.gpu_utilization_percent = gpu.load * 100
            except Exception as e:
                print(f"Warning: Could not collect GPU utilization: {e}")
        
        self.resource_samples.append(metrics)
        return metrics
        
    def compute_latency_percentiles(self, percentiles: List[int] = [50, 90, 95, 99]) -> Dict[str, Dict[int, float]]:
        """
        Compute latency percentiles across all samples.
        
        Args:
            percentiles: List of percentiles to compute (e.g., [50, 90, 95, 99])
            
        Returns:
            Dictionary with percentiles for each latency metric
        """
        if not self.latency_samples:
            return {}
        
        ttft_values = [s.time_to_first_token * 1000 for s in self.latency_samples]
        e2e_values = [s.end_to_end_latency * 1000 for s in self.latency_samples]
        
        result = {
            "ttft_ms": {},
            "end_to_end_ms": {}
        }
        
        for p in percentiles:
            result["ttft_ms"][f"p{p}"] = np.percentile(ttft_values, p)
            result["end_to_end_ms"][f"p{p}"] = np.percentile(e2e_values, p)
            
        return result
        
    def compute_throughput(self) -> ThroughputMetrics:
        """
        Compute throughput metrics from collected samples.
        
        Returns:
            ThroughputMetrics object
        """
        if not self.start_time or not self.end_time:
            raise ValueError("Benchmark not started or ended properly")
        
        duration = self.end_time - self.start_time
        total_requests = len(self.latency_samples)
        
        # Calculate total tokens (input + output)
        # This is a simplified version - actual implementation should track tokens per request
        total_tokens = total_requests * 200  # Placeholder: average tokens per request
        
        return ThroughputMetrics(
            total_tokens=total_tokens,
            total_requests=total_requests,
            duration_seconds=duration
        )
        
    def compute_average_resources(self) -> ResourceMetrics:
        """
        Compute average resource utilization across all samples.
        
        Returns:
            ResourceMetrics with averaged values
        """
        if not self.resource_samples:
            return ResourceMetrics()
        
        avg_metrics = ResourceMetrics(
            gpu_memory_used_mb=np.mean([r.gpu_memory_used_mb for r in self.resource_samples]),
            gpu_memory_total_mb=self.resource_samples[0].gpu_memory_total_mb,
            gpu_utilization_percent=np.mean([r.gpu_utilization_percent for r in self.resource_samples]),
            cpu_percent=np.mean([r.cpu_percent for r in self.resource_samples]),
            ram_used_gb=np.mean([r.ram_used_gb for r in self.resource_samples]),
            ram_total_gb=self.resource_samples[0].ram_total_gb,
        )
        
        return avg_metrics
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a complete summary of all collected metrics.
        
        Returns:
            Dictionary with all metrics
        """
        summary = {
            "latency": self.compute_latency_percentiles(),
            "throughput": self.compute_throughput().to_dict(),
            "resources": self.compute_average_resources().to_dict(),
            "num_samples": len(self.latency_samples),
        }
        
        return summary


def calculate_cost_metrics(
    model_config: Dict[str, Any],
    input_tokens: int,
    output_tokens: int,
    duration_seconds: float
) -> CostMetrics:
    """
    Calculate cost metrics for a benchmark run.
    
    Args:
        model_config: Model configuration from models.yaml
        input_tokens: Number of input tokens processed
        output_tokens: Number of output tokens generated
        duration_seconds: Duration of the benchmark run
        
    Returns:
        CostMetrics object
    """
    # Infrastructure cost (for self-hosted models)
    infra_cost = model_config.get("cost_per_hour", 0.0)
    
    # API costs (for API-based models)
    api_input_cost = model_config.get("cost_per_1m_tokens_input", 0.0)
    api_output_cost = model_config.get("cost_per_1m_tokens_output", 0.0)
    
    return CostMetrics(
        infrastructure_cost_per_hour=infra_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_per_1m_input_tokens=api_input_cost,
        cost_per_1m_output_tokens=api_output_cost,
        duration_seconds=duration_seconds
    )
