"""
KV Cache Optimization for LLM Inference

The KV (Key-Value) cache stores attention keys and values between
token generations, avoiding recomputation. Proper configuration
significantly impacts latency and throughput.

Strategies:
- DEFAULT:   vLLM's built-in paged attention (good baseline)
- OPTIMIZED: Tuned gpu_memory_utilization for max cache size
- DISABLED:  No caching (baseline comparison only)
"""

from dataclasses import dataclass
from typing import Dict, Any
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class KVCacheConfig:
    """
    Configuration for KV cache behaviour.

    Attributes:
        name:                   Friendly label
        gpu_memory_utilization: Fraction of GPU VRAM reserved for vLLM
                                (higher = larger KV cache, but risks OOM)
        max_num_seqs:           Max concurrent sequences in a batch
        max_num_batched_tokens: Max total tokens across the batch
        swap_space_gb:          CPU RAM used as overflow cache (GB)
        description:            Human-readable notes
    """
    name: str
    gpu_memory_utilization: float
    max_num_seqs: int
    max_num_batched_tokens: int
    swap_space_gb: int
    description: str = ""

    def to_vllm_kwargs(self) -> Dict[str, Any]:
        """Return kwargs to pass directly to vLLM LLM()"""
        return {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs":           self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "swap_space":             self.swap_space_gb,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":                   self.name,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs":           self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "swap_space_gb":          self.swap_space_gb,
            "description":            self.description,
        }


# Pre-defined KV cache configurations
KVCACHE_CONFIGS: Dict[str, KVCacheConfig] = {
    "default": KVCacheConfig(
        name="Default",
        gpu_memory_utilization=0.90,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        swap_space_gb=4,
        description="vLLM defaults. Good general-purpose baseline.",
    ),
    "optimized": KVCacheConfig(
        name="Optimized",
        gpu_memory_utilization=0.95,
        max_num_seqs=512,
        max_num_batched_tokens=16384,
        swap_space_gb=8,
        description=(
            "Higher memory utilization for larger KV cache. "
            "Better throughput but less headroom before OOM."
        ),
    ),
    "conservative": KVCacheConfig(
        name="Conservative",
        gpu_memory_utilization=0.75,
        max_num_seqs=128,
        max_num_batched_tokens=4096,
        swap_space_gb=2,
        description=(
            "Lower utilization for stability. "
            "Use when running multiple services on same GPU."
        ),
    ),
}


def get_kvcache_config(name: str) -> KVCacheConfig:
    """
    Get KV cache configuration by name.

    Args:
        name: Config name ('default', 'optimized', 'conservative')

    Returns:
        KVCacheConfig instance
    """
    name = name.lower()
    if name not in KVCACHE_CONFIGS:
        raise ValueError(
            f"Unknown KV cache config: '{name}'. "
            f"Available: {list(KVCACHE_CONFIGS.keys())}"
        )
    return KVCACHE_CONFIGS[name]


def estimate_kvcache_size(
    gpu_memory_gb: float,
    model_memory_gb: float,
    gpu_memory_utilization: float,
) -> Dict[str, float]:
    """
    Estimate available KV cache size.

    Args:
        gpu_memory_gb:          Total GPU VRAM (e.g. 24 for A10G)
        model_memory_gb:        Memory used by model weights
        gpu_memory_utilization: Fraction reserved by vLLM

    Returns:
        Dict with cache size estimates
    """
    reserved_gb    = gpu_memory_gb * gpu_memory_utilization
    available_gb   = reserved_gb - model_memory_gb
    cache_gb       = max(available_gb, 0)
    cache_pct      = (cache_gb / gpu_memory_gb) * 100

    return {
        "total_gpu_gb":    gpu_memory_gb,
        "model_memory_gb": model_memory_gb,
        "reserved_gb":     round(reserved_gb, 2),
        "kv_cache_gb":     round(cache_gb, 2),
        "kv_cache_pct":    round(cache_pct, 1),
    }