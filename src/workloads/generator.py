"""
Workload Generator for LLM Benchmarking
"""

from typing import List, Dict, Any
import yaml

from .scenarios import (
    generate_qa_prompts,
    generate_summarization_prompts,
    generate_code_generation_prompts,
    generate_conversation_prompts,
)
from src.utils import get_logger

logger = get_logger(__name__)


class WorkloadGenerator:
    """Generate workloads based on configuration"""
    
    def __init__(self, config_path: str = "config/benchmark_config.yaml"):
        self.config = self._load_config(config_path)
        self.workloads = self.config.get("workloads", [])
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_workload(self, workload_name: str, num_samples: int = None) -> List[str]:
        """
        Generate prompts for a specific workload.
        
        Args:
            workload_name: Name of workload (qa, summarization, etc.)
            num_samples: Number of samples to generate
            
        Returns:
            List of prompts
        """
        # Find workload config
        workload_config = None
        for w in self.workloads:
            if w["name"] == workload_name:
                workload_config = w
                break
        
        if not workload_config:
            raise ValueError(f"Workload '{workload_name}' not found")
        
        # Get number of samples
        if num_samples is None:
            num_samples = workload_config.get("num_samples", 100)
        
        logger.info(f"Generating {num_samples} prompts for: {workload_name}")
        
        # Generate based on type
        if workload_name == "qa":
            prompts = generate_qa_prompts(num_samples)
        elif workload_name == "summarization":
            prompts = generate_summarization_prompts(num_samples)
        elif workload_name == "code_generation":
            prompts = generate_code_generation_prompts(num_samples)
        elif workload_name == "conversation":
            prompts = generate_conversation_prompts(num_samples)
        else:
            raise ValueError(f"Unknown workload: {workload_name}")
        
        return prompts
    
    def generate_all_workloads(self) -> Dict[str, List[str]]:
        """Generate all configured workloads"""
        all_workloads = {}
        for workload_config in self.workloads:
            name = workload_config["name"]
            prompts = self.generate_workload(name)
            all_workloads[name] = prompts
        return all_workloads