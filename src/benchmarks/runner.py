"""
Benchmark Runner - Main orchestrator for running benchmarks
"""

import time
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from src.models import model_loader
# , BaseModel
from src.workloads import generator
from src.metrics import MetricsCollector, LatencyMetrics, calculate_cost_metrics
from src.utils import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:
    """
    Main benchmark orchestrator.
    
    Coordinates:
    - Model loading
    - Workload generation
    - Benchmark execution
    - Results collection and storage
    """
    
    def __init__(self, output_dir: str = "results/baseline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_loader = model_loader()
        self.workload_generator = generator()
        self.results = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_benchmark(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        workload_name: str,
        num_samples: int = 10,
        batch_sizes: List[int] = [1, 4, 8]
    ):
        """
        Run benchmark for a single model and workload.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration dict
            workload_name: Workload type
            num_samples: Number of prompts to test
            batch_sizes: List of batch sizes to test
        """
        logger.info(f"="*60)
        logger.info(f"Benchmarking: {model_name} on {workload_name}")
        logger.info(f"="*60)
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        try:
            model = self.model_loader.load_model(model_config)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return
        
        # Generate workload
        logger.info(f"Generating {num_samples} prompts for {workload_name}")
        prompts = self.workload_generator.generate_workload(workload_name, num_samples)
        
        # Run for each batch size
        for batch_size in batch_sizes:
            logger.info(f"Testing batch_size={batch_size}")
            
            collector = MetricsCollector()
            collector.start_benchmark()
            
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Process in batches
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                
                try:
                    # Generate responses
                    start_time = time.time()
                    responses = model.generate_batch(batch, max_tokens=512)
                    end_time = time.time()
                    
                    # Record metrics for each response
                    for response in responses:
                        latency = LatencyMetrics(
                            time_to_first_token=response.latency_ms / 1000,
                            inter_token_latency=[],
                            end_to_end_latency=response.latency_ms / 1000,
                        )
                        collector.record_latency(latency)
                        
                        total_input_tokens += response.input_tokens
                        total_output_tokens += response.output_tokens
                        
                        # Store detailed results
                        self.results.append({
                            "run_id": self.run_id,
                            "model_name": model_name,
                            "backend": model_config.get("backend", "unknown"),
                            "workload": workload_name,
                            "batch_size": batch_size,
                            "latency_ms": response.latency_ms,
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "timestamp": datetime.now().isoformat(),
                        })
                    
                    logger.info(f"  Batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} completed")
                
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue
            
            collector.end_benchmark()
            
            # Calculate costs
            cost_metrics = calculate_cost_metrics(
                model_config=model_config,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                duration_seconds=collector.end_time - collector.start_time
            )
            
            # Get summary
            summary = collector.get_summary()
            logger.info(f"Batch size {batch_size} summary:")
            logger.info(f"  Avg latency: {summary['latency']['ttft_ms']['p50']:.2f}ms (p50)")
            logger.info(f"  p95 latency: {summary['latency']['ttft_ms']['p95']:.2f}ms")
            logger.info(f"  Throughput: {summary['throughput']['tokens_per_second']:.2f} tokens/sec")
            logger.info(f"  Total cost: ${cost_metrics.total_cost:.4f}")
        
        logger.info(f"Benchmark complete for {model_name}")
    
    def save_results(self, filename: str = None):
        """Save results to CSV"""
        if filename is None:
            filename = f"benchmark_results_{self.run_id}.csv"
        
        filepath = self.output_dir / filename
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics by model and workload"""
        if not self.results:
            logger.warning("No results to summarize")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        summary = df.groupby(["model_name", "workload", "batch_size"]).agg({
            "latency_ms": ["mean", "median", lambda x: x.quantile(0.95), lambda x: x.quantile(0.99)],
            "input_tokens": "sum",
            "output_tokens": "sum",
        }).round(2)
        
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary.columns = [
            col.replace("<lambda_0>", "p95").replace("<lambda_1>", "p99") 
            for col in summary.columns
        ]
        
        return summary
    
    def cleanup(self):
        """Cleanup loaded models"""
        logger.info("Cleaning up models...")
        self.model_loader.unload_all()