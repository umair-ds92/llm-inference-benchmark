"""Run simple benchmarks and save results"""
import time
from typing import List
import pandas as pd
from src.models.model_loader import GPT35Model, ModelResponse
from src.workloads.scenarios import generate_qa_prompts

class SimpleBenchmarkRunner:
    def __init__(self, model):
        self.model = model
        self.results = []
    
    def run_benchmark(self, num_samples: int = 10):
        """Run benchmark on Q&A workload"""
        print(f"Generating {num_samples} prompts...")
        prompts = generate_qa_prompts(num_samples)
        
        print(f"Running benchmarks...")
        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{num_samples}] Processing...")
            response = self.model.generate(prompt)
            
            self.results.append({
                "prompt": prompt,
                "response": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
            })
        
        print("Benchmark complete!")
    
    def get_summary(self):
        """Get performance summary"""
        df = pd.DataFrame(self.results)
        
        summary = {
            "total_requests": len(df),
            "avg_latency_ms": df["latency_ms"].mean(),
            "p50_latency_ms": df["latency_ms"].quantile(0.5),
            "p95_latency_ms": df["latency_ms"].quantile(0.95),
            "p99_latency_ms": df["latency_ms"].quantile(0.99),
            "total_input_tokens": df["input_tokens"].sum(),
            "total_output_tokens": df["output_tokens"].sum(),
        }
        
        return summary
    
    def save_results(self, filepath: str = "results/quick_benchmark.csv"):
        """Save results to CSV"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")