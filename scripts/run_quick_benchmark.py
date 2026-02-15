#!/usr/bin/env python3
"""Quick benchmark script"""
import sys
sys.path.insert(0, ".")

from src.models.model_loader import GPT35Model
from src.benchmarks.runner import SimpleBenchmarkRunner

def main():
    print("="*60)
    print("QUICK BENCHMARK - GPT-3.5")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    model = GPT35Model()
    
    # Create runner
    runner = SimpleBenchmarkRunner(model)
    
    # Run benchmark
    print("\n2. Running benchmark...")
    runner.run_benchmark(num_samples=10)
    
    # Get summary
    print("\n3. Results:")
    print("-"*60)
    summary = runner.get_summary()
    for key, value in summary.items():
        print(f"{key:20s}: {value:.2f}")
    
    # Save results
    print("\n4. Saving results...")
    runner.save_results()
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()