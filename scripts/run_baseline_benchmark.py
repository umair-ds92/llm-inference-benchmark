#!/usr/bin/env python3
"""
Run baseline benchmarks across models and workloads
"""

import sys
import argparse
import yaml

sys.path.insert(0, ".")

from src.benchmarks import runner
from src.utils import setup_logger

def load_models_config(config_path: str = "config/models.yaml"):
    """Load model configurations"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("models", [])

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference benchmarks")
    parser.add_argument("--models", nargs="+", help="Models to benchmark")
    parser.add_argument("--workloads", nargs="+", default=["qa"], help="Workloads to test")
    parser.add_argument("--num-samples", type=int, default=20, help="Samples per workload")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8], help="Batch sizes")
    parser.add_argument("--quick-test", action="store_true", help="Quick test (5 samples)")
    parser.add_argument("--output-dir", default="results/baseline", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level="INFO", console=True)
    
    # Quick test mode
    if args.quick_test:
        args.num_samples = 5
        args.batch_sizes = [1]
        print("🚀 QUICK TEST MODE: 5 samples, batch_size=1")
    
    # Load model configs
    all_models = load_models_config()
    
    # Filter models
    if args.models:
        models_to_test = [m for m in all_models if m["name"] in args.models]
    else:
        # Default: API models only (no GPU needed)
        models_to_test = [m for m in all_models if "api" in m.get("backend", "")]
    
    if not models_to_test:
        print("❌ No models selected.")
        print(f"Available: {[m['name'] for m in all_models]}")
        return
    
    print("="*80)
    print("LLM INFERENCE BASELINE BENCHMARK")
    print("="*80)
    print(f"Models: {[m['name'] for m in models_to_test]}")
    print(f"Workloads: {args.workloads}")
    print(f"Samples: {args.num_samples}")
    print(f"Batch sizes: {args.batch_sizes}")
    print("="*80)
    print()
    
    # Create runner
    runner = runner(output_dir=args.output_dir)
    
    # Run benchmarks
    for model_config in models_to_test:
        for workload in args.workloads:
            try:
                runner.run_benchmark(
                    model_name=model_config["name"],
                    model_config=model_config,
                    workload_name=workload,
                    num_samples=args.num_samples,
                    batch_sizes=args.batch_sizes,
                )
            except Exception as e:
                print(f"❌ Error: {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    filepath = runner.save_results()
    print(f"✅ Saved: {filepath}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary = runner.get_summary_stats()
    print(summary)
    
    # Cleanup
    runner.cleanup()
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()