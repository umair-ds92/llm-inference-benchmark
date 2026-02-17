#!/usr/bin/env python3
"""
Run Optimization Experiments for LLM Inference

Tests quantization (INT8/FP16), KV cache strategies,
and batch sizes. Compares quality and performance.

Usage:
    # Quick test (API models, small samples)
    python scripts/run_optimization_experiments.py --quick-test

    # Full experiments on a specific model
    python scripts/run_optimization_experiments.py \
        --model llama-2-7b \
        --quantizations fp16 int8 \
        --batch-sizes 1 4 8 16 \
        --num-samples 50

    # API models only (no GPU required)
    python scripts/run_optimization_experiments.py \
        --api-only --num-samples 20
"""

import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List

sys.path.insert(0, ".")

import yaml
from src.utils import setup_logger, get_logger
from src.models import ModelLoader
from src.workloads import WorkloadGenerator
from src.optimizations import (
    get_quantization_config,
    get_all_configs,
    BatchOptimizer,
)
from src.evaluation import QualityEvaluator

logger = get_logger(__name__)

OUTPUT_DIR = Path("results/optimized")


def load_model_config(model_name: str, config_path: str = "config/models.yaml") -> dict:
    """Load a single model's config by name."""
    with open(config_path) as f:
        all_models = yaml.safe_load(f).get("models", [])
    for m in all_models:
        if m["name"] == model_name:
            return m
    raise ValueError(f"Model '{model_name}' not found in {config_path}")


def run_quantization_experiment(
    model_name: str,
    model_config: dict,
    prompts: List[str],
    quantizations: List[str],
    evaluator: QualityEvaluator,
    loader: ModelLoader,
) -> List[dict]:
    """
    Run quantization experiments for a model.

    Loads the model once per quantization level, runs benchmarks,
    evaluates quality, and records everything.
    """
    results = []
    baseline_quality = None

    for quant_name in quantizations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_name} + {quant_name.upper()}")
        logger.info(f"{'='*60}")

        # Get quantization config
        quant_config = get_quantization_config(quant_name)

        # Build model config with this quantization
        exp_config = {**model_config, "quantization": quant_config.dtype}
        run_id = f"{model_name}_{quant_name}_{datetime.now():%Y%m%d_%H%M%S}"

        # ── Performance benchmarking ───────────────────────────────────────
        import time, numpy as np
        latencies = []
        output_tokens_list = []

        try:
            # For vLLM: re-load with new dtype
            if exp_config.get("backend") == "vllm":
                loader.unload_model(model_name)

            model = loader.load_model({**exp_config, "name": run_id})
        except Exception as e:
            logger.error(f"Could not load {model_name} with {quant_name}: {e}")
            results.append({
                "run_id": run_id,
                "model_name": model_name,
                "quantization": quant_name,
                "error": str(e),
            })
            continue

        for i in range(0, len(prompts), 8):   # batch_size=8
            batch = prompts[i:i+8]
            t0 = time.time()
            try:
                responses = model.generate_batch(batch, max_tokens=256)
                t1 = time.time()
                batch_ms = (t1 - t0) * 1000
                for r in responses:
                    latencies.append(batch_ms / len(batch))
                    output_tokens_list.append(r.output_tokens)
            except Exception as e:
                logger.error(f"Batch error: {e}")

        latencies_arr = np.array(latencies) if latencies else np.array([0])
        total_tokens  = sum(output_tokens_list)

        perf = {
            "mean_latency_ms":  round(float(np.mean(latencies_arr)),  2),
            "p50_latency_ms":   round(float(np.percentile(latencies_arr, 50)), 2),
            "p95_latency_ms":   round(float(np.percentile(latencies_arr, 95)), 2),
            "p99_latency_ms":   round(float(np.percentile(latencies_arr, 99)), 2),
            "throughput_tok_s": round(total_tokens / max(sum(latencies)/1000, 0.001), 2),
        }

        # ── Quality evaluation ─────────────────────────────────────────────
        qa_prompts = [p for p in prompts if "?" in p][:20]
        quality = evaluator.evaluate_accuracy(model, qa_prompts)
        quality.model_name   = model_name
        quality.quantization = quant_name

        if quant_name == "fp16":
            baseline_quality = quality

        if baseline_quality and quant_name != "fp16":
            quality = evaluator.compare_quality(baseline_quality, quality)

        # ── Memory estimate ────────────────────────────────────────────────
        params_b = float(model_config.get("parameters", "7B").rstrip("B"))
        from src.optimizations.quantization import estimate_memory_usage
        mem = estimate_memory_usage(params_b, quant_config)

        # ── Combine all metrics ────────────────────────────────────────────
        record = {
            "run_id":          run_id,
            "model_name":      model_name,
            "quantization":    quant_name,
            "dtype":           quant_config.dtype,
            **perf,
            "task_accuracy":   quality.task_accuracy,
            "accuracy_delta":  quality.accuracy_delta,
            "est_memory_gb":   mem["total_memory_gb"],
            "memory_reduction":quant_config.memory_reduction,
            "timestamp":       datetime.now().isoformat(),
        }
        results.append(record)

        logger.info(
            f"{quant_name.upper()}: p95={perf['p95_latency_ms']:.0f}ms, "
            f"accuracy={quality.task_accuracy:.2%}, "
            f"memory={mem['total_memory_gb']:.1f}GB"
        )

        loader.unload_model(run_id)

    return results


def run_batch_size_experiment(
    model_name: str,
    model_config: dict,
    prompts: List[str],
    batch_sizes: List[int],
    loader: ModelLoader,
    sla_target_ms: float = 500.0,
) -> List[dict]:
    """Run batch size optimisation experiments."""
    logger.info(f"\nBatch size experiment: {model_name}, sizes={batch_sizes}")

    try:
        model = loader.load_model(model_config)
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        return []

    optimizer = BatchOptimizer(sla_target_ms=sla_target_ms)
    results   = optimizer.run_experiments(model, prompts, batch_sizes)
    optimal   = optimizer.find_optimal(results)

    if optimal:
        logger.info(
            f"✅ Optimal batch_size={optimal.batch_size}: "
            f"p95={optimal.p95_latency_ms:.0f}ms, "
            f"{optimal.throughput_tokens_per_sec:.1f} tok/s"
        )

    return [r.to_dict() for r in results]


def main():

    parser = argparse.ArgumentParser(description="Run optimization experiments")
    parser.add_argument("--model",           default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--quantizations",   nargs="+", default=["fp16", "int8"],
                        help="Quantization levels to test")
    parser.add_argument("--batch-sizes",     nargs="+", type=int, default=[1, 4, 8, 16],
                        help="Batch sizes to test")
    parser.add_argument("--num-samples",     type=int, default=20,
                        help="Prompts per experiment")
    parser.add_argument("--sla-target-ms",   type=float, default=500.0,
                        help="p95 latency SLA target in ms")
    parser.add_argument("--api-only",        action="store_true",
                        help="Test API models only (no GPU required)")
    parser.add_argument("--quick-test",      action="store_true",
                        help="Quick test: 5 samples, batch_sizes=[1,4]")
    args = parser.parse_args()

    setup_logger(log_level="INFO", console=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick_test:
        args.num_samples = 5
        args.batch_sizes = [1, 4]
        logger.info("🚀 QUICK TEST MODE")

    # ── Load config & generate prompts ────────────────────────────────────
    try:
        model_config = load_model_config(args.model)
    except ValueError as e:
        print(f"❌ {e}")
        return

    gen     = WorkloadGenerator()
    prompts = gen.generate_workload("qa", num_samples=args.num_samples)

    loader    = ModelLoader()
    evaluator = QualityEvaluator()

    print("=" * 70)
    print("LLM OPTIMIZATION EXPERIMENTS")
    print("=" * 70)
    print(f"Model:          {args.model}")
    print(f"Quantizations:  {args.quantizations}")
    print(f"Batch sizes:    {args.batch_sizes}")
    print(f"Samples:        {args.num_samples}")
    print(f"SLA target:     {args.sla_target_ms}ms p95")
    print("=" * 70)

    all_quant_results  = []
    all_batch_results  = []

    # ── Quantization experiments ──────────────────────────────────────────
    if model_config.get("backend") == "vllm":
        print("\n📊 QUANTIZATION EXPERIMENTS")
        quant_results = run_quantization_experiment(
            model_name=args.model,
            model_config=model_config,
            prompts=prompts,
            quantizations=args.quantizations,
            evaluator=evaluator,
            loader=loader,
        )
        all_quant_results.extend(quant_results)
    else:
        logger.info(f"Skipping quantization (API model: {args.model})")

    # ── Batch size experiments ────────────────────────────────────────────
    print("\n⚡ BATCH SIZE EXPERIMENTS")
    batch_results = run_batch_size_experiment(
        model_name=args.model,
        model_config=model_config,
        prompts=prompts,
        batch_sizes=args.batch_sizes,
        loader=loader,
        sla_target_ms=args.sla_target_ms,
    )
    all_batch_results.extend(batch_results)

    # ── Save results ──────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if all_quant_results:
        quant_path = OUTPUT_DIR / f"quantization_results_{ts}.csv"
        pd.DataFrame(all_quant_results).to_csv(quant_path, index=False)
        print(f"\n✅ Quantization results: {quant_path}")

    if all_batch_results:
        batch_path = OUTPUT_DIR / f"batch_results_{ts}.csv"
        pd.DataFrame(all_batch_results).to_csv(batch_path, index=False)
        print(f"✅ Batch results:        {batch_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)

    if all_quant_results:
        df_q = pd.DataFrame(all_quant_results)
        print("\nQuantization Results:")
        print(df_q[["quantization", "p95_latency_ms",
                     "task_accuracy", "est_memory_gb"]].to_string(index=False))

    if all_batch_results:
        df_b = pd.DataFrame(all_batch_results)
        print("\nBatch Size Results:")
        print(df_b[["batch_size", "p95_latency_ms",
                     "throughput_tokens_per_sec", "meets_sla"]].to_string(index=False))

    print("\n✅ EXPERIMENTS COMPLETE!")
    loader.unload_all()


if __name__ == "__main__":
    main()