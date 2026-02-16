```markdown
# Running Benchmarks Guide

## Quick Start

### Run with API Models (No GPU needed)

```bash
# Quick test with GPT-3.5 (requires OPENAI_API_KEY)
python scripts/run_baseline_benchmark.py --quick-test

# Full test with multiple workloads
python scripts/run_baseline_benchmark.py \
    --workloads qa summarization code_generation \
    --num-samples 20
```

### Run with Self-Hosted Models (Requires GPU)

```bash
# Test Llama-2-7B
python scripts/run_baseline_benchmark.py \
    --models llama-2-7b \
    --workloads qa \
    --batch-sizes 1 4 8 16

# Test multiple models
python scripts/run_baseline_benchmark.py \
    --models llama-2-7b mistral-7b-instruct \
    --workloads qa summarization \
    --num-samples 50 \
    --batch-sizes 1 4 8
```

## CLI Options

```
--models MODEL [MODEL ...]
    Models to benchmark (from config/models.yaml)
    Examples: llama-2-7b, gpt-3.5-turbo, claude-haiku

--workloads WORKLOAD [WORKLOAD ...]
    Workloads to test
    Options: qa, summarization, code_generation, conversation
    Default: qa

--num-samples N
    Number of prompts per workload
    Default: 20

--batch-sizes SIZE [SIZE ...]
    Batch sizes to test
    Default: 1 4 8

--quick-test
    Run quick test (5 samples, batch_size=1)

--output-dir DIR
    Output directory for results
    Default: results/baseline
```

## Output Files

Results are saved to `results/baseline/`:

- `benchmark_results_YYYYMMDD_HHMMSS.csv` - Detailed results
- Each row contains: model, workload, batch_size, latency, tokens, etc.

## Analyzing Results

### Using Jupyter Notebook

```bash
jupyter notebook notebooks/01_baseline_analysis.ipynb
```

### Using Pandas

```python
import pandas as pd

df = pd.read_csv('results/baseline/benchmark_results_20260214_120000.csv')

# Summary stats
summary = df.groupby('model_name').agg({
    'latency_ms': ['mean', 'median', lambda x: x.quantile(0.95)],
    'input_tokens': 'sum',
    'output_tokens': 'sum'
})
print(summary)

# Best model by latency
best_model = df.groupby('model_name')['latency_ms'].mean().idxmin()
print(f"Fastest model: {best_model}")
```

## Example Workflows

### Compare API vs Self-Hosted

```bash
# Benchmark API model
python scripts/run_baseline_benchmark.py \
    --models gpt-3.5-turbo \
    --workloads qa \
    --num-samples 100

# Benchmark self-hosted
python scripts/run_baseline_benchmark.py \
    --models llama-2-7b \
    --workloads qa \
    --num-samples 100

# Compare in notebook
jupyter notebook notebooks/01_baseline_analysis.ipynb
```

### Find Optimal Batch Size

```bash
python scripts/run_baseline_benchmark.py \
    --models llama-2-7b \
    --workloads qa \
    --num-samples 50 \
    --batch-sizes 1 2 4 8 16 32
```

Then analyze to find the batch size with best latency/throughput trade-off.

### Test All Workloads

```bash
python scripts/run_baseline_benchmark.py \
    --models mistral-7b-instruct \
    --workloads qa summarization code_generation conversation \
    --num-samples 30
```

## Troubleshooting

### GPU Out of Memory

```bash
# Use smaller batch sizes
--batch-sizes 1 2 4

# Or use smaller model
--models phi-3-mini
```

### API Rate Limits

```bash
# Use fewer samples
--num-samples 10

# Add delays in code (modify runner.py)
import time
time.sleep(1)  # Between requests
```

### Missing API Keys

```bash
# Check .env file
cat .env | grep API_KEY

# Set manually
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

## Next Steps

After running baseline benchmarks:

1. Review results in Jupyter notebook
2. Identify best models for your use case
3. Move to next step for optimization experiments
4. Test quantization, KV cache, etc.
```