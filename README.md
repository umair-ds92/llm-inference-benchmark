# LLM Inference Benchmarking System

A comprehensive framework for benchmarking Large Language Model (LLM) inference performance across multiple models, hardware configurations, and optimization strategies. This project provides actionable insights for production deployment decisions by systematically evaluating the trade-offs between latency, throughput, cost, and quality.

## 🎯 Project Overview

Organizations deploying LLMs in production face critical decisions about model selection, hardware provisioning, and optimization strategies. This benchmarking system provides:

- **Multi-model comparison**: Benchmark 6+ LLMs (Llama-2-7B/13B, Mistral-7B, Phi-3, GPT-3.5, Claude-Haiku)
- **Realistic workloads**: Simulate production traffic patterns (Q&A, summarization, code generation)
- **Comprehensive metrics**: Measure TTFT, ITL, throughput, memory usage, and cost
- **Optimization analysis**: Test quantization, batching, and KV cache strategies
- **Decision framework**: Generate cost-performance Pareto frontiers and deployment recommendations

## 🚀 Key Features

- ✅ **Production-focused**: Real-world workload simulation and SLA validation
- ✅ **Cost-aware**: Detailed cost modeling for infrastructure and API usage
- ✅ **Optimization-ready**: Built-in support for quantization and performance tuning
- ✅ **Reproducible**: Automated setup and containerized deployment
- ✅ **Extensible**: Easy to add new models, metrics, and workload scenarios

## 📊 Sample Results

> *Llama-2-7B with vLLM + INT8 quantization achieves 85 tokens/sec throughput at 120ms p95 latency, 60% cheaper than API-based models for high-volume workloads*

> *Mistral-7B-Instruct provides the best cost-performance ratio for general-purpose enterprise use cases*

## 🏗️ Architecture

```
llm-inference-benchmark/
├── config/                  # Model and benchmark configurations
├── src/
│   ├── metrics/            # Metrics collection and storage
│   ├── models/             # Model deployment and management
│   ├── workloads/          # Traffic generation and scenarios
│   ├── benchmarks/         # Benchmark execution engine
│   ├── optimizations/      # Quantization, batching, KV cache
│   ├── analysis/           # Cost modeling and Pareto analysis
│   └── visualization/      # Dashboard and plotting
├── infrastructure/         # AWS/cloud deployment scripts
├── notebooks/              # Analysis and reporting
├── results/                # Benchmark data and reports
└── tests/                  # Unit and integration tests
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.1+ (for local testing)
- AWS account (for production benchmarks)
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-inference-benchmark.git
cd llm-inference-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use Poetry
poetry install
```

### Basic Usage

```bash
# Run a simple benchmark
python -m src.benchmarks.runner --config config/benchmark_config.yaml

# Launch the dashboard
python -m src.visualization.dashboard

# Generate a report
python -m src.analysis.generate_report --results results/baseline/
```

## 📦 Supported Models

### Open Source Models (via vLLM)
- **Llama-2-7B** - Strong general-purpose model
- **Llama-2-13B** - Larger variant for complex tasks
- **Mistral-7B-Instruct** - Efficient instruction-following
- **Phi-3-mini** - Compact high-performance model

### API-Based Models
- **GPT-3.5-Turbo** - OpenAI's production model
- **Claude-Haiku** - Anthropic's fast, affordable model

## 🎯 Workload Scenarios

1. **Question Answering** - Short prompts (50-200 tokens), factual responses
2. **Document Summarization** - Long prompts (500-2000 tokens), concise outputs
3. **Code Generation** - Technical prompts, structured code outputs
4. **Conversational** - Multi-turn dialogues with context

## 📈 Metrics Collected

- **Time to First Token (TTFT)**: Initial response latency
- **Inter-Token Latency (ITL)**: Streaming performance
- **Throughput**: Tokens per second
- **Memory Usage**: GPU/RAM utilization
- **Cost per 1M tokens**: Infrastructure and API costs
- **Quality Metrics**: Perplexity, task accuracy (optional)

## 🔧 Configuration

Edit `config/benchmark_config.yaml` to customize:

```yaml
models:
  - name: "llama-2-7b"
    backend: "vllm"
    quantization: "fp16"
    
workloads:
  - name: "qa"
    prompt_length: [50, 100, 200]
    batch_sizes: [1, 4, 8, 16]
    
infrastructure:
  instance_type: "g5.xlarge"
  region: "us-east-1"
```

## 🚀 AWS Deployment

```bash
# Deploy infrastructure (Terraform)
cd infrastructure
terraform init
terraform apply

# Or use the helper script
./infrastructure/deploy_aws.sh --instance-type g5.xlarge
```

## 📊 Analysis & Visualization

The project includes Jupyter notebooks for detailed analysis:

- `01_baseline_analysis.ipynb` - Initial model comparison
- `02_optimization_analysis.ipynb` - Quantization and batching trade-offs
- `03_final_analysis.ipynb` - Complete results and recommendations

Interactive dashboard for real-time exploration:

```bash
python -m src.visualization.dashboard
# Opens at http://localhost:8501
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_metrics.py
```

## 📚 Documentation

- [Setup Guide](docs/setup.md) - Detailed installation instructions
- [Architecture](docs/architecture.md) - System design and components
- [API Reference](docs/api_reference.md) - Code documentation
- [Optimization Guide](docs/optimization_guide.md) - Performance tuning tips
- [Deployment Guide](docs/deployment_guide.md) - Production recommendations

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- Uses [HuggingFace Transformers](https://huggingface.co/transformers) for model loading
- Inspired by production deployments at scale
