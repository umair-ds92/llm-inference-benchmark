# Setup Guide - LLM Inference Benchmarking System

This guide provides detailed instructions for setting up the benchmarking system locally and on AWS.

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [AWS Cloud Setup](#aws-cloud-setup)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites

- **Python**: 3.11 or higher
- **CUDA**: 12.1+ (for GPU support)
- **NVIDIA GPU**: With at least 16GB VRAM (24GB recommended)
- **Git**: For version control
- **Operating System**: Linux (Ubuntu 20.04+ recommended) or WSL2 on Windows

### Step 1: Clone the Repository

```bash
git clone https://github.com/umair-ds92/llm-inference-benchmark.git
cd llm-inference-benchmark
```

### Step 2: Create Virtual Environment

Using `venv`:

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using `conda`:

```bash
conda create -n llm-bench python=3.10
conda activate llm-bench
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Or use Poetry
poetry install
```

### Step 4: Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
```

### Step 5: Set Up Environment Variables

Create a `.env` file:

```bash
cat > .env << EOF
# API Keys (if using API-based models)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# HuggingFace Token (for gated models)
HF_TOKEN=your_huggingface_token_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/benchmark.log

# Database
DATABASE_URL=sqlite:///results/benchmark.db
EOF
```

### Step 6: Create Required Directories

```bash
mkdir -p logs results/baseline results/optimized data
```

---

## AWS Cloud Setup

### Prerequisites

- **AWS Account**: With appropriate permissions
- **AWS CLI**: Installed and configured
- **SSH Key**: For instance access

### Step 1: Configure AWS CLI

```bash
aws configure
```

Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., `us-east-1`)
- Default output format (`json`)

### Step 2: Deploy Infrastructure

```bash
# Make script executable
chmod +x infrastructure/deploy_aws.sh

# Deploy with default settings (g5.xlarge)
./infrastructure/deploy_aws.sh

# Or customize instance type
INSTANCE_TYPE=g5.2xlarge ./infrastructure/deploy_aws.sh
```

The script will:
1. Create a security group
2. Create/verify SSH key pair
3. Find the latest Deep Learning AMI
4. Launch EC2 instance
5. Configure storage and networking

### Step 3: Connect to Instance

```bash
# SSH into the instance (IP will be shown in output)
ssh -i ~/.ssh/llm-benchmark-key.pem ubuntu@<INSTANCE_IP>
```

### Step 4: Set Up on Instance

Once connected:

```bash
# Clone repository
git clone https://github.com/umair-ds92/llm-inference-benchmark.git
cd llm-inference-benchmark

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Configuration

### Model Configuration

Edit `config/models.yaml` to:

1. **Add/remove models**:
```yaml
models:
  - name: "your-custom-model"
    backend: "vllm"
    model_id: "organization/model-name"
    quantization: "fp16"
```

2. **Adjust costs**:
```yaml
cost_per_hour: 1.50  # Your instance cost
```

3. **Change quantization**:
```yaml
quantization: "int8"  # fp16, int8, int4
```

### Benchmark Configuration

Edit `config/benchmark_config.yaml` to:

1. **Select models**:
```yaml
model_selection:
  preset: "quick_test"  # or custom list
```

2. **Configure workloads**:
```yaml
workloads:
  - name: "qa"
    num_samples: 100
    prompt_length_range: [50, 200]
```

3. **Set batch sizes**:
```yaml
batch_sizes: [1, 4, 8, 16]
```

---

## Verification

### Test 1: Import Modules

```bash
python -c "from src.metrics import MetricsCollector; print('Imports successful')"
```

### Test 2: Run Tests

```bash
pytest tests/test_metrics.py -v
```

Expected output:
```
tests/test_metrics.py::TestLatencyMetrics::test_latency_metrics_creation PASSED
tests/test_metrics.py::TestLatencyMetrics::test_latency_metrics_to_dict PASSED
...
=================== X passed in Y.YYs ===================
```

### Test 3: Check GPU Access

```bash
python -c "
from src.metrics import MetricsCollector
collector = MetricsCollector()
metrics = collector.collect_resource_metrics()
print(f'GPU Memory: {metrics.gpu_memory_used_mb:.0f}MB / {metrics.gpu_memory_total_mb:.0f}MB')
"
```

### Test 4: Verify Configuration Loading

```bash
python -c "
import yaml
with open('config/models.yaml') as f:
    config = yaml.safe_load(f)
    print(f'Loaded {len(config[\"models\"])} models')
"
```

---

## Troubleshooting

### CUDA/GPU Issues

**Problem**: `CUDA not available`

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues

**Problem**: `CUDA out of memory`

**Solution**:
```yaml
# Reduce gpu_memory_utilization in config/models.yaml
gpu_memory_utilization: 0.7  # Instead of 0.9

# Or use smaller batch sizes in config/benchmark_config.yaml
batch_sizes: [1, 2, 4]  # Instead of [1, 4, 8, 16]
```

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### AWS Instance Issues

**Problem**: Cannot connect to instance

**Solution**:
```bash
# Check instance status
aws ec2 describe-instances --instance-ids <INSTANCE_ID>

# Verify security group allows SSH
aws ec2 describe-security-groups --group-ids <SG_ID>

# Check SSH key permissions
chmod 400 ~/.ssh/llm-benchmark-key.pem
```

### HuggingFace Authentication

**Problem**: Cannot download gated models

**Solution**:
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token in .env
echo "HF_TOKEN=your_token_here" >> .env
```

---

## Next Steps

After successful setup:

1. **Review configurations**: Check `config/models.yaml` and `config/benchmark_config.yaml`
2. **Test small benchmark**: Run on a single model first
3. **Monitor resources**: Use `nvidia-smi` to watch GPU usage
4. **Scale up**: Once verified, run full benchmark suite

For running your first benchmark, see the main [README.md](../README.md).