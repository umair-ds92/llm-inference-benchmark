# Production Deployment Guide

## Overview

This guide provides best practices for deploying LLM inference in production
based on benchmark results and real-world experience.

## Model Selection Framework

### Step 1: Define Requirements

Before choosing a model, clearly define:

| Requirement | Question | Example |
|-------------|----------|---------|
| **Latency SLA** | What's your p95 latency target? | <500ms for interactive apps |
| **Volume** | How many tokens/month? | 100M tokens/month |
| **Budget** | What's your monthly budget? | $5,000/month |
| **Quality** | Minimum accuracy needed? | >85% on your tasks |
| **Infrastructure** | Self-host or API? | Prefer self-hosted |

### Step 2: Choose Deployment Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                 DECISION TREE                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Volume < 10M tokens/month?                                │
│  └─ YES → Use API (GPT-3.5 or Claude-Haiku)               │
│  └─ NO  → Continue...                                      │
│                                                             │
│  Need < 200ms p95 latency?                                 │
│  └─ YES → Small model + GPU (Phi-3-mini or Mistral-7B)    │
│  └─ NO  → Continue...                                      │
│                                                             │
│  Budget constrained?                                        │
│  └─ YES → Llama-2-7B + INT8 quantization                  │
│  └─ NO  → Llama-2-13B or Mistral-7B for best quality      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Optimization Strategies

### 1. Quantization

**Recommended**: INT8 for production self-hosted models

```
┌──────────────────────────────────────────────────────┐
│ Quantization Trade-offs                              │
├──────────────────────────────────────────────────────┤
│ FP16:  Baseline (14GB for 7B model)                 │
│ INT8:  45% memory reduction, <2% quality loss       │
│ INT4:  70% memory reduction, ~5% quality loss       │
└──────────────────────────────────────────────────────┘
```

**When to use**:
- INT8: Default for production (best balance)
- FP16: When quality is critical, have GPU headroom
- INT4: Experimental only, significant quality impact

### 2. Batch Size

**Recommended**: 8-16 for most workloads

```python
# Finding optimal batch size
from src.optimizations import BatchOptimizer

optimizer = BatchOptimizer(sla_target_ms=500)
results = optimizer.run_experiments(
    model, prompts,
    batch_sizes=[1, 4, 8, 16, 32]
)
optimal = optimizer.find_optimal()
```

**Guidelines**:
- Real-time chat: batch_size=1-4
- Interactive apps: batch_size=8-16
- Batch processing: batch_size=32+

### 3. KV Cache Configuration

**Recommended**: `optimized` for production

```yaml
kv_cache_config:
  strategy: optimized
  gpu_memory_utilization: 0.95
  max_num_seqs: 512
```

## Infrastructure Sizing

### GPU Selection

| Workload | GPU | Model Fit | Cost/Hour |
|----------|-----|-----------|-----------|
| Light (<10 req/s) | g5.xlarge (A10G 24GB) | Llama-2-7B INT8 | $1.01 |
| Medium (10-50 req/s) | g5.2xlarge (A10G 24GB) | Llama-2-13B INT8 | $1.52 |
| Heavy (50+ req/s) | p4d.24xlarge (8x A100) | Multiple models | $32.77 |

### Scaling Strategy

**Horizontal Scaling** (Recommended):
```
Load Balancer
    ├─ Instance 1: Llama-2-7B (g5.xlarge)
    ├─ Instance 2: Llama-2-7B (g5.xlarge)
    └─ Instance 3: Llama-2-7B (g5.xlarge)
```

**Benefits**:
- Better fault tolerance
- Easier to scale
- Lower cost per instance

**Vertical Scaling**:
```
Single Instance: Llama-2-13B (p4d.24xlarge)
```

**Use when**:
- Need larger model
- Low latency critical
- Request volume predictable

## Cost Optimization

### Break-Even Analysis

**API vs Self-Hosted**:

```
GPT-3.5 Turbo API:     $0.0005/1K input + $0.0015/1K output
Llama-2-7B INT8:       $0.0042/1K tokens (g5.xlarge, 100% util)

Break-even point: ~30M tokens/month
```

**Recommendation**:
- < 30M tokens/month: Use API
- 30-100M tokens/month: Consider self-hosted
- > 100M tokens/month: Definitely self-host

### Cost Reduction Strategies

1. **Use INT8 quantization**: -45% memory, enables smaller GPUs
2. **Optimize batch size**: 2-3x throughput improvement
3. **Spot instances**: -70% cost (with proper handling)
4. **Reserved instances**: -40% cost (1-year commit)
5. **Auto-scaling**: Only run GPUs when needed

## Monitoring & Alerting

### Key Metrics to Track

```python
# Essential monitoring
metrics = {
    "latency_p95_ms": 500,     # SLA breach alert
    "latency_p99_ms": 1000,    # Critical alert
    "throughput_req_s": 10,    # Capacity planning
    "gpu_utilization": 0.8,    # Cost optimization
    "error_rate": 0.01,        # Quality alert
}
```

### Prometheus + Grafana Setup

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'llm_inference'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

## Production Checklist

Before going live:

- [ ] Load test at 2x expected traffic
- [ ] Set up monitoring & alerts
- [ ] Configure auto-scaling
- [ ] Test failover scenarios
- [ ] Document incident response
- [ ] Set up cost tracking
- [ ] Implement rate limiting
- [ ] Test model rollback procedure
- [ ] Configure log aggregation
- [ ] Set up on-call rotation

## Common Pitfalls

### ❌ Don't:
- Run FP16 on 24GB GPU (wastes money)
- Use batch_size=1 for batch workloads (low throughput)
- Skip quality evaluation (hidden degradation)
- Over-provision "just in case" (wastes budget)
- Ignore p99 latency (users notice outliers)

### ✅ Do:
- Start with INT8 quantization
- Test multiple batch sizes
- Monitor quality metrics
- Right-size infrastructure
- Plan for p99, not average

## Case Studies

### Case Study 1: E-Commerce Chatbot

**Requirements**:
- Latency: <300ms p95
- Volume: 50M tokens/month
- Budget: $3,000/month

**Solution**:
- Model: Mistral-7B-Instruct + INT8
- Infrastructure: 2x g5.xlarge (auto-scaling)
- Batch size: 8
- Cost: $2,400/month ($600 under budget)
- Latency: 250ms p95 ✅

### Case Study 2: Document Summarization

**Requirements**:
- Latency: <2s acceptable
- Volume: 200M tokens/month
- Budget: $8,000/month

**Solution**:
- Model: Llama-2-13B + INT8
- Infrastructure: 3x g5.2xlarge
- Batch size: 32
- Cost: $6,800/month
- Latency: 1.2s p95 ✅

---