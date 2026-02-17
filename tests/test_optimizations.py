"""Tests for optimization modules"""

import pytest
from src.optimizations.quantization import (
    QuantizationConfig,
    get_quantization_config,
    get_all_configs,
    estimate_memory_usage,
    QUANTIZATION_CONFIGS,
)
from src.optimizations.kv_cache import (
    KVCacheConfig,
    get_kvcache_config,
    estimate_kvcache_size,
    KVCACHE_CONFIGS,
)
from src.optimizations.batching import BatchConfig, BatchOptimizer, BatchExperimentResult
from src.evaluation.quality_metrics import QualityResult, QualityEvaluator


# ─── Quantization tests ────────────────────────────────────────────────────────

class TestQuantizationConfig:
    def test_get_fp16_config(self):
        cfg = get_quantization_config("fp16")
        assert cfg.dtype == "float16"
        assert cfg.memory_reduction == 1.0
        assert cfg.expected_quality_loss == 0.0

    def test_get_int8_config(self):
        cfg = get_quantization_config("int8")
        assert cfg.dtype == "int8"
        assert cfg.memory_reduction < 1.0       # memory savings
        assert cfg.expected_quality_loss < 0.05  # <5% quality loss

    def test_get_int4_config(self):
        cfg = get_quantization_config("int4")
        assert cfg.dtype == "int4"
        assert cfg.memory_reduction < 0.55

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError):
            get_quantization_config("bf16_unknown")

    def test_all_configs_returned(self):
        configs = get_all_configs()
        assert len(configs) == 3  # fp16, int8, int4

    def test_to_dict(self):
        cfg = get_quantization_config("int8")
        d   = cfg.to_dict()
        assert "name"           in d
        assert "dtype"          in d
        assert "memory_reduction" in d

    def test_estimate_memory_7b(self):
        cfg = get_quantization_config("fp16")
        mem = estimate_memory_usage(7.0, cfg)
        # 7B params * 2 bytes * 1.2 overhead ÷ 1024^3 ≈ 15.6 GB
        assert 10 < mem["total_memory_gb"] < 20

    def test_int8_uses_less_memory_than_fp16(self):
        fp16_mem = estimate_memory_usage(7.0, get_quantization_config("fp16"))
        int8_mem = estimate_memory_usage(7.0, get_quantization_config("int8"))
        assert int8_mem["total_memory_gb"] < fp16_mem["total_memory_gb"]

    def test_int4_uses_less_memory_than_int8(self):
        int8_mem = estimate_memory_usage(7.0, get_quantization_config("int8"))
        int4_mem = estimate_memory_usage(7.0, get_quantization_config("int4"))
        assert int4_mem["total_memory_gb"] < int8_mem["total_memory_gb"]


# ─── KV cache tests ────────────────────────────────────────────────────────────

class TestKVCacheConfig:
    def test_get_default_config(self):
        cfg = get_kvcache_config("default")
        assert cfg.gpu_memory_utilization == 0.90
        assert cfg.max_num_seqs == 256

    def test_get_optimized_config(self):
        cfg = get_kvcache_config("optimized")
        assert cfg.gpu_memory_utilization > 0.90

    def test_get_conservative_config(self):
        cfg = get_kvcache_config("conservative")
        assert cfg.gpu_memory_utilization < 0.90

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError):
            get_kvcache_config("turbo_mode")

    def test_to_vllm_kwargs(self):
        cfg    = get_kvcache_config("default")
        kwargs = cfg.to_vllm_kwargs()
        assert "gpu_memory_utilization"  in kwargs
        assert "max_num_seqs"            in kwargs
        assert "max_num_batched_tokens"  in kwargs

    def test_estimate_kvcache_size(self):
        info = estimate_kvcache_size(
            gpu_memory_gb=24.0,
            model_memory_gb=14.0,
            gpu_memory_utilization=0.90,
        )
        assert info["kv_cache_gb"] > 0
        assert info["total_gpu_gb"] == 24.0


# ─── Batching tests ───────────────────────────────────────────────────────────

class TestBatchConfig:
    def test_batch_config_creation(self):
        cfg = BatchConfig(batch_size=8)
        assert cfg.batch_size == 8
        assert cfg.strategy == "static"

    def test_to_dict(self):
        cfg = BatchConfig(batch_size=8, strategy="dynamic")
        d   = cfg.to_dict()
        assert d["batch_size"] == 8
        assert d["strategy"]   == "dynamic"


class TestBatchOptimizer:
    def test_init(self):
        opt = BatchOptimizer(sla_target_ms=500)
        assert opt.sla_target_ms == 500
        assert opt.results == []

    def test_find_optimal_empty(self):
        opt = BatchOptimizer()
        assert opt.find_optimal([]) is None

    def test_find_optimal_selects_highest_throughput(self):
        opt = BatchOptimizer(sla_target_ms=500)

        # Two configs both meeting SLA - should pick higher throughput
        r1 = BatchExperimentResult(
            batch_size=4, strategy="static",
            mean_latency_ms=100, p50_latency_ms=95,
            p95_latency_ms=200, p99_latency_ms=300,
            throughput_req_per_sec=10, throughput_tokens_per_sec=500,
            meets_sla=True, sla_target_ms=500,
        )
        r2 = BatchExperimentResult(
            batch_size=16, strategy="static",
            mean_latency_ms=200, p50_latency_ms=180,
            p95_latency_ms=450, p99_latency_ms=490,
            throughput_req_per_sec=8, throughput_tokens_per_sec=1200,
            meets_sla=True, sla_target_ms=500,
        )
        best = opt.find_optimal([r1, r2])
        assert best.batch_size == 16   # higher throughput

    def test_find_optimal_none_when_all_fail_sla(self):
        opt = BatchOptimizer(sla_target_ms=50)   # very tight SLA

        r = BatchExperimentResult(
            batch_size=1, strategy="static",
            mean_latency_ms=500, p50_latency_ms=490,
            p95_latency_ms=800, p99_latency_ms=900,
            throughput_req_per_sec=2, throughput_tokens_per_sec=100,
            meets_sla=False, sla_target_ms=50,
        )
        assert opt.find_optimal([r]) is None


# ─── Quality metric tests ──────────────────────────────────────────────────────

class TestQualityResult:
    def test_quality_result_creation(self):
        qr = QualityResult(
            model_name="llama-2-7b",
            quantization="int8",
            task_accuracy=0.85,
            num_samples=100,
        )
        assert qr.task_accuracy == 0.85
        assert qr.num_samples   == 100

    def test_to_dict(self):
        qr = QualityResult(
            model_name="llama-2-7b",
            quantization="int8",
            task_accuracy=0.85,
        )
        d = qr.to_dict()
        assert "model_name"   in d
        assert "quantization" in d
        assert "task_accuracy" in d

    def test_quality_degradation_pct(self):
        qr = QualityResult(
            model_name="llama-2-7b",
            quantization="int8",
            task_accuracy=0.85,
            accuracy_delta=-0.02,   # 2% worse than baseline
        )
        assert qr.quality_degradation_pct() == 2.0


class TestQualityEvaluator:
    def test_evaluator_init(self):
        ev = QualityEvaluator()
        assert ev.QA_ANSWERS is not None
        assert len(ev.QA_ANSWERS) >= 10

    def test_compare_quality(self):
        ev = QualityEvaluator()

        baseline = QualityResult(
            model_name="m", quantization="fp16",
            task_accuracy=0.90, rouge_l_score=0.70,
        )
        optimized = QualityResult(
            model_name="m", quantization="int8",
            task_accuracy=0.88, rouge_l_score=0.68,
        )
        result = ev.compare_quality(baseline, optimized)

        assert result.accuracy_delta == pytest.approx(-0.02, abs=1e-6)
        assert result.rouge_delta    == pytest.approx(-0.02, abs=1e-6)