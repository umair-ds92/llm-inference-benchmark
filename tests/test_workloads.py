"""Tests for workload generation"""

import pytest
from src.workloads.scenarios import (
    generate_qa_prompts,
    generate_summarization_prompts,
    generate_code_generation_prompts,
    generate_conversation_prompts,
)
from src.workloads import WorkloadGenerator


class TestWorkloadScenarios:
    """Test scenario generation functions"""
    
    def test_qa_prompts(self):
        prompts = generate_qa_prompts(10)
        assert len(prompts) == 10
        assert all(isinstance(p, str) for p in prompts)
        assert all(len(p) > 0 for p in prompts)
    
    def test_summarization_prompts(self):
        prompts = generate_summarization_prompts(5)
        assert len(prompts) == 5
        assert all("Summarize" in p for p in prompts)
    
    def test_code_generation_prompts(self):
        prompts = generate_code_generation_prompts(10)
        assert len(prompts) == 10
        assert all(isinstance(p, str) for p in prompts)
    
    def test_conversation_prompts(self):
        prompts = generate_conversation_prompts(5)
        assert len(prompts) == 5
        assert all("User:" in p and "Assistant:" in p for p in prompts)


class TestWorkloadGenerator:
    """Test WorkloadGenerator class"""
    
    def test_generator_init(self):
        generator = WorkloadGenerator()
        assert generator.workloads is not None
    
    def test_generate_qa_workload(self):
        generator = WorkloadGenerator()
        prompts = generator.generate_workload("qa", num_samples=5)
        assert len(prompts) == 5
    
    def test_unknown_workload(self):
        generator = WorkloadGenerator()
        with pytest.raises(ValueError):
            generator.generate_workload("unknown_workload")