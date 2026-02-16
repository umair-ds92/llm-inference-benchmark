"""Tests for model loading"""

import pytest
import os
from src.models import ModelLoader, OpenAIModel, ModelResponse


class TestModelLoader:
    """Test ModelLoader class"""
    
    def test_model_loader_init(self):
        loader = ModelLoader()
        assert len(loader.loaded_models) == 0
    
    def test_list_loaded_models(self):
        loader = ModelLoader()
        assert loader.list_loaded_models() == []


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestOpenAIModel:
    """Test OpenAI model"""
    
    def test_openai_model_init(self):
        config = {
            "name": "gpt-3.5-turbo",
            "model_id": "gpt-3.5-turbo",
            "backend": "openai_api"
        }
        model = OpenAIModel(config)
        assert model.name == "gpt-3.5-turbo"
    
    def test_openai_generate(self):
        config = {
            "name": "gpt-3.5-turbo",
            "model_id": "gpt-3.5-turbo",
            "backend": "openai_api"
        }
        model = OpenAIModel(config)
        response = model.generate("What is 2+2?", max_tokens=50)
        
        assert isinstance(response, ModelResponse)
        assert response.latency_ms > 0
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert len(response.text) > 0


class TestModelResponse:
    """Test ModelResponse dataclass"""
    
    def test_model_response_creation(self):
        response = ModelResponse(
            text="Hello world",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100.5
        )
        
        assert response.text == "Hello world"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.latency_ms == 100.5