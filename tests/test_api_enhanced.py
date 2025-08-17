import pytest
import os
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Ensure models are loaded by triggering training
from src.train_enhanced import train_sklearn_model

# Mock the environment before importing the app
with patch.dict(os.environ, {'USE_LLM_MODEL': 'true', 'USE_MOCK_LLM': 'true'}):
    from src.api_enhanced import app

# Setup models before creating test client
try:
    train_sklearn_model()
except:
    pass

client = TestClient(app)


class TestEnhancedAPI:
    """Test enhanced API with both sklearn and Gemma models."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "Enhanced Iris Classifier" in data["message"]
    
    def test_health_endpoint(self):
        """Test health endpoint with model information."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "sklearn_available" in data
        assert "gemma_available" in data
        assert "default_model" in data
        assert "models_loaded" in data
        assert "feature_names" in data
        assert "classes" in data
        
        assert isinstance(data["models_loaded"], list)
        assert len(data["models_loaded"]) > 0
    
    def test_models_endpoint(self):
        """Test models information endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        
        assert "sklearn_available" in data
        assert "gemma_available" in data
        assert "default_model" in data
        assert "models_loaded" in data
        
        assert isinstance(data["sklearn_available"], bool)
        assert isinstance(data["gemma_available"], bool)
        assert data["default_model"] in ["sklearn", "gemma"]
    
    def test_predict_endpoint_default_model(self):
        """Test prediction with default model."""
        sample_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=sample_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "species" in data
        assert "confidence" in data
        assert "model_used" in data
        assert data["species"] in ["setosa", "versicolor", "virginica"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["model_used"] in ["sklearn", "gemma"]
    
    def test_predict_endpoint_sklearn_model(self):
        """Test prediction with sklearn model explicitly."""
        sample_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict?model=sklearn", json=sample_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_used"] == "sklearn"
        assert data["species"] in ["setosa", "versicolor", "virginica"]
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_predict_endpoint_gemma_model(self):
        """Test prediction with Gemma model explicitly."""
        sample_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict?model=gemma", json=sample_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_used"] == "gemma"
        assert data["species"] in ["setosa", "versicolor", "virginica"]
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_predict_endpoint_invalid_model(self):
        """Test prediction with invalid model type."""
        sample_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict?model=invalid", json=sample_data)
        assert response.status_code == 400
        assert "Invalid model type" in response.json()["detail"]
    
    def test_predict_endpoint_setosa_classification(self):
        """Test setosa classification with both models."""
        setosa_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        # Test with sklearn
        response = client.post("/predict?model=sklearn", json=setosa_data)
        assert response.status_code == 200
        data = response.json()
        assert data["species"] == "setosa"
        
        # Test with Gemma (mock should also predict setosa for these small measurements)
        response = client.post("/predict?model=gemma", json=setosa_data)
        assert response.status_code == 200
        data = response.json()
        assert data["species"] == "setosa"
    
    def test_predict_batch_endpoint(self):
        """Test batch prediction."""
        batch_data = [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.0,
                "sepal_width": 3.0,
                "petal_length": 4.5,
                "petal_width": 1.5
            }
        ]
        
        response = client.post("/predict_batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 2
        for prediction in data:
            assert "species" in prediction
            assert "confidence" in prediction
            assert "model_used" in prediction
            assert prediction["species"] in ["setosa", "versicolor", "virginica"]
    
    def test_predict_batch_empty(self):
        """Test batch prediction with empty list."""
        response = client.post("/predict_batch", json=[])
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0
    
    def test_compare_models_endpoint(self):
        """Test model comparison endpoint."""
        sample_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/compare_models", json=sample_data)
        assert response.status_code == 200
        data = response.json()
        
        assert "input" in data
        assert "predictions" in data
        assert "timestamp" in data
        
        # Should have predictions from both models
        predictions = data["predictions"]
        assert "sklearn" in predictions or "gemma" in predictions
        
        # Check structure of predictions
        for model_name, prediction in predictions.items():
            if "error" not in prediction:
                assert "species" in prediction
                assert "confidence" in prediction
                assert "model_used" in prediction
    
    def test_predict_endpoint_invalid_data(self):
        """Test prediction with invalid input data."""
        invalid_data = {
            "sepal_length": "invalid",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error