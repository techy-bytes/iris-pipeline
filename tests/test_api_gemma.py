"""
Test suite for Gemma LLM API
"""

import pytest
from fastapi.testclient import TestClient
from src.api_gemma import app, gemma_predictor

@pytest.fixture(scope="module")
def client():
    """Create test client with startup event."""
    # Manually initialize the predictor for tests
    gemma_predictor.load_model()
    return TestClient(app)

def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "IRIS Gemma LLM API" in data["message"]
    assert "model_type" in data
    assert "model_loaded" in data

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_type" in data
    assert "api_version" in data
    assert data["api_version"] == "2.0.0"

def test_model_info(client):
    """Test model info endpoint.""" 
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "input_format" in data
    assert "output_format" in data
    assert data["input_format"] == "linguistic (conversational)"
    assert "capabilities" in data
    assert isinstance(data["capabilities"], list)

def test_llm_prediction(client):
    """Test LLM prediction endpoint."""
    # Test with typical Setosa measurements
    setosa_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict_llm", json=setosa_data)
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "species" in data
    assert "confidence" in data
    assert "reasoning" in data
    assert "model_type" in data
    
    # Verify data types
    assert isinstance(data["species"], str)
    assert isinstance(data["confidence"], float)
    assert isinstance(data["reasoning"], str)
    assert data["species"] in ["Setosa", "Versicolor", "Virginica"]
    assert 0.0 <= data["confidence"] <= 1.0
    
    # For small petal measurements, should predict Setosa
    assert data["species"] == "Setosa"

def test_llm_prediction_versicolor(client):
    """Test LLM prediction for Versicolor."""
    # Test with typical Versicolor measurements
    versicolor_data = {
        "sepal_length": 6.0,
        "sepal_width": 3.4,
        "petal_length": 4.5,
        "petal_width": 1.6
    }
    
    response = client.post("/predict_llm", json=versicolor_data)
    assert response.status_code == 200
    data = response.json()
    
    assert data["species"] == "Versicolor"
    assert data["confidence"] > 0.8
    assert "Versicolor" in data["reasoning"]

def test_llm_prediction_virginica(client):
    """Test LLM prediction for Virginica."""
    # Test with typical Virginica measurements
    virginica_data = {
        "sepal_length": 6.5,
        "sepal_width": 3.0,
        "petal_length": 5.5,
        "petal_width": 2.0
    }
    
    response = client.post("/predict_llm", json=virginica_data)
    assert response.status_code == 200
    data = response.json()
    
    assert data["species"] == "Virginica"
    assert data["confidence"] > 0.8
    assert "Virginica" in data["reasoning"]

def test_legacy_predict_endpoint(client):
    """Test legacy prediction endpoint redirects to LLM."""
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    
    # Should have same structure as LLM endpoint
    assert "species" in data
    assert "confidence" in data
    assert "reasoning" in data
    assert "model_type" in data

def test_invalid_input(client):
    """Test handling of invalid input."""
    # Missing required fields
    invalid_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5
        # Missing petal measurements
    }
    
    response = client.post("/predict_llm", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_negative_measurements(client):
    """Test handling of negative measurements."""
    negative_data = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict_llm", json=negative_data)
    # Should still work (model can handle edge cases)
    assert response.status_code == 200

def test_reasoning_quality(client):
    """Test that reasoning contains meaningful information."""
    test_cases = [
        {
            "data": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            "expected_species": "Setosa",
            "reasoning_keywords": ["small", "petal"]
        },
        {
            "data": {"sepal_length": 6.0, "sepal_width": 3.4, "petal_length": 4.5, "petal_width": 1.6},
            "expected_species": "Versicolor",
            "reasoning_keywords": ["medium", "petal"]
        },
        {
            "data": {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 2.0},
            "expected_species": "Virginica", 
            "reasoning_keywords": ["large", "petal"]
        }
    ]
    
    for test_case in test_cases:
        response = client.post("/predict_llm", json=test_case["data"])
        assert response.status_code == 200
        data = response.json()
        
        assert data["species"] == test_case["expected_species"]
        
        # Check that reasoning contains meaningful keywords
        reasoning_lower = data["reasoning"].lower()
        found_keywords = [kw for kw in test_case["reasoning_keywords"] if kw in reasoning_lower]
        assert len(found_keywords) > 0, f"Reasoning should contain keywords {test_case['reasoning_keywords']}, got: {data['reasoning']}"

def test_confidence_scores(client):
    """Test that confidence scores are reasonable."""
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict_llm", json=test_data)
    assert response.status_code == 200
    data = response.json()
    
    confidence = data["confidence"]
    assert 0.5 <= confidence <= 1.0, f"Confidence should be reasonable (0.5-1.0), got {confidence}"
    
    # For very clear cases (like Setosa with small petals), confidence should be high
    if data["species"] == "Setosa":
        assert confidence >= 0.9, f"Setosa prediction should have high confidence, got {confidence}"