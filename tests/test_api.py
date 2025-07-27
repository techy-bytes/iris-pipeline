import pytest
from fastapi.testclient import TestClient
import sys
import os
import tempfile
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import app, train_and_save_model

@pytest.fixture(autouse=True)
def setup_model():
    """Setup model for testing."""
    # Ensure model is trained before tests
    try:
        train_and_save_model()
    except:
        # If training fails, create a simple test dataset
        test_data = {
            'sepal_length': [5.1, 4.9, 6.0, 5.8],
            'sepal_width': [3.5, 3.0, 3.0, 2.7],
            'petal_length': [1.4, 1.4, 4.5, 5.1],
            'petal_width': [0.2, 0.2, 1.5, 1.9],
            'species': ['setosa', 'setosa', 'versicolor', 'virginica']
        }
        df = pd.DataFrame(test_data)
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/iris.csv', index=False)
        train_and_save_model()

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert data["status"] == "healthy"

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "feature_names" in data
    assert "classes" in data
    assert len(data["classes"]) == 3


def test_predict_endpoint_invalid_data():
    """Test the prediction endpoint with invalid data."""
    invalid_data = {
        "sepal_length": "invalid",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error



def test_predict_batch_empty():
    """Test batch prediction with empty list."""
    response = client.post("/predict_batch", json=[])
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0