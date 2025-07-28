#!/usr/bin/env python3
"""
Locust load testing script for the Iris ML API
This script tests the telemetry and performance under load
"""

from locust import HttpUser, task, between
import json
import random


class IrisAPIUser(HttpUser):
    """Simulates a user making requests to the Iris ML API."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts."""
        # Test samples for different iris species
        self.test_samples = [
            # Setosa samples
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "petal_width": 0.2},
            
            # Versicolor samples  
            {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},
            {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5},
            {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 4.9, "petal_width": 1.5},
            
            # Virginica samples
            {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
            {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9},
            {"sepal_length": 7.1, "sepal_width": 3.0, "petal_length": 5.9, "petal_width": 2.1},
        ]
    
    @task(3)
    def test_health_check(self):
        """Test the health endpoint (lighter weight)."""
        with self.client.get("/health", name="health_check", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Health check failed: {data}")
            else:
                response.failure(f"Health check returned {response.status_code}")
    
    @task(10)
    def test_single_prediction(self):
        """Test single prediction endpoint."""
        sample = random.choice(self.test_samples)
        
        with self.client.post(
            "/predict",
            json=sample,
            headers={"Content-Type": "application/json"},
            name="single_prediction",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "species" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure(f"Invalid prediction response: {data}")
            else:
                response.failure(f"Prediction failed with {response.status_code}: {response.text}")
    
    @task(2)
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        # Create batch of 2-5 samples
        batch_size = random.randint(2, 5)
        batch_data = random.choices(self.test_samples, k=batch_size)
        
        with self.client.post(
            "/predict_batch",
            json=batch_data,
            headers={"Content-Type": "application/json"},
            name="batch_prediction",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) == batch_size:
                    # Verify all predictions have required fields
                    if all("species" in pred and "confidence" in pred for pred in data):
                        response.success()
                    else:
                        response.failure("Invalid batch prediction response format")
                else:
                    response.failure(f"Batch size mismatch: expected {batch_size}, got {len(data) if isinstance(data, list) else 'non-list'}")
            else:
                response.failure(f"Batch prediction failed with {response.status_code}: {response.text}")
    
    @task(1)
    def test_root_endpoint(self):
        """Test the root endpoint."""
        with self.client.get("/", name="root_endpoint", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "message" in data:
                    response.success()
                else:
                    response.failure(f"Root endpoint response missing message: {data}")
            else:
                response.failure(f"Root endpoint failed with {response.status_code}")


class HighLoadUser(HttpUser):
    """High-frequency user for stress testing."""
    
    wait_time = between(0.1, 0.5)  # Very fast requests
    weight = 1  # Lower weight so fewer of these users
    
    def on_start(self):
        """Initialize with a few test samples."""
        self.test_samples = [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5},
            {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
        ]
    
    @task
    def rapid_predictions(self):
        """Make rapid predictions for stress testing."""
        sample = random.choice(self.test_samples)
        
        with self.client.post(
            "/predict",
            json=sample,
            headers={"Content-Type": "application/json"},
            name="rapid_prediction",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Rapid prediction failed: {response.status_code}")