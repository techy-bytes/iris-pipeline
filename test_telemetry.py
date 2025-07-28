#!/usr/bin/env python3
"""
Demo script to test the Iris ML service with telemetry
Based on professor's logging & telemetry demo requirements
"""

import requests
import json
import time
import sys

def test_service(base_url="http://localhost:8000"):
    """Test the iris ML service endpoints with telemetry logging."""
    
    print("🧪 Testing Iris ML Service with Telemetry")
    print("=" * 50)
    
    # Test health check
    print("\n📊 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test prediction endpoint with various samples
    test_samples = [
        {
            "name": "Setosa sample",
            "data": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        },
        {
            "name": "Versicolor sample", 
            "data": {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5}
        },
        {
            "name": "Virginica sample",
            "data": {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
        }
    ]
    
    print("\n🔮 Testing prediction endpoint...")
    for sample in test_samples:
        try:
            print(f"\n  Testing {sample['name']}:")
            print(f"  Input: {sample['data']}")
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/predict",
                json=sample['data'],
                headers={"Content-Type": "application/json"}
            )
            latency = round((time.time() - start_time) * 1000, 2)
            
            print(f"  Status: {response.status_code}")
            print(f"  Response: {json.dumps(response.json(), indent=4)}")
            print(f"  Latency: {latency}ms")
            
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
    
    # Test batch prediction
    print("\n📦 Testing batch prediction endpoint...")
    batch_data = [sample["data"] for sample in test_samples]
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict_batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        latency = round((time.time() - start_time) * 1000, 2)
        
        print(f"Status: {response.status_code}")
        print(f"Batch size: {len(batch_data)}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print(f"Latency: {latency}ms")
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
    
    # Test error handling
    print("\n🚨 Testing error handling...")
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"invalid": "data"},
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Error response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error test failed: {e}")
    
    print("\n✅ Testing completed!")
    print("\n💡 Telemetry features:")
    print("  - All requests generate trace IDs for correlation")
    print("  - Structured JSON logging for observability")  
    print("  - Performance metrics (latency) tracked")
    print("  - Exception handling with trace correlation")
    print("  - Ready for Google Cloud Trace and Logging integration")

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_service(base_url)